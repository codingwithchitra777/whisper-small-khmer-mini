"""
Khmer Subtitle API — FastAPI backend
Handles YouTube audio download (via yt-dlp) and Khmer ASR (via fine-tuned Whisper).

Uses WhisperProcessor + WhisperForConditionalGeneration directly (same as the
existing inference_whisper_khmer.py) to avoid pipeline beam-search compatibility
issues with this fine-tuned model's generation_config.json.
"""

import math
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# The fine-tuned model downloaded from Kaggle goes here; KHMER_MODEL_DIR overrides it
# (e.g. point it at another checkpoint to compare models in the demo).
MODEL_DIR = Path(os.environ.get("KHMER_MODEL_DIR", PROJECT_ROOT / "models" / "whisper-small-khmer-mini"))

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLING_RATE = 16_000
# Whisper's decoder holds 448 tokens, and Khmer tokenizes long (~400 tokens per 15 s of speech),
# so a full 30 s window gets cut off mid-transcript. Chunks stay at or under 15 s and are cut at
# the quietest pause between 8 s and 15 s, so words aren't split and no overlap is needed.
MIN_CHUNK_SECONDS = 2
MAX_CHUNK_SECONDS = 15
PAUSE_RELATIVE_RMS = 0.1   # a "pause" is quieter than 10% of the window's median loudness
MAX_NEW_TOKENS = 440       # 448 minus the 4 prompt tokens, with a little room
FRAME_SECONDS = 0.02       # energy frame for pause detection
PAUSE_SMOOTHING_FRAMES = 10  # look for ~200 ms of quiet, not a single quiet frame
SILENCE_RMS = 1e-3         # chunks quieter than this are skipped (avoids hallucinated text)

# ── Global model objects ──────────────────────────────────────────────────────
processor: Optional[WhisperProcessor] = None
model: Optional[WhisperForConditionalGeneration] = None
device: Optional[torch.device] = None


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global processor, model, device
    if not MODEL_DIR.exists():
        raise RuntimeError(f"Model directory not found: {MODEL_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_label = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"[INFO] Loading Khmer Whisper model from {MODEL_DIR} on {device_label}...")

    processor = WhisperProcessor.from_pretrained(str(MODEL_DIR))
    model = WhisperForConditionalGeneration.from_pretrained(str(MODEL_DIR)).to(device)
    model.eval()

    # ── Patch generation_config for newer transformers compatibility ──────────
    # The saved generation_config has eos_token_id as a list [50257], but newer
    # transformers expects an integer in SuppressTokensAtBeginLogitsProcessor.
    # Also, forced_decoder_ids=[[1, null], [2, 50359]] must be cleared to allow
    # language= and task= kwargs to work properly.
    gc = model.generation_config
    if isinstance(gc.eos_token_id, list):
        gc.eos_token_id = gc.eos_token_id[0]  # [50257] -> 50257
    gc.forced_decoder_ids = None              # let generate() set language/task
    gc.suppress_tokens = []                   # avoid double suppress_tokens conflict
    gc.begin_suppress_tokens = []

    print("[INFO] Model loaded successfully.")
    yield

    # Shutdown cleanup
    processor = None
    model = None
    print("[INFO] Model unloaded.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Khmer Subtitle API",
    description="Download YouTube audio and transcribe it with a fine-tuned Khmer Whisper model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────
class ProcessRequest(BaseModel):
    url: str


class Segment(BaseModel):
    start: float
    end: float
    text: str


class ProcessResponse(BaseModel):
    success: bool
    segments: list[Segment]
    total_segments: int
    duration_seconds: Optional[float] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_ytdlp_cmd() -> str:
    """Return the yt-dlp executable path (venv or system PATH)."""
    venv_ytdlp = PROJECT_ROOT / "py-venv" / "Scripts" / "yt-dlp.exe"
    if venv_ytdlp.exists():
        return str(venv_ytdlp)
    return "yt-dlp"


def _download_audio(url: str, tmp_dir: Path) -> Path:
    """Download best audio from YouTube URL, convert to 16kHz mono WAV."""
    yt_dlp = _get_ytdlp_cmd()
    output_template = str(tmp_dir / "audio.%(ext)s")

    cmd = [
        yt_dlp,
        "--no-playlist",
        "--no-warnings",
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", output_template,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail=f"yt-dlp failed: {result.stderr.strip()[:500]}",
        )

    candidates = list(tmp_dir.glob("audio.*"))
    if not candidates:
        raise HTTPException(status_code=500, detail="No audio file produced by yt-dlp.")

    audio_path = candidates[0]

    # Extra ffmpeg conversion if yt-dlp couldn't produce WAV directly
    if audio_path.suffix.lower() != ".wav":
        wav_path = tmp_dir / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(wav_path)],
            capture_output=True, timeout=120,
        )
        audio_path = wav_path

    return audio_path


def _split_on_pauses(audio_array: np.ndarray) -> list[tuple[int, int]]:
    """
    Split audio into (start, end) sample ranges of at most MAX_CHUNK_SECONDS. Each chunk ends at
    the latest real pause between MIN_CHUNK_SECONDS and MAX_CHUNK_SECONDS (long chunks, whole
    sentences); with no pause in that window, it ends at the quietest ~200 ms instead.
    """
    frame = int(FRAME_SECONDS * SAMPLING_RATE)
    n_frames = len(audio_array) // frame
    if n_frames == 0:
        return [(0, len(audio_array))] if len(audio_array) else []
    frames = audio_array[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    smoothed = np.convolve(rms, np.ones(PAUSE_SMOOTHING_FRAMES) / PAUSE_SMOOTHING_FRAMES, mode="same")

    min_frames = int(MIN_CHUNK_SECONDS / FRAME_SECONDS)
    max_frames = int(MAX_CHUNK_SECONDS / FRAME_SECONDS)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < n_frames:
        if n_frames - start <= max_frames:
            ranges.append((start * frame, len(audio_array)))
            break
        window = smoothed[start + min_frames : start + max_frames]
        threshold = PAUSE_RELATIVE_RMS * np.median(smoothed[start : start + max_frames])
        pauses = np.flatnonzero(window < threshold)
        cut = start + min_frames + int(pauses[-1] if len(pauses) else np.argmin(window))
        ranges.append((start * frame, cut * frame))
        start = cut
    return ranges


def _transcribe_with_timestamps(audio_path: Path) -> list[dict]:
    """
    Transcribe audio using the fine-tuned Whisper model with segment timestamps.

    Splits long audio at pauses into chunks of at most MAX_CHUNK_SECONDS, transcribes each with
    return_timestamps=True, and assembles the results with corrected offsets.
    """
    if processor is None or model is None or device is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again shortly.")

    # Load full audio with soundfile (avoids librosa/pkg_resources issues)
    # soundfile returns (samples, samplerate); resample if needed
    audio_raw, orig_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    # Convert stereo to mono if needed
    if audio_raw.ndim == 2:
        audio_raw = audio_raw.mean(axis=1)
    # Resample to 16kHz if the file isn't already (yt-dlp should handle this, but just in case)
    if orig_sr != SAMPLING_RATE:
        # Simple linear resampling using numpy
        duration = len(audio_raw) / orig_sr
        target_len = int(duration * SAMPLING_RATE)
        audio_array = np.interp(
            np.linspace(0, len(audio_raw) - 1, target_len),
            np.arange(len(audio_raw)),
            audio_raw,
        ).astype(np.float32)
    else:
        audio_array = audio_raw
    total_duration = len(audio_array) / SAMPLING_RATE

    segments: list[dict] = []

    for start_sample, end_sample in _split_on_pauses(audio_array):
        chunk = audio_array[start_sample:end_sample]
        if np.sqrt(np.mean(chunk ** 2)) < SILENCE_RMS:
            continue
        chunk_offset_sec = start_sample / SAMPLING_RATE
        chunk_end_sec = min(end_sample / SAMPLING_RATE, total_duration)

        # Prepare features
        input_features = processor(
            chunk,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
        ).input_features.to(device)

        with torch.no_grad():
            # return_timestamps produces token-level timestamps decoded as segments
            predicted_ids = model.generate(
                input_features,
                language="khmer",
                return_timestamps=True,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        # Decode with timestamps — returns list of dicts [{timestamp, text}]
        result = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True, output_offsets=True
        )

        # batch_decode with output_with_offsets returns a list of dicts per item
        # Each item has "text" and optionally "offsets" list
        for item in result:
            offsets = item.get("offsets", [])
            if offsets:
                for seg in offsets:
                    raw_ts = seg.get("timestamp", (0.0, None))
                    seg_start = (raw_ts[0] or 0.0) + chunk_offset_sec
                    # A missing end timestamp means the segment runs to the end of the chunk
                    seg_end = chunk_end_sec if raw_ts[1] is None else raw_ts[1] + chunk_offset_sec
                    seg_end = min(seg_end, chunk_end_sec)
                    text = seg.get("text", "").strip()
                    if text:
                        segments.append({
                            "start": round(seg_start, 2),
                            "end":   round(seg_end, 2),
                            "text":  text,
                        })
            else:
                # No offsets — treat entire chunk as one segment
                full_text = item.get("text", "").strip()
                if full_text:
                    segments.append({
                        "start": round(chunk_offset_sec, 2),
                        "end":   round(chunk_end_sec, 2),
                        "text":  full_text,
                    })

    # Chunks don't overlap, so no deduplication: repeated phrases in the video are kept.
    return segments


def _get_duration(audio_path: Path) -> Optional[float]:
    """Return audio duration in seconds using soundfile."""
    try:
        info = sf.info(str(audio_path))
        return round(info.duration, 2)
    except Exception:
        return None


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check — confirms the model is loaded."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_dir": str(MODEL_DIR),
    }


@app.post("/process", response_model=ProcessResponse)
async def process_youtube(request: ProcessRequest) -> ProcessResponse:
    """
    Main endpoint:
    1. Download audio from YouTube URL via yt-dlp (16kHz mono WAV).
    2. Split at pauses into chunks of at most 15 s.
    3. Transcribe each chunk with return_timestamps=True.
    4. Assemble timestamped Khmer segments.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="khmer_subtitle_"))
    try:
        audio_path = _download_audio(request.url, tmp_dir)
        duration   = _get_duration(audio_path)
        segments   = _transcribe_with_timestamps(audio_path)

        return ProcessResponse(
            success=True,
            segments=[Segment(**s) for s in segments],
            total_segments=len(segments),
            duration_seconds=duration,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
