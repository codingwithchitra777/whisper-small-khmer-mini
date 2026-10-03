"""
Khmer Subtitle API — FastAPI backend
Handles YouTube audio download (via yt-dlp) and Khmer ASR (via fine-tuned Whisper).

Uses WhisperProcessor + WhisperForConditionalGeneration directly (same as the
existing inference_whisper_khmer.py) to avoid pipeline beam-search compatibility
issues with this fine-tuned model's generation_config.json.
"""

import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

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
# Whisper's decoder holds 448 tokens, and Khmer tokenizes long (~400 tokens per 15 s of read
# speech, more for fast news delivery), so a full 30 s window gets cut off mid-transcript. Chunks
# stay at or under 10 s (15 s still garbled the endings of fast newsreader sentences) and end at a
# pause, so words aren't split and no overlap is needed.
MIN_CHUNK_SECONDS = 2
MAX_CHUNK_SECONDS = 10
PAUSE_RELATIVE_RMS = 0.1   # a "pause" is quieter than 10% of the window's median loudness
MAX_NEW_TOKENS = 440       # 448 minus the 4 prompt tokens, with a little room
# Khmer runs ~27 tokens/s of speech; capping each chunk near what its length needs stops a
# repetition loop early instead of letting it run to MAX_NEW_TOKENS.
BATCH_SIZE = 8             # chunks transcribed per generate() call (~1.5 GB of GPU memory in fp16)
TOKENS_PER_SECOND = 40
TOKEN_MARGIN = 24
REPEAT_RUN = re.compile(r"(.{2,15}?)\1{2,}")  # the same 2–15 chars 3+ times in a row (a decoding loop)
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
    # fp16 on the GPU roughly halves transcription time; CPU stays fp32 (fp16 is slow there).
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(str(MODEL_DIR), dtype=dtype).to(device)
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


class JobResponse(BaseModel):
    job_id: str
    status: str                       # queued | downloading | transcribing | done | error
    queue_position: int = 0           # jobs ahead of this one (queued status only)
    chunks_done: int = 0
    chunks_total: int = 0
    duration_seconds: Optional[float] = None
    segments: list[Segment] = []
    error: Optional[str] = None


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


def _clean_text(text: str) -> str:
    """Collapse decoding loops to one copy and drop the broken character a cut-off token leaves."""
    return REPEAT_RUN.sub(r"\1", text.replace("�", "")).strip()


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


def _transcribe_with_timestamps(
    audio_path: Path, on_progress: Optional[Callable[[int, int], None]] = None
) -> list[dict]:
    """
    Transcribe audio using the fine-tuned Whisper model with segment timestamps.

    Splits long audio at pauses into chunks of at most MAX_CHUNK_SECONDS, transcribes each with
    return_timestamps=True, and assembles the results with corrected offsets. `on_progress` is
    called with (chunks done, total chunks) after each batch.
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

    # (start s, end s, samples) for every chunk with sound in it
    chunks = [
        (start / SAMPLING_RATE, min(end / SAMPLING_RATE, total_duration), audio_array[start:end])
        for start, end in _split_on_pauses(audio_array)
        if np.sqrt(np.mean(audio_array[start:end] ** 2)) >= SILENCE_RMS
    ]

    # Chunks are transcribed BATCH_SIZE at a time: generation is one forward pass per token, and on a
    # laptop GPU the per-pass overhead dominates, so a batch costs about the same time as one chunk.
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        input_features = processor(
            [samples for _, _, samples in batch],
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
        ).input_features.to(device, dtype=model.dtype)
        longest = max(end - start for start, end, _ in batch)
        token_cap = min(MAX_NEW_TOKENS, int(longest * TOKENS_PER_SECOND) + TOKEN_MARGIN)

        with torch.no_grad():
            # return_timestamps produces token-level timestamps decoded as segments
            predicted_ids = model.generate(
                input_features,
                language="khmer",
                return_timestamps=True,
                max_new_tokens=token_cap,
            )

        # One {"text", "offsets": [{"text", "timestamp": (start, end)}]} per chunk; times are chunk-relative
        result = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True, output_offsets=True
        )

        for (chunk_offset_sec, chunk_end_sec, _), item in zip(batch, result):
            offsets = item.get("offsets", [])
            if offsets:
                for seg in offsets:
                    raw_ts = seg.get("timestamp", (0.0, None))
                    seg_start = (raw_ts[0] or 0.0) + chunk_offset_sec
                    # A missing end timestamp means the segment runs to the end of the chunk
                    seg_end = chunk_end_sec if raw_ts[1] is None else raw_ts[1] + chunk_offset_sec
                    seg_end = min(seg_end, chunk_end_sec)
                    text = _clean_text(seg.get("text", ""))
                    if text:
                        segments.append({
                            "start": round(seg_start, 2),
                            "end":   round(seg_end, 2),
                            "text":  text,
                        })
            else:
                # No offsets — treat entire chunk as one segment
                full_text = _clean_text(item.get("text", ""))
                if full_text:
                    segments.append({
                        "start": round(chunk_offset_sec, 2),
                        "end":   round(chunk_end_sec, 2),
                        "text":  full_text,
                    })

        if on_progress:
            on_progress(min(batch_start + BATCH_SIZE, len(chunks)), len(chunks))

    # Chunks don't overlap, so no deduplication: repeated phrases in the video are kept.
    return segments


def _get_duration(audio_path: Path) -> Optional[float]:
    """Return audio duration in seconds using soundfile."""
    try:
        info = sf.info(str(audio_path))
        return round(info.duration, 2)
    except Exception:
        return None


# ── Background jobs ───────────────────────────────────────────────────────────
# Cloudflare (the demo tunnel, RunPod's proxy) closes any request after 100 s, so a long video
# can't be processed inside one request. POST /jobs returns a job ID at once and the web app polls
# GET /jobs/{id}. A single worker thread runs jobs in order: there is one GPU, and running two
# videos at once would only make both slower.
JOB_KEEP_SECONDS = 3600  # finished jobs are forgotten after an hour

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_job_runner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcribe")


def _update_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs[job_id].update(fields, updated=time.time())


def _run_job(job_id: str, url: str) -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="khmer_subtitle_"))
    try:
        _update_job(job_id, status="downloading")
        audio_path = _download_audio(url, tmp_dir)
        _update_job(job_id, status="transcribing", duration_seconds=_get_duration(audio_path))
        segments = _transcribe_with_timestamps(
            audio_path,
            on_progress=lambda done, total: _update_job(job_id, chunks_done=done, chunks_total=total),
        )
        _update_job(job_id, status="done", segments=segments)
    except HTTPException as error:
        _update_job(job_id, status="error", error=str(error.detail), error_status=error.status_code)
    except Exception as error:  # report it to the user instead of leaving the job "running" forever
        _update_job(job_id, status="error", error=f"{type(error).__name__}: {error}", error_status=500)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _start_job(url: str):
    """Queue a job; returns (job_id, future)."""
    now = time.time()
    with _jobs_lock:
        for old_id in [j for j, job in _jobs.items()
                       if job["status"] in ("done", "error") and job["updated"] < now - JOB_KEEP_SECONDS]:
            del _jobs[old_id]
        job_id = uuid.uuid4().hex
        _jobs[job_id] = {"status": "queued", "created": now, "updated": now, "chunks_done": 0,
                         "chunks_total": 0, "duration_seconds": None, "segments": [], "error": None}
    return job_id, _job_runner.submit(_run_job, job_id, url)


def _job_view(job_id: str) -> JobResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found (it may have expired).")
        ahead = sum(1 for other in _jobs.values()
                    if other["status"] == "queued" and other["created"] < job["created"])
        return JobResponse(
            job_id=job_id,
            status=job["status"],
            queue_position=ahead if job["status"] == "queued" else 0,
            chunks_done=job["chunks_done"],
            chunks_total=job["chunks_total"],
            duration_seconds=job["duration_seconds"],
            segments=[Segment(**s) for s in job["segments"]] if job["status"] == "done" else [],
            error=job["error"],
        )


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


@app.post("/jobs", response_model=JobResponse, status_code=202)
def create_job(request: ProcessRequest) -> JobResponse:
    """
    Queue a YouTube URL for subtitling and return at once; poll GET /jobs/{job_id}.
    The job: download audio via yt-dlp (16 kHz mono WAV) → split at pauses into chunks of at
    most 10 s → transcribe with timestamps → timestamped Khmer segments.
    """
    job_id, _ = _start_job(request.url)
    return _job_view(job_id)


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    """Job status, progress in chunks, and the segments once status is "done"."""
    return _job_view(job_id)


@app.post("/process", response_model=ProcessResponse)
def process_youtube(request: ProcessRequest) -> ProcessResponse:
    """
    Same work as /jobs, answered in one request (for scripts and short videos). Behind a
    Cloudflare proxy, videos longer than ~4 minutes exceed its 100 s limit; use /jobs there.
    Runs in the same one-at-a-time queue as /jobs.
    """
    job_id, future = _start_job(request.url)
    future.result()
    with _jobs_lock:
        job = dict(_jobs[job_id])
    if job["status"] == "error":
        raise HTTPException(status_code=job.get("error_status", 500), detail=job["error"])
    return ProcessResponse(
        success=True,
        segments=[Segment(**s) for s in job["segments"]],
        total_segments=len(job["segments"]),
        duration_seconds=job["duration_seconds"],
    )
