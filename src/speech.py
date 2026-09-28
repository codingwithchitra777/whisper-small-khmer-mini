"""Audio/text helpers shared by dataset building, training and evaluation."""

import io
import random
import re
from dataclasses import dataclass
from math import gcd
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

SAMPLING_RATE = 16000
MIN_AUDIO_SECONDS = 0.3
MAX_AUDIO_SECONDS = 30.0  # Whisper's context window
MAX_LABEL_TOKENS = 448  # Whisper's max_target_positions


def load_audio(source) -> np.ndarray:
    """Read a path or encoded bytes into 16 kHz mono float32."""
    data = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    array, sampling_rate = sf.read(data, dtype="float32", always_2d=False)
    if array.ndim > 1:
        array = array.mean(axis=1)
    if sampling_rate != SAMPLING_RATE:
        # Sources are 16/22.05/44.1/48 kHz; polyphase resampling (librosa 0.10 breaks without pkg_resources).
        divisor = gcd(sampling_rate, SAMPLING_RATE)
        array = resample_poly(array, SAMPLING_RATE // divisor, sampling_rate // divisor).astype(np.float32)
    return array


def encode_flac(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, array, SAMPLING_RATE, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("​", "")).strip()


def remove_spaces(text: str) -> str:
    return re.sub(r"\s+", "", text)


class FeatureMaker:
    """Turns (audio bytes, text) into Whisper input features + label ids, or None if unusable.

    With `timestamp_fraction` > 0, that share of labels is written as `<|0.00|> text <|end|>`
    (no `<|notimestamps|>`), so the fine-tuned model keeps the timestamps the subtitle API needs.
    """

    def __init__(self, processor, timestamp_fraction: float = 0.0, seed: int = 0) -> None:
        self.processor = processor
        self.timestamp_fraction = timestamp_fraction
        self.rng = random.Random(seed)
        tokenizer = processor.tokenizer
        no_timestamps_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self.timestamp_prefix = [token for token in tokenizer.prefix_tokens if token != no_timestamps_id]

    def make_labels(self, text: str, seconds: float) -> List[int]:
        tokenizer = self.processor.tokenizer
        if self.rng.random() >= self.timestamp_fraction:
            return tokenizer(text).input_ids
        end = min(round(seconds / 0.02) * 0.02, MAX_AUDIO_SECONDS)
        body = tokenizer.encode(f"<|0.00|>{text}<|{end:.2f}|>", add_special_tokens=False)
        return self.timestamp_prefix + body + [tokenizer.eos_token_id]

    def __call__(self, audio: bytes, text: str) -> Optional[Dict[str, Any]]:
        array = load_audio(audio)
        if len(array) == 0 or len(array) > MAX_AUDIO_SECONDS * SAMPLING_RATE:
            return None
        labels = self.make_labels(normalize_text(text), len(array) / SAMPLING_RATE)
        if len(labels) > MAX_LABEL_TOKENS:
            return None
        features = self.processor.feature_extractor(array, sampling_rate=SAMPLING_RATE).input_features[0]
        return {"input_features": features, "labels": labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.eq(0), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_asr_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """WER, CER, and CER with spaces removed (Khmer spacing is inconsistent, so that one is fairest)."""
    import evaluate

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    predictions = [normalize_text(text) for text in predictions]
    references = [normalize_text(text) for text in references]
    return {
        "wer": wer_metric.compute(predictions=predictions, references=references),
        "cer": cer_metric.compute(predictions=predictions, references=references),
        "cer_no_space": cer_metric.compute(
            predictions=[remove_spaces(text) for text in predictions],
            references=[remove_spaces(text) for text in references],
        ),
    }
