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
    """Read a path or encoded bytes into 16 kHz mono float32.

    Whisper was trained on 16 kHz mono audio, so every clip is converted to that format first.

    Args:
        source: a file path (raw WAVs when building the dataset) or encoded audio bytes
            (the FLAC stored in the parquet files).

    Returns:
        1-D float32 array of samples in [-1, 1] at 16 kHz.
    """
    # soundfile reads from a path or a file-like object, so wrap raw bytes in BytesIO.
    data = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    array, sampling_rate = sf.read(data, dtype="float32", always_2d=False)
    if array.ndim > 1:
        # Stereo (samples x channels): average the channels into mono.
        array = array.mean(axis=1)
    if sampling_rate != SAMPLING_RATE:
        # Sources are 16/22.05/44.1/48 kHz; polyphase resampling (librosa 0.10 breaks without pkg_resources).
        # Resample by the reduced ratio 16000 / rate, e.g. 44100 Hz -> up 160, down 441.
        divisor = gcd(sampling_rate, SAMPLING_RATE)
        array = resample_poly(array, SAMPLING_RATE // divisor, sampling_rate // divisor).astype(np.float32)
    return array


def encode_flac(array: np.ndarray) -> bytes:
    """Compress 16 kHz audio to FLAC bytes for storage in the parquet files.

    FLAC is lossless and about half the size of WAV, which keeps the dataset upload to
    Kaggle/RunPod small. 16-bit samples are plenty for speech.

    Args:
        array: 1-D float32 samples at 16 kHz (output of `load_audio`).

    Returns:
        The encoded FLAC file as bytes (decode again with `load_audio`).
    """
    # Encode into memory instead of a temporary file.
    buffer = io.BytesIO()
    sf.write(buffer, array, SAMPLING_RATE, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def normalize_text(text: str) -> str:
    """Clean a transcript so every source has the same spacing.

    Removes zero-width spaces (U+200B, which Khmer text often uses as an invisible word
    separator), turns any run of whitespace (tabs, newlines, double spaces) into one space,
    and trims both ends.

    Args:
        text: raw transcript.

    Returns:
        The cleaned transcript.
    """
    return re.sub(r"\s+", " ", text.replace("​", "")).strip()


def remove_spaces(text: str) -> str:
    """Delete all whitespace.

    Khmer is written without spaces between words and the sources disagree on where to put
    them, so texts are compared without spaces: for `cer_no_space`, for assigning splits, and
    for finding sentences shared with ddd_test.

    Args:
        text: any transcript.

    Returns:
        The transcript with every space, tab and newline removed.
    """
    return re.sub(r"\s+", "", text)


class FeatureMaker:
    """Turns (audio bytes, text) into Whisper input features + label ids, or None if unusable.

    With `timestamp_fraction` > 0, that share of labels is written as `<|0.00|> text <|end|>`
    (no `<|notimestamps|>`), so the fine-tuned model keeps the timestamps the subtitle API needs.
    """

    def __init__(self, processor, timestamp_fraction: float = 0.0, seed: int = 0) -> None:
        """Prepare the tokens used to build labels.

        Args:
            processor: WhisperProcessor created with language="khmer", task="transcribe".
            timestamp_fraction: probability (0-1) that a label is written with timestamps.
            seed: seed for that random choice, so runs are reproducible.
        """
        self.processor = processor
        self.timestamp_fraction = timestamp_fraction
        # Own random generator, so the choice doesn't depend on (or change) the global one.
        self.rng = random.Random(seed)
        # The tokenizer's normal prefix is <|startoftranscript|><|km|><|transcribe|><|notimestamps|>.
        # Labels with timestamps use the same prefix without <|notimestamps|>.
        tokenizer = processor.tokenizer
        no_timestamps_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self.timestamp_prefix = [token for token in tokenizer.prefix_tokens if token != no_timestamps_id]

    def make_labels(self, text: str, seconds: float) -> List[int]:
        """Turn a transcript into the token ids the decoder must learn to output.

        Args:
            text: normalized transcript.
            seconds: clip length, used as the end timestamp.

        Returns:
            Either `<|startoftranscript|><|km|><|transcribe|><|notimestamps|> text <|endoftext|>`
            or, with probability `timestamp_fraction`,
            `<|startoftranscript|><|km|><|transcribe|><|0.00|> text <|end|><|endoftext|>`.
        """
        tokenizer = self.processor.tokenizer
        if self.rng.random() >= self.timestamp_fraction:
            # Plain label: the tokenizer adds the prefix and <|endoftext|> itself.
            return tokenizer(text).input_ids
        # Whisper timestamps come in 0.02 s steps and can't go past 30 s, so round the clip
        # length to the nearest step and cap it.
        end = min(round(seconds / 0.02) * 0.02, MAX_AUDIO_SECONDS)
        body = tokenizer.encode(f"<|0.00|>{text}<|{end:.2f}|>", add_special_tokens=False)
        return self.timestamp_prefix + body + [tokenizer.eos_token_id]

    def __call__(self, audio: bytes, text: str) -> Optional[Dict[str, Any]]:
        """Build one training example from a clip and its transcript.

        Args:
            audio: encoded audio bytes (FLAC from the parquet files).
            text: the clip's transcript.

        Returns:
            dict with `input_features` (80 x 3000 log-mel spectrogram) and `labels` (token ids),
            or None if the clip can't be used.
        """
        array = load_audio(audio)
        # Whisper's encoder sees at most 30 s; longer audio would be cut and no longer match the text.
        if len(array) == 0 or len(array) > MAX_AUDIO_SECONDS * SAMPLING_RATE:
            return None
        labels = self.make_labels(normalize_text(text), len(array) / SAMPLING_RATE)
        # The decoder has only 448 positions, so a longer label can't be trained on.
        if len(labels) > MAX_LABEL_TOKENS:
            return None
        # Log-mel spectrogram, padded with silence to 30 s; this is what the encoder reads.
        features = self.processor.feature_extractor(array, sampling_rate=SAMPLING_RATE).input_features[0]
        return {"input_features": features, "labels": labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Groups examples from `FeatureMaker` into one batch of tensors for the Trainer.

    Attributes:
        processor: WhisperProcessor, used to pad features and labels.
        decoder_start_token_id: id of <|startoftranscript|>.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Stack a list of examples into a batch.

        Args:
            features: examples, each with `input_features` and `labels`.

        Returns:
            dict with `input_features` (batch x 80 x 3000) and `labels` (batch x longest label),
            where label padding is -100.
        """
        # Spectrograms are all already 30 s long, so this just stacks them into one tensor.
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Labels differ in length: pad them to the longest one, then mark the padding as -100,
        # the value PyTorch's cross-entropy loss ignores, so padding doesn't count as an error.
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.eq(0), -100)

        # The model adds <|startoftranscript|> itself when it shifts labels into decoder inputs,
        # so drop it here or it would appear twice.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_asr_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """WER, CER, and CER with spaces removed (Khmer spacing is inconsistent, so that one is fairest).

    CER = (substituted + deleted + inserted characters) / characters in the reference. It can
    exceed 1.0 when the model outputs much more text than the reference (e.g. repeating itself).

    Args:
        predictions: model transcripts.
        references: correct transcripts, in the same order.

    Returns:
        dict with `wer`, `cer` and `cer_no_space`, each a fraction (0.25 = 25%).
    """
    # Imported here, not at the top, so build_dataset.py can import this module without `evaluate`.
    import evaluate

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    # Normalize both sides the same way so formatting differences don't count as errors.
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
