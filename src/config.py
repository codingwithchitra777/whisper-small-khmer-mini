"""Paths and dataset sources for the Khmer speech-to-text final project."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "khmer-asr-mini"  # built by src/build_dataset.py, uploaded to Kaggle
MODELS_DIR = ROOT_DIR / "models"
FINE_TUNED_MODEL_DIR = MODELS_DIR / "whisper-small-khmer-mini"
OUTPUTS_DIR = ROOT_DIR / "outputs"
TRAINING_OUTPUTS_DIR = OUTPUTS_DIR / "training"
EVALUATION_OUTPUTS_DIR = OUTPUTS_DIR / "evaluation"

BASE_MODEL = "openai/whisper-small"  # 241M trainable parameters

# Raw collected audio lives in the thesis workspace; build_dataset.py packs it into DATA_DIR.
SOURCE_ROOT = Path(os.environ.get("KHMER_SOURCE_ROOT", r"C:\workspace_thesis"))
DDD_TEST_FILE = SOURCE_ROOT / "data" / "khmer-speech-dataset" / "test.parquet"

# name -> (folder with line_index.tsv + wavs/, transcript delimiter, license, origin)
SOURCES = {
    "openslr42": (
        "src/app/open-slr/km_kh_male",
        "\t\t",
        "CC-BY-SA-4.0",
        "OpenSLR 42 (https://openslr.org/42/), 1 male speaker, word-segmented text",
    ),
    "km_speech_corpus": (
        "src/app/kh-copus/exported_data",
        "\t",
        "CC-BY-4.0",
        "https://huggingface.co/datasets/seanghay/km-speech-corpus",
    ),
    "kheng_info": (
        "src/app/kh-copus/exported_khmer_kheng_info_speech",
        "\t",
        "research use only",
        "https://huggingface.co/datasets/seanghay/khmer_kheng_info_speech (single words, kheng.info)",
    ),
    "mpwt": (
        "src/app/kh-copus/exported_khmer_mpwt_speech",
        "\t",
        "research use only",
        "https://huggingface.co/datasets/seanghay/khmer_mpwt_speech (Ministry of Public Works and Transport app)",
    ),
    "rfi_manual": (
        "src/app/kh-speech-copus",
        " ",
        "own collection (RFI Khmer audio)",
        "Clips cut and transcribed by hand from RFI Khmer news",
    ),
}
