"""Pack the collected Khmer speech datasets into Kaggle-ready parquet files.

    python -m src.build_dataset

Reads the 5 labelled sources listed in `config.SOURCES` (~17.9 h) from the thesis workspace, plus
the 2,034-clip unseen-speaker test sample of the Digital Divide Data (DDD) Khmer dataset, and
writes `data/khmer-asr-mini/`:

- train-0000N.parquet, validation.parquet, test.parquet  (columns: id, source, text, duration, audio)
- ddd_test.parquet  (out-of-domain test: speakers and sentences none of the sources contain)
- summary.json, README.md  (dataset card for Kaggle)

Audio is re-encoded to 16 kHz mono FLAC. A sentence always lands in exactly one split (assigned
by hashing its text without spaces), and training drops any sentence that is also in ddd_test.
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config import BASE_MODEL, DATA_DIR, DDD_TEST_FILE, SOURCE_ROOT, SOURCES
from src.speech import (
    MAX_AUDIO_SECONDS,
    MAX_LABEL_TOKENS,
    MIN_AUDIO_SECONDS,
    SAMPLING_RATE,
    encode_flac,
    load_audio,
    normalize_text,
    remove_spaces,
)

SHARD_ROWS = 5000
VALIDATION_PERCENT = 5
TEST_PERCENT = 5
# <|startoftranscript|> <|km|> <|transcribe|> + <|notimestamps|> or two timestamps + <|endoftext|>
SPECIAL_LABEL_TOKENS = 6


def read_transcripts(name: str, keep_openslr_spaces: bool) -> List[Dict]:
    folder, delimiter, _, _ = SOURCES[name]
    folder = SOURCE_ROOT / folder
    rows = []
    for line in (folder / "line_index.tsv").read_text(encoding="utf-8-sig").splitlines():
        line = line.strip("\r\n")
        if not line.strip():
            continue
        clip_id, _, text = line.partition(delimiter)
        text = normalize_text(text)
        if name == "openslr42" and not keep_openslr_spaces:
            text = remove_spaces(text)  # its spaces are word-segmentation marks, unlike every other source
        rows.append({"id": f"{name}/{clip_id.strip()}", "source": name, "text": text,
                     "path": str(folder / "wavs" / f"{clip_id.strip()}.wav")})
    return rows


def load_clip(row: Dict) -> Dict:
    try:
        array = load_audio(row["path"])
    except Exception as error:
        return {"error": f"{row['source']}: {type(error).__name__}: {error}"}
    duration = len(array) / SAMPLING_RATE
    if not row["text"] or not MIN_AUDIO_SECONDS <= duration <= MAX_AUDIO_SECONDS:
        return {"error": None}  # empty text or unusable length
    return {"id": row["id"], "source": row["source"], "text": row["text"],
            "duration": round(duration, 3), "audio": encode_flac(array)}


def load_ddd_test() -> pd.DataFrame:
    frame = pd.read_parquet(DDD_TEST_FILE)
    rows = []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="ddd_test"):
        array = load_audio(row["audio"]["bytes"])
        rows.append({"id": f"ddd/{row['speaker_id']}/{row['sentence_id'].replace(' ', '')}", "source": "ddd_test",
                     "text": normalize_text(row["transcript"]), "duration": round(len(array) / SAMPLING_RATE, 3),
                     "audio": encode_flac(array)})
    return pd.DataFrame(rows)


def split_of(text: str) -> str:
    bucket = int(hashlib.md5(remove_spaces(text).encode("utf-8")).hexdigest(), 16) % 100
    if bucket < TEST_PERCENT:
        return "test"
    if bucket < TEST_PERCENT + VALIDATION_PERCENT:
        return "validation"
    return "train"


def hours(frame: pd.DataFrame) -> float:
    return round(float(frame["duration"].sum()) / 3600, 2)


def write_card(summary: Dict) -> None:
    lines = [
        "# Khmer ASR mini dataset",
        "",
        "Short Khmer speech clips with transcripts, 16 kHz mono FLAC, packed for fine-tuning Whisper.",
        "Columns: `id`, `source`, `text`, `duration` (seconds), `audio` (FLAC bytes).",
        "",
        "| Split | Clips | Hours |",
        "| --- | --- | --- |",
    ]
    for split, info in summary["splits"].items():
        lines.append(f"| {split} | {info['clips']:,} | {info['hours']} |")
    lines += ["", "## Sources", "", "| Source | Clips | Hours | License | Origin |", "| --- | --- | --- | --- | --- |"]
    for name, info in summary["sources"].items():
        lines.append(f"| {name} | {info['clips']:,} | {info['hours']} | {info['license']} | {info['origin']} |")
    lines += [
        "",
        "`ddd_test` is a 2,034-clip sample of the Digital Divide Data Khmer ASR Cultural Dataset "
        "(https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset, CC-BY-SA-4.0): two speakers "
        "and paragraphs held out, used as an out-of-domain test set only.",
        "",
        "Some sources are licensed for research use only, so keep this dataset private.",
    ]
    (DATA_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Khmer ASR mini dataset")
    parser.add_argument("--keep-openslr-spaces", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    transcripts = [row for name in SOURCES for row in read_transcripts(name, args.keep_openslr_spaces)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(tqdm(pool.map(load_clip, transcripts, chunksize=64), total=len(transcripts), desc="Encoding clips"))
    errors = [result["error"] for result in results if "error" in result and result["error"]]
    frame = pd.DataFrame([result for result in results if "error" not in result])
    dropped = {"unreadable": len(errors), "empty_text_or_bad_length": len(results) - len(frame) - len(errors)}
    if errors:
        print(f"{len(errors)} unreadable clips, first: {errors[0]}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    lengths = [len(ids) for ids in tokenizer(frame["text"].tolist(), add_special_tokens=False)["input_ids"]]
    too_long = pd.Series(lengths, index=frame.index) + SPECIAL_LABEL_TOKENS > MAX_LABEL_TOKENS
    dropped["over_label_token_limit"] = int(too_long.sum())
    frame = frame[~too_long]

    ddd_test = load_ddd_test()
    in_ddd_test = frame["text"].map(remove_spaces).isin(set(ddd_test["text"].map(remove_spaces)))
    dropped["text_also_in_ddd_test"] = int(in_ddd_test.sum())
    frame = frame[~in_ddd_test]

    frame["split"] = frame["text"].map(split_of)
    frame = frame.sample(frac=1, random_state=42).reset_index(drop=True)  # mix sources within shards

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for old in DATA_DIR.glob("*.parquet"):
        old.unlink()
    columns = ["id", "source", "text", "duration", "audio"]
    train = frame[frame["split"] == "train"]
    for shard, start in enumerate(range(0, len(train), SHARD_ROWS)):
        train.iloc[start : start + SHARD_ROWS][columns].to_parquet(DATA_DIR / f"train-{shard:05d}.parquet", index=False)
    for split in ("validation", "test"):
        frame[frame["split"] == split][columns].to_parquet(DATA_DIR / f"{split}.parquet", index=False)
    ddd_test[columns].to_parquet(DATA_DIR / "ddd_test.parquet", index=False)

    splits = {split: frame[frame["split"] == split] for split in ("train", "validation", "test")}
    splits["ddd_test"] = ddd_test
    summary = {
        "splits": {name: {"clips": len(part), "hours": hours(part),
                          "per_source": part.groupby("source").size().to_dict()} for name, part in splits.items()},
        "sources": {
            name: {"clips": int((frame["source"] == name).sum()), "hours": hours(frame[frame["source"] == name]),
                   "license": SOURCES[name][2], "origin": SOURCES[name][3]}
            for name in SOURCES
        },
        "dropped": dropped,
        "openslr_spaces_removed": not args.keep_openslr_spaces,
    }
    (DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_card(summary)
    print(json.dumps({k: summary[k] for k in ("splits", "dropped")}, indent=2))
    size = sum(path.stat().st_size for path in DATA_DIR.glob("*")) / 1e9
    print(f"Wrote {DATA_DIR} ({size:.2f} GB)")


if __name__ == "__main__":
    main()
