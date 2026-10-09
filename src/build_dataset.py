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

SHARD_ROWS = 5000  # clips per train-*.parquet file, so no single file gets too big
VALIDATION_PERCENT = 5
TEST_PERCENT = 5  # the remaining 90% is train
# <|startoftranscript|> <|km|> <|transcribe|> + <|notimestamps|> or two timestamps + <|endoftext|>
SPECIAL_LABEL_TOKENS = 6


def read_transcripts(name: str, keep_openslr_spaces: bool) -> List[Dict]:
    """List the clips of one source from its `line_index.tsv`.

    Each line of that file is `<clip id><delimiter><transcript>`, and the audio is at
    `wavs/<clip id>.wav`. Audio isn't read here; `load_clip` does that in parallel later.

    Args:
        name: a key of `config.SOURCES`, e.g. "openslr42".
        keep_openslr_spaces: keep OpenSLR 42's word-segmentation spaces (off by default).

    Returns:
        One dict per clip with `id` ("<source>/<clip id>"), `source`, normalized `text` and the
        WAV `path`.
    """
    folder, delimiter, _, _ = SOURCES[name]
    folder = SOURCE_ROOT / folder
    rows = []
    # utf-8-sig drops the byte-order mark some of these files start with.
    for line in (folder / "line_index.tsv").read_text(encoding="utf-8-sig").splitlines():
        line = line.strip("\r\n")
        if not line.strip():
            continue
        # Split at the first delimiter only; the transcript itself may contain spaces.
        clip_id, _, text = line.partition(delimiter)
        text = normalize_text(text)
        if name == "openslr42" and not keep_openslr_spaces:
            text = remove_spaces(text)  # its spaces are word-segmentation marks, unlike every other source
        rows.append({"id": f"{name}/{clip_id.strip()}", "source": name, "text": text,
                     "path": str(folder / "wavs" / f"{clip_id.strip()}.wav")})
    return rows


def load_clip(row: Dict) -> Dict:
    """Read one WAV, check it, and re-encode it as 16 kHz mono FLAC.

    Runs in worker processes (see `main`), so it reports problems in its return value
    instead of raising.

    Args:
        row: a dict from `read_transcripts`.

    Returns:
        The finished dataset row (id, source, text, duration, audio), or `{"error": message}`
        if the file can't be read, or `{"error": None}` if the clip is skipped on purpose
        (empty transcript, or shorter than 0.3 s / longer than 30 s).
    """
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
    """Load the out-of-domain test set: 2,034 DDD clips from speakers no training source has.

    It is never split or trained on; it only measures how well the model handles new voices.

    Returns:
        DataFrame with the same columns as the other splits (id, source="ddd_test", text,
        duration, audio as 16 kHz FLAC).
    """
    frame = pd.read_parquet(DDD_TEST_FILE)
    rows = []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="ddd_test"):
        # Audio is stored as {"bytes": ...} (Hugging Face datasets format); convert it like the
        # other sources so every split has the same sampling rate and encoding.
        array = load_audio(row["audio"]["bytes"])
        rows.append({"id": f"ddd/{row['speaker_id']}/{row['sentence_id'].replace(' ', '')}", "source": "ddd_test",
                     "text": normalize_text(row["transcript"]), "duration": round(len(array) / SAMPLING_RATE, 3),
                     "audio": encode_flac(array)})
    return pd.DataFrame(rows)


def split_of(text: str) -> str:
    """Choose the split for a transcript: about 5% test, 5% validation, 90% train.

    The choice depends only on the text (MD5 hash without spaces -> bucket 0-99), not on random
    chance. So when several speakers read the same sentence, all of their clips land in the same
    split, and the test sets never contain a sentence the model was trained on. It also gives
    the same splits every time the dataset is rebuilt.

    Args:
        text: the clip's transcript.

    Returns:
        "test", "validation" or "train".
    """
    bucket = int(hashlib.md5(remove_spaces(text).encode("utf-8")).hexdigest(), 16) % 100
    if bucket < TEST_PERCENT:
        return "test"
    if bucket < TEST_PERCENT + VALIDATION_PERCENT:
        return "validation"
    return "train"


def hours(frame: pd.DataFrame) -> float:
    """Return the total audio length of `frame` in hours, rounded to 2 decimals."""
    return round(float(frame["duration"].sum()) / 3600, 2)


def write_card(summary: Dict) -> None:
    """Write `README.md` (the dataset card shown on Kaggle) next to the parquet files.

    Args:
        summary: the dict `main` saves as summary.json (clips and hours per split and per
            source, plus each source's license and origin).
    """
    # Overview and a table of clips/hours per split.
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
    # Where each source came from and its license (two are research-only).
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
    """Build the whole dataset from the raw sources.

    Steps: read all transcripts -> load and re-encode the audio in parallel -> drop unusable
    clips -> drop labels too long for Whisper -> drop sentences that are in ddd_test ->
    assign splits -> write the parquet files, summary.json and README.md.
    """
    parser = argparse.ArgumentParser(description="Build the Khmer ASR mini dataset")
    parser.add_argument("--keep-openslr-spaces", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # 1. Read every source's transcripts, then decode/resample/encode the audio on several CPU
    #    cores (one clip at a time is slow for ~20k clips).
    transcripts = [row for name in SOURCES for row in read_transcripts(name, args.keep_openslr_spaces)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(tqdm(pool.map(load_clip, transcripts, chunksize=64), total=len(transcripts), desc="Encoding clips"))
    # 2. Keep the good clips and count the dropped ones (unreadable files vs. clips skipped for
    #    empty text or bad length) for summary.json.
    errors = [result["error"] for result in results if "error" in result and result["error"]]
    frame = pd.DataFrame([result for result in results if "error" not in result])
    dropped = {"unreadable": len(errors), "empty_text_or_bad_length": len(results) - len(frame) - len(errors)}
    if errors:
        print(f"{len(errors)} unreadable clips, first: {errors[0]}")

    # 3. Drop transcripts that need more than 448 tokens with the special tokens included:
    #    Whisper's decoder can't output them, and Khmer uses many tokens per sentence.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    lengths = [len(ids) for ids in tokenizer(frame["text"].tolist(), add_special_tokens=False)["input_ids"]]
    too_long = pd.Series(lengths, index=frame.index) + SPECIAL_LABEL_TOKENS > MAX_LABEL_TOKENS
    dropped["over_label_token_limit"] = int(too_long.sum())
    frame = frame[~too_long]

    # 4. Remove every clip whose sentence also appears in ddd_test, compared without spaces,
    #    so the unseen-speaker test really contains nothing the model trained on.
    ddd_test = load_ddd_test()
    in_ddd_test = frame["text"].map(remove_spaces).isin(set(ddd_test["text"].map(remove_spaces)))
    dropped["text_also_in_ddd_test"] = int(in_ddd_test.sum())
    frame = frame[~in_ddd_test]

    # 5. Assign train/validation/test by sentence (see split_of), then shuffle.
    frame["split"] = frame["text"].map(split_of)
    frame = frame.sample(frac=1, random_state=42).reset_index(drop=True)  # mix sources within shards

    # 6. Write the parquet files. Delete old ones first, so leftover shards from a bigger previous
    #    build aren't read as part of the new train split.
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

    # 7. Statistics for the report: clips and hours per split and per source, and why clips
    #    were dropped. Saved as summary.json and turned into the README dataset card.
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
