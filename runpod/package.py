"""Build the two RunPod uploads.

    python runpod/package.py          # code zip (rebuild after every code change)
    python runpod/package.py --data   # also rebuild the data zip (after build_dataset)

- outputs/runpod/khmer-asr-mini-code.zip  -> src/*.py, requirements-train.txt, runpod/run.sh
- outputs/runpod/khmer-asr-mini-data.zip  -> khmer-asr-mini/*.parquet + summary.json + README.md
Upload both to /workspace on the pod, then follow the steps at the top of runpod/run.sh.
"""

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, OUTPUTS_DIR

OUT_DIR = OUTPUTS_DIR / "runpod"
CODE_FILES = ["requirements-train.txt", "runpod/run.sh"] + [f"src/{path.name}" for path in sorted((ROOT / "src").glob("*.py"))]


def write_zip(path: Path, files, compression) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for source, name in files:
            archive.write(source, arcname=name)
    print(f"Wrote {path} ({path.stat().st_size / 1e9:.2f} GB, {len(files)} files)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="store_true", help="Rebuild the data zip even if it exists")
    args = parser.parse_args()

    write_zip(OUT_DIR / "khmer-asr-mini-code.zip", [(ROOT / name, name) for name in CODE_FILES], zipfile.ZIP_DEFLATED)

    data_zip = OUT_DIR / "khmer-asr-mini-data.zip"
    if data_zip.exists() and not args.data:
        print(f"Kept {data_zip} (pass --data to rebuild)")
        return
    data_files = sorted(DATA_DIR.glob("*.parquet")) + [DATA_DIR / "summary.json", DATA_DIR / "README.md"]
    if not any(path.suffix == ".parquet" for path in data_files):
        raise SystemExit("No dataset yet: run  python -m src.build_dataset")
    # Audio is already FLAC, so storing without compression is as small and much faster.
    write_zip(data_zip, [(path, f"khmer-asr-mini/{path.name}") for path in data_files], zipfile.ZIP_STORED)


if __name__ == "__main__":
    main()
