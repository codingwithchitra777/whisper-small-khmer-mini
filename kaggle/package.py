"""Build the two Kaggle uploads.

    python kaggle/package.py

- outputs/kaggle/khmer-asr-mini-data.zip  -> Kaggle dataset `khmer-asr-mini-data` (upload once, private)
- outputs/kaggle/khmer-asr-mini-code.zip  -> Kaggle dataset `khmer-asr-mini-code` (re-upload as a new
  version whenever the code changes)
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, OUTPUTS_DIR

CODE_FILES = ["requirements-train.txt"] + [f"src/{path.name}" for path in sorted((ROOT / "src").glob("*.py"))]


def write_zip(path: Path, files, compression) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for source, name in files:
            archive.write(source, arcname=name)
    print(f"Wrote {path} ({path.stat().st_size / 1e9:.2f} GB, {len(files)} files)")


def main() -> None:
    data_files = sorted(DATA_DIR.glob("*.parquet")) + [DATA_DIR / "summary.json", DATA_DIR / "README.md"]
    if not any(path.suffix == ".parquet" for path in data_files):
        raise SystemExit("No dataset yet: run  python -m src.build_dataset")
    # Audio is already FLAC, so storing without compression is as small and much faster.
    write_zip(OUTPUTS_DIR / "kaggle" / "khmer-asr-mini-data.zip",
              [(path, f"khmer-asr-mini/{path.name}") for path in data_files], zipfile.ZIP_STORED)
    write_zip(OUTPUTS_DIR / "kaggle" / "khmer-asr-mini-code.zip",
              [(ROOT / name, name) for name in CODE_FILES], zipfile.ZIP_DEFLATED)


if __name__ == "__main__":
    main()
