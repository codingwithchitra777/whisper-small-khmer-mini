# CLAUDE.md

Final project for an AI course (brief: `docs/FinalProject.png`, due 2026-10-05): a Khmer
speech-to-subtitles web app with a fine-tuned `openai/whisper-small`. It is a slimmed-down copy
of the thesis project at `C:\workspace_thesis` (which trains on a much larger dataset) — keep the
two separate; don't import from the thesis code. Git repo: `github.com/codingwithchitra777/whisper-small-khmer-mini` (`main`);
data parquet, `models/` and `outputs/` are gitignored.

## Layout and commands

- `src/` is a package run with `python -m src.<module>` from the project root (no sys.path hacks).
  - `config.py` — all paths and the dataset source list (`SOURCES`, with licenses). Raw audio is
    read from the thesis workspace (`KHMER_SOURCE_ROOT`).
  - `speech.py` — shared audio/text/feature/metric helpers. Resampling uses
    `scipy.signal.resample_poly`; don't switch back to librosa (0.10 fails to import without
    `pkg_resources`).
  - `build_dataset.py` → `data/khmer-asr-mini/` (parquet, FLAC bytes). Splits are assigned by
    hashing each sentence's text, so a sentence never appears in two splits; training drops any
    sentence also in `ddd_test`. Preserve both properties if you change it.
  - `train.py` (`--sources` = ablation, `--time-limit-hours` + `--resume` for Kaggle's 12 h
    sessions; resume needs torch >= 2.6), `evaluate.py` (`--split test|ddd_test|validation`).
- `kaggle/package.py` builds the data and code zips; `kaggle/train_khmer_whisper.ipynb` runs
  smoke / train / ablation / evaluate on Kaggle T4 ×2. If you add a module under `src/`, it is
  packaged automatically (all `src/*.py`).
- `runpod/` is the RunPod alternative, independent of `kaggle/`: `runpod/package.py` builds the
  zips into `outputs/runpod/` (code zip includes `runpod/run.sh`); `runpod/run.sh` runs
  setup / smoke / train / ablation / evaluate / pack on one GPU under `/workspace`, with long
  runs in the background (logs in `/workspace/logs`).
- `api/main.py` (FastAPI) loads `models/whisper-small-khmer-mini` or `KHMER_MODEL_DIR`; it cuts
  audio at pauses into ≤10 s chunks because Khmer uses ~400 Whisper tokens per 15 s of read speech
  (more for fast news) against the 448-token decoder limit; 15 s chunks garbled sentence endings.
  Chunks are transcribed 8 per `generate()` call in fp16 on GPU (per-token overhead dominates on a
  laptop GPU). `webapp/` is the Next.js front end (proxies to the API on port 8000). Public demo:
  `tools/cloudflared.exe tunnel --url http://localhost:3000` (quick tunnel). Cloudflare cuts any
  request at 100 s, so the page uses background jobs: `POST /jobs` returns at once, the page polls
  `GET /jobs/{id}` (status, chunks done/total, segments). One worker thread runs jobs in order (one
  GPU); jobs live in memory for an hour. `/process` still exists but waits for the whole video.

Khmer transcripts: compare with `cer_no_space` (spacing is inconsistent across sources).
