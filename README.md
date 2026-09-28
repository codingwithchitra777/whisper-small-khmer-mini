# Khmer Speech-to-Text: Subtitles for Khmer Videos

Final project: an AI web app that turns Khmer speech in YouTube videos into timestamped Khmer
subtitles. It fine-tunes `openai/whisper-small` (241M trainable parameters, a Transformer
encoder-decoder) on ~18 hours of collected Khmer speech. Assignment brief: `docs/FinalProject.png`.

## Project layout

```text
src/
  config.py          paths + the list of dataset sources (with licenses)
  speech.py          audio loading/resampling, text normalization, Whisper features, metrics
  build_dataset.py   packs the collected datasets into data/khmer-asr-mini/ (parquet, FLAC audio)
  train.py           fine-tuning (Seq2SeqTrainer); --sources for the ablation
  evaluate.py        WER / CER / CER-without-spaces on the test sets, overall and per source
kaggle/
  package.py                 builds the two Kaggle upload zips (data, code)
  train_khmer_whisper.ipynb  Kaggle notebook: smoke / train / ablation / evaluate
api/main.py          FastAPI backend: YouTube URL -> audio -> pause-based chunks -> subtitles
webapp/              Next.js front end: URL input, video, subtitle timeline, SRT/TXT export
data/khmer-asr-mini/ the packed dataset (built locally, uploaded to Kaggle)
models/              fine-tuned models downloaded from Kaggle
outputs/             evaluation reports, Kaggle zips
```

## Dataset

Built by `python -m src.build_dataset` from datasets collected in the thesis workspace
(`KHMER_SOURCE_ROOT`, default `C:\workspace_thesis`). Full numbers: `data/khmer-asr-mini/summary.json`.

| Source | Clips | Content | License |
| --- | --- | --- | --- |
| `km_speech_corpus` | 14,943 | Read sentences, [seanghay/km-speech-corpus](https://huggingface.co/datasets/seanghay/km-speech-corpus) | CC-BY-4.0 |
| `openslr42` | 2,906 | Read news sentences, one male speaker, [OpenSLR 42](https://openslr.org/42/) | CC-BY-SA-4.0 |
| `kheng_info` | 3,097 | Single words, [seanghay/khmer_kheng_info_speech](https://huggingface.co/datasets/seanghay/khmer_kheng_info_speech) | research use only |
| `mpwt` | 2,058 | Traffic-rule questions, [seanghay/khmer_mpwt_speech](https://huggingface.co/datasets/seanghay/khmer_mpwt_speech) | research use only |
| `rfi_manual` | 83 | Our own clips cut and transcribed from RFI Khmer news | own collection |

| Split | Clips | Hours | Purpose |
| --- | --- | --- | --- |
| train | 20,758 | 15.99 | fine-tuning |
| validation | 1,150 | 0.93 | checkpoint selection during training |
| test | 1,179 | 0.92 | in-domain test (sentences never seen in training) |
| ddd_test | 2,034 | 4.66 | out-of-domain test: 2 unseen speakers + unseen paragraphs from the [Digital Divide Data Khmer ASR dataset](https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset) (CC-BY-SA-4.0) |

**Preparation:** all audio resampled to 16 kHz mono and stored as FLAC; clips outside 0.3–30 s or
with transcripts over Whisper's 448-token limit are dropped; zero-width spaces removed and
whitespace collapsed; OpenSLR's word-segmentation spaces removed so every source uses the same
spacing style; each sentence is assigned to exactly one split by hashing its text (no sentence
leaks between train and test); any training sentence that also appears in `ddd_test` is removed.

Two sources are research-use-only, so the Kaggle dataset must stay **private**.

## Experiments

| Model | Trained on | Role |
| --- | --- | --- |
| `openai/whisper-small`, zero-shot | – | baseline |
| `whisper-small-khmer-mini` | all sources (~16 h) | main model |
| `whisper-small-khmer-openslr-only` | OpenSLR 42 only (~3.6 h, one speaker) | ablation: does more, more varied data help? |

All three are scored on `test` and `ddd_test` with WER, CER and CER without spaces (Khmer
spacing is inconsistent, so CER without spaces is the fairest measure).

## Run it

**Train on Kaggle** (free T4 ×2):
1. `python kaggle/package.py` → upload `outputs/kaggle/khmer-asr-mini-data.zip` and
   `outputs/kaggle/khmer-asr-mini-code.zip` as two private Kaggle datasets.
2. Import `kaggle/train_khmer_whisper.ipynb` into Kaggle, attach both datasets, GPU T4 ×2, Internet on.
3. Commit with `MODE` = `smoke`, then `train`, then `ablation`, then `evaluate` (details in the notebook).
4. Download `models/whisper-small-khmer-mini/` from the `train` version's output into `models/`.

**Demo locally:**
```powershell
python -m venv py-venv; .\py-venv\Scripts\Activate.ps1; pip install -r requirements.txt
.\start-api.ps1      # FastAPI on http://localhost:8000 (loads models/whisper-small-khmer-mini)
.\start-webapp.ps1   # Next.js on http://localhost:3000
```
Set `KHMER_MODEL_DIR` to demo a different model folder.
