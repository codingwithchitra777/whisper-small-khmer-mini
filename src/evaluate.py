"""Score a Whisper model on the mini dataset's test sets.

    python -m src.evaluate --model openai/whisper-small --split test        # baseline
    python -m src.evaluate --model models/whisper-small-khmer-mini --split ddd_test

`test` = held-out sentences from the collected sources; `ddd_test` = unseen speakers and sentences
from the Digital Divide Data Khmer dataset (out-of-domain).
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config import DATA_DIR, EVALUATION_OUTPUTS_DIR, FINE_TUNED_MODEL_DIR
from src.speech import MAX_LABEL_TOKENS, SAMPLING_RATE, compute_asr_metrics, load_audio


def load_model(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    # Older checkpoints saved forced_decoder_ids and a list eos id, which clash with language= in generate().
    model.generation_config.forced_decoder_ids = None
    if isinstance(model.generation_config.eos_token_id, list):
        model.generation_config.eos_token_id = model.generation_config.eos_token_id[0]
    return processor, model, device, dtype


def transcribe(frame: pd.DataFrame, processor, model, device, dtype, batch_size: int) -> list:
    predictions = []
    for start in tqdm(range(0, len(frame), batch_size), desc="Transcribing"):
        arrays = [load_audio(audio) for audio in frame["audio"].iloc[start : start + batch_size]]
        features = processor.feature_extractor(arrays, sampling_rate=SAMPLING_RATE, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(features.to(device, dtype=dtype), language="khmer", task="transcribe",
                                 max_new_tokens=MAX_LABEL_TOKENS - 4)
        predictions.extend(processor.batch_decode(ids, skip_special_tokens=True))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Whisper model on the Khmer mini dataset")
    parser.add_argument("--model", default=str(FINE_TUNED_MODEL_DIR), help="Local path or Hugging Face id")
    parser.add_argument("--split", choices=["test", "ddd_test", "validation"], default="test")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--limit", type=int, default=0, help="Only score the first N clips")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default="", help="Report file name under outputs/evaluation/")
    args = parser.parse_args()

    frame = pd.read_parquet(Path(args.data_dir) / f"{args.split}.parquet")
    if args.limit:
        frame = frame.head(args.limit)

    processor, model, device, dtype = load_model(args.model)
    print(f"Device: {device} | model: {args.model} | {len(frame)} {args.split} clips")
    frame = frame.drop(columns=["audio"]).assign(prediction=transcribe(frame, processor, model, device, dtype, args.batch_size))

    report = {
        "model": args.model,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "split": args.split,
        "clips": len(frame),
        "hours": round(float(frame["duration"].sum()) / 3600, 2),
        "metrics": compute_asr_metrics(list(frame["prediction"]), list(frame["text"])),
        "per_source": {
            source: {"clips": len(rows), **compute_asr_metrics(list(rows["prediction"]), list(rows["text"]))}
            for source, rows in frame.groupby("source")
        },
        "samples": frame.head(20)[["source", "text", "prediction"]].to_dict("records"),
    }
    for name, value in report["metrics"].items():
        print(f"{name}: {value:.4f}")

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(args.model).name or args.model)
    output_path = EVALUATION_OUTPUTS_DIR / (args.output or f"{slug}_{args.split}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
