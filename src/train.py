"""Fine-tune Whisper-small on the Khmer ASR mini dataset.

    python -m src.train                          # all sources, 5 epochs
    python -m src.train --sources openslr42      # ablation: OpenSLR 42 only

Built for Kaggle's 12 h sessions: `--time-limit-hours` checkpoints and stops, `--resume` continues.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer_utils import get_last_checkpoint

from src.config import BASE_MODEL, DATA_DIR, FINE_TUNED_MODEL_DIR, TRAINING_OUTPUTS_DIR
from src.speech import MAX_LABEL_TOKENS, DataCollatorSpeechSeq2SeqWithPadding, FeatureMaker, compute_asr_metrics


class SpeechDataset(torch.utils.data.Dataset):
    """Parquet rows (FLAC bytes + text) turned into Whisper features on the fly."""

    def __init__(self, frame: pd.DataFrame, processor, timestamp_fraction: float = 0.0, seed: int = 0) -> None:
        self.frame = frame.reset_index(drop=True)
        self.make_features = FeatureMaker(processor, timestamp_fraction, seed)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        example = self.make_features(row["audio"], row["text"])
        if example is None:
            raise ValueError(f"Unusable clip {row['id']}; rebuild the dataset with src.build_dataset")
        return example


class TimeLimitCallback(TrainerCallback):
    """Stops and checkpoints before a hard session limit (Kaggle: 12 h), so `--resume` can continue."""

    def __init__(self, hours: float) -> None:
        self.deadline = time.time() + hours * 3600
        self.reached = False

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() >= self.deadline:
            self.reached = True
            control.should_save = True
            control.should_training_stop = True
        return control


def load_split(data_dir: Path, split: str, sources) -> pd.DataFrame:
    files = sorted(data_dir.glob(f"{split}-*.parquet")) or [data_dir / f"{split}.parquet"]
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    return frame[frame["source"].isin(sources)] if sources else frame


def build_model(args) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    # Language/task on the generation config (not forced_decoder_ids): generates cleanly on new transformers.
    model.generation_config.language = "khmer"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    if args.gradient_checkpointing:
        model.config.use_cache = False
    return model


def resume_checkpoint(args):
    if not args.resume:
        return None
    if tuple(int(part) for part in re.findall(r"\d+", torch.__version__)[:2]) < (2, 6):
        raise SystemExit(f"--resume needs torch >= 2.6 (found {torch.__version__}); pip install -r requirements-train.txt")
    output_dir = Path(args.output_dir)
    checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
    if checkpoint is None:
        print("No checkpoint found; starting from scratch")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on the Khmer ASR mini dataset")
    parser.add_argument("--model-name", default=BASE_MODEL)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--sources", default="", help="Comma-separated sources to train on (default: all)")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16, help="Per GPU")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=250, help="Evaluate and checkpoint every N steps")
    parser.add_argument("--eval-samples", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--timestamp-fraction", type=float, default=0.5,
                        help="Share of labels with timestamp tokens; keeps return_timestamps=True working in the API")
    parser.add_argument("--output-dir", default=str(TRAINING_OUTPUTS_DIR / "whisper-small-khmer-mini"))
    parser.add_argument("--final-dir", default=str(FINE_TUNED_MODEL_DIR))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--time-limit-hours", type=float, default=0, help="0 = no limit; 11 on Kaggle")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    data_dir = Path(args.data_dir)
    processor = WhisperProcessor.from_pretrained(args.model_name, language="khmer", task="transcribe")
    model = build_model(args)

    train_frame = load_split(data_dir, "train", sources)
    eval_frame = load_split(data_dir, "validation", sources).sample(frac=1, random_state=args.seed).head(args.eval_samples)
    train_dataset = SpeechDataset(train_frame, processor, args.timestamp_fraction, args.seed)
    eval_dataset = SpeechDataset(eval_frame, processor)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=3,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=MAX_LABEL_TOKENS,
        load_best_model_at_end=True,
        metric_for_best_model="cer_no_space",
        greater_is_better=False,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=["tensorboard"],
        seed=args.seed,
    )

    def compute_metrics(pred):
        pad_token_id = processor.tokenizer.pad_token_id
        prediction_ids, label_ids = pred.predictions, pred.label_ids
        prediction_ids[prediction_ids == -100] = pad_token_id
        label_ids[label_ids == -100] = pad_token_id
        return compute_asr_metrics(
            processor.batch_decode(prediction_ids, skip_special_tokens=True),
            processor.batch_decode(label_ids, skip_special_tokens=True),
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics,
    )

    output_dir = Path(args.output_dir)
    if trainer.is_world_process_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        run_config = {
            **vars(args),
            "sources_used": sorted(train_frame["source"].unique()),
            "train_clips": len(train_frame),
            "train_hours": round(float(train_frame["duration"].sum()) / 3600, 2),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or ["cpu"],
        }
        (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        print(json.dumps(run_config, indent=2))

    time_limit = TimeLimitCallback(args.time_limit_hours) if args.time_limit_hours else None
    if time_limit:
        trainer.add_callback(time_limit)
    trainer.train(resume_from_checkpoint=resume_checkpoint(args))

    if time_limit and time_limit.reached:
        print(f"Time limit reached at step {trainer.state.global_step}/{trainer.state.max_steps}; checkpoint saved.")
        print(f"Continue with the same command plus --resume (checkpoints in {output_dir})")
        return

    trainer.save_model(args.final_dir)  # collective under torchrun: every process calls it
    if not trainer.is_world_process_zero():
        return
    processor.save_pretrained(args.final_dir)
    (Path(args.final_dir) / "training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint} ({trainer.state.best_metric})")
    print(f"Saved model to {args.final_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
