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
        """Store the clips and the converter that turns each one into model inputs.

        Clips are converted lazily in `__getitem__` instead of all at once, so ~16 h of audio
        never has to sit in RAM as log-mel spectrograms.

        Args:
            frame: rows with columns id, source, text, duration, audio (FLAC bytes).
            processor: WhisperProcessor (feature extractor + tokenizer).
            timestamp_fraction: share of labels written with timestamp tokens (0.5 for training,
                0 for evaluation so validation labels are plain text).
            seed: seeds the random choice of which labels get timestamps.
        """
        # Reset the index so row positions 0..N-1 match the indices the DataLoader asks for.
        self.frame = frame.reset_index(drop=True)
        self.make_features = FeatureMaker(processor, timestamp_fraction, seed)

    def __len__(self) -> int:
        """Return the number of clips; the Trainer uses it to count steps per epoch."""
        return len(self.frame)

    def __getitem__(self, index: int):
        """Convert clip `index` into one training example.

        Returns:
            dict with `input_features` (80 x 3000 log-mel spectrogram of the audio, padded to 30 s)
            and `labels` (token ids of the Khmer transcript).

        Raises:
            ValueError: the clip is empty, longer than 30 s, or its text exceeds 448 tokens.
                build_dataset.py already filters these out, so this means the data is stale.
        """
        row = self.frame.iloc[index]
        example = self.make_features(row["audio"], row["text"])
        if example is None:
            raise ValueError(f"Unusable clip {row['id']}; rebuild the dataset with src.build_dataset")
        return example


class TimeLimitCallback(TrainerCallback):
    """Stops and checkpoints before a hard session limit (Kaggle: 12 h), so `--resume` can continue."""

    def __init__(self, hours: float) -> None:
        """Set the deadline `hours` from now.

        Args:
            hours: training time allowed in this session (e.g. 11 on Kaggle, leaving 1 h spare
                to save the checkpoint and outputs).
        """
        self.deadline = time.time() + hours * 3600
        # Read by main() after training to tell "stopped on time limit" apart from "finished".
        self.reached = False

    def on_step_end(self, args, state, control, **kwargs):
        """Called by the Trainer after every optimizer step.

        Once the deadline has passed, it asks the Trainer to save a checkpoint now and stop
        training. Otherwise it leaves `control` unchanged.
        """
        if time.time() >= self.deadline:
            self.reached = True
            control.should_save = True
            control.should_training_stop = True
        return control


def load_split(data_dir: Path, split: str, sources) -> pd.DataFrame:
    """Load one dataset split into a DataFrame.

    Args:
        data_dir: folder written by build_dataset.py.
        split: "train", "validation", "test" or "ddd_test".
        sources: source names to keep (e.g. ["openslr42"] for the ablation); empty keeps all.

    Returns:
        All rows of the split, filtered to `sources` if given.
    """
    # The train split is sharded into train-00000.parquet, train-00001.parquet, ...; the
    # smaller splits are a single file, so fall back to <split>.parquet when no shards exist.
    files = sorted(data_dir.glob(f"{split}-*.parquet")) or [data_dir / f"{split}.parquet"]
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    return frame[frame["source"].isin(sources)] if sources else frame


def build_model(args) -> WhisperForConditionalGeneration:
    """Load the pretrained Whisper model and prepare it for Khmer fine-tuning.

    All 241M parameters stay trainable (full fine-tune, no frozen layers or LoRA).

    Args:
        args: parsed command-line arguments; uses `model_name` and `gradient_checkpointing`.

    Returns:
        The model, set to transcribe Khmer.
    """
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    # Language/task on the generation config (not forced_decoder_ids): generates cleanly on new transformers.
    model.generation_config.language = "khmer"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    if args.gradient_checkpointing:
        # Gradient checkpointing recomputes activations during backprop instead of storing them:
        # less GPU memory, ~20-30% slower. The decoder's key/value cache is useless in training
        # and conflicts with it, so turn the cache off.
        model.config.use_cache = False
        # Non-reentrant: the default (reentrant) mode backpropagates through the encoder graph once per
        # decoder layer (encoder_hidden_states is passed by keyword) and fails on the second layer.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def resume_checkpoint(args):
    """Find the checkpoint to continue from.

    Args:
        args: parsed command-line arguments; uses `resume` and `output_dir`.

    Returns:
        Path of the newest `checkpoint-N` folder in `output_dir` when `--resume` is set and one
        exists, otherwise None (the Trainer then starts from the pretrained model).
    """
    if not args.resume:
        return None
    # Loading the saved optimizer/RNG state needs torch >= 2.6 (torch.load safety change), so
    # fail early with a clear message instead of after the model has loaded.
    if tuple(int(part) for part in re.findall(r"\d+", torch.__version__)[:2]) < (2, 6):
        raise SystemExit(f"--resume needs torch >= 2.6 (found {torch.__version__}); pip install -r requirements-train.txt")
    output_dir = Path(args.output_dir)
    checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
    if checkpoint is None:
        print("No checkpoint found; starting from scratch")
    return checkpoint


def main() -> None:
    """Run one fine-tuning job from the command line.

    Steps: parse arguments -> load processor, model and data -> configure the Trainer ->
    train (evaluating and checkpointing every `--eval-steps`) -> save the best checkpoint,
    judged by validation CER without spaces, to `--final-dir`.
    """
    # 1. Command-line options. The defaults are the main run; run.sh / the notebook override
    #    some of them (e.g. --grad-accum 2 --eval-steps 500 --gradient-checkpointing).
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

    # 2. Processor and model. The processor's tokenizer puts <|km|><|transcribe|> at the start of
    #    every label, so the model learns to emit Khmer transcription.
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    data_dir = Path(args.data_dir)
    processor = WhisperProcessor.from_pretrained(args.model_name, language="khmer", task="transcribe")
    model = build_model(args)

    # 3. Data. Validation is shuffled with a fixed seed and capped at --eval-samples clips, because
    #    evaluation runs full text generation, which is slow; 400 clips take about a minute.
    train_frame = load_split(data_dir, "train", sources)
    eval_frame = load_split(data_dir, "validation", sources).sample(frac=1, random_state=args.seed).head(args.eval_samples)
    train_dataset = SpeechDataset(train_frame, processor, args.timestamp_fraction, args.seed)
    eval_dataset = SpeechDataset(eval_frame, processor)

    # 4. Trainer settings. Mixed precision: bf16 on GPUs that support it (RTX 30xx/40xx, A100),
    #    fp16 otherwise (Kaggle's T4), full fp32 on CPU.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,  # effective batch = batch_size x grad_accum x GPUs
        learning_rate=args.learning_rate,  # small (1e-5) so the pretrained weights aren't wrecked
        warmup_steps=args.warmup_steps,  # ramp the learning rate up from 0, then decay linearly
        num_train_epochs=args.epochs,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,  # must match eval_steps for load_best_model_at_end
        save_total_limit=3,  # keep disk use bounded; the best checkpoint is never deleted
        logging_steps=25,
        predict_with_generate=True,  # evaluate on generated text (real CER), not just the loss
        generation_max_length=MAX_LABEL_TOKENS,
        load_best_model_at_end=True,
        metric_for_best_model="cer_no_space",  # Khmer spacing is inconsistent across sources
        greater_is_better=False,
        dataloader_num_workers=args.num_workers,  # decode FLAC + compute features in parallel
        remove_unused_columns=False,  # our dataset returns plain dicts, nothing to strip
        label_names=["labels"],
        report_to=["tensorboard"],
        seed=args.seed,
    )

    def compute_metrics(pred):
        """Score one evaluation pass; the Trainer calls this after generating on the validation set.

        Args:
            pred: EvalPrediction with `predictions` (generated token ids) and `label_ids`
                (reference token ids), both padded with -100.

        Returns:
            dict with wer, cer and cer_no_space (the Trainer logs them as eval_wer, ...).
        """
        # -100 marks padding (ignored by the loss) but isn't a real token id, so the tokenizer
        # can't decode it; swap it for the pad token, which skip_special_tokens then drops.
        pad_token_id = processor.tokenizer.pad_token_id
        prediction_ids, label_ids = pred.predictions, pred.label_ids
        prediction_ids[prediction_ids == -100] = pad_token_id
        label_ids[label_ids == -100] = pad_token_id
        return compute_asr_metrics(
            processor.batch_decode(prediction_ids, skip_special_tokens=True),
            processor.batch_decode(label_ids, skip_special_tokens=True),
        )

    # 5. The Trainer runs the loop: batches, forward/backward pass, optimizer (AdamW), learning-rate
    #    schedule, evaluation and checkpoints. The collator pads each batch's labels with -100.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics,
    )

    # 6. Record what this run used (settings, data size, GPUs) for the report. Only the main
    #    process writes files when training on several GPUs with torchrun.
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

    # 7. Train, optionally under a session time limit and/or continuing from the last checkpoint.
    time_limit = TimeLimitCallback(args.time_limit_hours) if args.time_limit_hours else None
    if time_limit:
        trainer.add_callback(time_limit)
    trainer.train(resume_from_checkpoint=resume_checkpoint(args))

    # Stopped early by the time limit: the checkpoint is saved, but the model isn't finished,
    # so don't write it to --final-dir yet.
    if time_limit and time_limit.reached:
        print(f"Time limit reached at step {trainer.state.global_step}/{trainer.state.max_steps}; checkpoint saved.")
        print(f"Continue with the same command plus --resume (checkpoints in {output_dir})")
        return

    # 8. Save the final model. load_best_model_at_end has already swapped in the checkpoint with
    #    the lowest validation cer_no_space, so this saves the best one, not the last one.
    trainer.save_model(args.final_dir)  # collective under torchrun: every process calls it
    if not trainer.is_world_process_zero():
        return
    # Save the processor next to the weights so the API can load both from one folder, plus the
    # full loss/metric history for plotting training curves.
    processor.save_pretrained(args.final_dir)
    (Path(args.final_dir) / "training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint} ({trainer.state.best_metric})")
    print(f"Saved model to {args.final_dir}")


if __name__ == "__main__":
    # Avoid the tokenizers fork warning/deadlock when DataLoader workers start.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
