#!/usr/bin/env bash
# Train and evaluate on a RunPod GPU pod. Ships inside the code zip built by runpod/package.py.
# Upload outputs/runpod/khmer-asr-mini-code.zip and khmer-asr-mini-data.zip to /workspace.
# Everything lives under /workspace, which survives stopping the pod (not terminating it).
#
#   cd /workspace && python -m zipfile -e khmer-asr-mini-code.zip project
#   bash project/runpod/run.sh setup      # after every pod start: install packages, unpack data
#   bash project/runpod/run.sh smoke      # ~5 min check, prints the speed
#   bash project/runpod/run.sh train      # main model, all sources (background)
#   bash project/runpod/run.sh ablation   # OpenSLR 42 only (background)
#   bash project/runpod/run.sh evaluate   # baseline, main, ablation on test + ddd_test (background)
#   bash project/runpod/run.sh status     # GPU use and the end of the latest log
#   bash project/runpod/run.sh pack       # /workspace/results.zip to download
#
# Overrides: EPOCHS=5 BATCH=16 GRAD_ACCUM=2 GC=1 NUM_WORKERS=8 EVAL_LIMIT=0
set -euo pipefail
SCRIPT=$(readlink -f "$0")

WORK=/workspace
PROJECT=$WORK/project
LOGS=$WORK/logs
EPOCHS=${EPOCHS:-5}
BATCH=${BATCH:-16}
GRAD_ACCUM=${GRAD_ACCUM:-2}  # 16 x 2 = effective batch 32, on one GPU
GC=${GC:-1}                  # gradient checkpointing: less memory, ~30% slower
NUM_WORKERS=${NUM_WORKERS:-8}
EVAL_LIMIT=${EVAL_LIMIT:-0}

data_dir() {
  local summary
  summary=$(find $WORK/data -name summary.json 2>/dev/null | head -1)
  [ -n "$summary" ] || { echo "No dataset under $WORK/data: run setup first" >&2; exit 1; }
  dirname "$summary"
}

common_args() {
  echo "--data-dir $(data_dir) --batch-size $BATCH --grad-accum $GRAD_ACCUM --num-workers $NUM_WORKERS" \
       "$([ "$GC" = 1 ] && echo --gradient-checkpointing)"
}

setup() {
  cd $PROJECT
  # The Pytorch 2.8 template (Ubuntu 24.04) marks the system Python externally managed (PEP 668), and torch
  # lives there, so install into it instead of a venv. Older pips ignore this variable.
  PIP_BREAK_SYSTEM_PACKAGES=1 pip install -q -r requirements-train.txt
  python - <<'EOF'
import torch
assert torch.cuda.is_available(), "No GPU visible"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
assert tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 6), "torch < 2.6: pick a newer PyTorch template"
EOF
  if ! find $WORK/data -name summary.json 2>/dev/null | grep -q .; then
    mkdir -p $WORK/data
    [ -f $WORK/khmer-asr-mini-data.zip ] || { echo "Upload khmer-asr-mini-data.zip to $WORK first" >&2; exit 1; }
    echo "Unpacking $WORK/khmer-asr-mini-data.zip ..."
    python -m zipfile -e $WORK/khmer-asr-mini-data.zip $WORK/data
  fi
  echo "Data: $(data_dir)"
  python -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['splits'], indent=1)[:1500])" "$(data_dir)/summary.json"
}

train_run() {  # checkpoint dir, final dir, extra args
  python -m src.train $(common_args) --epochs $EPOCHS --eval-steps 500 \
    --output-dir "$1" --final-dir "$2" --resume $3
  if [ -f "$2/config.json" ]; then
    echo "Finished: $2"
    cp "$1/run_config.json" "$2/" 2>/dev/null || true  # keep the hyperparameters with the model (and in pack)
    rm -rf "$1"  # checkpoints no longer needed
  fi
}

evaluate() {
  local split model label limit=""
  [ "$EVAL_LIMIT" != 0 ] && limit="--limit $EVAL_LIMIT"
  for label in baseline main ablation; do
    case $label in
      baseline) model=openai/whisper-small ;;
      main) model=$WORK/models/whisper-small-khmer-mini ;;
      ablation) model=$WORK/models/whisper-small-khmer-openslr-only ;;
    esac
    if [ "$label" != baseline ] && [ ! -f "$model/config.json" ]; then
      echo "Skipping $label: $model not trained yet"
      continue
    fi
    for split in test ddd_test; do
      python -m src.evaluate --model "$model" --split $split --data-dir "$(data_dir)" --batch-size 32 $limit \
        --output ${label}_${split}.json
    done
  done
  python - <<'EOF'
import json
from pathlib import Path
print(f"{'run':24s} {'clips':>6s} {'WER':>8s} {'CER':>8s} {'CER no space':>13s}")
for report in sorted(Path("outputs/evaluation").glob("*.json")):
    data = json.loads(report.read_text())
    m = data["metrics"]
    print(f"{report.stem:24s} {data['clips']:6d} {m['wer']:8.4f} {m['cer']:8.4f} {m['cer_no_space']:13.4f}")
EOF
}

# Long modes re-run this script in the background, so closing the browser tab doesn't stop them.
background() {
  mkdir -p $LOGS
  local log=$LOGS/$1.log
  RUNPOD_FOREGROUND=1 nohup bash "$SCRIPT" "$1" >> "$log" 2>&1 &
  echo "Started $1 in the background (pid $!)."
  echo "Watch:  tail -f $log    (Ctrl+C stops watching, not the run)"
}

mode=${1:-}
cd $PROJECT
export TOKENIZERS_PARALLELISM=false
# The Pytorch template sets HF_HUB_ENABLE_HF_TRANSFER=1 without installing hf_transfer, which breaks downloads.
export HF_HUB_ENABLE_HF_TRANSFER=0
case $mode in
  setup) setup ;;
  smoke)
    python -m src.train $(common_args) --epochs 0.1 --eval-steps 30 --eval-samples 64 \
      --output-dir $WORK/smoke --final-dir $WORK/smoke-model
    python - <<'EOF'
import glob, json
states = sorted(glob.glob("/workspace/smoke/checkpoint-*/trainer_state.json"), key=lambda p: int(p.split("-")[-1].split("/")[0]))
for entry in json.load(open(states[-1]))["log_history"] if states else []:
    if "eval_cer_no_space" in entry or "train_runtime" in entry:
        print(entry)
print("Speed check: train_runtime / steps = seconds per step; x ~3,000 steps = a full 5-epoch run.")
EOF
    rm -rf $WORK/smoke $WORK/smoke-model ;;
  train|ablation|evaluate)
    if [ -z "${RUNPOD_FOREGROUND:-}" ]; then background "$mode"; exit 0; fi
    case $mode in
      train) train_run $WORK/checkpoints-main $WORK/models/whisper-small-khmer-mini "" ;;
      # ~410 steps in total, so evaluate every 100 (not 500) to pick the best checkpoint like the main run
      ablation) train_run $WORK/checkpoints-ablation $WORK/models/whisper-small-khmer-openslr-only "--sources openslr42 --eval-steps 100" ;;
      evaluate) evaluate ;;
    esac ;;
  status)
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
    latest=$(ls -t $LOGS/*.log 2>/dev/null | head -1)
    [ -n "$latest" ] && { echo "== $latest"; tail -n 15 "$latest"; } || echo "No logs yet" ;;
  pack)
    cd $WORK
    python -m zipfile -c results.zip models project/outputs/evaluation
    ls -lh results.zip ;;
  *) sed -n 2,15p "$SCRIPT"; exit 1 ;;
esac
