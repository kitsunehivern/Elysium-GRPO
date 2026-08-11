#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CONFIG.yaml [train|eval|all]" >&2
  exit 2
fi

CONFIG_INPUT=$1
MODE=${2:-all}
if [[ "$MODE" != "train" && "$MODE" != "eval" && "$MODE" != "all" ]]; then
  echo "Mode must be train, eval, or all." >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=/home/stackops/dhviet/Elysium-GRPO
PYTHON_BIN=${PYTHON_BIN:-python}
DEEPSPEED_BIN=${DEEPSPEED_BIN:-deepspeed}
CUDA_DEVICES=${CUDA_DEVICES:-0}
EVAL_CUDA_DEVICES=${EVAL_CUDA_DEVICES:-$CUDA_DEVICES}
TRAIN_PORT=${TRAIN_PORT:-29691}
EVAL_PORT=${EVAL_PORT:-29887}
ALLOW_EXISTING=${ALLOW_EXISTING:-0}

if [[ "$CONFIG_INPUT" = /* ]]; then
  CONFIG_PATH=$CONFIG_INPUT
else
  CONFIG_PATH="$REPO_ROOT/$CONFIG_INPUT"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

mapfile -t CONFIG_VALUES < <(
  "$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import pathlib
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

print(cfg["output_dir"])
print(cfg["eval_dir"])
print(cfg["model"]["pretrained_model_name_or_path"])
print(cfg["model"]["trained_model_name_or_path"])
print(cfg["data"]["train"]["data_fetch"]["data_paths"][0]["anno_path"])
print(cfg["data"]["predict"]["data_fetch"]["anno_path"])
print(cfg["data"]["predict"]["data_fetch"]["image_folder"])
print(pathlib.Path(cfg["data"]["predict"]["data_fetch"]["anno_path"]).name)
PY
)

OUTPUT_DIR=${CONFIG_VALUES[0]}
EVAL_DIR=${CONFIG_VALUES[1]}
START_CHECKPOINT=${CONFIG_VALUES[2]}
TRAINED_MODEL=${CONFIG_VALUES[3]}
TRAIN_ANNOTATION=${CONFIG_VALUES[4]}
PREDICT_ANNOTATION=${CONFIG_VALUES[5]}
IMAGE_FOLDER=${CONFIG_VALUES[6]}
ANNOTATION_BASENAME=${CONFIG_VALUES[7]}

for required_path in \
  "$START_CHECKPOINT" \
  "$TRAIN_ANNOTATION" \
  "$PREDICT_ANNOTATION" \
  "$IMAGE_FOLDER"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    echo "Regenerate the configs with the correct --project-root and --dataset-root." >&2
    exit 1
  fi
done

mkdir -p "$EVAL_DIR/logs"
export WANDB_PROJECT=${WANDB_PROJECT:-Elysium-GRPO-Paper}
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$MODE" = "train" || "$MODE" = "all" ]]; then
  if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" && "$ALLOW_EXISTING" != "1" ]]; then
    echo "Refusing to train into non-empty output_dir: $OUTPUT_DIR" >&2
    echo "For an intentional resume/re-run, set resume_from_checkpoint in YAML and ALLOW_EXISTING=1." >&2
    exit 1
  fi

  (
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
      "$DEEPSPEED_BIN" --master_port="$TRAIN_PORT" \
      training/train_grpo.py --config "$CONFIG_PATH"
  ) 2>&1 | tee "$EVAL_DIR/logs/train.log"
fi

if [[ "$MODE" = "eval" || "$MODE" = "all" ]]; then
  if [[ ! -e "$TRAINED_MODEL" ]]; then
    echo "Trained model not found: $TRAINED_MODEL" >&2
    exit 1
  fi

  (
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES="$EVAL_CUDA_DEVICES" \
      "$DEEPSPEED_BIN" --master_port="$EVAL_PORT" \
      eval/eval.py --config "$CONFIG_PATH" --task SOT
  ) 2>&1 | tee "$EVAL_DIR/logs/eval.log"

  RAW_RESULTS="$EVAL_DIR/infer_results/$ANNOTATION_BASENAME"
  MERGED_RESULTS="$EVAL_DIR/infer_results/merged.jsonl"

  (
    cd "$REPO_ROOT"
    "$PYTHON_BIN" eval/merge_result.py \
      --files_to_merge="$RAW_RESULTS" \
      --output_file="$MERGED_RESULTS"
  )

  # Corrected evaluator: intentionally DO NOT pass --legacy_clamp_100.
  (
    cd "$REPO_ROOT"
    "$PYTHON_BIN" eval/otb.py "$MERGED_RESULTS"
  ) 2>&1 | tee "$EVAL_DIR/metrics_corrected.txt"

  (
    cd "$REPO_ROOT"
    "$PYTHON_BIN" eval/box_geometry_diagnostics.py "$MERGED_RESULTS" \
      --labels "$(basename "$OUTPUT_DIR")" \
      --json_output "$EVAL_DIR/geometry_diagnostics.json"
  ) 2>&1 | tee "$EVAL_DIR/geometry_diagnostics.txt"
fi
