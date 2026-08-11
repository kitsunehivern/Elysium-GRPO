#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-all}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${ELYSIUM_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}

CONFIGS=(
  "configs/uav123_reward_reverse_curriculum.yaml"
  "configs/uav123_reward_curriculum_tracking_only.yaml"
  "configs/uav123_reward_curriculum_format_only.yaml"
  "configs/uav123_reward_wo_geometry.yaml"
  "configs/uav123_reward_wo_trajectory.yaml"
  "configs/uav123_reward_iou_format_only.yaml"
)

BASE_TRAIN_PORT=${BASE_TRAIN_PORT:-31691}
BASE_EVAL_PORT=${BASE_EVAL_PORT:-32691}

for index in "${!CONFIGS[@]}"; do
  config=${CONFIGS[$index]}
  TRAIN_PORT=$((BASE_TRAIN_PORT + index)) \
  EVAL_PORT=$((BASE_EVAL_PORT + index)) \
  ELYSIUM_ROOT="$REPO_ROOT" \
    "$SCRIPT_DIR/run_experiment.sh" "$config" "$MODE"
done
