#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-all}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${ELYSIUM_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}

CONFIGS=(
  "configs/uav123_reward_full_curriculum.yaml"
  "configs/uav123_reward_wo_iou.yaml"
  "configs/uav123_reward_wo_center.yaml"
  "configs/uav123_reward_wo_area.yaml"
  "configs/uav123_reward_wo_aspect.yaml"
  "configs/uav123_reward_wo_temporal.yaml"
  "configs/uav123_reward_wo_validity.yaml"
  "configs/uav123_reward_wo_format.yaml"
  "configs/uav123_reward_fixed_early.yaml"
  "configs/uav123_reward_fixed_late.yaml"
  "configs/uav123_reward_fixed_mean.yaml"
)

BASE_TRAIN_PORT=${BASE_TRAIN_PORT:-29691}
BASE_EVAL_PORT=${BASE_EVAL_PORT:-30691}

for index in "${!CONFIGS[@]}"; do
  config=${CONFIGS[$index]}
  TRAIN_PORT=$((BASE_TRAIN_PORT + index)) \
  EVAL_PORT=$((BASE_EVAL_PORT + index)) \
  ELYSIUM_ROOT="$REPO_ROOT" \
    "$SCRIPT_DIR/run_experiment.sh" "$config" "$MODE"
done
