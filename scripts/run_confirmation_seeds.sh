#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-all}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${ELYSIUM_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}

CONFIGS=(
  "configs/uav123_reward_full_curriculum_seed123.yaml"
  "configs/uav123_reward_full_curriculum_seed2026.yaml"
  "configs/uav123_reward_fixed_mean_seed123.yaml"
  "configs/uav123_reward_fixed_mean_seed2026.yaml"
)

BASE_TRAIN_PORT=${BASE_TRAIN_PORT:-33691}
BASE_EVAL_PORT=${BASE_EVAL_PORT:-34691}

for index in "${!CONFIGS[@]}"; do
  config=${CONFIGS[$index]}
  TRAIN_PORT=$((BASE_TRAIN_PORT + index)) \
  EVAL_PORT=$((BASE_EVAL_PORT + index)) \
  ELYSIUM_ROOT="$REPO_ROOT" \
    "$SCRIPT_DIR/run_experiment.sh" "$config" "$MODE"
done
