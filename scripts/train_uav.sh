#!/bin/bash

# Check if at least one argument (the GPU ID) is provided
if [ -z "$1" ]; then
  echo "Error: No GPU ID provided."
  echo "Usage: $0 <gpu_id> [additional_args...]"
  echo "Example: $0 0"
  echo "Example: $0 0,1"
  exit 1
fi

# Assign the first argument to GPU_ID
GPU_ID=$1

# Shift the arguments so any remaining arguments can be passed to the python script
shift

echo "Starting training on CUDA_VISIBLE_DEVICES=$GPU_ID..."

# Run the command with the specified GPU and any extra arguments
CUDA_VISIBLE_DEVICES=$GPU_ID deepspeed --master_port=29501 training/train.py --config configs/baseline_uav123.yaml "$@"