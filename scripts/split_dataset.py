#!/usr/bin/env python3
import argparse
import random
from pathlib import Path


def read_jsonl_lines(path: Path):
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line.rstrip("\n"))
    return lines


def write_jsonl_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Split a JSONL dataset into train/test files.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--train-size", type=int, required=True, help="Number of train records")
    parser.add_argument("--test-size", type=int, required=True, help="Number of test records")
    parser.add_argument("--train-output", required=True, help="Full output path for train JSONL")
    parser.add_argument("--test-output", required=True, help="Full output path for test JSONL")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before splitting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    train_output = Path(args.train_output).resolve()
    test_output = Path(args.test_output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[DEBUG] input        = {input_path}")
    print(f"[DEBUG] train_output = {train_output}")
    print(f"[DEBUG] test_output  = {test_output}")
    print(f"[DEBUG] train_size   = {args.train_size}")
    print(f"[DEBUG] test_size    = {args.test_size}")
    print(f"[DEBUG] shuffle      = {args.shuffle}")
    print(f"[DEBUG] seed         = {args.seed}")

    lines = read_jsonl_lines(input_path)
    total = len(lines)
    needed = args.train_size + args.test_size

    print(f"[DEBUG] total records = {total}")

    if needed > total:
        raise ValueError(
            f"Requested {needed} records, but input only has {total} records."
        )

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(lines)

    train_lines = lines[:args.train_size]
    test_lines = lines[args.train_size:args.train_size + args.test_size]

    write_jsonl_lines(train_output, train_lines)
    write_jsonl_lines(test_output, test_lines)

    print()
    print("[DONE]")
    print(f"Train file: {train_output} ({len(train_lines)} records)")
    print(f"Test file : {test_output} ({len(test_lines)} records)")


if __name__ == "__main__":
    main()

"""
python scripts/split_dataset.py \
  --input /raid/hvtham/dcmquan/Elysium/datasets/train/ElysiumTrack-1M/ElysiumTrack-60K.jsonl \
  --train-size 50000 \
  --test-size 10000 \
  --train-output /raid/hvtham/dcmquan/Elysium/datasets/train/ElysiumTrack-1M/ElysiumTrack-50K.jsonl \
  --test-output /raid/hvtham/dcmquan/Elysium/datasets/test/ElysiumTrack-1M/ElysiumTrack-10K.jsonl \
  --shuffle \
  --seed 42
"""