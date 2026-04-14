#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Extract the first non-empty record from a JSONL file and save it as ElysiumTrack-1.jsonl"
    )
    parser.add_argument(
        "--jsonl-path",
        required=True,
        help="Path to the source JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save ElysiumTrack-1.jsonl. Default: same folder as input",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path).resolve()
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else jsonl_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ElysiumTrack-1.jsonl"

    first_line = None
    with jsonl_path.open("r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, start=1):
            if line.strip():
                first_line = line.rstrip("\n")
                print(f"[DEBUG] Found first non-empty record at line {line_num}")
                break

    if first_line is None:
        raise ValueError(f"No non-empty records found in: {jsonl_path}")

    with output_path.open("w", encoding="utf-8") as fout:
        fout.write(first_line + "\n")

    print(f"[DONE] Wrote first record to: {output_path}")


if __name__ == "__main__":
    main()

"""
python scripts/get_first_record.py \
  --jsonl-path /raid/hvtham/dcmquan/Elysium/datasets/train/ElysiumTrack-1M/ElysiumTrack-60K.jsonl \
  --output-dir /raid/hvtham/dcmquan/Elysium/datasets/train/ElysiumTrack-1M
"""