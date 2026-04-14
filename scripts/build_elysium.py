#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def format_count_name(n: int) -> str:
    if n >= 1_000_000:
        val = n / 1_000_000
        if abs(val - round(val)) < 1e-9:
            return f"{int(round(val))}M"
        return f"{val:.1f}M".rstrip("0").rstrip(".")
    if n >= 1_000:
        val = n / 1_000
        if abs(val - round(val)) < 1e-9:
            return f"{int(round(val))}K"
        return f"{val:.1f}K".rstrip("0").rstrip(".")
    return str(n)


def collect_local_vids(frames_root: Path):
    vids = set()
    for p in frames_root.iterdir():
        if p.is_dir():
            vids.add(p.name)
    return vids


def main():
    parser = argparse.ArgumentParser(
        description="Create a subset JSONL from ElysiumTrack by keeping only records whose vid exists in frames/<vid>/"
    )
    parser.add_argument(
        "--jsonl-path",
        required=True,
        help="Path to source JSONL, e.g. /path/to/ElysiumTrack-1M.jsonl",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root containing frames/, e.g. /path/to/ElysiumTrack-1M",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write output JSONL. Default: dataset-root",
    )
    parser.add_argument(
        "--prefix",
        default="ElysiumTrack",
        help="Output filename prefix. Default: ElysiumTrack",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=50000,
        help="Print progress every N records. Default: 50000",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    frames_root = dataset_root / "frames"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else dataset_root

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not frames_root.exists():
        raise FileNotFoundError(f"Frames root not found: {frames_root}")

    print(f"[DEBUG] jsonl_path   = {jsonl_path}")
    print(f"[DEBUG] dataset_root = {dataset_root}")
    print(f"[DEBUG] frames_root  = {frames_root}")
    print(f"[DEBUG] output_dir   = {output_dir}")
    print("[DEBUG] Collecting local vid folders...")

    local_vids = collect_local_vids(frames_root)
    print(f"[DEBUG] Local video folders found: {len(local_vids):,}")

    total_records = 0
    parsed_records = 0
    kept_records = 0
    bad_json = 0
    kept_lines = []

    with jsonl_path.open("r", encoding="utf-8") as fin:
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            total_records += 1

            try:
                record = json.loads(line)
            except Exception as e:
                bad_json += 1
                print(f"[WARN] line={lineno} bad json: {e}")
                continue

            parsed_records += 1
            vid = str(record.get("vid", ""))

            if vid in local_vids:
                kept_lines.append(json.dumps(record, ensure_ascii=False))
                kept_records += 1

            if total_records % args.print_every == 0:
                print(
                    f"[DEBUG] processed={total_records:,} "
                    f"parsed={parsed_records:,} "
                    f"kept={kept_records:,}"
                )

    suffix = format_count_name(kept_records)
    output_name = f"{args.prefix}-{suffix}.jsonl"
    output_path = output_dir / output_name

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for line in kept_lines:
            fout.write(line + "\n")

    print()
    print("#" * 100)
    print("[FINAL SUMMARY]")
    print(f"Total records read : {total_records:,}")
    print(f"Parsed records     : {parsed_records:,}")
    print(f"Bad JSON lines     : {bad_json:,}")
    print(f"Matched records    : {kept_records:,}")
    print(f"Output file        : {output_path}")


if __name__ == "__main__":
    main()

"""
python scripts/build_elysium.py \
  --jsonl-path /raid/hvtham/dcmquan/Elysium/datasets/train/ElysiumTrack-1M/ElysiumTrack-1M.jsonl \
  --dataset-root /raid/hvtham/dcmquan/Elysium/datasets/train/ElysiumTrack-1M
"""