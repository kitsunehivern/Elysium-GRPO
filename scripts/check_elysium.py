#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def collect_local_vids(frames_root: Path):
    """
    Collect local video ids from frames/<vid>/ directories.
    """
    vids = set()
    total_dirs = 0

    for p in frames_root.iterdir():
        if p.is_dir():
            total_dirs += 1
            vids.add(p.name)

    return vids, total_dirs


def count_matching_records(jsonl_path: Path, dataset_root: Path, print_every: int = 50000):
    frames_root = dataset_root / "frames"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not frames_root.exists():
        raise FileNotFoundError(f"Frames folder not found: {frames_root}")

    print(f"[DEBUG] jsonl_path   = {jsonl_path}")
    print(f"[DEBUG] dataset_root = {dataset_root}")
    print(f"[DEBUG] frames_root  = {frames_root}")
    print("[DEBUG] Scanning local frames folders...")

    local_vids, total_local_vid_dirs = collect_local_vids(frames_root)

    print(f"[DEBUG] Local video folders found: {total_local_vid_dirs}")
    print()

    total_records = 0
    parsed_records = 0
    records_vid_dir_exists = 0
    records_any_frame_exists = 0
    records_all_frames_exist = 0

    total_frame_refs = 0
    total_found_frames = 0
    total_missing_frames = 0

    sample_vid_dir_exists = []
    sample_any_frame_exists = []
    sample_all_frames_exist = []
    sample_missing = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total_records += 1

            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[WARN] line {lineno}: failed to parse JSON: {e}")
                continue

            parsed_records += 1

            vid = str(rec.get("vid", ""))
            rec_id = rec.get("id", "N/A")
            frames = rec.get("frames", [])

            if not isinstance(frames, list):
                frames = []

            vid_dir_exists = vid in local_vids

            if vid_dir_exists:
                records_vid_dir_exists += 1
                if len(sample_vid_dir_exists) < 5:
                    sample_vid_dir_exists.append((lineno, rec_id, vid))

            found_count = 0
            missing_count = 0

            for rel_path in frames:
                if not isinstance(rel_path, str):
                    missing_count += 1
                    total_missing_frames += 1
                    continue

                total_frame_refs += 1
                abs_path = dataset_root / rel_path

                if abs_path.exists():
                    found_count += 1
                    total_found_frames += 1
                else:
                    missing_count += 1
                    total_missing_frames += 1

            if found_count > 0:
                records_any_frame_exists += 1
                if len(sample_any_frame_exists) < 5:
                    sample_any_frame_exists.append((lineno, rec_id, vid, found_count, len(frames)))

            if len(frames) > 0 and missing_count == 0:
                records_all_frames_exist += 1
                if len(sample_all_frames_exist) < 5:
                    sample_all_frames_exist.append((lineno, rec_id, vid, len(frames)))

            if missing_count > 0 and len(sample_missing) < 5:
                sample_missing.append((lineno, rec_id, vid, found_count, missing_count, len(frames)))

            if total_records % print_every == 0:
                print(
                    f"[DEBUG] processed={total_records:,} "
                    f"parsed={parsed_records:,} "
                    f"vid_dir={records_vid_dir_exists:,} "
                    f"any_frame={records_any_frame_exists:,} "
                    f"all_frames={records_all_frames_exist:,}"
                )

    print()
    print("#" * 100)
    print("[FINAL SUMMARY]")
    print(f"Total records in JSONL parsed         : {parsed_records:,}")
    print(f"Records whose vid folder exists       : {records_vid_dir_exists:,}")
    print(f"Records with >=1 local frame          : {records_any_frame_exists:,}")
    print(f"Records with all frames available     : {records_all_frames_exist:,}")
    print()
    print(f"Total frame refs checked              : {total_frame_refs:,}")
    print(f"Total found frames                    : {total_found_frames:,}")
    print(f"Total missing frames                  : {total_missing_frames:,}")
    print()

    if parsed_records > 0:
        print("[RATIOS]")
        print(f"vid folder exists rate               : {records_vid_dir_exists / parsed_records:.2%}")
        print(f">=1 frame exists rate                : {records_any_frame_exists / parsed_records:.2%}")
        print(f"all frames available rate            : {records_all_frames_exist / parsed_records:.2%}")
        print()

    print("[SAMPLES] vid folder exists")
    for x in sample_vid_dir_exists:
        print(f"  line={x[0]} id={x[1]} vid={x[2]}")

    print("[SAMPLES] any frame exists")
    for x in sample_any_frame_exists:
        print(f"  line={x[0]} id={x[1]} vid={x[2]} found={x[3]}/{x[4]}")

    print("[SAMPLES] all frames exist")
    for x in sample_all_frames_exist:
        print(f"  line={x[0]} id={x[1]} vid={x[2]} frames={x[3]}")

    print("[SAMPLES] partial/missing")
    for x in sample_missing:
        print(f"  line={x[0]} id={x[1]} vid={x[2]} found={x[3]} missing={x[4]} total={x[5]}")


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python count_usable_records.py /path/to/ElysiumTrack-1M.json /path/to/dataset_root")
        print()
        print("Example:")
        print("  python count_usable_records.py /data/ElysiumTrack-1M/ElysiumTrack-1M.json /data/ElysiumTrack-1M_extracted")
        sys.exit(1)

    jsonl_path = Path(sys.argv[1]).resolve()
    dataset_root = Path(sys.argv[2]).resolve()

    count_matching_records(jsonl_path, dataset_root)


if __name__ == "__main__":
    main()

"""
python scripts/check_elysium.py /raid/hvtham/dcmquan/Elysium/datasets/train/ElysiumTrack-1M/ElysiumTrack-1M.jsonl /raid/hvtham/dcmquan/Elysium/datasets/train/ElysiumTrack-1M
"""