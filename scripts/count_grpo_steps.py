#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def count_clips_for_video(num_frames: int, clip_len: int) -> int:
    """
    Matches GRPOClipDataset behavior:

      stride = clip_len - 1
      starts: 0, stride, 2*stride, ...
      keep clip only if it has at least 2 frames

    Equivalent count:
      if num_frames < 2: 0
      else: ceil((num_frames - 1) / (clip_len - 1))
    """
    if num_frames < 2:
        return 0

    stride = clip_len - 1
    return math.ceil((num_frames - 1) / stride)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno", type=str, required=True, help="Path to JSONL annotation file")
    parser.add_argument("--clip_len", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--show_each", action="store_true")
    args = parser.parse_args()

    anno_path = Path(args.anno)

    total_records = 0
    total_frames = 0
    total_clips = 0

    min_frames = None
    max_frames = 0
    min_clips = None
    max_clips = 0

    per_video = []

    with anno_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            vid = item.get("vid", f"line_{line_no}")
            frames = item["frames"]
            boxes = item.get("box", None)

            num_frames = len(frames)

            if boxes is not None and len(boxes) != num_frames:
                raise ValueError(
                    f"{vid}: frames/boxes length mismatch: "
                    f"{num_frames} frames vs {len(boxes)} boxes"
                )

            num_clips = count_clips_for_video(num_frames, args.clip_len)

            total_records += 1
            total_frames += num_frames
            total_clips += num_clips

            min_frames = num_frames if min_frames is None else min(min_frames, num_frames)
            max_frames = max(max_frames, num_frames)

            min_clips = num_clips if min_clips is None else min(min_clips, num_clips)
            max_clips = max(max_clips, num_clips)

            per_video.append((vid, num_frames, num_clips))

    steps_per_epoch = math.ceil(total_clips / args.batch_size)

    print("========== GRPO Epoch Step Count ==========")
    print(f"annotation file      : {anno_path}")
    print(f"records/videos       : {total_records}")
    print(f"total frames         : {total_frames}")
    print(f"clip_len             : {args.clip_len}")
    print(f"stride               : {args.clip_len - 1}")
    print(f"batch_size           : {args.batch_size}")
    print(f"total clips          : {total_clips}")
    print(f"steps per epoch      : {steps_per_epoch}")
    print()
    print(f"min frames/video     : {min_frames}")
    print(f"max frames/video     : {max_frames}")
    print(f"min clips/video      : {min_clips}")
    print(f"max clips/video      : {max_clips}")

    if args.show_each:
        print()
        print("========== Per Video ==========")
        for vid, num_frames, num_clips in per_video:
            print(f"{vid}\tframes={num_frames}\tclips={num_clips}")

    print()
    print("Formula:")
    print("  clips_per_video = ceil((num_frames - 1) / (clip_len - 1))")
    print("  steps_per_epoch = ceil(total_clips / batch_size)")
    print()
    print("Note:")
    print("  group_size does NOT change steps_per_epoch.")
    print("  group_size only changes how many completions are sampled per step.")


if __name__ == "__main__":
    main()

"""
python scripts/count_grpo_steps.py \
  --anno /raid/hvtham/dhviet/UAV123_Elysium/short_train/annotation.jsonl \
  --clip_len 8 \
  --batch_size 1 \
  --show_each
"""
