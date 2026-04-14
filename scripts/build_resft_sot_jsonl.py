import os
import json
import math
import argparse
from typing import Any, Dict, List, Tuple


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def norm_box_to_pixel_box(
    box: List[float],
    frame_w: int,
    frame_h: int,
) -> List[int]:
    """
    Convert normalized [x1, y1, x2, y2] in [0,1] to integer pixel coords.
    Keeps the exact style close to your current baseline outputs.
    """
    x1 = clamp_int(int(round(box[0] * frame_w)), 0, frame_w - 1)
    y1 = clamp_int(int(round(box[1] * frame_h)), 0, frame_h - 1)
    x2 = clamp_int(int(round(box[2] * frame_w)), 0, frame_w - 1)
    y2 = clamp_int(int(round(box[3] * frame_h)), 0, frame_h - 1)

    # ensure valid ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]


def box_to_str(box: List[int]) -> str:
    return f"[{box[0]},{box[1]},{box[2]},{box[3]}]"


def make_sot_question(num_frames: int, init_box: List[int]) -> str:
    frame_tokens = ", ".join([f"Frame {i}: <image>" for i in range(1, num_frames + 1)])
    return (
        f"{frame_tokens}\n"
        f"This is a video showing an object with coordinates {box_to_str(init_box)} in Frame 1. "
        f"Please provide the detailed coordinates of the object in each frame."
    )


def make_answer(boxes: List[List[int]]) -> str:
    parts = [f"Frame {i}: {box_to_str(b)}" for i, b in enumerate(boxes, start=1)]
    return ", ".join(parts)


def slice_with_overlap(
    frames: List[str],
    boxes: List[List[int]],
    clip_len: int,
    overlap: int,
) -> List[Tuple[int, int]]:
    """
    Return (start, end) indices, end exclusive.
    Example: clip_len=8, overlap=1 -> stride=7
    """
    assert clip_len > 0
    assert 0 <= overlap < clip_len

    n = len(frames)
    if n == 0:
        return []

    if n <= clip_len:
        return [(0, n)]

    stride = clip_len - overlap
    clips = []
    start = 0
    while start < n:
        end = min(start + clip_len, n)
        clips.append((start, end))
        if end == n:
            break
        start += stride
    return clips


def should_keep_record(
    record: Dict[str, Any],
    min_frames: int,
) -> bool:
    frames = record.get("frames", [])
    box = record.get("box", [])
    frame_size = record.get("frame_size", None)

    if not isinstance(frames, list) or len(frames) < min_frames:
        return False
    if not isinstance(box, list) or len(box) != len(frames):
        return False
    if not isinstance(frame_size, list) or len(frame_size) != 2:
        return False
    return True


def build_train_record(
    source_item: Dict[str, Any],
    clip_index: int,
    clip_frames: List[str],
    clip_boxes_norm: List[List[float]],
    clip_boxes_px: List[List[int]],
    task: str = "SOT",
) -> Dict[str, Any]:
    question = make_sot_question(len(clip_frames), clip_boxes_px[0])
    answer = make_answer(clip_boxes_px)

    vid_raw = str(source_item["vid"])
    record_id = f"{vid_raw}|{clip_index}"

    vqa_payload = [
        {
            "from": "human",
            "value": question,
        },
        {
            "from": "gpt",
            "value": answer,
        },
    ]

    # Keep extra fields so you can inspect / debug later
    out = {
        "source": source_item.get("source", ""),
        "id": record_id,
        "vid": record_id,
        "orig_vid": source_item.get("vid"),
        "parquet_id": source_item.get("parquet_id", ""),
        "frames": clip_frames,
        "box": clip_boxes_norm,
        "frame_size": source_item.get("frame_size"),
        "task": task,
        "question": question,
        "answer": answer,
        # safest option: include a vqa object because your config uses label_key: vqa
        "vqa": json.dumps(vqa_payload, ensure_ascii=False),
        # useful metadata for later filtering / replay
        "object_class": source_item.get("object_class", ""),
        "object_description": source_item.get("object_description", ""),
        "caption": source_item.get("caption", ""),
        "metadata": source_item.get("metadata", ""),
        "clip_index": clip_index,
        "num_frames": len(clip_frames),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to source JSONL")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL")
    parser.add_argument("--image-folder", type=str, default="", help="Optional root image folder for existence checks")
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--require-images-exist", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    kept = 0
    skipped = 0
    built = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line_idx, line in enumerate(fin):
            if not line.strip():
                continue

            item = json.loads(line)

            if not should_keep_record(item, args.min_frames):
                skipped += 1
                continue

            frames: List[str] = item["frames"]
            boxes_norm: List[List[float]] = item["box"]
            frame_w, frame_h = int(item["frame_size"][0]), int(item["frame_size"][1])

            if args.require_images_exist and args.image_folder:
                missing = False
                for f in frames:
                    full_path = os.path.join(args.image_folder, f)
                    if not os.path.exists(full_path):
                        missing = True
                        break
                if missing:
                    skipped += 1
                    continue

            boxes_px = [norm_box_to_pixel_box(b, frame_w, frame_h) for b in boxes_norm]
            clip_ranges = slice_with_overlap(frames, boxes_px, args.clip_len, args.overlap)

            boxes_norm = item["box"]
            boxes_px = [norm_box_to_pixel_box(b, frame_w, frame_h) for b in boxes_norm]

            for clip_idx, (start, end) in enumerate(clip_ranges):
                clip_frames = frames[start:end]
                clip_boxes_norm = boxes_norm[start:end]
                clip_boxes_px = boxes_px[start:end]

                out = build_train_record(
                    source_item=item,
                    clip_index=clip_idx,
                    clip_frames=clip_frames,
                    clip_boxes_norm=clip_boxes_norm,
                    clip_boxes_px=clip_boxes_px,
                    task="SOT",
                )

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                built += 1
                
            kept += 1

            if args.limit > 0 and built >= args.limit:
                break

    print(
        json.dumps(
            {
                "source_records_kept": kept,
                "source_records_skipped": skipped,
                "output_clips_written": built,
                "output_path": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""
python scripts/build_resft_sot_jsonl.py \
  --input /raid/hvtham/dcmquan/Elysium/datasets/train/val500/ElysiumTrack-val1.jsonl \
  --output /raid/hvtham/dcmquan/Elysium/datasets/resft/resft_sot_stage1.jsonl \
  --image-folder /raid/hvtham/dcmquan/Elysium/datasets/train \
  --clip-len 8 \
  --overlap 1 \
  --min-frames 3
"""
