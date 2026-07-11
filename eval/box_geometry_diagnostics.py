"""Diagnose center-vs-box-geometry drift in merged tracking predictions.

Expected input is the JSONL produced by ``eval/merge_result.py``. The script uses
normalized TLBR coordinates directly, so area/aspect ratios do not depend on image
resolution.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

Box = Tuple[float, float, float, float]
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
BRACKET_RE = re.compile(r"[\[\{]([^\]\}]+)[\]\}]")


def parse_boxes(text: str) -> List[Box]:
    boxes: List[Box] = []
    for match in BRACKET_RE.finditer(text or ""):
        values = NUMBER_RE.findall(match.group(1))
        if len(values) < 4:
            continue
        boxes.append(tuple(float(v) for v in values[:4]))  # type: ignore[arg-type]
    return boxes


def box_size(box: Box) -> Tuple[float, float]:
    return box[2] - box[0], box[3] - box[1]


def box_center(box: Box) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def valid_geometry(box: Box) -> bool:
    w, h = box_size(box)
    return all(math.isfinite(v) for v in box) and w > 0.0 and h > 0.0


def iou(pred: Box, gt: Box) -> float:
    if not valid_geometry(pred) or not valid_geometry(gt):
        return 0.0
    ix1, iy1 = max(pred[0], gt[0]), max(pred[1], gt[1])
    ix2, iy2 = min(pred[2], gt[2]), min(pred[3], gt[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    pw, ph = box_size(pred)
    gw, gh = box_size(gt)
    union = pw * ph + gw * gh - inter
    return inter / union if union > 0.0 else 0.0


def boundary_overflow(box: Box, scale: float = 100.0) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, -x1) + max(0.0, -y1) + max(0.0, x2 - scale) + max(0.0, y2 - scale)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _safe_mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _safe_median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else float("nan")


def diagnose_file(path: str) -> Dict[str, float]:
    frame_iou: List[float] = []
    center_error: List[float] = []
    center_error_gt_diag: List[float] = []
    width_ratio: List[float] = []
    height_ratio: List[float] = []
    area_ratio: List[float] = []
    aspect_ratio_ratio: List[float] = []
    abs_log_area_ratio: List[float] = []
    abs_log_aspect_ratio: List[float] = []
    motion_vector_error: List[float] = []
    excess_jump: List[float] = []
    overflow: List[float] = []
    invalid_pred = 0
    total_pairs = 0
    sequences = 0

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            item = json.loads(raw_line)
            pred_boxes = parse_boxes(item.get("predict", ""))
            gt_boxes = parse_boxes(item.get("gt", ""))
            usable = min(len(pred_boxes), len(gt_boxes))
            if usable == 0:
                continue
            sequences += 1
            prev_pred = None
            prev_gt = None
            for pred, gt in zip(pred_boxes[:usable], gt_boxes[:usable]):
                if not valid_geometry(gt):
                    prev_pred = None
                    prev_gt = None
                    continue
                total_pairs += 1
                if not valid_geometry(pred):
                    invalid_pred += 1
                    frame_iou.append(0.0)
                    prev_pred = None
                    prev_gt = None
                    continue

                pw, ph = box_size(pred)
                gw, gh = box_size(gt)
                pcx, pcy = box_center(pred)
                gcx, gcy = box_center(gt)
                ce = math.hypot(pcx - gcx, pcy - gcy)
                diag = max(math.hypot(gw, gh), 1e-8)
                wr = pw / gw
                hr = ph / gh
                ar = (pw * ph) / (gw * gh)
                arr = (pw / ph) / (gw / gh)

                frame_iou.append(iou(pred, gt))
                center_error.append(ce)
                center_error_gt_diag.append(ce / diag)
                width_ratio.append(wr)
                height_ratio.append(hr)
                area_ratio.append(ar)
                aspect_ratio_ratio.append(arr)
                abs_log_area_ratio.append(abs(math.log(max(ar, 1e-12))))
                abs_log_aspect_ratio.append(abs(math.log(max(arr, 1e-12))))
                overflow.append(boundary_overflow(pred))

                if prev_pred is not None and prev_gt is not None:
                    p0x, p0y = box_center(prev_pred)
                    g0x, g0y = box_center(prev_gt)
                    pdx, pdy = pcx - p0x, pcy - p0y
                    gdx, gdy = gcx - g0x, gcy - g0y
                    motion_vector_error.append(math.hypot(pdx - gdx, pdy - gdy))
                    excess_jump.append(max(0.0, math.hypot(pdx, pdy) - math.hypot(gdx, gdy)))

                prev_pred, prev_gt = pred, gt

    valid_predictions = len(frame_iou) - invalid_pred
    return {
        "sequences": float(sequences),
        "gt_frames": float(total_pairs),
        "valid_pred_rate": valid_predictions / total_pairs if total_pairs else float("nan"),
        "mean_iou": _safe_mean(frame_iou),
        "center_error_0_100_mean": _safe_mean(center_error),
        "center_error_gt_diag_mean": _safe_mean(center_error_gt_diag),
        "center_error_gt_diag_p90": _percentile(center_error_gt_diag, 0.90),
        "width_ratio_median": _safe_median(width_ratio),
        "height_ratio_median": _safe_median(height_ratio),
        "area_ratio_median": _safe_median(area_ratio),
        "area_ratio_p10": _percentile(area_ratio, 0.10),
        "area_ratio_p90": _percentile(area_ratio, 0.90),
        "oversized_area_rate_gt_1_5": sum(v > 1.5 for v in area_ratio) / len(area_ratio) if area_ratio else float("nan"),
        "undersized_area_rate_lt_0_67": sum(v < (2.0 / 3.0) for v in area_ratio) / len(area_ratio) if area_ratio else float("nan"),
        "abs_log_area_ratio_mean": _safe_mean(abs_log_area_ratio),
        "aspect_ratio_ratio_median": _safe_median(aspect_ratio_ratio),
        "abs_log_aspect_ratio_mean": _safe_mean(abs_log_aspect_ratio),
        "motion_vector_error_mean": _safe_mean(motion_vector_error),
        "excess_jump_mean": _safe_mean(excess_jump),
        "out_of_boundary_rate": sum(v > 0.0 for v in overflow) / len(overflow) if overflow else float("nan"),
        "boundary_overflow_mean": _safe_mean(overflow),
    }


def _fmt(value: float, digits: int = 4) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.{digits}f}"


def print_summary(labels: Sequence[str], summaries: Sequence[Dict[str, float]]) -> None:
    columns = [
        ("Model", None),
        ("IoU", "mean_iou"),
        ("Center/GTdiag", "center_error_gt_diag_mean"),
        ("Area med", "area_ratio_median"),
        ("|log area|", "abs_log_area_ratio_mean"),
        ("AR med", "aspect_ratio_ratio_median"),
        ("|log AR|", "abs_log_aspect_ratio_mean"),
        ("Motion err", "motion_vector_error_mean"),
        ("OOB rate", "out_of_boundary_rate"),
    ]
    widths = [max(len(name), max((len(label) for label in labels), default=0)) if key is None else len(name) for name, key in columns]
    rows: List[List[str]] = []
    for label, summary in zip(labels, summaries):
        row = [label]
        for _, key in columns[1:]:
            row.append(_fmt(summary[key]))  # type: ignore[index]
        rows.append(row)
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    header = " | ".join(name.ljust(width) for (name, _), width in zip(columns, widths))
    sep = "-+-".join("-" * width for width in widths)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="Merged JSONL result files")
    parser.add_argument("--labels", nargs="*", help="Optional labels matching files")
    parser.add_argument("--json_output", help="Optional path to write full summaries")
    args = parser.parse_args()

    labels = args.labels or [Path(path).parent.parent.name or Path(path).stem for path in args.files]
    if len(labels) != len(args.files):
        parser.error("--labels must contain exactly one label per file")

    summaries = [diagnose_file(path) for path in args.files]
    print_summary(labels, summaries)

    if args.json_output:
        payload = {label: summary for label, summary in zip(labels, summaries)}
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
