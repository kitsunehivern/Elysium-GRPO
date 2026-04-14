from __future__ import annotations
from typing import List, Optional, Dict, Any
import math


def is_valid_box(box: Optional[List[int]], image_size: List[int]) -> bool:
    """
    Check if box is valid and inside image bounds.
    image_size = [width, height]
    """
    if box is None:
        return False
    if len(box) != 4:
        return False

    x1, y1, x2, y2 = box
    w, h = image_size

    if not (0 <= x1 < x2 <= w):
        return False
    if not (0 <= y1 < y2 <= h):
        return False
    return True


def box_area(box: List[int]) -> float:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(box_a: List[int], box_b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union = box_area(box_a) + box_area(box_b) - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def center(box: List[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_distance(box_a: List[int], box_b: List[int]) -> float:
    ax, ay = center(box_a)
    bx, by = center(box_b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def smoothness_penalty(pred_boxes: List[Optional[List[int]]]) -> float:
    """
    Penalize large jumps between consecutive predicted boxes.
    """
    penalty = 0.0
    prev = None
    for box in pred_boxes:
        if prev is not None and box is not None:
            penalty += center_distance(prev, box)
        prev = box
    return penalty


def score_trajectory(
    pred_boxes: List[Optional[List[int]]],
    gt_boxes: List[List[int]],
    image_size: List[int],
) -> Dict[str, Any]:
    """
    Produce a scalar score and diagnostics.
    Higher is better.
    """
    assert len(pred_boxes) == len(gt_boxes), "Prediction/GT length mismatch"

    valid_count = 0
    ious = []
    center_errors = []

    for pred, gt in zip(pred_boxes, gt_boxes):
        if is_valid_box(pred, image_size):
            valid_count += 1
            ious.append(iou(pred, gt))
            center_errors.append(center_distance(pred, gt))
        else:
            ious.append(0.0)
            center_errors.append(1000.0)

    format_valid_rate = valid_count / max(1, len(gt_boxes))
    mean_iou = sum(ious) / len(ious)
    mean_center_error = sum(center_errors) / len(center_errors)
    jitter = smoothness_penalty(pred_boxes) / max(1, len(pred_boxes) - 1)

    # Reward design for first pass:
    # + high IoU
    # + valid boxes
    # - center error
    # - excessive jitter
    score = (
        5.0 * mean_iou
        + 1.0 * format_valid_rate
        - 0.01 * mean_center_error
        - 0.005 * jitter
    )

    return {
        "score": score,
        "mean_iou": mean_iou,
        "format_valid_rate": format_valid_rate,
        "mean_center_error": mean_center_error,
        "jitter": jitter,
    }