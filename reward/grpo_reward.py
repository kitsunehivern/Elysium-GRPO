import math
import re
from typing import List, Dict, Any

import torch


BOX_PATTERN_BRACKET = re.compile(r"\[([\d\s,]+)\]")
BOX_PATTERN_TAG = re.compile(r"{<(\d+)><(\d+)><(\d+)><(\d+)>}")


def parse_box_from_raw_text(text: str) -> List[List[float]]:
    """
    Parse generated boxes into normalized [0,1] xyxy.
    Supports:
      [23,45,46,72]
      {<23><45><46><72>}
    """
    coords = []

    raw_tag = re.findall(BOX_PATTERN_TAG, text)
    if len(raw_tag) > 0:
        for xyxy in raw_tag:
            box = [float(v) / 100.0 for v in xyxy[:4]]
            if len(box) == 4:
                coords.append(box)
        return coords

    raw_bracket = re.findall(BOX_PATTERN_BRACKET, text)
    for xyxy_str in raw_bracket:
        vals = []
        for coord in xyxy_str.replace(" ", "").split(","):
            if coord != "":
                vals.append(float(coord) / 100.0)
        vals = vals[:4]
        if len(vals) < 4:
            if len(coords) > 0:
                vals = coords[-1]
            else:
                vals = [0.0, 0.0, 0.0, 0.0]
        coords.append(vals)
    return coords


def normalize_abs_box(box, image_size):
    w, h = image_size
    x1, y1, x2, y2 = box
    return [
        float(x1) / float(w),
        float(y1) / float(h),
        float(x2) / float(w),
        float(y2) / float(h),
    ]

from typing import Any, List
import torch


def _to_python(obj: Any):
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, tuple):
        return [_to_python(x) for x in obj]
    if isinstance(obj, list):
        return [_to_python(x) for x in obj]
    return obj


def _is_scalar(x):
    return isinstance(x, (int, float))


def canonicalize_gt_boxes(gt_boxes_abs: Any) -> List[List[float]]:
    """
    Convert collated GT boxes into a canonical shape:
        [[x1, y1, x2, y2], ...]
    Handles common weird cases from collate_fn:
      - [[[x1,y1,x2,y2], ...]]              (extra batch dim)
      - [x1, y1, x2, y2]                    (single box)
      - [[x1],[y1],[x2],[y2]]               (single box, transposed)
      - [[x1s...],[y1s...],[x2s...],[y2s...]] (T x 4 transposed to 4 x T)
      - tensors / nested tensors
    """
    x = _to_python(gt_boxes_abs)

    # strip repeated singleton wrappers
    while isinstance(x, list) and len(x) == 1 and isinstance(x[0], (list, tuple)):
        x = x[0]

    # case A: single box => wrap into list of boxes
    if isinstance(x, list) and len(x) == 4 and all(_is_scalar(v) for v in x):
        return [[float(v) for v in x]]

    # case B: already list of boxes
    if (
        isinstance(x, list)
        and len(x) > 0
        and all(isinstance(b, (list, tuple)) for b in x)
        and all(len(b) == 4 for b in x)
        and all(all(_is_scalar(v) for v in b) for b in x)
    ):
        return [[float(v) for v in b] for b in x]

    # case C: transposed format: 4 rows, each row is a time-series
    # Example:
    #   [[27,26,26], [21,21,21], [63,64,65], [74,72,71]]
    if (
        isinstance(x, list)
        and len(x) == 4
        and all(isinstance(row, (list, tuple)) for row in x)
    ):
        rows = []
        for row in x:
            flat_row = []
            for v in row:
                # unwrap [27] -> 27
                while isinstance(v, (list, tuple)) and len(v) == 1:
                    v = v[0]
                if not _is_scalar(v):
                    raise ValueError(f"Unsupported nested GT box entry: {v}")
                flat_row.append(float(v))
            rows.append(flat_row)

        T = min(len(r) for r in rows)
        if T == 0:
            raise ValueError("Empty transposed GT box rows.")
        return [[rows[0][i], rows[1][i], rows[2][i], rows[3][i]] for i in range(T)]

    raise ValueError(f"Unsupported gt_boxes_abs structure: {str(x)[:300]}")

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = box_area(box_a) + box_area(box_b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def box_center(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def center_distance(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def scale_error(box_a, box_b):
    aw = max(1e-6, box_a[2] - box_a[0])
    ah = max(1e-6, box_a[3] - box_a[1])
    bw = max(1e-6, box_b[2] - box_b[0])
    bh = max(1e-6, box_b[3] - box_b[1])
    return abs(aw - bw) + abs(ah - bh)


def is_valid_box(box):
    x1, y1, x2, y2 = box
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def pad_or_trim_boxes(pred_boxes, gt_boxes):
    target_len = len(gt_boxes)

    if len(pred_boxes) != target_len:
        print(f"Why pred and gt have different length? PRED: {pred_boxes}, GT: {gt_boxes}")

    pred_boxes = pred_boxes[:target_len]
    while len(pred_boxes) < target_len:
        pred_boxes.append([0.0, 0.0, 0.0, 0.0])
    return pred_boxes


def extract_step_end_token_indices(text: str, tokenizer) -> List[int]:
    """
    Map each parsed box occurrence to the token index (within completion only)
    where that frame-step ends.

    We approximate step boundaries by the end of each box string in the generated text.
    """
    matches = list(BOX_PATTERN_TAG.finditer(text))
    if len(matches) == 0:
        matches = list(BOX_PATTERN_BRACKET.finditer(text))

    end_token_indices = []
    for m in matches:
        prefix = text[:m.end()]
        tok = tokenizer(prefix, add_special_tokens=False, return_attention_mask=False)
        end_token_indices.append(len(tok["input_ids"]))
    return end_token_indices

def assert_gt_boxes_are_norm_xyxy(gt_boxes):
    for i, b in enumerate(gt_boxes):
        if len(b) != 4:
            raise ValueError(f"GT box {i} does not have 4 values: {b}")

        x1, y1, x2, y2 = [float(v) for v in b]

        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError(
                f"GT box {i} is not normalized xyxy in [0,1]: {b}"
            )

def compute_step_rewards(
    pred_boxes_norm: List[List[float]],
    gt_boxes_abs,
    image_size,
):
    gt_boxes_norm = canonicalize_gt_boxes(gt_boxes_abs)
    assert_gt_boxes_are_norm_xyxy(gt_boxes_norm)
    pred_boxes_norm = pad_or_trim_boxes(pred_boxes_norm, gt_boxes_norm)

    # print("======================================== DEBUG Step Rewards Computation ========================================")

    # print(f"GT boxes (abs): {gt_boxes_abs}")
    # print(f"GT boxes (norm): {gt_boxes_norm}")
    # print(f"Pred boxes (norm): {pred_boxes_norm}")


    step_rewards = []
    valid_count = 0
    ious = []
    center_errs = []
    jitters = []

    prev_box = None
    # print(f"Comparing {len(pred_boxes_norm)} predicted boxes to {len(gt_boxes_norm)} GT boxes...")
    for pb, gb in zip(pred_boxes_norm, gt_boxes_norm):
        # print(f"Comparing pred box {pb} to GT box {gb}...")

        valid = is_valid_box(pb)
        valid_bonus = 0.2 if valid else -1.0

        if valid:
            valid_count += 1
            iou = box_iou(pb, gb)
            c_err = center_distance(pb, gb)
            s_err = scale_error(pb, gb)
        else:
            iou = 0.0
            c_err = 2.0
            s_err = 2.0

        if prev_box is None:
            jitter = 0.0
        else:
            jitter = center_distance(prev_box, pb)
        prev_box = pb

        reward = (
            2.0 * iou
            + valid_bonus
            - 0.25 * c_err
            - 0.10 * s_err
            - 0.05 * jitter
        )
        step_rewards.append(float(reward))
        ious.append(float(iou))
        center_errs.append(float(c_err))
        jitters.append(float(jitter))

        # print(f"Step {len(step_rewards)}: valid={valid} iou={iou:.4f} c_err={c_err:.4f} s_err={s_err:.4f} jitter={jitter:.4f} reward={reward:.4f}")

    # print("---------------------------------------- Step Rewards Summary ---------------------------------------")
    # print(f"Total valid boxes: {valid_count} / {len(gt_boxes_norm)}")
    # print(f"Final step rewards: {step_rewards}")
    # print(f"Mean IoU: {sum(ious) / len(ious):.4f}")
    # print(f"Mean center error: {sum(center_errs) / len(center_errs):.4f}")
    # print(f"Mean jitter: {sum(jitters) / len(jitters):.4f}")
    # print(f"Valid box rate: {valid_count} / {len(gt_boxes_norm)} = {valid_count / len(gt_boxes_norm):.4f}")

    # print("================================================================================================================")

    return {
        "step_rewards": step_rewards,
        "mean_iou": sum(ious) / len(ious),
        "mean_center_error": sum(center_errs) / len(center_errs),
        "format_valid_rate": valid_count / len(gt_boxes_norm),
        "mean_jitter": sum(jitters) / len(jitters),
        "pred_boxes": pred_boxes_norm,
    }

def safe_mean(xs, default=0.0):
    if len(xs) == 0:
        return float(default)
    return float(sum(xs) / len(xs))

def motion_mismatch(prev_pred, cur_pred, prev_gt, cur_gt):
    """
    Penalize wrong motion, not motion itself.

    Old jitter:
      distance(pred_t, pred_t-1)

    Better jitter:
      distance(pred_motion, gt_motion)
    """
    p0x, p0y = box_center(prev_pred)
    p1x, p1y = box_center(cur_pred)

    g0x, g0y = box_center(prev_gt)
    g1x, g1y = box_center(cur_gt)

    pred_dx = p1x - p0x
    pred_dy = p1y - p0y

    gt_dx = g1x - g0x
    gt_dy = g1y - g0y

    return math.sqrt((pred_dx - gt_dx) ** 2 + (pred_dy - gt_dy) ** 2)

def compute_sequence_reward(
    pred_boxes_norm: List[List[float]],
    gt_boxes,
    image_size,
    ignore_first_frame: bool = False,
):
    """
    Sequence-level reward for GRPO SOT training.

    Assumptions:
      - GT boxes are normalized xyxy in [0,1].
      - Predicted boxes are parsed as normalized xyxy in [0,1].
      - For SOT, Frame 1 is given in the prompt, so reward should focus on Frame 2..T.

    Reward:
      + 5.0  * mean IoU
      + 1.0  * complete
      + 0.5  * valid rate
      - 0.05 * mean center error
      - 0.02 * mean scale error
      - 0.01 * mean motion mismatch
      - 0.20 * extra box rate
    """
    gt_boxes_norm = canonicalize_gt_boxes(gt_boxes)
    assert_gt_boxes_are_norm_xyxy(gt_boxes_norm)

    T = len(gt_boxes_norm)
    if T == 0:
        raise ValueError("Empty GT boxes.")

    raw_pred_count = len(pred_boxes_norm)

    # Missing boxes become invalid [0,0,0,0], not repeated-last.
    pred_boxes_aligned = pad_or_trim_boxes(pred_boxes_norm, gt_boxes_norm)

    if ignore_first_frame and T > 1:
        active_indices = list(range(1, T))
    else:
        active_indices = list(range(T))

    valid_flags = [is_valid_box(b) for b in pred_boxes_aligned]

    ious = []
    center_errs = []
    scale_errs = []
    motion_errs = []

    for t in active_indices:
        pred_box = pred_boxes_aligned[t]
        gt_box = gt_boxes_norm[t]

        valid = valid_flags[t]

        if valid:
            iou = box_iou(pred_box, gt_box)
            center_err = center_distance(pred_box, gt_box)
            scale_err_val = scale_error(pred_box, gt_box)
        else:
            iou = 0.0
            center_err = 1.0
            scale_err_val = 1.0

        ious.append(float(iou))
        center_errs.append(float(center_err))
        scale_errs.append(float(scale_err_val))

        # Motion mismatch is only meaningful when current and previous pred boxes are valid.
        if t > 0 and valid_flags[t] and valid_flags[t - 1]:
            motion_errs.append(
                motion_mismatch(
                    pred_boxes_aligned[t - 1],
                    pred_boxes_aligned[t],
                    gt_boxes_norm[t - 1],
                    gt_boxes_norm[t],
                )
            )

    mean_iou = safe_mean(ious, default=0.0)
    mean_center_error = safe_mean(center_errs, default=1.0)
    mean_scale_error = safe_mean(scale_errs, default=1.0)
    mean_jitter = safe_mean(motion_errs, default=0.0)

    valid_rate = safe_mean(
        [1.0 if valid_flags[t] else 0.0 for t in active_indices],
        default=0.0,
    )

    # Strict completeness: model should emit at least one box per frame.
    complete = 1.0 if raw_pred_count >= T else 0.0

    # Diagnostic only: partial completion ratio.
    complete_rate = min(raw_pred_count, T) / float(T)

    # Penalize coordinate spam.
    extra_rate = max(0, raw_pred_count - T) / float(T)

    sequence_reward = (
        5.0 * mean_iou
        + 1.0 * complete
        + 0.5 * valid_rate
        - 0.05 * mean_center_error
        - 0.02 * mean_scale_error
        - 0.01 * mean_jitter
        - 0.20 * extra_rate
    )

    return {
        "sequence_reward": float(sequence_reward),
        "mean_iou": float(mean_iou),
        "complete": float(complete),
        "complete_rate": float(complete_rate),
        "format_valid_rate": float(valid_rate),
        "mean_center_error": float(mean_center_error),
        "mean_scale_error": float(mean_scale_error),
        "mean_jitter": float(mean_jitter),
        "extra_rate": float(extra_rate),
        "num_pred_boxes": int(raw_pred_count),
        "num_gt_boxes": int(T),
        "pred_boxes": pred_boxes_aligned,
    }
