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


def pad_or_trim_boxes(pred_boxes: List[List[float]], target_len: int) -> List[List[float]]:
    pred_boxes = pred_boxes[:target_len]
    if len(pred_boxes) == 0:
        pred_boxes = [[0.0, 0.0, 0.0, 0.0] for _ in range(target_len)]
    while len(pred_boxes) < target_len:
        pred_boxes.append(pred_boxes[-1])
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


def compute_step_rewards(
    pred_boxes_norm: List[List[float]],
    gt_boxes_abs,
    image_size,
):
    gt_boxes_abs = canonicalize_gt_boxes(gt_boxes_abs)
    gt_boxes_norm = gt_boxes_abs
    pred_boxes_norm = pad_or_trim_boxes(pred_boxes_norm, len(gt_boxes_norm))

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

def compute_sequence_reward(
    pred_boxes_norm: List[List[float]],
    gt_boxes,
    image_size,
    ignore_first_frame: bool = False,
):
    info = compute_step_rewards(pred_boxes_norm, gt_boxes, image_size)
    start = 1 if ignore_first_frame and len(info["step_rewards"]) > 1 else 0
    active = info["step_rewards"][start:]
    info["sequence_reward"] = float(sum(active) / max(1, len(active)))
    return info

def build_process_advantages(
    completion_token_lens: List[int],
    step_end_token_indices: List[List[int]],
    step_rewards: List[List[float]],
) -> List[List[float]]:
    """
    DeepSeekMath process supervision:
    normalize process rewards within the sampled group,
    then advantage at token t = sum of normalized rewards from future steps. :contentReference[oaicite:5]{index=5}
    """
    flat = []
    for rewards in step_rewards:
        flat.extend(rewards)

    if len(flat) == 0:
        flat = [0.0]
    
    mean_r = sum(flat) / len(flat)
    std_r = (sum((x - mean_r) ** 2 for x in flat) / len(flat)) ** 0.5
    std_r = max(std_r, 1e-6)

    normalized_step_rewards = []
    for rewards in step_rewards:
        normalized_step_rewards.append([(r - mean_r) / std_r for r in rewards])

    advantages = []
    for comp_len, ends, norm_rewards in zip(completion_token_lens, step_end_token_indices, normalized_step_rewards):
        adv = [0.0 for _ in range(comp_len)]
        # token positions are 1-based in ends; adv list is 0-based
        for t in range(comp_len):
            s = 0.0
            for step_end, r in zip(ends, norm_rewards):
                if step_end >= (t + 1):
                    s += r
            adv[t] = s
        advantages.append(adv)
    return advantages