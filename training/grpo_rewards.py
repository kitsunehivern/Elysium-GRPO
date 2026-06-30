from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Number = float
Box = Tuple[Number, Number, Number, Number]


@dataclass
class TrackingComponentWeights:
    iou: float = 0.65
    center: float = 0.15
    temporal: float = 0.10
    validity: float = 0.10


@dataclass
class FinalRewardWeights:
    format: float = 0.10
    accuracy: float = 0.90
    semantic: float = 0.00


@dataclass
class RewardConfig:
    tracking_weights: TrackingComponentWeights = field(default_factory=TrackingComponentWeights)
    final_weights: FinalRewardWeights = field(default_factory=FinalRewardWeights)
    coordinate_scale: float = 100.0
    center_tau: float = 10.0
    temporal_tau: float = 20.0
    count_mismatch_penalty: float = 0.50
    clamp_for_metrics: bool = True
    semantic_gate: float = 0.05
    format_style: str = "answer_only"
    require_frame_prefix: bool = False


BOX_RE = re.compile(
    r"[\[\{]\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*[\]\}]"
)
FRAME_BOX_RE = re.compile(
    r"Frame\s*\d+\s*:\s*[\[\{]\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*[\]\}]",
    flags=re.IGNORECASE,
)
COT_RE = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", flags=re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)


def extract_answer_text(text: str) -> str:
    """Return content inside <answer> tags if present; otherwise return text."""

    if not isinstance(text, str):
        return ""
    match = ANSWER_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def extract_think_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    match = THINK_RE.search(text)
    return match.group(1).strip() if match else ""


def extract_video_description_span(text: str, max_words: int = 96) -> str:
    think = extract_think_text(text) or text
    if "." in think:
        think = think[think.find(".") + 1 :].lstrip()
    words = think.split()
    return " ".join(words[:max_words])


def parse_boxes(text: str, prefer_frame_prefix: bool = False) -> List[Box]:
    if not isinstance(text, str):
        return []
    answer = extract_answer_text(text)
    regex = FRAME_BOX_RE if prefer_frame_prefix else BOX_RE
    matches = list(regex.finditer(answer))
    if prefer_frame_prefix and not matches:
        matches = list(BOX_RE.finditer(answer))
    boxes: List[Box] = []
    for match in matches:
        try:
            vals = tuple(float(match.group(i)) for i in range(1, 5))
            boxes.append(vals)  # type: ignore[arg-type]
        except Exception:
            continue
    return boxes


def _valid_box(box: Box, scale: float = 100.0) -> bool:
    x1, y1, x2, y2 = box
    return 0.0 <= x1 < x2 <= scale and 0.0 <= y1 < y2 <= scale


def _clamp_box(box: Box, scale: float = 100.0) -> Box:
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), scale)
    y1 = min(max(y1, 0.0), scale)
    x2 = min(max(x2, 0.0), scale)
    y2 = min(max(y2, 0.0), scale)
    if x2 <= x1:
        x1, x2 = min(x1, x2), max(x1, x2)
        x2 = min(scale, x2 + 1e-6)
    if y2 <= y1:
        y1, y2 = min(y1, y2), max(y1, y2)
        y2 = min(scale, y2 + 1e-6)
    return x1, y1, x2, y2


def box_iou(pred: Box, gt: Box, scale: float = 100.0, clamp: bool = True) -> float:
    if clamp:
        pred = _clamp_box(pred, scale)
        gt = _clamp_box(gt, scale)
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    p_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    g_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = p_area + g_area - inter
    if union <= 0:
        return 0.0
    return max(0.0, min(1.0, inter / union))


def box_center(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def dense_center_reward(pred_boxes: Sequence[Box], gt_boxes: Sequence[Box], tau: float) -> float:
    if not pred_boxes or not gt_boxes:
        return 0.0
    vals = []
    for pred, gt in zip(pred_boxes, gt_boxes):
        pcx, pcy = box_center(pred)
        gcx, gcy = box_center(gt)
        dist = math.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2)
        vals.append(math.exp(-dist / max(tau, 1e-6)))
    return sum(vals) / max(1, len(gt_boxes))


def temporal_consistency_reward(pred_boxes: Sequence[Box], gt_boxes: Sequence[Box], tau: float) -> float:
    if len(gt_boxes) <= 1:
        return 1.0 if pred_boxes else 0.0
    if len(pred_boxes) <= 1:
        return 0.0
    vals = []
    for i in range(1, min(len(pred_boxes), len(gt_boxes))):
        pc0, pc1 = box_center(pred_boxes[i - 1]), box_center(pred_boxes[i])
        gc0, gc1 = box_center(gt_boxes[i - 1]), box_center(gt_boxes[i])
        pv = (pc1[0] - pc0[0], pc1[1] - pc0[1])
        gv = (gc1[0] - gc0[0], gc1[1] - gc0[1])
        err = math.sqrt((pv[0] - gv[0]) ** 2 + (pv[1] - gv[1]) ** 2)
        vals.append(math.exp(-err / max(tau, 1e-6)))
    # Missing late frames should reduce temporal score.
    denom = max(1, len(gt_boxes) - 1)
    return sum(vals) / denom


def validity_reward(pred_boxes: Sequence[Box], scale: float) -> float:
    if not pred_boxes:
        return 0.0
    return sum(1.0 for b in pred_boxes if _valid_box(b, scale)) / len(pred_boxes)


def format_reward(text: str, expected_num_boxes: int, cfg: RewardConfig) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    boxes = parse_boxes(text, prefer_frame_prefix=cfg.require_frame_prefix)
    count_ok = len(boxes) == expected_num_boxes

    if cfg.format_style == "cot_answer":
        cot_ok = COT_RE.fullmatch(text.strip()) is not None
        # Require the tags, then give partial credit for a parseable but wrong-count answer.
        if cot_ok and count_ok:
            return 1.0
        if cot_ok and boxes:
            return 0.6
        if boxes:
            return 0.25
        return 0.0

    # Original Elysium answer-only format.
    if count_ok:
        return 1.0
    if boxes:
        return max(0.1, 1.0 - cfg.count_mismatch_penalty * abs(len(boxes) - expected_num_boxes) / max(expected_num_boxes, 1))
    return 0.0


def trajectory_accuracy_reward(pred_boxes: Sequence[Box], gt_boxes: Sequence[Box], cfg: RewardConfig) -> Tuple[float, Dict[str, float]]:
    """Compute the geometry/accuracy reward R_track."""

    if not gt_boxes:
        return 0.0, {"iou": 0.0, "center": 0.0, "temporal": 0.0, "validity": 0.0}
    if not pred_boxes:
        return 0.0, {"iou": 0.0, "center": 0.0, "temporal": 0.0, "validity": 0.0}

    usable = min(len(pred_boxes), len(gt_boxes))
    pred_used = list(pred_boxes[:usable])
    gt_used = list(gt_boxes[:usable])

    iou_sum = sum(box_iou(p, g, cfg.coordinate_scale, cfg.clamp_for_metrics) for p, g in zip(pred_used, gt_used))
    # Divide by gt length, not matched length, so missing frames are penalized.
    iou_score = iou_sum / max(1, len(gt_boxes))
    center_score = dense_center_reward(pred_used, gt_used, cfg.center_tau) * (usable / max(1, len(gt_boxes)))
    temporal_score = temporal_consistency_reward(pred_used, gt_used, cfg.temporal_tau)
    valid_score = validity_reward(pred_boxes, cfg.coordinate_scale)

    # Mild penalty for too many/few boxes, separate from the format reward.
    count_delta = abs(len(pred_boxes) - len(gt_boxes))
    count_factor = max(0.0, 1.0 - cfg.count_mismatch_penalty * count_delta / max(1, len(gt_boxes)))

    w = cfg.tracking_weights
    denom = max(1e-8, w.iou + w.center + w.temporal + w.validity)
    track = (w.iou * iou_score + w.center * center_score + w.temporal * temporal_score + w.validity * valid_score) / denom
    track *= count_factor
    track = max(0.0, min(1.0, track))
    return track, {
        "iou": iou_score,
        "center": center_score,
        "temporal": temporal_score,
        "validity": valid_score,
        "count_factor": count_factor,
    }


def compute_tracking_reward(
    completion: str,
    ground_truth: str,
    cfg: RewardConfig,
    semantic_reward_value: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    pred_boxes = parse_boxes(completion, prefer_frame_prefix=cfg.require_frame_prefix)
    gt_boxes = parse_boxes(ground_truth, prefer_frame_prefix=False)
    r_format = format_reward(completion, len(gt_boxes), cfg)
    r_track, parts = trajectory_accuracy_reward(pred_boxes, gt_boxes, cfg)

    fw = cfg.final_weights
    r_sem = max(0.0, min(1.0, float(semantic_reward_value)))
    gated_sem = r_sem if r_track > cfg.semantic_gate else 0.0
    reward = fw.format * r_format + fw.accuracy * r_track + fw.semantic * gated_sem
    reward = max(0.0, min(1.0, reward))

    metrics = {
        "reward": reward,
        "format": r_format,
        "accuracy": r_track,
        "semantic": r_sem,
        "semantic_gated": gated_sem,
        "num_pred_boxes": float(len(pred_boxes)),
        "num_gt_boxes": float(len(gt_boxes)),
        **parts,
    }
    return reward, metrics


def compute_batch_tracking_rewards(
    completions: Sequence[str],
    ground_truths: Sequence[str],
    cfg: Optional[RewardConfig] = None,
    semantic_rewards: Optional[Sequence[float]] = None,
) -> Tuple[List[float], Dict[str, float]]:
    cfg = cfg or RewardConfig()
    if semantic_rewards is None:
        semantic_rewards = [0.0] * len(completions)

    rewards: List[float] = []
    metrics_accum: Dict[str, List[float]] = {}
    for completion, gt, sem in zip(completions, ground_truths, semantic_rewards):
        reward, metrics = compute_tracking_reward(completion, gt, cfg, semantic_reward_value=sem)
        rewards.append(reward)
        for key, value in metrics.items():
            metrics_accum.setdefault(key, []).append(float(value))

    metrics_mean = {key: sum(vals) / max(1, len(vals)) for key, vals in metrics_accum.items()}
    return rewards, metrics_mean
