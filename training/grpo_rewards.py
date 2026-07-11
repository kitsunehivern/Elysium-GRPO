from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field, fields
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

Number = float
Box = Tuple[Number, Number, Number, Number]


@dataclass
class TrackingComponentWeights:
    # Legacy components. Defaults intentionally match the original implementation.
    iou: float = 0.65
    center: float = 0.15
    temporal: float = 0.10
    validity: float = 0.10

    # Optional geometry/motion components. Their zero defaults make every existing
    # YAML file numerically backward compatible.
    size: float = 0.00
    aspect: float = 0.00
    boundary: float = 0.00
    jump: float = 0.00


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

    # New optional reward temperatures/tolerances. They have no effect while the
    # corresponding component weight is zero.
    size_tau: float = 0.70
    aspect_tau: float = 0.50
    boundary_tau: float = 5.0
    jump_tau: float = 10.0
    jump_margin: float = 2.0


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


def _positive_geometry(box: Box) -> bool:
    x1, y1, x2, y2 = box
    return x2 > x1 and y2 > y1


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


def box_size(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


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
    """Motion-aware temporal reward.

    This is deliberately *not* a smoothness reward. It compares predicted center
    displacement with ground-truth displacement, so matching fast motion is fully
    rewarded while a static prediction during fast target motion is penalized.
    """

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


def size_reward(pred_boxes: Sequence[Box], gt_boxes: Sequence[Box], tau: float) -> float:
    """Symmetric area-ratio reward using log space.

    ``exp(-|log(area_pred / area_gt)| / tau)`` gives the same penalty to an area
    ratio r and 1/r, preventing a systematic preference for oversized boxes.
    """

    if not pred_boxes or not gt_boxes:
        return 0.0
    vals: List[float] = []
    eps = 1e-8
    for pred, gt in zip(pred_boxes, gt_boxes):
        pw, ph = box_size(pred)
        gw, gh = box_size(gt)
        if pw <= 0.0 or ph <= 0.0 or gw <= 0.0 or gh <= 0.0:
            vals.append(0.0)
            continue
        log_ratio_error = abs(math.log((pw * ph + eps) / (gw * gh + eps)))
        vals.append(math.exp(-log_ratio_error / max(tau, 1e-6)))
    return sum(vals) / max(1, len(gt_boxes))


def aspect_reward(pred_boxes: Sequence[Box], gt_boxes: Sequence[Box], tau: float) -> float:
    """Symmetric aspect-ratio reward in log space."""

    if not pred_boxes or not gt_boxes:
        return 0.0
    vals: List[float] = []
    eps = 1e-8
    for pred, gt in zip(pred_boxes, gt_boxes):
        pw, ph = box_size(pred)
        gw, gh = box_size(gt)
        if pw <= 0.0 or ph <= 0.0 or gw <= 0.0 or gh <= 0.0:
            vals.append(0.0)
            continue
        pred_ar = (pw + eps) / (ph + eps)
        gt_ar = (gw + eps) / (gh + eps)
        log_ratio_error = abs(math.log(pred_ar / gt_ar))
        vals.append(math.exp(-log_ratio_error / max(tau, 1e-6)))
    return sum(vals) / max(1, len(gt_boxes))


def _boundary_overflow(box: Box, scale: float) -> float:
    x1, y1, x2, y2 = box
    return (
        max(0.0, -x1)
        + max(0.0, -y1)
        + max(0.0, x2 - scale)
        + max(0.0, y2 - scale)
    )


def boundary_reward(
    pred_boxes: Sequence[Box],
    gt_boxes: Sequence[Box],
    scale: float,
    tau: float,
) -> float:
    """Soft, GT-aware boundary reward.

    Legacy ``validity_reward`` already performs a binary ordering *and* in-frame
    check. This component instead penalizes the predicted overflow only when it is
    larger than the GT overflow. Consequently, a correct partially out-of-view box
    is not punished solely because the annotated target crosses an image boundary.
    """

    if not pred_boxes or not gt_boxes:
        return 0.0
    vals: List[float] = []
    for pred, gt in zip(pred_boxes, gt_boxes):
        excess_overflow = max(
            0.0,
            _boundary_overflow(pred, scale) - _boundary_overflow(gt, scale),
        )
        vals.append(math.exp(-excess_overflow / max(tau, 1e-6)))
    # Divide by GT length so missing frames are penalized consistently with IoU.
    return sum(vals) / max(1, len(gt_boxes))


def jump_reward(
    pred_boxes: Sequence[Box],
    gt_boxes: Sequence[Box],
    tau: float,
    margin: float,
) -> float:
    """Penalize only motion larger than the target's actual motion.

    For each transition, ``excess = max(0, |delta_pred| - |delta_gt| - margin)``.
    Therefore, matching a fast-moving target is not discouraged. Directional and
    vector mismatch is already captured by ``temporal_consistency_reward``.
    """

    if len(gt_boxes) <= 1:
        return 1.0 if pred_boxes else 0.0
    if len(pred_boxes) <= 1:
        return 0.0
    vals: List[float] = []
    for i in range(1, min(len(pred_boxes), len(gt_boxes))):
        pc0, pc1 = box_center(pred_boxes[i - 1]), box_center(pred_boxes[i])
        gc0, gc1 = box_center(gt_boxes[i - 1]), box_center(gt_boxes[i])
        pred_motion = math.hypot(pc1[0] - pc0[0], pc1[1] - pc0[1])
        gt_motion = math.hypot(gc1[0] - gc0[0], gc1[1] - gc0[1])
        excess = max(0.0, pred_motion - gt_motion - max(0.0, margin))
        vals.append(math.exp(-excess / max(tau, 1e-6)))
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
        return max(
            0.1,
            1.0
            - cfg.count_mismatch_penalty
            * abs(len(boxes) - expected_num_boxes)
            / max(expected_num_boxes, 1),
        )
    return 0.0


def trajectory_accuracy_reward(
    pred_boxes: Sequence[Box], gt_boxes: Sequence[Box], cfg: RewardConfig
) -> Tuple[float, Dict[str, float]]:
    """Compute the geometry/accuracy reward R_track."""

    empty_parts = {
        "iou": 0.0,
        "center": 0.0,
        "temporal": 0.0,
        "validity": 0.0,
        "size": 0.0,
        "aspect": 0.0,
        "boundary": 0.0,
        "jump": 0.0,
        "count_factor": 0.0,
    }
    if not gt_boxes or not pred_boxes:
        return 0.0, empty_parts

    usable = min(len(pred_boxes), len(gt_boxes))
    pred_used = list(pred_boxes[:usable])
    gt_used = list(gt_boxes[:usable])

    iou_sum = sum(
        box_iou(p, g, cfg.coordinate_scale, cfg.clamp_for_metrics)
        for p, g in zip(pred_used, gt_used)
    )
    # Divide by gt length, not matched length, so missing frames are penalized.
    iou_score = iou_sum / max(1, len(gt_boxes))
    matched_fraction = usable / max(1, len(gt_boxes))
    center_score = dense_center_reward(pred_used, gt_used, cfg.center_tau) * matched_fraction
    temporal_score = temporal_consistency_reward(pred_used, gt_used, cfg.temporal_tau)
    valid_score = validity_reward(pred_boxes, cfg.coordinate_scale)
    size_score = size_reward(pred_used, gt_used, cfg.size_tau) * matched_fraction
    aspect_score = aspect_reward(pred_used, gt_used, cfg.aspect_tau) * matched_fraction
    boundary_score = boundary_reward(pred_used, gt_used, cfg.coordinate_scale, cfg.boundary_tau)
    jump_score = jump_reward(pred_used, gt_used, cfg.jump_tau, cfg.jump_margin)

    # Mild penalty for too many/few boxes, separate from the format reward.
    count_delta = abs(len(pred_boxes) - len(gt_boxes))
    count_factor = max(
        0.0,
        1.0
        - cfg.count_mismatch_penalty * count_delta / max(1, len(gt_boxes)),
    )

    w = cfg.tracking_weights
    weighted_components = (
        (w.iou, iou_score),
        (w.center, center_score),
        (w.temporal, temporal_score),
        (w.validity, valid_score),
        (w.size, size_score),
        (w.aspect, aspect_score),
        (w.boundary, boundary_score),
        (w.jump, jump_score),
    )
    denom = max(1e-8, sum(weight for weight, _ in weighted_components))
    track = sum(weight * score for weight, score in weighted_components) / denom
    track *= count_factor
    track = max(0.0, min(1.0, track))
    return track, {
        "iou": iou_score,
        "center": center_score,
        "temporal": temporal_score,
        "validity": valid_score,
        "size": size_score,
        "aspect": aspect_score,
        "boundary": boundary_score,
        "jump": jump_score,
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
        reward, metrics = compute_tracking_reward(
            completion, gt, cfg, semantic_reward_value=sem
        )
        rewards.append(reward)
        for key, value in metrics.items():
            metrics_accum.setdefault(key, []).append(float(value))

    metrics_mean = {
        key: sum(vals) / max(1, len(vals)) for key, vals in metrics_accum.items()
    }
    return rewards, metrics_mean


# ---------------------------------------------------------------------------
# YAML parsing and curriculum scheduling
# ---------------------------------------------------------------------------


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _dataclass_from_mapping(cls, values: object, defaults=None):
    values_map = _as_mapping(values)
    base = defaults if defaults is not None else cls()
    kwargs = {}
    for item in fields(cls):
        current = getattr(base, item.name)
        raw = values_map.get(item.name, current)
        kwargs[item.name] = float(raw)
    return cls(**kwargs)


def reward_config_from_dict(raw_reward_cfg: object, defaults: Optional[RewardConfig] = None) -> RewardConfig:
    """Parse ``grpo.reward`` while preserving old YAML behavior.

    Unknown keys are ignored. Missing new keys get zero component weights, so
    all existing configs produce the same scalar reward as before this extension.
    """

    raw = _as_mapping(raw_reward_cfg)
    base = copy.deepcopy(defaults) if defaults is not None else RewardConfig()
    tracking = _dataclass_from_mapping(
        TrackingComponentWeights,
        raw.get("tracking_weights", {}),
        defaults=base.tracking_weights,
    )
    final = _dataclass_from_mapping(
        FinalRewardWeights,
        raw.get("final_weights", {}),
        defaults=base.final_weights,
    )

    scalar_float_fields = {
        "coordinate_scale",
        "center_tau",
        "temporal_tau",
        "count_mismatch_penalty",
        "semantic_gate",
        "size_tau",
        "aspect_tau",
        "boundary_tau",
        "jump_tau",
        "jump_margin",
    }
    kwargs = {
        "tracking_weights": tracking,
        "final_weights": final,
    }
    for name in scalar_float_fields:
        kwargs[name] = float(raw.get(name, getattr(base, name)))
    kwargs["clamp_for_metrics"] = bool(raw.get("clamp_for_metrics", base.clamp_for_metrics))
    kwargs["format_style"] = str(raw.get("format_style", base.format_style))
    kwargs["require_frame_prefix"] = bool(
        raw.get("require_frame_prefix", base.require_frame_prefix)
    )
    return RewardConfig(**kwargs)


def _lerp(a: float, b: float, alpha: float) -> float:
    return float(a) + (float(b) - float(a)) * float(alpha)


def _interpolate_dataclass(left, right, alpha: float):
    cls = type(left)
    return cls(
        **{
            item.name: _lerp(getattr(left, item.name), getattr(right, item.name), alpha)
            for item in fields(cls)
        }
    )


def _interpolate_reward_config(early: RewardConfig, late: RewardConfig, alpha: float) -> RewardConfig:
    alpha = max(0.0, min(1.0, float(alpha)))
    numeric_names = (
        "coordinate_scale",
        "center_tau",
        "temporal_tau",
        "count_mismatch_penalty",
        "semantic_gate",
        "size_tau",
        "aspect_tau",
        "boundary_tau",
        "jump_tau",
        "jump_margin",
    )
    kwargs = {
        "tracking_weights": _interpolate_dataclass(
            early.tracking_weights, late.tracking_weights, alpha
        ),
        "final_weights": _interpolate_dataclass(
            early.final_weights, late.final_weights, alpha
        ),
    }
    for name in numeric_names:
        kwargs[name] = _lerp(getattr(early, name), getattr(late, name), alpha)

    # Non-numeric behavior should not switch halfway through a run. Keep early
    # values during interpolation, then use late at the completed transition.
    source = late if alpha >= 1.0 else early
    kwargs["clamp_for_metrics"] = source.clamp_for_metrics
    kwargs["format_style"] = source.format_style
    kwargs["require_frame_prefix"] = source.require_frame_prefix
    return RewardConfig(**kwargs)


def reward_config_weight_metrics(cfg: RewardConfig) -> Dict[str, float]:
    tw = cfg.tracking_weights
    fw = cfg.final_weights
    return {
        "curriculum/weight_iou": tw.iou,
        "curriculum/weight_center": tw.center,
        "curriculum/weight_temporal": tw.temporal,
        "curriculum/weight_validity": tw.validity,
        "curriculum/weight_size": tw.size,
        "curriculum/weight_aspect": tw.aspect,
        "curriculum/weight_boundary": tw.boundary,
        "curriculum/weight_jump": tw.jump,
        "curriculum/weight_format": fw.format,
        "curriculum/weight_accuracy": fw.accuracy,
        "curriculum/weight_semantic": fw.semantic,
    }


def resolve_curriculum_reward_config(
    base_cfg: RewardConfig,
    curriculum_cfg: object,
    global_step: int,
    max_steps: int,
) -> Tuple[RewardConfig, Dict[str, float]]:
    """Return the active reward configuration for the current optimizer step.

    Supported YAML shape::

        curriculum:
          enabled: true
          schedule: linear       # linear | cosine | step
          start_step: 400        # or start_ratio: 0.20
          end_step: 1400         # or end_ratio: 0.70
          early:
            tracking_weights: {...}
            final_weights: {...}
          late:
            tracking_weights: {...}
            final_weights: {...}

    ``early`` and ``late`` are partial overrides of the legacy/base reward config.
    If the whole block is absent or disabled, ``base_cfg`` is returned unchanged.
    """

    raw = _as_mapping(curriculum_cfg)
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        metrics = {
            "curriculum/enabled": 0.0,
            "curriculum/progress": 0.0,
            "curriculum/phase": 0.0,
            **reward_config_weight_metrics(base_cfg),
        }
        return base_cfg, metrics

    early_cfg = reward_config_from_dict(raw.get("early", {}), defaults=base_cfg)
    late_cfg = reward_config_from_dict(raw.get("late", {}), defaults=base_cfg)

    effective_max_steps = max(1, int(max_steps))
    if "start_step" in raw:
        start_step = float(raw["start_step"])
    else:
        start_step = float(raw.get("start_ratio", 0.0)) * effective_max_steps

    if "end_step" in raw:
        end_step = float(raw["end_step"])
    else:
        end_step = float(raw.get("end_ratio", 0.60)) * effective_max_steps

    if end_step <= start_step:
        end_step = start_step + 1.0

    raw_progress = (float(global_step) - start_step) / (end_step - start_step)
    progress = max(0.0, min(1.0, raw_progress))
    schedule = str(raw.get("schedule", "linear")).lower()
    if schedule in {"step", "two_stage", "two-stage"}:
        alpha = 0.0 if float(global_step) < end_step else 1.0
    elif schedule == "cosine":
        alpha = 0.5 - 0.5 * math.cos(math.pi * progress)
    else:
        alpha = progress

    active = _interpolate_reward_config(early_cfg, late_cfg, alpha)
    if float(global_step) < start_step:
        phase = 0.0  # early hold
    elif float(global_step) >= end_step:
        phase = 2.0  # late hold
    else:
        phase = 1.0  # transition

    metrics = {
        "curriculum/enabled": 1.0,
        "curriculum/progress": float(alpha),
        "curriculum/phase": phase,
        "curriculum/start_step": float(start_step),
        "curriculum/end_step": float(end_step),
        **reward_config_weight_metrics(active),
    }
    return active, metrics
