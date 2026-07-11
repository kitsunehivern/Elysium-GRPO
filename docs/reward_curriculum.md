# Curriculum and geometry-aware GRPO rewards

This extension keeps every existing reward config backward compatible. If the new
fields are omitted, or their weights remain `0.0`, the scalar reward is identical
to the previous implementation.

## What changed

The legacy tracking reward remains:

\[
R_{track}=w_{iou}R_{iou}+w_cR_{center}+w_tR_{temporal}+w_vR_{validity}.
\]

Four optional components are now available:

### Size reward

\[
R_{size}=\exp\left(-\frac{|\log(A_p/A_g)|}{\tau_{size}}\right).
\]

The log area ratio is symmetric: a box with twice the GT area and a box with half
the GT area receive the same size score. This discourages both oversized and
undersized boxes without introducing a directional bias.

### Aspect-ratio reward

\[
R_{aspect}=\exp\left(-\frac{|\log((w_p/h_p)/(w_g/h_g))|}{\tau_{aspect}}\right).
\]

This separates shape drift from area drift. It is motivated by the same box
geometry factors used in Complete-IoU: overlap, center distance, and aspect ratio.

### Boundary reward

\[
R_{boundary}=\exp(-\max(0,d^p_{overflow}-d^g_{overflow})/\tau_{boundary}),
\]

where overflow is the sum of coordinate distances outside `[0, coordinate_scale]`.
The legacy `R_validity` already checks both coordinate ordering and whether the box
is inside this range. `R_boundary` is a softer, GT-aware refinement: a one-unit
*excess* overflow is less severe than a large excess overflow, while matching a GT
box that is partially out of view is not penalized.

### Excess-jump reward

\[
R_{jump}=\exp\left(-\frac{\max(0,\|\Delta c_p\|-\|\Delta c_g\|-m)}{\tau_{jump}}\right).
\]

It only penalizes predicted movement larger than the target's actual movement plus
a margin. Thus, it does not reward static boxes during fast motion.

## Important: the old temporal reward was already motion-aware

The original code already computes

\[
R_{temporal}=\exp\left(-\frac{\|\Delta c_p-\Delta c_g\|}{\tau_{temporal}}\right),
\]

not `-|pred_t - pred_{t-1}|`. Therefore, it already rewards matching GT motion and
penalizes a static prediction when the target moves. The new `R_jump` is a separate,
optional safeguard for extreme excess motion.

## Curriculum configuration

A curriculum is an optional partial override under `grpo.reward.curriculum`:

```yaml
reward:
  tracking_weights:
    iou: 0.55
    center: 0.30
    temporal: 0.05
    validity: 0.10

  final_weights:
    format: 0.05
    accuracy: 0.95
    semantic: 0.00

  curriculum:
    enabled: true
    schedule: linear       # linear, cosine, or step
    start_step: 400        # hold early weights before this step
    end_step: 1400         # hold late weights after this step

    early:
      tracking_weights:
        iou: 0.40
        center: 0.35
        temporal: 0.10
        validity: 0.15
      final_weights:
        format: 0.20
        accuracy: 0.80
        semantic: 0.00

    late:
      tracking_weights:
        iou: 0.45
        center: 0.25
        size: 0.10
        aspect: 0.05
        temporal: 0.10
        validity: 0.05
      final_weights:
        format: 0.10
        accuracy: 0.90
        semantic: 0.00
```

`early` and `late` are partial configs. Missing keys inherit the base reward config.
You may use `start_ratio` and `end_ratio` instead of explicit steps.

The trainer logs the active schedule and weights, for example:

- `reward/curriculum/progress`
- `reward/curriculum/phase` (`0=early`, `1=transition`, `2=late`)
- `reward/curriculum/weight_iou`
- `reward/curriculum/weight_size`
- `reward/curriculum/weight_aspect`
- all component rewards under `reward/iou`, `reward/size`, etc.

## Recommended controlled experiments

Do not enable every new term in the first run. Use the same SFT checkpoint, seed,
sampling settings, reference model, optimizer state policy, and 2,000-step budget.

1. **Static v2 control:** current v2 reward, no curriculum.
2. **Curriculum only:** `configs/sft_grpo_uav123_curriculum.yaml`.
3. **Curriculum + size/aspect:** `configs/sft_grpo_uav123_curriculum_geometry.yaml`.
4. **Boundary/jump ablation:** starting from experiment 3, allocate small weights by
   reducing IoU/center weights; do not simply add weights without checking the sum.

Suggested boundary/jump late weights for a later ablation:

```yaml
tracking_weights:
  iou: 0.42
  center: 0.23
  size: 0.10
  aspect: 0.05
  temporal: 0.08
  validity: 0.05
  boundary: 0.04
  jump: 0.03
```

## Full-state resume

All training entry points now accept an optional top-level field:

```yaml
resume_from_checkpoint: /path/to/output_dir/checkpoint-1000
```

This is passed to `Trainer.train(resume_from_checkpoint=...)`. Keep the GRPO
`reference_model_path` fixed to the original SFT checkpoint when resuming the same
GRPO run. A named model-only snapshot such as `..._1000` is for evaluation; use the
Hugging Face/DeepSpeed `output_dir/checkpoint-1000` directory for full-state resume.

## Current one-update GRPO clipping caveat

The current trainer generates samples and evaluates `old_logps` and `new_logps` from
the same policy before the optimizer update. Therefore the ratio is exactly one in
that loss call and `clip_epsilon` does not activate. This is consistent with a
one-update (`mu=1`) GRPO objective, but it means comparing v1-v5 through
`clip_epsilon` alone is not a meaningful ablation unless multiple optimizer updates
are made from the same rollout. The parameter is retained for config compatibility.

## Research basis

- Shao et al., *DeepSeekMath* (GRPO and group-relative advantages), 2024.
- Freitag et al., *Curriculum Reinforcement Learning for Complex Reward Functions*, 2024/2025.
- Zheng et al., *Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression*, 2019/2020.
- Müller et al., *TrackingNet*, ECCV 2018 (success/IoU, precision, normalized precision, and tracking attributes such as scale variation, aspect-ratio change, and fast motion).
- Gao et al., *Scaling Laws for Reward Model Overoptimization*, ICML 2023.
- Hugging Face TRL GRPO documentation (dynamic custom rewards can access trainer state; multiple reward functions can be weighted).

## Box-geometry diagnostic

After merging predictions, compare checkpoints directly:

```bash
python eval/box_geometry_diagnostics.py \
  outputs/sft2k_sft2k/infer_results/merged.jsonl \
  outputs/sft10k_sft10k/infer_results/merged.jsonl \
  --labels SFT2k+SFT2k SFT10k+SFT10k \
  --json_output outputs/geometry_comparison.json
```

The most useful columns for the professor's hypothesis are center error versus
`|log area|` and `|log AR|`. If center error remains stable while the latter two
increase, the longer model still localizes the target but predicts worse width,
height, or aspect ratio.

## Evaluation correctness warning

The original `eval/otb.py` converted 0–100 boxes to pixels and then applied
`.clamp(1, 100)` to x, y, width, and height. This corrupts geometry on frames larger
than 100 pixels. The corrected evaluator no longer applies that cap. Use
`--legacy_clamp_100` only to reproduce old numbers. All models in a comparison
must be re-evaluated with the same corrected evaluator.

`eval/eval.py` now writes fresh per-rank files and merges them, instead of letting
all ranks append to one file or mixing a rerun with old predictions.
