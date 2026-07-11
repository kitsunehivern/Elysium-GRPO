import math
import unittest

from training.grpo_rewards import (
    RewardConfig,
    TrackingComponentWeights,
    aspect_reward,
    boundary_reward,
    jump_reward,
    resolve_curriculum_reward_config,
    reward_config_from_dict,
    size_reward,
    temporal_consistency_reward,
    trajectory_accuracy_reward,
)


class RewardComponentTests(unittest.TestCase):
    def test_temporal_is_motion_aware_not_static_smoothing(self):
        gt = [(0, 0, 10, 10), (30, 0, 40, 10)]
        follows_fast_motion = [(0, 0, 10, 10), (30, 0, 40, 10)]
        stays_static = [(0, 0, 10, 10), (0, 0, 10, 10)]
        self.assertAlmostEqual(temporal_consistency_reward(follows_fast_motion, gt, 20.0), 1.0)
        self.assertLess(temporal_consistency_reward(stays_static, gt, 20.0), 1.0)

    def test_size_reward_is_symmetric_for_large_and_small_boxes(self):
        gt = [(0, 0, 10, 10)]
        twice_area = [(0, 0, 20, 10)]
        half_area = [(0, 0, 5, 10)]
        self.assertAlmostEqual(size_reward(twice_area, gt, 0.7), size_reward(half_area, gt, 0.7), places=7)

    def test_aspect_reward_detects_shape_drift(self):
        gt = [(0, 0, 10, 10)]
        correct = [(0, 0, 20, 20)]
        wrong = [(0, 0, 20, 5)]
        self.assertGreater(aspect_reward(correct, gt, 0.5), aspect_reward(wrong, gt, 0.5))

    def test_boundary_reward_is_soft_and_gt_aware(self):
        gt_inside = [(0, 0, 10, 10)]
        inside = boundary_reward([(0, 0, 10, 10)], gt_inside, 100.0, 5.0)
        slightly_out = boundary_reward([(-1, 0, 10, 10)], gt_inside, 100.0, 5.0)
        far_out = boundary_reward([(-20, 0, 10, 10)], gt_inside, 100.0, 5.0)
        self.assertEqual(inside, 1.0)
        self.assertGreater(slightly_out, far_out)
        gt_out_of_view = [(-5, 0, 10, 10)]
        self.assertEqual(
            boundary_reward(gt_out_of_view, gt_out_of_view, 100.0, 5.0),
            1.0,
        )

    def test_jump_does_not_penalize_matching_fast_motion(self):
        gt = [(0, 0, 10, 10), (30, 0, 40, 10)]
        matching = [(0, 0, 10, 10), (30, 0, 40, 10)]
        excessive = [(0, 0, 10, 10), (70, 0, 80, 10)]
        self.assertEqual(jump_reward(matching, gt, 10.0, 2.0), 1.0)
        self.assertLess(jump_reward(excessive, gt, 10.0, 2.0), 1.0)


class BackwardCompatibilityTests(unittest.TestCase):
    def test_missing_new_weights_preserve_legacy_denominator(self):
        raw = {
            "tracking_weights": {"iou": 0.55, "center": 0.30, "temporal": 0.05, "validity": 0.10},
            "final_weights": {"format": 0.05, "accuracy": 0.95, "semantic": 0.0},
        }
        cfg = reward_config_from_dict(raw)
        self.assertEqual(cfg.tracking_weights.size, 0.0)
        self.assertEqual(cfg.tracking_weights.aspect, 0.0)
        self.assertEqual(cfg.tracking_weights.boundary, 0.0)
        self.assertEqual(cfg.tracking_weights.jump, 0.0)
        self.assertAlmostEqual(
            cfg.tracking_weights.iou
            + cfg.tracking_weights.center
            + cfg.tracking_weights.temporal
            + cfg.tracking_weights.validity,
            1.0,
        )

    def test_disabled_curriculum_returns_base_config(self):
        base = RewardConfig()
        active, metrics = resolve_curriculum_reward_config(base, {}, 500, 2000)
        self.assertIs(active, base)
        self.assertEqual(metrics["curriculum/enabled"], 0.0)

    def test_linear_curriculum_interpolates_weights(self):
        base = RewardConfig()
        curriculum = {
            "enabled": True,
            "schedule": "linear",
            "start_step": 100,
            "end_step": 300,
            "early": {
                "tracking_weights": {"iou": 0.4, "center": 0.6, "temporal": 0.0, "validity": 0.0},
                "final_weights": {"format": 0.2, "accuracy": 0.8, "semantic": 0.0},
            },
            "late": {
                "tracking_weights": {"iou": 0.8, "center": 0.2, "temporal": 0.0, "validity": 0.0},
                "final_weights": {"format": 0.1, "accuracy": 0.9, "semantic": 0.0},
            },
        }
        active, metrics = resolve_curriculum_reward_config(base, curriculum, 200, 1000)
        self.assertAlmostEqual(metrics["curriculum/progress"], 0.5)
        self.assertAlmostEqual(active.tracking_weights.iou, 0.6)
        self.assertAlmostEqual(active.tracking_weights.center, 0.4)
        self.assertAlmostEqual(active.final_weights.format, 0.15)

    def test_geometry_weights_change_track_reward_only_when_enabled(self):
        pred = [(5, 5, 25, 15)]  # same center as GT, wrong shape/size
        gt = [(10, 5, 20, 15)]
        legacy = RewardConfig(
            tracking_weights=TrackingComponentWeights(iou=0.0, center=1.0, temporal=0.0, validity=0.0)
        )
        geometry = RewardConfig(
            tracking_weights=TrackingComponentWeights(
                iou=0.0, center=0.5, temporal=0.0, validity=0.0, size=0.25, aspect=0.25
            )
        )
        legacy_score, _ = trajectory_accuracy_reward(pred, gt, legacy)
        geometry_score, _ = trajectory_accuracy_reward(pred, gt, geometry)
        self.assertAlmostEqual(legacy_score, 1.0)
        self.assertLess(geometry_score, legacy_score)


if __name__ == "__main__":
    unittest.main()
