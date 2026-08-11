#!/usr/bin/env python3
"""Validate experiment invariants and expected reward changes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict

import yaml


CORE = (
    "full_curriculum",
    "wo_iou",
    "wo_center",
    "wo_area",
    "wo_aspect",
    "wo_temporal",
    "wo_validity",
    "wo_format",
    "fixed_early",
    "fixed_late",
    "fixed_mean",
)
CONSTANT_TOP_LEVEL = (
    "bf16",
    "max_steps",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "lr_scheduler_type",
    "gradient_checkpointing",
)
CONSTANT_GRPO = (
    "num_generations",
    "max_new_tokens",
    "min_new_tokens",
    "do_sample",
    "temperature",
    "top_p",
    "clip_epsilon",
    "beta_kl",
    "use_reference_model",
    "reference_model_path",
    "skip_zero_std_groups",
    "min_reward_std",
)


def load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), abs_tol=1e-9):
        raise AssertionError(f"{label}: got {actual}, expected {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs_dir",
        type=Path,
        nargs="?",
        default=Path("configs"),
    )
    args = parser.parse_args()

    configs = {
        name: load(args.configs_dir / f"uav123_reward_{name}.yaml")
        for name in CORE
    }
    full = configs["full_curriculum"]

    for name, cfg in configs.items():
        for key in CONSTANT_TOP_LEVEL:
            if cfg[key] != full[key]:
                raise AssertionError(f"{name}: changed top-level invariant {key}")
        for key in CONSTANT_GRPO:
            if cfg["grpo"][key] != full["grpo"][key]:
                raise AssertionError(f"{name}: changed GRPO invariant {key}")
        if cfg["model"]["pretrained_model_name_or_path"] != full["model"]["pretrained_model_name_or_path"]:
            raise AssertionError(f"{name}: changed SFT start checkpoint")

    outputs = [cfg["output_dir"] for cfg in configs.values()]
    evals = [cfg["eval_dir"] for cfg in configs.values()]
    if len(outputs) != len(set(outputs)) or len(evals) != len(set(evals)):
        raise AssertionError("Output or evaluation directories are not unique")

    component_mapping = {
        "wo_iou": "iou",
        "wo_center": "center",
        "wo_area": "size",
        "wo_aspect": "aspect",
        "wo_temporal": "temporal",
        "wo_validity": "validity",
    }
    for name, component in component_mapping.items():
        reward = configs[name]["grpo"]["reward"]
        for phase in ("early", "late"):
            assert_close(
                reward["curriculum"][phase]["tracking_weights"][component],
                0.0,
                f"{name}.{phase}.{component}",
            )

    wo_format = configs["wo_format"]["grpo"]["reward"]["curriculum"]
    for phase in ("early", "late"):
        assert_close(
            wo_format[phase]["final_weights"]["format"],
            0.0,
            f"wo_format.{phase}.format",
        )
        assert_close(
            wo_format[phase]["final_weights"]["accuracy"],
            1.0,
            f"wo_format.{phase}.accuracy",
        )

    fixed_mean = configs["fixed_mean"]["grpo"]["reward"]
    if fixed_mean["curriculum"].get("enabled", True):
        raise AssertionError("fixed_mean must disable curriculum")
    expected_mean = {
        "iou": 0.4275,
        "center": 0.2950,
        "size": 0.0550,
        "aspect": 0.0275,
        "temporal": 0.1000,
        "validity": 0.0950,
    }
    for key, value in expected_mean.items():
        assert_close(
            fixed_mean["tracking_weights"][key],
            value,
            f"fixed_mean.{key}",
        )
    assert_close(fixed_mean["final_weights"]["format"], 0.145, "fixed_mean.format")
    assert_close(fixed_mean["final_weights"]["accuracy"], 0.855, "fixed_mean.accuracy")

    for name, cfg in configs.items():
        training_sum = sum(
            float(value)
            for value in cfg["grpo"]["reward"]["tracking_weights"].values()
        )
        if training_sum <= 0.0:
            raise AssertionError(f"{name}: base tracking weights sum to zero")

    print(f"Validated {len(configs)} core configs.")


if __name__ == "__main__":
    main()
