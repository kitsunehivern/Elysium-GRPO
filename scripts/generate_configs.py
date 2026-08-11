#!/usr/bin/env python3
"""Generate the UAV123 reward-ablation and curriculum-control YAML files."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


PROJECT_ROOT_DEFAULT = Path(
    "/home/stackops/dhviet/ElysiumGRPO/Elysium-main"
)
DATASET_ROOT_DEFAULT = Path("/home/stackops/dhviet/UAV123_Elysium")

TRACKING_KEYS = (
    "iou",
    "center",
    "size",
    "aspect",
    "temporal",
    "validity",
    "boundary",
    "jump",
)

EARLY_TRACKING = {
    "iou": 0.40,
    "center": 0.35,
    "size": 0.00,
    "aspect": 0.00,
    "temporal": 0.10,
    "validity": 0.15,
    "boundary": 0.00,
    "jump": 0.00,
}
LATE_TRACKING = {
    "iou": 0.45,
    "center": 0.25,
    "size": 0.10,
    "aspect": 0.05,
    "temporal": 0.10,
    "validity": 0.05,
    "boundary": 0.00,
    "jump": 0.00,
}
MEAN_TRACKING = {
    "iou": 0.4275,
    "center": 0.2950,
    "size": 0.0550,
    "aspect": 0.0275,
    "temporal": 0.1000,
    "validity": 0.0950,
    "boundary": 0.0000,
    "jump": 0.0000,
}

EARLY_FINAL = {"format": 0.20, "accuracy": 0.80, "semantic": 0.00}
LATE_FINAL = {"format": 0.10, "accuracy": 0.90, "semantic": 0.00}
MEAN_FINAL = {"format": 0.1450, "accuracy": 0.8550, "semantic": 0.0000}


def complete_tracking(values: Dict[str, float]) -> Dict[str, float]:
    return {key: float(values.get(key, 0.0)) for key in TRACKING_KEYS}


def curriculum_block(
    early_tracking: Dict[str, float] = EARLY_TRACKING,
    late_tracking: Dict[str, float] = LATE_TRACKING,
    early_final: Dict[str, float] = EARLY_FINAL,
    late_final: Dict[str, float] = LATE_FINAL,
    start_step: int = 400,
    end_step: int = 1400,
) -> Dict[str, Any]:
    return {
        "enabled": True,
        "schedule": "linear",
        "start_step": start_step,
        "end_step": end_step,
        "early": {
            "tracking_weights": complete_tracking(early_tracking),
            "final_weights": deepcopy(early_final),
        },
        "late": {
            "tracking_weights": complete_tracking(late_tracking),
            "final_weights": deepcopy(late_final),
        },
    }


def make_base_config(
    project_root: Path,
    dataset_root: Path,
    variant: str,
    seed: int = 42,
) -> Dict[str, Any]:
    output_dir = (
        project_root
        / "checkpoints"
        / "paper_ablation_uav123"
        / variant
    )
    eval_dir = (
        project_root
        / "outputs"
        / "paper_ablation_uav123"
        / variant
    )
    sft_checkpoint = project_root / "checkpoints" / "sft2k_uav123"

    reward = {
        "format_style": "answer_only",
        "require_frame_prefix": False,
        "coordinate_scale": 100.0,
        "center_tau": 10.0,
        "temporal_tau": 20.0,
        "size_tau": 0.70,
        "aspect_tau": 0.50,
        "boundary_tau": 5.0,
        "jump_tau": 10.0,
        "jump_margin": 2.0,
        "count_mismatch_penalty": 0.50,
        "clamp_for_metrics": True,
        "semantic_gate": 0.05,
        # These are fallbacks only while curriculum.enabled=true.
        "tracking_weights": complete_tracking(
            {
                "iou": 0.55,
                "center": 0.30,
                "temporal": 0.05,
                "validity": 0.10,
            }
        ),
        "final_weights": {
            "format": 0.05,
            "accuracy": 0.95,
            "semantic": 0.00,
        },
        "curriculum": curriculum_block(),
    }

    return {
        "bf16": True,
        "seed": seed,
        "num_train_epochs": 1,
        "max_steps": 2000,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "evaluation_strategy": "no",
        "save_strategy": "steps",
        "save_steps": 1000,
        "save_total_limit": 2,
        "learning_rate": 2.0e-6,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "logging_steps": 1,
        "tf32": True,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "report_to": "wandb",
        "run_name": f"paper-uav123-{variant}-seed{seed}",
        "output_dir": str(output_dir),
        "eval_dir": str(eval_dir),
        "deepspeed": "./configs/zero2.json",
        "model": {
            "pretrained_model_name_or_path": str(sft_checkpoint),
            "trained_model_name_or_path": str(output_dir),
        },
        "grpo": {
            "num_generations": 8,
            "max_new_tokens": 256,
            "min_new_tokens": 4,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "clip_epsilon": 0.2,
            "beta_kl": 0.005,
            "use_reference_model": True,
            "reference_model_path": str(sft_checkpoint),
            "skip_zero_std_groups": True,
            "min_reward_std": 0.02,
            "reference_torch_dtype": "bf16",
            "freeze_visual_encoder": True,
            "freeze_llm": False,
            "freeze_adapter": False,
            "freeze_projector": False,
            "prompt_suffix": "\nReturn only frame-wise boxes.",
            "reward": reward,
            "semantic_reward": {
                "enabled": False,
                "model_name_or_path": "siglip-so400m-patch14-384",
                "torch_dtype": "bf16",
                "scale": 2.0,
                "max_frames": 4,
                "text_span_words": 96,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
            },
        },
        "data": {
            "train": {
                "data_fetch": {
                    "data_paths": [
                        {
                            "anno_path": str(
                                dataset_root / "train" / "annotation.jsonl"
                            ),
                            "image_folder": str(dataset_root / "frames"),
                        }
                    ],
                    "batch_sizes": [1],
                    "num_workers": 0,
                    "num_readers": [2],
                    "key_mapping": None,
                    "multi_round_qa": True,
                },
                "data_preprocess": {
                    "trust_remote_code": True,
                    "with_visual": True,
                    "frames_key": "frames",
                    "label_key": "vqa",
                    "task_type": "vqa",
                    "tokenizer": "lmsys/vicuna-7b-v1.5",
                    "max_seq_len": 512,
                    "max_prompt_len": 256,
                    "vqa_processor_params": {"box_format": "ours_v1"},
                    "online_vqa_processor_params": {
                        "task": "SOT",
                        "box_format": "ours_v1",
                        "fix_prompt": True,
                    },
                    "timestamp_params": {
                        "frame_prefix_pattern": "Frame {i}: ",
                        "offset": 1,
                        "remove_single_frame_timestamp": True,
                        "sep": "; ",
                        "remove_last_sep": False,
                        "end_symbol": "\n",
                    },
                    "sample_method": "random_clip",
                    "clip_frames": [4, 8],
                    "clip_interval": [1, 8],
                    "extra_sample_keys": ["box"],
                    "num_segments": 1,
                    "verbose": True,
                    "training": True,
                    "frames_ops": {
                        "Resize": {"size": [336, 336]},
                        "ToTensor": {},
                        "Normalize": {
                            "mean": [0.48145466, 0.4578275, 0.40821073],
                            "std": [0.26862954, 0.26130258, 0.27577711],
                        },
                    },
                },
            },
            "predict": {
                "data_fetch": {
                    "anno_path": str(
                        dataset_root / "test" / "annotation.jsonl"
                    ),
                    "image_folder": str(dataset_root / "frames"),
                    "batch_sizes": [1],
                    "num_workers": 1,
                    "num_readers": [2],
                    "key_mapping": None,
                },
                "data_preprocess": {
                    "with_visual": True,
                    "frames_key": "frames",
                    "sample_method": "random_clip",
                    "label_key": "vqa",
                    "task_type": "vqa",
                    "tokenizer": "lmsys/vicuna-7b-v1.5",
                    "max_seq_len": 512,
                    "max_prompt_len": 256,
                    "vqa_processor_params": {"box_format": "ours_v1"},
                    "online_vqa_processor_params": {"task": "SOT"},
                    "num_segments": 1,
                    "verbose": True,
                    "training": False,
                    "frames_ops": {
                        "Resize": {"size": [336, 336]},
                        "ToTensor": {},
                        "Normalize": {
                            "mean": [0.48145466, 0.4578275, 0.40821073],
                            "std": [0.26862954, 0.26130258, 0.27577711],
                        },
                    },
                },
            },
        },
    }


def set_identity(
    config: Dict[str, Any],
    project_root: Path,
    variant: str,
    seed: int = 42,
) -> None:
    output_dir = (
        project_root
        / "checkpoints"
        / "paper_ablation_uav123"
        / variant
    )
    eval_dir = (
        project_root
        / "outputs"
        / "paper_ablation_uav123"
        / variant
    )
    config["seed"] = seed
    config["run_name"] = f"paper-uav123-{variant}-seed{seed}"
    config["output_dir"] = str(output_dir)
    config["eval_dir"] = str(eval_dir)
    config["model"]["trained_model_name_or_path"] = str(output_dir)


def zero_component(config: Dict[str, Any], component: str) -> None:
    reward = config["grpo"]["reward"]
    reward["tracking_weights"][component] = 0.0
    reward["curriculum"]["early"]["tracking_weights"][component] = 0.0
    reward["curriculum"]["late"]["tracking_weights"][component] = 0.0


def set_fixed(
    config: Dict[str, Any],
    tracking: Dict[str, float],
    final: Dict[str, float],
) -> None:
    reward = config["grpo"]["reward"]
    reward["tracking_weights"] = complete_tracking(tracking)
    reward["final_weights"] = deepcopy(final)
    reward["curriculum"] = {"enabled": False}


def build_variants(
    project_root: Path,
    dataset_root: Path,
) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    variants: List[Tuple[str, str, str, Dict[str, Any]]] = []

    def fresh(name: str) -> Dict[str, Any]:
        return make_base_config(project_root, dataset_root, name)

    full = fresh("full_curriculum")
    variants.append(
        (
            "core",
            "full_curriculum",
            "Full v6 geometry-aware linear curriculum",
            full,
        )
    )

    component_labels = {
        "iou": "Remove IoU reward",
        "center": "Remove center-localization reward",
        "size": "Remove area/scale reward (YAML field: size)",
        "aspect": "Remove aspect-ratio reward",
        "temporal": "Remove motion-vector temporal reward",
        "validity": "Remove validity reward",
    }
    slug_for_component = {"size": "area"}
    for component, description in component_labels.items():
        slug = f"wo_{slug_for_component.get(component, component)}"
        cfg = fresh(slug)
        zero_component(cfg, component)
        variants.append(("core", slug, description, cfg))

    wo_format = fresh("wo_format")
    reward = wo_format["grpo"]["reward"]
    reward["final_weights"] = {
        "format": 0.0,
        "accuracy": 1.0,
        "semantic": 0.0,
    }
    for phase in ("early", "late"):
        reward["curriculum"][phase]["final_weights"] = {
            "format": 0.0,
            "accuracy": 1.0,
            "semantic": 0.0,
        }
    variants.append(("core", "wo_format", "Remove format reward", wo_format))

    fixed_early = fresh("fixed_early")
    set_fixed(fixed_early, EARLY_TRACKING, EARLY_FINAL)
    variants.append(
        (
            "core",
            "fixed_early",
            "Use the curriculum early reward for all 2,000 steps",
            fixed_early,
        )
    )

    fixed_late = fresh("fixed_late")
    set_fixed(fixed_late, LATE_TRACKING, LATE_FINAL)
    variants.append(
        (
            "core",
            "fixed_late",
            "Use the curriculum late reward for all 2,000 steps",
            fixed_late,
        )
    )

    fixed_mean = fresh("fixed_mean")
    set_fixed(fixed_mean, MEAN_TRACKING, MEAN_FINAL)
    variants.append(
        (
            "core",
            "fixed_mean",
            "Time-mean exposure-matched fixed reward",
            fixed_mean,
        )
    )

    reverse = fresh("reverse_curriculum")
    reverse_reward = reverse["grpo"]["reward"]
    reverse_reward["tracking_weights"] = complete_tracking(MEAN_TRACKING)
    reverse_reward["final_weights"] = deepcopy(MEAN_FINAL)
    reverse_reward["curriculum"] = curriculum_block(
        early_tracking=LATE_TRACKING,
        late_tracking=EARLY_TRACKING,
        early_final=LATE_FINAL,
        late_final=EARLY_FINAL,
        start_step=600,
        end_step=1600,
    )
    variants.append(
        (
            "optional_curriculum",
            "reverse_curriculum",
            "Late-to-early order with matched 55/45 regime exposure",
            reverse,
        )
    )

    tracking_only = fresh("curriculum_tracking_only")
    tracking_reward = tracking_only["grpo"]["reward"]
    tracking_reward["tracking_weights"] = complete_tracking(MEAN_TRACKING)
    tracking_reward["final_weights"] = deepcopy(MEAN_FINAL)
    tracking_reward["curriculum"] = curriculum_block(
        early_tracking=EARLY_TRACKING,
        late_tracking=LATE_TRACKING,
        early_final=MEAN_FINAL,
        late_final=MEAN_FINAL,
    )
    variants.append(
        (
            "optional_curriculum",
            "curriculum_tracking_only",
            "Curriculum only inside R_track; fixed format/tracking mixture",
            tracking_only,
        )
    )

    format_only = fresh("curriculum_format_only")
    format_reward = format_only["grpo"]["reward"]
    format_reward["tracking_weights"] = complete_tracking(MEAN_TRACKING)
    format_reward["final_weights"] = deepcopy(MEAN_FINAL)
    format_reward["curriculum"] = curriculum_block(
        early_tracking=MEAN_TRACKING,
        late_tracking=MEAN_TRACKING,
        early_final=EARLY_FINAL,
        late_final=LATE_FINAL,
    )
    variants.append(
        (
            "optional_curriculum",
            "curriculum_format_only",
            "Curriculum only in format/tracking mixture; fixed R_track",
            format_only,
        )
    )

    wo_geometry = fresh("wo_geometry")
    zero_component(wo_geometry, "size")
    zero_component(wo_geometry, "aspect")
    variants.append(
        (
            "optional_grouped",
            "wo_geometry",
            "Grouped removal of area and aspect rewards",
            wo_geometry,
        )
    )

    wo_trajectory = fresh("wo_trajectory")
    zero_component(wo_trajectory, "temporal")
    zero_component(wo_trajectory, "validity")
    variants.append(
        (
            "optional_grouped",
            "wo_trajectory",
            "Grouped removal of temporal and validity rewards",
            wo_trajectory,
        )
    )

    iou_format = fresh("iou_format_only")
    iou_only = complete_tracking({"iou": 1.0})
    iou_reward = iou_format["grpo"]["reward"]
    iou_reward["tracking_weights"] = deepcopy(iou_only)
    iou_reward["curriculum"] = curriculum_block(
        early_tracking=iou_only,
        late_tracking=iou_only,
        early_final=EARLY_FINAL,
        late_final=LATE_FINAL,
    )
    variants.append(
        (
            "optional_grouped",
            "iou_format_only",
            "Simple IoU plus format reward baseline",
            iou_format,
        )
    )
    return variants


def dump_yaml(
    path: Path,
    config: Dict[str, Any],
    family: str,
    description: str,
) -> None:
    header = (
        "# Generated by scripts/paper_ablation/generate_configs.py\n"
        f"# Experiment family: {family}\n"
        f"# Purpose: {description}\n"
        "# IMPORTANT: the YAML field `size` is the paper's area/scale reward.\n"
    )
    body = yaml.safe_dump(
        config,
        sort_keys=False,
        default_flow_style=False,
        width=1000,
    )
    path.write_text(header + body, encoding="utf-8")


def write_manifest(
    path: Path,
    rows: Iterable[Tuple[str, str, str, str, int]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["family", "variant", "config_file", "description", "seed"]
        )
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
        help="Elysium-GRPO repository root used in generated absolute paths",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT_DEFAULT,
        help="UAV123_Elysium root containing train/, test/, and frames/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination for YAML files; defaults to configs",
    )
    parser.add_argument(
        "--confirmation-seeds",
        type=int,
        nargs="*",
        default=[123, 2026],
        help="Extra GRPO seeds for full_curriculum and fixed_mean",
    )
    args = parser.parse_args()

    project_root = args.project_root.expanduser()
    dataset_root = args.dataset_root.expanduser()
    bundle_root = Path(__file__).resolve().parents[2]
    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir
        else bundle_root / "configs" / "paper_ablation_uav123"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = build_variants(project_root, dataset_root)
    manifest_rows: List[Tuple[str, str, str, str, int]] = []
    by_name: Dict[str, Tuple[str, str, Dict[str, Any]]] = {}
    for family, variant, description, config in variants:
        filename = f"uav123_reward_{variant}.yaml"
        dump_yaml(output_dir / filename, config, family, description)
        manifest_rows.append(
            (family, variant, filename, description, int(config["seed"]))
        )
        by_name[variant] = (family, description, config)

    for seed in args.confirmation_seeds:
        if seed == 42:
            continue
        for source_variant in ("full_curriculum", "fixed_mean"):
            family, description, source_config = by_name[source_variant]
            seeded = deepcopy(source_config)
            seeded_variant = f"{source_variant}_seed{seed}"
            set_identity(seeded, project_root, seeded_variant, seed=seed)
            filename = f"uav123_reward_{seeded_variant}.yaml"
            dump_yaml(
                output_dir / filename,
                seeded,
                "confirmation",
                f"{description}; confirmation seed {seed}",
            )
            manifest_rows.append(
                (
                    "confirmation",
                    seeded_variant,
                    filename,
                    f"{description}; confirmation seed {seed}",
                    seed,
                )
            )

    write_manifest(output_dir / "experiment_manifest.csv", manifest_rows)
    print(f"Wrote {len(manifest_rows)} configs to {output_dir}")


if __name__ == "__main__":
    main()
