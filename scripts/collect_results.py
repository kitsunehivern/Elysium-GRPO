#!/usr/bin/env python3
"""Collect corrected OTB and geometry diagnostics into one CSV table."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Optional

import yaml


METRIC_PATTERNS = {
    "auc": re.compile(r"^auc:\s+.*?([-+]?\d+(?:\.\d+)?)"),
    "precision": re.compile(
        r"^prec_score:\s+.*?([-+]?\d+(?:\.\d+)?)"
    ),
    "normalized_precision": re.compile(
        r"^norm_prec_score:\s+.*?([-+]?\d+(?:\.\d+)?)"
    ),
}


def parse_scalar_metrics(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8").splitlines():
        for key, pattern in METRIC_PATTERNS.items():
            match = pattern.search(line.strip())
            if match:
                metrics[key] = float(match.group(1))
    return metrics


def parse_geometry(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload:
        return {}
    first = next(iter(payload.values()))
    return {str(key): float(value) for key, value in first.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_ablation_uav123_results.csv"),
    )
    args = parser.parse_args()

    rows = []
    all_keys = {"variant", "config", "seed", "status"}
    for config_path in sorted(args.configs_dir.glob("*.yaml")):
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        eval_dir = Path(config["eval_dir"])
        scalar = parse_scalar_metrics(eval_dir / "metrics_corrected.txt")
        geometry = parse_geometry(eval_dir / "geometry_diagnostics.json")
        row: Dict[str, object] = {
            "variant": config_path.stem.removeprefix("uav123_reward_"),
            "config": str(config_path),
            "seed": int(config["seed"]),
            "status": "complete" if scalar else "missing",
            **scalar,
            **geometry,
        }
        rows.append(row)
        all_keys.update(row)

    preferred = [
        "variant",
        "seed",
        "status",
        "auc",
        "precision",
        "normalized_precision",
        "mean_iou",
        "center_error_gt_diag_mean",
        "abs_log_area_ratio_mean",
        "abs_log_aspect_ratio_mean",
        "motion_vector_error_mean",
        "valid_pred_rate",
        "out_of_boundary_rate",
        "config",
    ]
    fields = preferred + sorted(all_keys.difference(preferred))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
