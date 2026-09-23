#!/usr/bin/env python3
"""Post-hoc margin and normalized-error audit for the R071 candidate scaling run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--family", default="mur_rollout")
    return parser.parse_args()


def episode_margin_and_error(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray):
    """Per-episode top-two margin, max absolute error, mean error, flip and regret."""

    episodes = truth.shape[0]
    margins = np.full(episodes, np.nan, dtype=float)
    max_abs = np.full(episodes, np.nan, dtype=float)
    mean_abs = np.full(episodes, np.nan, dtype=float)
    flips = np.zeros(episodes, dtype=bool)
    strict_flips = np.zeros(episodes, dtype=bool)
    regrets = np.full(episodes, np.nan, dtype=float)
    for row in range(episodes):
        eligible = ~mask[row]
        if eligible.sum() < 2:
            continue
        true_values = truth[row, eligible]
        pred_values = pred[row, eligible]
        order = np.sort(true_values)[::-1]
        margins[row] = order[0] - order[1]
        abs_err = np.abs(pred_values - true_values)
        max_abs[row] = float(np.max(abs_err))
        mean_abs[row] = float(np.mean(abs_err))
        true_choice = int(np.argmax(true_values))
        pred_choice = int(np.argmax(pred_values))
        regrets[row] = float(true_values[true_choice] - true_values[pred_choice])
        flips[row] = regrets[row] > 1e-12
        strict_flips[row] = pred_choice != true_choice
    return margins, max_abs, mean_abs, flips, strict_flips, regrets


def summarize_arrays(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    # pred/truth/mask: (states, episodes, candidates)
    state_rows = []
    pooled_abs_error = []
    pooled_truth = []
    for state in range(pred.shape[0]):
        eligible = ~mask[state]
        # Candidate-level pooled error and utility scale.
        err = pred[state][eligible] - truth[state][eligible]
        pooled_abs_error.append(np.abs(err))
        pooled_truth.append(truth[state][eligible])
        margin, max_abs, mean_abs, flips, strict_flips, regrets = episode_margin_and_error(
            pred[state], truth[state], mask[state]
        )
        valid = np.isfinite(margin)
        if not np.any(valid):
            continue
        state_rows.append(
            {
                "margin": margin[valid],
                "max_abs": max_abs[valid],
                "mean_abs": mean_abs[valid],
                "flip": flips[valid],
                "strict_flip": strict_flips[valid],
                "regret": regrets[valid],
            }
        )

    abs_error = np.concatenate(pooled_abs_error)
    truth_values = np.concatenate(pooled_truth)
    utility_mae = float(np.mean(abs_error))
    target_sd = float(np.std(truth_values))
    target_absmean = float(np.mean(np.abs(truth_values)))
    nmae_sd = utility_mae / target_sd if target_sd > 0 else float("nan")
    nmae_absmean = utility_mae / target_absmean if target_absmean > 0 else float("nan")

    margins = np.concatenate([row["margin"] for row in state_rows])
    max_abs = np.concatenate([row["max_abs"] for row in state_rows])
    mean_abs = np.concatenate([row["mean_abs"] for row in state_rows])
    flips = np.concatenate([row["flip"] for row in state_rows]).astype(float)
    strict_flips = np.concatenate([row["strict_flip"] for row in state_rows]).astype(float)
    regrets = np.concatenate([row["regret"] for row in state_rows])

    positive = margins > 1e-12
    ratio_max = max_abs[positive] / margins[positive]
    ratio_mean = mean_abs[positive] / margins[positive]
    bound_all = max_abs > margins / 2.0
    bound_positive = max_abs[positive] > margins[positive] / 2.0
    return {
        "utility_mae": utility_mae,
        "target_sd": target_sd,
        "target_absmean": target_absmean,
        "nmae_sd": nmae_sd,
        "nmae_absmean": nmae_absmean,
        "mean_top2_margin": float(np.mean(margins)),
        "median_top2_margin": float(np.median(margins)),
        "mean_positive_top2_margin": float(np.mean(margins[positive])) if np.any(positive) else float("nan"),
        "zero_margin_rate": float(1.0 - np.mean(positive)),
        "mean_max_abs_error": float(np.mean(max_abs)),
        "mean_ratio_max_error_margin_positive": float(np.mean(ratio_max)) if ratio_max.size else float("nan"),
        "median_ratio_max_error_margin_positive": float(np.median(ratio_max)) if ratio_max.size else float("nan"),
        "mean_ratio_mean_error_margin_positive": float(np.mean(ratio_mean)) if ratio_mean.size else float("nan"),
        "bound_rate_max_error_gt_half_margin": float(np.mean(bound_all)),
        "bound_rate_positive_margin": float(np.mean(bound_positive)) if np.any(positive) else float("nan"),
        "decision_error_rate": float(np.mean(flips)),
        "decision_accuracy": float(1.0 - np.mean(flips)),
        "strict_top1_accuracy": float(1.0 - np.mean(strict_flips)),
        "mean_regret": float(np.mean(regrets)),
        "episodes": int(margins.size),
        "states": int(pred.shape[0]),
    }


def main() -> None:
    args = parse_args()
    records = []
    for k_dir in sorted(args.raw_root.glob("k*"), key=lambda p: int(p.name[1:])):
        k = int(k_dir.name[1:])
        for npz_path in sorted(k_dir.glob("*/utility_arrays.npz")):
            run_dir = npz_path.parent
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            seed = int(result["config"]["seed"])
            data = np.load(npz_path)
            family = args.family
            pred = data[f"{family}_prediction"]
            truth = data[f"{family}_truth"]
            mask = data[f"{family}_mask"]
            row = {
                "candidate_count": k,
                "seed": seed,
                **summarize_arrays(pred, truth, mask),
            }
            records.append(row)

    if not records:
        raise SystemExit(f"no utility_arrays.npz found under {args.raw_root}")

    fieldnames = list(records[0].keys())
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    metric_names = [name for name in fieldnames if name not in {"candidate_count", "seed"}]
    aggregate = {}
    for k in sorted({int(row["candidate_count"]) for row in records}):
        subset = [row for row in records if int(row["candidate_count"]) == k]
        aggregate[str(k)] = {
            metric: {
                "mean": float(np.mean([float(row[metric]) for row in subset])),
                "std": float(np.std([float(row[metric]) for row in subset], ddof=1)),
            }
            for metric in metric_names
        }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    header = ["candidate_count", "seed"] + [name for name in metric_names]
    print("\t".join(header))
    for row in records:
        print("\t".join(str(row[name]) for name in header))


if __name__ == "__main__":
    main()
