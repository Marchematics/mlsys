#!/usr/bin/env python3
"""Redundancy-adaptive selection, with the threshold chosen on held-out targets.

The pool-geometry probe shows that relevance ranking and pooling everything
trade places as the candidate pool becomes less redundant with the anchor. This
script turns that observation into a deployable rule and validates it honestly:
the threshold tau on the pool statistic (mean |correlation| between the anchor
and the candidates) is selected on one half of the targets and evaluated on the
other half, repeated over splits, so nothing is tuned on the reported cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def candidate_novelty(dataset: str, target: int, mode: str, candidates: int = 16) -> float:
    """Share of candidate variance not explained by the anchor series.

    Label-free and computed on the training split only: regress each candidate
    series on the anchor series and keep the residual variance share. Low values
    mean the candidates are redundant with the anchor.
    """

    import numpy as np

    from pami_traffic import build_candidate_pool

    data_root = ROOT.parent / "data" / "staeformer" / dataset
    if not data_root.exists():
        data_root = ROOT / "data" / "staeformer" / dataset
    values = np.load(data_root / "data.npz")["data"].astype("float32")[..., 0]
    index = np.load(data_root / "index.npz")
    stop = int(index["train"][:, 1].max() + 1)
    values = values[:stop]
    correlation = np.corrcoef(values.T)
    pool = build_candidate_pool(
        correlation[target], target_sensor=target, candidate_count=candidates, seed=20260920, mode=mode
    )
    anchor = values[:, target]
    design = np.stack([np.ones_like(anchor), anchor], axis=1)
    beta, *_ = np.linalg.lstsq(design, values[:, pool], rcond=None)
    residual = values[:, pool] - design @ beta
    return float(np.mean(residual.var(axis=0) / values[:, pool].var(axis=0)))


def load(root: Path) -> dict:
    cells: dict[str, dict[tuple[int, int], dict]] = {}
    for cell in root.glob("*_target*_seed*"):
        record = json.loads((cell / "result.json").read_text(encoding="utf-8"))
        record.setdefault("dataset", "PEMSBAY" if "PEMSBAY" in root.name.upper() else "METRLA")
        key = (int(record["target_sensor"]), int(record["seed"]))
        cells.setdefault(record["mode"], {})[key] = record
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates", default="relevance,pool_all,mutual_information,dpp,mmr")
    parser.add_argument("--budget-column", default="mean_prediction_gain")
    parser.add_argument("--statistic", default="anchor_similarity", choices=["anchor_similarity", "novelty"])
    args = parser.parse_args()

    cells: dict[str, dict[tuple[int, int], dict]] = {}
    for root in args.run_root:
        for mode, entries in load(root).items():
            cells.setdefault(mode, {}).update(entries)
    modes = sorted(cells)
    keys = sorted(set.intersection(*[set(cells[m]) for m in modes]))
    policies = [p for p in args.candidates.split(",") if p]

    novelty_cache: dict = {}

    def value(mode, key, policy):
        metrics = cells[mode][key]["metrics"]
        return float(metrics[policy][args.budget_column]) if policy in metrics else float("nan")

    def anchor(mode, key):
        dataset = cells[mode][key].get("dataset", "METRLA")
        if args.statistic == "novelty":
            cache_key = (dataset, key[0], mode)
            if cache_key not in novelty_cache:
                novelty_cache[cache_key] = candidate_novelty(dataset, key[0], mode)
            return novelty_cache[cache_key]
        return float(cells[mode][key]["pool_mean_anchor_similarity"])

    report = {
        "statistic": args.statistic,
        "modes": modes,
        "cells": len(keys),
        "per_mode": {
            mode: {
                "mean_statistic": float(np.mean([anchor(mode, k) for k in keys])),
                "policy_means": {
                    policy: float(np.mean([value(mode, k, policy) for k in keys])) for policy in policies
                },
            }
            for mode in modes
        },
    }

    targets = sorted({k[0] for k in keys})
    thresholds = (
        [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        if args.statistic == "novelty"
        else [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    )
    rng = np.random.default_rng(0)
    splits = 200
    adaptive_scores, chosen = [], []
    for _ in range(splits):
        shuffled = rng.permutation(targets)
        half = len(shuffled) // 2
        calibration = set(shuffled[:half].tolist())
        evaluation = set(shuffled[half:].tolist())
        best_tau, best_score = None, -np.inf
        for tau in thresholds:
            score = 0.0
            count = 0
            for mode in modes:
                for key in keys:
                    if key[0] not in calibration:
                        continue
                    policy = "pool_all" if anchor(mode, key) > tau else "relevance"
                    score += value(mode, key, policy)
                    count += 1
            if count and score / count > best_score:
                best_tau, best_score = tau, score / count
        chosen.append(best_tau)
        total, count = 0.0, 0
        for mode in modes:
            for key in keys:
                if key[0] not in evaluation:
                    continue
                policy = "pool_all" if anchor(mode, key) > best_tau else "relevance"
                total += value(mode, key, policy)
                count += 1
        adaptive_scores.append(total / max(count, 1))
    adaptive_scores = np.asarray(adaptive_scores)
    report["adaptive"] = {
        "splits": splits,
        "thresholds": thresholds,
        "chosen_threshold_median": float(np.median(chosen)),
        "chosen_threshold_counts": {str(t): int(chosen.count(t)) for t in thresholds if chosen.count(t)},
        "mean_held_out_gain": float(adaptive_scores.mean()),
        "ci": [
            float(np.quantile(adaptive_scores, 0.025)),
            float(np.quantile(adaptive_scores, 0.975)),
        ],
        "fixed_relevance": float(np.mean([value(m, k, "relevance") for m in modes for k in keys])),
        "fixed_pool_all": float(np.mean([value(m, k, "pool_all") for m in modes for k in keys])),
        "oracle_pick": float(np.mean([
            max(value(m, k, p) for p in policies) for m in modes for k in keys
        ])),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "adaptive_rule.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["adaptive"], indent=2))
    for mode in modes:
        entry = report["per_mode"][mode]
        print(f"{mode:16s} statistic {entry['mean_statistic']:.3f}  " + "  ".join(
            f"{p} {entry['policy_means'][p]:+.5f}" for p in policies
        ))


if __name__ == "__main__":
    main()
