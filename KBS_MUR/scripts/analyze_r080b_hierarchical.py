#!/usr/bin/env python3
"""Hierarchical target-level analysis for multi-target R-MUR runs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/derived/R080b_hierarchical_summary"
DATASETS = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]

SOURCES = {
    101: ROOT / "results/raw/R080b_multitarget_rmur_s1_20260920_2340",
    202: ROOT / "results/raw/R080b_multitarget_rmur_s202_diag_20260921",
    303: ROOT / "results/raw/R080b_multitarget_rmur_s303_diag_20260921",
}


def load_targets(dataset: str):
    records = []
    for seed, root in SOURCES.items():
        data_dir = root / dataset
        if not data_dir.exists():
            continue
        for target_dir in sorted(data_dir.glob("target*_seed*")):
            result_path = target_dir / "result.json"
            gains_path = target_dir / "episode_gains.npz"
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            record = {
                "seed": seed,
                "target": int(result.get("target_sensor", target_dir.name.split("_")[0].replace("target", ""))),
                "metrics": result["metrics"],
            }
            if gains_path.exists():
                arrays = np.load(gains_path)
                record["gains"] = {key: arrays[key] for key in arrays.files}
            records.append(record)
    return records


def target_level_bootstrap(values: np.ndarray, *, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = values.size
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = np.mean(values[idx])
    return float(np.mean(stats)), float(np.quantile(stats, .025)), float(np.quantile(stats, .975))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    rows = [
        "dataset,targets,seed_target_cells,G_RMUR_mean,G_static_mean,G_cached_mean,"
        "delta_static,delta_static_lo,delta_static_hi,delta_cached,delta_cached_lo,delta_cached_hi,"
        "positive_target_fraction,positive_gain_target_fraction,R_oracle_pooled"
    ]
    for dataset in DATASETS:
        records = load_targets(dataset)
        if not records:
            continue
        by_target = {}
        for rec in records:
            by_target.setdefault(rec["target"], []).append(rec)
        targets = sorted(by_target)
        delta_static_targets = []
        delta_cached_targets = []
        positive_gain_targets = []
        q4_values = []
        static_values = []
        cached_values = []
        og_values = []
        for target in targets:
            recs = by_target[target]
            q4 = np.asarray([r["metrics"]["ranked_response_q4"]["mean_prediction_gain"] for r in recs], dtype=float)
            st = np.asarray([r["metrics"]["static_utility"]["mean_prediction_gain"] for r in recs], dtype=float)
            ca = np.asarray([r["metrics"]["cached_mur"]["mean_prediction_gain"] for r in recs], dtype=float)
            og = np.asarray([r["metrics"]["oracle_greedy"]["mean_prediction_gain"] for r in recs], dtype=float)
            ds = q4 - st
            dc = q4 - ca
            delta_static_targets.append(float(np.mean(ds)))
            delta_cached_targets.append(float(np.mean(dc)))
            positive_gain_targets.append(bool(np.mean(q4) > 0.0))
            q4_values.extend(q4.tolist())
            static_values.extend(st.tolist())
            cached_values.extend(ca.tolist())
            og_values.extend(og.tolist())
        delta_static_targets = np.asarray(delta_static_targets, dtype=float)
        delta_cached_targets = np.asarray(delta_cached_targets, dtype=float)
        q4_values = np.asarray(q4_values, dtype=float)
        static_values = np.asarray(static_values, dtype=float)
        cached_values = np.asarray(cached_values, dtype=float)
        og_values = np.asarray(og_values, dtype=float)
        mean_static, lo_static, hi_static = target_level_bootstrap(delta_static_targets, seed=1)
        mean_cached, lo_cached, hi_cached = target_level_bootstrap(delta_cached_targets, seed=2)
        summary[dataset] = {
            "targets": int(len(targets)),
            "seed_target_cells": int(len(records)),
            "G_RMUR_mean": float(np.mean(q4_values)),
            "G_static_mean": float(np.mean(static_values)),
            "G_cached_mean": float(np.mean(cached_values)),
            "delta_static": float(np.mean(delta_static_targets)),
            "delta_static_ci": [lo_static, hi_static],
            "delta_cached": float(np.mean(delta_cached_targets)),
            "delta_cached_ci": [lo_cached, hi_cached],
            "positive_target_fraction": float(np.mean(delta_static_targets > 0.0)),
            "positive_gain_target_fraction": float(np.mean(positive_gain_targets)),
            "R_oracle_pooled": float(np.sum(q4_values) / np.sum(og_values)) if np.sum(og_values) > 0 else float("nan"),
        }
        rows.append(
            f"{dataset},{len(targets)},{len(records)},{np.mean(q4_values):.5f},{np.mean(static_values):.5f},"
            f"{np.mean(cached_values):.5f},{np.mean(delta_static_targets):.5f},{lo_static:.5f},{hi_static:.5f},"
            f"{np.mean(delta_cached_targets):.5f},{lo_cached:.5f},{hi_cached:.5f},"
            f"{np.mean(delta_static_targets > 0.0):.4f},{np.mean(positive_gain_targets):.4f},"
            f"{summary[dataset]['R_oracle_pooled']:.4f}"
        )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
