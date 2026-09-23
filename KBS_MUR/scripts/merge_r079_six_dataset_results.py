#!/usr/bin/env python3
"""Merge the completed six-dataset R-MUR runs into one summary."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "METRLA": ROOT / "results/raw/R079_response_router_groupA_s5_20260920_2030/METRLA",
    "PEMSBAY": ROOT / "results/raw/R079_PEMSBAY_s5_20260920_2155/PEMSBAY",
    "PEMS03": ROOT / "results/raw/R079_response_router_groupB_s5_20260920_2030/PEMS03",
    "PEMS04": ROOT / "results/raw/R079_PEMS04_s5_20260920_2155/PEMS04",
    "PEMS07": ROOT / "results/raw/R079_response_router_groupC_s5_20260920_2030/PEMS07",
    "PEMS08": ROOT / "results/raw/R079_PEMS08_s5_20260920_2155/PEMS08",
}
OUT_DIR = ROOT / "results/derived/R079_six_dataset_summary"


def t_interval(values):
    from scipy.stats import t as student_t

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    crit = float(student_t.ppf(0.975, df=n - 1))
    return mean - crit * se, mean + crit * se


def main() -> None:
    summary = {}
    rows = ["dataset,oracle_static,oracle_greedy,H_state,static_utility,cached_mur,ranked_rmur_q2,ranked_rmur_q4,ranked_rmur_q8,ranked_rmur_full,R_oracle_q4,R_deploy_q4"]
    for dataset, source in SOURCES.items():
        results = [json.loads(path.read_text()) for path in sorted(source.glob("seed*/result.json"))]
        if len(results) != 5:
            raise RuntimeError(f"{dataset}: expected 5 results, found {len(results)}")
        policies = list(results[0]["metrics"].keys())
        agg = {}
        for policy in policies:
            agg[policy] = {}
            for metric in results[0]["metrics"][policy]:
                values = np.asarray([r["metrics"][policy][metric] for r in results], dtype=float)
                agg[policy][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
        g_os = agg["oracle_static"]["mean_prediction_gain"]["mean"]
        g_og = agg["oracle_greedy"]["mean_prediction_gain"]["mean"]
        g_static = agg["static_utility"]["mean_prediction_gain"]["mean"]
        g_q4 = agg["ranked_response_q4"]["mean_prediction_gain"]["mean"]
        headroom = (g_og - g_os) / g_og if g_og > 0 else float("nan")
        r_oracle = g_q4 / g_og if g_og > 0 else float("nan")
        r_deploy = (g_q4 - g_static) / (g_og - g_static) if g_og > g_static else float("nan")
        q4_seed_values = np.asarray([r["metrics"]["ranked_response_q4"]["mean_prediction_gain"] for r in results], dtype=float)
        q4_ci = t_interval(q4_seed_values)
        summary[dataset] = {
            "seek_count": len(results),
            "quantities": {
                "oracle_static_gain": g_os,
                "oracle_greedy_gain": g_og,
                "state_conditioning_headroom": headroom,
                "R_oracle_q4": r_oracle,
                "R_deploy_q4": r_deploy,
                "ranked_q4_seed_ci_low": q4_ci[0],
                "ranked_q4_seed_ci_high": q4_ci[1],
            },
            "metrics": agg,
        }
        rows.append(
            f"{dataset},{g_os:.5f},{g_og:.5f},{headroom:.4f},"
            f"{g_static:.5f},{agg['cached_mur']['mean_prediction_gain']['mean']:.5f},"
            f"{agg['ranked_response_q2']['mean_prediction_gain']['mean']:.5f},"
            f"{g_q4:.5f},{agg['ranked_response_q8']['mean_prediction_gain']['mean']:.5f},"
            f"{agg['ranked_response_full']['mean_prediction_gain']['mean']:.5f},"
            f"{r_oracle:.4f},{r_deploy:.4f}"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
