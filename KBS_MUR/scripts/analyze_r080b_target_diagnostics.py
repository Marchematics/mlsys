#!/usr/bin/env python3
"""Post-hoc target-level diagnostics for the multi-target R-MUR runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/derived/R080b_target_diagnostics"
SOURCES = {
    202: ROOT / "results/raw/R080b_multitarget_rmur_s202_diag_20260921",
    303: ROOT / "results/raw/R080b_multitarget_rmur_s303_diag_20260921",
    101: ROOT / "results/raw/R080b_multitarget_rmur_s101_diag_metrla_20260921",
}
DATASETS = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]


def diagnostics_for(root: Path, dataset: str):
    records = []
    data_dir = root / dataset
    if not data_dir.exists():
        return records
    for run_dir in sorted(data_dir.glob("target*_seed*")):
        result_path = run_dir / "result.json"
        diag_path = run_dir / "diagnostics.npz"
        if not result_path.exists() or not diag_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        records.append((run_dir, result))
    return records


def score_metrics(predicted: np.ndarray, true: np.ndarray):
    mask = np.isfinite(true) & np.isfinite(predicted)
    if mask.sum() < 4:
        return float("nan"), float("nan")
    x = predicted[mask]
    y = true[mask]
    rho = float(spearmanr(x, y).statistic)
    if np.unique((y > 0).astype(int)).size < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score((y > 0).astype(int), x))
    return rho, auc


def cached_regret(diag):
    scores = diag["cached_score"]
    true = diag["true_marginal"]
    states = diag["state_mask"].astype(bool)
    regrets = []
    for step in range(scores.shape[0]):
        for row in range(scores.shape[1]):
            eligible = ~states[step, row]
            if not np.any(eligible):
                continue
            values = true[step, row, eligible]
            choices = scores[step, row, eligible]
            if values.size < 2:
                continue
            true_best = float(np.max(values))
            cached_choice = int(np.argmax(choices))
            regrets.append(true_best - float(values[cached_choice]))
    return float(np.mean(regrets)) if regrets else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_records = []
    for seed, root in SOURCES.items():
        for dataset in DATASETS:
            for run_dir, result in diagnostics_for(root, dataset):
                diag = np.load(run_dir / "diagnostics.npz")
                true = diag["true_marginal"]
                ranked = diag["ranked_score"]
                cached = diag["cached_score"]
                rho_rank, auc_rank = score_metrics(ranked, true)
                rho_cache, auc_cache = score_metrics(cached, true)
                margin = diag["margin"]
                margins = margin[np.isfinite(margin)]
                target = int(result.get("target_sensor", run_dir.name.split("_")[0].replace("target", "")))
                q4 = result["metrics"]["ranked_response_q4"]["mean_prediction_gain"]
                static = result["metrics"]["static_utility"]["mean_prediction_gain"]
                cached_gain = result["metrics"]["cached_mur"]["mean_prediction_gain"]
                all_records.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "seed": seed,
                        "delta_static": float(q4 - static),
                        "delta_cached": float(q4 - cached_gain),
                        "spearman_ranked": rho_rank,
                        "spearman_cached": rho_cache,
                        "auc_ranked": auc_rank,
                        "auc_cached": auc_cache,
                        "mean_margin": float(np.mean(margins)) if margins.size else float("nan"),
                        "cached_regret": cached_regret(diag),
                        "ranked_regret": float(np.nanmean(diag["regret"])),
                    }
                )
    (OUT / "per_target_seed.csv").write_text(
        "dataset,target,seed,delta_static,delta_cached,spearman_ranked,spearman_cached,auc_ranked,auc_cached,mean_margin,cached_regret,ranked_regret\n"
        + "\n".join(
            f"{r['dataset']},{r['target']},{r['seed']},{r['delta_static']:.6f},{r['delta_cached']:.6f},"
            f"{r['spearman_ranked']:.6f},{r['spearman_cached']:.6f},{r['auc_ranked']:.6f},{r['auc_cached']:.6f},"
            f"{r['mean_margin']:.6f},{r['cached_regret']:.6f},{r['ranked_regret']:.6f}"
            for r in all_records
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(all_records)} target-seed diagnostic records")


if __name__ == "__main__":
    main()
