"""Does a pilot on one time period predict the deployment on another? (R155)

R154 validated the measurement protocol as a deployment decision when the pilot
and the deployment are drawn from the same period: a 24-target pilot reproduced
the full-sample verdict 96% of the time. That leaves the boundary that matters
most in practice --- the pilot is necessarily labelled *earlier* than the period
the router will run on.

This script answers it with a two-split run of the six-dataset ridge protocol:
the router and pool-everything are scored on the calibration period (the pilot)
and on the test period (the deployment) for the same targets. It reports

* the per-target association between the pilot and deployment differences, and
* the verdict the pilot would have produced versus the verdict on the
  deployment, at both target-pair and dataset level.

Usage
-----
python experiments/analyze_pilot_shift.py \
    --run-root /root/icl_ess_threshold/KBS_MUR/results/raw/R155_pilot_shift_20260925_1000 \
    --out results/derived/R155_pilot_shift
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

DATASETS = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]
ROUTER = "ranked_response_q4"


def load(root: Path, dataset: str) -> dict[int, dict[str, float]]:
    """Per target: the router-minus-pooling difference on each split."""

    out: dict[int, dict[str, float]] = {}
    for cell in sorted((root / dataset).glob("target*_seed*")):
        result = cell / "result.json"
        if not result.exists():
            continue
        record = json.loads(result.read_text())
        test = record.get("metrics") or {}
        validation = record.get("metrics_validation")
        if not validation or ROUTER not in test or ROUTER not in validation:
            continue
        out[int(record["target_sensor"])] = {
            "pilot": float(validation[ROUTER]["mean_prediction_gain"])
            - float(validation["pool_all"]["mean_prediction_gain"]),
            "deployment": float(test[ROUTER]["mean_prediction_gain"])
            - float(test["pool_all"]["mean_prediction_gain"]),
            "pilot_router": float(validation[ROUTER]["mean_prediction_gain"]),
            "deployment_router": float(test[ROUTER]["mean_prediction_gain"]),
            "pilot_pool": float(validation["pool_all"]["mean_prediction_gain"]),
            "deployment_pool": float(test["pool_all"]["mean_prediction_gain"]),
        }
    return out


def verdict(values: np.ndarray, *, draws: int = 10000, seed: int = 0) -> str:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    low, high = np.quantile(means, 0.025), np.quantile(means, 0.975)
    if low > 0:
        return "AHEAD"
    if high < 0:
        return "BELOW"
    return "UNRESOLVED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = {"run_root": str(args.run_root), "datasets": {}}
    pooled_pilot, pooled_deploy = [], []
    agree = total = 0
    for dataset in DATASETS:
        cells = load(args.run_root, dataset)
        if not cells:
            continue
        pilot = np.asarray([cells[t]["pilot"] for t in sorted(cells)], dtype=float)
        deploy = np.asarray([cells[t]["deployment"] for t in sorted(cells)], dtype=float)
        entry = {
            "targets": int(pilot.size),
            "pilot_mean": float(pilot.mean()),
            "deployment_mean": float(deploy.mean()),
            "pilot_verdict": verdict(pilot),
            "deployment_verdict": verdict(deploy),
            "sign_agreement": float(np.mean(np.sign(pilot) == np.sign(deploy))),
            "pearson": {"r": float(pearsonr(pilot, deploy).statistic),
                        "p": float(pearsonr(pilot, deploy).pvalue)},
            "spearman": {"rho": float(spearmanr(pilot, deploy).statistic),
                         "p": float(spearmanr(pilot, deploy).pvalue)},
        }
        entry["verdict_agrees"] = entry["pilot_verdict"] == entry["deployment_verdict"]
        summary["datasets"][dataset] = entry
        pooled_pilot.append(pilot)
        pooled_deploy.append(deploy)
        agree += int(entry["verdict_agrees"])
        total += 1

    if pooled_pilot:
        p = np.concatenate(pooled_pilot)
        d = np.concatenate(pooled_deploy)
        pc = np.concatenate([x - x.mean() for x in pooled_pilot])
        dc = np.concatenate([x - x.mean() for x in pooled_deploy])
        summary["pooled"] = {
            "targets": int(p.size),
            "pearson": {"r": float(pearsonr(p, d).statistic), "p": float(pearsonr(p, d).pvalue)},
            "pearson_within_dataset": {"r": float(pearsonr(pc, dc).statistic),
                                       "p": float(pearsonr(pc, dc).pvalue)},
            "spearman_within_dataset": {"rho": float(spearmanr(pc, dc).statistic),
                                        "p": float(spearmanr(pc, dc).pvalue)},
            "sign_agreement": float(np.mean(np.sign(p) == np.sign(d))),
        }
        summary["dataset_verdict_agreement"] = f"{agree}/{total}"

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pilot_shift.json").write_text(json.dumps(summary, indent=1))

    print(f"{'dataset':9s} {'n':>4s} {'pilot Δ':>9s} {'deploy Δ':>9s} {'pilot verdict':>14s} "
          f"{'deploy verdict':>15s} {'sign agree':>11s} {'r':>7s}")
    for dataset in DATASETS:
        e = summary["datasets"].get(dataset)
        if not e:
            continue
        print(f"{dataset:9s} {e['targets']:>4d} {e['pilot_mean']:>+9.4f} {e['deployment_mean']:>+9.4f} "
              f"{e['pilot_verdict']:>14s} {e['deployment_verdict']:>15s} "
              f"{e['sign_agreement']:>11.2f} {e['pearson']['r']:>+7.3f}")
    if "pooled" in summary:
        p = summary["pooled"]
        print(f"\npooled targets: {p['targets']}")
        print(f"  within-dataset Pearson r={p['pearson_within_dataset']['r']:+.3f} "
              f"(p={p['pearson_within_dataset']['p']:.4f})")
        print(f"  per-target sign agreement {p['sign_agreement']:.2f}")
        print(f"  dataset-level verdict agreement {summary['dataset_verdict_agreement']}")
    print(f"\nwrote {args.out / 'pilot_shift.json'}")


if __name__ == "__main__":
    main()
