"""Does structural headroom predict where routing succeeds? (R153)

The protocol defines structural headroom as ``H_state = (G_seq - G_stand)/G_seq``
and treats it as the opportunity a router could capture. That makes a natural
prediction: targets (or datasets) with more headroom should benefit more from
routing. This script tests it in the regime where routing works, using the
six-dataset ridge protocol.

Two questions:

1. Per target -- within a dataset, does ``H_state`` order the router's
   advantage over pooling?
2. Per dataset -- does mean headroom order the mean advantage across the six
   benchmarks?

It also reports the effect size per dataset (mean over its standard error and
the fraction of targets where routing wins), which is what distinguishes a
genuinely marginal dataset from an under-powered one.

Usage
-----
python experiments/analyze_headroom_predictiveness.py \
    --out results/derived/R153_headroom_predictiveness
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
KBS_RAW = ROOT / "KBS_MUR/results/raw"
DATASETS = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]
LEARNED_RUNS = [
    "R080b_multitarget_rmur_s1_20260920_2340",
    "R080b_multitarget_rmur_s202_diag_20260921",
    "R080b_multitarget_rmur_s303_diag_20260921",
]
PRIMARY = "ranked_response_q4"


def load(dataset: str) -> dict[int, dict[int, dict[str, float]]]:
    per_target: dict[int, dict[int, dict[str, float]]] = {}
    for run in LEARNED_RUNS:
        for cell in sorted((KBS_RAW / run / dataset).glob("target*_seed*")):
            result = cell / "result.json"
            if not result.exists():
                continue
            record = json.loads(result.read_text())
            metrics = record.get("metrics") or {}
            if PRIMARY not in metrics:
                continue
            per_target.setdefault(int(record["target_sensor"]), {})[int(record["seed"])] = {
                name: float(value["mean_prediction_gain"])
                for name, value in metrics.items()
                if isinstance(value, dict)
            }
    return per_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = {"provenance": {"learned_runs": LEARNED_RUNS, "primary_policy": PRIMARY}, "datasets": {}}
    pooled_h, pooled_d = [], []
    for dataset in DATASETS:
        per_target = load(dataset)
        headroom, advantage = [], []
        for target, seeds in sorted(per_target.items()):
            standalone = np.mean([m["oracle_static"] for m in seeds.values()])
            sequential = np.mean([m["oracle_greedy"] for m in seeds.values()])
            delta = np.mean([m[PRIMARY] - m["pool_all"] for m in seeds.values()])
            if sequential > 0:
                headroom.append(float((sequential - standalone) / sequential))
                advantage.append(float(delta))
        h = np.asarray(headroom)
        d = np.asarray(advantage)
        pooled_h.append(h)
        pooled_d.append(d)
        se = d.std(ddof=1) / np.sqrt(d.size)
        summary["datasets"][dataset] = {
            "targets": int(d.size),
            "mean_advantage": float(d.mean()),
            "sd_advantage": float(d.std(ddof=1)),
            "se_advantage": float(se),
            "t_statistic": float(d.mean() / se),
            "fraction_targets_router_wins": float(np.mean(d > 0)),
            "mean_headroom": float(h.mean()),
            "sd_headroom": float(h.std(ddof=1)),
            "pearson_headroom_vs_advantage": {
                "r": float(pearsonr(h, d).statistic), "p": float(pearsonr(h, d).pvalue)
            },
            "spearman_headroom_vs_advantage": {
                "rho": float(spearmanr(h, d).statistic), "p": float(spearmanr(h, d).pvalue)
            },
        }

    h = np.concatenate(pooled_h)
    d = np.concatenate(pooled_d)
    # within-dataset (fixed-effects) association
    hc = np.concatenate([x - x.mean() for x in pooled_h])
    dc = np.concatenate([x - x.mean() for x in pooled_d])
    summary["pooled"] = {
        "targets": int(h.size),
        "pearson_raw": {"r": float(pearsonr(h, d).statistic), "p": float(pearsonr(h, d).pvalue)},
        "spearman_raw": {"rho": float(spearmanr(h, d).statistic), "p": float(spearmanr(h, d).pvalue)},
        "pearson_within_dataset": {"r": float(pearsonr(hc, dc).statistic), "p": float(pearsonr(hc, dc).pvalue)},
        "spearman_within_dataset": {"rho": float(spearmanr(hc, dc).statistic), "p": float(spearmanr(hc, dc).pvalue)},
    }
    # dataset-level ordering: does mean headroom order mean advantage?
    means_h = np.asarray([summary["datasets"][x]["mean_headroom"] for x in DATASETS])
    means_d = np.asarray([summary["datasets"][x]["mean_advantage"] for x in DATASETS])
    summary["across_datasets"] = {
        "pearson_mean_headroom_vs_mean_advantage": {
            "r": float(pearsonr(means_h, means_d).statistic),
            "p": float(pearsonr(means_h, means_d).pvalue),
            "n": len(DATASETS),
        },
        "spearman_mean_headroom_vs_mean_advantage": {
            "rho": float(spearmanr(means_h, means_d).statistic),
            "p": float(spearmanr(means_h, means_d).pvalue),
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "headroom_predictiveness.json").write_text(json.dumps(summary, indent=1))

    print(f"{'dataset':9s} {'n':>4s} {'mean d':>9s} {'se':>7s} {'t':>7s} {'win frac':>9s} "
          f"{'mean H':>8s} {'r(H,d)':>8s}")
    for dataset in DATASETS:
        e = summary["datasets"][dataset]
        print(f"{dataset:9s} {e['targets']:>4d} {e['mean_advantage']:>+9.4f} {e['se_advantage']:>7.4f} "
              f"{e['t_statistic']:>+7.2f} {e['fraction_targets_router_wins']:>9.2f} "
              f"{e['mean_headroom']:>8.4f} {e['pearson_headroom_vs_advantage']['r']:>+8.3f}")
    p = summary["pooled"]
    print(f"\npooled targets: {p['targets']}")
    print(f"  within-dataset Pearson r={p['pearson_within_dataset']['r']:+.3f} "
          f"(p={p['pearson_within_dataset']['p']:.3f}), Spearman={p['spearman_within_dataset']['rho']:+.3f}")
    a = summary["across_datasets"]
    print(f"  across datasets (n=6): Pearson r={a['pearson_mean_headroom_vs_mean_advantage']['r']:+.3f} "
          f"(p={a['pearson_mean_headroom_vs_mean_advantage']['p']:.3f})")
    print(f"\nwrote {args.out / 'headroom_predictiveness.json'}")


if __name__ == "__main__":
    main()
