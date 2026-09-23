"""Does the theory's degenerate-regime prediction hold? (R151)

The squared-loss identity gives

    E[m(j|A) | x, A] = <g_A(x), d_j> - ||d_j||^2 - lambda c_j,   g_A = 2(mu - f_A).

When the anchor is strong and adapted, the identifiable coefficient ``g_A`` is
close to zero because ``mu - f_A`` is dominated by irreducible noise. The score
then collapses to ``-||d_j||^2``, so the optimal action is to select the
candidates that perturb the predictor *least*. That is a sharp, falsifiable
prediction for regime B: ``min_disturbance_*`` should be the strongest
learning-free policy, and may even beat pooling.

This script tests it against the strong-backbone protocol, using the
response-geometry baselines that had never been run on the neural expert.

Usage
-----
python experiments/analyze_response_geometry.py \
    --run-root results/raw/R151_response_geometry_20260924_0910 \
    --out results/derived/R151_response_geometry
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STRONG_BACKBONE = ROOT / "results/raw/R103_strong_backbone_gate_20260921_0350"

GEOMETRY = [
    "min_disturbance_static",
    "min_disturbance_greedy",
    "max_disturbance_static",
    "consensus_alignment",
    "anti_consensus_alignment",
    "last_k",
]
LEARNED = ["ranked_response_q4", "ranked_response_q8", "cached_mur", "static_utility"]
CLASSICAL = ["random", "relevance", "mmr", "dpp", "facility_location", "mutual_information",
             "kcenter", "kmeans_representatives"]
REFERENCES = ["pool_all", "oracle_static", "oracle_greedy"]


def load(root: Path) -> dict:
    """Read per-cell policy gains from either result schema."""

    cells: dict[tuple[int, int], dict[str, float]] = {}
    if not root.exists():
        return cells
    for cell in sorted(root.glob("target*_seed*")):
        record = json.loads((cell / "result.json").read_text())
        block = record.get("metrics") or record.get("policies") or {}
        gains = {}
        for policy, value in block.items():
            if isinstance(value, dict) and "mean_prediction_gain" in value:
                gains[policy] = float(value["mean_prediction_gain"])
        if gains:
            cells[(int(record["target_sensor"]), int(record["seed"]))] = gains
    return cells


def merged(dataset: str, run_root: Path) -> dict:
    """The geometry run holds only the geometry policies; the strong-backbone
    run of R103 holds pooling, the learned routers and the classical rules at
    exactly the same targets, seeds and protocol, so the two are merged per
    cell rather than re-running either."""

    cells = load(run_root / dataset)
    reference = load(STRONG_BACKBONE / dataset)
    out = {}
    for key, gains in cells.items():
        if key in reference:
            out[key] = {**reference[key], **gains}
    return out


def paired_ci(values: np.ndarray, *, draws: int = 10000, seed: int = 20260925):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(means, 0.025)),
            "high": float(np.quantile(means, 0.975)), "n": int(values.size)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()

    summary = {"run_root": str(args.run_root), "datasets": {}}
    for dataset in ("METRLA", "PEMSBAY"):
        cells = merged(dataset, args.run_root)
        if not cells:
            print(f"{dataset}: no cells")
            continue
        entry = {"cells": len(cells), "policy": {}, "verdict": {}}
        pool = np.asarray([cells[k]["pool_all"] for k in sorted(cells)], dtype=float)
        for policy in GEOMETRY + LEARNED + CLASSICAL + REFERENCES:
            values = np.asarray([cells[k][policy] for k in sorted(cells) if policy in cells[k]], dtype=float)
            if values.size == 0:
                continue
            record = paired_ci(values, draws=args.draws)
            if policy != "pool_all" and values.size == pool.size:
                deltas = values - pool
                record["delta_vs_pool_all"] = paired_ci(deltas, draws=args.draws)
            entry["policy"][policy] = record
        # Theory prediction: with g_A ~ 0 the score collapses to -||d_j||^2, so
        # the least-perturbing policy should be the best learning-free rule and
        # should beat full pooling.
        candidates = [p for p in entry["policy"] if p not in REFERENCES]
        ranked = sorted(candidates, key=lambda p: -entry["policy"][p]["mean"])
        disturbance = entry["policy"].get("min_disturbance_static", {})
        delta = disturbance.get("delta_vs_pool_all", {})
        entry["verdict"] = {
            "best_policy_overall": ranked[0] if ranked else None,
            "best_geometry_policy": next((p for p in ranked if p in GEOMETRY), None),
            "min_disturbance_rank": ranked.index("min_disturbance_static") + 1
            if "min_disturbance_static" in ranked else None,
            "policies_compared": len(ranked),
            "min_disturbance_mean": disturbance.get("mean"),
            "min_disturbance_delta_vs_pool": delta.get("mean"),
            "min_disturbance_ci": [delta.get("low"), delta.get("high")],
            # the prediction under test: least perturbation wins AND beats pooling
            "prediction_beats_pooling": bool(delta.get("low", -1.0) > 0),
            "prediction_is_best_learning_free": entry["verdict_best_is_disturbance"]
            if "verdict_best_is_disturbance" in entry else ranked[:1] == ["min_disturbance_static"],
        }
        summary["datasets"][dataset] = entry

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "response_geometry.json").write_text(json.dumps(summary, indent=1))

    for dataset, entry in summary["datasets"].items():
        print(f"\n=== {dataset}  ({entry['cells']} cells)")
        ordered = sorted(entry["policy"].items(), key=lambda kv: -kv[1]["mean"])
        for policy, record in ordered:
            delta = record.get("delta_vs_pool_all")
            tail = ""
            if delta:
                star = "*" if (delta["low"] > 0 or delta["high"] < 0) else " "
                tail = f"  vs pool {delta['mean']:+.4f} [{delta['low']:+.4f},{delta['high']:+.4f}] {star}"
            print(f"  {policy:26s} {record['mean']:+.4f}{tail}")
        v = entry["verdict"]
        print(f"  -> best overall: {v['best_policy_overall']}; "
              f"min_disturbance_static ranks {v['min_disturbance_rank']}/{v['policies_compared']} "
              f"({v['min_disturbance_mean']:+.4f}, Δ vs pool {v['min_disturbance_delta_vs_pool']:+.4f} "
              f"[{v['min_disturbance_ci'][0]:+.4f},{v['min_disturbance_ci'][1]:+.4f}])")
        print(f"  -> theory prediction (least perturbation wins and beats pooling): "
              f"{v['prediction_beats_pooling']} / {v['prediction_is_best_learning_free']}")
    print(f"\nwrote {args.out / 'response_geometry.json'}")


if __name__ == "__main__":
    main()
