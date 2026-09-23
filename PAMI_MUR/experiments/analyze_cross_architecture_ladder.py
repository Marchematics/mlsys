#!/usr/bin/env python3
"""Cross-architecture expert ladder: does realized value fall as the expert improves?

Corollary 1 predicts that the identifiable part of a candidate's value is the
fixed predictor's own residual, so routing should realise a smaller share of the
structural opportunity as the expert gets stronger --- independently of the
architecture. This script assembles rungs of *different* architectures on
identical cells (METR-LA, the eight evenly spaced targets, seed 101) and reports
expert strength next to structural opportunity and realized value.

Strength is the anchor-only mean squared error on held-out windows, taken from
the released artifacts (documented per rung in the output JSON).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "KBS_MUR" / "scripts"))

from run_traffic_response_router_sweep import paired_ci  # noqa: E402

RUNGS = [
    {
        "rung": "ridge",
        "architecture": "per-target subset ridge (linear)",
        "strength": 0.3782,
        "strength_source": "results/derived/R104_expert_strength_metrla/summary.json (empty mask)",
        "run": None,  # filled from --ridge-root
        "seeds": [101],
    },
    {
        "rung": "deepsets",
        "architecture": "masked-mean MLP (permutation invariant)",
        "strength": 0.3625,
        "strength_source": "results/raw/R101_expert_pretrain_20260921_0237/METRLA/deepsets_seed101/summary.json",
        "run": "results/raw/R111_deepsets_metrla",
        "seeds": [101],
    },
    {
        "rung": "st_small",
        "architecture": "spatio-temporal transformer, 1 layer d=32",
        "strength": 0.3721,
        "strength_source": "results/raw/R132_expert_ladder_*/pretrain/METRLA/small/summary.json",
        "run": None,  # filled from --ladder-root
        "seeds": [101],
    },
    {
        "rung": "st_medium",
        "architecture": "spatio-temporal transformer, 2 layers d=64",
        "strength": 0.3578,
        "strength_source": "results/raw/R132_expert_ladder_*/pretrain/METRLA/medium/summary.json",
        "run": None,
        "seeds": [101],
    },
    {
        "rung": "st_large",
        "architecture": "spatio-temporal transformer, 3 layers d=128",
        "strength": 0.3562,
        "strength_source": "results/raw/R101_expert_pretrain_20260921_0237/METRLA/seed101/summary.json",
        "run": None,
        "seeds": [101],
    },
]


def per_cell(root: Path, policy: str, seeds: set[int]) -> dict[int, float]:
    values: dict[int, list[float]] = {}
    for cell in Path(root).glob("target*_seed*"):
        result = cell / "result.json"
        if not result.exists():
            continue
        record = json.loads(result.read_text(encoding="utf-8"))
        seed = int(record.get("seed", cell.name.split("_")[1].replace("seed", "")))
        if seed not in seeds:
            continue
        metrics = record.get("metrics", {}).get(policy)
        if metrics is None:
            continue
        target = int(record.get("target_sensor", cell.name.split("_")[0].replace("target", "")))
        values.setdefault(target, []).append(float(metrics["mean_prediction_gain"]))
    return {t: float(np.mean(v)) for t, v in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder-root", type=Path, required=True)
    parser.add_argument("--ridge-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for rung in RUNGS:
        run = Path(rung["run"]) if rung["run"] else None
        if rung["rung"] == "ridge":
            run = args.ridge_root
        elif rung["rung"].startswith("st_"):
            run = args.ladder_root / f"eval_{rung['rung'].split('_')[1]}"
        if run is None or not Path(run).exists():
            continue
        seeds = set(rung["seeds"])
        policies = {
            p: per_cell(run, p, seeds)
            for p in ("oracle_static", "oracle_greedy", "ranked_response_q4", "static_utility", "random")
        }
        required = ("oracle_static", "oracle_greedy", "ranked_response_q4")
        usable = {p: v for p, v in policies.items() if v}
        missing_required = [p for p in required if p not in usable]
        if missing_required:
            print(f"  skipping rung {rung['rung']}: missing policies {missing_required}")
            continue
        common = sorted(set.intersection(*[set(v) for v in usable.values()]))
        if not common:
            continue
        means = {p: float(np.mean([v[t] for t in common])) for p, v in usable.items()}
        shares = np.asarray([
            policies["ranked_response_q4"][t] / policies["oracle_static"][t]
            for t in common if policies["oracle_static"][t] > 1e-9
        ])
        rows.append({
            "rung": rung["rung"],
            "architecture": rung["architecture"],
            "anchor_only_mse": rung["strength"],
            "strength_source": rung["strength_source"],
            "cells": len(common),
            "oracle_static_gain": means["oracle_static"],
            "oracle_greedy_gain": means["oracle_greedy"],
            "learned_gain": means["ranked_response_q4"],
            "static_gain": means.get("static_utility"),
            "random_gain": means.get("random"),
            "realized_share": float(shares.mean()) if shares.size else float("nan"),
            "realized_share_ci": paired_ci(shares) if shares.size > 1 else None,
        })
    rows.sort(key=lambda r: -r["anchor_only_mse"])  # weakest first
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cross_architecture_ladder.json").write_text(
        json.dumps({"rungs": rows, "cells": "METR-LA, eight evenly spaced targets, seed 101"}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{'rung':10s} {'architecture':44s} {'anchor MSE':>10s} {'oracle_s':>9s} {'learned':>9s} {'realized':>9s}")
    for row in rows:
        print(f"{row['rung']:10s} {row['architecture']:44s} {row['anchor_only_mse']:10.4f} "
              f"{row['oracle_static_gain']:+9.4f} {row['learned_gain']:+9.4f} {row['realized_share']:9.3f}")
    strengths = np.asarray([r["anchor_only_mse"] for r in rows])
    shares = np.asarray([r["realized_share"] for r in rows])
    if strengths.size > 2:
        print(f"\ncorr(anchor-only MSE, realized share) = {np.corrcoef(strengths, shares)[0,1]:+.3f} "
              f"(positive means weaker experts realise more)")


if __name__ == "__main__":
    main()
