#!/usr/bin/env python3
"""Expert-strength ladder: test the identifiability prediction.

Corollary 1 says that the term a router must estimate is the fixed predictor's
own residual, so a stronger predictor should leave *less realizable* value even
though the *structural* opportunity (the oracle ladder) persists. This script
assembles a ladder of experts of increasing strength on the same targets and
measures, for each rung:

  * expert strength      anchor-only validation MSE from the pretraining summary
  * structural value     standalone and sequential oracle gains
  * realizable value     learned router gain and its share of the oracle gain
  * signal at the screen response-free (static) gain against random selection

The prediction is a decline in realizable share with expert strength, with the
oracle gain not declining comparably.
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


def cell_means(run_root: Path, policy: str) -> dict[tuple[int, int], float]:
    out = {}
    for cell in Path(run_root).glob("target*_seed*"):
        result = cell / "result.json"
        if not result.exists():
            continue
        record = json.loads(result.read_text(encoding="utf-8"))
        metrics = record.get("metrics", {}).get(policy)
        if metrics is None:
            continue
        target = int(record.get("target_sensor", cell.name.split("_")[0].replace("target", "")))
        seed = int(record.get("seed", cell.name.split("_")[1].replace("seed", "")))
        out[(target, seed)] = float(metrics["mean_prediction_gain"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder-root", type=Path, required=True)
    parser.add_argument("--variants", default="small,medium,large")
    parser.add_argument("--primary", default="ranked_response_q4")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for variant in [v for v in args.variants.split(",") if v]:
        eval_root = args.ladder_root / f"eval_{variant}"
        if not eval_root.exists():
            continue
        summary_path = args.ladder_root / "pretrain" / "METRLA" / variant / "summary.json"
        if not summary_path.exists():
            # the strongest rung reuses the original pretraining run
            summary_path = ROOT / "PAMI_MUR/results/raw/R101_expert_pretrain_20260921_0237/METRLA/seed101/summary.json"
        strength = None
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            final = summary.get("final_validation") or {}
            strength = final.get("empty", {}).get("mse") or final.get("average", {}).get("mse")
        policies = {p: cell_means(eval_root, p) for p in (
            "oracle_static", "oracle_greedy", args.primary, "static_utility", "cached_mur", "random"
        )}
        common = sorted(set.intersection(*[set(v) for v in policies.values() if v]))
        if not common:
            continue
        means = {p: float(np.mean([v[k] for k in common])) for p, v in policies.items()}
        realized = np.asarray([
            policies[args.primary][k] / policies["oracle_static"][k]
            for k in common if policies["oracle_static"][k] > 1e-9
        ])
        rows.append({
            "variant": variant,
            "cells": len(common),
            "anchor_only_mse": strength,
            "oracle_static_gain": means["oracle_static"],
            "oracle_greedy_gain": means["oracle_greedy"],
            "learned_gain": means[args.primary],
            "static_gain": means["static_utility"],
            "cached_gain": means["cached_mur"],
            "random_gain": means["random"],
            "realized_share": float(realized.mean()) if realized.size else float("nan"),
            "realized_share_ci": paired_ci(realized) if realized.size > 1 else None,
            "learned_minus_random": paired_ci(np.asarray([
                policies[args.primary][k] - policies["random"][k] for k in common
            ])),
        })
    rows.sort(key=lambda r: (r["anchor_only_mse"] is None, r["anchor_only_mse"]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ladder.json").write_text(json.dumps({"variants": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"{'variant':8s} {'anchor MSE':>10s} {'oracle_s':>9s} {'oracle_g':>9s} {'learned':>9s} {'static':>9s} {'random':>8s} {'realized':>9s}")
    for row in rows:
        print(
            f"{row['variant']:8s} {('n/a' if row['anchor_only_mse'] is None else format(row['anchor_only_mse'], '.4f')):>10} "
            f"{row['oracle_static_gain']:+9.4f} {row['oracle_greedy_gain']:+9.4f} {row['learned_gain']:+9.4f} "
            f"{row['static_gain']:+9.4f} {row['random_gain']:+8.4f} {row['realized_share']:+9.3f}"
        )


if __name__ == "__main__":
    main()
