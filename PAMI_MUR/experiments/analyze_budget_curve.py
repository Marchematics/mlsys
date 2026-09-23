#!/usr/bin/env python3
"""Analyze the budget curve (R121) and the backward-vs-forward oracle (R120)."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    curves = [
        json.loads((cell / "curve.json").read_text(encoding="utf-8"))["curve"]
        for cell in sorted(args.curve_root.glob("target*_seed*"))
    ]
    if not curves:
        raise SystemExit("no curve cells")
    ks = sorted(int(k) for k in curves[0])
    random_curve = {k: float(np.mean([c[str(k)]["random_mean"] for c in curves])) for k in ks}
    oracle_curve = {k: float(np.mean([c[str(k)]["oracle_mean"] for c in curves])) for k in ks}
    report = {
        "cells": len(curves),
        "random_curve": random_curve,
        "oracle_curve": oracle_curve,
        "oracle_argmax_k": max(ks, key=lambda k: oracle_curve[k]),
        "oracle_max": max(oracle_curve.values()),
        "random_monotone": bool(all(random_curve[k] <= random_curve[k + 1] + 1e-9 for k in ks[:-1])),
        "pool_all": random_curve[max(ks)],
        "oracle_at_budget_4": oracle_curve[4],
        "gap_pool_all_to_oracle_max": max(oracle_curve.values()) - random_curve[max(ks)],
    }

    forward, backward = [], []
    for cell in sorted(args.oracle_root.glob("target*_seed*")):
        arrays = np.load(cell / "episode_gains.npz")
        forward.append(float(arrays["oracle_forward"].mean()))
        backward.append(float(arrays["oracle_backward"].mean()))
    forward = np.asarray(forward)
    backward = np.asarray(backward)
    difference = backward - forward
    report["backward_vs_forward"] = {
        "cells": int(difference.size),
        "forward_mean": float(forward.mean()),
        "backward_mean": float(backward.mean()),
        "difference": paired_ci(difference),
        "backward_wins": int(np.sum(difference > 0)),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "budget_and_direction.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
