#!/usr/bin/env python3
"""Recompute the R112 redundancy diagnostics from the frozen predictors.

The redundancy numbers in ``result.json`` were produced by an earlier version of
``redundancy_diagnostics`` that measured a cluster member's marginal with the
whole budget already spent.  This script rebuilds each cell's test batch (same
seeds as the driver) and re-measures the diagnostics with the corrected
definition -- the marginal of a second member given *its own cluster's seed*
only -- writing one JSON per environment.  No model is retrained and no raw run
is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAMI = HERE.parents[1]
ROOT = PAMI.parent
KBS = ROOT / "KBS_MUR"
for _path in (HERE, PAMI / "experiments", KBS / "src", KBS / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from demo_protocol import BENCHMARK_CONFIGS, load_protocol  # noqa: E402
from incontext_predictor import DemoExpertOps, load_checkpoint  # noqa: E402
from redundant_pool import build_redundant_batch  # noqa: E402
from run_r112_redundant_demo import redundancy_diagnostics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--benchmarks", default="cifar10,cifar100,svhn,eurosat,dtd")
    parser.add_argument("--copy-jitter", type=float, default=0.0)
    parser.add_argument("--feature-root", type=Path, default=PAMI / "data/vision_features")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    payload: dict = {"source_run": str(args.run_root), "copy_jitter": args.copy_jitter, "cells": {}}
    for name in [item.strip() for item in args.benchmarks.split(",") if item.strip()]:
        protocol = load_protocol(args.feature_root, name)
        config = BENCHMARK_CONFIGS[name]
        per_seed = []
        for seed in seeds:
            checkpoint = args.run_root / name / f"seed{seed}" / "predictor.pt"
            if not checkpoint.exists():
                print(f"missing {checkpoint}", flush=True)
                continue
            batch, info = build_redundant_batch(
                protocol.features,
                protocol.mean,
                protocol.components,
                query_source=protocol.splits.test,
                pool_source=protocol.splits.pool,
                distractor_pool=protocol.splits.distractor_pool,
                query_split="test",
                episodes_per_class=config.test_episodes_per_class,
                seed=seed + 2_000,
                config=config,
                copy_jitter=args.copy_jitter,
            )
            model, _ = load_checkpoint(checkpoint, device=args.device)
            ops = DemoExpertOps(model, device=args.device, chunk_pairs=2048)
            diagnostics = redundancy_diagnostics(ops, batch, info)
            per_seed.append({key: value for key, value in diagnostics.items() if not key.startswith("per_episode")})
            del model, ops
        if per_seed:
            payload["cells"][name] = {
                key: float(np.mean([entry[key] for entry in per_seed])) for key in per_seed[0]
            }
            payload["cells"][name]["per_seed"] = per_seed
            print(
                json.dumps(
                    {
                        "benchmark": name,
                        "first": round(payload["cells"][name]["mean_first_member_marginal"], 4),
                        "second": round(payload["cells"][name]["mean_second_member_marginal"], 4),
                        "ratio": round(
                            payload["cells"][name]["mean_second_member_marginal"]
                            / max(payload["cells"][name]["mean_first_member_marginal"], 1e-9),
                            4,
                        ),
                    }
                ),
                flush=True,
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
