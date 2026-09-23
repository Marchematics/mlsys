#!/usr/bin/env python3
"""R115 diagnostics on the R112 pools for the prototype predictor.

Produces the two artefacts the report needs to compare predictor families:

* ``utility_curve.json`` -- mean query-batch CE reduction for k = 0..4 selected
  demonstrations, for the relevance prefix, the single-cluster (maximally
  redundant) prefix, the sequential oracle and the standalone oracle
  (``curve_for_cell`` is imported unchanged from the R112 diagnostic);
* ``redundancy.json`` -- second-member marginal as a share of the first-member
  marginal on the R112 redundancy pools (``redundancy_diagnostics`` imported
  unchanged from the R112 driver).
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
from diagnose_r112_utility_curve import curve_for_cell  # noqa: E402
from incontext_predictor import DemoExpertOps  # noqa: E402
from prototype_predictor import load_prototype_checkpoint  # noqa: E402
from redundant_pool import build_redundant_batch  # noqa: E402
from run_r112_redundant_demo import redundancy_diagnostics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--benchmarks", default="cifar10,cifar100,svhn,eurosat,dtd")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--feature-root", type=Path, default=PAMI / "data/vision_features")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    curve_artifact = {"source_run": str(args.run_root), "cells": {}}
    redundancy_artifact = {"source_run": str(args.run_root), "cells": {}}
    for name in [item.strip() for item in args.benchmarks.split(",") if item.strip()]:
        protocol = load_protocol(args.feature_root, name)
        config = BENCHMARK_CONFIGS[name]
        curves, diagnostics = [], []
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
            )
            model, _ = load_prototype_checkpoint(checkpoint, device=args.device)
            ops = DemoExpertOps(model, device=args.device, chunk_episodes=16, chunk_pairs=1024)
            if seed == seeds[0]:
                curves.append(curve_for_cell(ops, batch, info, budget=config.budget))
            diagnostics.append(redundancy_diagnostics(ops, batch, info))
            del model, ops
        if curves:
            curve_artifact["cells"][name] = curves[0]
        if diagnostics:
            redundancy_artifact["cells"][name] = {
                key: float(np.mean([entry[key] for entry in diagnostics]))
                for key in diagnostics[0]
                if not key.startswith("per_episode")
            }
            redundancy_artifact["cells"][name]["per_seed"] = [
                {key: value for key, value in entry.items() if not key.startswith("per_episode")}
                for entry in diagnostics
            ]
            diagnostics_entry = redundancy_artifact["cells"][name]
            print(
                json.dumps(
                    {
                        "benchmark": name,
                        "first": round(diagnostics_entry["mean_first_member_marginal"], 4),
                        "second": round(diagnostics_entry["mean_second_member_marginal"], 4),
                        "ratio": round(
                            diagnostics_entry["mean_second_member_marginal"]
                            / max(diagnostics_entry["mean_first_member_marginal"], 1e-9),
                            4,
                        ),
                        "within": round(diagnostics_entry["within_cluster_similarity"], 3),
                    }
                ),
                flush=True,
            )
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "utility_curve.json").write_text(json.dumps(curve_artifact, indent=2) + "\n", encoding="utf-8")
    (args.out_root / "redundancy.json").write_text(
        json.dumps(redundancy_artifact, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out_root}/utility_curve.json and redundancy.json", flush=True)


if __name__ == "__main__":
    main()
