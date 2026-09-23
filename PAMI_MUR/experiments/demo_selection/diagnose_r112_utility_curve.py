#!/usr/bin/env python3
"""R112 mechanism diagnostic: the utility curve behind H_state ~ 0.

For seed 101 of each benchmark the frozen predictor is evaluated on the test
episodes with 0..4 demonstrations chosen in four ways:

* ``relevance``: the k most query-relevant candidates (the kNN baseline's prefix);
* ``one_cluster``: k members of the single most relevant cluster (maximally
  redundant prefix);
* ``greedy``: the sequential oracle prefix;
* ``static``: the standalone-oracle prefix (top-k by m(j|empty)).

The curve shows where the marginal value of an additional demonstration dies,
which is what prevents any selection rule from separating itself.
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
from run_r110_demo_selection import relevance_of  # noqa: E402


def curve_for_cell(ops: DemoExpertOps, batch, info, *, budget: int) -> dict:
    episodes = batch.episodes
    relevance = relevance_of(batch)
    empty = np.zeros((episodes, batch.candidate_count), dtype=bool)
    base = ops.loss(batch, empty)
    order = np.argsort(-relevance, axis=1, kind="stable")
    seeds = info.seed_index

    def losses(mask: np.ndarray) -> np.ndarray:
        return ops.loss(batch, mask)

    result: dict[str, list[float]] = {"relevance": [], "one_cluster": [], "greedy": [], "static": []}
    for k in range(budget + 1):
        mask = np.zeros((episodes, batch.candidate_count), dtype=bool)
        if k:
            mask[np.arange(episodes)[:, None], order[:, :k]] = True
        result["relevance"].append(float(np.mean(base - losses(mask))))
    for k in range(budget + 1):
        mask = np.zeros((episodes, batch.candidate_count), dtype=bool)
        for episode in range(episodes):
            cluster = 0  # the most relevant cluster is the first one by construction
            members = np.flatnonzero(info.cluster_id[episode] == cluster)
            members = members[np.argsort(-relevance[episode, members], kind="stable")][:k]
            mask[episode, members] = True
        result["one_cluster"].append(float(np.mean(base - losses(mask))))

    marginal = ops.candidate_marginals(batch, empty)
    static_order = np.argsort(-marginal, axis=1, kind="stable")
    greedy_mask = np.zeros((episodes, batch.candidate_count), dtype=bool)
    static_mask = np.zeros((episodes, batch.candidate_count), dtype=bool)
    result["greedy"].append(0.0)
    result["static"].append(0.0)
    for k in range(budget):
        static_mask[np.arange(episodes), static_order[:, k]] = True
        result["static"].append(float(np.mean(base - losses(static_mask))))
        marginal = ops.candidate_marginals(batch, greedy_mask)
        choice = np.argmax(marginal, axis=1)
        greedy_mask[np.arange(episodes), choice] = True
        result["greedy"].append(float(np.mean(base - losses(greedy_mask))))
    return {
        "base_ce": float(np.mean(base)),
        "curves": result,
        "greedy_prefix": result["greedy"],
        "static_prefix": result["static"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=PAMI / "data/vision_features")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--benchmarks", default="cifar10,cifar100,svhn,eurosat,dtd")
    parser.add_argument("--copy-jitter", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    artifact = {"source_run": str(args.run_root), "seed": args.seed, "cells": {}}
    for name in [item.strip() for item in args.benchmarks.split(",") if item.strip()]:
        checkpoint = args.run_root / name / f"seed{args.seed}" / "predictor.pt"
        if not checkpoint.exists():
            print(f"missing {checkpoint}", flush=True)
            continue
        protocol = load_protocol(args.feature_root, name)
        config = BENCHMARK_CONFIGS[name]
        batch, info = build_redundant_batch(
            protocol.features,
            protocol.mean,
            protocol.components,
            query_source=protocol.splits.test,
            pool_source=protocol.splits.pool,
            distractor_pool=protocol.splits.distractor_pool,
            query_split="test",
            episodes_per_class=config.test_episodes_per_class,
            seed=args.seed + 2_000,
            config=config,
            copy_jitter=args.copy_jitter,
        )
        model, _ = load_checkpoint(checkpoint, device=args.device)
        ops = DemoExpertOps(model, device=args.device, chunk_pairs=2048)
        artifact["cells"][name] = curve_for_cell(ops, batch, info, budget=config.budget)
        print(
            json.dumps({"benchmark": name, **{k: [round(v, 4) for v in vals] for k, vals in artifact["cells"][name]["curves"].items()}}),
            flush=True,
        )
        del model, ops
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
