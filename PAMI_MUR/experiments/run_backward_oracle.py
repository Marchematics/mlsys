#!/usr/bin/env python3
"""Backward elimination versus forward selection (oracle-style diagnostic).

The protocol only ever *adds* contexts. If the utility is not submodular, the
best four contexts found by pruning down from the full pool can differ from the
best four found by greedy addition, and the gap says whether forward selection
is limited by search direction or by identifiability. Both policies here use
the true marginals of the adapted expert, so they are oracles.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KBS = ROOT / "KBS_MUR"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KBS / "src"))

from neural_expert import SubsetExpertOps, finetune_expert, load_checkpoint  # noqa: E402
from pami_traffic import build_candidate_pool, load_traffic_data, make_neural_batch  # noqa: E402


def oracle_forward(batch, ops, budget: int) -> np.ndarray:
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    rows = np.arange(batch.episodes)
    for _ in range(budget):
        marginal = ops.candidate_marginals(batch, selected)
        choice = np.argmax(marginal, axis=1)
        selected[rows, choice] = True
    return selected


def oracle_backward(batch, ops, budget: int) -> np.ndarray:
    """Start from the full pool, repeatedly drop the most harmful candidate."""

    selected = np.ones((batch.episodes, batch.candidate_count), dtype=bool)
    rows = np.arange(batch.episodes)
    width = batch.candidate_count
    for _ in range(width - budget):
        marginal = ops.candidate_marginals(batch, selected)
        # marginal[j] is the gain of adding j; for a selected j the loss change
        # of *removing* it is measured directly.
        loss_with = ops.squared_error(batch, selected)
        best_gain = np.full(batch.episodes, -np.inf)
        best_candidate = np.zeros(batch.episodes, dtype=np.int64)
        for candidate in range(width):
            eligible = selected[:, candidate]
            if not np.any(eligible):
                continue
            reduced = selected.copy()
            reduced[eligible, candidate] = False
            gain = loss_with[eligible] - ops.squared_error(batch, reduced)[eligible]
            improve = gain > best_gain[eligible]
            index = np.flatnonzero(eligible)[improve]
            best_gain[index] = gain[improve]
            best_candidate[index] = candidate
        selected[rows, best_candidate] = False
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/staeformer")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--target-count", type=int, default=8)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--calibration-episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=1000)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)
    data = load_traffic_data(args.data_root / args.dataset)
    targets = np.unique(np.linspace(0, data.num_nodes - 1, args.target_count, dtype=np.int64))
    for target in targets:
        target = int(target)
        candidates = build_candidate_pool(
            data.correlation[target], target_sensor=target, candidate_count=16, seed=20260920
        )
        for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
            started = time.time()
            model, _ = load_checkpoint(args.expert_root / f"seed{seed}" / "expert.pt", device=args.device)
            model = copy.deepcopy(model)
            rng = np.random.default_rng(seed + 2_000)
            batches = []
            for starts, count in (
                (data.train_times, args.train_episodes),
                (data.validation_times, args.calibration_episodes),
                (data.test_times, args.test_episodes),
            ):
                times = rng.choice(starts, size=min(count, len(starts)), replace=False)
                batches.append(make_neural_batch(data, times, target_sensor=target, candidate_sensors=candidates))
            batches = tuple(batches)
            finetune_expert(
                model, batches[0], epochs=args.finetune_epochs, batch_size=256, learning_rate=3e-4,
                budget=args.budget, seed=seed + 7_777, device=args.device, validation_batch=batches[1],
            )
            ops = SubsetExpertOps(model, device=args.device)
            test = batches[2]
            empty = np.zeros((test.episodes, test.candidate_count), dtype=bool)
            base_loss = ops.squared_error(test, empty)
            forward = oracle_forward(test, ops, args.budget)
            backward = oracle_backward(test, ops, args.budget)
            gains = {
                "oracle_forward": base_loss - ops.squared_error(test, forward),
                "oracle_backward": base_loss - ops.squared_error(test, backward),
            }
            cell = args.out_root / f"target{target}_seed{seed}"
            cell.mkdir(parents=True)
            np.savez_compressed(cell / "episode_gains.npz", **{k: v.astype(np.float32) for k, v in gains.items()})
            (cell / "result.json").write_text(
                json.dumps(
                    {
                        "dataset": args.dataset,
                        "target_sensor": target,
                        "seed": seed,
                        "policies": {k: {"mean_prediction_gain": float(np.mean(v))} for k, v in gains.items()},
                        "wall_seconds": time.time() - started,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"target": target, "seed": seed,
                              **{k: round(float(np.mean(v)), 5) for k, v in gains.items()}}), flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
