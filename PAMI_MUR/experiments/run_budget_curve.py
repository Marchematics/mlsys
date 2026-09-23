#!/usr/bin/env python3
"""Budget curve: random-k, oracle-k and pool-all under a strong predictor.

Measures how the value of context selection depends on the budget itself. For
each k the script evaluates (a) the mean gain of random k-subsets, (b) the gain
of the greedy forward oracle restricted to k, and (c) the full pool, so the
crossing point between "select k" and "use everything" is read off directly.
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


def oracle_k(batch, ops, budget: int) -> np.ndarray:
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    rows = np.arange(batch.episodes)
    for _ in range(budget):
        marginal = ops.candidate_marginals(batch, selected)
        choice = np.argmax(marginal, axis=1)
        selected[rows, choice] = True
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/staeformer")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--target-count", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--draws", type=int, default=3, help="random subsets per k per episode")
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--calibration-episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=500)
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
            data.correlation[target], target_sensor=target,
            candidate_count=args.candidate_count, seed=20260920,
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
                budget=args.candidate_count, seed=seed + 7_777, device=args.device,
                validation_batch=batches[1],
            )
            ops = SubsetExpertOps(model, device=args.device)
            test = batches[2]
            empty = np.zeros((test.episodes, args.candidate_count), dtype=bool)
            base_loss = ops.squared_error(test, empty)
            curve = {}
            for k in range(1, args.candidate_count + 1):
                random_gains = []
                for draw in range(args.draws):
                    draw_rng = np.random.default_rng(seed + 100 * k + draw)
                    mask = np.zeros((test.episodes, args.candidate_count), dtype=bool)
                    for row in range(test.episodes):
                        mask[row, draw_rng.choice(args.candidate_count, size=k, replace=False)] = True
                    random_gains.append(base_loss - ops.squared_error(test, mask))
                selected = oracle_k(test, ops, k)
                curve[k] = {
                    "random_mean": float(np.mean(np.concatenate(random_gains))),
                    "oracle_mean": float(np.mean(base_loss - ops.squared_error(test, selected))),
                }
            cell = args.out_root / f"target{target}_seed{seed}"
            cell.mkdir(parents=True)
            (cell / "curve.json").write_text(
                json.dumps(
                    {
                        "dataset": args.dataset,
                        "target_sensor": target,
                        "seed": seed,
                        "episodes": test.episodes,
                        "draws": args.draws,
                        "curve": curve,
                        "wall_seconds": time.time() - started,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "target": target,
                        "seed": seed,
                        "random_k4": round(curve[4]["random_mean"], 5),
                        "oracle_k4": round(curve[4]["oracle_mean"], 5),
                        "random_k16": round(curve[16]["random_mean"], 5),
                    }
                ),
                flush=True,
            )
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
