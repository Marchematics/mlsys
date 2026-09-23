#!/usr/bin/env python3
"""Learned subset-selection baseline on the strong-backbone protocol.

Runs the policy-gradient selector of ``policy_selector.py`` on exactly the same
cells (dataset, target, seed), expert checkpoints, fine-tuning seeds and test
batches as ``run_strong_backbone.py``, so the resulting episode gains can be
paired with the stored gains of the gate run.
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
sys.path.insert(0, str(KBS / "scripts"))

from neural_expert import SubsetExpertOps, finetune_expert, load_checkpoint  # noqa: E402
from pami_traffic import (  # noqa: E402
    build_candidate_pool,
    load_traffic_data,
    make_neural_batch,
    target_set,
)
from policy_selector import LearnedPolicySelector, select_learned_policy, train_policy_selector  # noqa: E402


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/staeformer")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--targets", default="", help="explicit target list; default is --target-count")
    parser.add_argument("--target-count", type=int, default=16)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--calibration-episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=2000)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--policy-epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)
    data = load_traffic_data(args.data_root / args.dataset)
    targets = (
        np.asarray(parse_ints(args.targets), dtype=np.int64)
        if args.targets.strip()
        else target_set(data.num_nodes, args.target_count)
    )
    for target in targets:
        target = int(target)
        candidates = build_candidate_pool(
            data.correlation[target], target_sensor=target,
            candidate_count=args.candidate_count, seed=20260920,
        )
        for seed in parse_ints(args.seeds):
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
                model, batches[0], epochs=args.finetune_epochs, batch_size=256,
                learning_rate=3e-4, budget=args.budget, seed=seed + 7_777,
                device=args.device, validation_batch=batches[1],
            )
            ops = SubsetExpertOps(model, device=args.device)
            policy = LearnedPolicySelector(embedding_dim=batches[0].history_length)
            info = train_policy_selector(
                policy, batches[0], ops, budget=args.budget, epochs=args.policy_epochs,
                batch_size=256, seed=seed + 8_888, device=args.device,
            )
            empty = np.zeros((batches[2].episodes, args.candidate_count), dtype=bool)
            base_loss = ops.squared_error(batches[2], empty)
            selected = select_learned_policy(policy, batches[2], ops, budget=args.budget, device=args.device)
            gains = base_loss - ops.squared_error(batches[2], selected)
            cell = args.out_root / f"target{target}_seed{seed}"
            cell.mkdir(parents=True)
            np.savez_compressed(cell / "episode_gains.npz", learned_policy=gains.astype(np.float32))
            (cell / "result.json").write_text(
                json.dumps(
                    {
                        "dataset": args.dataset,
                        "target_sensor": target,
                        "seed": seed,
                        "policy": "learned_policy",
                        "mean_gain": float(np.mean(gains)),
                        "negative_transfer_rate": float(np.mean(gains < 0.0)),
                        "mean_selected_contexts": float(np.mean(selected.sum(axis=1))),
                        "policy_epochs": args.policy_epochs,
                        "final_train_reward": info["mean_reward_history"][-1],
                        "wall_seconds": time.time() - started,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"target": target, "seed": seed, "mean_gain": float(np.mean(gains))}), flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
