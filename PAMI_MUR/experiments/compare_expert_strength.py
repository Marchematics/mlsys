#!/usr/bin/env python3
"""Expert-strength comparison for the strong-backbone confirmation.

Both experts see exactly the same training episodes and are evaluated on the
same held-out windows with the same mask families, so the comparison isolates
the function class: per-target ridge regression versus the subset-capable
nonlinear backbone (zero-shot and adapted).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KBS = ROOT / "KBS_MUR"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KBS / "src"))

from mur.traffic import fit_subset_expert, squared_error  # noqa: E402

from neural_expert import SubsetExpertOps, finetune_expert, load_checkpoint, sample_subset_masks  # noqa: E402
from pami_traffic import (  # noqa: E402
    build_candidate_pool,
    load_traffic_data,
    make_neural_batch,
    target_set,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/staeformer")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--targets", type=int, default=8)
    parser.add_argument("--target-list", default="",
                        help="explicit comma-separated target sensors, overriding --targets")
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--validation-episodes", type=int, default=1000)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    data = load_traffic_data(args.data_root / args.dataset)
    model, config = load_checkpoint(args.expert_root / f"seed{args.seed}" / "expert.pt", device=args.device)
    ops = SubsetExpertOps(model, device=args.device)
    if args.target_list.strip():
        targets = np.asarray([int(t) for t in args.target_list.split(",")], dtype=np.int64)
    else:
        targets = target_set(data.num_nodes, args.targets)
    rng = np.random.default_rng(args.seed + 2_000)
    rng_masks = np.random.default_rng(7)

    families = ("empty", "size2", "size4", "full")
    records = []
    for target in targets:
        target = int(target)
        pool = build_candidate_pool(
            data.correlation[target], target_sensor=target,
            candidate_count=args.candidate_count, seed=20260920,
        )
        train_times = rng.choice(data.train_times, size=args.train_episodes, replace=False)
        validation_times = rng.choice(data.validation_times, size=args.validation_episodes, replace=False)
        train_batch = make_neural_batch(data, train_times, target_sensor=target, candidate_sensors=pool)
        validation_batch = make_neural_batch(
            data, validation_times, target_sensor=target, candidate_sensors=pool
        )
        ridge = fit_subset_expert(train_batch, seed=args.seed + 3_000, repeats=3, ridge_penalty=10.0)
        adapted = copy.deepcopy(model)
        finetune_expert(
            adapted, train_batch, epochs=args.finetune_epochs, batch_size=256,
            learning_rate=3e-4, budget=args.budget, seed=args.seed + 7_777,
            device=args.device, validation_batch=validation_batch,
        )
        adapted_ops = SubsetExpertOps(adapted, device=args.device)
        masks = {
            "empty": np.zeros((validation_batch.episodes, args.candidate_count), dtype=bool),
            "full": np.ones((validation_batch.episodes, args.candidate_count), dtype=bool),
        }
        for name, size in (("size2", 2), ("size4", args.budget)):
            masks[name] = sample_subset_masks(
                rng_masks, validation_batch.episodes, args.candidate_count, budget=size,
                full_probability=0.0, empty_probability=0.0, long_probability=0.0,
            )
        record = {"target_sensor": target, "families": {}}
        for name in families:
            mask = masks[name]
            ridge_mse = float(np.mean(squared_error(validation_batch, mask, ridge)))
            zero_shot = float(np.mean(ops.squared_error(validation_batch, mask)))
            adapted_mse = float(np.mean(adapted_ops.squared_error(validation_batch, mask)))
            record["families"][name] = {
                "ridge_mse": ridge_mse,
                "neural_zero_shot_mse": zero_shot,
                "neural_adapted_mse": adapted_mse,
                "adapted_over_ridge": adapted_mse / ridge_mse if ridge_mse > 0 else float("nan"),
            }
        records.append(record)
        print(json.dumps({"target": target, **{k: round(v["adapted_over_ridge"], 3) for k, v in record["families"].items()}}), flush=True)

    summary = {"dataset": args.dataset, "seed": args.seed, "targets": len(records), "records": records}
    for name in families:
        ridge = np.mean([r["families"][name]["ridge_mse"] for r in records])
        zero = np.mean([r["families"][name]["neural_zero_shot_mse"] for r in records])
        adapted = np.mean([r["families"][name]["neural_adapted_mse"] for r in records])
        summary[f"{name}_ridge_mse"] = float(ridge)
        summary[f"{name}_neural_zero_shot_mse"] = float(zero)
        summary[f"{name}_neural_adapted_mse"] = float(adapted)
        summary[f"{name}_adapted_over_ridge"] = float(adapted / ridge)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
