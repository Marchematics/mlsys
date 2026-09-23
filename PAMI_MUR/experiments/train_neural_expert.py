#!/usr/bin/env python3
"""Train a subset-capable nonlinear expert on one traffic benchmark.

The expert is trained once per (dataset, seed) on the training split only:
candidate pools and normalisation come from the training split, and every
training sample uses a random subset of the candidate pool so that arbitrary
subsets are in distribution at routing time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from neural_expert import (  # noqa: E402
    SubsetExpertOps,
    build_backbone,
    sample_subset_masks,
    save_checkpoint,
)
from pami_traffic import load_traffic_data, pools_for_targets, target_set  # noqa: E402


def build_samples(values: np.ndarray, nodes: np.ndarray, times: np.ndarray, *, history: int, horizon: int):
    steps = times[:, None] - history + np.arange(history)[None, :]
    anchor = values[steps, nodes[:, None]]
    future_steps = times[:, None] + np.arange(horizon)[None, :]
    target = values[future_steps, nodes[:, None]]
    return anchor.astype(np.float32), target.astype(np.float32), steps.astype(np.int64)


def candidate_tensor(values: np.ndarray, steps: np.ndarray, candidate_nodes: np.ndarray) -> np.ndarray:
    """values indexed as (batch, history, candidates) -> (batch, candidates, history)."""

    block = values[steps[:, :, None], candidate_nodes[:, None, :]]
    return np.ascontiguousarray(np.transpose(block, (0, 2, 1)), dtype=np.float32)


def evaluate_validation(
    model,
    ops: SubsetExpertOps,
    values: np.ndarray,
    nodes: np.ndarray,
    times: np.ndarray,
    pools: dict[int, np.ndarray],
    *,
    candidate_count: int,
    history_length: int,
    horizon: int,
):
    """MSE/MAE of the expert on fixed mask families over the validation split."""

    from pami_traffic import NeuralTrafficBatch

    rng = np.random.default_rng(7)
    families = {
        "empty": lambda n: np.zeros((n, candidate_count), dtype=bool),
        "full": lambda n: np.ones((n, candidate_count), dtype=bool),
        "size2": lambda n: sample_subset_masks(
            rng, n, candidate_count, budget=2, full_probability=0.0,
            empty_probability=0.0, long_probability=0.0,
        ),
        "size4": lambda n: sample_subset_masks(
            rng, n, candidate_count, budget=4, full_probability=0.0,
            empty_probability=0.0, long_probability=0.0,
        ),
    }
    squared = {name: [] for name in families}
    absolute = {name: [] for name in families}
    for node in np.unique(nodes):
        rows = np.flatnonzero(nodes == node)
        pool = pools[int(node)]
        anchor, target, steps = build_samples(
            values, nodes[rows], times[rows], history=history_length, horizon=horizon
        )
        candidate = candidate_tensor(
            values, steps, np.broadcast_to(pool, (len(rows), candidate_count))
        )
        batch = NeuralTrafficBatch(
            anchor_history=anchor,
            candidate_history=candidate,
            target_future=target,
            target_sensor=int(node),
            candidate_sensors=pool,
            time_index=times[rows].astype(np.int64),
        )
        for name, factory in families.items():
            prediction = ops.predict_masks(batch, factory(len(rows))[None])[0]
            error = prediction - batch.target_future
            squared[name].append(float(np.mean(error ** 2)))
            absolute[name].append(float(np.mean(np.abs(error))))
    results = {
        name: {"mse": float(np.mean(squared[name])), "mae": float(np.mean(absolute[name]))}
        for name in families
    }
    results["average"] = {
        "mse": float(np.mean([results[name]["mse"] for name in families])),
        "mae": float(np.mean([results[name]["mae"] for name in families])),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "data/staeformer")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--backbone", default="subset_transformer")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--windows-per-node", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    args.out.mkdir(parents=True)
    started = time.time()

    data = load_traffic_data(
        args.data_root / args.dataset,
        history_length=args.history_length,
        horizon=args.horizon,
    )
    nodes = np.arange(data.num_nodes, dtype=np.int64)
    pools = pools_for_targets(data, nodes, candidate_count=args.candidate_count, seed=20260920)
    pool_matrix = np.stack([pools[int(node)] for node in nodes])

    rng = np.random.default_rng(args.seed + 11)
    train_nodes, train_times = [], []
    for node in nodes:
        times = rng.choice(data.train_times, size=args.windows_per_node, replace=False)
        train_nodes.append(np.full(args.windows_per_node, node, dtype=np.int64))
        train_times.append(times)
    train_nodes = np.concatenate(train_nodes)
    train_times = np.concatenate(train_times)

    validation_targets = target_set(data.num_nodes, 32)
    rng_val = np.random.default_rng(20260920)
    val_nodes = np.repeat(validation_targets, 32)
    val_times = np.concatenate(
        [rng_val.choice(data.validation_times, size=32, replace=False) for _ in validation_targets]
    )

    torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    backbone_kwargs = {"d_model": args.d_model}
    if args.backbone == "subset_transformer":
        backbone_kwargs.update(num_layers=args.num_layers, nhead=args.num_heads)
    model = build_backbone(
        args.backbone,
        data.num_nodes,
        args.history_length,
        args.horizon,
        **backbone_kwargs,
    ).to(args.device)
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = torch.nn.MSELoss()

    ops = SubsetExpertOps(model, device=args.device)
    best = {"score": float("inf"), "epoch": -1, "state": None}
    history = []
    train_rng = np.random.default_rng(args.seed + 21)
    for epoch in range(args.epochs):
        model.train()
        order = train_rng.permutation(len(train_nodes))
        total, seen = 0.0, 0
        epoch_started = time.time()
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            node_block = train_nodes[index]
            anchor, target, steps = build_samples(
                data.values, node_block, train_times[index],
                history=args.history_length, horizon=args.horizon,
            )
            candidate = candidate_tensor(
                data.values, steps, pool_matrix[node_block]
            )
            masks = sample_subset_masks(
                train_rng, len(index), args.candidate_count, budget=args.budget
            )
            time_block = train_times[index]
            time_features = torch.from_numpy(
                np.stack([time_block % 288, (time_block // 288) % 7], axis=1)
            ).to(args.device)
            prediction = model(
                torch.from_numpy(anchor).to(args.device),
                torch.from_numpy(candidate).to(args.device),
                torch.from_numpy(masks).to(args.device),
                torch.from_numpy(node_block).to(args.device),
                torch.from_numpy(pool_matrix[node_block]).to(args.device),
                time_features,
            )
            loss = loss_fn(prediction, torch.from_numpy(target).to(args.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(index)
            seen += len(index)
        scheduler.step()
        model.eval()
        validation = evaluate_validation(
            model, ops, data.values, val_nodes, val_times, pools,
            candidate_count=args.candidate_count,
            history_length=args.history_length,
            horizon=args.horizon,
        )
        score = validation["average"]["mse"]
        history.append(
            {
                "epoch": epoch,
                "train_mse": total / max(seen, 1),
                "validation": validation,
                "seconds": time.time() - epoch_started,
            }
        )
        print(
            f"[{args.dataset} seed={args.seed}] epoch {epoch + 1:03d} "
            f"train {total / max(seen, 1):.5f} val_avg {score:.5f} "
            f"(empty {validation['empty']['mse']:.5f} full {validation['full']['mse']:.5f})",
            flush=True,
        )
        if score < best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        elif epoch - best["epoch"] >= args.patience:
            print(f"early stop at epoch {epoch + 1}", flush=True)
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    config = {
        "dataset": args.dataset,
        "seed": args.seed,
        "backbone": args.backbone,
        "num_nodes": data.num_nodes,
        "history_length": args.history_length,
        "horizon": args.horizon,
        "candidate_count": args.candidate_count,
        "budget": args.budget,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "windows_per_node": args.windows_per_node,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "parameters": parameter_count,
        "backbone_kwargs": backbone_kwargs,
        "candidate_pool_seed": 20260920,
    }
    save_checkpoint(args.out / "expert.pt", model, backbone=args.backbone, config=config, history=history)
    summary = {
        "config": config,
        "best_epoch": best["epoch"],
        "best_validation_average_mse": best["score"],
        "final_validation": history[best["epoch"]]["validation"] if best["epoch"] >= 0 else None,
        "wall_seconds": time.time() - started,
        "history": history,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("best_epoch", "best_validation_average_mse", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
