#!/usr/bin/env python3
"""Train a subset-capable transformer predictor on a traffic benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subset_predictor import SubsetTransformerPredictor  # noqa: E402


def build_candidate_pool(values, train_stop, target_sensor, candidate_count, seed=20260920):
    corr = np.corrcoef(values[:train_stop].T)[target_sensor]
    corr[target_sensor] = -np.inf
    order = np.argsort(-corr, kind="stable")
    high = order[: max(candidate_count // 2, 1)]
    middle_start = len(high)
    middle_stop = min(middle_start + max(candidate_count // 3, 1), len(order))
    middle = order[middle_start:middle_stop]
    remaining = np.setdiff1d(order, np.concatenate([high, middle]), assume_unique=False)
    rng = np.random.default_rng(seed)
    distractors = rng.choice(remaining, size=candidate_count - len(high) - len(middle), replace=False)
    return np.concatenate([high, middle, distractors]).astype(np.int64)


def make_arrays(values, times, target_sensor, candidate_sensors, history_length, horizon):
    anchor = np.stack([values[t - history_length:t, target_sensor] for t in times])
    candidates = np.stack([values[t - history_length:t, candidate_sensors].T for t in times])
    target = np.stack([values[t:t + horizon, target_sensor] for t in times])
    return anchor.astype(np.float32), candidates.astype(np.float32), target.astype(np.float32)


def sample_masks(batch_size, candidate_count, rng, budget=None):
    if budget is None:
        budget = candidate_count
    masks = np.zeros((batch_size, candidate_count), dtype=bool)
    sizes = rng.integers(0, budget + 1, size=batch_size)
    for row, size in enumerate(sizes):
        if size:
            masks[row, rng.choice(candidate_count, size=int(size), replace=False)] = True
    return masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "data/staeformer")
    parser.add_argument("--dataset", default="METRLA")
    parser.add_argument("--target-sensor", type=int, default=0)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-windows", type=int, default=4000)
    parser.add_argument("--val-windows", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.data_root / args.dataset / "data.npz")["data"].astype(np.float32)
    index = np.load(args.data_root / args.dataset / "index.npz")
    values = data[..., 0]
    train_starts = index["train"][:, 1]
    val_starts = index["val"][:, 1]
    train_stop = int(train_starts.max() + 1)
    mean = values[:train_stop].mean(axis=0)
    sd = values[:train_stop].std(axis=0)
    values = (values - mean) / np.maximum(sd, 1e-6)
    candidates = build_candidate_pool(values, train_stop, args.target_sensor, args.candidate_count)
    rng = np.random.default_rng(1234)
    train_times = rng.choice(train_starts, size=min(args.train_windows, len(train_starts)), replace=False)
    val_times = rng.choice(val_starts, size=min(args.val_windows, len(val_starts)), replace=False)
    train_a, train_c, train_y = make_arrays(values, train_times, args.target_sensor, candidates, args.history_length, args.horizon)
    val_a, val_c, val_y = make_arrays(values, val_times, args.target_sensor, candidates, args.history_length, args.horizon)
    model = SubsetTransformerPredictor(args.history_length, args.horizon).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_a))
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            masks = sample_masks(len(idx), args.candidate_count, rng, budget=args.candidate_count)
            anchor = torch.from_numpy(train_a[idx]).float().to(args.device)
            cand = torch.from_numpy(train_c[idx]).float().to(args.device)
            y = torch.from_numpy(train_y[idx]).float().to(args.device)
            pred = model(anchor, cand, torch.from_numpy(masks).to(args.device))
            loss = loss_fn(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(idx)
        model.eval()
        with torch.no_grad():
            val_masks = sample_masks(len(val_a), args.candidate_count, rng, budget=args.candidate_count)
            pred = model(
                torch.from_numpy(val_a).float().to(args.device),
                torch.from_numpy(val_c).float().to(args.device),
                torch.from_numpy(val_masks).to(args.device),
            )
            val_loss = float(loss_fn(pred, torch.from_numpy(val_y).float().to(args.device)).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch + 1:03d} train {total / len(train_a):.5f} val {val_loss:.5f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "history_length": args.history_length,
                "horizon": args.horizon,
                "candidate_count": args.candidate_count,
                "target_sensor": args.target_sensor,
                "candidate_sensors": candidates.tolist(),
            },
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
