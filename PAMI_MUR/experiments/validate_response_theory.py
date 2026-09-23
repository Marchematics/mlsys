#!/usr/bin/env python3
"""Numerical validation of the response-utility theory (PAMI addition A3).

Two checks:
  * squared loss, real forecasting expert: the identity is exact and the
    response carries the interaction term that a state-only score misses;
  * cross-entropy, controlled in-context classifier: the two-sided bound of
    Proposition 3 holds and its tightness is measured.

Both checks report out-of-sample R^2 of the same ridge head with and without
response features, which is the empirical content of Proposition 4.
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from neural_expert import (  # noqa: E402
    SubsetExpertOps,
    build_backbone,
    load_checkpoint,
    sample_subset_masks,
)
from pami_traffic import (  # noqa: E402
    build_candidate_pool,
    load_traffic_data,
    make_neural_batch,
    target_set,
)


def ridge_r2(features: np.ndarray, target: np.ndarray, *, penalty: float = 1e-2, seed: int = 0):
    """Out-of-sample R^2 of a ridge head on a random half split."""

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(target))
    cut = len(order) // 2
    train, test = order[:cut], order[cut:]
    design = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    gram = design[train].T @ design[train] + penalty * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design[train].T @ target[train])
    prediction = design[test] @ weights
    residual = target[test] - prediction
    total = target[test] - target[test].mean()
    return float(1.0 - np.sum(residual ** 2) / max(np.sum(total ** 2), 1e-12))


def squared_loss_check(args):
    data = load_traffic_data(args.data_root / args.dataset)
    model, config = load_checkpoint(Path(args.checkpoint), device=args.device)
    ops = SubsetExpertOps(model, device=args.device)
    targets = target_set(data.num_nodes, args.targets)
    rng = np.random.default_rng(args.seed)
    candidate_count = int(config.get("candidate_count", 16))
    budget = 4
    identity_error, response_penalty_error = [], []
    state_rows, response_rows, targets_out = [], [], []
    bound_pairs = 0
    for target in targets:
        target = int(target)
        pool = build_candidate_pool(
            data.correlation[target], target_sensor=target,
            candidate_count=candidate_count, seed=20260920,
        )
        times = rng.choice(data.test_times, size=args.episodes, replace=False)
        batch = make_neural_batch(data, times, target_sensor=target, candidate_sensors=pool)
        for _ in range(args.states):
            selected = np.zeros((batch.episodes, candidate_count), dtype=bool)
            sizes = rng.integers(0, budget, size=batch.episodes)
            for row, size in enumerate(sizes):
                if size:
                    selected[row, rng.choice(candidate_count, size=int(size), replace=False)] = True
            base = ops.predict(batch, selected)
            rows, columns = [], []
            for row in range(batch.episodes):
                remaining = np.flatnonzero(~selected[row])
                if remaining.size == 0:
                    continue
                for candidate in rng.choice(remaining, size=min(args.pairs_per_state, remaining.size), replace=False):
                    rows.append(row)
                    columns.append(int(candidate))
            if not rows:
                continue
            rows = np.asarray(rows, dtype=np.int64)
            columns = np.asarray(columns, dtype=np.int64)
            after = ops.pair_predict(batch, selected, rows, columns)
            delta = after - base[rows]
            y = batch.target_future[rows]
            marginal = np.mean((base[rows] - y) ** 2, axis=1) - np.mean((after - y) ** 2, axis=1)
            first = np.mean(2.0 * delta * (y - base[rows]), axis=1)
            penalty = np.mean(delta ** 2, axis=1)
            identity_error.extend(np.abs(marginal - (first - penalty)).tolist())
            response_penalty_error.extend(np.abs(penalty - np.mean(delta ** 2, axis=1)).tolist())
            bound_pairs += int(rows.size)
            selected_counts = selected[rows].sum(axis=1)
            selected_mean = np.stack([
                batch.candidate_history[row, selected[row]].mean(axis=0)
                if selected_counts[i] else np.zeros(batch.history_length)
                for i, row in enumerate(rows)
            ])
            state_rows.extend(list(np.concatenate([
                base[rows], batch.anchor_history[rows], selected_mean,
                (selected_counts / budget)[:, None],
            ], axis=1)))
            response_rows.extend(list(np.concatenate([
                delta,
                np.stack([np.mean(np.abs(delta), axis=1), np.linalg.norm(delta, axis=1),
                          np.max(np.abs(delta), axis=1)], axis=1),
            ], axis=1)))
            targets_out.extend(marginal.tolist())
    state_features = np.asarray(state_rows, dtype=np.float64)
    response_features = np.asarray(response_rows, dtype=np.float64)
    values = np.asarray(targets_out, dtype=np.float64)
    r2_state = ridge_r2(state_features, values, seed=1)
    r2_both = ridge_r2(np.concatenate([state_features, response_features], axis=1), values, seed=1)
    return {
        "family": "squared loss (real forecasting expert)",
        "samples": int(values.size),
        "max_identity_error": float(np.max(identity_error)),
        "mean_identity_error": float(np.mean(identity_error)),
        "penalty_check_max_error": float(np.max(response_penalty_error)),
        "r2_state_only": r2_state,
        "r2_state_plus_response": r2_both,
        "r2_gain": r2_both - r2_state,
        "marginal_mean": float(values.mean()),
        "marginal_std": float(values.std()),
    }


def make_icl_problem(seed: int, classes: int = 4, dimension: int = 16, candidates: int = 8):
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(classes, dimension)) * 1.5
    def sample(labels, noise=1.0):
        return means[labels] + noise * rng.normal(size=(len(labels), dimension))
    train_labels = rng.integers(0, classes, size=4000)
    train_x = sample(train_labels)
    train_y = np.eye(classes, dtype=np.float32)[train_labels]
    return {
        "means": means, "train_x": train_x.astype(np.float32), "train_y": train_y,
        "classes": classes, "dimension": dimension, "candidates": candidates, "rng": rng,
    }


class InContextClassifier(torch.nn.Module):
    def __init__(self, dimension: int, classes: int, d_model: int = 64):
        super().__init__()
        self.embed = torch.nn.Linear(dimension, d_model)
        self.label_embed = torch.nn.Embedding(classes, d_model)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128, dropout=0.0,
            batch_first=True, activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=2)
        self.head = torch.nn.Linear(d_model, classes)

    def forward(self, query, demonstrations, labels, selected_mask):
        tokens = torch.cat(
            [self.embed(query).unsqueeze(1), self.embed(demonstrations) + self.label_embed(labels)],
            dim=1,
        )
        padding = torch.cat(
            [torch.zeros(query.shape[0], 1, dtype=torch.bool, device=query.device), ~selected_mask],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        return self.head(encoded[:, 0])


def cross_entropy_check(args):
    problem = make_icl_problem(args.seed)
    classes, dimension, candidates = problem["classes"], problem["dimension"], problem["candidates"]
    rng = problem["rng"]
    model = InContextClassifier(dimension, classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_x = torch.from_numpy(problem["train_x"])
    train_labels = torch.from_numpy(np.argmax(problem["train_y"], axis=1)).long()
    for step in range(args.steps):
        index = torch.from_numpy(rng.integers(0, len(train_x), size=args.batch_size))
        query = train_x[index]
        demos = train_x[index][:, None, :].expand(len(index), candidates, dimension).clone()
        demo_labels = train_labels[index][:, None].expand(len(index), candidates).clone()
        masks = torch.from_numpy(sample_subset_masks(rng, len(index), candidates, budget=4))
        logits = model(query, demos, demo_labels, masks)
        loss = loss_fn(logits, train_labels[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()

    def logits_for(query, demos, labels, mask):
        with torch.no_grad():
            return model(
                torch.from_numpy(query).float(),
                torch.from_numpy(demos).float(),
                torch.from_numpy(labels).long(),
                torch.from_numpy(mask),
            ).numpy()

    rng = np.random.default_rng(args.seed + 5)
    state_rows, response_rows, values, lower_gap, upper_gap = [], [], [], [], []
    for _ in range(args.states):
        query_labels = rng.integers(0, classes, size=args.query_batch)
        query = problem["means"][query_labels] + rng.normal(size=(args.query_batch, dimension))
        demo_labels = rng.integers(0, classes, size=(args.query_batch, candidates))
        demos = problem["means"][demo_labels] + rng.normal(size=(args.query_batch, candidates, dimension))
        selected = np.zeros((args.query_batch, candidates), dtype=bool)
        sizes = rng.integers(0, 4, size=args.query_batch)
        for row, size in enumerate(sizes):
            if size:
                selected[row, rng.choice(candidates, size=int(size), replace=False)] = True
        base_logits = logits_for(query, demos, demo_labels, selected)
        one_hot = np.eye(classes)[query_labels]
        base_probability = np.exp(base_logits - base_logits.max(axis=1, keepdims=True))
        base_probability /= base_probability.sum(axis=1, keepdims=True)
        base_loss = -np.log(np.maximum(base_probability[np.arange(args.query_batch), query_labels], 1e-12))
        for candidate in range(candidates):
            eligible = ~selected[:, candidate]
            if not eligible.any():
                continue
            augmented = selected.copy()
            augmented[eligible, candidate] = True
            after_logits = logits_for(query, demos, demo_labels, augmented)
            probability = np.exp(after_logits - after_logits.max(axis=1, keepdims=True))
            probability /= probability.sum(axis=1, keepdims=True)
            after_loss = -np.log(np.maximum(probability[np.arange(args.query_batch), query_labels], 1e-12))
            delta = after_logits - base_logits
            marginal = float(np.mean(base_loss[eligible] - after_loss[eligible]))
            first = float(np.mean(np.sum((one_hot - base_probability)[eligible] * delta[eligible], axis=1)))
            magnitude = float(np.mean(np.sum(delta[eligible] ** 2, axis=1)))
            values.append(marginal)
            lower_gap.append(marginal - (first - 0.25 * magnitude))
            upper_gap.append(first - marginal)
            state_rows.append(
                np.concatenate([
                    base_probability[eligible].mean(axis=0),
                    selected[eligible].mean(axis=0) / max(candidates, 1),
                ])
            )
            response_rows.append(
                np.concatenate([
                    delta[eligible].mean(axis=0), [np.mean(np.linalg.norm(delta[eligible], axis=1)),
                                                   magnitude],
                ])
            )
    values = np.asarray(values)
    state_features = np.asarray(state_rows, dtype=np.float64)
    response_features = np.asarray(response_rows, dtype=np.float64)
    r2_state = ridge_r2(state_features, values, seed=3)
    r2_both = ridge_r2(np.concatenate([state_features, response_features], axis=1), values, seed=3)
    lower = np.asarray(lower_gap)
    upper = np.asarray(upper_gap)
    tightness = upper / np.maximum(upper + lower, 1e-12)
    return {
        "family": "cross-entropy (frozen in-context classifier)",
        "samples": int(values.size),
        "bound_violations": int(np.sum(lower < -1e-9) + np.sum(upper < -1e-9)),
        "min_lower_gap": float(lower.min()),
        "min_upper_gap": float(upper.min()),
        "tightness_mean": float(tightness.mean()),
        "tightness_quantiles": [float(np.quantile(tightness, q)) for q in (0.05, 0.5, 0.95)],
        "r2_state_only": r2_state,
        "r2_state_plus_response": r2_both,
        "r2_gain": r2_both - r2_state,
        "marginal_mean": float(values.mean()),
        "marginal_std": float(values.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["squared", "ce", "both"], default="both")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/staeformer")
    parser.add_argument("--dataset", default="METRLA")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--pairs-per-state", type=int, default=3)
    parser.add_argument("--targets", type=int, default=4)
    parser.add_argument("--query-batch", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}
    if args.mode in ("squared", "both"):
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for the squared-loss check")
        report["checks"]["squared"] = squared_loss_check(args)
    if args.mode in ("ce", "both"):
        report["checks"]["cross_entropy"] = cross_entropy_check(args)
    (args.out / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
