#!/usr/bin/env python3
"""Can any model predict state-conditioned utility from observable features?

Fast, closed-form probe for the observability limit of the strong-backbone
setting. Builds the feature set a router can actually see at decision time
(anchor history, candidate history, candidate identity, and the expert response
including its direction and the observable base-response interaction), then
fits (a) ridge regression in closed form and (b) a small MLP, and reports the
out-of-sample R^2, the within-episode Spearman correlation and the utility of
the greedy top-4 policy they induce.

Reference points from the same cell: random selection, the learned Static
Utility router, and the standalone oracle.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def features(batch, ops, *, response: bool):
    episodes, candidates = batch.episodes, batch.candidate_count
    empty = np.zeros((episodes, candidates), dtype=bool)
    base = ops.predict(batch, empty)
    rows, columns = np.nonzero(~empty)
    delta = np.zeros((episodes, candidates, batch.horizon), dtype=np.float32)
    delta[rows, columns] = ops.pair_predict(batch, empty, rows, columns) - base[rows]
    blocks = [
        np.repeat(batch.anchor_history[:, None, :], candidates, axis=1).reshape(episodes * candidates, -1),
        batch.candidate_history.reshape(episodes * candidates, -1),
        np.broadcast_to(
            np.eye(candidates, dtype=np.float32)[None], (episodes, candidates, candidates)
        ).reshape(episodes * candidates, -1),
        np.broadcast_to(
            np.asarray(batch.candidate_sensors, dtype=np.float32)[None, :, None], (episodes, candidates, 1)
        ).reshape(episodes * candidates, -1),
    ]
    if response:
        blocks.extend([
            delta.reshape(episodes * candidates, -1),
            (np.linalg.norm(delta, axis=2) ** 2).reshape(-1, 1),
            np.sum(np.repeat(base[:, None, :], candidates, axis=1) * delta, axis=2).reshape(-1, 1),
        ])
    return np.concatenate(blocks, axis=1).astype(np.float64)


def ridge_fit_predict(train_x, train_y, test_x, penalty=1.0):
    design = np.concatenate([train_x, np.ones((len(train_x), 1))], axis=1)
    gram = design.T @ design + penalty * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ train_y)
    test_design = np.concatenate([test_x, np.ones((len(test_x), 1))], axis=1)
    return test_design @ weights


def mlp_fit_predict(train_x, train_y, test_x, *, epochs=300, hidden=128, seed=0):
    torch.manual_seed(seed)
    mean, sd = train_x.mean(0), train_x.std(0) + 1e-6
    xt = torch.from_numpy(((train_x - mean) / sd)).float()
    yt = torch.from_numpy(train_y).float()
    xt_test = torch.from_numpy(((test_x - mean) / sd)).float()
    model = torch.nn.Sequential(
        torch.nn.Linear(xt.shape[1], hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(xt).squeeze(-1), yt)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return model(xt_test).squeeze(-1).numpy()


def policy_gain(marginal, predicted, budget):
    order = np.argsort(-predicted.reshape(marginal.shape), axis=1)[:, :budget]
    return float(np.mean(np.take_along_axis(marginal, order, axis=1).sum(axis=1)))


def within_episode_spearman(marginal, predicted):
    from scipy.stats import spearmanr

    predicted = predicted.reshape(marginal.shape)
    values = [spearmanr(marginal[row], predicted[row]).statistic for row in range(marginal.shape[0])]
    return float(np.nanmean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/staeformer")
    parser.add_argument("--dataset", default="METRLA")
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--test-episodes", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ridge-only", action="store_true")
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--calibration-episodes", type=int, default=1000)
    args = parser.parse_args()

    data = load_traffic_data(args.data_root / args.dataset)
    candidates = build_candidate_pool(
        data.correlation[args.target], target_sensor=args.target, candidate_count=16, seed=20260920
    )
    rng = np.random.default_rng(args.seed + 2_000)
    train_batch = make_neural_batch(
        data, rng.choice(data.train_times, size=args.train_episodes, replace=False),
        target_sensor=args.target, candidate_sensors=candidates,
    )
    test_batch = make_neural_batch(
        data, rng.choice(data.test_times, size=args.test_episodes, replace=False),
        target_sensor=args.target, candidate_sensors=candidates,
    )
    model, _ = load_checkpoint(args.expert_root / f"seed{args.seed}" / "expert.pt", device=args.device)
    # match the gate protocol: adapt on the target's training episodes
    calibration_batch = make_neural_batch(
        data, rng.choice(data.validation_times, size=args.calibration_episodes, replace=False),
        target_sensor=args.target, candidate_sensors=candidates,
    )
    finetune_expert(
        model, train_batch, epochs=args.finetune_epochs, batch_size=256, learning_rate=3e-4,
        budget=args.budget, seed=args.seed + 7_777, device=args.device, validation_batch=calibration_batch,
    )
    ops = SubsetExpertOps(model, device=args.device)
    train_marginal = ops.candidate_marginals(train_batch, np.zeros((train_batch.episodes, 16), dtype=bool))
    test_marginal = ops.candidate_marginals(test_batch, np.zeros((test_batch.episodes, 16), dtype=bool))
    oracle_gain = policy_gain(test_marginal, test_marginal, args.budget)
    random_gain = float(np.mean(test_marginal) * args.budget)
    report = {
        "dataset": args.dataset, "target": args.target, "seed": args.seed,
        "test_episodes": test_batch.episodes,
        "oracle_static_gain": oracle_gain,
        "random_gain": random_gain,
        "marginal_std": float(np.std(test_marginal)),
        "probes": {},
    }
    for label, use_response in (("state_only", False), ("state_plus_response", True)):
        train_x = features(train_batch, ops, response=use_response)
        test_x = features(test_batch, ops, response=use_response)
        train_y = train_marginal.reshape(-1)
        for learner in ("ridge", "mlp") if not args.ridge_only else ("ridge",):
            predicted = (
                ridge_fit_predict(train_x, train_y, test_x)
                if learner == "ridge"
                else mlp_fit_predict(train_x, train_y, test_x)
            )
            residual = test_marginal.reshape(-1) - predicted
            total = test_marginal.reshape(-1) - test_marginal.mean()
            entry = {
                "r2": float(1.0 - np.sum(residual ** 2) / max(np.sum(total ** 2), 1e-12)),
                "within_episode_spearman": within_episode_spearman(test_marginal, predicted),
                "policy_gain": policy_gain(test_marginal, predicted, args.budget),
                "policy_over_oracle": policy_gain(test_marginal, predicted, args.budget) / oracle_gain,
            }
            if learner == "ridge":
                # permutation control: same pipeline, labels shuffled (guards
                # against feature/label misalignment)
                permuted = np.random.default_rng(12345).permutation(len(train_y))
                control = ridge_fit_predict(train_x, train_y[permuted], test_x)
                control_residual = test_marginal.reshape(-1) - control
                entry["permutation_r2"] = float(
                    1.0 - np.sum(control_residual ** 2) / max(np.sum(total ** 2), 1e-12)
                )
                in_sample = ridge_fit_predict(train_x, train_y, train_x)
                entry["in_sample_r2"] = float(
                    1.0 - np.sum((train_y - in_sample) ** 2) / max(np.sum((train_y - train_y.mean()) ** 2), 1e-12)
                )
            report["probes"][f"{label}_{learner}"] = entry
            print(json.dumps({"target": args.target, **report["probes"][f"{label}_{learner}"]}), flush=True)
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "observability.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "observability.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "probes"}, indent=2))


if __name__ == "__main__":
    main()
