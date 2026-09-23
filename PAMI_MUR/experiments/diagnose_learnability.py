#!/usr/bin/env python3
"""Is state-conditioned utility learnable from observable features at all?

Fits a strong nonparametric regressor (gradient boosting) to predict the true
standalone marginal of every candidate from everything observable at routing
time: anchor history, candidate history, candidate identity, and the expert
response (vector, magnitude, and the observable base-response interaction).
Also fits the same model with the *cached* (response-free) features only.

If even this model cannot rank candidates, the bottleneck is observability, not
router capacity; if it can, the bottleneck is the router's feature set.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KBS = ROOT / "KBS_MUR"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KBS / "src"))

from mur.traffic import fit_subset_expert, squared_error  # noqa: E402

from neural_expert import SubsetExpertOps, finetune_expert, load_checkpoint  # noqa: E402
from pami_traffic import build_candidate_pool, load_traffic_data, make_neural_batch  # noqa: E402


def build_features(batch, ops, marginals, *, include_response=True, include_identity=True):
    episodes, candidates = marginals.shape
    anchor = np.repeat(batch.anchor_history[:, None, :], candidates, axis=1)
    history = batch.candidate_history
    empty = np.zeros((episodes, candidates), dtype=bool)
    base = ops.predict(batch, empty)
    rows, columns = np.nonzero(~empty)
    delta = np.zeros((episodes, candidates, batch.horizon), dtype=np.float32)
    delta[rows, columns] = ops.pair_predict(batch, empty, rows, columns) - base[rows]
    blocks = [anchor.reshape(episodes * candidates, -1), history.reshape(episodes * candidates, -1)]
    if include_identity:
        identity = np.zeros((episodes, candidates, candidates), dtype=np.float32)
        identity[:, np.arange(candidates), np.arange(candidates)] = 1.0
        blocks.append(identity.reshape(episodes * candidates, -1))
        pool_features = np.repeat(
            np.asarray(batch.candidate_sensors, dtype=np.float32)[None, :, None], episodes, axis=0
        )
        blocks.append(pool_features.reshape(episodes * candidates, -1))
    if include_response:
        blocks.append(delta.reshape(episodes * candidates, -1))
        blocks.append(np.linalg.norm(delta, axis=2).reshape(-1, 1) ** 2)
        interaction = np.sum(np.repeat(base[:, None, :], candidates, axis=1) * delta, axis=2)
        blocks.append(interaction.reshape(-1, 1))
    return np.concatenate(blocks, axis=1), base, delta


def evaluate_policy(marginals, predicted, budget):
    order = np.argsort(-predicted, axis=1)[:, :budget]
    rows = np.arange(marginals.shape[0])[:, None]
    return float(np.mean(marginals[rows, order].sum(axis=1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/staeformer")
    parser.add_argument("--dataset", default="METRLA")
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--targets", default="0,13")
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--calibration-episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=2000)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from sklearn.ensemble import HistGradientBoostingRegressor
    from scipy.stats import spearmanr

    data = load_traffic_data(args.data_root / args.dataset)
    model, _ = load_checkpoint(args.expert_root / f"seed{args.seed}" / "expert.pt", device=args.device)
    report = {"dataset": args.dataset, "seed": args.seed, "budget": args.budget, "targets": {}}
    for target in [int(item) for item in args.targets.split(",") if item.strip()]:
        candidates = build_candidate_pool(
            data.correlation[target], target_sensor=target, candidate_count=16, seed=20260920
        )
        rng = np.random.default_rng(args.seed + 2_000)
        train_batch = make_neural_batch(
            data, rng.choice(data.train_times, size=args.train_episodes, replace=False),
            target_sensor=target, candidate_sensors=candidates,
        )
        calibration_batch = make_neural_batch(
            data, rng.choice(data.validation_times, size=args.calibration_episodes, replace=False),
            target_sensor=target, candidate_sensors=candidates,
        )
        test_batch = make_neural_batch(
            data, rng.choice(data.test_times, size=args.test_episodes, replace=False),
            target_sensor=target, candidate_sensors=candidates,
        )
        adapted = copy.deepcopy(model)
        finetune_expert(
            adapted, train_batch, epochs=args.finetune_epochs, batch_size=256, learning_rate=3e-4,
            budget=args.budget, seed=args.seed + 7_777, device=args.device, validation_batch=calibration_batch,
        )
        ops = SubsetExpertOps(adapted, device=args.device)
        empty_train = np.zeros((train_batch.episodes, 16), dtype=bool)
        empty_test = np.zeros((test_batch.episodes, 16), dtype=bool)
        train_marginal = ops.candidate_marginals(train_batch, empty_train)
        test_marginal = ops.candidate_marginals(test_batch, empty_test)
        record = {
            "oracle_static_mean": float(np.mean(np.sort(test_marginal, axis=1)[:, -args.budget:].sum(axis=1))),
            "random_mean": float(np.mean(test_marginal) * args.budget),
            "marginal_std": float(np.std(test_marginal)),
        }
        for label, kwargs in (
            ("response_free", dict(include_response=False, include_identity=True)),
            ("response_aware", dict(include_response=True, include_identity=True)),
        ):
            train_x, _, _ = build_features(train_batch, ops, train_marginal, **kwargs)
            test_x, _, _ = build_features(test_batch, ops, test_marginal, **kwargs)
            model_gb = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, max_depth=4, random_state=0, early_stopping=True, n_iter_no_change=8)
            model_gb.fit(train_x, train_marginal.reshape(-1))
            predicted = model_gb.predict(test_x).reshape(test_marginal.shape)
            residual = test_marginal - predicted
            total = test_marginal - test_marginal.mean()
            record[f"{label}_r2"] = float(1.0 - np.sum(residual ** 2) / max(np.sum(total ** 2), 1e-12))
            record[f"{label}_policy_gain"] = evaluate_policy(test_marginal, predicted, args.budget)
            correlations = [
                spearmanr(test_marginal[row], predicted[row]).statistic for row in range(0, len(test_marginal), 5)
            ]
            record[f"{label}_mean_spearman"] = float(np.nanmean(correlations))
        # How much of the oracle value does a perfect ranker of the *true*
        # marginals capture? Upper bound for any observable-feature policy.
        record["observable_ceiling_ratio"] = record["response_aware_policy_gain"] / record["oracle_static_mean"]
        record["random_ratio"] = record["random_mean"] / record["oracle_static_mean"]
        report["targets"][str(target)] = record
        print(json.dumps({"target": target, **record}), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "learnability.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
