#!/usr/bin/env python3
"""Audit Electricity for oracle headroom before any representation changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mur.calibration import SymmetricUtilityCalibrator  # noqa: E402
from mur.electricity import (  # noqa: E402
    expert_prediction,
    load_electricity_data,
    observed_candidate_marginals,
    pack_selected,
    sample_context_batch,
    squared_error,
)
from mur.metrics import decision_metrics  # noqa: E402
from mur.synthetic_experiment import (  # noqa: E402
    predict_pair_dataset,
    sample_pair_dataset,
    select_mur,
    train_utility_model,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_oracle_static(batch, budget: int) -> np.ndarray:
    empty = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    marginal = observed_candidate_marginals(batch, empty)
    order = np.argsort(-marginal, axis=1, kind="stable")[:, :budget]
    selected = empty.copy()
    for row in range(batch.episodes):
        for candidate in order[row]:
            if marginal[row, candidate] <= 0.0:
                break
            selected[row, candidate] = True
    return selected


def select_oracle_greedy(batch, budget: int) -> np.ndarray:
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for _ in range(budget):
        marginal = observed_candidate_marginals(batch, selected)
        choice = np.argmax(marginal, axis=1)
        active = marginal[np.arange(batch.episodes), choice] > 0.0
        selected[np.arange(batch.episodes)[active], choice[active]] = True
    return selected


def prediction_deltas(batch, selected: np.ndarray) -> np.ndarray:
    mask = np.asarray(selected, dtype=bool)
    before = expert_prediction(batch, mask)
    output = np.zeros((batch.episodes, batch.candidate_count), dtype=np.float32)
    for candidate in range(batch.candidate_count):
        eligible = ~mask[:, candidate]
        if not np.any(eligible):
            continue
        augmented = mask.copy()
        augmented[eligible, candidate] = True
        output[eligible, candidate] = (
            expert_prediction(batch, augmented)[eligible] - before[eligible]
        )
    return output


def summarize(batch, selected, oracle) -> dict[str, float]:
    base = squared_error(batch, np.zeros_like(selected))
    routed = squared_error(batch, selected)
    oracle_loss = squared_error(batch, oracle)
    return decision_metrics(base, routed, oracle_loss, np.sum(selected, axis=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", type=Path, default=ROOT.parent / "data/uci_electricity/electricity_float32.npz")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--train-episodes", type=int, default=1024)
    parser.add_argument("--calibration-episodes", type=int, default=512)
    parser.add_argument("--test-episodes", type=int, default=1024)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--same-client-fraction", type=float, default=0.5)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--context-points", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    data = load_electricity_data(
        str(args.cache_path),
        active_fraction_threshold=0.90,
        split_seed=20260719,
        train_client_count=160,
        validation_client_count=60,
        train_week_stop=32,
        harmonics=4,
    )

    def make_batch(clients, start, stop, episodes, offset):
        return sample_context_batch(
            data,
            clients=clients,
            week_start=start,
            week_stop=stop,
            episodes=episodes,
            candidate_count=args.candidate_count,
            same_client_fraction=args.same_client_fraction,
            context_points=args.context_points,
            ridge_penalty=1.0,
            rng=np.random.default_rng(args.seed + offset),
        )

    train = make_batch(data.train_clients, 0, 32, args.train_episodes, 10_000)
    calibration = make_batch(data.validation_clients, 32, 40, args.calibration_episodes, 20_000)
    test = make_batch(data.test_clients, 40, 52, args.test_episodes, 30_000)
    oracle_static = select_oracle_static(test, args.budget)
    oracle_greedy = select_oracle_greedy(test, args.budget)
    base = np.zeros_like(oracle_greedy)
    static_gain = float(np.mean(squared_error(test, base) - squared_error(test, oracle_static)))
    greedy_gain = float(np.mean(squared_error(test, base) - squared_error(test, oracle_greedy)))

    full_pairs = sample_pair_dataset(
        train,
        max_budget=args.budget,
        states_per_episode=4,
        seed=args.seed + 40_000,
        static_only=False,
        marginal_fn=observed_candidate_marginals,
        pack_fn=pack_selected,
        delta_fn=prediction_deltas,
    )
    full_model = train_utility_model(
        full_pairs,
        max_budget=args.budget,
        seed=args.seed + 50_000,
        device=args.device,
        epochs=args.epochs,
    )
    calibration_pairs = sample_pair_dataset(
        calibration,
        max_budget=args.budget,
        states_per_episode=4,
        seed=args.seed + 60_000,
        static_only=False,
        marginal_fn=observed_candidate_marginals,
        pack_fn=pack_selected,
        delta_fn=prediction_deltas,
    )
    calibration_prediction = predict_pair_dataset(full_model, calibration_pairs, device=args.device)
    calibrator = SymmetricUtilityCalibrator.fit(
        calibration_prediction, calibration_pairs.target, alpha=0.1
    )
    exact_delta_selected = select_mur(
        full_model,
        test,
        budget=args.budget,
        device=args.device,
        pack_fn=pack_selected,
        delta_fn=lambda batch, selected: prediction_deltas(batch, selected),
    )
    metrics = {
        "base_only": summarize(test, base, oracle_greedy),
        "oracle_static": summarize(test, oracle_static, oracle_greedy),
        "oracle_greedy": summarize(test, oracle_greedy, oracle_greedy),
        "mur_exact_delta": summarize(test, exact_delta_selected, oracle_greedy),
    }
    result = {
        "run_id": "R073_electricity_oracle_audit",
        "status": "audit",
        "config": {
            "seed": args.seed,
            "train_episodes": args.train_episodes,
            "calibration_episodes": args.calibration_episodes,
            "test_episodes": args.test_episodes,
            "candidate_count": args.candidate_count,
            "same_client_fraction": args.same_client_fraction,
            "budget": args.budget,
            "context_points": args.context_points,
            "epochs": args.epochs,
            "device": args.device,
            "calibration_alpha": 0.1,
            "calibration_radius": calibrator.radius,
        },
        "metrics": metrics,
        "oracle_quantities": {
            "gain_base_to_oracle_static": static_gain,
            "gain_base_to_oracle_greedy": greedy_gain,
            "state_conditioning_headroom": (
                (greedy_gain - static_gain) / greedy_gain if greedy_gain > 0.0 else float("nan")
            ),
            "exact_delta_gain": metrics["mur_exact_delta"]["mean_prediction_gain"],
        },
        "wall_seconds": time.time() - started,
        "provenance": {
            "cache_path": str(args.cache_path),
            "cache_sha256": sha256(args.cache_path),
            "electricity_module_sha256": sha256(ROOT / "src/mur/electricity.py"),
            "synthetic_experiment_module_sha256": sha256(ROOT / "src/mur/synthetic_experiment.py"),
        },
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
