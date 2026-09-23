#!/usr/bin/env python3
"""METR-LA oracle gate: test external utility and state-conditioning headroom."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mur.traffic import (
    TrafficBatch,
    expert_prediction,
    fit_subset_expert,
    make_batch,
    observed_candidate_marginals,
    squared_error,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def select_oracle_static(batch: TrafficBatch, expert, budget: int) -> np.ndarray:
    empty = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    marginal = observed_candidate_marginals(batch, empty, expert)
    order = np.argsort(-marginal, axis=1, kind="stable")[:, :budget]
    selected = empty.copy()
    for row in range(batch.episodes):
        for candidate in order[row]:
            if marginal[row, candidate] <= 0.0:
                break
            selected[row, candidate] = True
    return selected


def select_oracle_greedy(batch: TrafficBatch, expert, budget: int) -> np.ndarray:
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for _ in range(budget):
        marginal = observed_candidate_marginals(batch, selected, expert)
        choice = np.argmax(marginal, axis=1)
        active = marginal[np.arange(batch.episodes), choice] > 0.0
        selected[np.arange(batch.episodes)[active], choice[active]] = True
    return selected


def summarize(batch: TrafficBatch, expert, selected: np.ndarray, oracle: np.ndarray) -> dict[str, float]:
    base = squared_error(batch, np.zeros_like(selected), expert)
    routed = squared_error(batch, selected, expert)
    oracle_loss = squared_error(batch, oracle, expert)
    gain = base - routed
    oracle_gain = base - oracle_loss
    positive = oracle_gain > 0.0
    return {
        "mean_prediction_gain": float(np.mean(gain)),
        "negative_transfer_rate": float(np.mean(routed > base)),
        "mean_utility_recovery": float(np.mean(gain[positive] / oracle_gain[positive]))
        if np.any(positive)
        else float("nan"),
        "mean_selected_contexts": float(np.mean(np.sum(selected, axis=1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--target-sensor-index", type=int, default=0)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--validation-episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=2000)
    parser.add_argument("--expert-repeats", type=int, default=3)
    parser.add_argument("--ridge-penalty", type=float, default=10.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("train-fraction must lie in (0,1)")
    if not 0.0 < args.validation_fraction < 1.0 - args.train_fraction:
        raise ValueError("validation-fraction leaves no test split")

    frame = pd.read_csv(args.data_path)
    if "date" not in frame.columns:
        raise ValueError("traffic CSV must contain a date column")
    sensor_columns = [column for column in frame.columns if column not in {"date", "OT"}]
    values = frame[sensor_columns].to_numpy(dtype=np.float32)
    if np.any(~np.isfinite(values)):
        raise ValueError("traffic data contains non-finite values")
    train_stop = int(values.shape[0] * args.train_fraction)
    validation_stop = int(values.shape[0] * (args.train_fraction + args.validation_fraction))
    target = args.target_sensor_index
    if not 0 <= target < values.shape[1]:
        raise ValueError("target sensor index is out of range")

    train_mean = values[:train_stop].mean(axis=0)
    train_sd = values[:train_stop].std(axis=0)
    values = (values - train_mean) / np.maximum(train_sd, 1e-6)
    corr = np.corrcoef(values[:train_stop].T)[target]
    corr[target] = -np.inf
    order = np.argsort(-corr, kind="stable")
    high = order[: max(args.candidate_count // 2, 1)]
    middle_start = len(high)
    middle_stop = min(middle_start + max(args.candidate_count // 3, 1), len(order))
    middle = order[middle_start:middle_stop]
    remaining = np.setdiff1d(order, np.concatenate([high, middle]), assume_unique=False)
    candidate_construction_seed = 20260920
    rng = np.random.default_rng(candidate_construction_seed)
    distractor_count = args.candidate_count - len(high) - len(middle)
    distractors = rng.choice(remaining, size=distractor_count, replace=False)
    candidate_sensors = np.concatenate([high, middle, distractors]).astype(np.int64)
    if len(np.unique(candidate_sensors)) != args.candidate_count:
        raise RuntimeError("candidate sensor construction produced duplicates")

    def sample_times(start: int, stop: int, count: int, offset: int) -> np.ndarray:
        low = start + args.history_length
        high_time = stop - args.horizon
        if high_time <= low:
            raise ValueError("split is too short for the requested windows")
        return rng_for_times.integers(low, high_time, size=count)

    rng_for_times = np.random.default_rng(args.seed + 2_000)
    train_times = sample_times(0, train_stop, args.train_episodes, 0)
    validation_times = sample_times(train_stop, validation_stop, args.validation_episodes, 1)
    test_times = sample_times(validation_stop, values.shape[0], args.test_episodes, 2)
    train_batch = make_batch(
        values,
        train_times,
        target_sensor=target,
        candidate_sensors=candidate_sensors,
        history_length=args.history_length,
        horizon=args.horizon,
    )
    validation_batch = make_batch(
        values,
        validation_times,
        target_sensor=target,
        candidate_sensors=candidate_sensors,
        history_length=args.history_length,
        horizon=args.horizon,
    )
    test_batch = make_batch(
        values,
        test_times,
        target_sensor=target,
        candidate_sensors=candidate_sensors,
        history_length=args.history_length,
        horizon=args.horizon,
    )
    expert = fit_subset_expert(
        train_batch,
        seed=args.seed + 3_000,
        repeats=args.expert_repeats,
        ridge_penalty=args.ridge_penalty,
    )
    oracle_static = select_oracle_static(test_batch, expert, args.budget)
    oracle_greedy = select_oracle_greedy(test_batch, expert, args.budget)
    base = np.zeros_like(oracle_greedy)
    metrics = {
        "base_only": summarize(test_batch, expert, base, oracle_greedy),
        "oracle_static": summarize(test_batch, expert, oracle_static, oracle_greedy),
        "oracle_greedy": summarize(test_batch, expert, oracle_greedy, oracle_greedy),
    }
    base_gain = metrics["oracle_greedy"]["mean_prediction_gain"]
    static_gain = metrics["oracle_static"]["mean_prediction_gain"]
    result = {
        "run_id": "R074_traffic_oracle_gate",
        "status": "oracle_gate",
        "config": {
            "seed": args.seed,
            "target_sensor_index": target,
            "candidate_sensors": candidate_sensors.tolist(),
            "candidate_construction_seed": candidate_construction_seed,
            "candidate_sensor_correlations": corr[candidate_sensors].tolist(),
            "candidate_count": args.candidate_count,
            "budget": args.budget,
            "history_length": args.history_length,
            "horizon": args.horizon,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "train_episodes": args.train_episodes,
            "validation_episodes": args.validation_episodes,
            "test_episodes": args.test_episodes,
            "expert_repeats": args.expert_repeats,
            "ridge_penalty": args.ridge_penalty,
            "sensor_count": values.shape[1],
            "observation_count": values.shape[0],
        },
        "metrics": metrics,
        "oracle_quantities": {
            "gain_base_to_oracle_static": static_gain,
            "gain_base_to_oracle_greedy": base_gain,
            "state_conditioning_headroom": (base_gain - static_gain) / base_gain
            if base_gain > 0.0
            else float("nan"),
        },
        "wall_seconds": time.time() - started,
        "provenance": {
            "data_path": str(args.data_path),
            "data_sha256": sha256(args.data_path),
            "traffic_module_sha256": sha256(ROOT / "src" / "mur" / "traffic.py"),
        },
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
