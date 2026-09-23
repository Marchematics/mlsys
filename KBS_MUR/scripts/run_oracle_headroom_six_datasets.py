#!/usr/bin/env python3
"""Oracle state-conditioning headroom audit on six standard traffic datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mur.traffic import (  # noqa: E402
    fit_subset_expert,
    make_batch,
    observed_candidate_marginals,
    squared_error,
)


DATASETS = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]


def select_oracle_static(batch, expert, budget: int) -> np.ndarray:
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


def select_oracle_greedy(batch, expert, budget: int) -> np.ndarray:
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for _ in range(budget):
        marginal = observed_candidate_marginals(batch, selected, expert)
        choice = np.argmax(marginal, axis=1)
        active = marginal[np.arange(batch.episodes), choice] > 0.0
        selected[np.arange(batch.episodes)[active], choice[active]] = True
    return selected


def summarize(batch, expert, selected: np.ndarray, oracle: np.ndarray) -> dict[str, float]:
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


def build_candidate_pool(
    values: np.ndarray,
    *,
    train_stop: int,
    target_sensor: int,
    candidate_count: int,
    seed: int,
) -> np.ndarray:
    corr = np.corrcoef(values[:train_stop].T)[target_sensor]
    corr[target_sensor] = -np.inf
    order = np.argsort(-corr, kind="stable")
    high = order[: max(candidate_count // 2, 1)]
    middle_start = len(high)
    middle_stop = min(middle_start + max(candidate_count // 3, 1), len(order))
    middle = order[middle_start:middle_stop]
    remaining = np.setdiff1d(order, np.concatenate([high, middle]), assume_unique=False)
    rng = np.random.default_rng(seed)
    distractor_count = candidate_count - len(high) - len(middle)
    distractors = rng.choice(remaining, size=distractor_count, replace=False)
    return np.concatenate([high, middle, distractors]).astype(np.int64)


def run_seed(
    dataset_dir: Path,
    *,
    seed: int,
    candidate_count: int,
    budget: int,
    history_length: int,
    horizon: int,
    train_episodes: int,
    test_episodes: int,
    expert_repeats: int,
    ridge_penalty: float,
) -> dict:
    data = np.load(dataset_dir / "data.npz")["data"].astype(np.float32)
    index = np.load(dataset_dir / "index.npz")
    values = data[..., 0]
    train_starts = index["train"][:, 1]
    test_starts = index["test"][:, 1]
    train_stop = int(train_starts.max() + 1)
    # Per-sensor standardization from the training segment.
    mean = values[:train_stop].mean(axis=0)
    sd = values[:train_stop].std(axis=0)
    values = (values - mean) / np.maximum(sd, 1e-6)
    target_sensor = 0
    candidate_sensors = build_candidate_pool(
        values,
        train_stop=train_stop,
        target_sensor=target_sensor,
        candidate_count=candidate_count,
        seed=20260920,
    )
    rng = np.random.default_rng(seed + 2_000)
    train_times = rng.choice(train_starts, size=min(train_episodes, len(train_starts)), replace=False)
    test_times = rng.choice(test_starts, size=min(test_episodes, len(test_starts)), replace=False)
    train_batch = make_batch(
        values,
        train_times,
        target_sensor=target_sensor,
        candidate_sensors=candidate_sensors,
        history_length=history_length,
        horizon=horizon,
    )
    test_batch = make_batch(
        values,
        test_times,
        target_sensor=target_sensor,
        candidate_sensors=candidate_sensors,
        history_length=history_length,
        horizon=horizon,
    )
    expert = fit_subset_expert(
        train_batch,
        seed=seed + 3_000,
        repeats=expert_repeats,
        ridge_penalty=ridge_penalty,
    )
    oracle_static = select_oracle_static(test_batch, expert, budget)
    oracle_greedy = select_oracle_greedy(test_batch, expert, budget)
    static_gain = summarize(test_batch, expert, oracle_static, oracle_greedy)["mean_prediction_gain"]
    greedy_gain = summarize(test_batch, expert, oracle_greedy, oracle_greedy)["mean_prediction_gain"]
    return {
        "dataset": dataset_dir.name,
        "seed": seed,
        "target_sensor": target_sensor,
        "candidate_sensors": candidate_sensors.tolist(),
        "candidate_count": candidate_count,
        "budget": budget,
        "history_length": history_length,
        "horizon": horizon,
        "train_episodes": int(train_times.size),
        "test_episodes": int(test_times.size),
        "gain_base_to_oracle_static": float(static_gain),
        "gain_base_to_oracle_greedy": float(greedy_gain),
        "state_conditioning_headroom": float((greedy_gain - static_gain) / greedy_gain)
        if greedy_gain > 0.0
        else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data" / "staeformer")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--test-episodes", type=int, default=2000)
    parser.add_argument("--expert-repeats", type=int, default=3)
    parser.add_argument("--ridge-penalty", type=float, default=10.0)
    args = parser.parse_args()
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)
    records = []
    for dataset in datasets:
        dataset_dir = args.data_root / dataset
        for seed in seeds:
            started = time.time()
            record = run_seed(
                dataset_dir,
                seed=seed,
                candidate_count=args.candidate_count,
                budget=args.budget,
                history_length=args.history_length,
                horizon=args.horizon,
                train_episodes=args.train_episodes,
                test_episodes=args.test_episodes,
                expert_repeats=args.expert_repeats,
                ridge_penalty=args.ridge_penalty,
            )
            record["wall_seconds"] = time.time() - started
            records.append(record)
            print(json.dumps(record, indent=2))
    (args.out_root / "records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    rows = ["dataset,seed,gain_base_to_oracle_static,gain_base_to_oracle_greedy,state_conditioning_headroom"]
    for record in records:
        rows.append(
            f"{record['dataset']},{record['seed']},{record['gain_base_to_oracle_static']:.8f},"
            f"{record['gain_base_to_oracle_greedy']:.8f},{record['state_conditioning_headroom']:.8f}"
        )
    (args.out_root / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    aggregate = {}
    for dataset in datasets:
        subset = [r for r in records if r["dataset"] == dataset]
        aggregate[dataset] = {
            "oracle_static_gain_mean": float(np.mean([r["gain_base_to_oracle_static"] for r in subset])),
            "oracle_static_gain_std": float(np.std([r["gain_base_to_oracle_static"] for r in subset], ddof=1)),
            "oracle_greedy_gain_mean": float(np.mean([r["gain_base_to_oracle_greedy"] for r in subset])),
            "oracle_greedy_gain_std": float(np.std([r["gain_base_to_oracle_greedy"] for r in subset], ddof=1)),
            "state_conditioning_headroom_mean": float(np.mean([r["state_conditioning_headroom"] for r in subset])),
            "state_conditioning_headroom_std": float(np.std([r["state_conditioning_headroom"] for r in subset], ddof=1)),
        }
    (args.out_root / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_root": str(args.out_root), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
