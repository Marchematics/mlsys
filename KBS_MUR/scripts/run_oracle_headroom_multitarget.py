#!/usr/bin/env python3
"""Multi-target oracle headroom audit on standard traffic datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mur.traffic import fit_subset_expert, make_batch, observed_candidate_marginals, squared_error  # noqa: E402


def build_candidate_pool(corr_row, *, target_sensor, candidate_count, seed):
    corr = np.asarray(corr_row, dtype=float).copy()
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


def select_oracle_static(batch, expert, budget):
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


def select_oracle_greedy(batch, expert, budget):
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for _ in range(budget):
        marginal = observed_candidate_marginals(batch, selected, expert)
        choice = np.argmax(marginal, axis=1)
        active = marginal[np.arange(batch.episodes), choice] > 0.0
        selected[np.arange(batch.episodes)[active], choice[active]] = True
    return selected


def target_set(num_nodes: int, count: int) -> np.ndarray:
    if count >= num_nodes:
        return np.arange(num_nodes, dtype=np.int64)
    return np.unique(np.linspace(0, num_nodes - 1, count, dtype=np.int64))


def prepare_dataset(dataset_dir: Path):
    data = np.load(dataset_dir / "data.npz")["data"].astype(np.float32)
    index = np.load(dataset_dir / "index.npz")
    values = data[..., 0]
    train_starts = index["train"][:, 1]
    test_starts = index["test"][:, 1]
    train_stop = int(train_starts.max() + 1)
    mean = values[:train_stop].mean(axis=0)
    sd = values[:train_stop].std(axis=0)
    values = (values - mean) / np.maximum(sd, 1e-6)
    corr = np.corrcoef(values[:train_stop].T)
    return {
        "values": values,
        "train_starts": train_starts,
        "test_starts": test_starts,
        "corr": corr,
        "num_nodes": int(values.shape[1]),
    }


def run_one(prepared, target_sensor, seed, *, candidate_count, budget, history_length, horizon,
            train_episodes, test_episodes, expert_repeats, ridge_penalty):
    values = prepared["values"]
    train_starts = prepared["train_starts"]
    test_starts = prepared["test_starts"]
    candidates = build_candidate_pool(
        prepared["corr"][target_sensor],
        target_sensor=target_sensor,
        candidate_count=candidate_count,
        seed=20260920,
    )
    rng = np.random.default_rng(seed + 2_000)
    train_times = rng.choice(train_starts, size=min(train_episodes, len(train_starts)), replace=False)
    test_times = rng.choice(test_starts, size=min(test_episodes, len(test_starts)), replace=False)
    train_batch = make_batch(values, train_times, target_sensor=target_sensor, candidate_sensors=candidates,
                             history_length=history_length, horizon=horizon)
    test_batch = make_batch(values, test_times, target_sensor=target_sensor, candidate_sensors=candidates,
                            history_length=history_length, horizon=horizon)
    expert = fit_subset_expert(train_batch, seed=seed + 3_000, repeats=expert_repeats, ridge_penalty=ridge_penalty)
    os = select_oracle_static(test_batch, expert, budget)
    og = select_oracle_greedy(test_batch, expert, budget)
    base = squared_error(test_batch, np.zeros_like(og), expert)
    g_os = float(np.mean(base - squared_error(test_batch, os, expert)))
    g_og = float(np.mean(base - squared_error(test_batch, og, expert)))
    return {
        "dataset": prepared.get("dataset", ""),
        "target_sensor": int(target_sensor),
        "seed": seed,
        "candidate_sensors": candidates.tolist(),
        "gain_base_to_oracle_static": g_os,
        "gain_base_to_oracle_greedy": g_og,
        "state_conditioning_headroom": float((g_og - g_os) / g_og) if g_og > 0 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data" / "staeformer")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--datasets", default="METRLA,PEMSBAY,PEMS03,PEMS04,PEMS07,PEMS08")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--target-count", type=int, default=32)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--test-episodes", type=int, default=2000)
    parser.add_argument("--expert-repeats", type=int, default=3)
    parser.add_argument("--ridge-penalty", type=float, default=10.0)
    args = parser.parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)
    records = []
    for dataset in datasets:
        dataset_dir = args.data_root / dataset
        prepared = prepare_dataset(dataset_dir)
        prepared["dataset"] = dataset
        targets = target_set(prepared["num_nodes"], args.target_count)
        for target in targets:
            for seed in seeds:
                started = time.time()
                record = run_one(
                    prepared,
                    int(target),
                    seed,
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
        print(f"completed {dataset} with {len(targets)} targets and {len(seeds)} seeds")
    (args.out_root / "records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    rows = ["dataset,target_sensor,seed,gain_base_to_oracle_static,gain_base_to_oracle_greedy,state_conditioning_headroom"]
    for r in records:
        rows.append(
            f"{r['dataset']},{r['target_sensor']},{r['seed']},"
            f"{r['gain_base_to_oracle_static']:.8f},{r['gain_base_to_oracle_greedy']:.8f},"
            f"{r['state_conditioning_headroom']:.8f}"
        )
    (args.out_root / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    aggregate = {}
    for dataset in datasets:
        subset = [r for r in records if r["dataset"] == dataset]
        head = np.asarray([r["state_conditioning_headroom"] for r in subset], dtype=float)
        aggregate[dataset] = {
            "targets": int(len({r["target_sensor"] for r in subset})),
            "seed_target_runs": len(subset),
            "H_state_mean": float(np.mean(head)),
            "H_state_median": float(np.median(head)),
            "H_state_std": float(np.std(head, ddof=1)),
            "H_state_positive_fraction": float(np.mean(head > 0.0)),
            "H_state_p10": float(np.quantile(head, .10)),
            "H_state_p90": float(np.quantile(head, .90)),
            "oracle_static_gain_mean": float(np.mean([r["gain_base_to_oracle_static"] for r in subset])),
            "oracle_greedy_gain_mean": float(np.mean([r["gain_base_to_oracle_greedy"] for r in subset])),
        }
    (args.out_root / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_root": str(args.out_root), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
