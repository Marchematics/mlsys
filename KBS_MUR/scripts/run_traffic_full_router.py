#!/usr/bin/env python3
"""Train and evaluate the MUR comparison after the METR-LA oracle gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mur.synthetic_experiment import (  # noqa: E402
    sample_pair_dataset,
    select_mur,
    select_static,
    train_utility_model,
)
from mur.traffic import (  # noqa: E402
    TrafficBatch,
    fit_subset_expert,
    make_batch,
    observed_candidate_marginals,
    pack_selected,
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


def select_relevance(relevance: np.ndarray, budget: int) -> np.ndarray:
    order = np.argsort(-relevance, axis=1, kind="stable")[:, :budget]
    selected = np.zeros_like(relevance, dtype=bool)
    selected[np.arange(relevance.shape[0])[:, None], order] = True
    return selected


def select_mmr(batch: TrafficBatch, relevance: np.ndarray, budget: int, gamma: float = .5) -> np.ndarray:
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    vectors = batch.candidate_history.astype(float)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)
    similarity = np.einsum("bik,bjk->bij", vectors, vectors)
    for row in range(batch.episodes):
        for step in range(min(budget, batch.candidate_count)):
            remaining = np.flatnonzero(~selected[row])
            if step == 0:
                adjusted = relevance[row, remaining]
            else:
                redundancy = np.max(similarity[row, remaining][:, selected[row]], axis=1)
                adjusted = relevance[row, remaining] - gamma * redundancy
            selected[row, int(remaining[int(np.argmax(adjusted))])] = True
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


def load_protocol(data_path: Path, gate_result: Path, seed: int):
    gate = json.loads(gate_result.read_text(encoding="utf-8"))
    cfg = gate["config"]
    frame = pd.read_csv(data_path)
    sensor_columns = [column for column in frame.columns if column not in {"date", "OT"}]
    values = frame[sensor_columns].to_numpy(dtype=np.float32)
    train_stop = int(values.shape[0] * cfg["train_fraction"])
    validation_stop = int(values.shape[0] * (cfg["train_fraction"] + cfg["validation_fraction"]))
    mean = values[:train_stop].mean(axis=0)
    sd = values[:train_stop].std(axis=0)
    values = (values - mean) / np.maximum(sd, 1e-6)
    rng = np.random.default_rng(seed + 2_000)

    def times(start: int, stop: int, count: int) -> np.ndarray:
        low = start + cfg["history_length"]
        high = stop - cfg["horizon"]
        return rng.integers(low, high, size=count)

    candidate_sensors = np.asarray(cfg["candidate_sensors"], dtype=np.int64)
    target = int(cfg["target_sensor_index"])
    batches = []
    for start, stop, count in (
        (0, train_stop, cfg["train_episodes"]),
        (train_stop, validation_stop, cfg["validation_episodes"]),
        (validation_stop, values.shape[0], cfg["test_episodes"]),
    ):
        batches.append(
            make_batch(
                values,
                times(start, stop, count),
                target_sensor=target,
                candidate_sensors=candidate_sensors,
                history_length=cfg["history_length"],
                horizon=cfg["horizon"],
            )
        )
    return cfg, batches, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--gate-result", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    cfg, (train, calibration, test), _ = load_protocol(args.data_path, args.gate_result, args.seed)
    budget = int(cfg["budget"])
    expert = fit_subset_expert(
        train,
        seed=args.seed + 3_000,
        repeats=int(cfg["expert_repeats"]),
        ridge_penalty=float(cfg["ridge_penalty"]),
    )
    marginal_fn = lambda batch, selected: observed_candidate_marginals(batch, selected, expert)
    static_pairs = sample_pair_dataset(
        train,
        max_budget=budget,
        states_per_episode=4,
        seed=args.seed + 4_000,
        static_only=True,
        marginal_fn=marginal_fn,
        pack_fn=pack_selected,
    )
    mur_pairs = sample_pair_dataset(
        train,
        max_budget=budget,
        states_per_episode=4,
        seed=args.seed + 5_000,
        static_only=False,
        marginal_fn=marginal_fn,
        pack_fn=pack_selected,
    )
    static_model = train_utility_model(
        static_pairs, max_budget=budget, seed=args.seed + 6_000, device=args.device, epochs=args.epochs
    )
    mur_model = train_utility_model(
        mur_pairs, max_budget=budget, seed=args.seed + 7_000, device=args.device, epochs=args.epochs
    )
    candidate_corr = np.asarray(cfg["candidate_sensor_correlations"], dtype=np.float32)
    relevance = np.broadcast_to(candidate_corr[None, :], (test.episodes, test.candidate_count)).copy()
    selections = {
        "base_only": np.zeros((test.episodes, test.candidate_count), dtype=bool),
        "pool_all": np.ones((test.episodes, test.candidate_count), dtype=bool),
        "relevance": select_relevance(relevance, budget),
        "mmr": select_mmr(test, relevance, budget),
        "static_utility": select_static(
            static_model, test, budget=budget, device=args.device, pack_fn=pack_selected
        ),
        "mur": select_mur(
            mur_model, test, budget=budget, device=args.device, pack_fn=pack_selected
        ),
        "oracle_static": select_oracle_static(test, expert, budget),
    }
    oracle_greedy = select_oracle_greedy(test, expert, budget)
    metrics = {
        name: summarize(test, expert, selected, oracle_greedy)
        for name, selected in selections.items()
    }
    metrics["oracle_greedy"] = summarize(test, expert, oracle_greedy, oracle_greedy)
    oracle_gain = metrics["oracle_greedy"]["mean_prediction_gain"]
    static_oracle_gain = metrics["oracle_static"]["mean_prediction_gain"]
    result = {
        "run_id": "R075_traffic_full_router",
        "status": "confirmatory",
        "config": {**cfg, "seed": args.seed, "router_epochs": args.epochs, "device": args.device},
        "metrics": metrics,
        "oracle_quantities": {
            "gain_base_to_oracle_static": static_oracle_gain,
            "gain_base_to_oracle_greedy": oracle_gain,
            "state_conditioning_headroom": (oracle_gain - static_oracle_gain) / oracle_gain
            if oracle_gain > 0.0
            else float("nan"),
        },
        "wall_seconds": time.time() - started,
        "provenance": {
            "data_path": str(args.data_path),
            "data_sha256": sha256(args.data_path),
            "gate_result": str(args.gate_result),
            "traffic_module_sha256": sha256(ROOT / "src" / "mur" / "traffic.py"),
        },
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
