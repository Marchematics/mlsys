#!/usr/bin/env python3
"""Compute, latency and memory evidence for response-aware routing (A4).

For a single (dataset, target, seed) cell this measures, at deployment batch
size, how many expert evaluations each policy spends, how long the response
stage takes, and the peak device memory, so the accuracy--compute trade-off of
the shortlist can be reported with real numbers instead of probe counts.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KBS = ROOT / "KBS_MUR"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KBS / "src"))
sys.path.insert(0, str(KBS / "scripts"))

from mur.synthetic_experiment import score_state  # noqa: E402

from fast_router import train_ranked_response_model_fast, train_utility_model_fast  # noqa: E402
from neural_expert import SubsetExpertOps, finetune_expert, load_checkpoint  # noqa: E402
from pami_traffic import (  # noqa: E402
    build_candidate_pool,
    load_traffic_data,
    make_neural_batch,
    target_set,
)
from run_strong_backbone import (  # noqa: E402
    build_full_state_data,
    sample_pair_dataset_fast,
    select_oracle_greedy,
    select_oracle_static,
    select_ranked_response_mur,
)


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def device_memory() -> dict:
    if not torch.cuda.is_available():
        return {"peak_allocated_mb": 0.0, "peak_reserved_mb": 0.0}
    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/staeformer")
    parser.add_argument("--dataset", default="METRLA")
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--q-values", default="2,4,8,16")
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--calibration-episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=2000)
    parser.add_argument("--router-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    data = load_traffic_data(args.data_root / args.dataset)
    candidates = build_candidate_pool(
        data.correlation[args.target], target_sensor=args.target,
        candidate_count=args.candidate_count, seed=20260920,
    )
    model, _ = load_checkpoint(args.expert_root / f"seed{args.seed}" / "expert.pt", device=args.device)
    model = copy.deepcopy(model)
    rng = np.random.default_rng(args.seed + 2_000)
    batches = []
    for starts, count in (
        (data.train_times, args.train_episodes),
        (data.validation_times, args.calibration_episodes),
        (data.test_times, args.test_episodes),
    ):
        times = rng.choice(starts, size=min(count, len(starts)), replace=False)
        batches.append(make_neural_batch(data, times, target_sensor=args.target, candidate_sensors=candidates))
    batches = tuple(batches)
    finetune_expert(
        model, batches[0], epochs=args.finetune_epochs, batch_size=256, learning_rate=3e-4,
        budget=args.budget, seed=args.seed + 7_777, device=args.device, validation_batch=batches[1],
    )
    ops = SubsetExpertOps(model, device=args.device)
    budget = args.budget
    static_pairs = sample_pair_dataset_fast(
        batches[0], ops, max_budget=budget, states_per_episode=4, seed=args.seed + 4_000, static_only=True
    )
    cached_pairs = sample_pair_dataset_fast(
        batches[0], ops, max_budget=budget, states_per_episode=4, seed=args.seed + 5_000, static_only=False
    )
    static_model = train_utility_model_fast(
        static_pairs, max_budget=budget, seed=args.seed + 6_000, device=args.device, epochs=args.router_epochs
    )
    cached_model = train_utility_model_fast(
        cached_pairs, max_budget=budget, seed=args.seed + 7_000, device=args.device, epochs=args.router_epochs
    )
    full_data = build_full_state_data(
        batches[0], ops, max_budget=budget, states_per_episode=4, seed=args.seed + 9_000
    )
    ranked_model = train_ranked_response_model_fast(
        full_data, max_budget=budget, seed=args.seed + 10_000, device=args.device,
        epochs=args.router_epochs, batch_size=128, beta=1.0,
    )

    test = batches[2]
    empty = np.zeros((test.episodes, test.candidate_count), dtype=bool)
    report = {
        "dataset": args.dataset,
        "target": args.target,
        "seed": args.seed,
        "test_episodes": test.episodes,
        "candidate_count": test.candidate_count,
        "budget": budget,
        "device": args.device,
        "policies": {},
    }

    def timed_select(function, *positional, **keywords):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        ops.reset_profile()
        started = time.perf_counter()
        output = function(*positional, **keywords)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return output, {
            "wall_seconds": elapsed,
            "seconds_per_episode_ms": 1000.0 * elapsed / test.episodes,
            "expert_calls": ops.calls,
            "expert_rows": ops.rows,
            "expert_seconds": ops.seconds,
            "expert_rows_per_episode": ops.rows / test.episodes,
            **device_memory(),
        }

    _, report["policies"]["empty_state"] = timed_select(ops.predict, test, empty)
    _, report["policies"]["oracle_static"] = timed_select(select_oracle_static, test, ops, budget)
    _, report["policies"]["oracle_greedy"] = timed_select(select_oracle_greedy, test, ops, budget)
    for q in parse_ints(args.q_values):
        label = "response_full" if q >= test.candidate_count else f"response_q{q}"
        _, stats = timed_select(
            select_ranked_response_mur, cached_model, ranked_model, test, ops,
            budget=budget, q=q, device=args.device,
        )
        stats["expert_rows_per_episode"] = stats["expert_rows"] / test.episodes
        stats["fraction_of_full_pool_rows"] = stats["expert_rows"] / (
            (test.candidate_count + 1) * budget * test.episodes
        )
        report["policies"][label] = stats

    # Screening alone (cached shortlist) carries no expert cost.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    _ = score_state(cached_model, test, empty, device=args.device, pack_fn=ops.pack)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report["policies"]["screening_only_q4"] = {
        "wall_seconds": elapsed,
        "seconds_per_episode_ms": 1000.0 * elapsed / test.episodes,
        "expert_calls": 0,
        "expert_rows": 0,
        **device_memory(),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "compute_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
