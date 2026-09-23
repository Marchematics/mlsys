#!/usr/bin/env python3
"""Backbone-generalisation check with a subset-capable neural predictor.

Repeats the routing comparison with the ridge expert replaced by a small MLP
that consumes the same anchor and masked candidate histories. The question is
whether the ordering of the learned policies survives a nonlinear predictor
family, not whether the forecasts improve, so the protocol is smaller than the
main multi-target sweep: fewer targets, one run, and a reduced episode budget.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mur.neural_expert import fit_neural_expert  # noqa: E402
from mur.traffic import make_batch, squared_error  # noqa: E402
from run_response_router_multitarget import build_candidate_pool, prepare_dataset, target_set  # noqa: E402
from run_traffic_ranked_response_router_sweep import evaluate_seed  # noqa: E402

POLICIES = [
    "base_only",
    "pool_all",
    "relevance",
    "mmr",
    "static_utility",
    "cached_mur",
    "ranked_response_q4",
    "oracle_static",
    "oracle_greedy",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data" / "staeformer")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", default="101")
    parser.add_argument("--target-count", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-episodes", type=int, default=1500)
    parser.add_argument("--calibration-episodes", type=int, default=500)
    parser.add_argument("--test-episodes", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--expert-epochs", type=int, default=30)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset_root = args.out_root / args.dataset
    if dataset_root.exists():
        raise FileExistsError(f"dataset output already exists: {dataset_root}")
    dataset_root.mkdir(parents=True)

    prepared = prepare_dataset(args.data_root / args.dataset)
    targets = target_set(prepared["num_nodes"], args.target_count)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    for target_sensor in targets:
        target_sensor = int(target_sensor)
        candidates = build_candidate_pool(
            prepared["corr"][target_sensor],
            target_sensor=target_sensor,
            candidate_count=args.candidate_count,
            seed=20260920,
        )
        corr = prepared["corr"][target_sensor]
        cfg = {
            "budget": args.budget,
            "expert_repeats": 0,
            "ridge_penalty": 0.0,
            "candidate_sensor_correlations": corr[candidates].astype(np.float32).tolist(),
            "candidate_sensors": candidates.tolist(),
            "target_sensor_index": target_sensor,
        }
        for seed in seeds:
            started = time.time()
            rng = np.random.default_rng(seed + 2_000)
            batches = []
            for starts, count in (
                (prepared["train_starts"], args.train_episodes),
                (prepared["validation_starts"], args.calibration_episodes),
                (prepared["test_starts"], args.test_episodes),
            ):
                times = rng.choice(starts, size=min(count, len(starts)), replace=False)
                batches.append(
                    make_batch(
                        prepared["values"],
                        times,
                        target_sensor=target_sensor,
                        candidate_sensors=candidates,
                        history_length=args.history_length,
                        horizon=args.horizon,
                    )
                )
            train, _calibration, test = batches
            expert = fit_neural_expert(
                train, seed=seed + 3_000, epochs=args.expert_epochs, device=args.device
            )
            result, gains, diagnostics = evaluate_seed(
                seed,
                cfg,
                batches,
                expert,
                q_values=[4],
                epochs=args.epochs,
                beta=args.beta,
                hard_negative_q=0,
                device=args.device,
                save_diagnostics=False,
            )
            # record the neural expert's own anchor-only loss for reference
            empty = np.zeros((test.episodes, test.candidate_count), dtype=bool)
            result["metrics"]["neural_anchor_loss"] = {
                "mean_prediction_gain": 0.0,
                "negative_transfer_rate": 0.0,
                "mean_utility_recovery": 0.0,
                "mean_selected_contexts": 0.0,
            }
            result["neural_anchor_mse"] = float(squared_error(test, empty, expert).mean())
            result["dataset"] = args.dataset
            result["target_sensor"] = target_sensor
            result["seed"] = seed
            result["wall_seconds"] = time.time() - started
            seed_dir = dataset_root / f"target{target_sensor}_seed{seed}"
            seed_dir.mkdir(parents=True)
            (seed_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"completed target {target_sensor} for {args.dataset}", flush=True)


if __name__ == "__main__":
    main()
