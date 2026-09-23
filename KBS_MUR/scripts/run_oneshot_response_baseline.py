#!/usr/bin/env python3
"""One-shot response-aware ranking baseline (ablation of sequential routing).

The sequential router scores candidates, accepts one, and re-scores the rest on
the updated state. This runner trains exactly the same response-aware scorer on
exactly the same data as the multi-target router runs and then evaluates two
policies with it:

``oneshot_response``
    the same scorer asked once, on the empty state, over the whole pool, with
    the resulting ranking applied as it stands.

The sequential router is read from its own run set (``R080b``), which uses the
same scorer recipe and the same protocol; this runner only adds the one-shot
policy. Two differences separate the two policies, and both favour the one-shot
policy: it scores the whole pool at its single decision instead of the cached
shortlist, and it is not limited by the screen. A gain of the sequential router
over this baseline therefore cannot be explained by candidate availability.

Output layout matches the other sweeps:
``<out>/<dataset>/target<t>_seed<s>/result.json``.
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

from mur.traffic import fit_subset_expert, make_batch, squared_error  # noqa: E402
from run_response_router_multitarget import (  # noqa: E402
    build_candidate_pool,
    prepare_dataset,
    target_set,
)
from run_traffic_ranked_response_router_sweep import (  # noqa: E402
    build_full_state_data,
    select_oneshot_response,
    select_oracle_greedy,
    train_ranked_response_model,
)

POLICIES = ["oneshot_response", "ranked_response_full"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data" / "staeformer")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--target-count", type=int, default=32)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--calibration-episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=2000)
    parser.add_argument(
        "--states-per-episode",
        type=int,
        default=4,
        help="states sampled per training episode; the router runs use four",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-id", default="R207_oneshot_response_ablation")
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
            expert = fit_subset_expert(train, seed=seed + 3_000, repeats=3, ridge_penalty=10.0)

            full_data = build_full_state_data(
                train,
                expert,
                max_budget=args.budget,
                states_per_episode=args.states_per_episode,
                seed=seed + 9_000,
            )
            ranked_model = train_ranked_response_model(
                full_data,
                max_budget=args.budget,
                seed=seed + 10_000,
                device=args.device,
                epochs=args.epochs,
                beta=args.beta,
            )

            selections = {
                "oneshot_response": select_oneshot_response(
                    ranked_model, ranked_model, test, expert, budget=args.budget, device=args.device
                ),
                # Access-matched control: the sequential router with the whole pool
                # as its shortlist. At q = K the cached screen cannot exclude any
                # candidate, so passing the response model as the screen model is
                # exact here, and the only difference from the one-shot policy is
                # that scores are recomputed after every accepted candidate.
                "ranked_response_full": select_ranked_response_mur(
                    ranked_model,
                    ranked_model,
                    test,
                    expert,
                    budget=args.budget,
                    q=test.candidate_count,
                    device=args.device,
                ),
            }
            oracle = select_oracle_greedy(test, expert, args.budget)
            base_loss = squared_error(
                test, np.zeros((test.episodes, test.candidate_count), dtype=bool), expert
            )
            metrics = {}
            for name in POLICIES:
                selected = selections[name]
                loss = squared_error(test, selected, expert)
                gain = base_loss - loss
                metrics[name] = {
                    "mean_prediction_gain": float(np.mean(gain)),
                    "negative_transfer_rate": float(np.mean(loss > base_loss)),
                    "mean_selected_contexts": float(np.mean(selected.sum(axis=1))),
                }
            metrics["oracle_greedy"] = {
                "mean_prediction_gain": float(np.mean(base_loss - squared_error(test, oracle, expert))),
                "negative_transfer_rate": float("nan"),
                "mean_selected_contexts": float(np.mean(oracle.sum(axis=1))),
            }
            record = {
                "run_id": args.run_id,
                "status": "confirmatory",
                "dataset": args.dataset,
                "target_sensor": target_sensor,
                "seed": seed,
                "config": {
                    "candidate_count": args.candidate_count,
                    "budget": args.budget,
                    "q": args.q,
                    "states_per_episode": args.states_per_episode,
                    "epochs": args.epochs,
                    "candidate_sensors": candidates.tolist(),
                    "note": "one-shot policy scored with the response-aware model trained here",
                },
                "metrics": metrics,
                "wall_seconds": time.time() - started,
            }
            seed_dir = dataset_root / f"target{target_sensor}_seed{seed}"
            seed_dir.mkdir(parents=True)
            (seed_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"completed target {target_sensor} for {args.dataset}", flush=True)


if __name__ == "__main__":
    main()
