#!/usr/bin/env python3
"""R060/R061 Electricity smoke: multi-context frozen-ridge routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
    sample_pair_dataset,
    score_state,
    select_mur,
    select_mur_interval,
    select_static,
    train_utility_model,
    predict_pair_dataset,
)
from mur.calibration import SymmetricUtilityCalibrator  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _batch_audit(batch, *, clients, week_start: int, week_stop: int) -> dict[str, int]:
    clients = np.asarray(clients)
    return {
        "target_client_outside_split": int(np.sum(~np.isin(batch.target_client, clients))),
        "candidate_client_outside_split": int(np.sum(~np.isin(batch.candidate_client, clients))),
        "anchor_week_outside_split": int(
            np.sum((batch.anchor_week < week_start) | (batch.anchor_week >= week_stop))
        ),
        "candidate_week_outside_split": int(
            np.sum((batch.candidate_week < week_start) | (batch.candidate_week >= week_stop))
        ),
        "candidate_same_as_anchor_week": int(
            np.sum(batch.candidate_week == batch.anchor_week[:, None])
        ),
        "candidate_not_historical": int(
            np.sum(batch.candidate_week >= batch.anchor_week[:, None])
        ),
    }


def select_relevance(batch, budget: int) -> np.ndarray:
    order = np.argsort(-batch.relevance, axis=1, kind="stable")[:, :budget]
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    selected[np.arange(batch.episodes)[:, None], order] = True
    return selected


def select_coverage(batch, budget: int) -> np.ndarray:
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for row in range(batch.episodes):
        covered: set[int] = set()
        for _ in range(budget):
            remaining = np.flatnonzero(~selected[row])
            adjusted = batch.relevance[row, remaining].astype(float)
            adjusted -= np.asarray(
                [2.0 if int(batch.candidate_client[row, index]) in covered else 0.0 for index in remaining]
            )
            choice = int(remaining[int(np.argmax(adjusted))])
            selected[row, choice] = True
            covered.add(int(batch.candidate_client[row, choice]))
    return selected


def select_oracle(batch, budget: int) -> np.ndarray:
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for _ in range(budget):
        marginal = observed_candidate_marginals(batch, selected)
        choice = np.argmax(marginal, axis=1)
        active = marginal[np.arange(batch.episodes), choice] > 0.0
        selected[np.arange(batch.episodes)[active], choice[active]] = True
    return selected


def summarize(batch, selected, oracle) -> dict[str, float]:
    base = squared_error(batch, np.zeros_like(selected))
    routed = squared_error(batch, selected)
    oracle_loss = squared_error(batch, oracle)
    out = decision_metrics(base, routed, oracle_loss, np.sum(selected, axis=1))
    duplicates = []
    for row in range(batch.episodes):
        groups = batch.candidate_client[row, selected[row]]
        duplicates.append(len(groups) - len(np.unique(groups)))
    out["mean_duplicate_selections"] = float(np.mean(duplicates))
    return out


def prediction_deltas(batch, selected: np.ndarray) -> np.ndarray:
    """Exact frozen-expert prediction deltas for the high-cost diagnostic."""

    mask = np.asarray(selected, dtype=bool)
    before = expert_prediction(batch, mask)
    output = np.zeros((batch.episodes, batch.candidate_count), dtype=np.float32)
    for candidate in range(batch.candidate_count):
        eligible = ~mask[:, candidate]
        if not np.any(eligible):
            continue
        augmented = mask.copy()
        augmented[eligible, candidate] = True
        output[eligible, candidate] = expert_prediction(batch, augmented)[eligible] - before[eligible]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", type=Path, default=ROOT.parent / "data/uci_electricity/electricity_float32.npz")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--train-episodes", type=int, default=2048)
    parser.add_argument("--calibration-episodes", type=int, default=1024)
    parser.add_argument("--test-episodes", type=int, default=2048)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--same-client-fraction", type=float, default=0.5)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--context-points", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.time()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    run_instance_id = f"R061_seed{args.seed}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(started))}"
    data = load_electricity_data(
        str(args.cache_path),
        active_fraction_threshold=0.90,
        split_seed=20260719,
        train_client_count=160,
        validation_client_count=60,
        train_week_stop=32,
        harmonics=4,
    )
    train = sample_context_batch(
        data,
        clients=data.train_clients,
        week_start=0,
        week_stop=32,
        episodes=args.train_episodes,
        candidate_count=args.candidate_count,
        same_client_fraction=args.same_client_fraction,
        context_points=args.context_points,
        ridge_penalty=1.0,
        rng=np.random.default_rng(args.seed + 10_000),
    )
    calibration = sample_context_batch(
        data,
        clients=data.validation_clients,
        week_start=32,
        week_stop=40,
        episodes=args.calibration_episodes,
        candidate_count=args.candidate_count,
        same_client_fraction=args.same_client_fraction,
        context_points=args.context_points,
        ridge_penalty=1.0,
        rng=np.random.default_rng(args.seed + 20_000),
    )
    test = sample_context_batch(
        data,
        clients=data.test_clients,
        week_start=40,
        week_stop=52,
        episodes=args.test_episodes,
        candidate_count=args.candidate_count,
        same_client_fraction=args.same_client_fraction,
        context_points=args.context_points,
        ridge_penalty=1.0,
        rng=np.random.default_rng(args.seed + 30_000),
    )
    static_pairs = sample_pair_dataset(
        train,
        max_budget=args.budget,
        states_per_episode=4,
        seed=args.seed + 40_000,
        static_only=True,
        marginal_fn=observed_candidate_marginals,
        pack_fn=pack_selected,
    )
    mur_pairs = sample_pair_dataset(
        train,
        max_budget=args.budget,
        states_per_episode=4,
        seed=args.seed + 50_000,
        static_only=False,
        marginal_fn=observed_candidate_marginals,
        pack_fn=pack_selected,
    )
    static_model = train_utility_model(
        static_pairs,
        max_budget=args.budget,
        seed=args.seed + 60_000,
        device=args.device,
        epochs=args.epochs,
    )
    mur_model = train_utility_model(
        mur_pairs,
        max_budget=args.budget,
        seed=args.seed + 70_000,
        device=args.device,
        epochs=args.epochs,
    )
    mur_full_model = None
    full_calibrator = None
    if args.include_full:
        full_pairs = sample_pair_dataset(
            train,
            max_budget=args.budget,
            states_per_episode=4,
            seed=args.seed + 55_000,
            static_only=False,
            marginal_fn=observed_candidate_marginals,
            pack_fn=pack_selected,
            delta_fn=prediction_deltas,
        )
        mur_full_model = train_utility_model(
            full_pairs,
            max_budget=args.budget,
            seed=args.seed + 75_000,
            device=args.device,
            epochs=args.epochs,
        )
    cal_pairs = sample_pair_dataset(
        calibration,
        max_budget=args.budget,
        states_per_episode=4,
        seed=args.seed + 80_000,
        static_only=False,
        marginal_fn=observed_candidate_marginals,
        pack_fn=pack_selected,
    )
    cal_prediction = predict_pair_dataset(mur_model, cal_pairs, device=args.device)
    calibrator = SymmetricUtilityCalibrator.fit(cal_prediction, cal_pairs.target, alpha=0.1)
    if mur_full_model is not None:
        full_cal_pairs = sample_pair_dataset(
            calibration,
            max_budget=args.budget,
            states_per_episode=4,
            seed=args.seed + 85_000,
            static_only=False,
            marginal_fn=observed_candidate_marginals,
            pack_fn=pack_selected,
            delta_fn=prediction_deltas,
        )
        full_cal_prediction = predict_pair_dataset(
            mur_full_model, full_cal_pairs, device=args.device
        )
        full_calibrator = SymmetricUtilityCalibrator.fit(
            full_cal_prediction, full_cal_pairs.target, alpha=0.1
        )
    base = np.zeros((test.episodes, test.candidate_count), dtype=bool)
    pool_all = np.ones_like(base)
    selections = {
        "base_only": base,
        "pool_all": pool_all,
        "relevance": select_relevance(test, args.budget),
        "coverage": select_coverage(test, args.budget),
        "static_utility": select_static(
            static_model, test, budget=args.budget, device=args.device, pack_fn=pack_selected
        ),
        "mur_light": select_mur(
            mur_model, test, budget=args.budget, device=args.device, pack_fn=pack_selected
        ),
        "mur_interval": select_mur_interval(
            mur_model,
            test,
            calibrator,
            budget=args.budget,
            device=args.device,
            pack_fn=pack_selected,
        ),
    }
    if mur_full_model is not None:
        selections["mur_full"] = select_mur(
            mur_full_model,
            test,
            budget=args.budget,
            device=args.device,
            pack_fn=pack_selected,
            delta_fn=prediction_deltas,
        )
        selections["mur_full_interval"] = select_mur_interval(
            mur_full_model,
            test,
            full_calibrator,
            budget=args.budget,
            device=args.device,
            pack_fn=pack_selected,
            delta_fn=prediction_deltas,
        )
    oracle = select_oracle(test, args.budget)
    result = {
        "run_id": "R061",
        "run_instance_id": run_instance_id,
        "status": "smoke",
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
            "include_full": args.include_full,
            "full_calibration_radius": full_calibrator.radius if full_calibrator else None,
            "active_fraction_threshold": 0.90,
            "split_seed": 20260719,
            "train_client_count": 160,
            "validation_client_count": 60,
            "train_week_stop": 32,
            "harmonics": 4,
            "ridge_penalty": 1.0,
            "states_per_episode": 4,
            "week_windows": {"train": [0, 32], "calibration": [32, 40], "test": [40, 52]},
        },
        "metrics": {name: summarize(test, selected, oracle) for name, selected in selections.items()},
        "metrics_oracle": summarize(test, oracle, oracle),
        "wall_seconds": time.time() - started,
        "provenance": {
            "started_utc": started_utc,
            "script_sha256": _sha256(Path(__file__)),
            "electricity_module_sha256": _sha256(ROOT / "src/mur/electricity.py"),
            "synthetic_experiment_module_sha256": _sha256(ROOT / "src/mur/synthetic_experiment.py"),
            "model_module_sha256": _sha256(ROOT / "src/mur/model.py"),
            "metrics_module_sha256": _sha256(ROOT / "src/mur/metrics.py"),
            "calibration_module_sha256": _sha256(ROOT / "src/mur/calibration.py"),
            "cache_path": str(args.cache_path),
            "cache_sha256": _sha256(args.cache_path),
            "split_audit": {
                "train": _batch_audit(train, clients=data.train_clients, week_start=0, week_stop=32),
                "calibration": _batch_audit(
                    calibration, clients=data.validation_clients, week_start=32, week_stop=40
                ),
                "test": _batch_audit(test, clients=data.test_clients, week_start=40, week_stop=52),
            },
        },
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
