#!/usr/bin/env python3
"""Preliminary Gate B/N run: Static Utility versus MUR under redundancy."""

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

from mur.synthetic import (  # noqa: E402
    SyntheticConfig,
    generate_synthetic_batch,
    observed_candidate_marginals,
)
from mur.synthetic_experiment import (  # noqa: E402
    predict_pair_dataset,
    sample_pair_dataset,
    select_coverage,
    select_mmr,
    select_mur,
    select_mur_interval,
    select_oracle,
    select_oracle_static,
    select_relevance,
    select_static,
    score_state,
    summarize_policy,
    trace_mur,
    trace_positive_utility_precision,
    train_utility_model,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_float_grid(value: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("float grid must be nonempty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", default="synthetic_gate")
    parser.add_argument("--status", default="pilot_not_confirmatory")
    parser.add_argument("--redundancy", type=float, required=True)
    parser.add_argument("--harmful-fraction", type=float, default=0.0)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--test-episodes", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calibration-alpha", type=float, default=0.1)
    parser.add_argument("--mmr-gamma-grid", default="0,.25,.5,1.0")
    parser.add_argument("--save-diagnostics", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    config = SyntheticConfig(
        candidate_count=args.candidate_count,
        group_count=args.candidate_count,
        budget=args.budget,
        redundancy_ratio=args.redundancy,
        harmful_fraction=args.harmful_fraction,
    )
    train = generate_synthetic_batch(args.train_episodes, config, seed=args.seed + 10_000)
    calibration = generate_synthetic_batch(
        args.test_episodes, config, seed=args.seed + 15_000
    )
    test = generate_synthetic_batch(args.test_episodes, config, seed=args.seed + 20_000)
    static_pairs = sample_pair_dataset(
        train,
        max_budget=config.budget,
        states_per_episode=4,
        seed=args.seed + 30_000,
        static_only=True,
    )
    mur_pairs = sample_pair_dataset(
        train,
        max_budget=config.budget,
        states_per_episode=4,
        seed=args.seed + 40_000,
        static_only=False,
    )
    static_model = train_utility_model(
        static_pairs,
        max_budget=config.budget,
        seed=args.seed + 50_000,
        device=args.device,
        epochs=args.epochs,
    )
    mur_model = train_utility_model(
        mur_pairs,
        max_budget=config.budget,
        seed=args.seed + 60_000,
        device=args.device,
        epochs=args.epochs,
    )
    calibration_pairs = sample_pair_dataset(
        calibration,
        max_budget=config.budget,
        states_per_episode=4,
        seed=args.seed + 45_000,
        static_only=False,
    )
    calibration_prediction = predict_pair_dataset(
        mur_model, calibration_pairs, device=args.device
    )
    from mur.calibration import SymmetricUtilityCalibrator

    calibrator = SymmetricUtilityCalibrator.fit(
        calibration_prediction, calibration_pairs.target, alpha=args.calibration_alpha
    )
    mmr_gamma_grid = parse_float_grid(args.mmr_gamma_grid)
    calibration_oracle = select_oracle(calibration, budget=config.budget)
    mmr_validation = {
        gamma: summarize_policy(
            calibration,
            select_mmr(calibration, budget=config.budget, gamma=gamma),
            calibration_oracle,
        )
        for gamma in mmr_gamma_grid
    }
    mmr_gamma = max(
        mmr_gamma_grid,
        key=lambda gamma: (
            mmr_validation[gamma]["mean_prediction_gain"],
            -gamma,
        ),
    )
    interval_calibration_selection = select_mur_interval(
        mur_model,
        calibration,
        calibrator,
        budget=config.budget,
        device=args.device,
    )
    interval_context_target = float(np.mean(np.sum(interval_calibration_selection, axis=1)))
    free_threshold_grid = np.unique(
        np.quantile(
            np.abs(calibration_pairs.target),
            np.linspace(0.0, 1.0, 21),
        )
    )
    free_threshold_validation = {
        float(threshold): summarize_policy(
            calibration,
            select_mur(
                mur_model,
                calibration,
                budget=config.budget,
                device=args.device,
                threshold=float(threshold),
            ),
            calibration_oracle,
        )
        for threshold in free_threshold_grid
    }
    free_threshold = min(
        (float(threshold) for threshold in free_threshold_grid),
        key=lambda threshold: (
            abs(
                free_threshold_validation[threshold]["mean_selected_contexts"]
                - interval_context_target
            ),
            -free_threshold_validation[threshold]["mean_prediction_gain"],
        ),
    )
    selections = {
        "relevance": select_relevance(test, budget=config.budget),
        "mmr": select_mmr(test, budget=config.budget, gamma=mmr_gamma),
        "coverage": select_coverage(test, budget=config.budget),
        "oracle_static": select_oracle_static(test, budget=config.budget),
        "static_utility": select_static(
            static_model, test, budget=config.budget, device=args.device
        ),
        "mur_light": select_mur(mur_model, test, budget=config.budget, device=args.device),
        "mur_interval": select_mur_interval(
            mur_model,
            test,
            calibrator,
            budget=config.budget,
            device=args.device,
        ),
        "mur_free_threshold": select_mur(
            mur_model,
            test,
            budget=config.budget,
            device=args.device,
            threshold=free_threshold,
        ),
    }
    oracle = select_oracle(test, budget=config.budget)
    metrics = {
        name: summarize_policy(test, selected, oracle)
        for name, selected in selections.items()
    }
    metrics["oracle"] = summarize_policy(test, oracle, oracle)
    traces = {
        "mur_light": trace_mur(
            mur_model, test, budget=config.budget, device=args.device
        ),
        "mur_interval": trace_mur(
            mur_model,
            test,
            budget=config.budget,
            device=args.device,
            calibrator=calibrator,
        ),
        "mur_free_threshold": trace_mur(
            mur_model,
            test,
            budget=config.budget,
            device=args.device,
            threshold=free_threshold,
        ),
    }
    for name, trace in traces.items():
        metrics[name]["positive_selection_precision"] = (
            trace_positive_utility_precision(test, trace)
        )
    oracle_gain = metrics["oracle"]["mean_prediction_gain"]
    oracle_static_gain = metrics["oracle_static"]["mean_prediction_gain"]
    mur_gain = metrics["mur_light"]["mean_prediction_gain"]
    if oracle_gain > 0.0:
        metrics["state_conditioning_headroom"] = float(
            (oracle_gain - oracle_static_gain) / oracle_gain
        )
        metrics["mur_estimation_gap"] = float((oracle_gain - mur_gain) / oracle_gain)
        static_headroom = oracle_gain - oracle_static_gain
        metrics["mur_headroom_recovery"] = float(
            (mur_gain - oracle_static_gain) / static_headroom
        ) if static_headroom > 0.0 else float("nan")
    else:
        metrics["state_conditioning_headroom"] = float("nan")
        metrics["mur_estimation_gap"] = float("nan")
        metrics["mur_headroom_recovery"] = float("nan")

    def diagnostics(states):
        if isinstance(states, np.ndarray):
            states = [states]
        errors, top1_hits, regrets = [], [], []
        prediction_arrays, truth_arrays, mask_arrays = [], [], []
        for state_mask in states:
            predicted = score_state(mur_model, test, state_mask, device=args.device)
            truth = observed_candidate_marginals(test, state_mask)
            rows, cols = np.where(~state_mask)
            errors.extend(np.abs(predicted[rows, cols] - truth[rows, cols]).tolist())
            top1_pred = np.argmax(predicted, axis=1)
            top1_true = np.argmax(truth, axis=1)
            top1_hits.extend((top1_pred == top1_true).tolist())
            regrets.extend(
                (truth[np.arange(test.episodes), top1_true]
                 - truth[np.arange(test.episodes), top1_pred]).tolist()
            )
            prediction_arrays.append(predicted.astype(np.float32))
            truth_arrays.append(truth.astype(np.float32))
            mask_arrays.append(np.asarray(state_mask, dtype=bool))
        stats = {
            "utility_mae": float(np.mean(errors)),
            "top1_accuracy": float(np.mean(top1_hits)),
            "one_step_regret": float(np.mean(regrets)),
        }
        return (
            stats,
            np.stack(prediction_arrays),
            np.stack(truth_arrays),
            np.stack(mask_arrays),
        )

    rng_state = np.random.default_rng(args.seed + 90_000)
    random_states = np.zeros((test.episodes, test.candidate_count), dtype=bool)
    for row in range(test.episodes):
        size = int(rng_state.integers(0, config.budget))
        if size:
            random_states[row, rng_state.choice(test.candidate_count, size=size, replace=False)] = True
    oracle_states = []
    state = np.zeros_like(random_states)
    for _ in range(config.budget):
        oracle_states.append(state.copy())
        true = observed_candidate_marginals(test, state)
        choice = np.argmax(true, axis=1)
        active = true[np.arange(test.episodes), choice] > 0
        state[np.arange(test.episodes)[active], choice[active]] = True
    mur_states = []
    state = np.zeros_like(random_states)
    for _ in range(config.budget):
        mur_states.append(state.copy())
        predicted = score_state(mur_model, test, state, device=args.device)
        choice = np.argmax(predicted, axis=1)
        active = predicted[np.arange(test.episodes), choice] > 0
        state[np.arange(test.episodes)[active], choice[active]] = True
    diagnostic_payload = {}
    state_diagnostics = {}
    for family, states in (
        ("random", random_states),
        ("oracle_prefix", oracle_states),
        ("mur_rollout", mur_states),
    ):
        stats, predictions, truths, masks = diagnostics(states)
        state_diagnostics[family] = stats
        diagnostic_payload[f"{family}_prediction"] = predictions
        diagnostic_payload[f"{family}_truth"] = truths
        diagnostic_payload[f"{family}_mask"] = masks

    result = {
        "run_id": args.run_id,
        "status": args.status,
        "config": {
            **config.__dict__,
            "seed": args.seed,
            "train_episodes": args.train_episodes,
            "test_episodes": args.test_episodes,
            "epochs": args.epochs,
            "device": args.device,
            "calibration_alpha": args.calibration_alpha,
            "mmr_gamma_grid": mmr_gamma_grid,
            "mmr_gamma_selected": mmr_gamma,
            "mmr_validation": mmr_validation,
            "calibration_radius": calibrator.radius,
            "free_threshold_grid": free_threshold_grid.tolist(),
            "free_threshold_selected": free_threshold,
            "free_threshold_target_contexts": interval_context_target,
            "free_threshold_validation": free_threshold_validation,
        },
        "metrics": metrics,
        "state_diagnostics": state_diagnostics,
        "wall_seconds": time.time() - started,
        "provenance": {
            "script_sha256": sha256(Path(__file__)),
            "synthetic_sha256": sha256(ROOT / "src" / "mur" / "synthetic.py"),
            "experiment_sha256": sha256(
                ROOT / "src" / "mur" / "synthetic_experiment.py"
            ),
        },
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.save_diagnostics:
        diagnostic_payload["candidate_count"] = np.asarray([config.candidate_count], dtype=np.int64)
        diagnostic_payload["budget"] = np.asarray([config.budget], dtype=np.int64)
        diagnostic_payload["seed"] = np.asarray([args.seed], dtype=np.int64)
        np.savez_compressed(args.out / "utility_arrays.npz", **diagnostic_payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
