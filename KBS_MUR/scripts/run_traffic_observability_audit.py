#!/usr/bin/env python3
"""Diagnostic probe ladder for traffic marginal-utility observability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_traffic_full_router import load_protocol  # noqa: E402
from mur.traffic import (  # noqa: E402
    TrafficBatch,
    expert_prediction,
    fit_subset_expert,
    observed_candidate_marginals,
)


def _state_masks(batch: TrafficBatch, budget: int, states_per_episode: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(states_per_episode):
        state = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
        for row in range(batch.episodes):
            size = int(rng.integers(0, budget))
            if size:
                state[row, rng.choice(batch.candidate_count, size=size, replace=False)] = True
        states.append(state)
    return states


def _build_rows(
    batch: TrafficBatch,
    expert,
    *,
    budget: int,
    states_per_episode: int,
    seed: int,
) -> dict[str, np.ndarray]:
    x1_rows, x2_rows, x3_rows = [], [], []
    labels, state_ids, candidate_ids = [], [], []
    state_offset = 0
    for state in _state_masks(batch, budget, states_per_episode, seed):
        base_prediction = expert_prediction(batch, state, expert)
        marginal = observed_candidate_marginals(batch, state, expert)
        selected_count = np.sum(state, axis=1, keepdims=True).astype(np.float32)
        selected_sum = np.einsum("bk,bkh->bh", state.astype(np.float32), batch.candidate_history)
        selected_count_safe = np.maximum(selected_count, 1.0)
        selected_mean = selected_sum / selected_count_safe
        for candidate in range(batch.candidate_count):
            active = ~state[:, candidate]
            if not np.any(active):
                continue
            augmented = state.copy()
            augmented[:, candidate] = True
            augmented_prediction = expert_prediction(batch, augmented, expert)
            query = batch.anchor_history[active]
            selected = selected_mean[active]
            candidate_history = batch.candidate_history[active, candidate]
            difference = selected - candidate_history
            product = selected * candidate_history
            size = selected_count[active]
            x1_rows.append(
                np.concatenate([query, selected, candidate_history, difference, product, size], axis=1)
            )
            response_summary = np.column_stack(
                [
                    np.mean(base_prediction[active], axis=1),
                    np.mean(augmented_prediction[active], axis=1),
                    np.mean(augmented_prediction[active] - base_prediction[active], axis=1),
                    np.mean(np.abs(augmented_prediction[active] - base_prediction[active]), axis=1),
                ]
            )
            x2_rows.append(
                np.concatenate([x1_rows[-1][-int(np.sum(active)) :], response_summary], axis=1)
            )
            x3_rows.append(
                np.concatenate(
                    [
                        x2_rows[-1][-int(np.sum(active)) :],
                        base_prediction[active],
                        augmented_prediction[active],
                        augmented_prediction[active] - base_prediction[active],
                    ],
                    axis=1,
                )
            )
            labels.append(marginal[active, candidate].astype(np.float32))
            state_ids.append(
                (np.arange(batch.episodes)[active] + state_offset * batch.episodes).astype(np.int64)
            )
            candidate_ids.append(np.full(np.sum(active), candidate, dtype=np.int64))
        state_offset += 1
    return {
        "x1": np.concatenate(x1_rows).astype(np.float32),
        "x2": np.concatenate(x2_rows).astype(np.float32),
        "x3": np.concatenate(x3_rows).astype(np.float32),
        "label": np.concatenate(labels).astype(np.float32),
        "state_id": np.concatenate(state_ids),
        "candidate_id": np.concatenate(candidate_ids),
    }


def _fit_probe(x_train, y_train, x_cal, y_cal, *, classification: bool, seed: int):
    candidates = []
    for depth in (3, 6, None):
        for l2 in (1e-3, 1.0):
            if classification:
                model = HistGradientBoostingClassifier(
                    max_iter=150, learning_rate=.08, max_depth=depth,
                    l2_regularization=l2, random_state=seed
                )
                model.fit(x_train, y_train > 0.0)
                score = model.predict_proba(x_cal)[:, 1]
                metric = roc_auc_score(y_cal > 0.0, score)
                candidates.append((metric, model))
            else:
                model = HistGradientBoostingRegressor(
                    max_iter=150, learning_rate=.08, max_depth=depth,
                    l2_regularization=l2, random_state=seed
                )
                model.fit(x_train, y_train)
                prediction = model.predict(x_cal)
                metric = -float(np.mean(np.abs(prediction - y_cal)))
                candidates.append((metric, model))
    return max(candidates, key=lambda item: item[0])[1]


def _evaluate(regressor, classifier, rows: dict[str, np.ndarray], candidate_count: int) -> dict[str, float]:
    prediction = regressor.predict(rows["x"])
    label = rows["label"]
    sign_score = classifier.predict_proba(rows["x"])[:, 1]
    by_state = {}
    for index, state_id in enumerate(rows["state_id"]):
        by_state.setdefault(int(state_id), []).append(index)
    top1_hits, regrets = [], []
    for indices in by_state.values():
        indices = np.asarray(indices, dtype=np.int64)
        pred_best = indices[np.argmax(prediction[indices])]
        true_best = indices[np.argmax(label[indices])]
        top1_hits.append(float(rows["candidate_id"][pred_best] == rows["candidate_id"][true_best]))
        regrets.append(float(label[true_best] - label[pred_best]))
    return {
        "mae": float(np.mean(np.abs(prediction - label))),
        "spearman": float(spearmanr(prediction, label).statistic),
        "positive_auc": float(roc_auc_score(label > 0.0, sign_score)),
        "top1_accuracy": float(np.mean(top1_hits)),
        "one_step_regret": float(np.mean(regrets)),
        "positive_rate": float(np.mean(label > 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--gate-result", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--states-per-episode", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    cfg, (train, calibration, test), _ = load_protocol(args.data_path, args.gate_result, args.seed)
    expert = fit_subset_expert(
        train,
        seed=args.seed + 3_000,
        repeats=int(cfg["expert_repeats"]),
        ridge_penalty=float(cfg["ridge_penalty"]),
    )
    train_rows = _build_rows(
        train, expert, budget=int(cfg["budget"]), states_per_episode=args.states_per_episode, seed=args.seed + 4_000
    )
    calibration_rows = _build_rows(
        calibration, expert, budget=int(cfg["budget"]), states_per_episode=args.states_per_episode, seed=args.seed + 5_000
    )
    test_rows = _build_rows(
        test, expert, budget=int(cfg["budget"]), states_per_episode=args.states_per_episode, seed=args.seed + 6_000
    )
    results = {}
    for family in ("x1", "x2", "x3"):
        train_family = {"x": train_rows[family], "label": train_rows["label"], "state_id": train_rows["state_id"], "candidate_id": train_rows["candidate_id"]}
        cal_family = {"x": calibration_rows[family], "label": calibration_rows["label"], "state_id": calibration_rows["state_id"], "candidate_id": calibration_rows["candidate_id"]}
        test_family = {"x": test_rows[family], "label": test_rows["label"], "state_id": test_rows["state_id"], "candidate_id": test_rows["candidate_id"]}
        regressor = _fit_probe(
            train_family["x"], train_family["label"], cal_family["x"], cal_family["label"], classification=False, seed=args.seed
        )
        classifier = _fit_probe(
            train_family["x"], train_family["label"], cal_family["x"], cal_family["label"], classification=True, seed=args.seed
        )
        results[family] = {
            "feature_dim": int(train_family["x"].shape[1]),
            "metrics": _evaluate(regressor, classifier, test_family, int(cfg["candidate_count"])),
        }
    result = {
        "run_id": "R076_traffic_utility_observability",
        "status": "diagnostic_probe",
        "config": {
            "seed": args.seed,
            "states_per_episode": args.states_per_episode,
            "candidate_count": int(cfg["candidate_count"]),
            "budget": int(cfg["budget"]),
            "feature_definition": {
                "x1": "raw query/selected/candidate history interactions",
                "x2": "x1 plus scalar expert response summaries",
                "x3": "x2 plus full expert forecast trajectories and delta",
            },
        },
        "results": results,
        "wall_seconds": time.time() - started,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
