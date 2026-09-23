#!/usr/bin/env python3
"""Fixed-set acquisition baselines for the multi-target protocol (corrected).

Supersedes ``run_fixed_set_baselines.py``: the negative-transfer rate is now a
per-episode comparison, the cluster structure is estimated on the validation
split, and the gradient-influence score is ranked in the direction of loss
reduction.


These baselines come from the feature-selection and active-feature-acquisition
literature and are adapted to the context-selection protocol as follows. Each
one chooses a *fixed* candidate set per target using only the training and
validation splits and the frozen predictor, and then applies that set to every
test episode. They therefore answer a question the deployed router must answer:
does conditioning each decision on the episode's selected state buy anything
over the best set that can be chosen offline?

Policies
--------
forward_val          greedy forward selection by validation loss reduction
                     (best-subset / optimal-pursuit style)
forward_backward_val the same followed by one elimination pass (selection and
                     elimination criterion)
adaptive_k           relevance ordering with the budget chosen on validation
                     (adaptive-k style context selection)
ids_cluster          clustering of the candidate pool with one representative
                     per cluster (iterative demonstration selection style)
lookahead_val        two-step non-greedy acquisition chosen on validation
                     (nongreedy active feature acquisition style)

No model is trained and no test-time label is used. Output layout matches the
other sweeps: ``<out>/<dataset>/target<t>_seed<s>/result.json``.
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

from mur.traffic import expert_prediction, fit_subset_expert, make_batch, squared_error  # noqa: E402
from run_response_router_multitarget import build_candidate_pool, prepare_dataset, target_set  # noqa: E402

# Reported policies. ``forward_backward_val`` is computed as well and stored in
# the record for the audit of the elimination criterion, but it is not reported
# as a separate row: the elimination pass selected the same set as plain forward
# selection on every target-run that was audited.
POLICIES = ["forward_val", "adaptive_k", "ids_cluster", "lookahead_val", "grad_influence"]
AUDIT_POLICIES = POLICIES + ["forward_backward_val"]


def mask_for(batch, indices) -> np.ndarray:
    """Selection mask that applies the same candidate set to every episode."""

    mask = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for index in indices:
        mask[:, int(index)] = True
    return mask


def mean_loss(batch, expert, indices) -> float:
    return float(squared_error(batch, mask_for(batch, indices), expert).mean())


def forward_selection(batch, expert, budget: int, *, allow_elimination: bool) -> list[int]:
    """Greedy forward selection by validation loss, with optional elimination."""

    count = batch.candidate_count
    order: list[int] = []
    base = mean_loss(batch, expert, order)
    for _ in range(budget):
        gains = []
        for candidate in range(count):
            gains.append(-np.inf if candidate in order else base - mean_loss(batch, expert, order + [candidate]))
        best = int(np.argmax(gains))
        if gains[best] <= 0.0:
            break
        order = order + [best]
        base -= gains[best]
    if allow_elimination and len(order) >= 2:
        # One elimination pass: the classical criterion removes the weakest
        # element and reconsiders the best replacement for it. Repeating the
        # pass until no swap improves the validation loss degenerates into
        # near-exhaustive subset search, which is not the baseline we compare
        # against and is far more expensive than any other policy.
        best = (base, None)
        for position in range(len(order)):
            reduced = [j for k, j in enumerate(order) if k != position]
            for candidate in range(count):
                if candidate in order:
                    continue
                gain = base - mean_loss(batch, expert, reduced + [candidate])
                if gain > best[0]:
                    best = (gain, reduced + [candidate])
        if best[1] is not None:
            order = best[1]
    return order


def adaptive_budget(relevance: np.ndarray, batch, expert, budget: int) -> list[int]:
    """Relevance ordering with the number of contexts chosen on validation."""

    order = [int(index) for index in np.argsort(-relevance, kind="stable")[:budget]]
    base = mean_loss(batch, expert, [])
    best_k, best_gain = 1, -np.inf
    for k in range(1, budget + 1):
        gain = base - mean_loss(batch, expert, order[:k])
        if gain > best_gain:
            best_gain, best_k = gain, k
    return order[:best_k]


def cluster_representatives(batch, relevance: np.ndarray, budget: int, seed: int = 0) -> list[int]:
    """One representative per cluster of the candidate pool, by relevance."""

    vectors = np.asarray(batch.candidate_history, dtype=float).mean(axis=0)   # (K, history)
    vectors = vectors - vectors.mean(axis=0, keepdims=True)
    scale = vectors.std(axis=0, keepdims=True)
    vectors = vectors / np.maximum(scale, 1e-9)
    count = vectors.shape[0]
    rng = np.random.default_rng(seed)
    centers = [vectors[int(np.argmax(relevance))]]
    for _ in range(min(budget, count) - 1):
        distance = np.min(
            np.linalg.norm(vectors[:, None, :] - np.asarray(centers)[None, :, :], axis=2), axis=1
        )
        centers.append(vectors[int(np.argmax(distance))])
    centers = np.asarray(centers)
    assignment = np.argmin(np.linalg.norm(vectors[:, None, :] - centers[None, :, :], axis=2), axis=1)
    chosen: list[int] = []
    for cluster in range(centers.shape[0]):
        members = np.flatnonzero(assignment == cluster)
        if members.size == 0:
            continue
        chosen.append(int(members[int(np.argmax(relevance[members]))]))
    for candidate in np.argsort(-relevance, kind="stable"):
        if len(chosen) >= budget:
            break
        if int(candidate) not in chosen:
            chosen.append(int(candidate))
    return chosen[:budget]


def lookahead_selection(batch, expert, budget: int) -> list[int]:
    """Two-step non-greedy acquisition: best pair first, then greedy extension."""

    count = batch.candidate_count
    base = mean_loss(batch, expert, [])
    best_pair, best_gain = None, -np.inf
    for first in range(count):
        for second in range(count):
            if first == second:
                continue
            gain = base - mean_loss(batch, expert, [first, second])
            if gain > best_gain:
                best_gain, best_pair = gain, (first, second)
    if best_pair is None:
        return []
    order = [best_pair[0], best_pair[1]]
    base -= best_gain
    while len(order) < budget:
        gains = [
            -np.inf if candidate in order else base - mean_loss(batch, expert, order + [candidate])
            for candidate in range(count)
        ]
        best = int(np.argmax(gains))
        if gains[best] <= 0.0:
            break
        order = order + [best]
        base -= gains[best]
    return order


def gradient_influence(batch, expert, budget: int, seed: int = 0, subsets: int = 8) -> list[int]:
    """First-order influence per candidate, aggregated over random states.

    Adaptation of the gradient-estimation selection rule (Zhang et al., 2025) to
    this protocol. The response of the frozen predictor to a candidate supplies
    the first-order term of the squared-loss change,
    ``2 <f_A - y, d_j> + ||d_j||^2`` with ``d_j = f(A+j) - f(A)``; validation
    labels make the residual observable. The score of a candidate is averaged
    over randomly sampled states so that it does not depend on one state, the
    candidates are scored independently in linear time in the pool size, and the
    highest scoring candidates form the fixed set applied to the test episodes.
    """

    count = batch.candidate_count
    rng = np.random.default_rng(seed)
    totals = np.zeros(count, dtype=float)
    for _ in range(subsets):
        size = int(rng.integers(0, max(1, budget)))
        state = [int(i) for i in rng.choice(count, size=size, replace=False)] if size else []
        state_mask = mask_for(batch, state)
        base = expert_prediction(batch, state_mask, expert)
        residual = base - batch.target_future
        for candidate in range(count):
            if candidate in state:
                continue
            trial = state_mask.copy()
            trial[:, candidate] = True
            delta = expert_prediction(batch, trial, expert) - base
            # 2<d, r> + ||d||^2 is the *increase* in squared loss caused by
            # adding the candidate, so the influence score is its negation and
            # the candidates are ranked in ascending order of that quantity.
            totals[candidate] += float(
                (2.0 * (delta * residual).sum(axis=1) + (delta ** 2).sum(axis=1)).mean()
            )
    return [int(index) for index in np.argsort(totals, kind="stable")[:budget]]


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
        relevance = prepared["corr"][target_sensor][candidates].astype(np.float64)
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
            train, validation, test = batches
            expert = fit_subset_expert(train, seed=seed + 3_000, repeats=3, ridge_penalty=10.0)
            base_episode = squared_error(
                test, np.zeros((test.episodes, test.candidate_count), dtype=bool), expert
            )

            selections = {
                "forward_val": forward_selection(validation, expert, args.budget, allow_elimination=False),
                "forward_backward_val": forward_selection(validation, expert, args.budget, allow_elimination=True),
                "adaptive_k": adaptive_budget(relevance, validation, expert, args.budget),
                # The cluster structure is a statistic of the candidate pool and
                # is estimated on the validation split, never on the test set.
                "ids_cluster": cluster_representatives(validation, relevance, args.budget, seed=seed),
                "lookahead_val": lookahead_selection(validation, expert, args.budget),
                "grad_influence": gradient_influence(validation, expert, args.budget, seed=seed),
            }
            metrics = {}
            for name in AUDIT_POLICIES:
                indices = selections[name]
                mask = mask_for(test, indices)
                loss = squared_error(test, mask, expert)
                metrics[name] = {
                    # Per-episode comparison, matching src/mur/metrics.py so that
                    # the rates are comparable with every other policy family.
                    "mean_prediction_gain": float(np.mean(base_episode - loss)),
                    "negative_transfer_rate": float(np.mean(loss > base_episode)),
                    "mean_utility_recovery": float("nan"),
                    "mean_selected_contexts": float(mask.sum(axis=1).mean()),
                }
            record = {
                "run_id": "R206_fixed_set_baselines_v2",
                "status": "confirmatory",
                "dataset": args.dataset,
                "target_sensor": target_sensor,
                "seed": seed,
                "config": {
                    "candidate_count": args.candidate_count,
                    "budget": args.budget,
                    "candidate_sensors": candidates.tolist(),
                    "selected_indices": {name: [int(i) for i in indices] for name, indices in selections.items()},
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
