#!/usr/bin/env python3
"""Standard budgeted-subset baselines for the multi-target protocol.

Adds the canonical subset-selection families that the routing comparison does
not yet contain, evaluated on exactly the same episodes, candidate pools and
frozen predictor as the multi-target routing runs:

* facility location (submodular coverage of the candidate pool),
* relevance-weighted facility location,
* greedy DPP (log-determinant maximisation),
* k-center (max-min diversity).

None of these trains a model or evaluates the predictor during selection, so
the run is an evaluation-only comparison. Output layout matches the routing
runs: ``<out>/<dataset>/target<t>_seed<s>/result.json`` with a ``metrics``
mapping keyed by policy name.
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
from run_response_router_multitarget import build_candidate_pool, prepare_dataset, target_set  # noqa: E402

POLICIES = ["facility_location", "facility_location_relevance", "dpp_greedy", "kcenter"]
RELEVANCE_WEIGHT = 0.5


def cosine_similarity(batch) -> np.ndarray:
    """Cosine similarity between candidate histories, shape (episodes, K, K)."""

    vectors = np.asarray(batch.candidate_history, dtype=float)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)
    return np.einsum("bik,bjk->bij", vectors, vectors)


def greedy_submodular(similarity: np.ndarray, budget: int, relevance=None, weight: float = 0.0) -> np.ndarray:
    """Greedy maximisation of facility location, optionally relevance-weighted.

    The objective is the coverage sum over pool members,
    ``sum_i max_{k in A} s_ik``, which is monotone submodular; the greedy rule
    is the standard (1 - 1/e) approximation. Adding ``weight * relevance``
    turns it into a relevance-aware coverage rule.
    """

    episodes, candidates, _ = similarity.shape
    selected = np.zeros((episodes, candidates), dtype=bool)
    coverage = np.zeros((episodes, candidates), dtype=float)
    for _ in range(min(budget, candidates)):
        gain = np.maximum(similarity, coverage[:, :, None]).sum(axis=1) - coverage.sum(axis=1, keepdims=True)
        if relevance is not None and weight > 0.0:
            gain = gain + weight * relevance
        gain[selected] = -np.inf
        choice = np.argmax(gain, axis=1)
        selected[np.arange(episodes), choice] = True
        coverage = np.maximum(coverage, similarity[np.arange(episodes), choice, :])
    return selected


def greedy_dpp(similarity: np.ndarray, budget: int, *, jitter: float = 1e-3) -> np.ndarray:
    """Greedy MAP for a determinantal point process (log-determinant objective)."""

    episodes, candidates, _ = similarity.shape
    kernel = similarity + jitter * np.eye(candidates)[None, :, :]
    selected = np.zeros((episodes, candidates), dtype=bool)
    chosen: list[np.ndarray] = []
    for _ in range(min(budget, candidates)):
        gains = np.empty((episodes, candidates), dtype=float)
        for candidate in range(candidates):
            gains[:, candidate] = _logdet_gain(kernel, chosen, candidate)
        gains[selected] = -np.inf
        choice = np.argmax(gains, axis=1)
        selected[np.arange(episodes), choice] = True
        chosen.append(choice)
    return selected


def _logdet_gain(kernel: np.ndarray, chosen: list[np.ndarray], candidate: int) -> np.ndarray:
    """Log-determinant increase from adding one item, computed per episode."""

    episodes = kernel.shape[0]
    rows = np.arange(episodes)
    if not chosen:
        return np.log(np.maximum(kernel[rows, candidate, candidate], 1e-12))
    index = np.stack(chosen, axis=1)                                 # (E, m)
    sub = kernel[rows[:, None, None], index[:, :, None], index[:, None, :]]
    cross = kernel[rows[:, None], index, np.full_like(index, candidate)]
    diagonal = np.maximum(kernel[rows, candidate, candidate], 1e-12)
    solved = np.linalg.solve(sub, cross[:, :, None])[:, :, 0]        # (E, m)
    schur = diagonal - np.einsum("em,em->e", cross, solved)
    return np.log(np.maximum(schur, 1e-12))


def greedy_kcenter(similarity: np.ndarray, budget: int) -> np.ndarray:
    """Greedy max-min diversity: each pick is farthest from the selected set."""

    episodes, candidates, _ = similarity.shape
    distance = 1.0 - similarity
    selected = np.zeros((episodes, candidates), dtype=bool)
    best = np.full((episodes, candidates), np.inf)
    for _ in range(min(budget, candidates)):
        score = best.copy()
        score[selected] = -np.inf
        choice = np.argmax(score, axis=1)
        selected[np.arange(episodes), choice] = True
        best = np.minimum(best, distance[np.arange(episodes), choice, :])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data" / "staeformer")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", default="101")
    parser.add_argument("--target-count", type=int, default=32)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--calibration-episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=2000)
    args = parser.parse_args()

    prepared = prepare_dataset(args.data_root / args.dataset)
    targets = target_set(prepared["num_nodes"], args.target_count)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    dataset_root = args.out_root / args.dataset
    if dataset_root.exists():
        raise FileExistsError(f"dataset output already exists: {dataset_root}")
    dataset_root.mkdir(parents=True)

    for target_sensor in targets:
        target_sensor = int(target_sensor)
        candidates = build_candidate_pool(
            prepared["corr"][target_sensor],
            target_sensor=target_sensor,
            candidate_count=args.candidate_count,
            seed=20260920,
        )
        corr = prepared["corr"][target_sensor]
        relevance_row = corr[candidates].astype(np.float64)
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
            base = squared_error(test, np.zeros((test.episodes, test.candidate_count), dtype=bool), expert)
            similarity = cosine_similarity(test)
            relevance = np.broadcast_to(relevance_row[None, :], (test.episodes, test.candidate_count)).copy()
            selections = {
                "facility_location": greedy_submodular(similarity, args.budget),
                "facility_location_relevance": greedy_submodular(
                    similarity, args.budget, relevance=relevance, weight=RELEVANCE_WEIGHT
                ),
                "dpp_greedy": greedy_dpp(similarity, args.budget),
                "kcenter": greedy_kcenter(similarity, args.budget),
            }
            metrics = {}
            for name in POLICIES:
                selected = selections[name]
                loss = squared_error(test, selected, expert)
                metrics[name] = {
                    "mean_prediction_gain": float(np.mean(base - loss)),
                    "negative_transfer_rate": float(np.mean(loss > base)),
                    "mean_utility_recovery": float("nan"),
                    "mean_selected_contexts": float(selected.sum(axis=1).mean()),
                }
            record = {
                "run_id": "R202_subset_baselines",
                "status": "confirmatory",
                "dataset": args.dataset,
                "target_sensor": target_sensor,
                "seed": seed,
                "config": {
                    "candidate_count": args.candidate_count,
                    "budget": args.budget,
                    "relevance_weight": RELEVANCE_WEIGHT,
                    "policies": POLICIES,
                    "candidate_sensors": candidates.tolist(),
                },
                "metrics": metrics,
                "wall_seconds": time.time() - started,
            }
            seed_dir = args.out_root / args.dataset / f"target{target_sensor}_seed{seed}"
            seed_dir.mkdir(parents=True)
            (seed_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"completed target {target_sensor} for {args.dataset}", flush=True)


if __name__ == "__main__":
    main()
