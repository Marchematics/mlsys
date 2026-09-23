#!/usr/bin/env python3
"""Response-geometry baselines that need no utility training.

The exact identity gives m(j|A) = <g_A(x), d_{A,j}> - ||d_{A,j}||^2 - lambda c_j.
When the first-order term is not identifiable (which the observability probe
measures), the only *observable* term is the curvature penalty, so the
predicted-best candidates are those that disturb the frozen predictor least.
This script evaluates that counter-intuitive policy, its opposite, and simple
geometry controls on exactly the cells of the strong-backbone gate run, so the
episode gains can be paired with the stored ones.

Policies (all select exactly `budget` candidates):
  min_disturbance_static : smallest ||d||^2 at the empty state
  min_disturbance_greedy : smallest ||d||^2 recomputed at every step
  max_disturbance_static : largest ||d||^2 at the empty state (control)
  min_step_static        : smallest single-step change at the empty state
  last_k                 : the pool tail (least correlated candidates)
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

from neural_expert import SubsetExpertOps, finetune_expert, load_checkpoint  # noqa: E402
from pami_traffic import build_candidate_pool, load_traffic_data, make_neural_batch  # noqa: E402

POLICIES = (
    "consensus_alignment",
    "anti_consensus_alignment",
    "min_disturbance_static",
    "min_disturbance_greedy",
    "max_disturbance_static",
    "last_k",
    "mutual_information",
    "kcenter",
    "kmeans_representatives",
    "delfit_style",
)


def select_consensus_alignment(batch, ops, budget: int, *, agree: bool = True) -> np.ndarray:
    """Keep the contexts that move the predictor the way the full pool does.

    A label-free self-consistency rule: take the direction in which using *all*
    contexts moves the anchor-only prediction, and keep the candidates whose own
    response is most aligned with it (or most opposed, for the control). It uses
    no utility labels and no learned score, only the predictor's responses at
    two states plus one pooled state.
    """

    episodes, candidates = batch.episodes, batch.candidate_count
    empty = np.zeros((episodes, candidates), dtype=bool)
    base = ops.predict(batch, empty)
    rows, columns = np.nonzero(~empty)
    delta = ops.pair_predict(batch, empty, rows, columns) - base[rows]
    delta = delta.reshape(episodes, candidates, -1)
    full = ops.predict(batch, np.ones((episodes, candidates), dtype=bool))
    direction = (full - base)[:, None, :]
    alignment = (delta * direction).sum(axis=2)
    order = np.argsort(-alignment if agree else alignment, axis=1, kind="stable")[:, :budget]
    selected = np.zeros((episodes, candidates), dtype=bool)
    selected[np.arange(episodes)[:, None], order] = True
    return selected


def select_mutual_information(batch, budget: int) -> np.ndarray:
    """Greedy Gaussian sensor selection (conditional variance reduction).

    Each candidate contributes its own block of history features, and the
    target contributes the forecast horizon. Under the pooled training
    covariance the greedy rule adds the candidate that most reduces the
    conditional variance (equivalently, the largest conditional mutual
    information) of the target given the already selected candidates. This is
    the classical sensor-placement criterion and is label-free at selection
    time: it uses only the covariance estimated on the training split.
    """

    history = batch.candidate_history.astype(np.float64)
    episodes, candidates, width = history.shape
    future = batch.target_future.astype(np.float64)
    horizon = future.shape[1]
    joint = np.concatenate([history.reshape(episodes, candidates * width), future], axis=1)
    joint = joint - joint.mean(axis=0, keepdims=True)
    covariance = np.cov(joint, rowvar=False)
    covariance += 1e-6 * np.eye(covariance.shape[0])
    split = candidates * width
    feature_block = covariance[:split, :split]
    target_cross = covariance[split:, :split]
    target_block = covariance[split:, split:]

    selected: list[int] = []
    chosen: list[int] = []
    for _ in range(min(budget, candidates)):
        best_score, best_candidate = -np.inf, None
        for candidate in range(candidates):
            if candidate in selected:
                continue
            index = chosen + list(range(candidate * width, (candidate + 1) * width))
            sub = feature_block[np.ix_(index, index)]
            cross = target_cross[:, index]
            solved = np.linalg.solve(sub, cross.T)
            residual = target_block - cross @ solved
            score = -float(np.trace(residual))
            if score > best_score:
                best_score, best_candidate = score, candidate
        if best_candidate is None:
            break
        selected.append(best_candidate)
        chosen += list(range(best_candidate * width, (best_candidate + 1) * width))
    mask = np.zeros((episodes, candidates), dtype=bool)
    mask[:, selected] = True
    return mask


def candidate_vectors(batch) -> np.ndarray:
    """Flattened candidate histories, one row per (episode, candidate)."""

    episodes, candidates = batch.episodes, batch.candidate_count
    return batch.candidate_history.reshape(episodes * candidates, -1).astype(np.float64)


def select_kcenter(batch, budget: int, seed: int = 0) -> np.ndarray:
    """Greedy k-center: maximise the minimum distance to the selected set."""

    vectors = candidate_vectors(batch).reshape(batch.episodes, batch.candidate_count, -1)
    norms = np.maximum(np.linalg.norm(vectors, axis=2, keepdims=True), 1e-12)
    unit = vectors / norms
    similarity = np.einsum("bik,bjk->bij", unit, unit)
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    rows = np.arange(batch.episodes)
    start = int(np.argmax(np.linalg.norm(vectors, axis=2).mean(axis=0)))
    selected[:, start] = True
    for _ in range(budget - 1):
        masked = np.where(selected[:, None, :], similarity, -np.inf)
        coverage = masked.max(axis=2)
        scores = np.where(selected, np.inf, coverage)
        choice = np.argmin(scores, axis=1)
        selected[rows, choice] = True
    return selected


def select_kmeans_representatives(batch, budget: int, seed: int = 0) -> np.ndarray:
    """DELIFT-style representative selection.

    The cluster structure is estimated once per cell on the episode-averaged
    candidate representation, and each episode keeps the member of every
    cluster closest to that cluster's episode-specific centre.
    """

    from sklearn.cluster import KMeans

    history = batch.candidate_history
    episodes, candidates = batch.episodes, batch.candidate_count
    pooled = history.mean(axis=0)
    kmeans = KMeans(n_clusters=min(budget, candidates), n_init=4, random_state=seed).fit(pooled)
    labels = kmeans.labels_
    selected = np.zeros((episodes, candidates), dtype=bool)
    for cluster in range(int(labels.max()) + 1):
        members = np.flatnonzero(labels == cluster)
        if members.size == 0:
            continue
        block = history[:, members]
        centre = block.mean(axis=1, keepdims=True)
        distance = np.linalg.norm(block - centre, axis=2)
        medoid = members[np.argmin(distance, axis=1)]
        selected[np.arange(episodes), medoid] = True
    missing = budget - selected.sum(axis=1)
    for row in np.flatnonzero(missing > 0):
        remaining = np.flatnonzero(~selected[row])
        selected[row, remaining[: missing[row]]] = True
    return selected


def select_delfit_style(batch, relevance, budget: int, *, similarity: np.ndarray, seed: int = 0) -> np.ndarray:
    """Facility location evaluated one cluster at a time (DELIFT-style).

    DELIFT (ICLR 2025) combines a submodular coverage objective with k-means
    representatives. ``select_facility_location`` and
    ``select_kmeans_representatives`` implement the two components separately;
    this policy is their composition and is the baseline the paper reports
    under that name. k-means over-segments the pool into more clusters than the
    budget, so that every cluster has a medoid but the medoids alone do not fit
    the budget; the selection is then filled greedily by facility-location
    marginal gain over those medoids. With ``n_clusters == budget`` the two
    components would coincide and the policy would be identical to
    ``select_kmeans_representatives``, which is why the pool is
    over-segmented.
    """

    from sklearn.cluster import KMeans

    history = batch.candidate_history
    episodes, candidates = batch.episodes, batch.candidate_count
    pooled = history.mean(axis=0)
    # over-segment: medoids are the candidate pool for the coverage stage
    cluster_count = int(min(candidates, max(budget + 1, 2 * budget)))
    kmeans = KMeans(n_clusters=cluster_count, n_init=4, random_state=seed).fit(pooled)
    labels = kmeans.labels_

    # Facility-location coverage of the pool by the currently selected set.
    sim = np.asarray(similarity, dtype=np.float64)
    if sim.ndim == 2:  # (candidates, candidates) -- broadcast over episodes
        sim = np.broadcast_to(sim[None, :, :], (episodes, candidates, candidates))
    selected = np.zeros((episodes, candidates), dtype=bool)
    best = np.zeros((episodes, candidates), dtype=np.float64)
    rows = np.arange(episodes)
    for _ in range(budget):
        marginal = np.maximum(sim - best[:, None, :], 0.0).sum(axis=2)  # (episodes, candidates)
        # restrict to candidates that are representative of some cluster
        allowed = np.zeros_like(selected)
        for cluster in range(int(labels.max()) + 1):
            members = np.flatnonzero(labels == cluster)
            if members.size == 0:
                continue
            block = history[:, members]
            centre = block.mean(axis=1, keepdims=True)
            distance = np.linalg.norm(block - centre, axis=2)
            medoid = members[np.argmin(distance, axis=1)]
            allowed[rows, medoid] = True
        allowed &= ~selected
        # fall back to the whole pool for rows with no unused representative
        empty = ~allowed.any(axis=1)
        if empty.any():
            allowed[empty] = ~selected[empty]
        scores = np.where(allowed, marginal, -np.inf)
        choice = np.argmax(scores, axis=1)
        selected[rows, choice] = True
        best = np.maximum(best, sim[rows, choice])
    return selected



    """Squared response magnitude of every eligible candidate, (episodes, K)."""

    summary = ops.response_summary(batch, selected)
    magnitude = summary[..., 5].astype(np.float64) ** 2 * batch.horizon
    magnitude[selected] = np.inf
    return magnitude


def disturbance(ops: SubsetExpertOps, batch, selected: np.ndarray) -> np.ndarray:
    """Squared prediction change caused by adding each remaining candidate."""

    episodes, candidates = batch.episodes, batch.candidate_count
    base = ops.predict(batch, selected)
    rows = np.repeat(np.arange(episodes), candidates)
    columns = np.tile(np.arange(candidates), episodes)
    moved = ops.pair_predict(batch, selected, rows, columns)
    delta = (moved - base[rows]).reshape(episodes, candidates, -1)
    return np.einsum("bch,bch->bc", delta, delta)


def select_by_magnitude(batch, ops, budget: int, *, smallest: bool, greedy: bool) -> np.ndarray:
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    rows = np.arange(batch.episodes)
    if not greedy:
        magnitude = disturbance(ops, batch, selected)
        order = np.argsort(magnitude if smallest else -magnitude, axis=1, kind="stable")[:, :budget]
        selected[rows[:, None], order] = True
        return selected
    for _ in range(budget):
        magnitude = disturbance(ops, batch, selected)
        choice = np.argmin(magnitude, axis=1) if smallest else np.argmax(magnitude, axis=1)
        selected[rows, choice] = True
    return selected


def select_last_k(batch, budget: int) -> np.ndarray:
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    selected[:, -budget:] = True
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/staeformer")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--targets", default="", help="explicit target list; default 16 evenly spaced")
    parser.add_argument("--target-count", type=int, default=16)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--calibration-episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=2000)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)
    data = load_traffic_data(args.data_root / args.dataset)
    targets = (
        np.asarray([int(t) for t in args.targets.split(",") if t.strip()], dtype=np.int64)
        if args.targets.strip()
        else np.unique(np.linspace(0, data.num_nodes - 1, args.target_count, dtype=np.int64))
    )
    for target in targets:
        target = int(target)
        candidates = build_candidate_pool(
            data.correlation[target], target_sensor=target,
            candidate_count=args.candidate_count, seed=20260920,
        )
        for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
            started = time.time()
            model, _ = load_checkpoint(args.expert_root / f"seed{seed}" / "expert.pt", device=args.device)
            model = copy.deepcopy(model)
            rng = np.random.default_rng(seed + 2_000)
            batches = []
            for starts, count in (
                (data.train_times, args.train_episodes),
                (data.validation_times, args.calibration_episodes),
                (data.test_times, args.test_episodes),
            ):
                times = rng.choice(starts, size=min(count, len(starts)), replace=False)
                batches.append(make_neural_batch(data, times, target_sensor=target, candidate_sensors=candidates))
            batches = tuple(batches)
            finetune_expert(
                model, batches[0], epochs=args.finetune_epochs, batch_size=256, learning_rate=3e-4,
                budget=args.budget, seed=seed + 7_777, device=args.device, validation_batch=batches[1],
            )
            ops = SubsetExpertOps(model, device=args.device)
            test = batches[2]
            empty = np.zeros((test.episodes, test.candidate_count), dtype=bool)
            base_loss = ops.squared_error(test, empty)
            gains = {}
            gains["min_disturbance_static"] = base_loss - ops.squared_error(
                test, select_by_magnitude(test, ops, args.budget, smallest=True, greedy=False)
            )
            gains["max_disturbance_static"] = base_loss - ops.squared_error(
                test, select_by_magnitude(test, ops, args.budget, smallest=False, greedy=False)
            )
            gains["last_k"] = base_loss - ops.squared_error(test, select_last_k(test, args.budget))
            gains["min_disturbance_greedy"] = base_loss - ops.squared_error(
                test, select_by_magnitude(test, ops, args.budget, smallest=True, greedy=True)
            )
            gains["consensus_alignment"] = base_loss - ops.squared_error(
                test, select_consensus_alignment(test, ops, args.budget, agree=True)
            )
            gains["anti_consensus_alignment"] = base_loss - ops.squared_error(
                test, select_consensus_alignment(test, ops, args.budget, agree=False)
            )
            gains["mutual_information"] = base_loss - ops.squared_error(
                test, select_mutual_information(test, args.budget)
            )
            gains["kcenter"] = base_loss - ops.squared_error(test, select_kcenter(test, args.budget))
            gains["kmeans_representatives"] = base_loss - ops.squared_error(
                test, select_kmeans_representatives(test, args.budget)
            )
            cell = args.out_root / f"target{target}_seed{seed}"
            cell.mkdir(parents=True)
            np.savez_compressed(
                cell / "episode_gains.npz",
                **{name: np.asarray(values, dtype=np.float32) for name, values in gains.items()},
            )
            (cell / "result.json").write_text(
                json.dumps(
                    {
                        "dataset": args.dataset,
                        "target_sensor": target,
                        "seed": seed,
                        "policies": {
                            name: {
                                "mean_prediction_gain": float(np.mean(values)),
                                "negative_transfer_rate": float(np.mean(values < 0.0)),
                                "mean_selected_contexts": float(args.budget),
                            }
                            for name, values in gains.items()
                        },
                        "wall_seconds": time.time() - started,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "target": target,
                        "seed": seed,
                        **{name: round(float(np.mean(v)), 5) for name, v in gains.items()},
                    }
                ),
                flush=True,
            )
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
