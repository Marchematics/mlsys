"""Verbatim snapshot of the PAMI routing helpers used by the R110 family.

Copied unchanged from ``PAMI_MUR/experiments/run_strong_backbone.py``
(md5 3f1508c8036d1947c72fa7a99367c79e) because that workspace file is edited concurrently with this
workstream; pinning the revision keeps the R110 raw runs reproducible.  The
functions below are byte-identical copies -- no logic was modified.  The frozen
R-MUR implementation itself (pair sampling, utility models, ranked calibration,
paired CIs, selection policies) is still imported live from ``KBS_MUR``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
_PAMI = _HERE.parents[2]
for _path in (_REPO / "KBS_MUR" / "src", _REPO / "KBS_MUR" / "scripts", _PAMI / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mur.synthetic_experiment import sample_pair_dataset  # noqa: E402,F401  (re-exported)
from neural_expert import SubsetExpertOps  # noqa: E402,F401  (type annotation only)
from mur.synthetic_experiment import score_state  # noqa: E402,F401


def select_oracle_static(batch, ops: SubsetExpertOps, budget: int) -> np.ndarray:
    empty = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    marginal = ops.candidate_marginals(batch, empty)
    order = np.argsort(-marginal, axis=1, kind="stable")[:, :budget]
    selected = empty.copy()
    for row in range(batch.episodes):
        for candidate in order[row]:
            if marginal[row, candidate] <= 0.0:
                break
            selected[row, candidate] = True
    return selected

def select_oracle_greedy(batch, ops: SubsetExpertOps, budget: int) -> np.ndarray:
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for _ in range(budget):
        marginal = ops.candidate_marginals(batch, selected)
        choice = np.argmax(marginal, axis=1)
        active = marginal[np.arange(batch.episodes), choice] > 0.0
        selected[np.arange(batch.episodes)[active], choice[active]] = True
    return selected

def select_random(batch, budget: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for row in range(batch.episodes):
        chosen = rng.choice(batch.candidate_count, size=budget, replace=False)
        selected[row, chosen] = True
    return selected

def candidate_similarity(batch) -> np.ndarray:
    vectors = batch.candidate_history.astype(np.float64).reshape(batch.episodes, batch.candidate_count, -1)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)
    return np.einsum("bik,bjk->bij", vectors, vectors)

def normalized_relevance(batch, relevance: np.ndarray) -> np.ndarray:
    values = np.asarray(relevance, dtype=np.float64)
    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    return (values - low) / np.maximum(high - low, 1e-9)

def select_facility_location(
    batch, relevance: np.ndarray, budget: int, *, similarity: np.ndarray | None = None
) -> np.ndarray:
    """Relevance-weighted coverage: a standard submodular selection baseline."""

    if similarity is None:
        similarity = candidate_similarity(batch)
    weights = normalized_relevance(batch, relevance) + 1e-3
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    coverage = np.zeros((batch.episodes, batch.candidate_count), dtype=np.float64)
    rows = np.arange(batch.episodes)
    for _ in range(budget):
        gains = (weights[:, None, :] * np.maximum(similarity, coverage[:, :, None])).sum(axis=2)
        gains -= (weights * coverage).sum(axis=1, keepdims=True)
        gains[selected] = -np.inf
        choice = np.argmax(gains, axis=1)
        selected[rows, choice] = True
        coverage = np.maximum(coverage, similarity[rows, :, choice])
    return selected

def select_dpp(
    batch, relevance: np.ndarray, budget: int, *, similarity: np.ndarray | None = None, jitter: float = 1e-6
) -> np.ndarray:
    """Greedy MAP for a determinantal point process (diversity baseline)."""

    if similarity is None:
        similarity = candidate_similarity(batch)
    quality = normalized_relevance(batch, relevance) + 0.1
    kernel = quality[:, :, None] * similarity * quality[:, None, :]
    episodes, candidates = kernel.shape[:2]
    selected = np.zeros((episodes, candidates), dtype=bool)
    rows = np.arange(episodes)
    diagonal = np.diagonal(kernel, axis1=1, axis2=2).copy()
    for _ in range(budget):
        sub_index = [np.flatnonzero(selected[row]) for row in range(episodes)]
        residual = diagonal.copy()
        for row in range(episodes):
            index = sub_index[row]
            if index.size == 0:
                continue
            block = kernel[row][np.ix_(index, index)]
            chol = np.linalg.cholesky(block + jitter * np.eye(index.size))
            cross = kernel[row][np.ix_(index, np.arange(candidates))]
            solved = np.linalg.solve(chol, cross)
            residual[row] -= np.sum(solved ** 2, axis=0)
        residual[selected] = -np.inf
        choice = np.argmax(residual, axis=1)
        selected[rows, choice] = True
    return selected

def select_ranked_response_mur(
    screen_model,
    response_model,
    batch,
    ops: SubsetExpertOps,
    *,
    budget: int,
    q: int,
    device: str,
    projection: np.ndarray | None = None,
    include_interaction: bool = False,
    return_trace: bool = False,
):
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    candidate_count = batch.candidate_count
    q = max(1, min(int(q), candidate_count))
    trace = {key: [] for key in ("choice", "topq", "response_scores")}
    expert_rows = 0
    for _ in range(budget):
        before = ops.rows
        screen = score_state(screen_model, batch, selected, device=device, pack_fn=ops.pack)
        topq = np.argsort(-screen, axis=1, kind="stable")[:, :q]
        qmask = np.zeros_like(selected, dtype=bool)
        qmask[np.arange(batch.episodes)[:, None], topq] = True
        responses = ops.response_summary(
            batch, selected, candidate_mask=qmask, projection=projection,
            include_interaction=include_interaction,
        )

        def response_fn(batch_, selected_):
            return responses

        response_scores = score_state(
            response_model,
            batch,
            selected,
            device=device,
            pack_fn=ops.pack,
            response_fn=response_fn,
        )
        expert_rows += ops.rows - before
        masked = np.where(qmask, response_scores, -np.inf)
        choice = np.argmax(masked, axis=1)
        best = masked[np.arange(batch.episodes), choice]
        active = best > 0.0
        selected[np.arange(batch.episodes)[active], choice[active]] = True
        if return_trace:
            trace["choice"].append(choice.astype(np.int64))
            trace["topq"].append(topq.astype(np.int64))
            trace["response_scores"].append(response_scores.astype(np.float32))
    if return_trace:
        for key in trace:
            trace[key] = np.stack(trace[key], axis=0)
        return selected, trace, expert_rows
    return selected, expert_rows

def sample_pair_dataset_fast(
    batch,
    ops: SubsetExpertOps,
    *,
    max_budget: int,
    states_per_episode: int,
    seed: int,
    static_only: bool,
):
    """Pair dataset with the sampled candidate's marginal only.

    Identical to ``sample_pair_dataset`` for the marginals it consumes (the
    reference implementation evaluates the full pool and then indexes one
    column); evaluating just the sampled candidate avoids a factor ``K`` of
    redundant expert calls.
    """

    from mur.synthetic_experiment import PairDataset

    rng = np.random.default_rng(seed)
    if max_budget < 1 or max_budget > batch.candidate_count:
        raise ValueError("max_budget must be in [1, candidate_count]")
    pieces: dict[str, list[np.ndarray]] = {
        key: [] for key in ("query", "selected", "selected_mask", "candidate", "cost", "target", "group_id")
    }
    rows = np.arange(batch.episodes)
    for repetition in range(states_per_episode):
        selected_mask = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
        if not static_only:
            sizes = rng.integers(0, max_budget, size=batch.episodes)
            for row, size in enumerate(sizes):
                if size:
                    selected_mask[row, rng.choice(batch.candidate_count, size=int(size), replace=False)] = True
        candidate_index = np.empty(batch.episodes, dtype=np.int64)
        for row in rows:
            candidate_index[row] = int(rng.choice(np.flatnonzero(~selected_mask[row])))
        marginal = ops.pair_marginals(batch, selected_mask, candidate_index)
        selected, selected_valid = ops.pack(batch, selected_mask, max_budget=max_budget)
        pieces["query"].append(batch.query_embedding)
        pieces["selected"].append(selected)
        pieces["selected_mask"].append(selected_valid)
        pieces["candidate"].append(batch.candidate_embedding[rows, candidate_index])
        pieces["cost"].append(np.ones((batch.episodes, 1), dtype=np.float32))
        pieces["target"].append(marginal.astype(np.float32))
        pieces["group_id"].append((repetition * batch.episodes + rows).astype(np.int64))
    return PairDataset(**{key: np.concatenate(value) for key, value in pieces.items()})

def build_full_state_data(
    batch,
    ops: SubsetExpertOps,
    *,
    max_budget: int,
    states_per_episode: int,
    seed: int,
    projection: np.ndarray | None = None,
    include_interaction: bool = False,
):
    rng = np.random.default_rng(seed)
    query_rows, selected_rows, valid_rows = [], [], []
    candidate_rows, response_rows, target_rows = [], [], []
    for _ in range(states_per_episode):
        selected_mask = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
        sizes = rng.integers(0, max_budget, size=batch.episodes)
        for row, size in enumerate(sizes):
            if size:
                chosen = rng.choice(batch.candidate_count, size=int(size), replace=False)
                selected_mask[row, chosen] = True
        selected, selected_valid = ops.pack(batch, selected_mask, max_budget=max_budget)
        marginal, responses = ops.candidate_marginals_and_responses(
            batch, selected_mask, projection=projection, include_interaction=include_interaction
        )
        query_rows.append(batch.query_embedding)
        selected_rows.append(selected)
        valid_rows.append(selected_valid.astype(bool))
        candidate_rows.append(batch.candidate_embedding)
        response_rows.append(responses)
        target_rows.append(marginal.astype(np.float32))
    return {
        "query": np.concatenate(query_rows, axis=0),
        "selected": np.concatenate(selected_rows, axis=0),
        "selected_mask": np.concatenate(valid_rows, axis=0),
        "candidate": np.concatenate(candidate_rows, axis=0),
        "response": np.concatenate(response_rows, axis=0),
        "target": np.concatenate(target_rows, axis=0),
    }
