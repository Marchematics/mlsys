"""Training-pair construction and policy evaluation for controlled gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .metrics import decision_metrics
from .calibration import SymmetricUtilityCalibrator
from .model import MarginalUtilityRouter, RouterConfig, utility_training_loss
from .synthetic import (
    SyntheticBatch,
    observed_candidate_marginals,
    selected_set_tensors,
    squared_error,
)


@dataclass(frozen=True)
class PairDataset:
    query: np.ndarray
    selected: np.ndarray
    selected_mask: np.ndarray
    candidate: np.ndarray
    cost: np.ndarray
    target: np.ndarray
    group_id: np.ndarray
    prediction_delta: np.ndarray | None = None
    candidate_response: np.ndarray | None = None


def sample_pair_dataset(
    batch: SyntheticBatch,
    *,
    max_budget: int,
    states_per_episode: int,
    seed: int,
    static_only: bool,
    marginal_fn= None,
    pack_fn=None,
    delta_fn=None,
    response_fn=None,
) -> PairDataset:
    """Sample candidate labels at empty or nonempty selected-set states."""

    rng = np.random.default_rng(seed)
    if marginal_fn is None:
        from .synthetic import observed_candidate_marginals as marginal_fn
    if pack_fn is None:
        pack_fn = selected_set_tensors
    if max_budget < 1 or max_budget > batch.candidate_count:
        raise ValueError("max_budget must be in [1, candidate_count]")
    pieces: dict[str, list[np.ndarray]] = {
        key: []
        for key in ("query", "selected", "selected_mask", "candidate", "cost", "target", "group_id")
    }
    if delta_fn is not None:
        pieces["prediction_delta"] = []
    if response_fn is not None:
        pieces["candidate_response"] = []
    rows = np.arange(batch.episodes)
    for repetition in range(states_per_episode):
        selected_mask = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
        if not static_only:
            sizes = rng.integers(0, max_budget, size=batch.episodes)
            for row, size in enumerate(sizes):
                if size:
                    chosen = rng.choice(batch.candidate_count, size=int(size), replace=False)
                    selected_mask[row, chosen] = True
        candidate_index = np.empty(batch.episodes, dtype=np.int64)
        for row in rows:
            candidate_index[row] = int(rng.choice(np.flatnonzero(~selected_mask[row])))
        marginal = marginal_fn(batch, selected_mask)
        selected, selected_valid = pack_fn(batch, selected_mask, max_budget=max_budget)
        pieces["query"].append(batch.query_embedding)
        pieces["selected"].append(selected)
        pieces["selected_mask"].append(selected_valid)
        pieces["candidate"].append(batch.candidate_embedding[rows, candidate_index])
        pieces["cost"].append(np.ones((batch.episodes, 1), dtype=np.float32))
        pieces["target"].append(marginal[rows, candidate_index].astype(np.float32))
        if delta_fn is not None:
            deltas = delta_fn(batch, selected_mask)
            pieces["prediction_delta"].append(
                deltas[rows, candidate_index, None].astype(np.float32)
            )
        if response_fn is not None:
            responses = response_fn(batch, selected_mask)
            pieces["candidate_response"].append(
                responses[rows, candidate_index].astype(np.float32)
            )
        pieces["group_id"].append(
            (repetition * batch.episodes + rows).astype(np.int64)
        )
    return PairDataset(**{key: np.concatenate(value) for key, value in pieces.items()})


def train_utility_model(
    pairs: PairDataset,
    *,
    max_budget: int,
    seed: int,
    device: str,
    epochs: int = 30,
    batch_size: int = 512,
) -> MarginalUtilityRouter:
    """Fit a Huber utility model with optional delta or response features."""

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    use_delta = pairs.prediction_delta is not None
    use_response = pairs.candidate_response is not None
    response_dim = int(pairs.candidate_response.shape[1]) if use_response else 0
    config = RouterConfig(
        embedding_dim=pairs.query.shape[1],
        hidden_dim=64,
        set_dim=64,
        depth=2,
        max_budget=max_budget,
        use_prediction_delta=use_delta,
        response_dim=response_dim,
    )
    model = MarginalUtilityRouter(config).to(device)
    tensors = [
        torch.from_numpy(pairs.query).float(),
        torch.from_numpy(pairs.selected).float(),
        torch.from_numpy(pairs.selected_mask),
        torch.from_numpy(pairs.candidate).float(),
    ]
    if use_delta:
        tensors.append(torch.from_numpy(pairs.prediction_delta).float())
    if use_response:
        tensors.append(torch.from_numpy(pairs.candidate_response).float())
    tensors.extend(
        [
            torch.from_numpy(pairs.cost).float(),
            torch.from_numpy(pairs.target).float(),
        ]
    )
    dataset = TensorDataset(*tensors)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            cursor = 0
            query, selected, selected_mask, candidate = batch[cursor : cursor + 4]
            cursor += 4
            prediction_delta = None
            if use_delta:
                prediction_delta = batch[cursor]
                cursor += 1
            candidate_response = None
            if use_response:
                candidate_response = batch[cursor]
                cursor += 1
            cost, target = batch[cursor : cursor + 2]
            prediction = model(
                query.to(device),
                selected.to(device),
                selected_mask.to(device),
                candidate.to(device),
                prediction_delta.to(device) if prediction_delta is not None else None,
                cost.to(device),
                candidate_response.to(device) if candidate_response is not None else None,
            )
            loss = utility_training_loss(prediction, target.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def predict_pair_dataset(
    model: MarginalUtilityRouter,
    pairs: PairDataset,
    *,
    device: str,
    batch_size: int = 8192,
) -> np.ndarray:
    """Predict utility for a pair dataset with optional delta or response features."""

    outputs: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(pairs.target), batch_size):
        stop = min(start + batch_size, len(pairs.target))
        outputs.append(
            model(
                torch.from_numpy(pairs.query[start:stop]).float().to(device),
                torch.from_numpy(pairs.selected[start:stop]).float().to(device),
                torch.from_numpy(pairs.selected_mask[start:stop]).to(device),
                torch.from_numpy(pairs.candidate[start:stop]).float().to(device),
                torch.from_numpy(pairs.prediction_delta[start:stop]).float().to(device)
                if pairs.prediction_delta is not None
                else None,
                torch.from_numpy(pairs.cost[start:stop]).float().to(device),
                torch.from_numpy(pairs.candidate_response[start:stop]).float().to(device)
                if pairs.candidate_response is not None
                else None,
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(outputs)


@torch.no_grad()
def score_state(
    model: MarginalUtilityRouter,
    batch: SyntheticBatch,
    selected_mask: np.ndarray,
    *,
    device: str,
    inference_batch_size: int = 8192,
    pack_fn=None,
    delta_fn=None,
    response_fn=None,
) -> np.ndarray:
    """Score every candidate for each episode at the supplied state."""

    if pack_fn is None:
        pack_fn = selected_set_tensors
    selected, selected_valid = pack_fn(
        batch, selected_mask, max_budget=model.config.max_budget
    )
    episodes, candidates = selected_mask.shape
    rows = np.repeat(np.arange(episodes), candidates)
    columns = np.tile(np.arange(candidates), episodes)
    outputs = []
    deltas = delta_fn(batch, selected_mask) if delta_fn is not None else None
    responses = response_fn(batch, selected_mask) if response_fn is not None else None
    for start in range(0, rows.size, inference_batch_size):
        take = slice(start, start + inference_batch_size)
        row = rows[take]
        column = columns[take]
        outputs.append(
            model(
                torch.from_numpy(batch.query_embedding[row]).float().to(device),
                torch.from_numpy(selected[row]).float().to(device),
                torch.from_numpy(selected_valid[row]).to(device),
                torch.from_numpy(batch.candidate_embedding[row, column]).float().to(device),
                torch.from_numpy(deltas[row, column, None]).float().to(device)
                if deltas is not None
                else None,
                torch.ones((len(row), 1), device=device),
                torch.from_numpy(responses[row, column]).float().to(device)
                if responses is not None
                else None,
            )
            .cpu()
            .numpy()
        )
    score = np.concatenate(outputs).reshape(episodes, candidates)
    return np.where(selected_mask, -np.inf, score)


def select_static(
    model: MarginalUtilityRouter,
    batch: SyntheticBatch,
    *,
    budget: int,
    device: str,
    pack_fn=None,
    delta_fn=None,
    response_fn=None,
) -> np.ndarray:
    empty = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    score = score_state(
        model,
        batch,
        empty,
        device=device,
        pack_fn=pack_fn,
        delta_fn=delta_fn,
        response_fn=response_fn,
    )
    order = np.argsort(-score, axis=1, kind="stable")[:, :budget]
    selected = empty.copy()
    selected[np.arange(batch.episodes)[:, None], order] = True
    return selected


def select_mur(
    model: MarginalUtilityRouter,
    batch: SyntheticBatch,
    *,
    budget: int,
    device: str,
    threshold: float = 0.0,
    pack_fn=None,
    delta_fn=None,
    response_fn=None,
) -> np.ndarray:
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for _ in range(budget):
        score = score_state(
            model,
            batch,
            selected,
            device=device,
            pack_fn=pack_fn,
            delta_fn=delta_fn,
            response_fn=response_fn,
        )
        choice = np.argmax(score, axis=1)
        best = score[np.arange(batch.episodes), choice]
        active = best > threshold
        selected[np.arange(batch.episodes)[active], choice[active]] = True
    return selected


def trace_mur(
    model: MarginalUtilityRouter,
    batch: SyntheticBatch,
    *,
    budget: int,
    device: str,
    threshold: float = 0.0,
    calibrator: SymmetricUtilityCalibrator | None = None,
    pack_fn=None,
    delta_fn=None,
    response_fn=None,
) -> np.ndarray:
    """Return the selected candidate index at each sequential routing step."""

    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    trace = np.full((batch.episodes, budget), -1, dtype=np.int64)
    for step in range(budget):
        point = score_state(
            model,
            batch,
            selected,
            device=device,
            pack_fn=pack_fn,
            delta_fn=delta_fn,
            response_fn=response_fn,
        )
        if calibrator is not None:
            score, _ = calibrator.interval(point)
            step_threshold = 0.0
        else:
            score = point
            step_threshold = threshold
        choice = np.argmax(score, axis=1)
        best = score[np.arange(batch.episodes), choice]
        active = best > step_threshold
        rows = np.arange(batch.episodes)[active]
        trace[rows, step] = choice[active]
        selected[rows, choice[active]] = True
    return trace


def trace_positive_utility_precision(batch: SyntheticBatch, trace: np.ndarray) -> float:
    """Measure the fraction of sequential additions with positive true marginal utility."""

    indices = np.asarray(trace, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[0] != batch.episodes:
        raise ValueError("trace must have shape (episodes, steps)")
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    positive = 0
    total = 0
    for step in range(indices.shape[1]):
        choice = indices[:, step]
        active = choice >= 0
        if not np.any(active):
            continue
        marginal = observed_candidate_marginals(batch, selected)
        rows = np.arange(batch.episodes)[active]
        cols = choice[active]
        positive += int(np.sum(marginal[rows, cols] > 0.0))
        total += int(rows.size)
        selected[rows, cols] = True
    return float(positive / total) if total else float("nan")


def select_mur_interval(
    model: MarginalUtilityRouter,
    batch: SyntheticBatch,
    calibrator: SymmetricUtilityCalibrator,
    *,
    budget: int,
    device: str,
    pack_fn=None,
    delta_fn=None,
    response_fn=None,
) -> np.ndarray:
    """Risk-averse routing: add only candidates with a positive utility LCB."""

    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for _ in range(budget):
        point = score_state(
            model,
            batch,
            selected,
            device=device,
            pack_fn=pack_fn,
            delta_fn=delta_fn,
            response_fn=response_fn,
        )
        lower, _ = calibrator.interval(point)
        choice = np.argmax(lower, axis=1)
        best = lower[np.arange(batch.episodes), choice]
        active = best > 0.0
        selected[np.arange(batch.episodes)[active], choice[active]] = True
    return selected


def select_relevance(batch: SyntheticBatch, *, budget: int) -> np.ndarray:
    order = np.argsort(-batch.relevance, axis=1, kind="stable")[:, :budget]
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    selected[np.arange(batch.episodes)[:, None], order] = True
    return selected


def select_coverage(batch: SyntheticBatch, *, budget: int) -> np.ndarray:
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for row in range(batch.episodes):
        covered: set[int] = set()
        for _ in range(budget):
            remaining = np.flatnonzero(~selected[row])
            adjusted = batch.relevance[row, remaining].astype(float)
            adjusted -= np.asarray(
                [2.0 if int(batch.candidate_group[row, index]) in covered else 0.0 for index in remaining]
            )
            choice = int(remaining[int(np.argmax(adjusted))])
            selected[row, choice] = True
            covered.add(int(batch.candidate_group[row, choice]))
    return selected


def select_mmr(batch: SyntheticBatch, *, budget: int, gamma: float = 0.5) -> np.ndarray:
    """Maximum-marginal-relevance selection using observable embeddings."""

    if gamma < 0.0:
        raise ValueError("gamma must be nonnegative")
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    vectors = batch.candidate_embedding.astype(float)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    similarity = np.einsum("bik,bjk->bij", vectors, vectors)
    for row in range(batch.episodes):
        for step in range(min(budget, batch.candidate_count)):
            remaining = np.flatnonzero(~selected[row])
            if step == 0:
                adjusted = batch.relevance[row, remaining].astype(float)
            else:
                redundancy = np.max(similarity[row, remaining][:, selected[row]], axis=1)
                adjusted = batch.relevance[row, remaining] - gamma * redundancy
            selected[row, int(remaining[int(np.argmax(adjusted))])] = True
    return selected


def select_oracle_static(batch: SyntheticBatch, *, budget: int) -> np.ndarray:
    """Perfect standalone-utility ranking: true m(j|empty), no re-scoring."""

    empty = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    marginal = observed_candidate_marginals(batch, empty)
    order = np.argsort(-marginal, axis=1, kind="stable")[:, :budget]
    selected = empty.copy()
    for row in range(batch.episodes):
        for candidate in order[row]:
            if marginal[row, candidate] <= 0.0:
                break
            selected[row, candidate] = True
    return selected


def select_oracle(batch: SyntheticBatch, *, budget: int) -> np.ndarray:
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for _ in range(budget):
        marginal = observed_candidate_marginals(batch, selected)
        choice = np.argmax(marginal, axis=1)
        best = marginal[np.arange(batch.episodes), choice]
        active = best > 0.0
        selected[np.arange(batch.episodes)[active], choice[active]] = True
    return selected


def summarize_policy(
    batch: SyntheticBatch,
    selected: np.ndarray,
    oracle_selected: np.ndarray,
) -> dict[str, float]:
    empty = np.zeros_like(selected)
    metrics = decision_metrics(
        squared_error(batch, empty),
        squared_error(batch, selected),
        squared_error(batch, oracle_selected),
        np.sum(selected, axis=1),
    )
    duplicates = []
    for row in range(batch.episodes):
        groups = batch.candidate_group[row, selected[row]]
        duplicates.append(len(groups) - len(np.unique(groups)))
    metrics["mean_duplicate_selections"] = float(np.mean(duplicates))
    counts = np.sum(selected, axis=1)
    metrics["duplicate_selection_rate"] = float(
        np.mean(np.divide(duplicates, counts, out=np.zeros_like(counts, dtype=float), where=counts > 0))
    )
    return metrics
