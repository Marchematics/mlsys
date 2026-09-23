"""Frozen subset-capable expert for the METR-LA oracle gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrafficBatch:
    anchor_history: np.ndarray
    candidate_history: np.ndarray
    target_future: np.ndarray

    @property
    def episodes(self) -> int:
        return int(self.target_future.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_history.shape[1])

    @property
    def history_length(self) -> int:
        return int(self.anchor_history.shape[1])

    @property
    def horizon(self) -> int:
        return int(self.target_future.shape[1])

    @property
    def query_embedding(self) -> np.ndarray:
        return self.anchor_history.astype(np.float32)

    @property
    def candidate_embedding(self) -> np.ndarray:
        return self.candidate_history.astype(np.float32)


@dataclass(frozen=True)
class TrafficExpert:
    coefficient: np.ndarray
    candidate_count: int
    history_length: int
    horizon: int


def make_batch(
    values: np.ndarray,
    times: np.ndarray,
    *,
    target_sensor: int,
    candidate_sensors: np.ndarray,
    history_length: int,
    horizon: int,
) -> TrafficBatch:
    """Create leakage-free windows whose target starts at each supplied time."""

    data = np.asarray(values, dtype=np.float32)
    starts = np.asarray(times, dtype=np.int64)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("times must be a nonempty one-dimensional array")
    candidates = np.asarray(candidate_sensors, dtype=np.int64)
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("candidate_sensors must be a nonempty one-dimensional array")
    if np.any(starts < history_length) or np.any(starts + horizon > data.shape[0]):
        raise ValueError("window exceeds available traffic observations")
    anchor = np.stack([data[t - history_length : t, target_sensor] for t in starts])
    candidate = np.stack(
        [data[t - history_length : t, candidates].T for t in starts]
    )
    target = np.stack([data[t : t + horizon, target_sensor] for t in starts])
    return TrafficBatch(anchor, candidate, target)


def fit_subset_expert(
    batch: TrafficBatch,
    *,
    seed: int,
    subset_probability: float = 0.5,
    repeats: int = 2,
    ridge_penalty: float = 1.0,
) -> TrafficExpert:
    """Fit one frozen ridge expert on random context subsets."""

    if not 0.0 < subset_probability < 1.0 or repeats < 1 or ridge_penalty <= 0.0:
        raise ValueError("invalid subset-expert hyperparameters")
    rng = np.random.default_rng(seed)
    rows = []
    targets = []
    for repeat in range(repeats):
        if repeat == 0:
            masks = np.ones((batch.episodes, batch.candidate_count), dtype=bool)
        elif repeat == 1:
            masks = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
        else:
            masks = rng.random((batch.episodes, batch.candidate_count)) < subset_probability
        features = np.concatenate(
            [
                batch.anchor_history,
                (batch.candidate_history * masks[:, :, None]).reshape(
                    batch.episodes, -1
                ),
            ],
            axis=1,
        )
        rows.append(np.concatenate([np.ones((batch.episodes, 1)), features], axis=1))
        targets.append(batch.target_future)
    design = np.concatenate(rows, axis=0).astype(np.float64)
    target = np.concatenate(targets, axis=0).astype(np.float64)
    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += ridge_penalty
    coefficient = np.linalg.solve(gram, design.T @ target).astype(np.float32)
    return TrafficExpert(
        coefficient=coefficient,
        candidate_count=batch.candidate_count,
        history_length=batch.history_length,
        horizon=batch.horizon,
    )


def expert_prediction(
    batch: TrafficBatch,
    selected: np.ndarray,
    expert,
) -> np.ndarray:
    """Predict with the frozen expert.

    Accepts either the ridge ``TrafficExpert`` or a neural expert that exposes
    ``neural_prediction``, so that routing experiments can be repeated with a
    nonlinear predictor family.
    """

    if not isinstance(expert, TrafficExpert):
        from mur.neural_expert import neural_prediction

        return neural_prediction(batch, selected, expert)
    mask = np.asarray(selected, dtype=bool)
    if mask.shape != (batch.episodes, batch.candidate_count):
        raise ValueError("selected mask has the wrong shape")
    features = np.concatenate(
        [
            batch.anchor_history,
            (batch.candidate_history * mask[:, :, None]).reshape(batch.episodes, -1),
        ],
        axis=1,
    )
    design = np.concatenate([np.ones((batch.episodes, 1)), features], axis=1)
    return (design @ expert.coefficient).astype(np.float32)


def squared_error(batch: TrafficBatch, selected: np.ndarray, expert: TrafficExpert) -> np.ndarray:
    prediction = expert_prediction(batch, selected, expert)
    return np.mean(np.square(prediction - batch.target_future), axis=1)



def expert_predict_many(batch: TrafficBatch, masks: np.ndarray, expert) -> np.ndarray:
    """Predict for a stack of selection masks, shape (J, episodes, horizon).

    The ridge expert is evaluated with one matrix product; a neural expert is
    evaluated with one forward pass over the stacked masks. Both paths are
    equivalent to calling :func:`expert_prediction` once per mask, which the
    regression test checks against stored runs.
    """

    stack = np.asarray(masks, dtype=bool)
    if stack.ndim != 3 or stack.shape[1:] != (batch.episodes, batch.candidate_count):
        raise ValueError("mask stack has the wrong shape")
    if isinstance(expert, TrafficExpert):
        anchor = np.asarray(batch.anchor_history, dtype=np.float64)[None, :, :]
        candidates = np.asarray(batch.candidate_history, dtype=np.float64)[None, :, :, :]
        features = np.concatenate(
            [
                np.repeat(anchor, stack.shape[0], axis=0),
                (candidates * stack[:, :, :, None]).reshape(stack.shape[0], batch.episodes, -1),
            ],
            axis=2,
        )
        design = np.concatenate(
            [np.ones((stack.shape[0], batch.episodes, 1)), features], axis=2
        )
        return (design @ expert.coefficient).astype(np.float32)
    from mur.neural_expert import neural_predict_many

    return neural_predict_many(batch, stack, expert)


def observed_candidate_marginals(
    batch: TrafficBatch,
    selected: np.ndarray,
    expert: TrafficExpert,
) -> np.ndarray:
    mask = np.asarray(selected, dtype=bool)
    episodes, candidates = mask.shape
    stack = np.repeat(mask[None, :, :], candidates, axis=0)
    for candidate in range(candidates):
        stack[candidate, :, candidate] = True
    predictions = expert_predict_many(batch, stack, expert)
    base_prediction = expert_predict_many(batch, mask[None, :, :], expert)[0]
    base_loss = np.mean(np.square(base_prediction - batch.target_future), axis=1)
    losses = np.mean(np.square(predictions - batch.target_future[None, :, :]), axis=2)
    output = (base_loss[None, :] - losses).T
    return np.where(mask, -np.inf, output)


def response_summary(
    batch: TrafficBatch,
    selected: np.ndarray,
    expert: TrafficExpert,
    *,
    candidate_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compact frozen-expert response summaries for each candidate.

    Feature order: mean base forecast, mean forecast after adding the
    candidate, mean response delta, and mean absolute response delta.
    """

    mask = np.asarray(selected, dtype=bool)
    if mask.shape != (batch.episodes, batch.candidate_count):
        raise ValueError("selected mask has the wrong shape")
    if candidate_mask is None:
        candidate_mask = np.ones_like(mask, dtype=bool)
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=bool)
        if candidate_mask.shape != mask.shape:
            raise ValueError("candidate_mask has the wrong shape")
    base = expert_predict_many(batch, mask[None, :, :], expert)[0]
    episodes, candidates = mask.shape
    stack = np.repeat(mask[None, :, :], candidates, axis=0)
    for candidate in range(candidates):
        stack[candidate, :, candidate] = True
    after = expert_predict_many(batch, stack, expert)              # (K, E, H)
    delta = after - base[None, :, :]
    horizon = max(int(batch.horizon), 1)
    features = np.stack(
        [
            np.broadcast_to(np.mean(base, axis=1)[None, :], (candidates, episodes)),
            np.mean(after, axis=2),
            np.mean(delta, axis=2),
            np.std(delta, axis=2),
            np.mean(np.abs(delta), axis=2),
            np.linalg.norm(delta, axis=2) / np.sqrt(horizon),
            np.max(np.abs(delta), axis=2),
        ],
        axis=2,
    ).astype(np.float32)
    eligible = (~mask) & candidate_mask
    return np.where(eligible[:, :, None], features.transpose(1, 0, 2), 0.0).astype(np.float32)


def pack_selected(
    batch: TrafficBatch, selected: np.ndarray, *, max_budget: int
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(selected, dtype=bool)
    if mask.shape != (batch.episodes, batch.candidate_count):
        raise ValueError("selected mask has the wrong shape")
    values = np.zeros(
        (batch.episodes, max_budget, batch.history_length), dtype=np.float32
    )
    valid = np.zeros((batch.episodes, max_budget), dtype=bool)
    for row in range(batch.episodes):
        indices = np.flatnonzero(mask[row])
        if indices.size > max_budget:
            raise ValueError("selected set exceeds max_budget")
        values[row, : indices.size] = batch.candidate_history[row, indices]
        valid[row, : indices.size] = True
    return values, valid
