"""Multi-context Electricity episodes for the KBS routing experiment.

The loader follows the existing UCI Electricity preprocessing contract, while
the episode object changes the decision target from a pairwise relation to a
candidate-set loss marginal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray
SLOTS_PER_WEEK = 7 * 96
SLOTS_2014 = 365 * 96


@dataclass(frozen=True)
class ElectricityData:
    values: Array
    train_clients: Array
    validation_clients: Array
    test_clients: Array
    slot_features: Array
    normalization_mean: float
    normalization_sd: float


@dataclass(frozen=True)
class ElectricityContextBatch:
    target: Array
    query_embedding: Array
    candidate_embedding: Array
    candidate_client: Array
    target_client: Array
    anchor_week: Array
    candidate_week: Array
    query_position: Array
    anchor_gram: Array
    anchor_rhs: Array
    candidate_gram: Array
    candidate_rhs: Array
    query_features: Array
    ridge_penalty: float
    relevance: Array

    @property
    def episodes(self) -> int:
        return int(self.target.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_client.shape[1])


def _weekly_fourier_features(harmonics: int) -> Array:
    slot = np.arange(SLOTS_PER_WEEK, dtype=float)
    angle = 2.0 * np.pi * slot / SLOTS_PER_WEEK
    columns = [np.ones(SLOTS_PER_WEEK)]
    for harmonic in range(1, harmonics + 1):
        columns.extend([np.sin(harmonic * angle), np.cos(harmonic * angle)])
    return np.column_stack(columns)


def load_electricity_data(
    cache_path: str,
    *,
    active_fraction_threshold: float,
    split_seed: int,
    train_client_count: int,
    validation_client_count: int,
    train_week_stop: int,
    harmonics: int,
) -> ElectricityData:
    """Load and normalize the existing 2014 Electricity cache."""

    with np.load(cache_path, allow_pickle=True) as loaded:
        full = np.asarray(loaded["values"], dtype=np.float32)
    if full.ndim != 2 or full.shape[0] < SLOTS_2014:
        raise ValueError("electricity cache has the wrong shape")
    values = full[-SLOTS_2014:]
    active_rows = slice(0, train_week_stop * SLOTS_PER_WEEK)
    active = np.mean(values[active_rows] > 0.0, axis=0) > active_fraction_threshold
    eligible = np.flatnonzero(active)
    required = train_client_count + validation_client_count + 112
    if len(eligible) < required:
        raise ValueError(f"only {len(eligible)} eligible clients; need at least {required}")
    permutation = np.random.default_rng(split_seed).permutation(eligible)
    train_clients = np.sort(permutation[:train_client_count])
    validation_clients = np.sort(
        permutation[train_client_count : train_client_count + validation_client_count]
    )
    test_clients = np.sort(permutation[train_client_count + validation_client_count :])
    transformed = np.log1p(values.astype(np.float64))
    train_rows = slice(0, train_week_stop * SLOTS_PER_WEEK)
    reference = transformed[train_rows][:, train_clients]
    mean = float(np.mean(reference))
    sd = float(np.std(reference))
    if sd <= 0.0:
        raise ValueError("training normalization must have positive scale")
    transformed = ((transformed - mean) / sd).astype(np.float32)
    return ElectricityData(
        values=transformed,
        train_clients=train_clients,
        validation_clients=validation_clients,
        test_clients=test_clients,
        slot_features=_weekly_fourier_features(harmonics).astype(np.float32),
        normalization_mean=mean,
        normalization_sd=sd,
    )


def _ridge_contexts(
    data: ElectricityData,
    clients: Array,
    weeks: Array,
    *,
    context_points: int,
    rng: np.random.Generator,
    ridge_penalty: float,
) -> tuple[Array, Array, Array, Array, Array]:
    """Sample contexts and return client, beta, Gram, rhs, residual arrays."""

    count = len(clients)
    dimension = data.slot_features.shape[1]
    positions = rng.integers(SLOTS_PER_WEEK, size=(count, context_points))
    times = weeks[:, None] * SLOTS_PER_WEEK + positions
    y = data.values[times, clients[:, None]].astype(np.float64)
    x = data.slot_features[positions].astype(np.float64)
    gram = np.einsum("bki,bkj->bij", x, x)
    rhs = np.einsum("bki,bk->bi", x, y)
    beta = np.linalg.solve(
        gram + ridge_penalty * np.eye(dimension)[None], rhs[..., None]
    )[..., 0]
    fitted = np.einsum("bki,bi->bk", x, beta)
    residual = np.mean(np.square(y - fitted), axis=1)
    return clients, beta, gram, rhs, residual


def sample_context_batch(
    data: ElectricityData,
    *,
    clients: Array,
    week_start: int,
    week_stop: int,
    episodes: int,
    candidate_count: int,
    same_client_fraction: float = 0.5,
    context_points: int,
    ridge_penalty: float,
    rng: np.random.Generator,
) -> ElectricityContextBatch:
    """Create an anchor, query, and heterogeneous candidate context pool."""

    clients = np.asarray(clients, dtype=int)
    if len(clients) < 2:
        raise ValueError("at least two clients are required")
    if not 0.0 <= same_client_fraction <= 1.0:
        raise ValueError("same_client_fraction must lie in [0, 1]")
    if week_stop - week_start < 3:
        raise ValueError("at least three weeks are required")
    if not 0 < context_points < SLOTS_PER_WEEK:
        raise ValueError("context_points must leave at least one query slot outside context")
    dimension = data.slot_features.shape[1]
    target_client = rng.choice(clients, size=episodes)
    anchor_week = rng.integers(week_start + 1, week_stop, size=episodes)
    query_position = rng.integers(SLOTS_PER_WEEK, size=episodes)
    anchor_positions = rng.integers(SLOTS_PER_WEEK, size=(episodes, context_points))
    overlap = np.any(anchor_positions == query_position[:, None], axis=1)
    while np.any(overlap):
        query_position[overlap] = rng.integers(SLOTS_PER_WEEK, size=int(np.sum(overlap)))
        overlap = np.any(anchor_positions == query_position[:, None], axis=1)
    anchor_times = anchor_week[:, None] * SLOTS_PER_WEEK + anchor_positions
    anchor_y = data.values[anchor_times, target_client[:, None]].astype(np.float64)
    anchor_x = data.slot_features[anchor_positions].astype(np.float64)
    anchor_gram = np.einsum("bki,bkj->bij", anchor_x, anchor_x)
    anchor_rhs = np.einsum("bki,bk->bi", anchor_x, anchor_y)
    anchor_beta = np.linalg.solve(
        anchor_gram + ridge_penalty * np.eye(dimension)[None], anchor_rhs[..., None]
    )[..., 0]
    anchor_fit = np.einsum("bki,bi->bk", anchor_x, anchor_beta)
    anchor_residual = np.mean(np.square(anchor_y - anchor_fit), axis=1)
    query_x = data.slot_features[query_position].astype(np.float64)
    target_time = anchor_week * SLOTS_PER_WEEK + query_position
    target = data.values[target_time, target_client].astype(np.float32)

    candidate_client = np.empty((episodes, candidate_count), dtype=int)
    candidate_week = np.empty((episodes, candidate_count), dtype=int)
    for row in range(episodes):
        same = rng.random(candidate_count) < same_client_fraction
        candidate_client[row, same] = target_client[row]
        different = np.flatnonzero(~same)
        if different.size:
            choices = rng.choice(clients[clients != target_client[row]], size=different.size)
            candidate_client[row, different] = choices
        candidate_week[row] = rng.integers(week_start, anchor_week[row], size=candidate_count)

    flat_client, flat_beta, flat_gram, flat_rhs, flat_residual = _ridge_contexts(
        data,
        candidate_client.reshape(-1),
        candidate_week.reshape(-1),
        context_points=context_points,
        rng=rng,
        ridge_penalty=ridge_penalty,
    )
    candidate_beta = flat_beta.reshape(episodes, candidate_count, dimension)
    candidate_gram = flat_gram.reshape(episodes, candidate_count, dimension, dimension)
    candidate_rhs = flat_rhs.reshape(episodes, candidate_count, dimension)
    candidate_residual = flat_residual.reshape(episodes, candidate_count)
    cosine = np.sum(candidate_beta * anchor_beta[:, None, :], axis=-1) / np.maximum(
        np.linalg.norm(candidate_beta, axis=-1) * np.linalg.norm(anchor_beta, axis=1)[:, None],
        1e-8,
    )
    same_client = (candidate_client == target_client[:, None]).astype(float)
    relevance = cosine + 0.10 * same_client

    # All embeddings use observable context/query features only.
    embedding_dim = 3 * dimension + 3
    query_embedding = np.zeros((episodes, embedding_dim), dtype=np.float32)
    query_embedding[:, :dimension] = anchor_beta.astype(np.float32)
    query_embedding[:, dimension : 2 * dimension] = 0.0
    query_embedding[:, 2 * dimension : 3 * dimension] = query_x.astype(np.float32)
    query_embedding[:, -3] = anchor_residual.astype(np.float32)
    candidate_embedding = np.zeros(
        (episodes, candidate_count, embedding_dim), dtype=np.float32
    )
    candidate_embedding[:, :, :dimension] = candidate_beta.astype(np.float32)
    candidate_embedding[:, :, dimension : 2 * dimension] = (
        candidate_beta - anchor_beta[:, None, :]
    ).astype(np.float32)
    candidate_embedding[:, :, 2 * dimension : 3 * dimension] = query_x[:, None, :].astype(
        np.float32
    )
    candidate_embedding[:, :, -3] = candidate_residual.astype(np.float32)
    candidate_embedding[:, :, -2] = cosine.astype(np.float32)
    candidate_embedding[:, :, -1] = same_client.astype(np.float32)
    return ElectricityContextBatch(
        target=target,
        query_embedding=query_embedding,
        candidate_embedding=candidate_embedding,
        candidate_client=candidate_client,
        target_client=target_client.astype(np.int64),
        anchor_week=anchor_week.astype(np.int64),
        candidate_week=candidate_week.astype(np.int64),
        query_position=query_position.astype(np.int64),
        anchor_gram=anchor_gram.astype(np.float32),
        anchor_rhs=anchor_rhs.astype(np.float32),
        candidate_gram=candidate_gram.astype(np.float32),
        candidate_rhs=candidate_rhs.astype(np.float32),
        query_features=query_x.astype(np.float32),
        ridge_penalty=float(ridge_penalty),
        relevance=relevance.astype(np.float32),
    )


def expert_prediction(batch: ElectricityContextBatch, selected: Array) -> Array:
    """Frozen ridge expert prediction using the anchor plus selected contexts."""

    mask = np.asarray(selected, dtype=bool)
    if mask.shape != batch.candidate_client.shape:
        raise ValueError("selected mask has the wrong shape")
    gram = batch.anchor_gram.astype(np.float64).copy()
    rhs = batch.anchor_rhs.astype(np.float64).copy()
    gram += np.einsum("bk,bkij->bij", mask.astype(float), batch.candidate_gram)
    rhs += np.einsum("bk,bki->bi", mask.astype(float), batch.candidate_rhs)
    dimension = gram.shape[-1]
    beta = np.linalg.solve(
        gram + batch.ridge_penalty * np.eye(dimension)[None], rhs[..., None]
    )[..., 0]
    return np.einsum("bi,bi->b", batch.query_features, beta).astype(np.float32)


def squared_error(batch: ElectricityContextBatch, selected: Array) -> Array:
    prediction = expert_prediction(batch, selected)
    return np.square(prediction - batch.target)


def observed_candidate_marginals(batch: ElectricityContextBatch, selected: Array) -> Array:
    mask = np.asarray(selected, dtype=bool)
    before = squared_error(batch, mask)
    output = np.full(mask.shape, -np.inf, dtype=float)
    for candidate in range(batch.candidate_count):
        eligible = ~mask[:, candidate]
        if not np.any(eligible):
            continue
        augmented = mask.copy()
        augmented[eligible, candidate] = True
        output[eligible, candidate] = before[eligible] - squared_error(batch, augmented)[eligible]
    return output


def pack_selected(batch: ElectricityContextBatch, selected: Array, max_budget: int) -> tuple[Array, Array]:
    mask = np.asarray(selected, dtype=bool)
    values = np.zeros((batch.episodes, max_budget, batch.candidate_embedding.shape[-1]), dtype=np.float32)
    valid = np.zeros((batch.episodes, max_budget), dtype=bool)
    for row in range(batch.episodes):
        indices = np.flatnonzero(mask[row])
        if indices.size > max_budget:
            raise ValueError("selected set exceeds max_budget")
        values[row, : indices.size] = batch.candidate_embedding[row, indices]
        valid[row, : indices.size] = True
    return values, valid
