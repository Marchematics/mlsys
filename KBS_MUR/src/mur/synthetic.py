"""Controlled multi-context environment for redundancy and harm gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SyntheticConfig:
    candidate_count: int = 8
    group_count: int = 8
    budget: int = 2
    redundancy_ratio: float = 0.5
    harmful_fraction: float = 0.0
    anchor_noise: float = 1.0
    useful_residual_fraction: float = 0.1
    harmful_residual_fraction: float = 1.8

    def __post_init__(self) -> None:
        if min(self.candidate_count, self.group_count, self.budget) < 1:
            raise ValueError("counts and budget must be positive")
        if self.budget > self.candidate_count:
            raise ValueError("budget cannot exceed candidate_count")
        if not 0.0 <= self.redundancy_ratio < 1.0:
            raise ValueError("redundancy_ratio must lie in [0, 1)")
        if not 0.0 <= self.harmful_fraction <= 1.0:
            raise ValueError("harmful_fraction must lie in [0, 1]")


@dataclass(frozen=True)
class SyntheticBatch:
    target: Array
    anchor: Array
    query_embedding: Array
    candidate_embedding: Array
    candidate_group: Array
    candidate_measurement: Array
    relevance: Array
    harmful: Array

    @property
    def episodes(self) -> int:
        return int(self.target.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_group.shape[1])


def generate_synthetic_batch(
    episodes: int,
    config: SyntheticConfig,
    *,
    seed: int,
) -> SyntheticBatch:
    """Generate relation-positive candidates with controlled duplicate groups.

    Useful candidates move one output coordinate close to its target. Duplicate
    candidates carry exactly the same measurement, so their standalone value is
    positive while their marginal value after the first copy is exactly zero.
    Harmful candidates remain relation-positive but move the coordinate farther
    along the anchor-error direction.
    """

    if episodes < 1:
        raise ValueError("episodes must be positive")
    rng = np.random.default_rng(seed)
    cfg = config
    target = rng.normal(size=(episodes, cfg.group_count))
    anchor_error = cfg.anchor_noise * rng.normal(size=target.shape)
    anchor = target + anchor_error
    unique_count = max(1, int(round(cfg.candidate_count * (1.0 - cfg.redundancy_ratio))))
    unique_count = min(unique_count, cfg.group_count, cfg.candidate_count)

    candidate_group = np.empty((episodes, cfg.candidate_count), dtype=np.int64)
    candidate_measurement = np.empty((episodes, cfg.candidate_count), dtype=float)
    harmful = np.zeros((episodes, cfg.candidate_count), dtype=bool)
    relevance = rng.uniform(0.7, 1.0, size=(episodes, cfg.candidate_count))
    for row in range(episodes):
        groups = rng.choice(cfg.group_count, size=unique_count, replace=False)
        assignment = list(int(group) for group in groups)
        while len(assignment) < cfg.candidate_count:
            assignment.append(int(rng.choice(groups)))
        assignment = np.asarray(assignment, dtype=np.int64)
        rng.shuffle(assignment)
        candidate_group[row] = assignment
        base_measurement: dict[int, float] = {}
        for group in groups:
            base_measurement[int(group)] = float(
                target[row, group] + cfg.useful_residual_fraction * anchor_error[row, group]
            )
        candidate_measurement[row] = np.asarray(
            [base_measurement[int(group)] for group in assignment], dtype=float
        )
        harmful_count = int(round(cfg.harmful_fraction * cfg.candidate_count))
        if harmful_count:
            harmful_index = rng.choice(cfg.candidate_count, size=harmful_count, replace=False)
            harmful[row, harmful_index] = True
            group_index = assignment[harmful_index]
            candidate_measurement[row, harmful_index] = (
                target[row, group_index]
                + cfg.harmful_residual_fraction * anchor_error[row, group_index]
            )

    dimension = 3 * cfg.group_count + 2
    query_embedding = np.zeros((episodes, dimension), dtype=np.float32)
    query_embedding[:, : cfg.group_count] = anchor.astype(np.float32)
    candidate_embedding = np.zeros(
        (episodes, cfg.candidate_count, dimension), dtype=np.float32
    )
    for row in range(episodes):
        for column in range(cfg.candidate_count):
            group = int(candidate_group[row, column])
            candidate_embedding[row, column, group] = candidate_measurement[row, column]
            candidate_embedding[row, column, cfg.group_count + group] = 1.0
            candidate_embedding[row, column, 2 * cfg.group_count + group] = anchor[row, group]
            candidate_embedding[row, column, -2] = relevance[row, column]
            # The final slot is reserved for observable metadata. Harm type is
            # deliberately not exposed to the router.
            candidate_embedding[row, column, -1] = 0.0
    return SyntheticBatch(
        target=target.astype(np.float32),
        anchor=anchor.astype(np.float32),
        query_embedding=query_embedding,
        candidate_embedding=candidate_embedding,
        candidate_group=candidate_group,
        candidate_measurement=candidate_measurement.astype(np.float32),
        relevance=relevance.astype(np.float32),
        harmful=harmful,
    )


def expert_prediction(batch: SyntheticBatch, selected: Array) -> Array:
    """Return the frozen coordinate-wise expert for a boolean selected mask."""

    mask = np.asarray(selected, dtype=bool)
    if mask.shape != batch.candidate_group.shape:
        raise ValueError("selected mask has the wrong shape")
    prediction = batch.anchor.astype(float).copy()
    for row in range(batch.episodes):
        for group in np.unique(batch.candidate_group[row, mask[row]]):
            columns = mask[row] & (batch.candidate_group[row] == group)
            prediction[row, group] = float(np.mean(batch.candidate_measurement[row, columns]))
    return prediction.astype(np.float32)


def squared_error(batch: SyntheticBatch, selected: Array) -> Array:
    prediction = expert_prediction(batch, selected)
    return np.mean(np.square(prediction - batch.target), axis=1)


def observed_candidate_marginals(batch: SyntheticBatch, selected: Array) -> Array:
    """Return realized marginal loss reductions for every unselected candidate."""

    mask = np.asarray(selected, dtype=bool)
    before = squared_error(batch, mask)
    output = np.full(mask.shape, -np.inf, dtype=float)
    for candidate in range(batch.candidate_count):
        eligible = ~mask[:, candidate]
        if not np.any(eligible):
            continue
        augmented = mask.copy()
        augmented[eligible, candidate] = True
        after = squared_error(batch, augmented)
        output[eligible, candidate] = before[eligible] - after[eligible]
    return output


def selected_set_tensors(
    batch: SyntheticBatch,
    selected: Array,
    *,
    max_budget: int,
) -> tuple[Array, Array]:
    """Pack selected candidate embeddings into a padded set tensor."""

    mask = np.asarray(selected, dtype=bool)
    if mask.shape != batch.candidate_group.shape:
        raise ValueError("selected mask has the wrong shape")
    dimension = batch.candidate_embedding.shape[-1]
    values = np.zeros((batch.episodes, max_budget, dimension), dtype=np.float32)
    valid = np.zeros((batch.episodes, max_budget), dtype=bool)
    for row in range(batch.episodes):
        indices = np.flatnonzero(mask[row])
        if indices.size > max_budget:
            raise ValueError("selected set exceeds max_budget")
        values[row, : indices.size] = batch.candidate_embedding[row, indices]
        valid[row, : indices.size] = True
    return values, valid
