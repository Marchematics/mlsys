"""Data plumbing for the PAMI strong-backbone confirmation.

This mirrors the frozen traffic protocol used by the R-MUR evidence
(candidate pool of correlated sensors plus distractors, 12-step history,
12-step horizon, correlation computed on the training split only) so that the
only changed component is the prediction expert.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NeuralTrafficBatch:
    """Episode batch consumed by both the expert and the R-MUR router."""

    anchor_history: np.ndarray
    candidate_history: np.ndarray
    target_future: np.ndarray
    target_sensor: int
    candidate_sensors: np.ndarray
    time_index: np.ndarray | None = None

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

    @property
    def candidate_node_matrix(self) -> np.ndarray:
        """Candidate node ids as (episodes, candidates).

        The protocol stores one shared pool per batch, while pair-level expert
        evaluation uses a per-episode candidate list; both are supported.
        """

        sensors = np.asarray(self.candidate_sensors)
        if sensors.ndim == 1:
            return np.broadcast_to(sensors[None, :], (self.episodes, self.candidate_count))
        if sensors.shape != (self.episodes, self.candidate_count):
            raise ValueError("candidate_sensors has the wrong shape")
        return sensors


@dataclass(frozen=True)
class TrafficData:
    values: np.ndarray
    train_times: np.ndarray
    validation_times: np.ndarray
    test_times: np.ndarray
    correlation: np.ndarray
    num_nodes: int
    history_length: int
    horizon: int


def load_traffic_data(
    dataset_dir: Path,
    *,
    history_length: int = 12,
    horizon: int = 12,
) -> TrafficData:
    """Load one STAEformer-format dataset; statistics use the training split."""

    data = np.load(Path(dataset_dir) / "data.npz")["data"].astype(np.float32)
    index = np.load(Path(dataset_dir) / "index.npz")
    values = data[..., 0]
    starts = index["train"][:, 1].astype(np.int64)
    train_stop = int(starts.max() + 1)
    mean = values[:train_stop].mean(axis=0)
    sd = values[:train_stop].std(axis=0)
    values = ((values - mean) / np.maximum(sd, 1e-6)).astype(np.float32)
    correlation = np.corrcoef(values[:train_stop].T).astype(np.float32)
    usable = (starts >= history_length) & (starts + horizon <= values.shape[0])
    return TrafficData(
        values=values,
        train_times=starts[usable],
        validation_times=index["val"][:, 1].astype(np.int64),
        test_times=index["test"][:, 1].astype(np.int64),
        correlation=correlation,
        num_nodes=int(values.shape[1]),
        history_length=history_length,
        horizon=horizon,
    )


def build_candidate_pool(
    correlation_row: np.ndarray,
    *,
    target_sensor: int,
    candidate_count: int,
    seed: int,
    mode: str = "correlated",
) -> np.ndarray:
    """Candidate pool builder.

    ``correlated`` (default, the frozen protocol): half high-correlation
    neighbours, a third mid-correlation, the rest random distractors.
    ``anticorrelated``: the same layout but the head of the pool is drawn from
    the *least* correlated sensors, i.e. contexts that are likely to hurt.
    ``mid``: the head is drawn from the middle of the magnitude distribution,
    i.e. contexts that are informative but not redundant with the anchor.
    """

    corr = np.asarray(correlation_row, dtype=float).copy()
    if mode == "mid":
        # candidates with moderate absolute correlation: they carry information
        # that is neither redundant with the anchor nor irrelevant
        magnitude = np.abs(corr)
        magnitude[target_sensor] = -np.inf
        order = np.argsort(-magnitude, kind="stable")
        band = order[len(order) // 3 : 2 * len(order) // 3]
        order = np.concatenate([band, np.setdiff1d(order, band, assume_unique=False)])
    else:
        corr[target_sensor] = np.inf if mode == "anticorrelated" else -np.inf
        order = np.argsort(corr if mode == "anticorrelated" else -corr, kind="stable")
    high = order[: max(candidate_count // 2, 1)]
    middle_start = len(high)
    middle_stop = min(middle_start + max(candidate_count // 3, 1), len(order))
    middle = order[middle_start:middle_stop]
    remaining = np.setdiff1d(order, np.concatenate([high, middle]), assume_unique=False)
    rng = np.random.default_rng(seed)
    distractors = rng.choice(
        remaining, size=candidate_count - len(high) - len(middle), replace=False
    )
    return np.concatenate([high, middle, distractors]).astype(np.int64)


def pools_for_targets(
    data: TrafficData,
    targets: np.ndarray,
    *,
    candidate_count: int,
    seed: int,
) -> dict[int, np.ndarray]:
    return {
        int(node): build_candidate_pool(
            data.correlation[int(node)],
            target_sensor=int(node),
            candidate_count=candidate_count,
            seed=seed,
        )
        for node in np.asarray(targets, dtype=np.int64)
    }


def target_set(num_nodes: int, count: int) -> np.ndarray:
    if count >= num_nodes:
        return np.arange(num_nodes, dtype=np.int64)
    return np.unique(np.linspace(0, num_nodes - 1, count, dtype=np.int64))


def make_neural_batch(
    data: TrafficData,
    times: np.ndarray,
    *,
    target_sensor: int,
    candidate_sensors: np.ndarray,
) -> NeuralTrafficBatch:
    """Windows whose target horizon starts at each supplied time."""

    values = data.values
    history, horizon = data.history_length, data.horizon
    starts = np.asarray(times, dtype=np.int64)
    candidates = np.asarray(candidate_sensors, dtype=np.int64)
    if starts.size == 0:
        raise ValueError("times must be nonempty")
    if np.any(starts < history) or np.any(starts + horizon > values.shape[0]):
        raise ValueError("window exceeds available traffic observations")
    steps = starts[:, None] - history + np.arange(history)[None, :]
    anchor = values[steps, np.full_like(steps, int(target_sensor))]
    candidate = values[steps[:, :, None], candidates[None, None, :]]
    candidate = np.transpose(candidate, (0, 2, 1))
    target = np.stack([values[t : t + horizon, target_sensor] for t in starts])
    return NeuralTrafficBatch(
        anchor_history=np.ascontiguousarray(anchor, dtype=np.float32),
        candidate_history=np.ascontiguousarray(candidate, dtype=np.float32),
        target_future=np.ascontiguousarray(target, dtype=np.float32),
        target_sensor=int(target_sensor),
        candidate_sensors=candidates,
        time_index=starts.astype(np.int64),
    )


def sample_windows(
    data: TrafficData,
    nodes: np.ndarray,
    *,
    windows_per_node: int,
    seed: int,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (node, time) pairs uniformly from a split."""

    rng = np.random.default_rng(seed)
    times = data.train_times if split == "train" else data.validation_times
    nodes = np.asarray(nodes, dtype=np.int64)
    node_draw = rng.choice(nodes, size=windows_per_node * len(nodes), replace=True)
    time_draw = rng.choice(times, size=node_draw.size, replace=True)
    return node_draw.astype(np.int64), time_draw.astype(np.int64)
