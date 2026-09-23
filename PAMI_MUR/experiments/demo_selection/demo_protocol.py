"""Demo-selection protocol: episodes, batches and observable state features.

Second task family for the frozen R-MUR evidence.  The mapping from the traffic
protocol (``PAMI_MUR/experiments/pami_traffic.py``) is:

===========================  ==================================================
traffic                      demonstration selection
===========================  ==================================================
target sensor                target class ``c``
anchor window of sensor c    query batch of ``N`` test images of class ``c``
candidate sensor pool (16)   candidate demonstration pool (16 training images)
correlated neighbours (8)    8 in-class training images nearest to the query
                             batch mean in frozen-encoder embedding space
distractors                  other-class training images
anchor history (12 steps)    query-batch mean embedding (compact 64-dim PCA view)
candidate history            candidate image embedding (compact 64-dim PCA view)
target future                per-image class labels of the query batch
===========================  ==================================================

Two views of every image exist:

* ``*_features``: the raw L2-normalised frozen-CLIP ``ViT-L-14`` embedding
  (768-dim) consumed by the in-context predictor.
* ``*_embedding``: a 64-dim PCA view of the same embedding, fitted without
  labels on the benchmark's train split.  This is the *observable* state the
  R-MUR router and the similarity baselines see, playing the role of the
  normalised 12-step histories in the traffic protocol.

Train-split images are partitioned once (deterministically, per class) into a
POOL half that supplies candidate demonstrations everywhere and a QUERY half.
Test-split images supply evaluation queries.  Training query batches are drawn
from the whole train split *minus the episode's own candidate images*, so a
query image is never its own demonstration while every train image stays
usable; this mirrors the traffic protocol, where the target series and the
candidate series are always distinct.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vision_data import Features, load_features

CANDIDATE_COUNT = 16
BUDGET = 4
QUERY_BATCH_SIZE = 32
IN_CLASS_CANDIDATES = CANDIDATE_COUNT // 2
PROTOCOL_SEED = 20260921
PCA_DIM = 64


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    train_episodes_per_class: int
    test_episodes_per_class: int
    query_batch_size: int = QUERY_BATCH_SIZE
    candidate_count: int = CANDIDATE_COUNT
    budget: int = BUDGET
    in_class_candidates: int = IN_CLASS_CANDIDATES
    predictor_epochs: int = 200


BENCHMARK_CONFIGS: dict[str, BenchmarkConfig] = {
    "cifar10": BenchmarkConfig("cifar10", train_episodes_per_class=24, test_episodes_per_class=8),
    "cifar100": BenchmarkConfig("cifar100", train_episodes_per_class=6, test_episodes_per_class=2, predictor_epochs=120),
    "svhn": BenchmarkConfig("svhn", train_episodes_per_class=24, test_episodes_per_class=8),
    "eurosat": BenchmarkConfig("eurosat", train_episodes_per_class=24, test_episodes_per_class=8),
    "dtd": BenchmarkConfig("dtd", train_episodes_per_class=4, test_episodes_per_class=2),
}


# --------------------------------------------------------------------------
# Observable state
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoBatch:
    """Episode batch consumed by both the frozen predictor and the R-MUR router."""

    query_embedding: np.ndarray  # (E, d) compact query-batch summary (router view)
    candidate_embedding: np.ndarray  # (E, K, d) compact candidate view
    query_features: np.ndarray  # (E, N, 768) frozen-encoder features of the query images
    candidate_features: np.ndarray  # (E, K, 768) frozen-encoder features of the candidates
    query_labels: np.ndarray  # (E,) class label shared by every query image of the episode
    candidate_labels: np.ndarray  # (E, K)
    class_index: np.ndarray  # (E,) target class of the episode
    episode_index: np.ndarray  # (E,) episode id inside its class
    time_index: None = None

    @property
    def episodes(self) -> int:
        return int(self.query_features.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_features.shape[1])

    @property
    def query_batch_size(self) -> int:
        return int(self.query_features.shape[1])

    @property
    def candidate_history(self) -> np.ndarray:
        """Alias used by the generic MMR / DPP / facility-location baselines."""

        return self.candidate_embedding

    def select(self, rows: np.ndarray) -> "DemoBatch":
        rows = np.asarray(rows, dtype=np.int64)
        return DemoBatch(
            query_embedding=self.query_embedding[rows],
            candidate_embedding=self.candidate_embedding[rows],
            query_features=self.query_features[rows],
            candidate_features=self.candidate_features[rows],
            query_labels=self.query_labels[rows],
            candidate_labels=self.candidate_labels[rows],
            class_index=self.class_index[rows],
            episode_index=self.episode_index[rows],
        )


def fit_pca(features: Features, *, dim: int = PCA_DIM, cache_path: Path | None = None):
    """Unsupervised PCA of the frozen encoder output, fitted on the train split."""

    if cache_path is not None and Path(cache_path).exists():
        payload = np.load(cache_path)
        mean, components = payload["mean"], payload["components"]
        return mean.astype(np.float32), components.astype(np.float32)
    matrix = np.asarray(features.train_emb, dtype=np.float32)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    covariance = (centered.T @ centered) / max(matrix.shape[0] - 1, 1)
    values, vectors = np.linalg.eigh(covariance.astype(np.float64))
    order = np.argsort(-values)[:dim]
    components = vectors[:, order].T.astype(np.float32)
    # Deterministic sign convention (largest absolute loading positive).
    pivot = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), pivot])
    signs[signs == 0.0] = 1.0
    components *= signs[:, None]
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            mean=mean.astype(np.float32),
            components=components,
            explained_variance_ratio=(
                values[np.argsort(-values)][:dim] / max(values.sum(), 1e-12)
            ).astype(np.float64),
        )
    return mean, components


def project(matrix: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32) @ components.T - (mean @ components.T)
    return values.astype(np.float32)


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return (values / np.maximum(norm, 1e-12)).astype(np.float32)


# --------------------------------------------------------------------------
# Episode construction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitIndices:
    """Deterministic per-class index bookkeeping for one benchmark."""

    pool: list[np.ndarray]
    train_classes: list[np.ndarray]
    query_half: list[np.ndarray]
    test: list[np.ndarray]
    distractor_pool: np.ndarray


def split_indices(features: Features, *, seed: int = PROTOCOL_SEED) -> SplitIndices:
    train_labels = np.asarray(features.train_labels)
    test_labels = np.asarray(features.test_labels)
    pool: list[np.ndarray] = []
    query_half: list[np.ndarray] = []
    test: list[np.ndarray] = []
    train_by_class: list[np.ndarray] = []
    for class_index in range(features.num_classes):
        members = np.flatnonzero(train_labels == class_index)
        shuffled = members[np.random.default_rng(seed + class_index).permutation(members.size)]
        half = shuffled.size // 2
        pool.append(np.sort(shuffled[:half]))
        query_half.append(np.sort(shuffled[half:]))
        train_by_class.append(members)
        test.append(np.flatnonzero(test_labels == class_index))
    return SplitIndices(
        pool=pool,
        train_classes=[np.sort(members) for members in train_by_class],
        query_half=query_half,
        test=test,
        distractor_pool=np.sort(np.concatenate(pool)),
    )


def _draw_query_indices(
    available: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    """``count`` query images drawn without replacement, cycling if too few exist."""

    if available.size == 0:
        raise ValueError("no query images available for this class")
    picks: list[np.ndarray] = []
    remaining = available[rng.permutation(available.size)]
    while sum(piece.size for piece in picks) < count:
        picks.append(remaining)
        remaining = available[rng.permutation(available.size)]
    return np.concatenate(picks)[:count]


def build_demo_batch(
    features: Features,
    mean: np.ndarray,
    components: np.ndarray,
    *,
    query_source: list[np.ndarray],
    pool_source: list[np.ndarray],
    distractor_pool: np.ndarray,
    query_split: str,
    episodes_per_class: int,
    seed: int,
    config: BenchmarkConfig,
) -> DemoBatch:
    """Build one episode batch (train, calibration or test).

    ``pool_source`` supplies the in-class demonstration candidates for each
    target class; ``distractor_pool`` is the union of the other classes'
    candidate halves and supplies the distractors.
    """

    rng = np.random.default_rng(seed)
    train_labels = np.asarray(features.train_labels)
    if query_split not in {"train", "test"}:
        raise ValueError("query_split must be 'train' or 'test'")
    raw_train = np.asarray(features.train_emb, dtype=np.float32)
    raw_query_source = (
        np.asarray(features.test_emb, dtype=np.float32)
        if query_split == "test"
        else raw_train
    )
    distractor_pool = np.asarray(distractor_pool, dtype=np.int64)
    query_rows, candidate_rows = [], []
    class_rows, episode_rows = [], []
    candidate_label_rows = []
    for class_index in range(features.num_classes):
        available_query = np.asarray(query_source[class_index], dtype=np.int64)
        available_pool = np.asarray(pool_source[class_index], dtype=np.int64)
        in_class_pool = available_pool[train_labels[available_pool] == class_index]
        if in_class_pool.size < config.in_class_candidates:
            raise ValueError(
                f"class {class_index} has only {in_class_pool.size} pool images, "
                f"need {config.in_class_candidates}"
            )
        other_pool = distractor_pool[train_labels[distractor_pool] != class_index]
        distractor_count = config.candidate_count - config.in_class_candidates
        if other_pool.size < distractor_count:
            raise ValueError(
                f"class {class_index} has only {other_pool.size} distractor images, "
                f"need {distractor_count}"
            )
        pool_compact = l2_normalise(project(raw_train[in_class_pool], mean, components))
        for episode in range(episodes_per_class):
            if query_split == "train":
                # Draw the query batch from the whole train split minus *this
                # episode's* candidate images, so a query image is never its own
                # demonstration.  This mirrors the traffic protocol, where the
                # target series and the candidate series are distinct.
                probe = _draw_query_indices(available_query, config.query_batch_size, rng)
                probe_compact = l2_normalise(
                    project(raw_query_source[probe].mean(axis=0, keepdims=True), mean, components)
                )[0]
                nearest = in_class_pool[
                    np.argsort(-(pool_compact @ probe_compact), kind="stable")[
                        : config.in_class_candidates
                    ]
                ]
                distractors = rng.choice(other_pool, size=distractor_count, replace=False)
                candidates = np.concatenate([nearest, distractors]).astype(np.int64)
                draw_source = np.setdiff1d(available_query, candidates, assume_unique=False)
                query = _draw_query_indices(draw_source, config.query_batch_size, rng)
            else:
                query = _draw_query_indices(available_query, config.query_batch_size, rng)
                query_compact = l2_normalise(
                    project(raw_query_source[query].mean(axis=0, keepdims=True), mean, components)
                )[0]
                nearest = in_class_pool[
                    np.argsort(-(pool_compact @ query_compact), kind="stable")[
                        : config.in_class_candidates
                    ]
                ]
                distractors = rng.choice(other_pool, size=distractor_count, replace=False)
                candidates = np.concatenate([nearest, distractors]).astype(np.int64)
            query_rows.append(query)
            candidate_rows.append(candidates)
            class_rows.append(class_index)
            episode_rows.append(episode)
            candidate_label_rows.append(train_labels[candidates])
    query_index = np.stack(query_rows, axis=0)
    candidate_index = np.stack(candidate_rows, axis=0)
    query_features = raw_query_source[query_index]
    candidate_features = raw_train[candidate_index]
    return DemoBatch(
        query_embedding=l2_normalise(
            project(query_features.mean(axis=1), mean, components)
        ),
        candidate_embedding=l2_normalise(project(candidate_features, mean, components)),
        query_features=query_features,
        candidate_features=candidate_features,
        query_labels=np.asarray(class_rows, dtype=np.int64),
        candidate_labels=np.asarray(candidate_label_rows, dtype=np.int64),
        class_index=np.asarray(class_rows, dtype=np.int64),
        episode_index=np.asarray(episode_rows, dtype=np.int64),
    )


@dataclass(frozen=True)
class ProtocolData:
    features: Features
    mean: np.ndarray
    components: np.ndarray
    splits: SplitIndices
    config: BenchmarkConfig

    def train_batch(self, seed: int) -> DemoBatch:
        return build_demo_batch(
            self.features,
            self.mean,
            self.components,
            query_source=self.splits.train_classes,
            pool_source=self.splits.pool,
            distractor_pool=self.splits.distractor_pool,
            query_split="train",
            episodes_per_class=self.config.train_episodes_per_class,
            seed=seed,
            config=self.config,
        )

    def test_batch(self, seed: int) -> DemoBatch:
        return build_demo_batch(
            self.features,
            self.mean,
            self.components,
            query_source=self.splits.test,
            pool_source=self.splits.pool,
            distractor_pool=self.splits.distractor_pool,
            query_split="test",
            episodes_per_class=self.config.test_episodes_per_class,
            seed=seed,
            config=self.config,
        )


def load_protocol(
    feature_root: Path,
    name: str,
    *,
    pca_dim: int = PCA_DIM,
    protocol_seed: int = PROTOCOL_SEED,
    config: BenchmarkConfig | None = None,
) -> ProtocolData:
    features = load_features(Path(feature_root), name)
    cache_path = Path(feature_root) / name / f"pca{pca_dim}.npz"
    mean, components = fit_pca(features, dim=pca_dim, cache_path=cache_path)
    splits = split_indices(features, seed=protocol_seed)
    return ProtocolData(
        features=features,
        mean=mean,
        components=components,
        splits=splits,
        config=config or BENCHMARK_CONFIGS[name],
    )
