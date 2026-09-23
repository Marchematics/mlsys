"""Unit tests for the R112 redundancy-structured pool and the forced-budget ablation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve()
PAMI = HERE.parents[1]
sys.path.insert(0, str(PAMI / "experiments"))
sys.path.insert(0, str(PAMI / "experiments" / "demo_selection"))

from demo_protocol import BenchmarkConfig, DemoBatch, fit_pca, split_indices  # noqa: E402
from forced_budget import select_ranked_response_mur_forced  # noqa: E402
from frozen_helpers import select_ranked_response_mur  # noqa: E402
from incontext_predictor import DemoExpertOps, InContextClassifier  # noqa: E402
from redundant_pool import (  # noqa: E402
    CLUSTER_SIZE,
    N_CLUSTERS,
    build_redundant_batch,
)
from vision_data import Features  # noqa: E402

EMBED = 16


def clustered_features(
    *, classes: int = 4, per_class: int = 120, cluster_spread: float = 0.05, seed: int = 0
) -> Features:
    """Synthetic features with tight, well-populated within-class modes.

    Each class has three modes with ``per_class / 3`` members, so the pool half
    holds enough near-duplicates for every cluster to stay inside one mode.
    """

    rng = np.random.default_rng(seed)
    prototypes = rng.normal(size=(classes, 3, EMBED)).astype(np.float32) * 3.0
    train, labels = [], []
    for class_index in range(classes):
        for index in range(per_class):
            mode = index % 3
            vector = prototypes[class_index, mode] + rng.normal(size=EMBED).astype(np.float32) * cluster_spread
            train.append(vector)
            labels.append(class_index)
    train = np.asarray(train, dtype=np.float16)
    test = np.asarray(
        [prototypes[c, 0] + rng.normal(size=EMBED).astype(np.float32) * cluster_spread for c in range(classes)],
        dtype=np.float16,
    )
    return Features(
        benchmark="toy",
        train_emb=train,
        train_labels=np.asarray(labels, dtype=np.int64),
        test_emb=test,
        test_labels=np.arange(classes, dtype=np.int64),
        class_names=[str(index) for index in range(classes)],
        meta={},
    )


def toy_batch(episodes: int = 4, candidates: int = 16, queries: int = 3, classes: int = 3, seed: int = 0) -> DemoBatch:
    rng = np.random.default_rng(seed)
    query_labels = rng.integers(0, classes, size=episodes)
    return DemoBatch(
        query_embedding=rng.normal(size=(episodes, EMBED)).astype(np.float32),
        candidate_embedding=rng.normal(size=(episodes, candidates, EMBED)).astype(np.float32),
        query_features=rng.normal(size=(episodes, queries, EMBED)).astype(np.float32),
        candidate_features=rng.normal(size=(episodes, candidates, EMBED)).astype(np.float32),
        query_labels=query_labels.astype(np.int64),
        candidate_labels=rng.integers(0, classes, size=(episodes, candidates)).astype(np.int64),
        class_index=query_labels.astype(np.int64),
        episode_index=np.zeros(episodes, dtype=np.int64),
    )


class _RouterConfig:
    max_budget = 4


class ConstantRouter(torch.nn.Module):
    """Stub utility model returning a fixed score for every candidate."""

    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.value = float(value)
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.config = _RouterConfig()

    def forward(self, *args, **kwargs):  # noqa: D401 - test stub
        query = args[0]
        return torch.full((query.shape[0], 1), self.value, dtype=torch.float32, device=query.device)


def toy_ops(batch: DemoBatch, *, classes: int = 3) -> DemoExpertOps:
    torch.manual_seed(0)
    model = InContextClassifier(EMBED, classes, d_model=16, nhead=2, num_layers=1, feed_forward_dim=32, dropout=0.0)
    return DemoExpertOps(model, device="cpu", chunk_episodes=2, chunk_pairs=8)


def test_redundant_pool_structure():
    features = clustered_features()
    mean, components = fit_pca(features, dim=8)
    splits = split_indices(features, seed=1)
    config = BenchmarkConfig("toy", 1, 1, query_batch_size=4, candidate_count=16)
    batch, info = build_redundant_batch(
        features,
        mean,
        components,
        query_source=splits.test,
        pool_source=splits.pool,
        distractor_pool=splits.distractor_pool,
        query_split="test",
        episodes_per_class=1,
        seed=0,
        config=config,
    )
    assert batch.candidate_count == 16
    assert batch.episodes == 4
    for row in range(batch.episodes):
        cluster_id = info.cluster_id[row]
        target = int(batch.query_labels[row])
        assert np.sum(cluster_id >= 0) == N_CLUSTERS * CLUSTER_SIZE
        assert np.sum(cluster_id < 0) == 16 - N_CLUSTERS * CLUSTER_SIZE, "expected distractors"
        for cluster in range(N_CLUSTERS):
            members = np.flatnonzero(cluster_id == cluster)
            assert members.size == CLUSTER_SIZE
            assert np.all(batch.candidate_labels[row, members] == target)
            # mutual near-duplicates: single-linkage distance stays small
            vectors = batch.candidate_embedding[row, members]
            gram = vectors @ vectors.T
            off_diagonal = gram[~np.eye(CLUSTER_SIZE, dtype=bool)]
            assert off_diagonal.min() > 0.9, "cluster members must be mutually close"
        # distractors are other-class images
        assert np.all(batch.candidate_labels[row, cluster_id < 0] != target)
        # within-cluster similarity is strictly above between-cluster similarity
        assert np.all(info.within_similarity[row] > info.between_similarity[row])
    assert np.all(info.between_similarity > 0.0)
    assert np.all(info.seed_member_similarity > info.between_similarity)


def test_redundant_pool_excludes_query_images_for_train_episodes():
    features = clustered_features()
    mean, components = fit_pca(features, dim=8)
    splits = split_indices(features, seed=2)
    config = BenchmarkConfig("toy", 2, 1, query_batch_size=4, candidate_count=16)
    batch, _ = build_redundant_batch(
        features,
        mean,
        components,
        query_source=splits.train_classes,
        pool_source=splits.pool,
        distractor_pool=splits.distractor_pool,
        query_split="train",
        episodes_per_class=2,
        seed=3,
        config=config,
    )
    for row in range(batch.episodes):
        overlap = np.all(
            np.isclose(batch.candidate_features[row][:, None, :], batch.query_features[row][None], atol=1e-6),
            axis=-1,
        )
        assert not np.any(overlap), "a query image must not be its own demonstration"


def test_forced_budget_selects_exactly_budget():
    batch = toy_batch()
    ops = toy_ops(batch)
    screen = ConstantRouter(1.0)
    response = ConstantRouter(0.5)
    selected, rows = select_ranked_response_mur_forced(
        screen, response, batch, ops, budget=4, q=4, device="cpu"
    )
    assert selected.shape == (batch.episodes, batch.candidate_count)
    counts = selected.sum(axis=1)
    assert np.all(counts == 4), f"forced budget must select exactly 4, got {counts}"
    assert rows >= 0


def test_forced_matches_frozen_when_threshold_never_binds():
    """With non-negative scores the frozen rule always accepts, so the two agree."""

    batch = toy_batch()
    ops = toy_ops(batch)
    screen = ConstantRouter(1.0)
    response = ConstantRouter(1.0)
    frozen, _ = select_ranked_response_mur(screen, response, batch, ops, budget=4, q=4, device="cpu")
    forced, _ = select_ranked_response_mur_forced(screen, response, batch, ops, budget=4, q=4, device="cpu")
    assert np.array_equal(frozen, forced), "forced and thresholded selection must coincide when scores stay positive"
    assert np.all(frozen.sum(axis=1) == 4)


def test_forced_rule_selects_more_than_thresholded_when_scores_negative():
    """The ablation is what separates ranking from stopping: negative scores stop the frozen rule."""

    batch = toy_batch()
    ops = toy_ops(batch)
    screen = ConstantRouter(1.0)
    response = ConstantRouter(-1.0)
    frozen, _ = select_ranked_response_mur(screen, response, batch, ops, budget=4, q=4, device="cpu")
    forced, _ = select_ranked_response_mur_forced(screen, response, batch, ops, budget=4, q=4, device="cpu")
    assert np.all(frozen.sum(axis=1) == 0), "frozen rule must reject all negative-score candidates"
    assert np.all(forced.sum(axis=1) == 4), "forced rule must still fill the budget"


def test_redundancy_diagnostics_reports_diminishing_returns():
    """Same-cluster second members must not be more useful than cluster seeds."""

    from run_r112_redundant_demo import redundancy_diagnostics

    batch = toy_batch(episodes=3, candidates=16, queries=3, classes=3, seed=5)
    ops = toy_ops(batch)
    cluster_id = np.full((3, 16), -1, dtype=np.int64)
    for cluster in range(N_CLUSTERS):
        cluster_id[:, cluster * CLUSTER_SIZE : (cluster + 1) * CLUSTER_SIZE] = cluster
    from redundant_pool import RedundantPoolInfo

    info = RedundantPoolInfo(
        cluster_id=cluster_id,
        seed_index=np.tile(np.arange(0, N_CLUSTERS * CLUSTER_SIZE, CLUSTER_SIZE), (3, 1)),
        within_similarity=np.ones((3, N_CLUSTERS)),
        between_similarity=np.zeros(3),
        seed_member_similarity=np.ones((3, N_CLUSTERS)),
        r110_reference_similarity=np.zeros(3),
    )
    diagnostics = redundancy_diagnostics(ops, batch, info)
    assert set(diagnostics) >= {
        "mean_first_member_marginal",
        "mean_second_member_marginal",
        "mean_distractor_marginal",
        "between_cluster_similarity",
    }
    assert len(diagnostics["per_episode_first"]) == 3
    assert np.isfinite(diagnostics["mean_first_member_marginal"])
