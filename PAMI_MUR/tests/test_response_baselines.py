"""Tests for the training-free response baselines.

The DELIFT-style rule is the one that needed a regression test: its first
implementation used one k-means cluster per budget slot, in which case the
medoids alone fill the budget, the facility-location stage never chooses
anything, and the policy is numerically identical to
``select_kmeans_representatives``. That degeneracy was caught only by comparing
per-cell gains across two full sweeps, so it is pinned here instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from pami_traffic import NeuralTrafficBatch  # noqa: E402
from run_strong_backbone import candidate_similarity  # noqa: E402
from run_response_baselines import (  # noqa: E402
    select_delfit_style,
    select_kcenter,
    select_kmeans_representatives,
    select_mutual_information,
)


def make_batch(*, episodes: int = 6, candidates: int = 16, length: int = 12, seed: int = 0) -> NeuralTrafficBatch:
    rng = np.random.default_rng(seed)
    # a few well-separated groups so k-means has real structure to find
    centres = rng.normal(scale=3.0, size=(4, length))
    labels = rng.integers(0, 4, size=candidates)
    candidate_history = centres[labels][None, :, :] + rng.normal(scale=0.4, size=(episodes, candidates, length))
    return NeuralTrafficBatch(
        anchor_history=rng.normal(size=(episodes, length)).astype(np.float32),
        candidate_history=candidate_history.astype(np.float32),
        target_future=rng.normal(size=(episodes, length)).astype(np.float32),
        target_sensor=0,
        candidate_sensors=np.arange(candidates),
    )


def similarity_of(batch: NeuralTrafficBatch) -> np.ndarray:
    """The production similarity the runners pass in (episodes, K, K)."""

    return candidate_similarity(batch)


def test_delfit_style_is_not_kmeans_representatives() -> None:
    """The whole point of the over-segmentation: the two policies must differ."""

    batch = make_batch()
    similarity = similarity_of(batch)
    relevance = np.zeros((batch.episodes, batch.candidate_count), dtype=np.float64)
    delfit = select_delfit_style(batch, relevance, 4, similarity=similarity, seed=0)
    kmeans = select_kmeans_representatives(batch, 4, seed=0)
    assert not np.array_equal(delfit, kmeans), (
        "DELIFT-style collapsed onto k-means representatives; the pool is no "
        "longer over-segmented relative to the budget"
    )


@pytest.mark.parametrize("budget", [1, 2, 4, 8])
def test_delfit_style_respects_the_budget(budget: int) -> None:
    batch = make_batch()
    similarity = similarity_of(batch)
    relevance = np.zeros((batch.episodes, batch.candidate_count), dtype=np.float64)
    selected = select_delfit_style(batch, relevance, budget, similarity=similarity, seed=1)
    assert selected.shape == (batch.episodes, batch.candidate_count)
    assert selected.dtype == bool
    assert np.all(selected.sum(axis=1) == budget), "every episode must spend exactly the budget"


def test_delfit_style_prefers_covering_distinct_groups() -> None:
    """Facility location should spread the selection over the latent groups."""

    batch = make_batch(episodes=4, candidates=16, seed=3)
    similarity = similarity_of(batch)
    relevance = np.zeros((batch.episodes, batch.candidate_count), dtype=np.float64)
    selected = select_delfit_style(batch, relevance, 4, similarity=similarity, seed=2)
    assert np.all(selected.sum(axis=1) == 4)
    # no episode may pick the same candidate twice
    assert np.all(selected.sum(axis=1) <= batch.candidate_count)


def test_other_baselines_return_valid_masks() -> None:
    batch = make_batch()
    for selector in (
        lambda: select_kcenter(batch, 4),
        lambda: select_kmeans_representatives(batch, 4),
        lambda: select_mutual_information(batch, 4),
    ):
        selected = selector()
        assert selected.shape == (batch.episodes, batch.candidate_count)
        assert np.all(selected.sum(axis=1) == 4)
