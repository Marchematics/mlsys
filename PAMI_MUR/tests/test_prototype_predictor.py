"""Unit tests for the R115 query-comparative prototype predictor.

The point of the predictor is that a demonstration can only help through its
image content, so the tests check exactly that: labels alone cannot determine the
logits, padded/unselected demonstrations are inert, the query batch carries no
information across queries, and the frozen batched pair evaluation stays exact.
"""

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

from demo_protocol import DemoBatch  # noqa: E402
from incontext_predictor import DemoExpertOps  # noqa: E402
from prototype_predictor import PrototypeClassifier, train_prototype_predictor  # noqa: E402

EMBED = 16


def make_batch(
    episodes: int = 5, candidates: int = 6, queries: int = 4, classes: int = 3, seed: int = 0
) -> DemoBatch:
    """Toy batch.  Features are *class-structured* when ``structured=True``."""

    rng = np.random.default_rng(seed)
    query_labels = rng.integers(0, classes, size=episodes)
    candidate_labels = np.stack(
        [
            np.concatenate(
                [
                    np.full(candidates // 2, query_labels[row]),
                    rng.integers(0, classes, size=candidates - candidates // 2),
                ]
            )
            for row in range(episodes)
        ]
    ).astype(np.int64)
    return DemoBatch(
        query_embedding=rng.normal(size=(episodes, EMBED)).astype(np.float32),
        candidate_embedding=rng.normal(size=(episodes, candidates, EMBED)).astype(np.float32),
        query_features=rng.normal(size=(episodes, queries, EMBED)).astype(np.float32),
        candidate_features=rng.normal(size=(episodes, candidates, EMBED)).astype(np.float32),
        query_labels=query_labels.astype(np.int64),
        candidate_labels=candidate_labels,
        class_index=query_labels.astype(np.int64),
        episode_index=np.zeros(episodes, dtype=np.int64),
    )


def make_model(classes: int = 3, seed: int = 0) -> PrototypeClassifier:
    torch.manual_seed(seed)
    return PrototypeClassifier(EMBED, classes, proj_dim=8)


def make_ops(model: PrototypeClassifier) -> DemoExpertOps:
    return DemoExpertOps(model, device="cpu", chunk_episodes=2, chunk_pairs=8)


def mask_of(episodes: int, candidates: int, take: int) -> np.ndarray:
    mask = np.zeros((episodes, candidates), dtype=bool)
    mask[:, :take] = True
    return mask


def test_empty_context_is_uniform():
    """With no demonstrations every logit is the fixed orthogonal reference score."""

    batch = make_batch()
    model = make_model()
    ops = make_ops(model)
    logits = ops.predict(batch, np.zeros((batch.episodes, batch.candidate_count), dtype=bool))
    assert np.allclose(logits, 0.0), "no demonstrations must give the uniform reference prediction"
    loss = ops.loss(batch, np.zeros((batch.episodes, batch.candidate_count), dtype=bool))
    assert np.allclose(loss, np.log(3.0), atol=1e-5)


def test_labels_alone_cannot_determine_logits():
    """Same labels, different demonstration images must give different logits."""

    batch = make_batch()
    model = make_model()
    ops = make_ops(model)
    mask = mask_of(batch.episodes, batch.candidate_count, 3)
    reference = ops.predict(batch, mask)
    altered = DemoBatch(
        query_embedding=batch.query_embedding,
        candidate_embedding=batch.candidate_embedding,
        query_features=batch.query_features,
        candidate_features=batch.candidate_features.copy(),
        query_labels=batch.query_labels,
        candidate_labels=batch.candidate_labels,
        class_index=batch.class_index,
        episode_index=batch.episode_index,
    )
    rng = np.random.default_rng(3)
    altered.candidate_features[:, :3] = rng.normal(size=(batch.episodes, 3, EMBED)).astype(np.float32) * 4.0
    changed = ops.predict(altered, mask)
    assert not np.allclose(reference, changed), "logits must depend on demonstration images"
    # ... while choosing a *different subset of same-class demonstrations* keeps
    # the label multiset identical yet must change the prediction.
    same_class = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for episode in range(batch.episodes):
        members = np.flatnonzero(batch.candidate_labels[episode] == batch.query_labels[episode])
        same_class[episode, members[:3]] = True
    alternative = np.zeros_like(same_class)
    for episode in range(batch.episodes):
        members = np.flatnonzero(batch.candidate_labels[episode] == batch.query_labels[episode])
        if members.size >= 4:
            alternative[episode, members[1:4]] = True
        else:
            alternative[episode, members[:3]] = True
    assert np.array_equal(
        same_class.sum(axis=1), alternative.sum(axis=1)
    ), "the label multiset must be identical between the two selections"
    assert not np.allclose(
        ops.predict(batch, same_class), ops.predict(batch, alternative)
    ), "which same-class images are selected must matter"


def test_demo_order_is_irrelevant():
    batch = make_batch()
    model = make_model()
    ops = make_ops(model)
    mask = mask_of(batch.episodes, batch.candidate_count, 3)
    reference = ops.predict(batch, mask)
    order = np.array([2, 0, 1, 3, 4, 5])
    shuffled = DemoBatch(
        query_embedding=batch.query_embedding,
        candidate_embedding=batch.candidate_embedding[:, order],
        query_features=batch.query_features,
        candidate_features=batch.candidate_features[:, order],
        query_labels=batch.query_labels,
        candidate_labels=batch.candidate_labels[:, order],
        class_index=batch.class_index,
        episode_index=batch.episode_index,
    )
    assert np.allclose(reference, ops.predict(shuffled, mask), atol=1e-6)


def test_unselected_demos_are_exactly_inert():
    batch = make_batch()
    model = make_model()
    ops = make_ops(model)
    rng = np.random.default_rng(5)
    for _ in range(4):
        mask = rng.random((batch.episodes, batch.candidate_count)) < 0.5
        full = ops.predict(batch, mask)
        compact = DemoBatch(
            query_embedding=batch.query_embedding,
            candidate_embedding=batch.candidate_embedding,
            query_features=batch.query_features,
            candidate_features=batch.candidate_features * mask[:, :, None]
            + (1.0 - mask[:, :, None]) * 1e6,  # masked demos are wildly different
            query_labels=batch.query_labels,
            candidate_labels=batch.candidate_labels,
            class_index=batch.class_index,
            episode_index=batch.episode_index,
        )
        assert np.allclose(full, ops.predict(compact, mask), atol=1e-5)


def test_queries_are_independent():
    batch = make_batch()
    model = make_model()
    ops = make_ops(model)
    mask = mask_of(batch.episodes, batch.candidate_count, 3)
    reference = ops.predict(batch, mask)
    altered = DemoBatch(
        query_embedding=batch.query_embedding,
        candidate_embedding=batch.candidate_embedding,
        query_features=batch.query_features.copy(),
        candidate_features=batch.candidate_features,
        query_labels=batch.query_labels,
        candidate_labels=batch.candidate_labels,
        class_index=batch.class_index,
        episode_index=batch.episode_index,
    )
    altered.query_features[0, 2] = 5.0
    changed = ops.predict(altered, mask)
    for query in range(batch.query_batch_size):
        if query == 2:
            continue
        assert np.allclose(reference[0, query], changed[0, query], atol=1e-6)


def test_pair_predict_matches_full_mask():
    batch = make_batch(episodes=6, candidates=5)
    model = make_model()
    ops = make_ops(model)
    rng = np.random.default_rng(7)
    mask = rng.random((batch.episodes, batch.candidate_count)) < 0.4
    rows, columns = np.nonzero(~mask)
    paired = ops.pair_predict(batch, mask, rows, columns)
    for index, (row, column) in enumerate(zip(rows, columns)):
        augmented = mask.copy()
        augmented[row, column] = True
        assert np.allclose(paired[index], ops.predict(batch, augmented)[row], atol=1e-6)


def structured_batch(
    episodes: int = 24, candidates: int = 8, queries: int = 4, classes: int = 3, seed: int = 11
) -> DemoBatch:
    """Class-structured toy data: queries and same-class demos share a prototype."""

    rng = np.random.default_rng(seed)
    prototypes = rng.normal(size=(classes, EMBED)).astype(np.float32) * 2.0
    query_labels = rng.integers(0, classes, size=episodes)
    candidate_labels = np.stack(
        [
            np.concatenate(
                [
                    np.full(candidates // 2, query_labels[row]),
                    rng.integers(0, classes, size=candidates - candidates // 2),
                ]
            )
            for row in range(episodes)
        ]
    ).astype(np.int64)
    query_features = np.stack(
        [prototypes[query_labels[row]] + rng.normal(size=(queries, EMBED)) * 0.3 for row in range(episodes)]
    ).astype(np.float32)
    candidate_features = np.stack(
        [
            np.stack(
                [
                    prototypes[candidate_labels[row, index]] + rng.normal(size=EMBED) * 0.3
                    for index in range(candidates)
                ]
            )
            for row in range(episodes)
        ]
    ).astype(np.float32)
    return DemoBatch(
        query_embedding=query_features.mean(axis=1),
        candidate_embedding=candidate_features,
        query_features=query_features,
        candidate_features=candidate_features,
        query_labels=query_labels.astype(np.int64),
        candidate_labels=candidate_labels,
        class_index=query_labels.astype(np.int64),
        episode_index=np.zeros(episodes, dtype=np.int64),
    )


def test_prototype_training_learns_and_uses_demonstrations():
    batch = structured_batch(episodes=24, candidates=8, queries=4, classes=3, seed=11)
    model, info = train_prototype_predictor(
        batch, num_classes=3, budget=3, epochs=60, batch_size=8, seed=0, device="cpu", log_every=0
    )
    # The toy task is easy enough that a random projection already separates the
    # prototypes, so the contract is: clearly better than uniform, no divergence,
    # and demonstrations demonstrably used (real-data learning is checked by the
    # R115 smoke and sweep).
    assert info["final_train_ce"] < 0.75 * np.log(3.0)
    assert info["final_train_ce"] <= info["first_train_ce"] + 0.15
    assert np.isfinite(info["temperature"]) and np.isfinite(info["sharpness"])
    ops = make_ops(model)
    empty = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    relevant = np.zeros_like(empty)
    relevant[:, :3] = True
    assert np.mean(ops.loss(batch, relevant)) < 0.5 * np.mean(ops.loss(batch, empty))


def test_masked_softmax_never_divides_by_zero():
    """A class whose demonstrations are all unselected must stay at the reference score."""

    batch = make_batch(episodes=2, candidates=4, queries=2, classes=4, seed=2)
    model = make_model(classes=4)
    ops = make_ops(model)
    mask = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    mask[:, 0] = True  # only class of candidate 0 present
    logits = ops.predict(batch, mask)
    assert np.all(np.isfinite(logits))
    for episode in range(batch.episodes):
        present = int(batch.candidate_labels[episode, 0])
        for class_index in range(4):
            if class_index != present:
                assert np.allclose(logits[episode, :, class_index], 0.0)
