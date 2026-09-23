"""Unit tests for the R110 demonstration-selection family.

Covers the two properties the protocol depends on:

* an unselected demonstration is *exactly* inert (a key-padding mask is not a
  soft weighting), verified both against a compact token tensor and against
  the batched pair evaluation;
* the 7-dimensional response summary carries the frozen traffic semantics,
  computed in class-logit space.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Tiny CPU tests must not fight the shared 14-core box for BLAS threads.
torch.set_num_threads(1)

HERE = Path(__file__).resolve()
PAMI = HERE.parents[1]
sys.path.insert(0, str(PAMI / "experiments"))
sys.path.insert(0, str(PAMI / "experiments" / "demo_selection"))

from demo_protocol import (  # noqa: E402
    BenchmarkConfig,
    DemoBatch,
    build_demo_batch,
    fit_pca,
    l2_normalise,
    project,
    split_indices,
)
from incontext_predictor import (  # noqa: E402
    DemoExpertOps,
    InContextClassifier,
    cross_entropy_from_logits,
    softmax_np,
)
from train_predictor import train_incontext_predictor  # noqa: E402
from vision_data import Features  # noqa: E402

EMBED = 16  # small stand-in for the 768-dim CLIP embedding


def make_batch(
    episodes: int = 6,
    candidates: int = 7,
    queries: int = 5,
    classes: int = 4,
    seed: int = 0,
) -> DemoBatch:
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
    query_features = rng.normal(size=(episodes, queries, EMBED)).astype(np.float32)
    candidate_features = rng.normal(size=(episodes, candidates, EMBED)).astype(np.float32)
    return DemoBatch(
        query_embedding=l2_normalise(query_features.mean(axis=1)),
        candidate_embedding=l2_normalise(candidate_features),
        query_features=query_features,
        candidate_features=candidate_features,
        query_labels=query_labels.astype(np.int64),
        candidate_labels=candidate_labels,
        class_index=query_labels.astype(np.int64),
        episode_index=np.zeros(episodes, dtype=np.int64),
    )


def make_ops(batch: DemoBatch, *, seed: int = 0, classes: int = 4) -> DemoExpertOps:
    torch.manual_seed(seed)
    model = InContextClassifier(
        EMBED, classes, d_model=32, nhead=4, num_layers=2, feed_forward_dim=64, dropout=0.0
    )
    return DemoExpertOps(model, device="cpu", chunk_episodes=3, chunk_pairs=5)


def random_mask(rng, episodes: int, candidates: int, budget: int = 3) -> np.ndarray:
    mask = np.zeros((episodes, candidates), dtype=bool)
    for row in range(episodes):
        size = int(rng.integers(0, budget + 1))
        if size:
            mask[row, rng.choice(candidates, size=size, replace=False)] = True
    return mask


def test_unselected_demos_are_exactly_inert():
    """Removing masked demos from the tensor must not change the query logits."""

    batch = make_batch()
    ops = make_ops(batch)
    rng = np.random.default_rng(1)
    for _ in range(5):
        mask = random_mask(rng, batch.episodes, batch.candidate_count)
        with_mask = ops.predict(batch, mask)
        # Compact tensor: only the selected demos, in the same (sorted) order.
        compact_slots = [np.flatnonzero(mask[row]) for row in range(batch.episodes)]
        width = max(max(len(slots) for slots in compact_slots), 1)
        demo_features = np.zeros((batch.episodes, width, EMBED), dtype=np.float32)
        demo_labels = np.zeros((batch.episodes, width), dtype=np.int64)
        valid = np.zeros((batch.episodes, width), dtype=bool)
        for row, slots in enumerate(compact_slots):
            demo_features[row, : len(slots)] = batch.candidate_features[row, slots]
            demo_labels[row, : len(slots)] = batch.candidate_labels[row, slots]
            valid[row, : len(slots)] = True
        compact = ops._forward(batch.query_features, demo_features, valid, demo_labels)
        assert np.allclose(with_mask, compact, atol=1e-6), "masked demos changed the prediction"


def test_pair_predict_equals_full_mask():
    """Batched pair evaluation must equal the explicit augmented-set evaluation."""

    batch = make_batch(episodes=8, candidates=6)
    ops = make_ops(batch, seed=3)
    rng = np.random.default_rng(2)
    for _ in range(4):
        mask = random_mask(rng, batch.episodes, batch.candidate_count)
        rows, columns = np.nonzero(~mask)
        if rows.size == 0:
            continue
        paired = ops.pair_predict(batch, mask, rows, columns)
        for index, (row, column) in enumerate(zip(rows, columns)):
            augmented = mask.copy()
            augmented[row, column] = True
            reference = ops.predict(batch, augmented)[row]
            assert np.allclose(paired[index], reference, atol=1e-6), "pair prediction is not exact"


def test_candidate_marginals_match_loss_differences():
    batch = make_batch(episodes=4, candidates=5)
    ops = make_ops(batch, seed=5)
    rng = np.random.default_rng(3)
    mask = random_mask(rng, batch.episodes, batch.candidate_count, budget=2)
    marginal = ops.candidate_marginals(batch, mask)
    base = ops.loss(batch, mask)
    for row in range(batch.episodes):
        for column in range(batch.candidate_count):
            if mask[row, column]:
                assert marginal[row, column] == -np.inf
                continue
            augmented = mask.copy()
            augmented[row, column] = True
            expected = base[row] - ops.loss(batch, augmented)[row]
            assert marginal[row, column] == pytest.approx(expected, abs=1e-6)


def test_response_summary_semantics():
    """Feature order and definitions must match the frozen traffic summary."""

    batch = make_batch(episodes=3, candidates=5, queries=4)
    ops = make_ops(batch, seed=7)
    rng = np.random.default_rng(4)
    mask = random_mask(rng, batch.episodes, batch.candidate_count, budget=2)
    summary = ops.response_summary(batch, mask)
    base = ops.predict(batch, mask)
    for row in range(batch.episodes):
        for column in range(batch.candidate_count):
            if mask[row, column]:
                assert np.allclose(summary[row, column], 0.0)
                continue
            augmented = mask.copy()
            augmented[row, column] = True
            after = ops.predict(batch, augmented)[row]
            delta = (after - base[row]).reshape(-1)
            assert summary[row, column, 0] == pytest.approx(base[row].mean(), abs=1e-5)
            assert summary[row, column, 1] == pytest.approx(after.mean(), abs=1e-5)
            assert summary[row, column, 2] == pytest.approx(delta.mean(), abs=1e-5)
            assert summary[row, column, 3] == pytest.approx(delta.std(), abs=1e-5)
            assert summary[row, column, 4] == pytest.approx(np.abs(delta).mean(), abs=1e-5)
            assert summary[row, column, 5] == pytest.approx(
                np.linalg.norm(delta) / np.sqrt(delta.size), abs=1e-5
            )
            assert summary[row, column, 6] == pytest.approx(np.abs(delta).max(), abs=1e-5)


def test_response_summary_respects_candidate_mask():
    batch = make_batch(episodes=3, candidates=6)
    ops = make_ops(batch, seed=11)
    rng = np.random.default_rng(5)
    mask = random_mask(rng, batch.episodes, batch.candidate_count, budget=2)
    candidate_mask = np.zeros_like(mask)
    candidate_mask[:, :2] = True
    summary = ops.response_summary(batch, mask, candidate_mask=candidate_mask)
    outside = summary[:, 2:].reshape(batch.episodes, -1)
    assert np.allclose(outside, 0.0), "candidates outside the shortlist must stay zero"


def test_pack_selected_matches_embeddings():
    batch = make_batch(episodes=4, candidates=6)
    ops = make_ops(batch)
    rng = np.random.default_rng(6)
    mask = random_mask(rng, batch.episodes, batch.candidate_count, budget=3)
    values, valid = ops.pack(batch, mask, max_budget=3)
    for row in range(batch.episodes):
        slots = np.flatnonzero(mask[row])
        assert valid[row].sum() == slots.size
        if slots.size:
            assert np.allclose(values[row, : slots.size], batch.candidate_embedding[row, slots])


def test_cross_entropy_matches_torch():
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(4, 3, 5)).astype(np.float32)
    labels = rng.integers(0, 5, size=4)
    ours = cross_entropy_from_logits(logits, labels)
    reference = torch.nn.functional.cross_entropy(
        torch.from_numpy(logits.reshape(-1, logits.shape[-1])),
        torch.from_numpy(labels).repeat_interleave(logits.shape[1]),
    )
    assert float(reference) == pytest.approx(float(ours.mean()), abs=1e-6)


def test_softmax_hessian_sandwich_holds():
    """First-order term bounds the true marginal from above, Hessian from below."""

    rng = np.random.default_rng(8)
    classes = 6
    for _ in range(200):
        base = rng.normal(size=classes)
        delta = rng.normal(size=classes) * rng.uniform(0.01, 3.0)
        label = int(rng.integers(0, classes))
        one_hot = np.zeros(classes)
        one_hot[label] = 1.0
        p = softmax_np(base[None, :])[0]
        m = -np.log(p[label]) + np.log(softmax_np((base + delta)[None, :])[0][label])
        first = float((one_hot - p) @ delta)
        d2 = float(delta @ delta)
        assert m <= first + 1e-12
        assert m >= first - 0.25 * d2 - 1e-12


def test_incontext_predictor_training_uses_subsets():
    """A short training run must reduce the loss and keep the mask exact."""

    batch = make_batch(episodes=16, candidates=8, queries=4, classes=3, seed=11)
    model, info = train_incontext_predictor(
        batch, num_classes=3, budget=3, epochs=30, batch_size=8, seed=0, device="cpu", log_every=0
    )
    assert info["final_train_ce"] < info["first_train_ce"]
    ops = DemoExpertOps(model, device="cpu", chunk_episodes=4, chunk_pairs=8)
    mask = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    augmented = mask.copy()
    augmented[:, 0] = True
    with_empty = ops.predict(batch, mask)
    with_one = ops.predict(batch, augmented)
    assert not np.allclose(with_empty, with_one), "demonstrations must influence the prediction"


def test_split_indices_partition_train_split():
    rng = np.random.default_rng(0)
    features = Features(
        benchmark="toy",
        train_emb=rng.normal(size=(40, EMBED)).astype(np.float16),
        train_labels=np.repeat(np.arange(4), 10).astype(np.int64),
        test_emb=rng.normal(size=(8, EMBED)).astype(np.float16),
        test_labels=np.repeat(np.arange(4), 2).astype(np.int64),
        class_names=[str(index) for index in range(4)],
        meta={},
    )
    splits = split_indices(features, seed=1)
    for class_index in range(4):
        pool = splits.pool[class_index]
        query = splits.query_half[class_index]
        assert pool.size == 5 and query.size == 5
        assert np.intersect1d(pool, query).size == 0
        assert np.array_equal(np.sort(np.concatenate([pool, query])), np.flatnonzero(features.train_labels == class_index))


def test_demo_batch_structure_and_pool():
    """Pool = nearest in-class demos first, then other-class distractors."""

    rng = np.random.default_rng(2)
    train_emb = rng.normal(size=(80, EMBED)).astype(np.float16)
    train_labels = np.repeat(np.arange(4), 20).astype(np.int64)
    # Class-0 images are split into two clusters so nearest neighbours are well defined.
    train_emb[:10] += 1.0
    features = Features(
        benchmark="toy",
        train_emb=train_emb,
        train_labels=train_labels,
        test_emb=rng.normal(size=(8, EMBED)).astype(np.float16),
        test_labels=np.repeat(np.arange(4), 2).astype(np.int64),
        class_names=[str(index) for index in range(4)],
        meta={},
    )
    mean, components = fit_pca(features, dim=8)
    splits = split_indices(features, seed=1)
    batch = build_demo_batch(
        features,
        mean,
        components,
        query_source=splits.test,
        pool_source=splits.pool,
        distractor_pool=splits.distractor_pool,
        query_split="test",
        episodes_per_class=1,
        seed=0,
        config=BenchmarkConfig("toy", 1, 1, query_batch_size=4, candidate_count=6, in_class_candidates=3),
    )
    assert batch.episodes == 4
    assert batch.candidate_features.shape[1] == 6
    for row in range(batch.episodes):
        in_class = batch.candidate_labels[row, :3]
        distractors = batch.candidate_labels[row, 3:]
        assert np.all(in_class == batch.query_labels[row])
        assert np.all(distractors != batch.query_labels[row])
        pool_members = set(splits.pool[int(batch.query_labels[row])].tolist())
        for index in range(3):
            candidate = batch.candidate_features[row, index]
            # The candidate must come from the pool half of the target class.
            matching = np.flatnonzero(
                np.all(np.isclose(np.asarray(features.train_emb), candidate, atol=1e-3), axis=1)
            )
            assert any(int(item) in pool_members for item in matching)
        # Query images never appear among the candidates (disjoint halves).
        assert not np.any(
            np.all(np.isclose(batch.candidate_features[row][:, None, :], batch.query_features[row][None], atol=1e-6), axis=-1)
        )
    assert batch.query_embedding.shape == (4, 8)
    assert np.allclose(np.linalg.norm(batch.query_embedding, axis=1), 1.0, atol=1e-5)
    assert np.allclose(project(np.asarray(features.train_emb[:2], dtype=np.float32), mean, components).shape, (2, 8))


def test_fast_pair_sampler_matches_frozen_reference():
    """The fast pair sampler must produce exactly the frozen sampler's targets."""

    try:
        from run_strong_backbone import sample_pair_dataset_fast
    except ImportError:  # pragma: no cover - helper lives in the PAMI workspace
        pytest.skip("sample_pair_dataset_fast unavailable")
    from mur.synthetic_experiment import sample_pair_dataset

    batch = make_batch(episodes=6, candidates=6, queries=4, classes=3, seed=21)
    ops = make_ops(batch, seed=13, classes=3)
    budget = 3
    marginal_fn = lambda b, s: ops.candidate_marginals(b, s)  # noqa: E731
    pack_fn = lambda b, s, max_budget: ops.pack(b, s, max_budget=max_budget)  # noqa: E731
    for static_only in (True, False):
        reference = sample_pair_dataset(
            batch,
            max_budget=budget,
            states_per_episode=2,
            seed=7,
            static_only=static_only,
            marginal_fn=marginal_fn,
            pack_fn=pack_fn,
        )
        fast = sample_pair_dataset_fast(
            batch, ops, max_budget=budget, states_per_episode=2, seed=7, static_only=static_only
        )
        assert np.allclose(reference.target, fast.target, atol=1e-6)
        assert np.array_equal(reference.selected, fast.selected)
        assert np.array_equal(reference.selected_mask, fast.selected_mask)
        assert np.array_equal(reference.candidate, fast.candidate)


def test_query_tokens_are_conditionally_independent():
    """A query's logits must not depend on the other query images of its batch.

    Only the demonstrations and the query's own embedding may carry
    information; otherwise the same-class query batch votes on its shared label
    and the benchmark collapses.
    """

    batch = make_batch(episodes=3, candidates=6, queries=5, classes=3, seed=31)
    ops = make_ops(batch, seed=17, classes=3)
    mask = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    mask[:, :2] = True
    reference = ops.predict(batch, mask)
    rng = np.random.default_rng(5)
    perturbed = DemoBatch(
        query_embedding=batch.query_embedding,
        candidate_embedding=batch.candidate_embedding,
        query_features=batch.query_features.copy(),
        candidate_features=batch.candidate_features,
        query_labels=batch.query_labels,
        candidate_labels=batch.candidate_labels,
        class_index=batch.class_index,
        episode_index=batch.episode_index,
    )
    # Replace query image 3 of episode 0 with a totally different vector.
    perturbed.query_features[0, 3] = rng.normal(size=EMBED).astype(np.float32) * 5.0
    changed = ops.predict(perturbed, mask)
    for query in range(batch.query_batch_size):
        if query == 3:
            continue
        assert np.allclose(reference[0, query], changed[0, query], atol=1e-6), (
            "another query image changed this query's logits"
        )
    assert not np.allclose(reference[0, 3], changed[0, 3]), "the perturbed query must change"


def test_independence_mask_structure():
    model = InContextClassifier(EMBED, 4, d_model=16, nhead=2, num_layers=1, feed_forward_dim=32)
    mask = model.independence_mask(3, 2, device="cpu")
    assert mask.shape == (5, 5)
    assert not mask[0, 0] and not mask[1, 1]           # self attention allowed
    assert mask[0, 1] and mask[1, 0] and mask[2, 0]    # query-query blocked
    assert not mask[0, 3] and not mask[4, 3]           # query -> demo allowed, demo -> demo allowed
    assert mask[3, 0] and mask[4, 2]                   # demo -> query blocked
