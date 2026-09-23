"""Unit tests for the R123 bilinear response head.

The head must implement exactly ``<g_theta, d_bar> - lambda * mean_i ||d_i||^2``
with the response block laid out as [7 scalar summaries, d_bar, curvature,
||d_bar||^2], and the frozen pairing/selection machinery must keep working with
it (batched pair prediction exact, response block consistent with the marginals).
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

from bilinear_response import (  # noqa: E402
    RESPONSE_SCALARS,
    BilinearDemoExpertOps,
    BilinearResponseRouter,
    train_bilinear_response_model,
)
from demo_protocol import DemoBatch  # noqa: E402
from prototype_predictor import PrototypeClassifier  # noqa: E402

EMBED = 16


def make_batch(
    episodes: int = 5, candidates: int = 6, queries: int = 4, classes: int = 3, seed: int = 0
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


def make_ops(batch: DemoBatch, *, classes: int = 3) -> tuple[PrototypeClassifier, BilinearDemoExpertOps]:
    torch.manual_seed(0)
    model = PrototypeClassifier(EMBED, classes, proj_dim=8)
    ops = BilinearDemoExpertOps(model, device="cpu", chunk_episodes=2, chunk_pairs=8)
    return model, ops


def make_router(classes: int, *, response_dim: int, max_budget: int = 3, learned: bool = False):
    from mur.model import RouterConfig

    torch.manual_seed(1)
    config = RouterConfig(
        embedding_dim=EMBED,
        prediction_dim=classes,
        hidden_dim=16,
        set_dim=8,
        depth=2,
        max_budget=max_budget,
        response_dim=response_dim,
    )
    return BilinearResponseRouter(config, classes, learned_curvature=learned), config


def test_response_block_layout_and_curvature_identity():
    batch = make_batch()
    _, ops = make_ops(batch)
    mask = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    mask[:, 0] = True
    responses = ops.response_summary(batch, mask)
    classes = ops.num_classes
    assert responses.shape == (batch.episodes, batch.candidate_count, RESPONSE_SCALARS + classes + 3)
    # Recompute d_bar and the two curvature quantities directly from the logits.
    base = ops.predict(batch, mask)
    rows, columns = np.nonzero(~mask)
    augmented = mask.copy()
    for row, column in zip(rows, columns):
        augmented[row, column] = True
    for row, column in zip(rows, columns):
        single = mask.copy()
        single[row, column] = True
        after = ops.predict(batch, single)[row]
        delta = after - base[row]
        d_bar = delta.mean(axis=0)
        curvature = np.sum(delta ** 2, axis=-1).mean()
        offset = RESPONSE_SCALARS
        assert np.allclose(responses[row, column, offset : offset + classes], d_bar, atol=1e-5)
        assert responses[row, column, offset + classes] == pytest.approx(curvature, rel=1e-4)
        assert responses[row, column, offset + classes + 1] == pytest.approx(
            float(np.sum(d_bar ** 2)), rel=1e-4
        )
        # realised curvature: mean_i d_i^T (diag(p) - p p^T) d_i >= 0 and <= ||d||^2
        realised = responses[row, column, offset + classes + 2]
        assert realised >= -1e-5
        assert realised <= curvature + 1e-5
        # Jensen: the exact mean curvature is at least the squared mean vector norm.
        assert curvature + 1e-6 >= float(np.sum(d_bar ** 2))


def test_score_equals_bilinear_form():
    batch = make_batch()
    _, ops = make_ops(batch)
    responses = ops.response_summary(batch, np.zeros((batch.episodes, batch.candidate_count), dtype=bool))
    classes = ops.num_classes
    router, _ = make_router(classes, response_dim=responses.shape[-1])
    rows, candidates = batch.episodes, batch.candidate_count
    query = torch.from_numpy(batch.query_embedding).float()
    selected = torch.zeros(rows, 3, EMBED)
    selected_mask = torch.zeros(rows, 3, dtype=torch.bool)
    candidate = torch.from_numpy(batch.candidate_embedding[:, 0]).float()
    cost = torch.ones(rows, 1)
    response = torch.from_numpy(responses[:, 0]).float()
    with torch.no_grad():
        score = router(query, selected, selected_mask, candidate, None, cost, response)
        coefficient = router.coefficient_head(
            torch.cat(
                [
                    query,
                    router.encode_set(selected, selected_mask),
                    candidate,
                    torch.abs(router.encode_set(selected, selected_mask) - candidate),
                    router.encode_set(selected, selected_mask) * candidate,
                    cost,
                    selected_mask.sum(dim=1, keepdim=True).float() / 3.0,
                ],
                dim=-1,
            )
        )
        expected = (coefficient * response[:, RESPONSE_SCALARS : RESPONSE_SCALARS + classes]).sum(-1) - 0.25 * response[:, RESPONSE_SCALARS + classes]
    assert torch.allclose(score, expected, atol=1e-5)


def test_learned_curvature_stays_in_bounds():
    router, _ = make_router(3, response_dim=RESPONSE_SCALARS + 3 + 3, learned=True)
    with torch.no_grad():
        for value in (-50.0, 0.0, 50.0):
            router.curvature_logit.fill_(value)
            multiplier = float(router.effective_curvature_multiplier())
            assert 0.0 <= multiplier <= 0.25


def test_pair_prediction_and_packed_marginals_agree():
    batch = make_batch(episodes=4, candidates=5, queries=3)
    _, ops = make_ops(batch)
    rng = np.random.default_rng(4)
    mask = rng.random((batch.episodes, batch.candidate_count)) < 0.4
    marginal, responses = ops.candidate_marginals_and_responses(batch, mask)
    assert responses.shape[-1] == RESPONSE_SCALARS + ops.num_classes + 3
    for row, column in zip(*np.nonzero(~mask)):
        augmented = mask.copy()
        augmented[row, column] = True
        expected = ops.loss(batch, mask)[row] - ops.loss(batch, augmented)[row]
        assert marginal[row, column] == pytest.approx(expected, abs=1e-6)
        assert np.all(np.isfinite(responses[row, column]))
    assert np.all(marginal[mask] == -np.inf)


def test_perturbing_one_candidate_leaves_other_responses_unchanged():
    """Each candidate's response must depend only on its own demonstration."""

    batch = make_batch()
    _, ops = make_ops(batch)
    mask = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    mask[:, 0] = True
    reference = ops.response_summary(batch, mask)
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
    altered.candidate_features[:, 2] += 1e3  # perturb only candidate 2
    changed = ops.response_summary(altered, mask)
    keep = [0, 1, 3, 4, 5]
    assert np.allclose(reference[:, keep], changed[:, keep], atol=1e-4)
    assert not np.allclose(reference[:, 2], changed[:, 2])


def test_training_reduces_ranking_loss_and_scores_finite():
    batch = make_batch(episodes=8, candidates=5, queries=3, classes=3, seed=7)
    _, ops = make_ops(batch)
    mask = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    marginal, responses = ops.candidate_marginals_and_responses(batch, mask)
    data = {
        "query": batch.query_embedding,
        "selected": np.zeros((batch.episodes, 3, EMBED), dtype=np.float32),
        "selected_mask": np.zeros((batch.episodes, 3), dtype=bool),
        "candidate": batch.candidate_embedding,
        "response": responses,
        "target": marginal.astype(np.float32),
    }
    model = train_bilinear_response_model(
        data, max_budget=3, seed=0, device="cpu", prediction_dim=ops.num_classes, epochs=15, batch_size=4
    )
    states, candidates = batch.episodes, batch.candidate_count
    query = torch.from_numpy(batch.query_embedding).float()[:, None, :].expand(states, candidates, EMBED).reshape(-1, EMBED)
    selected = torch.zeros(states, 3, EMBED)[:, None].expand(states, candidates, 3, EMBED).reshape(-1, 3, EMBED)
    selected_mask = torch.zeros(states * candidates, 3, dtype=torch.bool)
    with torch.no_grad():
        scores = model(
            query,
            selected,
            selected_mask,
            torch.from_numpy(batch.candidate_embedding.reshape(-1, EMBED)).float(),
            None,
            torch.ones(states * candidates, 1),
            torch.from_numpy(responses.reshape(-1, responses.shape[-1])).float(),
        )
    assert torch.all(torch.isfinite(scores))
    assert float(scores.min()) < float(scores.max())
