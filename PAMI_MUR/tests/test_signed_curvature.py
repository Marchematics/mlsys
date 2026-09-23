"""Unit tests for the R124 signed-curvature head.

The head must implement ``<g_theta, d_bar> + c_theta(state) * kappa`` with a
*signed, unconstrained* ``c``, start from the fixed-sign solution when
``init_curvature = -0.25``, and keep working with the frozen pairing and
selection machinery.
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

from bilinear_response import RESPONSE_SCALARS, BilinearDemoExpertOps, BilinearResponseRouter  # noqa: E402
from demo_protocol import DemoBatch  # noqa: E402
from prototype_predictor import PrototypeClassifier  # noqa: E402
from signed_curvature import (  # noqa: E402
    SignedCurvatureRouter,
    curvature_summary,
    train_signed_response_model,
)

EMBED = 16


def make_batch(episodes: int = 5, candidates: int = 6, queries: int = 4, classes: int = 3, seed: int = 0) -> DemoBatch:
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


def make_ops(batch: DemoBatch, *, classes: int = 3):
    torch.manual_seed(0)
    model = PrototypeClassifier(EMBED, classes, proj_dim=8)
    return model, BilinearDemoExpertOps(model, device="cpu", chunk_episodes=2, chunk_pairs=8)


def make_config(classes: int, response_dim: int, max_budget: int = 3):
    from mur.model import RouterConfig

    return RouterConfig(
        embedding_dim=EMBED,
        prediction_dim=classes,
        hidden_dim=16,
        set_dim=8,
        depth=2,
        max_budget=max_budget,
        response_dim=response_dim,
    )


def rows_from(batch: DemoBatch, responses: np.ndarray, candidate_index: int = 0):
    rows = batch.episodes
    return (
        torch.from_numpy(batch.query_embedding).float(),
        torch.zeros(rows, 3, EMBED),
        torch.zeros(rows, 3, dtype=torch.bool),
        torch.from_numpy(batch.candidate_embedding[:, candidate_index]).float(),
        torch.ones(rows, 1),
        torch.from_numpy(responses[:, candidate_index]).float(),
    )


def test_score_is_first_order_plus_signed_curvature():
    batch = make_batch()
    _, ops = make_ops(batch)
    responses = ops.response_summary(batch, np.zeros((batch.episodes, batch.candidate_count), dtype=bool))
    classes = ops.num_classes
    router = SignedCurvatureRouter(make_config(classes, responses.shape[-1]), classes)
    query, selected, mask, candidate, cost, response = rows_from(batch, responses)
    with torch.no_grad():
        output = router.signed_head(
            router._features(query, selected, mask, candidate, cost)
        )
        coefficient = output[..., :classes]
        curvature_coefficient = output[..., classes]
        kappa = response[:, RESPONSE_SCALARS + classes]
        expected = (coefficient * response[:, RESPONSE_SCALARS : RESPONSE_SCALARS + classes]).sum(-1) + curvature_coefficient * kappa
        score = router(query, selected, mask, candidate, None, cost, response)
    assert torch.allclose(score, expected, atol=1e-5)


def test_initialisation_reproduces_the_fixed_sign_head():
    """With c frozen at -0.25 the signed head must equal the fixed-sign bilinear score."""

    batch = make_batch()
    _, ops = make_ops(batch)
    responses = ops.response_summary(batch, np.zeros((batch.episodes, batch.candidate_count), dtype=bool))
    classes = ops.num_classes
    config = make_config(classes, responses.shape[-1])
    torch.manual_seed(3)
    signed = SignedCurvatureRouter(config, classes, init_curvature=-0.25)
    signed.eval()
    query, selected, mask, candidate, cost, response = rows_from(batch, responses)
    with torch.no_grad():
        coefficients = signed.curvature_coefficients(query, selected, mask, candidate, cost)
        # starts at the fixed-sign solution (up to the small initial weight scale)
        assert torch.allclose(coefficients, torch.full_like(coefficients, -0.25), atol=2e-3)
        output = signed.signed_head(signed._features(query, selected, mask, candidate, cost))
        kappa = response[:, RESPONSE_SCALARS + classes]
        d_bar = response[:, RESPONSE_SCALARS : RESPONSE_SCALARS + classes]
        reference = (output[..., :classes] * d_bar).sum(-1) + coefficients * kappa
        score = signed(query, selected, mask, candidate, None, cost, response)
    assert torch.allclose(score, reference, atol=1e-6)


def test_curvature_coefficient_is_free_to_change_sign():
    """The head can express a positive association between response size and utility."""

    batch = make_batch()
    _, ops = make_ops(batch)
    responses = ops.response_summary(batch, np.zeros((batch.episodes, batch.candidate_count), dtype=bool))
    classes = ops.num_classes
    router = SignedCurvatureRouter(make_config(classes, responses.shape[-1]), classes, init_curvature=-0.25)
    query, selected, mask, candidate, cost, response = rows_from(batch, responses)
    with torch.no_grad():
        last = router.signed_head[-1]
        last.bias[classes] = 0.5  # force a positive coefficient
        last.weight[classes].zero_()
        score = router(query, selected, mask, candidate, None, cost, response)
        output = router.signed_head(router._features(query, selected, mask, candidate, cost))
        kappa = response[:, RESPONSE_SCALARS + classes]
        first_order = (
            output[..., :classes] * response[:, RESPONSE_SCALARS : RESPONSE_SCALARS + classes]
        ).sum(-1)
        assert torch.allclose(score - first_order, 0.5 * kappa, atol=1e-5)
        # the coefficient is unconstrained: values outside the fixed-sign range are reachable
        last.bias[classes] = 2.0
        coefficients = router.curvature_coefficients(query, selected, mask, candidate, cost)
        assert torch.allclose(coefficients, torch.full_like(coefficients, 2.0), atol=1e-5)
        assert torch.all(coefficients > 0.25)


def test_packed_marginals_and_training_are_consistent():
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
    model = train_signed_response_model(
        data, max_budget=3, seed=0, device="cpu", prediction_dim=3, epochs=10, batch_size=4
    )
    query, selected, mask_t, candidate, cost, response = rows_from(batch, responses)
    with torch.no_grad():
        scores = model(
            query[:, None, :].expand(batch.episodes, batch.candidate_count, EMBED).reshape(-1, EMBED),
            selected[:, None].expand(batch.episodes, batch.candidate_count, 3, EMBED).reshape(-1, 3, EMBED),
            mask_t[:, None].expand(batch.episodes, batch.candidate_count, 3).reshape(-1, 3),
            torch.from_numpy(batch.candidate_embedding.reshape(-1, EMBED)).float(),
            None,
            torch.ones(batch.episodes * batch.candidate_count, 1),
            torch.from_numpy(responses.reshape(-1, responses.shape[-1])).float(),
        )
    assert torch.all(torch.isfinite(scores))
    stats = curvature_summary(model, ops, batch, states=2, seed=1, budget=3)
    assert set(stats) >= {"mean", "std", "fraction_positive", "p_value_vs_zero"}
    assert np.isfinite(stats["mean"]) and np.isfinite(stats["p_value_vs_minus_0_25"])


def test_signed_head_differs_from_fixed_sign_after_training():
    """Training must move c away from the fixed -0.25 start somewhere."""

    batch = make_batch(episodes=8, candidates=5, queries=3, classes=3, seed=11)
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
    model = train_signed_response_model(
        data, max_budget=3, seed=0, device="cpu", prediction_dim=3, epochs=40, batch_size=4
    )
    stats = curvature_summary(model, ops, batch, states=2, seed=2, budget=3)
    assert abs(stats["mean"] + 0.25) > 1e-4 or stats["std"] > 1e-4
