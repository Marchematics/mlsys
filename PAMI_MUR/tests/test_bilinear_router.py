"""Tests for the theory-shaped bilinear response router."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT.parent / "KBS_MUR" / "src"))

from mur.model import RouterConfig  # noqa: E402

from bilinear_router import BilinearResponseRouter, build_router  # noqa: E402


def config(response_dim: int = 19) -> RouterConfig:
    return RouterConfig(embedding_dim=6, hidden_dim=16, set_dim=8, depth=2, max_budget=3, response_dim=response_dim)


def tensors(batch: int = 4):
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(batch, 6, generator=generator),
        torch.randn(batch, 3, 6, generator=generator),
        torch.rand(batch, 3, generator=generator) < 0.7,
        torch.randn(batch, 6, generator=generator),
        torch.ones(batch, 1),
        torch.randn(batch, 19, generator=generator),
    )


def test_score_matches_the_analytic_form():
    torch.manual_seed(0)
    model = BilinearResponseRouter(config(), horizon=12)
    query, selected, mask, candidate, cost, response = tensors()
    score = model(query, selected, mask, candidate, None, cost, response)
    delta = response[:, 7:19]
    with torch.no_grad():
        state = model.encode_set(selected, mask)
        features = torch.cat(
            [query, state, candidate, torch.abs(state - candidate), state * candidate, cost,
             mask.sum(1, keepdim=True).to(query.dtype) / 3.0],
            dim=-1,
        )
        coefficient = model.coefficient_head(features)
        expected = (coefficient * delta).sum(-1) - (delta ** 2).sum(-1)
    torch.testing.assert_close(score, expected)


def test_requires_the_raw_response_vector():
    torch.manual_seed(0)
    try:
        BilinearResponseRouter(config(response_dim=7), horizon=12)
    except ValueError as error:
        assert "raw response vector" in str(error)
    else:
        raise AssertionError("expected a ValueError when the response vector is missing")


def test_build_router_kinds():
    assert isinstance(build_router(config(), kind="mlp", horizon=12), torch.nn.Module)
    assert isinstance(build_router(config(), kind="bilinear", horizon=12), BilinearResponseRouter)
    try:
        build_router(config(), kind="nope", horizon=12)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown kind must raise")
