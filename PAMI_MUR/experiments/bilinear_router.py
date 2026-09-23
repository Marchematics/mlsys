"""Theory-shaped router: score = <g(state, candidate), d> - ||d||^2.

The exact identity of Proposition 4 says that, conditional on the state, the
squared-loss marginal utility is affine in the predictor response with a
state-dependent coefficient, minus the *known* curvature penalty:

    E[m(j|A) | x, A] = <g_A(x), d_{A,j}> - ||d_{A,j}||^2 - lambda c_j .

A generic MLP on concatenated features has to discover this bilinear structure
from data, which is exactly what the strong-backbone experiments show it fails
to do. This module hard-codes the structure instead: the encoder is identical to
the frozen router, but the head emits a vector in prediction space that is
dotted with the observed response, and the curvature penalty is subtracted
exactly. It is the minimal-capacity model consistent with the identity, so the
only thing left to learn is the state-dependent coefficient g.
"""

from __future__ import annotations

import torch
from torch import nn

from mur.model import MarginalUtilityRouter, RouterConfig


class BilinearResponseRouter(nn.Module):
    """Drop-in replacement whose score has the form implied by the identity."""

    RESPONSE_SCALARS = 7

    def __init__(self, config: RouterConfig, horizon: int) -> None:
        super().__init__()
        if config.response_dim < self.RESPONSE_SCALARS + horizon:
            raise ValueError(
                "the bilinear head needs the raw response vector appended to the "
                "scalar response features (use --response-identity)"
            )
        self.config = config
        self.horizon = int(horizon)
        self.set_phi = nn.Sequential(
            nn.Linear(config.embedding_dim, config.set_dim),
            nn.GELU(),
            nn.Linear(config.set_dim, config.set_dim),
        )
        self.set_rho = nn.Sequential(nn.Linear(config.set_dim, config.embedding_dim), nn.GELU())
        feature_dim = 5 * config.embedding_dim + 2
        if config.use_prediction_delta:
            feature_dim += config.prediction_dim
        layers: list[nn.Module] = []
        current = feature_dim
        for _ in range(config.depth):
            layers.extend([nn.Linear(current, config.hidden_dim), nn.GELU()])
            current = config.hidden_dim
        layers.append(nn.Linear(current, horizon))
        self.coefficient_head = nn.Sequential(*layers)

    def encode_set(self, selected: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
        mask = selected_mask.to(dtype=selected.dtype).unsqueeze(-1)
        transformed = self.set_phi(selected) * mask
        counts = mask.sum(dim=1)
        pooled = transformed.sum(dim=1) / counts.clamp_min(1.0)
        encoded = self.set_rho(pooled)
        return torch.where(counts > 0.0, encoded, torch.zeros_like(encoded))

    def forward(
        self,
        query: torch.Tensor,
        selected: torch.Tensor,
        selected_mask: torch.Tensor,
        candidate: torch.Tensor,
        prediction_delta: torch.Tensor | None,
        candidate_cost: torch.Tensor,
        candidate_response: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if candidate_response is None:
            raise ValueError("the bilinear head requires response features")
        state = self.encode_set(selected, selected_mask)
        selected_fraction = selected_mask.sum(dim=1, keepdim=True).to(query.dtype) / float(
            self.config.max_budget
        )
        parts = [query, state, candidate, torch.abs(state - candidate), state * candidate]
        if prediction_delta is not None:
            parts.append(prediction_delta)
        parts.extend([candidate_cost, selected_fraction])
        features = torch.cat(parts, dim=-1)
        coefficient = self.coefficient_head(features)
        offset = self.RESPONSE_SCALARS
        response = candidate_response[..., offset : offset + self.horizon]
        return (coefficient * response).sum(dim=-1) - (response ** 2).sum(dim=-1)


def build_router(config: RouterConfig, *, kind: str, horizon: int) -> nn.Module:
    if kind == "bilinear":
        return BilinearResponseRouter(config, horizon)
    if kind == "mlp":
        return MarginalUtilityRouter(config)
    raise ValueError(f"unknown router kind: {kind}")
