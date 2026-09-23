"""Small set-conditioned marginal-utility estimator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RouterConfig:
    embedding_dim: int
    prediction_dim: int = 1
    hidden_dim: int = 64
    set_dim: int = 64
    depth: int = 2
    max_budget: int = 8
    use_prediction_delta: bool = False
    response_dim: int = 0

    def __post_init__(self) -> None:
        if min(
            self.embedding_dim,
            self.prediction_dim,
            self.hidden_dim,
            self.set_dim,
            self.depth,
            self.max_budget,
        ) < 1:
            raise ValueError("all router dimensions and max_budget must be positive")
        if self.response_dim < 0:
            raise ValueError("response_dim must be nonnegative")


class MarginalUtilityRouter(nn.Module):
    """DeepSets state encoder plus one scalar marginal-utility head."""

    def __init__(self, config: RouterConfig) -> None:
        super().__init__()
        self.config = config
        self.set_phi = nn.Sequential(
            nn.Linear(config.embedding_dim, config.set_dim),
            nn.GELU(),
            nn.Linear(config.set_dim, config.set_dim),
        )
        self.set_rho = nn.Sequential(nn.Linear(config.set_dim, config.embedding_dim), nn.GELU())
        feature_dim = 5 * config.embedding_dim + 2
        if config.use_prediction_delta:
            feature_dim += config.prediction_dim
        feature_dim += config.response_dim
        layers: list[nn.Module] = []
        current = feature_dim
        for _ in range(config.depth):
            layers.extend([nn.Linear(current, config.hidden_dim), nn.GELU()])
            current = config.hidden_dim
        layers.append(nn.Linear(current, 1))
        self.utility_head = nn.Sequential(*layers)

    def encode_set(self, selected: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
        """Encode a padded selected set; empty sets map to the zero vector."""

        if selected.ndim != 3 or selected.shape[-1] != self.config.embedding_dim:
            raise ValueError("selected must have shape (batch, set_size, embedding_dim)")
        if selected_mask.shape != selected.shape[:2]:
            raise ValueError("selected_mask must match the first two selected dimensions")
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
        cfg = self.config
        if query.ndim != 2 or query.shape[-1] != cfg.embedding_dim:
            raise ValueError("query must have shape (batch, embedding_dim)")
        if candidate.shape != query.shape:
            raise ValueError("candidate and query must have equal shapes")
        if cfg.use_prediction_delta:
            if prediction_delta is None or prediction_delta.shape != (
                query.shape[0],
                cfg.prediction_dim,
            ):
                raise ValueError("prediction_delta is required and has the wrong shape")
        elif prediction_delta is not None:
            raise ValueError("prediction_delta must be None for MUR-light")
        if candidate_cost.shape != (query.shape[0], 1):
            raise ValueError("candidate_cost must have shape (batch, 1)")
        if cfg.response_dim:
            if candidate_response is None or candidate_response.shape != (
                query.shape[0],
                cfg.response_dim,
            ):
                raise ValueError("candidate_response is required and has the wrong shape")
        elif candidate_response is not None:
            raise ValueError("candidate_response must be None when response_dim is zero")
        state = self.encode_set(selected, selected_mask)
        selected_count = selected_mask.sum(dim=1, keepdim=True).to(query.dtype)
        selected_fraction = selected_count / float(cfg.max_budget)
        parts = [
            query,
            state,
            candidate,
            torch.abs(state - candidate),
            state * candidate,
        ]
        if prediction_delta is not None:
            parts.append(prediction_delta)
        parts.extend([candidate_cost, selected_fraction])
        if candidate_response is not None:
            parts.append(candidate_response)
        features = torch.cat(parts, dim=-1)
        return self.utility_head(features).squeeze(-1)


def utility_training_loss(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    *,
    group_ids: torch.Tensor | None = None,
    ranking_weight: float = 0.0,
) -> torch.Tensor:
    """Huber utility regression with optional within-state pairwise ranking."""

    if predicted.shape != observed.shape or predicted.ndim != 1:
        raise ValueError("predicted and observed must be equal-length vectors")
    if ranking_weight < 0.0:
        raise ValueError("ranking_weight must be nonnegative")
    regression = F.smooth_l1_loss(predicted, observed)
    if ranking_weight == 0.0:
        return regression
    if group_ids is None or group_ids.shape != predicted.shape:
        raise ValueError("group_ids are required for ranking loss")
    same_group = group_ids[:, None] == group_ids[None, :]
    target_difference = observed[:, None] - observed[None, :]
    valid = same_group & (target_difference > 0.0)
    if not torch.any(valid):
        return regression
    predicted_difference = predicted[:, None] - predicted[None, :]
    ranking = F.softplus(-predicted_difference[valid]).mean()
    return regression + ranking_weight * ranking
