"""Subset-capable transformer predictor for PAMI-level confirmation runs."""

from __future__ import annotations

import torch
from torch import nn


class SubsetTransformerPredictor(nn.Module):
    """Predict a target horizon from an anchor history and selected contexts."""

    def __init__(
        self,
        history_len: int,
        horizon: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        feed_forward_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.horizon = horizon
        self.anchor_embed = nn.Sequential(
            nn.Linear(history_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.candidate_embed = nn.Sequential(
            nn.Linear(history_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.type_embedding = nn.Parameter(torch.zeros(2, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feed_forward_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon),
        )

    def forward(
        self,
        anchor: torch.Tensor,
        candidates: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        if anchor.ndim != 2 or anchor.shape[-1] != self.history_len:
            raise ValueError("anchor must have shape (batch, history_len)")
        if candidates.ndim != 3 or candidates.shape[-1] != self.history_len:
            raise ValueError("candidates must have shape (batch, K, history_len)")
        if selected_mask.shape != candidates.shape[:2]:
            raise ValueError("selected_mask must have shape (batch, K)")

        batch = anchor.shape[0]
        anchor_token = self.anchor_embed(anchor).unsqueeze(1)
        candidate_tokens = self.candidate_embed(candidates)
        tokens = torch.cat([anchor_token, candidate_tokens], dim=1)
        type_ids = torch.cat(
            [
                torch.zeros(batch, 1, dtype=torch.long, device=anchor.device),
                torch.ones(batch, candidates.shape[1], dtype=torch.long, device=anchor.device),
            ],
            dim=1,
        )
        tokens = tokens + self.type_embedding[type_ids]
        padding = torch.cat(
            [
                torch.zeros(batch, 1, dtype=torch.bool, device=anchor.device),
                ~selected_mask,
            ],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        anchor_state = encoded[:, 0]
        candidate_states = encoded[:, 1:]
        weights = selected_mask.to(candidate_states.dtype).unsqueeze(-1)
        pooled = (candidate_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(torch.cat([anchor_state, pooled], dim=-1))
