"""R124: signed, unconstrained curvature coefficient for the second family.

R123's theory-shaped head scores ``<g_theta, d_bar> - lambda * kappa`` with
``lambda >= 0`` (fixed at 0.25, or learned inside ``[0, 0.25]``).  The R123 data
show the association between response magnitude ``kappa`` and true marginal
utility *flips sign across benchmarks*: ``corr(score, kappa)`` is +0.257 on
CIFAR-10 but -0.590 on CIFAR-100 and -0.360 on SVHN.  A term with a fixed
negative sign can express the negative association but structurally cannot
express the positive one, which is exactly where CIFAR-10 lost 0.011 nats.

This module adds a third head that releases that constraint:

    score(state, j) = <g_theta(state, candidate), d_bar_j> + c_theta(state) * kappa_j

with ``c_theta`` a *signed, unconstrained* scalar produced by the same network
(one extra output unit, initialised at -0.25 so training starts from the
fixed-sign solution).  Everything else -- response block, features, training
recipe, evaluation stack -- is inherited unchanged.

Note on scope: for squared loss the exact identity *fixes* the curvature
coefficient at ``c = -1`` (the second-order term is realised exactly, not
bounded), so a learned signed ``c`` is only meaningful for the cross-entropy
family; the traffic experiments keep the exact form.  Here ``c`` is reported
per benchmark (mean, spread, and a test against -0.25 and 0) so the mechanism
is visible: if ``c`` moves towards zero (or positive) exactly where the fixed
sign hurt, the sign constraint is confirmed as the binding restriction.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PAMI = HERE.parents[1]
ROOT = PAMI.parent
for _path in (HERE, PAMI / "experiments", ROOT / "KBS_MUR" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mur.model import RouterConfig  # noqa: E402
from bilinear_response import RESPONSE_SCALARS, BilinearResponseRouter  # noqa: E402


class SignedCurvatureRouter(BilinearResponseRouter):
    """``<g_theta, d_bar> + c_theta(state) * kappa`` with an unconstrained ``c``."""

    def __init__(
        self,
        config: RouterConfig,
        prediction_dim: int,
        *,
        init_curvature: float = -0.25,
    ) -> None:
        super().__init__(config, prediction_dim, curvature_mode="bound")
        self.init_curvature = float(init_curvature)
        feature_dim = 5 * config.embedding_dim + 2
        if config.use_prediction_delta:
            feature_dim += config.prediction_dim
        layers: list[nn.Module] = []
        current = feature_dim
        for _ in range(config.depth):
            layers.extend([nn.Linear(current, config.hidden_dim), nn.GELU()])
            current = config.hidden_dim
        layers.append(nn.Linear(current, self.prediction_dim + 1))
        self.signed_head = nn.Sequential(*layers)
        # Start from the fixed-sign solution: last unit's bias = init_curvature,
        # its incoming weights near zero so c is nearly constant at first.
        last = self.signed_head[-1]
        with torch.no_grad():
            last.bias.zero_()
            last.bias[self.prediction_dim] = self.init_curvature
            last.weight[self.prediction_dim].mul_(0.01)

    def _features(
        self,
        query: torch.Tensor,
        selected: torch.Tensor,
        selected_mask: torch.Tensor,
        candidate: torch.Tensor,
        candidate_cost: torch.Tensor,
    ) -> torch.Tensor:
        state = self.encode_set(selected, selected_mask)
        selected_fraction = selected_mask.sum(dim=1, keepdim=True).to(query.dtype) / float(
            self.config.max_budget
        )
        return torch.cat(
            [
                query,
                state,
                candidate,
                torch.abs(state - candidate),
                state * candidate,
                candidate_cost,
                selected_fraction,
            ],
            dim=-1,
        )

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
            raise ValueError("the signed head requires response features")
        output = self.signed_head(
            self._features(query, selected, selected_mask, candidate, candidate_cost)
        )
        coefficient = output[..., : self.prediction_dim]
        curvature_coefficient = output[..., self.prediction_dim]
        offset = RESPONSE_SCALARS
        response = candidate_response[..., offset : offset + self.prediction_dim]
        curvature = candidate_response[..., offset + self.prediction_dim]
        return (coefficient * response).sum(dim=-1) + curvature_coefficient * curvature

    @torch.no_grad()
    def curvature_coefficients(
        self,
        query: torch.Tensor,
        selected: torch.Tensor,
        selected_mask: torch.Tensor,
        candidate: torch.Tensor,
        candidate_cost: torch.Tensor,
    ) -> torch.Tensor:
        """The learned signed scalar ``c_theta(state, candidate)`` per row."""

        output = self.signed_head(
            self._features(query, selected, selected_mask, candidate, candidate_cost)
        )
        return output[..., self.prediction_dim]


def train_signed_response_model(
    data: dict,
    *,
    max_budget: int,
    seed: int,
    device: str,
    prediction_dim: int,
    epochs: int = 20,
    batch_size: int = 128,
    beta: float = 1.0,
    init_curvature: float = -0.25,
):
    """Frozen training recipe (Huber + gap-weighted pairwise ranking) for the signed head."""

    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    response_dim = int(data["response"].shape[-1])
    config = RouterConfig(
        embedding_dim=int(data["query"].shape[1]),
        prediction_dim=prediction_dim,
        hidden_dim=64,
        set_dim=64,
        depth=2,
        max_budget=max_budget,
        response_dim=response_dim,
    )
    model = SignedCurvatureRouter(
        config, prediction_dim, init_curvature=init_curvature
    ).to(device)
    tensors = [
        torch.from_numpy(data["query"]).float(),
        torch.from_numpy(data["selected"]).float(),
        torch.from_numpy(data["selected_mask"]).bool(),
        torch.from_numpy(data["candidate"]).float(),
        torch.from_numpy(data["response"]).float(),
        torch.from_numpy(data["target"]).float(),
    ]
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=min(batch_size, len(tensors[0])),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for query, selected, selected_mask, candidate, response, target in loader:
            states = query.shape[0]
            k = candidate.shape[1]
            query_flat = query.to(device)[:, None, :].expand(states, k, query.shape[1]).reshape(states * k, -1)
            selected_flat = (
                selected.to(device)[:, None, :, :]
                .expand(states, k, selected.shape[1], selected.shape[2])
                .reshape(states * k, selected.shape[1], selected.shape[2])
            )
            mask_flat = (
                selected_mask.to(device)[:, None, :]
                .expand(states, k, selected_mask.shape[1])
                .reshape(states * k, selected_mask.shape[1])
            )
            candidate_flat = candidate.to(device).reshape(states * k, -1)
            response_flat = response.to(device).reshape(states * k, -1)
            cost_flat = torch.ones((states * k, 1), device=device)
            prediction = model(
                query_flat, selected_flat, mask_flat, candidate_flat, None, cost_flat, response_flat
            ).reshape(states, k)
            target_dev = target.to(device)
            valid = torch.isfinite(target_dev)
            huber = F.smooth_l1_loss(prediction[valid], target_dev[valid])
            gap = target_dev[:, :, None] - target_dev[:, None, :]
            pair_mask = (
                torch.isfinite(target_dev)[:, :, None]
                & torch.isfinite(target_dev)[:, None, :]
                & (gap > 0.0)
            )
            if torch.any(pair_mask):
                prediction_gap = prediction[:, :, None] - prediction[:, None, :]
                weights = gap[pair_mask]
                ranking = (weights * F.softplus(-prediction_gap[pair_mask])).sum() / weights.sum().clamp_min(1e-8)
                loss = huber + beta * ranking
            else:
                loss = huber
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def curvature_summary(
    model: SignedCurvatureRouter, ops, batch, *, states: int, seed: int, budget: int = 4
) -> dict:
    """Statistics of the learned signed coefficient on sampled test states."""

    from scipy.stats import ttest_1samp

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(states):
        mask = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
        for row in range(batch.episodes):
            size = int(rng.integers(0, budget + 1))
            if size:
                mask[row, rng.choice(batch.candidate_count, size=min(size, batch.candidate_count), replace=False)] = True
        selected, selected_valid = ops.pack(batch, mask, max_budget=budget)
        rows = np.repeat(np.arange(batch.episodes), batch.candidate_count)
        columns = np.tile(np.arange(batch.candidate_count), batch.episodes)
        query = torch.from_numpy(batch.query_embedding[rows]).float().to(ops.device)
        chosen = torch.from_numpy(batch.candidate_embedding[rows, columns]).float().to(ops.device)
        packed = torch.from_numpy(selected[rows]).float().to(ops.device)
        packed_mask = torch.from_numpy(selected_valid[rows]).to(ops.device)
        cost = torch.ones((rows.size, 1), device=ops.device)
        coefficients = model.curvature_coefficients(query, packed, packed_mask, chosen, cost)
        values.append(coefficients.detach().cpu().numpy())
    flat = np.concatenate(values).astype(np.float64)
    per_state_means = np.asarray([float(np.mean(piece)) for piece in values])
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "fraction_positive": float(np.mean(flat > 0.0)),
        "fraction_above_minus_0_25": float(np.mean(flat > -0.25)),
        "samples": int(flat.size),
        "p_value_vs_zero": float(ttest_1samp(flat, 0.0).pvalue),
        "p_value_vs_minus_0_25": float(ttest_1samp(flat, -0.25).pvalue),
        "per_state_mean": per_state_means.tolist(),
        "per_state_mean_std": float(np.std(per_state_means)),
    }
