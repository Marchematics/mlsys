"""R123: theory-shaped bilinear response head for the R115 demonstration family.

The exact cross-entropy sandwich of Proposition 3 in
``PAMI_MUR/refine-logs/THEORY_GENERAL_LOSS.md`` is

    <e_y - p_A, d_{A,j}> - (1/4) E||d_{A,j}||^2  <=  m(j|A)  <=  <e_y - p_A, d_{A,j}>

where ``d_{A,j}`` is the change in class logits produced by adding demonstration
``j`` to the selected set ``A`` (the response convention already used by the R110
response summariser).  A generic scalar MLP over concatenated features has to
*discover* this bilinear structure; the head here hard-codes it:

    score(state, j) = <g_theta(state, candidate), d_bar_j> - lambda * kappa_j

with ``d_bar_j`` the query-batch mean logit change, ``kappa_j`` the *exact*
query-batch mean squared response norm ``mean_i ||d_{i,j}||^2``, and ``lambda``
fixed at ``0.25`` -- the proven spectral bound of the softmax cross-entropy
Hessian -- or optionally learned and clipped to ``[0, 0.25]``.  Because
``<g, d_bar> = mean_i <g, d_i>``, the score is exactly the average of the
proven per-query lower bound for a shared coefficient vector, i.e. it is the
minimal-capacity model consistent with Proposition 3.

The module mirrors ``PAMI_MUR/experiments/bilinear_router.py`` (same encoder
layout, same ``<coefficient, response> - curvature`` form, same response layout
with the seven scalar summary features first); only the curvature term and the
prediction dimension differ, because cross-entropy bounds the curvature by 1/4
instead of realising it exactly as in the squared-loss identity.
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
from incontext_predictor import DemoExpertOps, cross_entropy_from_logits  # noqa: E402

RESPONSE_SCALARS = 7
# cur = kappa^2 index (mean_i ||d_i||^2), cur + 1 = ||d_bar||^2, cur + 2 = realised curvature
CURVATURE_MODES = {
    # (index offset from RESPONSE_SCALARS + prediction_dim, lambda)
    "bound": (0, 0.25),      # 0.25 * mean_i ||d_i||^2 : proven worst-case bound (Prop. 3)
    "realised": (2, 0.5),    # 0.5 * mean_i d_i^T H(p_{A,i}) d_i : exact second-order term
}


class BilinearResponseRouter(nn.Module):
    """Score = <g(state, candidate), d_bar> - lambda * mean_i ||d_i||^2."""

    def __init__(
        self,
        config: RouterConfig,
        prediction_dim: int,
        *,
        curvature_multiplier: float = 0.25,
        learned_curvature: bool = False,
        curvature_mode: str = "bound",
    ) -> None:
        super().__init__()
        if curvature_mode not in CURVATURE_MODES:
            raise ValueError(f"unknown curvature mode: {curvature_mode}")
        self.config = config
        self.prediction_dim = int(prediction_dim)
        self.curvature_mode = curvature_mode
        self.curvature_multiplier = float(CURVATURE_MODES[curvature_mode][1])
        if curvature_multiplier != 0.25:
            self.curvature_multiplier = float(curvature_multiplier)
        self.learned_curvature = bool(learned_curvature)
        needed = RESPONSE_SCALARS + self.prediction_dim + 3
        if config.response_dim < needed:
            raise ValueError(
                f"the bilinear head needs response_dim >= {needed} "
                f"(7 scalars + {self.prediction_dim} response coordinates + curvature terms), "
                f"got {config.response_dim}"
            )
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
        layers.append(nn.Linear(current, self.prediction_dim))
        self.coefficient_head = nn.Sequential(*layers)
        if self.learned_curvature:
            # sigmoid(0) = 0.5 -> start at 0.125, inside the admissible [0, 0.25]
            self.curvature_logit = nn.Parameter(torch.zeros(1))

    def effective_curvature_multiplier(self) -> torch.Tensor:
        if not self.learned_curvature:
            return torch.tensor(self.curvature_multiplier)
        return 0.25 * torch.sigmoid(self.curvature_logit)

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
        offset = RESPONSE_SCALARS
        response = candidate_response[..., offset : offset + self.prediction_dim]
        curvature_index = offset + self.prediction_dim + CURVATURE_MODES[self.curvature_mode][0]
        curvature = candidate_response[..., curvature_index]
        scale = self.effective_curvature_multiplier().to(candidate_response.device)
        return (coefficient * response).sum(dim=-1) - scale * curvature


class BilinearDemoExpertOps(DemoExpertOps):
    """``DemoExpertOps`` whose response block is (7 scalars, d_bar, curvature, ||d_bar||^2)."""

    def __init__(self, model, device: str = "cpu", **kwargs) -> None:
        super().__init__(model, device=device, **kwargs)
        self.prediction_dim = int(getattr(model, "num_classes", 0) or getattr(model, "prediction_dim", 0))
        if not self.prediction_dim:
            raise ValueError("cannot infer the prediction dimension from the model")

    @property
    def response_width(self) -> int:
        return RESPONSE_SCALARS + self.prediction_dim + 3

    def _packed_from_chunks(self, batch, mask, base_logits, rows, columns) -> np.ndarray:
        output = np.zeros((batch.episodes, batch.candidate_count, self.response_width), dtype=np.float32)
        if rows.size == 0:
            return output
        grid = max(int(batch.query_batch_size) * self.prediction_dim, 1)
        for take, logits in self.pair_predict_chunks(batch, mask, rows, columns):
            row = rows[take]
            column = columns[take]
            delta = logits - base_logits[row]  # (P, N, C)
            flat = delta.reshape(delta.shape[0], -1)
            mean_delta = delta.mean(axis=1)  # (P, C)
            output[row, column, 0] = base_logits[row].reshape(row.size, -1).mean(axis=1)
            output[row, column, 1] = logits.reshape(row.size, -1).mean(axis=1)
            output[row, column, 2] = flat.mean(axis=1)
            output[row, column, 3] = flat.std(axis=1)
            output[row, column, 4] = np.abs(flat).mean(axis=1)
            output[row, column, 5] = np.linalg.norm(flat, axis=1) / np.sqrt(grid)
            output[row, column, 6] = np.abs(flat).max(axis=1)
            offset = RESPONSE_SCALARS
            output[row, column, offset : offset + self.prediction_dim] = mean_delta
            output[row, column, offset + self.prediction_dim] = np.sum(delta ** 2, axis=-1).mean(axis=1)
            output[row, column, offset + self.prediction_dim + 1] = np.sum(mean_delta ** 2, axis=-1)
            output[row, column, offset + self.prediction_dim + 2] = self._realised_curvature(
                base_logits[row], delta
            )
        return output

    @staticmethod
    def _realised_curvature(base_logits: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """mean_i d_i^T H(p_{A,i}) d_i with H = diag(p) - p p^T at the *base* point.

        Both ``p_A`` and ``d`` are observable at routing time, so the realised
        second-order term is observable too; the exact second-order expansion uses
        the Hessian on the segment, this is its value at the base point.
        """

        shifted = base_logits - base_logits.max(axis=-1, keepdims=True)
        exponent = np.exp(shifted)
        probabilities = exponent / exponent.sum(axis=-1, keepdims=True)
        weighted = np.sum(probabilities * delta ** 2, axis=-1)
        centred = np.sum(probabilities * delta, axis=-1) ** 2
        return (weighted - centred).mean(axis=1)

    def response_summary(
        self,
        batch,
        selected: np.ndarray,
        *,
        candidate_mask: np.ndarray | None = None,
        projection: np.ndarray | None = None,
        include_interaction: bool = False,
    ) -> np.ndarray:
        if projection is not None or include_interaction:
            raise ValueError("the bilinear response block does not support extra projections")
        mask = np.asarray(selected, dtype=bool)
        if candidate_mask is None:
            candidate_mask = np.ones_like(mask, dtype=bool)
        else:
            candidate_mask = np.asarray(candidate_mask, dtype=bool)
            if candidate_mask.shape != mask.shape:
                raise ValueError("candidate_mask has the wrong shape")
        eligible = (~mask) & candidate_mask
        rows, columns = np.nonzero(eligible)
        base = self.predict(batch, mask)
        return self._packed_from_chunks(batch, mask, base, rows, columns)

    def candidate_marginals_and_responses(
        self,
        batch,
        selected: np.ndarray,
        *,
        projection: np.ndarray | None = None,
        include_interaction: bool = False,
    ):
        """Marginals and packed responses from a single shared pass over the pairs."""

        if projection is not None or include_interaction:
            raise ValueError("the bilinear response block does not support extra projections")
        mask = np.asarray(selected, dtype=bool)
        rows, columns = np.nonzero(~mask)
        marginal = np.full(mask.shape, -np.inf, dtype=np.float64)
        output = np.zeros((batch.episodes, batch.candidate_count, self.response_width), dtype=np.float32)
        if rows.size == 0:
            return marginal, output
        base = self.predict(batch, mask)
        base_loss = cross_entropy_from_logits(base, batch.query_labels)
        grid = max(int(batch.query_batch_size) * self.prediction_dim, 1)
        for take, logits in self.pair_predict_chunks(batch, mask, rows, columns):
            row = rows[take]
            column = columns[take]
            delta = logits - base[row]
            flat = delta.reshape(delta.shape[0], -1)
            mean_delta = delta.mean(axis=1)
            marginal[row, column] = base_loss[row] - cross_entropy_from_logits(
                logits, batch.query_labels[row]
            )
            output[row, column, 0] = base[row].reshape(row.size, -1).mean(axis=1)
            output[row, column, 1] = logits.reshape(row.size, -1).mean(axis=1)
            output[row, column, 2] = flat.mean(axis=1)
            output[row, column, 3] = flat.std(axis=1)
            output[row, column, 4] = np.abs(flat).mean(axis=1)
            output[row, column, 5] = np.linalg.norm(flat, axis=1) / np.sqrt(grid)
            output[row, column, 6] = np.abs(flat).max(axis=1)
            offset = RESPONSE_SCALARS
            output[row, column, offset : offset + self.prediction_dim] = mean_delta
            output[row, column, offset + self.prediction_dim] = np.sum(delta ** 2, axis=-1).mean(axis=1)
            output[row, column, offset + self.prediction_dim + 1] = np.sum(mean_delta ** 2, axis=-1)
            output[row, column, offset + self.prediction_dim + 2] = self._realised_curvature(
                base[row], delta
            )
        return marginal, output

def train_bilinear_response_model(
    data: dict,
    *,
    max_budget: int,
    seed: int,
    device: str,
    prediction_dim: int,
    epochs: int = 20,
    batch_size: int = 128,
    beta: float = 1.0,
    curvature_multiplier: float = 0.25,
    learned_curvature: bool = False,
    curvature_mode: str = "bound",
):
    """Frozen training recipe (Huber + gap-weighted pairwise ranking) for the bilinear head."""

    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    candidate_count = int(data["candidate"].shape[1])
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
    model = BilinearResponseRouter(
        config,
        prediction_dim,
        curvature_multiplier=curvature_multiplier,
        learned_curvature=learned_curvature,
        curvature_mode=curvature_mode,
    ).to(device)
    tensors = [
        torch.from_numpy(data["query"]).float(),
        torch.from_numpy(data["selected"]).float(),
        torch.from_numpy(data["selected_mask"]).bool(),
        torch.from_numpy(data["candidate"]).float(),
        torch.from_numpy(data["response"]).float(),
        torch.from_numpy(data["target"]).float(),
    ]
    dataset = TensorDataset(*tensors)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
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
                query_flat,
                selected_flat,
                mask_flat,
                candidate_flat,
                None,
                cost_flat,
                response_flat,
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
