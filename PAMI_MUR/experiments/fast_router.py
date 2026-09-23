"""Fast, procedure-equivalent versions of the frozen R-MUR training loops.

The reference implementations in ``KBS_MUR`` build a ``TensorDataset`` and a
``DataLoader`` for every utility model. Profiling the strong-backbone runs
showed that the loader (not the expert or the model) dominates the wall clock,
because it fetches millions of individual rows on the host.

These functions keep the *procedure* identical -- same seed, same uniform
random minibatches without replacement, same batch size, same loss, same
optimizer, same epoch count -- but move the whole dataset to the device once
and slice it directly instead of fetching rows one at a time.

They do not replay the reference RNG stream bit for bit: PyTorch's DataLoader
draws an integer from the sampler generator for iterator seeding
(``_base_seed``) whose position in the stream is undocumented and empirically
not a constant per epoch. The drawn permutations are therefore a different but
equally valid realisation of the same sampling distribution. The *training
data* is bit-identical, which is what the equivalence test enforces.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from mur.model import MarginalUtilityRouter, RouterConfig, utility_training_loss

from bilinear_router import build_router


def _move(tensor: torch.Tensor, device: str, dtype=None) -> torch.Tensor:
    tensor = torch.from_numpy(tensor) if not torch.is_tensor(tensor) else tensor
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor.contiguous().to(device)


def train_utility_model_fast(
    pairs,
    *,
    max_budget: int,
    seed: int,
    device: str,
    epochs: int = 30,
    batch_size: int = 512,
    hidden_dim: int = 64,
    set_dim: int = 64,
    depth: int = 2,
) -> MarginalUtilityRouter:
    """Equivalent to ``mur.synthetic_experiment.train_utility_model``."""

    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    use_delta = pairs.prediction_delta is not None
    use_response = pairs.candidate_response is not None
    response_dim = int(pairs.candidate_response.shape[1]) if use_response else 0
    config = RouterConfig(
        embedding_dim=pairs.query.shape[1],
        hidden_dim=hidden_dim,
        set_dim=set_dim,
        depth=depth,
        max_budget=max_budget,
        use_prediction_delta=use_delta,
        response_dim=response_dim,
    )
    model = MarginalUtilityRouter(config).to(device)
    query = _move(pairs.query, device, torch.float32)
    selected = _move(pairs.selected, device, torch.float32)
    selected_mask = _move(pairs.selected_mask, device)
    candidate = _move(pairs.candidate, device, torch.float32)
    prediction_delta = _move(pairs.prediction_delta, device, torch.float32) if use_delta else None
    candidate_response = _move(pairs.candidate_response, device, torch.float32) if use_response else None
    cost = _move(pairs.cost, device, torch.float32)
    target = _move(pairs.target, device, torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    episodes = int(target.shape[0])
    for _ in range(epochs):
        model.train()
        order = torch.randperm(episodes, generator=generator).to(device)
        for start in range(0, episodes, batch_size):
            index = order[start : start + batch_size]
            prediction = model(
                query[index],
                selected[index],
                selected_mask[index],
                candidate[index],
                None if prediction_delta is None else prediction_delta[index],
                cost[index],
                None if candidate_response is None else candidate_response[index],
            )
            loss = utility_training_loss(prediction, target[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def train_ranked_response_model_fast(
    data: dict,
    *,
    max_budget: int,
    seed: int,
    device: str,
    epochs: int = 20,
    batch_size: int = 128,
    beta: float = 1.0,
    hidden_dim: int = 64,
    set_dim: int = 64,
    depth: int = 2,
    curvature_corrected: bool = False,
    horizon: int = 12,
    router_kind: str = "mlp",
) -> MarginalUtilityRouter:
    """Equivalent to ``run_traffic_ranked_response_router_sweep``'s trainer."""

    import torch.nn.functional as F

    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    candidate_count = int(data["candidate"].shape[1])
    response_dim = int(data["response"].shape[-1])
    config = RouterConfig(
        embedding_dim=int(data["query"].shape[1]),
        hidden_dim=hidden_dim,
        set_dim=set_dim,
        depth=depth,
        max_budget=max_budget,
        response_dim=response_dim,
    )
    model = build_router(config, kind=router_kind, horizon=horizon).to(device)
    query = _move(data["query"], device, torch.float32)
    selected = _move(data["selected"], device, torch.float32)
    selected_mask = _move(data["selected_mask"], device)
    candidate = _move(data["candidate"], device, torch.float32)
    response = _move(data["response"], device, torch.float32)
    target = _move(data["target"], device, torch.float32)
    if curvature_corrected:
        # The exact identity is m = <g_A, d> - ||d||^2. The second term is
        # observable at routing time, so it is removed from the regression
        # target and re-subtracted explicitly at inference: the head then only
        # has to learn the identifiable interaction term.
        d_squared = response[..., 5] ** 2 * horizon
        target = target + d_squared
    states = int(target.shape[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(states, generator=generator).to(device)
        for start in range(0, states, batch_size):
            index = order[start : start + batch_size]
            chunk = int(index.numel())
            query_flat = query[index][:, None, :].expand(chunk, candidate_count, query.shape[1]).reshape(chunk * candidate_count, -1)
            selected_flat = (
                selected[index][:, None, :, :]
                .expand(chunk, candidate_count, selected.shape[1], selected.shape[2])
                .reshape(chunk * candidate_count, selected.shape[1], selected.shape[2])
            )
            mask_flat = (
                selected_mask[index][:, None, :]
                .expand(chunk, candidate_count, selected_mask.shape[1])
                .reshape(chunk * candidate_count, selected_mask.shape[1])
            )
            candidate_flat = candidate[index].reshape(chunk * candidate_count, -1)
            response_flat = response[index].reshape(chunk * candidate_count, -1)
            cost_flat = torch.ones((chunk * candidate_count, 1), device=device)
            prediction = model(
                query_flat, selected_flat, mask_flat, candidate_flat, None, cost_flat, response_flat
            ).reshape(chunk, candidate_count)

            target_chunk = target[index]
            valid = torch.isfinite(target_chunk)
            huber = F.smooth_l1_loss(prediction[valid], target_chunk[valid])
            gap = target_chunk[:, :, None] - target_chunk[:, None, :]
            pair_mask = (
                torch.isfinite(target_chunk)[:, :, None]
                & torch.isfinite(target_chunk)[:, None, :]
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


def train_utility_model_reference(pairs, *, max_budget: int, seed: int, device: str, epochs: int = 30, batch_size: int = 512):
    """Reference loader-based trainer, used by the equivalence test only."""

    from mur.synthetic_experiment import train_utility_model

    return train_utility_model(
        pairs, max_budget=max_budget, seed=seed, device=device, epochs=epochs, batch_size=batch_size
    )


class CurvatureCorrectedRouter(torch.nn.Module):
    """Wraps a router whose head predicts the interaction term <g_A, d>.

    Inference subtracts the observable curvature penalty ``||d||^2`` that was
    removed from the target during training, so the effective score is an
    estimate of the marginal utility with the known part treated exactly.
    """

    def __init__(self, base, horizon: int) -> None:
        super().__init__()
        self.base = base
        self.horizon = int(horizon)

    @property
    def config(self):
        return self.base.config

    def forward(self, query, selected, selected_mask, candidate, delta, cost, response):
        output = self.base(query, selected, selected_mask, candidate, delta, cost, response)
        if response is None:
            return output
        return output - response[..., 5] ** 2 * self.horizon
