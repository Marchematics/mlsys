"""Learned subset-selection baseline (policy gradient over candidate choices).

This is the standard "learn to select a subset" competitor: a set-conditioned
policy scores candidates at the current state and samples one candidate per
step without replacement; the policy is trained with REINFORCE against the
realised utility of the sampled subset. At evaluation time the policy is
greedy. It is deliberately a *different* inductive bias from R-MUR: it never
predicts a marginal utility and never sees predictor responses.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class LearnedPolicySelector(nn.Module):
    def __init__(self, embedding_dim: int, *, hidden_dim: int = 64, set_dim: int = 64) -> None:
        super().__init__()
        self.set_phi = nn.Sequential(
            nn.Linear(embedding_dim, set_dim), nn.GELU(), nn.Linear(set_dim, set_dim)
        )
        self.set_rho = nn.Sequential(nn.Linear(set_dim, embedding_dim), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(3 * embedding_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        query: torch.Tensor,
        selected: torch.Tensor,
        selected_mask: torch.Tensor,
        candidate: torch.Tensor,
        budget: int,
    ) -> torch.Tensor:
        weights = selected_mask.to(selected.dtype).unsqueeze(-1)
        pooled = (self.set_phi(selected) * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        state = torch.cat(
            [query, self.set_rho(pooled), (weights.sum(dim=1) / max(budget, 1))], dim=-1
        )
        episodes, candidate_count = candidate.shape[:2]
        features = torch.cat(
            [state[:, None, :].expand(episodes, candidate_count, state.shape[-1]), candidate], dim=-1
        )
        return self.head(features).squeeze(-1)


def train_policy_selector(
    model: LearnedPolicySelector,
    batch,
    ops,
    *,
    budget: int,
    epochs: int = 20,
    batch_size: int = 256,
    seed: int = 0,
    device: str = "cuda:0",
    learning_rate: float = 1e-3,
    cost: float = 0.0,
) -> dict:
    """REINFORCE with a batch-mean baseline on the realised set utility."""

    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model.to(device)
    rng = np.random.default_rng(seed)
    query = torch.from_numpy(batch.query_embedding).float().to(device)
    candidate = torch.from_numpy(batch.candidate_embedding).float().to(device)
    history = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(batch.episodes)
        totals, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            index_tensor = torch.from_numpy(index.astype(np.int64)).to(device)
            sub_batch = _sub_batch(batch, index)
            selected = torch.zeros((len(index), batch.candidate_count), dtype=torch.bool, device=device)
            log_probability = torch.zeros(len(index), device=device)
            for _ in range(budget):
                packed, valid = ops.pack(sub_batch, selected.cpu().numpy(), max_budget=budget)
                logits = model(
                    query[index_tensor],
                    torch.from_numpy(packed).float().to(device),
                    torch.from_numpy(valid).to(device),
                    candidate[index_tensor],
                    budget,
                )
                logits = logits.masked_fill(selected.clone(), float("-inf"))
                distribution = Categorical(logits=logits)
                choice = distribution.sample()
                log_probability = log_probability + distribution.log_prob(choice)
                selected[torch.arange(len(index), device=device), choice] = True
            reward = torch.from_numpy(ops.squared_error(sub_batch, selected.cpu().numpy())).float().to(device)
            reward = reward - cost * budget
            advantage = reward - reward.mean()
            loss = -(advantage.detach() * log_probability).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals += float(reward.mean().item()) * len(index)
            seen += len(index)
        history.append(totals / max(seen, 1))
    model.eval()
    return {"epochs": epochs, "mean_reward_history": history}


def _sub_batch(batch, index: np.ndarray):
    from pami_traffic import NeuralTrafficBatch

    return NeuralTrafficBatch(
        anchor_history=batch.anchor_history[index],
        candidate_history=batch.candidate_history[index],
        target_future=batch.target_future[index],
        target_sensor=batch.target_sensor,
        candidate_sensors=batch.candidate_sensors,
        time_index=None if batch.time_index is None else batch.time_index[index],
    )


@torch.no_grad()
def select_learned_policy(model: LearnedPolicySelector, batch, ops, *, budget: int, device: str) -> np.ndarray:
    """Greedy deployment of the learned policy."""

    model.eval()
    query = torch.from_numpy(batch.query_embedding).float().to(device)
    candidate = torch.from_numpy(batch.candidate_embedding).float().to(device)
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    for _ in range(budget):
        packed, valid = ops.pack(batch, selected, max_budget=budget)
        logits = model(
            query,
            torch.from_numpy(packed).float().to(device),
            torch.from_numpy(valid).to(device),
            candidate,
            budget,
        )
        masked = logits.masked_fill(torch.from_numpy(selected).to(device), float("-inf"))
        choice = torch.argmax(masked, dim=1).cpu().numpy()
        selected[np.arange(batch.episodes), choice] = True
    return selected
