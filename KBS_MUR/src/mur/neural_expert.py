"""Subset-capable neural predictor used for the backbone-generalisation check.

The predictor has the same interface as the ridge expert in ``mur.traffic`` but
is a small multilayer perceptron over the anchor history and the masked
candidate histories, trained on random context subsets. It exists to test
whether the ordering ``cached-only router < response-aware router`` survives a
predictor family that is not linear, not to improve forecasting accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # torch is optional for the CPU-only analyses
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass(frozen=True)
class NeuralExpert:
    """Frozen subset-capable MLP predictor."""

    weights: dict[str, np.ndarray]
    candidate_count: int
    history_length: int
    horizon: int
    hidden: int


if nn is not None:

    class _MLP(nn.Module):
        def __init__(self, input_dim: int, hidden: int, horizon: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, horizon),
            )

        def forward(self, x):  # noqa: D102
            return self.net(x)


def fit_neural_expert(
    batch,
    *,
    seed: int,
    hidden: int = 128,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    subset_probability: float = 0.5,
    device: str = "cpu",
) -> NeuralExpert:
    """Train the MLP on random context subsets of the training episodes."""

    if torch is None:
        raise RuntimeError("torch is required to fit the neural expert")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    anchor = torch.from_numpy(np.asarray(batch.anchor_history, dtype=np.float32))
    candidates = torch.from_numpy(np.asarray(batch.candidate_history, dtype=np.float32))
    target = torch.from_numpy(np.asarray(batch.target_future, dtype=np.float32))
    episodes, candidate_count, history = candidates.shape
    input_dim = history + candidate_count * history
    model = _MLP(input_dim, hidden, target.shape[1]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for _epoch in range(epochs):
        masks = torch.from_numpy(
            (rng.random((episodes, candidate_count)) < subset_probability).astype(np.float32)
        )
        # keep the empty and the full subset represented in every epoch
        if episodes >= 2:
            masks[0] = 0.0
            masks[1] = 1.0
        order = torch.from_numpy(rng.permutation(episodes))
        for start in range(0, episodes, batch_size):
            index = order[start : start + batch_size]
            features = torch.cat(
                [anchor[index], (candidates[index] * masks[index][:, :, None]).reshape(index.numel(), -1)],
                dim=1,
            ).to(device)
            prediction = model(features)
            loss = loss_fn(prediction, target[index].to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    weights = {name: tensor.detach().cpu().numpy().astype(np.float32) for name, tensor in model.state_dict().items()}
    return NeuralExpert(
        weights=weights,
        candidate_count=candidate_count,
        history_length=history,
        horizon=int(target.shape[1]),
        hidden=hidden,
    )


def neural_prediction(batch, selected: np.ndarray, expert: NeuralExpert) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is required to evaluate the neural expert")
    mask = np.asarray(selected, dtype=bool)
    if mask.shape != (batch.episodes, batch.candidate_count):
        raise ValueError("selected mask has the wrong shape")
    features = np.concatenate(
        [
            np.asarray(batch.anchor_history, dtype=np.float32),
            (np.asarray(batch.candidate_history, dtype=np.float32) * mask[:, :, None]).reshape(
                batch.episodes, -1
            ),
        ],
        axis=1,
    )
    model = _MLP(features.shape[1], expert.hidden, expert.horizon)
    model.load_state_dict({name: torch.from_numpy(value) for name, value in expert.weights.items()})
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(features)).numpy()
    return prediction.astype(np.float32)


def neural_predict_many(batch, masks: np.ndarray, expert: NeuralExpert) -> np.ndarray:
    """Predict for a stack of masks, shape (J, episodes, horizon)."""

    if torch is None:
        raise RuntimeError("torch is required to evaluate the neural expert")
    stack = np.asarray(masks, dtype=bool)
    episodes, candidates = stack.shape[1], stack.shape[2]
    anchor = np.asarray(batch.anchor_history, dtype=np.float32)
    candidate = np.asarray(batch.candidate_history, dtype=np.float32)
    count = stack.shape[0]
    features = np.concatenate(
        [
            np.broadcast_to(anchor[None, :, :], (count, episodes, anchor.shape[1])),
            (candidate[None, :, :, :] * stack[:, :, :, None]).reshape(count, episodes, -1),
        ],
        axis=2,
    ).reshape(count * episodes, -1)
    model = _MLP(features.shape[1], expert.hidden, expert.horizon)
    model.load_state_dict({name: torch.from_numpy(value) for name, value in expert.weights.items()})
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(features)).numpy()
    return prediction.reshape(count, episodes, expert.horizon).astype(np.float32)
