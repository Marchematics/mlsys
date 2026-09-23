"""Subset-capable nonlinear experts for the PAMI strong-backbone confirmation.

Both backbones consume an anchor history and an *arbitrary* subset of
candidate-node histories. They are trained with subset dropout so that every
subset size is in-distribution; the expert is never a full-input predictor
that is merely masked at evaluation time.

Two backbones are provided:
  * ``SubsetSTTransformer``: token transformer over anchor plus candidates.
  * ``SubsetDeepSets``: permutation-invariant MLP with masked mean pooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


class SubsetSTTransformer(nn.Module):
    """Anchor-plus-candidates token transformer with subset padding masks."""

    def __init__(
        self,
        num_nodes: int,
        history_length: int,
        horizon: int,
        *,
        d_model: int = 128,
        node_dim: int = 32,
        nhead: int = 4,
        num_layers: int = 3,
        feed_forward_dim: int = 256,
        dropout: float = 0.1,
        steps_per_day: int = 288,
        days_per_week: int = 7,
    ) -> None:
        super().__init__()
        self.history_length = history_length
        self.horizon = horizon
        self.steps_per_day = steps_per_day
        self.node_embedding = nn.Embedding(num_nodes, node_dim)
        self.node_projection = nn.Linear(node_dim, d_model)
        self.history_embedding = nn.Sequential(
            nn.Linear(history_length, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.type_embedding = nn.Parameter(torch.zeros(2, d_model))
        self.time_embedding = nn.Embedding(steps_per_day, d_model)
        self.week_embedding = nn.Embedding(days_per_week, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feed_forward_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head_norm = nn.LayerNorm(2 * d_model + 1)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon),
        )
        # Explicit linear path from the anchor history: persistence is a strong
        # predictor for these benchmarks and the head should not have to
        # rediscover it through the pooled anchor token.
        self.anchor_skip = nn.Linear(history_length, horizon)

    def forward(
        self,
        anchor: torch.Tensor,
        candidates: torch.Tensor,
        selected_mask: torch.Tensor,
        target_node: torch.Tensor,
        candidate_nodes: torch.Tensor,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = anchor.shape[0]
        anchor_token = (
            self.history_embedding(anchor)
            + self.node_projection(self.node_embedding(target_node))
            + self.type_embedding[0]
        )
        if time_features is not None:
            anchor_token = (
                anchor_token
                + self.time_embedding(time_features[:, 0])
                + self.week_embedding(time_features[:, 1])
            )
        anchor_token = anchor_token.unsqueeze(1)
        candidate_tokens = (
            self.history_embedding(candidates)
            + self.node_projection(self.node_embedding(candidate_nodes))
            + self.type_embedding[1]
        )
        tokens = torch.cat([anchor_token, candidate_tokens], dim=1)
        padding = torch.cat(
            [
                torch.zeros(batch, 1, dtype=torch.bool, device=anchor.device),
                ~selected_mask,
            ],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        weights = selected_mask.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded[:, 1:] * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        empty = (weights.sum(dim=1) < 0.5).to(encoded.dtype)
        features = torch.cat([encoded[:, 0], pooled, empty], dim=-1)
        return self.head(self.head_norm(features)) + self.anchor_skip(anchor)


class SubsetDeepSets(nn.Module):
    """Permutation-invariant nonlinear expert with masked mean pooling."""

    def __init__(
        self,
        num_nodes: int,
        history_length: int,
        horizon: int,
        *,
        d_model: int = 128,
        node_dim: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, node_dim)
        self.node_projection = nn.Linear(node_dim, d_model)
        self.history_embedding = nn.Sequential(
            nn.Linear(history_length, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.phi = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(
        self,
        anchor: torch.Tensor,
        candidates: torch.Tensor,
        selected_mask: torch.Tensor,
        target_node: torch.Tensor,
        candidate_nodes: torch.Tensor,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        anchor_state = self.history_embedding(anchor) + self.node_projection(
            self.node_embedding(target_node)
        )
        candidate_states = self.phi(
            self.history_embedding(candidates)
            + self.node_projection(self.node_embedding(candidate_nodes))
        )
        weights = selected_mask.to(candidate_states.dtype).unsqueeze(-1)
        pooled = (candidate_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        empty = (weights.sum(dim=1) < 0.5).to(candidate_states.dtype)
        return self.head(torch.cat([anchor_state, pooled, empty], dim=-1))


BACKBONES = {
    "subset_transformer": SubsetSTTransformer,
    "subset_deepsets": SubsetDeepSets,
}


def build_backbone(
    name: str, num_nodes: int, history_length: int, horizon: int, **kwargs
) -> nn.Module:
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone: {name}")
    return BACKBONES[name](num_nodes, history_length, horizon, **kwargs)


@dataclass
class SubsetExpertOps:
    """Frozen-expert operations mirroring the ridge expert interface."""

    model: nn.Module
    device: str = "cuda:0"
    chunk_episodes: int = 2048
    calls: int = 0
    rows: int = 0
    seconds: float = 0.0
    _profile: bool = field(default=False)

    def __post_init__(self) -> None:
        self.model.eval()
        self.model.to(self.device)

    def reset_profile(self) -> None:
        self.calls = 0
        self.rows = 0
        self.seconds = 0.0
        self._profile = True

    @torch.no_grad()
    def predict_masks(self, batch, masks: np.ndarray) -> np.ndarray:
        """Predictions for a stack of selection masks, shape (M, E, horizon)."""

        masks = np.asarray(masks, dtype=bool)
        if masks.ndim != 3:
            raise ValueError("masks must have shape (M, episodes, candidates)")
        model_count, episodes, candidate_count = masks.shape
        if episodes != batch.episodes or candidate_count != batch.candidate_count:
            raise ValueError("mask shape does not match the batch")
        horizon = batch.horizon
        output = np.empty((model_count, episodes, horizon), dtype=np.float32)
        base_nodes = batch.candidate_node_matrix
        candidate_nodes = np.broadcast_to(
            base_nodes[None], (model_count, episodes, candidate_count)
        )
        for start in range(0, episodes, self.chunk_episodes):
            stop = min(start + self.chunk_episodes, episodes)
            width = stop - start
            anchor = torch.from_numpy(
                np.ascontiguousarray(batch.anchor_history[start:stop])
            ).to(self.device)
            candidates = torch.from_numpy(
                np.ascontiguousarray(batch.candidate_history[start:stop])
            ).to(self.device)
            expanded_anchor = anchor[None].expand(model_count, width, -1).reshape(
                model_count * width, -1
            )
            expanded_candidates = (
                candidates[None]
                .expand(model_count, width, candidate_count, -1)
                .reshape(model_count * width, candidate_count, -1)
            )
            expanded_mask = torch.from_numpy(
                np.ascontiguousarray(masks[:, start:stop])
            ).reshape(model_count * width, candidate_count).to(self.device)
            target_node = torch.full(
                (model_count * width,),
                int(batch.target_sensor),
                dtype=torch.long,
                device=self.device,
            )
            expanded_nodes = torch.from_numpy(
                np.ascontiguousarray(candidate_nodes[:, start:stop]).reshape(
                    model_count * width, candidate_count
                )
            ).to(self.device)
            time_features = None
            if getattr(batch, "time_index", None) is not None:
                steps_per_day = int(getattr(self.model, "steps_per_day", 288))
                times = np.asarray(batch.time_index)[start:stop]
                features = np.stack(
                    [times % steps_per_day, (times // steps_per_day) % 7], axis=1
                )
                time_features = torch.from_numpy(
                    np.ascontiguousarray(
                        np.broadcast_to(features[None], (model_count, width, 2)).reshape(
                            model_count * width, 2
                        )
                    )
                ).to(self.device)
            if self._profile:
                torch.cuda.synchronize() if str(self.device).startswith("cuda") else None
                import time as _time

                started = _time.perf_counter()
                prediction = self.model(
                    expanded_anchor,
                    expanded_candidates,
                    expanded_mask,
                    target_node,
                    expanded_nodes,
                    time_features,
                )
                torch.cuda.synchronize() if str(self.device).startswith("cuda") else None
                self.seconds += _time.perf_counter() - started
                self.calls += 1
                self.rows += model_count * width
            else:
                prediction = self.model(
                    expanded_anchor,
                    expanded_candidates,
                    expanded_mask,
                    target_node,
                    expanded_nodes,
                    time_features,
                )
            output[:, start:stop] = (
                prediction.reshape(model_count, width, horizon).float().cpu().numpy()
            )
        return output

    def predict(self, batch, selected: np.ndarray) -> np.ndarray:
        return self.predict_masks(batch, np.asarray(selected, dtype=bool)[None])[0]

    def _selected_index_table(self, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Padded selected-candidate indices and counts per episode."""

        rows, columns = np.nonzero(selected)
        counts = selected.sum(axis=1).astype(np.int64)
        width = max(int(counts.max(initial=0)), 1)
        table = np.zeros((selected.shape[0], width), dtype=np.int64)
        if rows.size:
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            positions = np.arange(rows.size) - np.repeat(starts, counts)
            table[rows, positions] = columns
        return table, counts

    def pair_predict(
        self,
        batch,
        selected: np.ndarray,
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> np.ndarray:
        """Predictions for (episode, candidate) pairs added to their own state.

        Each pair is evaluated with the episode's selected set plus the single
        candidate, in one batched forward pass. This is exact (unselected
        candidates are inert) and lets the response stage pay for ``q``
        candidate evaluations instead of the full pool.
        """

        from pami_traffic import NeuralTrafficBatch

        rows = np.asarray(rows, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        if rows.size == 0:
            return np.zeros((0, batch.horizon), dtype=np.float32)
        mask = np.asarray(selected, dtype=bool)
        table, counts = self._selected_index_table(mask)
        width = table.shape[1] + 1
        candidate_list = np.zeros((rows.size, width), dtype=np.int64)
        candidate_list[:, : table.shape[1]] = table[rows]
        # The new candidate takes the slot right after that episode's selected
        # set, so the valid prefix is contiguous and the rest stays masked.
        candidate_list[np.arange(rows.size), counts[rows]] = columns
        pair_mask = np.arange(width)[None, :] < (counts[rows] + 1)[:, None]
        pair_batch = NeuralTrafficBatch(
            anchor_history=batch.anchor_history[rows],
            candidate_history=batch.candidate_history[rows[:, None], candidate_list],
            target_future=batch.target_future[rows],
            target_sensor=batch.target_sensor,
            candidate_sensors=batch.candidate_sensors[candidate_list],
            time_index=None if batch.time_index is None else batch.time_index[rows],
        )
        return self.predict_masks(pair_batch, pair_mask[None])[0]

    def squared_error(self, batch, selected: np.ndarray) -> np.ndarray:
        prediction = self.predict(batch, selected)
        return np.mean(np.square(prediction - batch.target_future), axis=1)

    def pair_losses(
        self, batch, selected: np.ndarray, rows: np.ndarray, columns: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Base loss and augmented loss for the supplied (episode, candidate) pairs."""

        base = self.predict(batch, selected)
        base_loss = np.mean(np.square(base - batch.target_future), axis=1)
        after = self.pair_predict(batch, selected, rows, columns)
        after_loss = np.mean(np.square(after - batch.target_future[rows]), axis=1)
        return base, base_loss, after, after_loss

    def pair_marginals(
        self, batch, selected: np.ndarray, columns: np.ndarray
    ) -> np.ndarray:
        """Marginal utility of one specified candidate per episode, shape (E,)."""

        mask = np.asarray(selected, dtype=bool)
        rows = np.arange(batch.episodes)
        columns = np.asarray(columns, dtype=np.int64)
        base_loss = np.mean(np.square(self.predict(batch, mask) - batch.target_future), axis=1)
        after = self.pair_predict(batch, mask, rows, columns)
        after_loss = np.mean(np.square(after - batch.target_future), axis=1)
        return base_loss - after_loss

    def candidate_marginals(self, batch, selected: np.ndarray):
        """Utility gain of adding each candidate, shape (E, K); -inf if taken."""

        mask = np.asarray(selected, dtype=bool)
        base = self.squared_error(batch, mask)
        rows, columns = np.nonzero(~mask)
        output = np.full(mask.shape, -np.inf, dtype=np.float64)
        if rows.size:
            after = self.pair_predict(batch, mask, rows, columns)
            losses = np.mean(np.square(after - batch.target_future[rows]), axis=1)
            output[rows, columns] = base[rows] - losses
        return output

    def candidate_marginals_and_responses(
        self,
        batch,
        selected: np.ndarray,
        *,
        projection: np.ndarray | None = None,
        include_interaction: bool = False,
    ):
        """Marginals and response features from one shared set of pair evaluations."""

        mask = np.asarray(selected, dtype=bool)
        rows, columns = np.nonzero(~mask)
        marginal = np.full(mask.shape, -np.inf, dtype=np.float64)
        width = 7 + (0 if projection is None else int(np.asarray(projection).shape[1]))
        width += 1 if include_interaction else 0
        output = np.zeros((batch.episodes, batch.candidate_count, width), dtype=np.float32)
        if rows.size == 0:
            return marginal, output
        base = self.predict(batch, mask)
        base_loss = np.mean(np.square(base - batch.target_future), axis=1)
        after = self.pair_predict(batch, mask, rows, columns)
        delta = after - base[rows]
        marginal[rows, columns] = base_loss[rows] - np.mean(
            np.square(after - batch.target_future[rows]), axis=1
        )
        horizon = max(int(batch.horizon), 1)
        output[rows, columns, 0] = np.mean(base[rows], axis=1)
        output[rows, columns, 1] = np.mean(after, axis=1)
        output[rows, columns, 2] = np.mean(delta, axis=1)
        output[rows, columns, 3] = np.std(delta, axis=1)
        output[rows, columns, 4] = np.mean(np.abs(delta), axis=1)
        output[rows, columns, 5] = np.linalg.norm(delta, axis=1) / np.sqrt(horizon)
        output[rows, columns, 6] = np.max(np.abs(delta), axis=1)
        offset = 7
        if projection is not None:
            component_count = int(np.asarray(projection).shape[1])
            output[rows, columns, offset : offset + component_count] = (
                delta @ np.asarray(projection, dtype=np.float32)
            )
            offset += component_count
        if include_interaction:
            # <f_A, d>: the part of the exact squared-loss identity that is
            # observable at routing time (the label-dependent part must be
            # predicted from the state).
            output[rows, columns, offset] = np.sum(base[rows] * delta, axis=1)
        return marginal, output

    def response_summary(
        self,
        batch,
        selected: np.ndarray,
        *,
        candidate_mask: np.ndarray | None = None,
        projection: np.ndarray | None = None,
        include_interaction: bool = False,
    ) -> np.ndarray:
        """Compact response features; order matches the ridge-expert features.

        Only candidates that are both requested (``candidate_mask``) and still
        selectable are evaluated; the rest stay zero exactly as in the ridge
        protocol. Cost therefore scales with the number of requested
        candidates, not with the pool size.
        """

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
        width = 7 + (0 if projection is None else int(np.asarray(projection).shape[1]))
        width += 1 if include_interaction else 0
        output = np.zeros((batch.episodes, batch.candidate_count, width), dtype=np.float32)
        if rows.size == 0:
            return output
        after = self.pair_predict(batch, mask, rows, columns)
        delta = after - base[rows]
        horizon = max(int(batch.horizon), 1)
        output[rows, columns, 0] = np.mean(base[rows], axis=1)
        output[rows, columns, 1] = np.mean(after, axis=1)
        output[rows, columns, 2] = np.mean(delta, axis=1)
        output[rows, columns, 3] = np.std(delta, axis=1)
        output[rows, columns, 4] = np.mean(np.abs(delta), axis=1)
        output[rows, columns, 5] = np.linalg.norm(delta, axis=1) / np.sqrt(horizon)
        output[rows, columns, 6] = np.max(np.abs(delta), axis=1)
        offset = 7
        if projection is not None:
            component_count = int(np.asarray(projection).shape[1])
            output[rows, columns, offset : offset + component_count] = (
                delta @ np.asarray(projection, dtype=np.float32)
            )
            offset += component_count
        if include_interaction:
            output[rows, columns, offset] = np.sum(base[rows] * delta, axis=1)
        return output

    def pack(self, batch, selected: np.ndarray, *, max_budget: int):
        return pack_selected(batch, selected, max_budget=max_budget)


def pack_selected(batch, selected: np.ndarray, *, max_budget: int):
    """Same packing as the ridge protocol: selected histories, zero padded."""

    mask = np.asarray(selected, dtype=bool)
    if mask.shape != (batch.episodes, batch.candidate_count):
        raise ValueError("selected mask has the wrong shape")
    values = np.zeros((batch.episodes, max_budget, batch.history_length), dtype=np.float32)
    valid = np.zeros((batch.episodes, max_budget), dtype=bool)
    for row in range(batch.episodes):
        indices = np.flatnonzero(mask[row])
        if indices.size > max_budget:
            raise ValueError("selected set exceeds max_budget")
        values[row, : indices.size] = batch.candidate_history[row, indices]
        valid[row, : indices.size] = True
    return values, valid


def sample_subset_masks(
    rng: np.random.Generator,
    episodes: int,
    candidate_count: int,
    *,
    budget: int,
    full_probability: float = 0.05,
    empty_probability: float = 0.05,
    long_probability: float = 0.15,
) -> np.ndarray:
    """Subset dropout over sizes: short sets dominate, longer sets are covered.

    The routing protocol evaluates subsets of size at most ``budget`` most of
    the time, but oracle-style diagnostics also score the full candidate pool.
    The mixture keeps both regimes in distribution.
    """

    masks = np.zeros((episodes, candidate_count), dtype=bool)
    draw = rng.random(episodes)
    sizes = rng.integers(0, budget + 1, size=episodes)
    long_draw = rng.random(episodes) < long_probability
    if candidate_count > budget:
        sizes = np.where(long_draw, rng.integers(budget + 1, candidate_count + 1, size=episodes), sizes)
    sizes = np.where(draw < full_probability, candidate_count, sizes)
    sizes = np.where(
        (draw >= full_probability) & (draw < full_probability + empty_probability), 0, sizes
    )
    for row, size in enumerate(sizes):
        if size:
            masks[row, rng.choice(candidate_count, size=int(size), replace=False)] = True
    return masks


def fit_response_projection(
    ops: "SubsetExpertOps",
    batch,
    *,
    components: int,
    budget: int,
    states: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """PCA basis for the predictor response, fitted on training episodes only.

    The frozen protocol summarises a response by scalar statistics, which
    discard its direction. The first-order theory relates utility to the inner
    product of the response with a state-dependent coefficient, so a
    direction-preserving compact summary is the theoretically matched feature
    set. The basis is estimated from the train split and then frozen.
    """

    if components <= 0:
        raise ValueError("components must be positive")
    rng = np.random.default_rng(seed)
    collected = []
    for _ in range(states):
        selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
        sizes = rng.integers(0, budget + 1, size=batch.episodes)
        for row, size in enumerate(sizes):
            if size:
                selected[row, rng.choice(batch.candidate_count, size=int(size), replace=False)] = True
        rows, columns = np.nonzero(~selected)
        if rows.size == 0:
            continue
        base = ops.predict(batch, selected)
        after = ops.pair_predict(batch, selected, rows, columns)
        collected.append(after - base[rows])
    if not collected:
        raise ValueError("no responses collected for the projection")
    responses = np.concatenate(collected, axis=0).astype(np.float64)
    responses -= responses.mean(axis=0, keepdims=True)
    covariance = responses.T @ responses / max(len(responses) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(-values)[:components]
    return np.ascontiguousarray(vectors[:, order], dtype=np.float32)


def finetune_expert(
    model: nn.Module,
    batch,
    *,
    epochs: int = 8,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    budget: int = 4,
    seed: int = 0,
    device: str = "cuda:0",
    validation_batch=None,
) -> dict:
    """Adapt a pretrained expert on one episode batch.

    The ridge expert in the frozen protocol is fitted on exactly this batch
    (`fit_subset_expert(batches[0], ...)`), so this routine keeps the backbone
    comparison apples to apples: identical training episodes, identical
    subset-dropout distribution, different function class. When a validation
    batch (the calibration split of the protocol) is supplied, the best epoch
    is selected on it, which prevents adaptation from overfitting the
    training episodes.
    """

    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(seed)
    anchor = torch.from_numpy(batch.anchor_history).to(device)
    candidate = torch.from_numpy(batch.candidate_history).to(device)
    target = torch.from_numpy(batch.target_future).to(device)
    nodes = torch.full((batch.episodes,), int(batch.target_sensor), dtype=torch.long, device=device)
    pool = torch.from_numpy(
        np.broadcast_to(
            batch.candidate_sensors[None], (batch.episodes, batch.candidate_count)
        ).copy()
    ).to(device)
    time_features = None
    if getattr(batch, "time_index", None) is not None:
        steps_per_day = int(getattr(model, "steps_per_day", 288))
        times = np.asarray(batch.time_index)
        time_features = torch.from_numpy(
            np.stack([times % steps_per_day, (times // steps_per_day) % 7], axis=1)
        ).to(device)
    history = []
    best = {"mse": float("inf"), "epoch": -1, "state": None}
    if validation_batch is not None:
        validation_rng = np.random.default_rng(seed + 1)
        validation_masks = [
            sample_subset_masks(
                validation_rng, validation_batch.episodes, validation_batch.candidate_count,
                budget=size, full_probability=0.0, empty_probability=0.0, long_probability=0.0,
            )
            for size in (0, 2, budget)
        ]
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(batch.episodes)
        total, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            index_tensor = torch.from_numpy(index.astype(np.int64)).to(device)
            masks = sample_subset_masks(rng, len(index), batch.candidate_count, budget=budget)
            prediction = model(
                anchor[index_tensor],
                candidate[index_tensor],
                torch.from_numpy(masks).to(device),
                nodes[index_tensor],
                pool[index_tensor],
                None if time_features is None else time_features[index_tensor],
            )
            loss = loss_fn(prediction, target[index_tensor])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(index)
            seen += len(index)
        history.append(total / max(seen, 1))
        if validation_batch is not None:
            model.eval()
            with torch.no_grad():
                scorer = SubsetExpertOps(model, device=device)
                score = float(np.mean([
                    np.mean(scorer.squared_error(validation_batch, mask)) for mask in validation_masks
                ]))
            if score < best["mse"]:
                best = {
                    "mse": score,
                    "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }
    if validation_batch is not None and best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    return {
        "epochs": epochs,
        "final_train_mse": history[-1] if history else float("nan"),
        "history": history,
        "validation_selected": validation_batch is not None,
        "best_epoch": best["epoch"],
        "best_validation_mse": None if best["epoch"] < 0 else best["mse"],
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    backbone: str,
    config: dict,
    history: Sequence[dict],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": backbone,
            "state_dict": model.state_dict(),
            "config": dict(config),
            "history": list(history),
        },
        path,
    )


def load_checkpoint(path: Path, *, device: str) -> tuple[nn.Module, dict]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = payload["config"]
    model = build_backbone(
        payload["backbone"],
        int(config["num_nodes"]),
        int(config["history_length"]),
        int(config["horizon"]),
        **config.get("backbone_kwargs", {}),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, config
