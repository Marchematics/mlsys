"""Frozen in-context classifier and its subset-evaluation ops (R110).

The predictor is a small transformer over tokens ``[query_1..query_N] +
[demo_1..demo_M]``.  A demo token is ``linear(embedding) + label_embedding``
plus a type embedding; a query token is ``linear(embedding)`` plus the other
type embedding.  Class logits are read from the query positions.  Unselected
demonstrations are removed with a key-padding mask.

The architecture deliberately has **no positional encoding**: the query output
is then invariant to the order of the demonstration tokens, which is what makes
the batched pair evaluation below exact rather than merely close.

``DemoExpertOps`` mirrors ``SubsetExpertOps`` in
``PAMI_MUR/experiments/neural_expert.py`` (same batched pair-prediction
pattern, same 7-dimensional response summarisation), with the squared-error
reduction replaced by query-batch cross-entropy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn


class InContextClassifier(nn.Module):
    """Query + demonstrations token transformer producing per-query class logits."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        *,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        feed_forward_dim: int = 256,
        dropout: float = 0.1,
        independent_queries: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.independent_queries = bool(independent_queries)
        self.query_projection = nn.Linear(embed_dim, d_model)
        self.demo_projection = nn.Linear(embed_dim, d_model)
        self.label_embedding = nn.Embedding(num_classes, d_model)
        self.type_embedding = nn.Parameter(torch.zeros(2, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feed_forward_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        # No positional embedding is registered: demonstration order cannot
        # change the query output.
        self._mask_cache: dict[tuple[int, int], torch.Tensor] = {}

    def independence_mask(self, queries: int, demos: int, *, device) -> torch.Tensor:
        """Attention mask that makes every query token self-contained.

        ``True`` marks a forbidden attention pair.  A query token may attend to
        itself and to the demonstration tokens, but not to other query tokens;
        demonstration tokens attend only to demonstrations.  Each query image is
        therefore scored from its own embedding plus the selected demonstration
        set, exactly like an independent anchor window in the traffic protocol.
        Without this mask the same-class query images of one batch can simply
        vote on their shared label and the benchmark collapses (quantified in
        the R110 report).
        """

        key = (int(queries), int(demos))
        cached = self._mask_cache.get(key)
        if cached is not None and cached.device == torch.device(device):
            return cached
        mask = torch.zeros(key[0] + key[1], key[0] + key[1], dtype=torch.bool, device=device)
        diagonal = torch.arange(key[0], device=device)
        mask[: key[0], : key[0]] = True
        mask[diagonal, diagonal] = False
        mask[key[0] :, : key[0]] = True
        self._mask_cache[key] = mask
        return mask

    def forward(
        self,
        query_features: torch.Tensor,
        demo_features: torch.Tensor,
        demo_mask: torch.Tensor,
        demo_labels: torch.Tensor,
    ) -> torch.Tensor:
        batch, queries = query_features.shape[0], query_features.shape[1]
        demos = demo_features.shape[1]
        query_tokens = self.query_projection(query_features) + self.type_embedding[0]
        demo_tokens = (
            self.demo_projection(demo_features)
            + self.label_embedding(demo_labels)
            + self.type_embedding[1]
        )
        tokens = torch.cat([query_tokens, demo_tokens], dim=1)
        padding = torch.cat(
            [
                torch.zeros(batch, queries, dtype=torch.bool, device=query_features.device),
                ~demo_mask,
            ],
            dim=1,
        )
        encoded = self._encode(
            tokens,
            (
                self.independence_mask(queries, demos, device=query_features.device)
                if self.independent_queries and queries > 1
                else None
            ),
            padding,
        )
        return self.head(self.head_norm(encoded[:, :queries]))

    def _encode(self, tokens: torch.Tensor, attn_mask, padding: torch.Tensor) -> torch.Tensor:
        """Run the encoder while keeping padded positions finite.

        A demonstration row that is entirely padding-masked (its own key is
        masked and, under the independence mask, it may attend only to other
        demonstrations) would take a softmax over an empty set, i.e. NaN, and
        that NaN would poison the query rows through the value aggregation of
        the next layer.  Padded rows are therefore zeroed after every layer;
        they remain exactly inert because their keys stay padding-masked.
        """

        values = tokens
        for layer in self.encoder.layers:
            values = layer(values, src_mask=attn_mask, src_key_padding_mask=padding)
            values = values.masked_fill(padding.unsqueeze(-1), 0.0)
        if self.encoder.norm is not None:  # pragma: no cover - not used here
            values = self.encoder.norm(values)
        return values


def cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean cross-entropy of a query batch, per episode.  ``logits`` (E, N, C)."""

    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    shifted = values - values.max(axis=-1, keepdims=True)
    log_normaliser = np.log(np.exp(shifted).sum(axis=-1))
    picked = np.take_along_axis(shifted, labels[:, None, None], axis=-1)[:, :, 0]
    return np.mean(log_normaliser - picked, axis=1)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


@dataclass
class DemoExpertOps:
    """Frozen-predictor operations with the frozen-protocol call signature."""

    model: InContextClassifier
    device: str = "cuda:0"
    chunk_episodes: int = 32
    chunk_pairs: int = 1024
    rows: int = 0
    calls: int = 0
    seconds: float = 0.0
    _profile: bool = field(default=False)

    def __post_init__(self) -> None:
        self.model.eval()
        self.model.to(self.device)

    # -- accounting --------------------------------------------------------
    def reset_profile(self) -> None:
        self.rows = 0
        self.calls = 0
        self.seconds = 0.0
        self._profile = True

    @property
    def num_classes(self) -> int:
        return int(self.model.num_classes)

    # -- forward -----------------------------------------------------------
    def _forward(
        self,
        query_features: np.ndarray,
        demo_features: np.ndarray,
        demo_mask: np.ndarray,
        demo_labels: np.ndarray,
    ) -> np.ndarray:
        query = torch.from_numpy(np.ascontiguousarray(query_features, dtype=np.float32)).to(self.device)
        demos = torch.from_numpy(np.ascontiguousarray(demo_features, dtype=np.float32)).to(self.device)
        mask = torch.from_numpy(np.ascontiguousarray(demo_mask, dtype=bool)).to(self.device)
        labels = torch.from_numpy(np.ascontiguousarray(demo_labels, dtype=np.int64)).to(self.device)
        if self._profile and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        import time as _time

        started = _time.perf_counter()
        with torch.no_grad():
            logits = self.model(query, demos, mask, labels)
        if self._profile and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        if self._profile:
            self.seconds += _time.perf_counter() - started
            self.calls += 1
            self.rows += int(query_features.shape[0])
        return logits.float().cpu().numpy()

    @staticmethod
    def _demo_slots(mask: np.ndarray) -> np.ndarray:
        """Candidate index per demonstration slot: selected indices first (sorted)."""

        rows, columns = np.nonzero(mask)
        counts = mask.sum(axis=1).astype(np.int64)
        width = max(int(counts.max(initial=0)), 1)
        table = np.zeros((mask.shape[0], width), dtype=np.int64)
        if rows.size:
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            positions = np.arange(rows.size) - np.repeat(starts, counts)
            table[rows, positions] = columns
        return table

    def predict_masks(self, batch, masks: np.ndarray) -> np.ndarray:
        """Logits for a stack of selection masks, shape (M, E, N, C)."""

        masks = np.asarray(masks, dtype=bool)
        if masks.ndim != 3:
            raise ValueError("masks must have shape (M, episodes, candidates)")
        model_count, episodes, candidate_count = masks.shape
        if episodes != batch.episodes or candidate_count != batch.candidate_count:
            raise ValueError("mask shape does not match the batch")
        queries = batch.query_batch_size
        classes = self.num_classes
        output = np.empty((model_count, episodes, queries, classes), dtype=np.float32)
        for start in range(0, episodes, self.chunk_episodes):
            stop = min(start + self.chunk_episodes, episodes)
            width = stop - start
            query = batch.query_features[start:stop]
            candidates = batch.candidate_features[start:stop]
            labels = batch.candidate_labels[start:stop]
            for model_index in range(model_count):
                mask = masks[model_index, start:stop]
                slots = self._demo_slots(mask)
                slot_mask = np.arange(slots.shape[1])[None, :] < mask.sum(axis=1, keepdims=True)
                demo_features = candidates[np.arange(width)[:, None], slots]
                demo_labels = labels[np.arange(width)[:, None], slots]
                output[model_index, start:stop] = self._forward(
                    query, demo_features, slot_mask, demo_labels
                )
        return output

    def predict(self, batch, selected: np.ndarray) -> np.ndarray:
        return self.predict_masks(batch, np.asarray(selected, dtype=bool)[None])[0]

    def pair_predict(
        self,
        batch,
        selected: np.ndarray,
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> np.ndarray:
        """Logits for (episode, candidate) pairs added to their own state.

        Each pair is evaluated with the episode's selected set plus the single
        candidate, in one batched forward pass.  The candidate is inserted at
        its *sorted* position, so the unmasked token sequence is identical to
        the one ``predict_masks`` would build for the augmented set; unselected
        candidates are inert either way.
        """

        rows = np.asarray(rows, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        if rows.size == 0:
            return np.zeros((0, batch.query_batch_size, self.num_classes), dtype=np.float32)
        return np.concatenate(
            [piece for _, piece in self.pair_predict_chunks(batch, selected, rows, columns)], axis=0
        )

    def pair_predict_chunks(
        self,
        batch,
        selected: np.ndarray,
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> Iterator[tuple[slice, np.ndarray]]:
        """Chunked version of :meth:`pair_predict` (bounds host memory)."""

        mask = np.asarray(selected, dtype=bool)
        rows = np.asarray(rows, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        table = self._demo_slots(mask)
        width = table.shape[1] + 1
        counts = mask.sum(axis=1).astype(np.int64)
        for start in range(0, rows.size, self.chunk_pairs):
            stop = min(start + self.chunk_pairs, rows.size)
            take = slice(start, stop)
            row = rows[take]
            column = columns[take]
            positions = np.arange(width - 1)[None, :]
            slot_valid = positions < counts[row][:, None]
            base = np.where(slot_valid, table[row], batch.candidate_count + 1)
            insert = (base < column[:, None]).sum(axis=1)
            slots = np.broadcast_to(np.arange(width)[None, :], (row.size, width))
            left = np.take_along_axis(base, np.clip(slots, 0, width - 2), axis=1)
            right = np.take_along_axis(base, np.clip(slots - 1, 0, width - 2), axis=1)
            pair_slots = np.where(
                slots == insert[:, None],
                column[:, None],
                np.where(slots > insert[:, None], right, left),
            )
            # Slots at or beyond counts+1 are masked; keep their indices legal.
            pair_slots = np.clip(pair_slots, 0, batch.candidate_count - 1)
            pair_mask = slots < (counts[row] + 1)[:, None]
            position = np.arange(row.size)[:, None]
            demo_features = batch.candidate_features[row[:, None], pair_slots]
            demo_labels = batch.candidate_labels[row[:, None], pair_slots]
            yield take, self._forward(
                batch.query_features[row], demo_features, pair_mask, demo_labels
            )

    # -- losses and frozen-expert interface --------------------------------
    def loss(self, batch, selected: np.ndarray) -> np.ndarray:
        """Query-batch mean cross-entropy per episode, shape (E,)."""

        logits = self.predict(batch, selected)
        return cross_entropy_from_logits(logits, batch.query_labels)

    def accuracy(self, batch, selected: np.ndarray) -> np.ndarray:
        """Query-image accuracy per episode, shape (E,)."""

        logits = self.predict(batch, selected)
        prediction = np.argmax(logits, axis=-1)
        return np.mean(prediction == batch.query_labels[:, None], axis=1)

    def candidate_marginals(self, batch, selected: np.ndarray) -> np.ndarray:
        """Utility gain of adding each candidate, shape (E, K); -inf if taken."""

        mask = np.asarray(selected, dtype=bool)
        base = self.loss(batch, mask)
        rows, columns = np.nonzero(~mask)
        output = np.full(mask.shape, -np.inf, dtype=np.float64)
        if rows.size:
            for take, logits in self.pair_predict_chunks(batch, mask, rows, columns):
                row = rows[take]
                column = columns[take]
                after = cross_entropy_from_logits(logits, batch.query_labels[row])
                output[row, column] = base[row] - after
        return output

    def pair_marginals(self, batch, selected: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Marginal utility of one specified candidate per episode, shape (E,)."""

        mask = np.asarray(selected, dtype=bool)
        rows = np.arange(batch.episodes)
        columns = np.asarray(columns, dtype=np.int64)
        base = self.loss(batch, mask)
        after = self.pair_predict(batch, mask, rows, columns)
        return base - cross_entropy_from_logits(after, batch.query_labels)

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
        width = 7
        if projection is not None:
            width += int(np.asarray(projection).shape[1])
        if include_interaction:
            width += 1
        output = np.zeros((batch.episodes, batch.candidate_count, width), dtype=np.float32)
        if rows.size == 0:
            return marginal, output
        base = self.predict(batch, mask)
        base_loss = cross_entropy_from_logits(base, batch.query_labels)
        grid = max(int(batch.query_batch_size) * self.num_classes, 1)
        for take, logits in self.pair_predict_chunks(batch, mask, rows, columns):
            row = rows[take]
            column = columns[take]
            delta = logits - base[row]
            flat = delta.reshape(delta.shape[0], -1)
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
            offset = 7
            if projection is not None:
                weight = np.asarray(projection, dtype=np.float32)
                output[row, column, offset : offset + weight.shape[1]] = flat @ weight
                offset += weight.shape[1]
            if include_interaction:
                output[row, column, offset] = np.sum(
                    base[row].reshape(row.size, -1) * flat, axis=1
                )
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
        """Compact predictor-response summaries, shape (E, K, 7).

        Feature order matches ``response_summary`` in
        ``KBS_MUR/src/mur/traffic.py`` exactly, with the predictor's *class
        logits* playing the role of the traffic expert's forecast vector:

        0 mean base logit, 1 mean logit after adding the candidate, 2 mean
        logit change, 3 std of the change, 4 mean absolute change, 5 RMS of
        the change, 6 max absolute change.
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
        width = 7
        if projection is not None:
            width += int(np.asarray(projection).shape[1])
        if include_interaction:
            width += 1
        output = np.zeros((batch.episodes, batch.candidate_count, width), dtype=np.float32)
        if rows.size == 0:
            return output
        grid = max(int(batch.query_batch_size) * self.num_classes, 1)
        for take, logits in self.pair_predict_chunks(batch, mask, rows, columns):
            row = rows[take]
            column = columns[take]
            delta = logits - base[row]
            flat = delta.reshape(delta.shape[0], -1)
            output[row, column, 0] = base[row].reshape(row.size, -1).mean(axis=1)
            output[row, column, 1] = logits.reshape(row.size, -1).mean(axis=1)
            output[row, column, 2] = flat.mean(axis=1)
            output[row, column, 3] = flat.std(axis=1)
            output[row, column, 4] = np.abs(flat).mean(axis=1)
            output[row, column, 5] = np.linalg.norm(flat, axis=1) / np.sqrt(grid)
            output[row, column, 6] = np.abs(flat).max(axis=1)
            offset = 7
            if projection is not None:
                weight = np.asarray(projection, dtype=np.float32)
                output[row, column, offset : offset + weight.shape[1]] = flat @ weight
                offset += weight.shape[1]
            if include_interaction:
                output[row, column, offset] = np.sum(
                    base[row].reshape(row.size, -1) * flat, axis=1
                )
        return output

    def pack(self, batch, selected: np.ndarray, *, max_budget: int):
        """Packed selected demonstration *embeddings* plus validity mask."""

        mask = np.asarray(selected, dtype=bool)
        if mask.shape != (batch.episodes, batch.candidate_count):
            raise ValueError("selected mask has the wrong shape")
        width = int(batch.candidate_embedding.shape[-1])
        values = np.zeros((batch.episodes, max_budget, width), dtype=np.float32)
        valid = np.zeros((batch.episodes, max_budget), dtype=bool)
        for row in range(batch.episodes):
            indices = np.flatnonzero(mask[row])
            if indices.size > max_budget:
                raise ValueError("selected set exceeds max_budget")
            values[row, : indices.size] = batch.candidate_embedding[row, indices]
            valid[row, : indices.size] = True
        return values, valid


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: InContextClassifier,
    *,
    config: dict,
    history: list[dict],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "predictor": "incontext_classifier",
            "state_dict": model.state_dict(),
            "config": dict(config),
            "history": list(history),
        },
        path,
    )


def load_checkpoint(path: Path, *, device: str) -> tuple[InContextClassifier, dict]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = payload["config"]
    model = InContextClassifier(
        int(config["embed_dim"]), int(config["num_classes"]), **config.get("model_kwargs", {})
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, config
