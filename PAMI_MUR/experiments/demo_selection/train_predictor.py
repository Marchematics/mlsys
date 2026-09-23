"""Subset-dropout training of the frozen in-context demonstration classifier.

Training episodes come from the TRAIN split only (query images from the query
half, candidate demonstrations from the disjoint pool half).  Every gradient
step draws a fresh random subset of the candidate pool with the frozen
protocol's subset-dropout mixture (mostly sizes ``0..B``, occasionally longer
and occasionally the full pool), so the predictor is a subset model and never a
full-context model that is merely masked at evaluation time.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAMI = HERE.parents[1]
for _path in (HERE, PAMI / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from incontext_predictor import InContextClassifier  # noqa: E402
from neural_expert import sample_subset_masks  # noqa: E402  (frozen protocol helper)


def train_incontext_predictor(
    batch,
    *,
    num_classes: int,
    budget: int,
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    seed: int = 0,
    device: str = "cuda:0",
    full_probability: float = 0.05,
    empty_probability: float = 0.05,
    long_probability: float = 0.15,
    grad_clip: float = 1.0,
    log_every: int = 0,
    independent_queries: bool = True,
) -> tuple[InContextClassifier, dict]:
    """Fit the in-context classifier with subset dropout on one episode batch."""

    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    embed_dim = int(batch.query_features.shape[-1])
    model = InContextClassifier(
        embed_dim, num_classes, independent_queries=independent_queries
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    query = torch.from_numpy(np.ascontiguousarray(batch.query_features, dtype=np.float32)).to(device)
    demos = torch.from_numpy(np.ascontiguousarray(batch.candidate_features, dtype=np.float32)).to(device)
    demo_labels = torch.from_numpy(np.ascontiguousarray(batch.candidate_labels, dtype=np.int64)).to(device)
    query_labels = torch.from_numpy(np.ascontiguousarray(batch.query_labels, dtype=np.int64)).to(device)
    episodes = batch.episodes
    queries = batch.query_batch_size
    candidates = batch.candidate_count

    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(episodes)
        total, seen = 0.0, 0
        for start in range(0, episodes, batch_size):
            index = order[start : start + batch_size]
            index_tensor = torch.from_numpy(index.astype(np.int64)).to(device)
            masks = sample_subset_masks(
                rng,
                len(index),
                candidates,
                budget=budget,
                full_probability=full_probability,
                empty_probability=empty_probability,
                long_probability=long_probability,
            )
            mask_tensor = torch.from_numpy(masks).to(device)
            logits = model(query[index_tensor], demos[index_tensor], mask_tensor, demo_labels[index_tensor])
            targets = query_labels[index_tensor, None].expand(-1, queries).reshape(-1)
            loss = F.cross_entropy(logits.reshape(-1, num_classes), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += float(loss.item()) * len(index)
            seen += len(index)
        record = {"epoch": epoch, "train_ce": total / max(seen, 1)}
        history.append(record)
        if log_every and (epoch + 1) % log_every == 0:
            print(f"    predictor epoch {epoch + 1}/{epochs} train_ce={record['train_ce']:.4f}", flush=True)
    model.eval()
    info = {
        "history": history,
        "independent_queries": bool(independent_queries),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "budget": budget,
        "final_train_ce": history[-1]["train_ce"] if history else float("nan"),
        "first_train_ce": history[0]["train_ce"] if history else float("nan"),
        "episodes": int(episodes),
        "subset_dropout": {
            "full_probability": full_probability,
            "empty_probability": empty_probability,
            "long_probability": long_probability,
        },
    }
    return model, info
