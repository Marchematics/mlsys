"""R115: query-comparative prototype predictor (new module; R110/R112 untouched).

R112 diagnosed that the in-context transformer's utility is label-driven: replacing
candidate features by noise costs only ~18 % of the first-member marginal, and
equalising labels flips every marginal negative.  That is a property of the
predictor family, not of demonstration selection.

This module implements a predictor whose utility can only come from the
demonstration *images relative to the query*:

* a shared learned linear metric ``z(x) = normalise(W x)`` maps the query and the
  demonstrations into a comparison space (no bias, so no label-only path);
* per class, the prototype is the **attention-weighted mean of the selected
  demonstrations of that class**, with weights ``softmax_m(beta * cos(z_q, z_m))``
  and a learned sharpness ``beta``;
* the logit of a class is ``tau * cos(z_q, prototype_c)`` with a learned
  temperature ``tau``; classes with no selected demonstration receive the fixed
  orthogonal reference score 0 (so with an empty context every logit is 0 and the
  prediction is uniform -- there is no learned classification head anywhere);
* a class can therefore only be predicted well if demonstrations *of that class*
  have been selected **and** their images are close to the query.

The forward signature matches ``InContextClassifier`` exactly, so the whole
R110/R112 evaluation stack (``DemoExpertOps`` with its batched pair prediction,
candidate marginals, 7-dimensional response summary and packing) is reused
unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PAMI = HERE.parents[1]
for _path in (HERE, PAMI / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from neural_expert import sample_subset_masks  # noqa: E402  (frozen protocol helper)


class PrototypeClassifier(nn.Module):
    """Attention-weighted prototype classifier with a learned comparison metric."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        *,
        proj_dim: int = 128,
        init_temperature: float = 2.0,
        init_sharpness: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.projection = nn.Linear(embed_dim, proj_dim, bias=False)
        self.log_temperature = nn.Parameter(torch.tensor(float(init_temperature)))
        self.log_sharpness = nn.Parameter(torch.tensor(float(init_sharpness)))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_temperature)

    @property
    def sharpness(self) -> torch.Tensor:
        return F.softplus(self.log_sharpness)

    def forward(
        self,
        query_features: torch.Tensor,
        demo_features: torch.Tensor,
        demo_mask: torch.Tensor,
        demo_labels: torch.Tensor,
    ) -> torch.Tensor:
        batch, queries = query_features.shape[0], query_features.shape[1]
        query = F.normalize(self.projection(query_features), dim=-1)
        demos = F.normalize(self.projection(demo_features), dim=-1)
        similarity = torch.einsum("bnp,bmp->bnm", query, demos)
        attention = torch.exp(self.sharpness * similarity) * demo_mask.unsqueeze(1)
        logits = torch.zeros(
            batch, queries, self.num_classes, device=query_features.device, dtype=query.dtype
        )
        valid = demo_labels[demo_mask]
        if valid.numel() == 0:
            return logits
        for class_index in torch.unique(valid):
            class_index = int(class_index)
            class_mask = (demo_labels == class_index) & demo_mask
            weights = attention * class_mask.unsqueeze(1)
            mass = weights.sum(dim=-1, keepdim=True)
            prototype = torch.einsum("bnm,bmp->bnp", weights, demos) / mass.clamp_min(1e-6)
            prototype = F.normalize(prototype, dim=-1)
            score = self.temperature * torch.einsum("bnp,bnp->bn", prototype, query)
            logits[:, :, class_index] = torch.where(
                mass.squeeze(-1) > 0, score, torch.zeros_like(score)
            )
        return logits


def train_prototype_predictor(
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
    proj_dim: int = 128,
    full_probability: float = 0.05,
    empty_probability: float = 0.05,
    long_probability: float = 0.15,
    grad_clip: float = 1.0,
    log_every: int = 0,
) -> tuple[PrototypeClassifier, dict]:
    """Subset-dropout training on train episodes only (same recipe as R110/R112)."""

    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    embed_dim = int(batch.query_features.shape[-1])
    model = PrototypeClassifier(embed_dim, num_classes, proj_dim=proj_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
            logits = model(
                query[index_tensor],
                demos[index_tensor],
                torch.from_numpy(masks).to(device),
                demo_labels[index_tensor],
            )
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
            print(f"    prototype epoch {epoch + 1}/{epochs} train_ce={record['train_ce']:.4f}", flush=True)
    model.eval()
    info = {
        "history": history,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "budget": budget,
        "proj_dim": proj_dim,
        "final_train_ce": history[-1]["train_ce"] if history else float("nan"),
        "first_train_ce": history[0]["train_ce"] if history else float("nan"),
        "episodes": int(episodes),
        "predictor": "prototype_classifier",
        "temperature": float(model.temperature.detach().cpu()),
        "sharpness": float(model.sharpness.detach().cpu()),
    }
    return model, info


def save_prototype_checkpoint(path: Path, model: PrototypeClassifier, *, config: dict, history: list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "predictor": "prototype_classifier",
            "state_dict": model.state_dict(),
            "config": dict(config),
            "history": list(history),
        },
        path,
    )


def load_prototype_checkpoint(path: Path, *, device: str) -> tuple[PrototypeClassifier, dict]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = payload["config"]
    model = PrototypeClassifier(
        int(config["embed_dim"]), int(config["num_classes"]), **config.get("model_kwargs", {})
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, config
