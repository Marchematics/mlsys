#!/usr/bin/env python3
"""Regime map: how pool redundancy explains which policy family wins.

For every regime the paper evaluates, measure a single comparable quantity ---
the redundancy of the candidate pool, defined as the mean pairwise similarity
between candidates relative to their similarity to the anchor --- and place it
next to the performance of each policy family (relevance, diversity, pooling
everything, learned utility routing, oracle). The point is to show that the
*same* heuristic changes sign across regimes, and that pool geometry predicts
which side of zero it lands on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KBS = ROOT / "KBS_MUR"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KBS / "src"))

from pami_traffic import build_candidate_pool  # noqa: E402


def traffic_redundancy(dataset: str, targets: np.ndarray, candidate_count: int = 16) -> dict:
    data = np.load(ROOT / "data/staeformer" / dataset / "data.npz")["data"].astype(np.float32)[..., 0]
    index = np.load(ROOT / "data/staeformer" / dataset / "index.npz")
    train_stop = int(index["train"][:, 1].max() + 1)
    values = data[:train_stop]
    correlation = np.corrcoef(values.T)
    within, to_anchor = [], []
    for target in targets:
        target = int(target)
        pool = build_candidate_pool(
            correlation[target], target_sensor=target, candidate_count=candidate_count, seed=20260920
        )
        block = np.abs(correlation[np.ix_(pool, pool)])
        off_diagonal = block[~np.eye(len(pool), dtype=bool)]
        within.append(float(off_diagonal.mean()))
        to_anchor.append(float(np.abs(correlation[target, pool]).mean()))
    return {
        "dataset": dataset,
        "targets": int(len(targets)),
        "mean_pairwise_candidate_similarity": float(np.mean(within)),
        "mean_candidate_to_anchor_similarity": float(np.mean(to_anchor)),
    }


def demo_redundancy(benchmark: str, candidate_count: int = 16, nearest: int = 8, draws: int = 20) -> dict:
    features_dir = ROOT / "PAMI_MUR/data/vision_features" / benchmark
    embeddings = np.load(features_dir / "train_emb.npy").astype(np.float32)
    labels = np.load(features_dir / "train_labels.npy")
    norms = np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    embeddings = embeddings / norms
    rng = np.random.default_rng(0)
    within, to_anchor = [], []
    classes = np.unique(labels)
    for _ in range(draws):
        anchor_class = int(rng.choice(classes))
        in_class = np.flatnonzero(labels == anchor_class)
        anchor = embeddings[rng.choice(in_class)]
        similarity = embeddings @ anchor
        order = np.argsort(-similarity)
        nearest_pool = order[:nearest]
        other = np.flatnonzero(labels != anchor_class)
        distractors = rng.choice(other, size=candidate_count - nearest, replace=False)
        pool = np.concatenate([nearest_pool, distractors])
        block = np.abs(embeddings[pool] @ embeddings[pool].T)
        off = block[~np.eye(len(pool), dtype=bool)]
        within.append(float(off.mean()))
        to_anchor.append(float(np.abs(embeddings[pool] @ anchor).mean()))
    return {
        "benchmark": benchmark,
        "draws": draws,
        "mean_pairwise_candidate_similarity": float(np.mean(within)),
        "mean_candidate_to_anchor_similarity": float(np.mean(to_anchor)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"traffic": {}, "demos": {}}
    strong_targets = np.unique(np.linspace(0, 206, 16, dtype=np.int64))
    ridge_targets = np.unique(np.linspace(0, 206, 32, dtype=np.int64))
    report["traffic"]["METRLA_strong_16targets"] = traffic_redundancy("METRLA", strong_targets)
    report["traffic"]["METRLA_ridge_32targets"] = traffic_redundancy("METRLA", ridge_targets)
    for benchmark in ("cifar10", "cifar100", "svhn", "eurosat", "dtd"):
        directory = ROOT / "PAMI_MUR/data/vision_features" / benchmark
        if (directory / "train_emb.npy").exists():
            report["demos"][benchmark] = demo_redundancy(benchmark)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "regime_map.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
