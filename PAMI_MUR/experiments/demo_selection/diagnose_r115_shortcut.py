#!/usr/bin/env python3
"""R115 prediction (iii): is the label shortcut closed?

For (state, candidate) samples of the frozen R115 predictor we fit two ridge
probes that predict the realised marginal utility ``m(j|A)``:

* **label probe** -- features are a function of the *label multiset of A* and of
  the candidate's label only: how many selected demonstrations carry the query's
  label, how many carry the candidate's label, how many carry any other label,
  whether the candidate's label equals the query's, the candidate's label count
  inside A, plus the selected-set size.  No image information whatsoever.
* **response probe** -- the seven-dimensional frozen-predictor response summary
  (mean base logit, mean logit after adding the candidate, mean/std/mean-abs/RMS/
  max-abs of the logit change), i.e. what R-MUR's reranking stage actually sees.

Probes are fitted on train-split episodes and evaluated out of sample on
test-split episodes (no episode overlap).  A permutation control repeats the
label probe with shuffled targets.  Prediction (iii): the label probe's
out-of-sample R^2 must be substantially lower than the response probe's.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAMI = HERE.parents[1]
ROOT = PAMI.parent
KBS = ROOT / "KBS_MUR"
for _path in (HERE, PAMI / "experiments", KBS / "src", KBS / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from demo_protocol import BENCHMARK_CONFIGS, load_protocol  # noqa: E402
from incontext_predictor import DemoExpertOps  # noqa: E402
from prototype_predictor import load_prototype_checkpoint  # noqa: E402
from redundant_pool import build_redundant_batch  # noqa: E402


def label_features(batch, selected: np.ndarray, rows: np.ndarray, candidate_labels: np.ndarray) -> np.ndarray:
    """Label-multiset features only (no image information).

    One row per (episode, candidate) pair, so every feature is computed on the
    pair's own episode.
    """

    selected_rows = selected[rows]
    labels_rows = batch.candidate_labels[rows]
    query_labels = batch.query_labels[rows]
    selected_count = selected_rows.sum(axis=1, keepdims=True).astype(np.float64)
    same_query = (selected_rows & (labels_rows == query_labels[:, None])).sum(axis=1, keepdims=True)
    candidate_is_query = (candidate_labels == query_labels).astype(np.float64)[:, None]
    candidate_in_state = (
        selected_rows & (labels_rows == candidate_labels[:, None])
    ).sum(axis=1, keepdims=True)
    other = selected_count - same_query
    return np.concatenate(
        [
            selected_count,
            same_query.astype(np.float64),
            other.astype(np.float64),
            candidate_is_query,
            candidate_in_state.astype(np.float64),
            np.ones_like(selected_count),  # intercept column
        ],
        axis=1,
    )


def collect_probe_data(ops: DemoExpertOps, batch, *, states: int, budget: int, seed: int):
    rng = np.random.default_rng(seed)
    label_rows, response_rows, target_rows = [], [], []
    for _ in range(states):
        size = rng.integers(0, budget + 1)
        mask = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
        for row in range(batch.episodes):
            if size:
                mask[row, rng.choice(batch.candidate_count, size=int(size), replace=False)] = True
        responses = ops.response_summary(batch, mask)
        marginal = ops.candidate_marginals(batch, mask)
        rows, columns = np.nonzero(~mask)
        candidate_labels = batch.candidate_labels[rows, columns]
        label_rows.append(label_features(batch, mask, rows, candidate_labels))
        response_rows.append(responses[rows, columns])
        target_rows.append(marginal[rows, columns])
    return (
        np.concatenate(label_rows),
        np.concatenate(response_rows),
        np.concatenate(target_rows),
    )


def ridge_r2(train_x, train_y, test_x, test_y, *, alpha: float = 1.0) -> dict:
    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    train_x = (train_x - mean) / scale
    test_x = (test_x - mean) / scale
    target_mean = train_y.mean()
    gram = train_x.T @ train_x + alpha * np.eye(train_x.shape[1])
    weights = np.linalg.solve(gram, train_x.T @ (train_y - target_mean))
    prediction = test_x @ weights + target_mean
    residual = float(np.mean((test_y - prediction) ** 2))
    variance = float(np.var(test_y))
    return {
        "r2": float(1.0 - residual / variance) if variance > 0 else float("nan"),
        "mse": residual,
        "n_train": int(train_x.shape[0]),
        "n_test": int(test_x.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmarks", default="cifar10,cifar100,svhn,eurosat,dtd")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--feature-root", type=Path, default=PAMI / "data/vision_features")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    artifact = {
        "source_run": str(args.run_root),
        "states_per_split": args.states,
        "predictor": "R115 learned-metric attention prototype",
        "cells": {},
    }
    for name in [item.strip() for item in args.benchmarks.split(",") if item.strip()]:
        protocol = load_protocol(args.feature_root, name)
        config = BENCHMARK_CONFIGS[name]
        per_seed = []
        for seed in seeds:
            checkpoint = args.run_root / name / f"seed{seed}" / "predictor.pt"
            if not checkpoint.exists():
                print(f"missing {checkpoint}", flush=True)
                continue
            model, _ = load_prototype_checkpoint(checkpoint, device=args.device)
            ops = DemoExpertOps(model, device=args.device, chunk_episodes=16, chunk_pairs=1024)
            train_batch, _ = build_redundant_batch(
                protocol.features,
                protocol.mean,
                protocol.components,
                query_source=protocol.splits.train_classes,
                pool_source=protocol.splits.pool,
                distractor_pool=protocol.splits.distractor_pool,
                query_split="train",
                episodes_per_class=config.train_episodes_per_class,
                seed=seed + 1_000,
                config=config,
            )
            test_batch, _ = build_redundant_batch(
                protocol.features,
                protocol.mean,
                protocol.components,
                query_source=protocol.splits.test,
                pool_source=protocol.splits.pool,
                distractor_pool=protocol.splits.distractor_pool,
                query_split="test",
                episodes_per_class=config.test_episodes_per_class,
                seed=seed + 2_000,
                config=config,
            )
            train_label, train_response, train_target = collect_probe_data(
                ops, train_batch, states=args.states, budget=config.budget, seed=seed + 11
            )
            test_label, test_response, test_target = collect_probe_data(
                ops, test_batch, states=args.states, budget=config.budget, seed=seed + 12
            )
            rng = np.random.default_rng(seed + 13)
            permutation = rng.permutation(train_target.size)
            entry = {
                "label_probe": ridge_r2(train_label, train_target, test_label, test_target),
                "response_probe": ridge_r2(train_response, train_target, test_response, test_target),
                "label_probe_permuted_control": ridge_r2(
                    train_label, train_target[permutation], test_label, test_target
                ),
            }
            entry["gap"] = entry["response_probe"]["r2"] - entry["label_probe"]["r2"]
            per_seed.append(entry)
            print(
                json.dumps(
                    {
                        "benchmark": name,
                        "seed": seed,
                        "label_r2": round(entry["label_probe"]["r2"], 4),
                        "response_r2": round(entry["response_probe"]["r2"], 4),
                        "gap": round(entry["gap"], 4),
                        "permuted_r2": round(entry["label_probe_permuted_control"]["r2"], 4),
                    }
                ),
                flush=True,
            )
        if per_seed:
            artifact["cells"][name] = {
                "label_r2": float(np.mean([entry["label_probe"]["r2"] for entry in per_seed])),
                "response_r2": float(np.mean([entry["response_probe"]["r2"] for entry in per_seed])),
                "gap": float(np.mean([entry["gap"] for entry in per_seed])),
                "permuted_r2": float(
                    np.mean([entry["label_probe_permuted_control"]["r2"] for entry in per_seed])
                ),
                "per_seed": per_seed,
            }
    cells = artifact["cells"]
    artifact["summary"] = {
        "mean_label_r2": float(np.mean([cell["label_r2"] for cell in cells.values()])) if cells else float("nan"),
        "mean_response_r2": float(np.mean([cell["response_r2"] for cell in cells.values()])) if cells else float("nan"),
        "mean_gap": float(np.mean([cell["gap"] for cell in cells.values()])) if cells else float("nan"),
        "mean_permuted_r2": float(np.mean([cell["permuted_r2"] for cell in cells.values()])) if cells else float("nan"),
        "benchmarks_label_below_response": int(sum(cell["gap"] > 0 for cell in cells.values())),
        "benchmarks": len(cells),
    }
    artifact["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
