#!/usr/bin/env python3
"""R123: compare the bilinear head against the frozen scalar head on identical data.

For each benchmark/seed we rebuild the cell's train and test batches, sample
states, build the full-state training data once, train both response heads with
the frozen recipe and identical seeds, and then evaluate on *test* states:

* Spearman rank correlation between the head's score and the true marginal
  ``m(j|A)`` over candidates of the same state (the ranking that selection
  actually uses);
* out-of-sample R^2 of an affine calibration fitted on train states;
* top-1 agreement with the true best candidate and the regret of the head's
  argmax.

This makes the mechanism visible: if the bilinear head estimates the interaction
term better, its ranking correlation and calibration must improve.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PAMI = HERE.parents[1]
ROOT = PAMI.parent
KBS = ROOT / "KBS_MUR"
for _path in (HERE, PAMI / "experiments", KBS / "src", KBS / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mur.synthetic_experiment import score_state  # noqa: E402
from run_traffic_ranked_response_router_sweep import train_ranked_response_model  # noqa: E402

from bilinear_response import BilinearDemoExpertOps, train_bilinear_response_model  # noqa: E402
from demo_protocol import BENCHMARK_CONFIGS, load_protocol  # noqa: E402
from frozen_helpers import build_full_state_data  # noqa: E402
from incontext_predictor import DemoExpertOps  # noqa: E402
from prototype_predictor import load_prototype_checkpoint  # noqa: E402
from redundant_pool import build_redundant_batch  # noqa: E402


def sample_states(rng, episodes: int, candidates: int, budget: int, states: int) -> np.ndarray:
    masks = np.zeros((states, episodes, candidates), dtype=bool)
    for index in range(states):
        size = int(rng.integers(0, budget + 1))
        for row in range(episodes):
            if size:
                masks[index, row, rng.choice(candidates, size=min(size, candidates), replace=False)] = True
    return masks


def head_scores(model, ops, batch, states: np.ndarray) -> np.ndarray:
    """Score every (state, candidate) of a test batch with one head."""

    output = np.full(states.shape, -np.inf, dtype=np.float64)
    for index in range(states.shape[0]):
        response = ops.response_summary(batch, states[index])
        scores = score_state(
            model,
            batch,
            states[index],
            device=str(ops.device),
            pack_fn=ops.pack,
            response_fn=lambda batch_, selected_, response=response: response,
        )
        output[index] = scores
    return output


def evaluate_head(
    scores: np.ndarray,
    targets: np.ndarray,
    states: np.ndarray,
    curvature: np.ndarray | None = None,
    mean_vector_norm: np.ndarray | None = None,
) -> dict:
    spearman, top1, regrets = [], [], []
    corr_curvature, corr_norm = [], []
    for index in range(states.shape[0]):
        eligible = ~states[index]
        rows = np.arange(states.shape[0] if False else targets.shape[0])
        for row in rows:
            mask = eligible[row]
            if mask.sum() < 2:
                continue
            predicted = scores[index, row, mask]
            truth = targets[index, row, mask]
            finite = np.isfinite(predicted) & np.isfinite(truth)
            if finite.sum() < 2:
                continue
            statistic = spearmanr(predicted[finite], truth[finite]).statistic
            if np.isfinite(statistic):
                spearman.append(float(statistic))
            if curvature is not None:
                value = spearmanr(predicted[finite], curvature[index, row, mask][finite]).statistic
                if np.isfinite(value):
                    corr_curvature.append(float(value))
            if mean_vector_norm is not None:
                value = spearmanr(predicted[finite], mean_vector_norm[index, row, mask][finite]).statistic
                if np.isfinite(value):
                    corr_norm.append(float(value))
            best_true = float(np.max(truth[finite]))
            best_pred = float(truth[finite][int(np.argmax(predicted[finite]))])
            regrets.append(best_true - best_pred)
            if np.argmax(predicted[finite]) == np.argmax(truth[finite]):
                top1.append(1.0)
            else:
                top1.append(0.0)
    return {
        "spearman_mean": float(np.mean(spearman)) if spearman else float("nan"),
        "spearman_with_curvature": float(np.mean(corr_curvature)) if corr_curvature else float("nan"),
        "spearman_with_mean_vector_norm": float(np.mean(corr_norm)) if corr_norm else float("nan"),
        "top1_agreement": float(np.mean(top1)) if top1 else float("nan"),
        "mean_regret": float(np.mean(regrets)) if regrets else float("nan"),
        "samples": int(len(spearman)),
    }


def calibrate(train_x, train_y, test_x, test_y) -> float:
    train_x = np.asarray(train_x, dtype=np.float64).reshape(-1)
    test_x = np.asarray(test_x, dtype=np.float64).reshape(-1)
    train_y = np.asarray(train_y, dtype=np.float64).reshape(-1)
    test_y = np.asarray(test_y, dtype=np.float64).reshape(-1)
    design = np.stack([train_x, np.ones_like(train_x)], axis=1)
    weights = np.linalg.lstsq(design, train_y, rcond=None)[0]
    prediction = np.stack([test_x, np.ones_like(test_x)], axis=1) @ weights
    residual = float(np.mean((test_y - prediction) ** 2))
    variance = float(np.var(test_y))
    return float(1.0 - residual / variance) if variance > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmarks", default="cifar10,cifar100,svhn,eurosat,dtd")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--router-epochs", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--feature-root", type=Path, default=PAMI / "data/vision_features")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    artifact = {"source_run": str(args.run_root), "cells": {}}
    for name in [item.strip() for item in args.benchmarks.split(",") if item.strip()]:
        protocol = load_protocol(args.feature_root, name)
        config = BENCHMARK_CONFIGS[name]
        budget = config.budget
        entries = []
        for seed in seeds:
            checkpoint = args.run_root / name / f"seed{seed}" / "predictor.pt"
            if not checkpoint.exists():
                print(f"missing {checkpoint}", flush=True)
                continue
            train_batch, _ = build_redundant_batch(
                protocol.features, protocol.mean, protocol.components,
                query_source=protocol.splits.train_classes, pool_source=protocol.splits.pool,
                distractor_pool=protocol.splits.distractor_pool, query_split="train",
                episodes_per_class=config.train_episodes_per_class, seed=seed + 1_000, config=config,
            )
            test_batch, _ = build_redundant_batch(
                protocol.features, protocol.mean, protocol.components,
                query_source=protocol.splits.test, pool_source=protocol.splits.pool,
                distractor_pool=protocol.splits.distractor_pool, query_split="test",
                episodes_per_class=config.test_episodes_per_class, seed=seed + 2_000, config=config,
            )
            model, _ = load_prototype_checkpoint(checkpoint, device=args.device)
            ops = DemoExpertOps(model, device=args.device, chunk_episodes=16, chunk_pairs=1024)
            bilinear_ops = BilinearDemoExpertOps(model, device=args.device, chunk_episodes=16, chunk_pairs=1024)
            frozen_data = build_full_state_data(
                train_batch, ops, max_budget=budget, states_per_episode=args.states, seed=seed + 9_000
            )
            bilinear_data = build_full_state_data(
                train_batch, bilinear_ops, max_budget=budget, states_per_episode=args.states, seed=seed + 9_000
            )
            frozen_model = train_ranked_response_model(
                frozen_data, max_budget=budget, seed=seed + 10_000, device=args.device,
                epochs=args.router_epochs, beta=args.beta,
            )
            bilinear_model = train_bilinear_response_model(
                bilinear_data, max_budget=budget, seed=seed + 10_000, device=args.device,
                prediction_dim=ops.num_classes, epochs=args.router_epochs, beta=args.beta,
            )
            rng = np.random.default_rng(seed + 21)
            train_states = sample_states(rng, train_batch.episodes, train_batch.candidate_count, budget, 2)
            test_states = sample_states(rng, test_batch.episodes, test_batch.candidate_count, budget, 2)
            entry = {}
            for label, head, head_ops, batch, states in (
                ("frozen", frozen_model, ops, test_batch, test_states),
                ("bilinear", bilinear_model, bilinear_ops, test_batch, test_states),
            ):
                targets = np.stack(
                    [head_ops.candidate_marginals(batch, state) for state in states]
                )
                scores = head_scores(head, head_ops, batch, states)
                # curvature / mean-vector norm come from the predictor, not the head
                packed = np.stack([bilinear_ops.response_summary(batch, state) for state in states])
                offset = 7 + ops.num_classes
                entry[label] = evaluate_head(
                    scores,
                    targets,
                    states,
                    curvature=packed[..., offset],
                    mean_vector_norm=packed[..., offset + 1],
                )
                # affine calibration fitted on train states, evaluated on test states
                train_scores = head_scores(
                    head, head_ops, train_batch, train_states
                )
                train_targets = np.stack(
                    [head_ops.candidate_marginals(train_batch, state) for state in train_states]
                )
                train_scores = train_scores.reshape(-1)
                train_targets = train_targets.reshape(-1)
                valid = np.isfinite(train_scores) & np.isfinite(train_targets)
                test_scores = scores.reshape(-1)
                test_targets = targets.reshape(-1)
                test_valid = np.isfinite(test_scores) & np.isfinite(test_targets)
                entry[label]["r2_calibrated"] = calibrate(
                    train_scores[valid], train_targets[valid], test_scores[test_valid], test_targets[test_valid]
                )
            entries.append(entry)
            print(
                json.dumps(
                    {
                        "benchmark": name,
                        "seed": seed,
                        "frozen_spearman": round(entry["frozen"]["spearman_mean"], 4),
                        "bilinear_spearman": round(entry["bilinear"]["spearman_mean"], 4),
                        "frozen_r2": round(entry["frozen"]["r2_calibrated"], 4),
                        "bilinear_r2": round(entry["bilinear"]["r2_calibrated"], 4),
                        "frozen_top1": round(entry["frozen"]["top1_agreement"], 4),
                        "bilinear_top1": round(entry["bilinear"]["top1_agreement"], 4),
                        "frozen_corr_curv": round(entry["frozen"]["spearman_with_curvature"], 4),
                        "bilinear_corr_curv": round(entry["bilinear"]["spearman_with_curvature"], 4),
                    }
                ),
                flush=True,
            )
            del model, ops, bilinear_ops
        if entries:
            artifact["cells"][name] = {
                f"{label}_{key}": float(np.mean([entry[label][key] for entry in entries]))
                for label in ("frozen", "bilinear")
                for key in (
                    "spearman_mean",
                    "top1_agreement",
                    "mean_regret",
                    "r2_calibrated",
                    "spearman_with_curvature",
                    "spearman_with_mean_vector_norm",
                )
            }
            artifact["cells"][name]["per_seed"] = entries
    cells = artifact["cells"]
    artifact["summary"] = {
        "mean_frozen_spearman": float(np.mean([cell["frozen_spearman_mean"] for cell in cells.values()])) if cells else float("nan"),
        "mean_bilinear_spearman": float(np.mean([cell["bilinear_spearman_mean"] for cell in cells.values()])) if cells else float("nan"),
        "mean_frozen_r2": float(np.mean([cell["frozen_r2_calibrated"] for cell in cells.values()])) if cells else float("nan"),
        "mean_bilinear_r2": float(np.mean([cell["bilinear_r2_calibrated"] for cell in cells.values()])) if cells else float("nan"),
        "benchmarks_bilinear_spearman_better": int(sum(cell["bilinear_spearman_mean"] > cell["frozen_spearman_mean"] for cell in cells.values())),
        "benchmarks_bilinear_r2_better": int(sum(cell["bilinear_r2_calibrated"] > cell["frozen_r2_calibrated"] for cell in cells.values())),
        "mean_frozen_corr_curvature": float(np.mean([cell["frozen_spearman_with_curvature"] for cell in cells.values()])) if cells else float("nan"),
        "mean_bilinear_corr_curvature": float(np.mean([cell["bilinear_spearman_with_curvature"] for cell in cells.values()])) if cells else float("nan"),
        "benchmarks": len(cells),
    }
    artifact["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
