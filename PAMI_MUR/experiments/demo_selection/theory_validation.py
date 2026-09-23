#!/usr/bin/env python3
"""R110 theory validation: the first-order / Hessian sandwich for softmax CE.

For a fixed predictor and a query image with label ``y`` let

    p_A = softmax(logits(f_A)),  d = logits(f_{A u {j}}) - logits(f_A),
    m   = CE(f_A, y) - CE(f_{A u {j}}, y)          (true marginal, nats)
    first = (e_y - p_A) . d                         (first-order term)

Because the softmax cross-entropy Hessian with respect to the logits satisfies
``0 <= H <= 0.5 I``, the exact Taylor remainder gives

    first - 0.25 * ||d||^2  <=  m  <=  first.

This script evaluates both sides on frozen predictors and states, per query
image and averaged over the query batch (where ``||d||^2`` is averaged over
queries, which keeps the inequality valid by Jensen).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAMI = HERE.parents[1]
for _path in (HERE, PAMI / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from demo_protocol import BENCHMARK_CONFIGS, load_protocol  # noqa: E402
from incontext_predictor import (  # noqa: E402
    DemoExpertOps,
    cross_entropy_from_logits,
    load_checkpoint,
    softmax_np,
)


def _random_mask(rng: np.random.Generator, episodes: int, candidates: int, budget: int) -> np.ndarray:
    mask = np.zeros((episodes, candidates), dtype=bool)
    sizes = rng.integers(0, budget, size=episodes)
    for row, size in enumerate(sizes):
        if size:
            mask[row, rng.choice(candidates, size=int(size), replace=False)] = True
    return mask


def collect_cell(
    ops: DemoExpertOps,
    batch,
    *,
    states: int,
    budget: int,
    seed: int,
    store_rows: bool = True,
) -> dict:
    """Per-query and per-(state, candidate) theory samples for one predictor."""

    rng = np.random.default_rng(seed)
    per_query_m, per_query_first, per_query_d2 = [], [], []
    aggregate_m, aggregate_first, aggregate_d2_mean, aggregate_d2_meanlogit = [], [], [], []
    episodes = batch.episodes
    queries = batch.query_batch_size
    labels = np.asarray(batch.query_labels, dtype=np.int64)
    one_hot = np.zeros((episodes, queries, ops.num_classes), dtype=np.float64)
    one_hot[np.arange(episodes)[:, None], np.arange(queries)[None, :], labels[:, None]] = 1.0
    for _ in range(states):
        mask = _random_mask(rng, episodes, batch.candidate_count, budget)
        base = np.asarray(ops.predict(batch, mask), dtype=np.float64)
        base_ce = cross_entropy_from_logits(base, labels)
        p_base = softmax_np(base)
        rows, columns = np.nonzero(~mask)
        for take, logits in ops.pair_predict_chunks(batch, mask, rows, columns):
            row = rows[take]
            column = columns[take]
            after = np.asarray(logits, dtype=np.float64)
            after_ce = cross_entropy_from_logits(after, labels[row])
            delta = after - base[row]
            first_query = np.sum((one_hot[row] - p_base[row]) * delta, axis=-1)
            per_query_m.append(
                (
                    _per_query_ce(base[row], one_hot[row]) - _per_query_ce(after, one_hot[row])
                ).reshape(-1)
            )
            per_query_first.append(np.asarray(first_query).reshape(-1))
            per_query_d2.append(np.sum(delta ** 2, axis=-1).reshape(-1))
            aggregate_m.append(base_ce[row] - after_ce)
            aggregate_first.append(first_query.mean(axis=1))
            aggregate_d2_mean.append((delta ** 2).sum(axis=-1).mean(axis=1))
            aggregate_d2_meanlogit.append((delta.mean(axis=1) ** 2).sum(axis=-1))
    payload = {
        "n_state_candidate": int(sum(piece.size for piece in aggregate_m)),
        "n_query": int(sum(piece.size for piece in per_query_m)),
        "aggregate": _summarise(
            np.concatenate(aggregate_m),
            np.concatenate(aggregate_first),
            np.concatenate(aggregate_d2_mean),
        ),
        "aggregate_mean_logit_bound": _summarise(
            np.concatenate(aggregate_m),
            np.concatenate(aggregate_first),
            np.concatenate(aggregate_d2_meanlogit),
        ),
        "per_query": _summarise(
            np.concatenate(per_query_m),
            np.concatenate(per_query_first),
            np.concatenate(per_query_d2),
        ),
    }
    if store_rows:
        payload["rows"] = {
            "m": np.concatenate(aggregate_m).astype(np.float64).tolist(),
            "first": np.concatenate(aggregate_first).astype(np.float64).tolist(),
            "d_squared": np.concatenate(aggregate_d2_mean).astype(np.float64).tolist(),
        }
    return payload


def _per_query_ce(logits: np.ndarray, one_hot: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_normaliser = np.log(np.exp(shifted).sum(axis=-1))
    picked = np.sum(shifted * one_hot, axis=-1)
    return log_normaliser - picked


def _summarise(m: np.ndarray, first: np.ndarray, d_squared: np.ndarray) -> dict:
    m = np.asarray(m, dtype=np.float64)
    first = np.asarray(first, dtype=np.float64)
    d_squared = np.asarray(d_squared, dtype=np.float64)
    upper_slack = first - m
    lower_slack = m - (first - 0.25 * d_squared)
    tolerance = 1e-9
    upper_violations = int(np.sum(upper_slack < -tolerance))
    lower_violations = int(np.sum(lower_slack < -tolerance))
    result = {
        "samples": int(m.size),
        "mean_m": float(m.mean()),
        "mean_first": float(first.mean()),
        "mean_d_squared": float(d_squared.mean()),
        "median_d_squared": float(np.median(d_squared)),
        "upper_slack_mean": float(upper_slack.mean()),
        "lower_slack_mean": float(lower_slack.mean()),
        "upper_slack_min": float(upper_slack.min()),
        "lower_slack_min": float(lower_slack.min()),
        "upper_violations": upper_violations,
        "lower_violations": lower_violations,
        "sandwich_violations": upper_violations + lower_violations,
        "violation_rate": float((upper_violations + lower_violations) / max(m.size, 1)),
        "finite": bool(np.all(np.isfinite(m)) and np.all(np.isfinite(first))),
    }
    if m.size > 2 and np.std(m) > 0 and np.std(first) > 0:
        result["pearson_first_m"] = float(np.corrcoef(first, m)[0, 1])
        from scipy.stats import spearmanr

        statistic = spearmanr(first, m).statistic
        result["spearman_first_m"] = float(np.asarray(statistic).reshape(-1)[0])
        slope, intercept = np.polyfit(first, m, 1)
        residual = m - (slope * first + intercept)
        result["r2_first_vs_m"] = float(1.0 - residual.var() / m.var())
        result["r2_first_no_intercept"] = float(
            1.0 - np.mean((m - first) ** 2) / m.var()
        )
        result["frac_m_positive"] = float(np.mean(m > 0))
        result["frac_first_positive"] = float(np.mean(first > 0))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True, help="raw R110 run directory")
    parser.add_argument("--feature-root", type=Path, default=PAMI / "data/vision_features")
    parser.add_argument("--out", type=Path, default=PAMI / "results/derived/R110_response_theory_ce/validation.json")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--benchmarks", default="cifar10,cifar100,svhn,eurosat,dtd")
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--store-rows", action="store_true", default=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    benchmarks = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    artifact: dict = {
        "run_id": "R110_response_theory_ce",
        "source_run": str(args.run_root),
        "definition": {
            "predictor": "frozen in-context classifier (query batch + demonstration tokens)",
            "loss": "query-batch mean softmax cross-entropy (nats)",
            "m": "mean_i CE_i(f_A, y_i) - CE_i(f_{A u {j}}, y_i)",
            "first": "mean_i (e_{y_i} - softmax(logits(f_A))_i) . (logits(f_{A u {j}})_i - logits(f_A)_i)",
            "d_squared": "mean_i ||logits(f_{A u {j}})_i - logits(f_A)_i||^2",
            "bound": "first - 0.25 * d_squared <= m <= first (softmax-CE Hessian 0 <= H <= 0.5 I)",
            "aggregate_rows": "one row per (state, candidate); the aggregate bound uses mean_i ||d_i||^2 (Jensen); the mean-logit variant uses ||mean_i d_i||^2",
        },
        "config": {
            "seeds": seeds,
            "benchmarks": benchmarks,
            "states_per_episode_batch": args.states,
            "budget": 4,
            "candidate_count": 16,
            "tolerance": 1e-9,
        },
        "per_benchmark": {},
        "per_cell": {},
    }
    total_state_candidate = 0
    total_query = 0
    for name in benchmarks:
        protocol = load_protocol(args.feature_root, name)
        budget = BENCHMARK_CONFIGS[name].budget
        cells = []
        for seed in seeds:
            checkpoint = args.run_root / name / f"seed{seed}" / "predictor.pt"
            if not checkpoint.exists():
                print(f"missing {checkpoint}, skipping", flush=True)
                continue
            model, _ = load_checkpoint(checkpoint, device=args.device)
            ops = DemoExpertOps(model, device=args.device, chunk_pairs=2048)
            batch = protocol.test_batch(seed)
            payload = collect_cell(
                ops, batch, states=args.states, budget=budget, seed=seed, store_rows=args.store_rows
            )
            artifact["per_cell"][f"{name}_seed{seed}"] = {
                key: value for key, value in payload.items() if key != "rows"
            }
            cells.append(payload)
            total_state_candidate += payload["n_state_candidate"]
            total_query += payload["n_query"]
            print(
                f"[{name} seed{seed}] rows={payload['n_state_candidate']} "
                f"violations={payload['aggregate']['sandwich_violations']} "
                f"spearman={payload['aggregate'].get('spearman_first_m'):.4f}",
                flush=True,
            )
            del model, ops
        merged_rows = {
            key: np.concatenate([np.asarray(cell["rows"][key]) for cell in cells]).tolist()
            for key in ("m", "first", "d_squared")
        }
        artifact["per_benchmark"][name] = {
            "seeds": seeds,
            "aggregate": _summarise(
                np.asarray(merged_rows["m"]),
                np.asarray(merged_rows["first"]),
                np.asarray(merged_rows["d_squared"]),
            ),
            "rows": merged_rows,
        }
    pooled_m = np.concatenate(
        [np.asarray(artifact["per_benchmark"][name]["rows"]["m"]) for name in artifact["per_benchmark"]]
    )
    pooled_first = np.concatenate(
        [np.asarray(artifact["per_benchmark"][name]["rows"]["first"]) for name in artifact["per_benchmark"]]
    )
    pooled_d2 = np.concatenate(
        [np.asarray(artifact["per_benchmark"][name]["rows"]["d_squared"]) for name in artifact["per_benchmark"]]
    )
    artifact["summary"] = {
        "benchmarks": list(artifact["per_benchmark"].keys()),
        "total_state_candidate_samples": int(total_state_candidate),
        "total_query_level_samples": int(total_query),
        "pooled_aggregate": _summarise(pooled_m, pooled_first, pooled_d2),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": artifact["summary"]}, indent=2)[:2000], flush=True)


if __name__ == "__main__":
    main()
