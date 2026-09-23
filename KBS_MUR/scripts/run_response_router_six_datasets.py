#!/usr/bin/env python3
"""Run response-aware MUR on the six standard traffic benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mur.traffic import fit_subset_expert, make_batch  # noqa: E402
from run_traffic_ranked_response_router_sweep import evaluate_seed  # noqa: E402


DATASETS = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]


def build_candidate_pool(values, *, train_stop, target_sensor, candidate_count, seed):
    corr = np.corrcoef(values[:train_stop].T)[target_sensor]
    corr[target_sensor] = -np.inf
    order = np.argsort(-corr, kind="stable")
    high = order[: max(candidate_count // 2, 1)]
    middle_start = len(high)
    middle_stop = min(middle_start + max(candidate_count // 3, 1), len(order))
    middle = order[middle_start:middle_stop]
    remaining = np.setdiff1d(order, np.concatenate([high, middle]), assume_unique=False)
    rng = np.random.default_rng(seed)
    distractors = rng.choice(remaining, size=candidate_count - len(high) - len(middle), replace=False)
    return np.concatenate([high, middle, distractors]).astype(np.int64)


def build_protocol(dataset_dir: Path, *, candidate_count, history_length, horizon, seed):
    data = np.load(dataset_dir / "data.npz")["data"].astype(np.float32)
    index = np.load(dataset_dir / "index.npz")
    values = data[..., 0]
    train_starts = index["train"][:, 1]
    validation_starts = index["val"][:, 1]
    test_starts = index["test"][:, 1]
    train_stop = int(train_starts.max() + 1)
    mean = values[:train_stop].mean(axis=0)
    sd = values[:train_stop].std(axis=0)
    values = (values - mean) / np.maximum(sd, 1e-6)
    target_sensor = 0
    candidate_sensors = build_candidate_pool(
        values,
        train_stop=train_stop,
        target_sensor=target_sensor,
        candidate_count=candidate_count,
        seed=20260920,
    )
    corr = np.corrcoef(values[:train_stop].T)[target_sensor]
    rng = np.random.default_rng(seed + 2_000)

    def sample(starts, count):
        return rng.choice(starts, size=min(count, len(starts)), replace=False)

    batches = []
    for starts, count in (
        (train_starts, 4000),
        (validation_starts, 1000),
        (test_starts, 2000),
    ):
        batches.append(
            make_batch(
                values,
                sample(starts, count),
                target_sensor=target_sensor,
                candidate_sensors=candidate_sensors,
                history_length=history_length,
                horizon=horizon,
            )
        )
    cfg = {
        "budget": 4,
        "expert_repeats": 3,
        "ridge_penalty": 10.0,
        "candidate_sensor_correlations": corr[candidate_sensors].astype(np.float32).tolist(),
        "candidate_sensors": candidate_sensors.tolist(),
        "target_sensor_index": target_sensor,
    }
    return cfg, tuple(batches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data" / "staeformer")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--q-values", default="2,4,8,16")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    q_values = [int(item) for item in args.q_values.split(",") if item.strip()]
    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)
    records = []
    for dataset in datasets:
        dataset_dir = args.data_root / dataset
        for seed in seeds:
            started = time.time()
            cfg, batches = build_protocol(dataset_dir, candidate_count=16, history_length=12, horizon=12, seed=seed)
            expert = fit_subset_expert(
                batches[0],
                seed=seed + 3_000,
                repeats=int(cfg["expert_repeats"]),
                ridge_penalty=float(cfg["ridge_penalty"]),
            )
            result, gains = evaluate_seed(
                seed,
                cfg,
                batches,
                expert,
                q_values=q_values,
                epochs=args.epochs,
                beta=args.beta,
                hard_negative_q=0,
                device=args.device,
            )
            result["dataset"] = dataset
            result["seed"] = seed
            result["wall_seconds"] = time.time() - started
            records.append(result)
            seed_dir = args.out_root / dataset / f"seed{seed}"
            seed_dir.mkdir(parents=True)
            (seed_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            np.savez_compressed(seed_dir / "episode_gains.npz", **{name: np.asarray(values) for name, values in gains.items()})
            print(json.dumps({"dataset": dataset, "seed": seed, "q4": result["metrics"].get("ranked_response_q4"), "full": result["metrics"].get("ranked_response_full")}, indent=2))

    metric_names = ["mean_prediction_gain", "negative_transfer_rate", "mean_utility_recovery", "mean_selected_contexts"]
    policy_names = list(records[0]["metrics"].keys())
    rows = ["dataset,seed,policy," + ",".join(metric_names)]
    for record in records:
        for policy in policy_names:
            metrics = record["metrics"][policy]
            rows.append(
                f"{record['dataset']},{record['seed']},{policy},"
                + ",".join(f"{metrics[name]:.8f}" for name in metric_names)
            )
    (args.out_root / "per_seed_metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    aggregate = {}
    for dataset in datasets:
        subset = [r for r in records if r["dataset"] == dataset]
        dataset_agg = {}
        for policy in policy_names:
            dataset_agg[policy] = {
                name: {
                    "mean": float(np.mean([r["metrics"][policy][name] for r in subset])),
                    "std": float(np.std([r["metrics"][policy][name] for r in subset], ddof=1)),
                }
                for name in metric_names
            }
        g_os = dataset_agg["oracle_static"]["mean_prediction_gain"]["mean"]
        g_og = dataset_agg["oracle_greedy"]["mean_prediction_gain"]["mean"]
        g_static = dataset_agg["static_utility"]["mean_prediction_gain"]["mean"]
        g_rmur = dataset_agg["ranked_response_q4"]["mean_prediction_gain"]["mean"]
        dataset_agg["quantities"] = {
            "oracle_static_gain": g_os,
            "oracle_greedy_gain": g_og,
            "state_conditioning_headroom": (g_og - g_os) / g_og if g_og > 0 else float("nan"),
            "R_oracle": g_rmur / g_og if g_og > 0 else float("nan"),
            "R_deploy": (g_rmur - g_static) / (g_og - g_static) if g_og > g_static else float("nan"),
        }
        aggregate[dataset] = dataset_agg
    (args.out_root / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")

    summary_rows = [
        "dataset,oracle_static,oracle_greedy,H_state,static_utility,cached_mur,ranked_rmur_q4,R_oracle,R_deploy"
    ]
    for dataset in datasets:
        q = aggregate[dataset]["quantities"]
        p = aggregate[dataset]
        summary_rows.append(
            f"{dataset},{q['oracle_static_gain']:.5f},{q['oracle_greedy_gain']:.5f},"
            f"{q['state_conditioning_headroom']:.4f},"
            f"{p['static_utility']['mean_prediction_gain']['mean']:.5f},"
            f"{p['cached_mur']['mean_prediction_gain']['mean']:.5f},"
            f"{p['ranked_response_q4']['mean_prediction_gain']['mean']:.5f},"
            f"{q['R_oracle']:.4f},{q['R_deploy']:.4f}"
        )
    (args.out_root / "summary.csv").write_text("\n".join(summary_rows) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
