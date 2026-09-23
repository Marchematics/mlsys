#!/usr/bin/env python3
"""Multi-target R-MUR evaluation on one standard traffic dataset."""

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


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def target_set(num_nodes: int, count: int) -> np.ndarray:
    if count >= num_nodes:
        return np.arange(num_nodes, dtype=np.int64)
    return np.unique(np.linspace(0, num_nodes - 1, count, dtype=np.int64))


def build_candidate_pool(corr_row, *, target_sensor, candidate_count, seed):
    corr = np.asarray(corr_row, dtype=float).copy()
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


def prepare_dataset(dataset_dir: Path):
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
    corr = np.corrcoef(values[:train_stop].T)
    return {
        "values": values,
        "train_starts": train_starts,
        "validation_starts": validation_starts,
        "test_starts": test_starts,
        "corr": corr,
        "num_nodes": int(values.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data" / "staeformer")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", default="101")
    parser.add_argument("--target-count", type=int, default=32)
    parser.add_argument("--q-values", default="2,4,8,16")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--calibration-episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collect-observables", action="store_true",
                        help="save the per-episode statistics a label-free confidence "
                             "gate could key on (screen margin, predicted margin, "
                             "predicted top score, response norm)")
    parser.add_argument("--extra-split", default="none", choices=["none", "validation"],
                        help="also score the router and pool-everything on the calibration "
                             "split, for the deployment-pilot study")
    args = parser.parse_args()
    dataset_dir = args.data_root / args.dataset
    prepared = prepare_dataset(dataset_dir)
    targets = target_set(prepared["num_nodes"], args.target_count)
    seeds = parse_ints(args.seeds)
    q_values = parse_ints(args.q_values)
    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)
    records = []
    rng_master = np.random.default_rng(20260920)
    for target_sensor in targets:
        target_sensor = int(target_sensor)
        candidates = build_candidate_pool(
            prepared["corr"][target_sensor],
            target_sensor=target_sensor,
            candidate_count=args.candidate_count,
            seed=20260920,
        )
        corr = prepared["corr"][target_sensor]
        cfg = {
            "budget": args.budget,
            "expert_repeats": 3,
            "ridge_penalty": 10.0,
            "candidate_sensor_correlations": corr[candidates].astype(np.float32).tolist(),
            "candidate_sensors": candidates.tolist(),
            "target_sensor_index": target_sensor,
        }
        for seed in seeds:
            started = time.time()
            rng = np.random.default_rng(seed + 2_000)
            batches = []
            for starts, count in (
                (prepared["train_starts"], args.train_episodes),
                (prepared["validation_starts"], args.calibration_episodes),
                (prepared["test_starts"], args.test_episodes),
            ):
                times = rng.choice(starts, size=min(count, len(starts)), replace=False)
                batches.append(
                    make_batch(
                        prepared["values"],
                        times,
                        target_sensor=target_sensor,
                        candidate_sensors=candidates,
                        history_length=args.history_length,
                        horizon=args.horizon,
                    )
                )
            batches = tuple(batches)
            expert = fit_subset_expert(
                batches[0],
                seed=seed + 3_000,
                repeats=int(cfg["expert_repeats"]),
                ridge_penalty=float(cfg["ridge_penalty"]),
            )
            result, gains, diagnostics = evaluate_seed(
                seed,
                cfg,
                batches,
                expert,
                q_values=q_values,
                epochs=args.epochs,
                beta=args.beta,
                hard_negative_q=0,
                device=args.device,
                save_diagnostics=True,
                extra_split=args.extra_split,
                collect_observables=args.collect_observables,
            )
            result["dataset"] = args.dataset
            result["target_sensor"] = target_sensor
            result["seed"] = seed
            result["wall_seconds"] = time.time() - started
            records.append(result)
            seed_dir = args.out_root / f"target{target_sensor}_seed{seed}"
            seed_dir.mkdir(parents=True)
            (seed_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            np.savez_compressed(seed_dir / "episode_gains.npz", **{name: np.asarray(values) for name, values in gains.items()})
            np.savez_compressed(seed_dir / "diagnostics.npz", **diagnostics)
        print(f"completed target {target_sensor} for {args.dataset}")

    metric_names = ["mean_prediction_gain", "negative_transfer_rate", "mean_utility_recovery", "mean_selected_contexts"]
    policy_names = list(records[0]["metrics"].keys())
    rows = ["dataset,target_sensor,seed,policy," + ",".join(metric_names)]
    for record in records:
        for policy in policy_names:
            metrics = record["metrics"][policy]
            rows.append(
                f"{record['dataset']},{record['target_sensor']},{record['seed']},{policy},"
                + ",".join(f"{metrics[name]:.8f}" for name in metric_names)
            )
    (args.out_root / "per_target_metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Aggregate target-level quantities.
    target_stats = []
    for target_sensor in sorted({r["target_sensor"] for r in records}):
        subset = [r for r in records if r["target_sensor"] == target_sensor]
        q4 = np.asarray([r["metrics"]["ranked_response_q4"]["mean_prediction_gain"] for r in subset], dtype=float)
        static = np.asarray([r["metrics"]["static_utility"]["mean_prediction_gain"] for r in subset], dtype=float)
        cached = np.asarray([r["metrics"]["cached_mur"]["mean_prediction_gain"] for r in subset], dtype=float)
        os_gain = np.asarray([r["metrics"]["oracle_static"]["mean_prediction_gain"] for r in subset], dtype=float)
        og_gain = np.asarray([r["metrics"]["oracle_greedy"]["mean_prediction_gain"] for r in subset], dtype=float)
        h = (og_gain - os_gain) / og_gain
        target_stats.append(
            {
                "target_sensor": target_sensor,
                "runs": int(len(subset)),
                "oracle_static_gain": float(np.mean(os_gain)),
                "oracle_greedy_gain": float(np.mean(og_gain)),
                "H_state": float(np.mean(h)),
                "static_utility_gain": float(np.mean(static)),
                "cached_mur_gain": float(np.mean(cached)),
                "ranked_rmur_q4_gain": float(np.mean(q4)),
                "R_oracle_q4": float(np.mean(q4) / np.mean(og_gain)) if np.mean(og_gain) > 0 else float("nan"),
                "R_deploy_q4": float((np.mean(q4) - np.mean(static)) / (np.mean(og_gain) - np.mean(static)))
                if np.mean(og_gain) > np.mean(static)
                else float("nan"),
            }
        )
    dataset_stats = {
        "dataset": args.dataset,
        "targets": len(target_stats),
        "seeds_per_target": len(seeds),
        "targets_positive_gain": sum(1 for t in target_stats if t["ranked_rmur_q4_gain"] > 0),
        "targets_gain_over_static": sum(1 for t in target_stats if t["ranked_rmur_q4_gain"] > t["static_utility_gain"]),
        "target_fraction_positive_gain": float(np.mean([t["ranked_rmur_q4_gain"] > 0 for t in target_stats])),
        "target_fraction_gain_over_static": float(np.mean([t["ranked_rmur_q4_gain"] > t["static_utility_gain"] for t in target_stats])),
        "H_state_mean": float(np.mean([t["H_state"] for t in target_stats])),
        "H_state_median": float(np.median([t["H_state"] for t in target_stats])),
        "R_oracle_mean": float(np.mean([t["R_oracle_q4"] for t in target_stats])),
        "R_deploy_mean": float(np.mean([t["R_deploy_q4"] for t in target_stats])),
        "target_records": target_stats,
    }
    (args.out_root / "target_statistics.json").write_text(json.dumps(dataset_stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dataset_stats, indent=2))


if __name__ == "__main__":
    main()
