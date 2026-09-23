#!/usr/bin/env python3
"""Why does relevance flip sign? Probe across candidate-pool geometries.

The ridge protocol measures relevance ranking (and DPP, MMR, facility location)
as *harmful*: worse than using no auxiliary context. The strong-backbone
protocol and the demonstration family have relevance near-neutral or
near-optimal. This probe holds the protocol fixed and varies only which part of
the correlation spectrum the candidate pool is drawn from:

  correlated      head = most correlated sensors (the frozen protocol)
  mid             head = moderate |correlation|: informative, not redundant
  anticorrelated  head = least correlated sensors

If redundancy with the anchor is what makes proxy ranking harmful, relevance
should improve as the pool moves from ``correlated`` to ``mid``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KBS = ROOT / "KBS_MUR"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KBS / "src"))
sys.path.insert(0, str(KBS / "scripts"))

from mur.traffic import fit_subset_expert, make_batch, squared_error  # noqa: E402

from pami_traffic import build_candidate_pool  # noqa: E402
from run_ridge_baselines import RidgeExpertOps  # noqa: E402
from run_strong_backbone import candidate_similarity, select_dpp  # noqa: E402
from run_traffic_full_router import select_mmr, select_oracle_static, select_relevance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/staeformer")
    parser.add_argument("--dataset", default="METRLA")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--modes", default="correlated,mid,anticorrelated")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--target-count", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--test-episodes", type=int, default=2000)
    parser.add_argument("--train-episodes", type=int, default=4000)
    args = parser.parse_args()

    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)
    data = np.load(args.data_root / args.dataset / "data.npz")["data"].astype(np.float32)[..., 0]
    index = np.load(args.data_root / args.dataset / "index.npz")
    train_starts, test_starts = index["train"][:, 1], index["test"][:, 1]
    train_stop = int(train_starts.max() + 1)
    mean, sd = data[:train_stop].mean(axis=0), data[:train_stop].std(axis=0)
    values = ((data - mean) / np.maximum(sd, 1e-6)).astype(np.float32)
    correlation = np.corrcoef(values[:train_stop].T)
    targets = np.unique(np.linspace(0, values.shape[1] - 1, args.target_count, dtype=np.int64))

    for mode in args.modes.split(","):
        for target in targets:
            target = int(target)
            candidates = build_candidate_pool(
                correlation[target], target_sensor=target,
                candidate_count=args.candidate_count, seed=20260920, mode=mode,
            )
            for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
                started = time.time()
                rng = np.random.default_rng(seed + 2_000)
                train_times = rng.choice(train_starts, size=args.train_episodes, replace=False)
                test_times = rng.choice(test_starts, size=args.test_episodes, replace=False)
                train = make_batch(values, train_times, target_sensor=target, candidate_sensors=candidates,
                                   history_length=12, horizon=12)
                test = make_batch(values, test_times, target_sensor=target, candidate_sensors=candidates,
                                  history_length=12, horizon=12)
                expert = fit_subset_expert(train, seed=seed + 3_000, repeats=3, ridge_penalty=10.0)
                ops = RidgeExpertOps(expert)
                empty = np.zeros((test.episodes, args.candidate_count), dtype=bool)
                base_loss = ops.squared_error(test, empty)
                relevance = np.broadcast_to(
                    correlation[target][candidates][None, :], (test.episodes, args.candidate_count)
                ).copy()
                similarity = candidate_similarity(test)
                selections = {
                    "relevance": select_relevance(relevance, args.budget),
                    "mmr": select_mmr(test, relevance, args.budget),
                    "dpp": select_dpp(test, relevance, args.budget, similarity=similarity),
                    "pool_all": np.ones((test.episodes, args.candidate_count), dtype=bool),
                    "random": np.stack([
                        np.isin(np.arange(args.candidate_count),
                                rng.choice(args.candidate_count, size=args.budget, replace=False))
                        for _ in range(test.episodes)
                    ]),
                    "oracle_static": select_oracle_static(test, expert, args.budget),
                }
                gains = {name: base_loss - ops.squared_error(test, mask) for name, mask in selections.items()}
                pool_block = np.abs(correlation[np.ix_(candidates, candidates)])
                record = {
                    "mode": mode,
                    "target_sensor": target,
                    "seed": seed,
                    "pool_mean_pairwise_similarity": float(
                        pool_block[~np.eye(len(candidates), dtype=bool)].mean()
                    ),
                    "pool_mean_anchor_similarity": float(np.abs(correlation[target][candidates]).mean()),
                    "metrics": {k: {"mean_prediction_gain": float(np.mean(v))} for k, v in gains.items()},
                    "wall_seconds": time.time() - started,
                }
                cell = args.out_root / f"{mode}_target{target}_seed{seed}"
                cell.mkdir(parents=True)
                np.savez_compressed(cell / "episode_gains.npz",
                                    **{k: np.asarray(v, dtype=np.float32) for k, v in gains.items()})
                (cell / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({"mode": mode, "target": target, "seed": seed,
                                  **{k: round(float(np.mean(v)), 5) for k, v in gains.items()}}), flush=True)


if __name__ == "__main__":
    main()
