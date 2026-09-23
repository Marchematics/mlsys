#!/usr/bin/env python3
"""The full baseline matrix in the ridge-expert protocol.

The strong-backbone study showed that selection is unidentifiable when the
fixed predictor is strong, so the protocol where context selection actually
pays is the ridge expert. This runner evaluates every training-free competitor
(DPP, facility location, mutual information, k-center, k-means representatives,
random, relevance, MMR, pool-all) on exactly the R080b cells, reusing the
frozen protocol (candidate pools, ridge expert fit, episode sampling) so the
results can be paired with the stored R-MUR episode gains.
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

from mur.traffic import expert_prediction, fit_subset_expert, make_batch, pack_selected, squared_error  # noqa: E402

from run_response_baselines import (  # noqa: E402
    select_delfit_style,
    select_kcenter,
    select_kmeans_representatives,
    select_mutual_information,
)
from run_strong_backbone import (  # noqa: E402
    candidate_similarity,
    select_dpp,
    select_facility_location,
    select_random,
)
from run_traffic_full_router import select_mmr, select_relevance  # noqa: E402


class RidgeExpertOps:
    """Adapter giving the ridge expert the interface the baselines expect."""

    def __init__(self, expert) -> None:
        self.expert = expert

    def predict(self, batch, selected: np.ndarray) -> np.ndarray:
        return expert_prediction(batch, selected, self.expert)

    def squared_error(self, batch, selected: np.ndarray) -> np.ndarray:
        return squared_error(batch, selected, self.expert)

    def pair_predict(self, batch, selected: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        augmented = np.asarray(selected, dtype=bool).copy()
        augmented[rows, columns] = True
        return self.predict(batch, augmented)[rows]

    def pack(self, batch, selected: np.ndarray, *, max_budget: int):
        return pack_selected(batch, selected, max_budget=max_budget)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/staeformer")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--target-count", type=int, default=32)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--test-episodes", type=int, default=2000)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--calibration-episodes", type=int, default=1000)
    args = parser.parse_args()

    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)

    data = np.load(args.data_root / args.dataset / "data.npz")["data"].astype(np.float32)
    index = np.load(args.data_root / args.dataset / "index.npz")
    values = data[..., 0]
    train_starts = index["train"][:, 1]
    validation_starts = index["val"][:, 1]
    test_starts = index["test"][:, 1]
    train_stop = int(train_starts.max() + 1)
    mean = values[:train_stop].mean(axis=0)
    sd = values[:train_stop].std(axis=0)
    values = ((values - mean) / np.maximum(sd, 1e-6)).astype(np.float32)
    correlation = np.corrcoef(values[:train_stop].T)
    num_nodes = values.shape[1]
    targets = np.unique(np.linspace(0, num_nodes - 1, args.target_count, dtype=np.int64))

    for target in targets:
        target = int(target)
        row = np.asarray(correlation[target], dtype=float).copy()
        row[target] = -np.inf
        order = np.argsort(-row, kind="stable")
        high = order[: max(args.candidate_count // 2, 1)]
        middle = order[len(high) : len(high) + max(args.candidate_count // 3, 1)]
        remaining = np.setdiff1d(order, np.concatenate([high, middle]), assume_unique=False)
        rng_pool = np.random.default_rng(20260920)
        distractors = rng_pool.choice(remaining, size=args.candidate_count - len(high) - len(middle), replace=False)
        candidates = np.concatenate([high, middle, distractors]).astype(np.int64)
        relevance_template = np.asarray(correlation[target], dtype=np.float32)[candidates]

        for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
            started = time.time()
            rng = np.random.default_rng(seed + 2_000)
            batches = []
            for starts, count in (
                (train_starts, args.train_episodes),
                (validation_starts, args.calibration_episodes),
                (test_starts, args.test_episodes),
            ):
                times = rng.choice(starts, size=min(count, len(starts)), replace=False)
                batches.append(
                    make_batch(
                        values, times, target_sensor=target, candidate_sensors=candidates,
                        history_length=12, horizon=12,
                    )
                )
            train, calibration, test = batches
            expert = fit_subset_expert(train, seed=seed + 3_000, repeats=3, ridge_penalty=10.0)
            ops = RidgeExpertOps(expert)
            empty = np.zeros((test.episodes, test.candidate_count), dtype=bool)
            base_loss = ops.squared_error(test, empty)
            relevance = np.broadcast_to(relevance_template[None, :], (test.episodes, test.candidate_count)).copy()
            similarity = candidate_similarity(test)

            selections = {
                "random": select_random(test, args.budget, seed + 99),
                "relevance": select_relevance(relevance, args.budget),
                "mmr": select_mmr(test, relevance, args.budget),
                "dpp": select_dpp(test, relevance, args.budget, similarity=similarity),
                "facility_location": select_facility_location(test, relevance, args.budget, similarity=similarity),
                "mutual_information": select_mutual_information(test, args.budget),
                "kcenter": select_kcenter(test, args.budget),
                "kmeans_representatives": select_kmeans_representatives(test, args.budget),
                "delfit_style": select_delfit_style(
                    test, relevance, args.budget, similarity=similarity, seed=seed,
                ),
                "pool_all": np.ones((test.episodes, test.candidate_count), dtype=bool),
                "base_only": empty,
            }
            gains = {
                name: base_loss - ops.squared_error(test, mask) for name, mask in selections.items()
            }
            cell = args.out_root / f"target{target}_seed{seed}"
            cell.mkdir(parents=True)
            np.savez_compressed(
                cell / "episode_gains.npz",
                **{name: np.asarray(v, dtype=np.float32) for name, v in gains.items()},
            )
            (cell / "result.json").write_text(
                json.dumps(
                    {
                        "dataset": args.dataset,
                        "target_sensor": target,
                        "seed": seed,
                        "metrics": {
                            name: {
                                "mean_prediction_gain": float(np.mean(values_)),
                                "negative_transfer_rate": float(np.mean(values_ < 0.0)),
                            }
                            for name, values_ in gains.items()
                        },
                        "wall_seconds": time.time() - started,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps({"target": target, "seed": seed,
                            **{k: round(float(np.mean(v)), 5) for k, v in gains.items()}}),
                flush=True,
            )


if __name__ == "__main__":
    main()
