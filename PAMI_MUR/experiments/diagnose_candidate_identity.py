#!/usr/bin/env python3
"""Is candidate identity informative about utility?

The frozen protocol embeds a candidate by its history only. The subset-capable
backbone additionally uses a learned node embedding, so this check asks whether
per-candidate utility is stable across episodes: if it is, a router that cannot
see node identity is handicapped by construction.

Reports:
  * variance decomposition of the standalone marginal across candidates and
    episodes,
  * stability of the per-candidate mean between two disjoint episode samples,
  * the test utility of a "candidate prior" policy that ranks candidates by
    their mean marginal estimated on other episodes.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KBS = ROOT / "KBS_MUR"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KBS / "src"))
sys.path.insert(0, str(KBS / "scripts"))

from mur.traffic import fit_subset_expert, squared_error  # noqa: E402

from neural_expert import SubsetExpertOps, finetune_expert, load_checkpoint  # noqa: E402
from pami_traffic import build_candidate_pool, load_traffic_data, make_neural_batch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/staeformer")
    parser.add_argument("--dataset", default="METRLA")
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--targets", default="0,13,27,41")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    data = load_traffic_data(args.data_root / args.dataset)
    model, _ = load_checkpoint(args.expert_root / f"seed{args.seed}" / "expert.pt", device=args.device)
    model = copy.deepcopy(model)
    report = {"dataset": args.dataset, "seed": args.seed, "budget": args.budget, "targets": {}}
    for target in [int(item) for item in args.targets.split(",") if item.strip()]:
        candidates = build_candidate_pool(
            data.correlation[target], target_sensor=target, candidate_count=16, seed=20260920
        )
        rng = np.random.default_rng(args.seed + 2_000)
        train_times = rng.choice(data.train_times, size=4000, replace=False)
        calibration_times = rng.choice(data.validation_times, size=1000, replace=False)
        test_times = rng.choice(data.test_times, size=args.episodes, replace=False)
        train_batch = make_neural_batch(data, train_times, target_sensor=target, candidate_sensors=candidates)
        calibration_batch = make_neural_batch(
            data, calibration_times, target_sensor=target, candidate_sensors=candidates
        )
        test_batch = make_neural_batch(data, test_times, target_sensor=target, candidate_sensors=candidates)
        adapted = copy.deepcopy(model)
        finetune_expert(
            adapted, train_batch, epochs=args.finetune_epochs, batch_size=256, learning_rate=3e-4,
            budget=args.budget, seed=args.seed + 7_777, device=args.device, validation_batch=calibration_batch,
        )
        ops = SubsetExpertOps(adapted, device=args.device)
        empty_test = np.zeros((test_batch.episodes, 16), dtype=bool)
        empty_calibration = np.zeros((calibration_batch.episodes, 16), dtype=bool)
        empty_train = np.zeros((train_batch.episodes, 16), dtype=bool)
        test_marginal = ops.candidate_marginals(test_batch, empty_test)
        calibration_marginal = ops.candidate_marginals(calibration_batch, empty_calibration)
        train_marginal = ops.candidate_marginals(train_batch, empty_train)
        ridge = fit_subset_expert(train_batch, seed=args.seed + 3_000, repeats=3, ridge_penalty=10.0)
        ridge_test_marginal = np.full_like(test_marginal, -np.inf)
        base = squared_error(test_batch, empty_test, ridge)
        for candidate in range(16):
            augmented = empty_test.copy()
            augmented[:, candidate] = True
            ridge_test_marginal[:, candidate] = base - squared_error(test_batch, augmented, ridge)

        between = float(np.var(test_marginal.mean(axis=0)))
        within = float(np.mean(test_marginal.var(axis=0)))
        correlation_neural = float(np.corrcoef(
            train_marginal.mean(axis=0), test_marginal.mean(axis=0)
        )[0, 1])
        correlation_ridge = float(np.corrcoef(
            train_marginal.mean(axis=0), ridge_test_marginal.mean(axis=0)
        )[0, 1])
        order = np.argsort(-train_marginal.mean(axis=0))[: args.budget]
        prior_selected = np.zeros_like(empty_test)
        prior_selected[:, order] = True
        base_loss = ops.squared_error(test_batch, empty_test)
        prior_gain = float(np.mean(base_loss - ops.squared_error(test_batch, prior_selected)))
        oracle_selected = np.zeros_like(empty_test)
        for row in range(len(empty_test)):
            oracle_selected[row, np.argsort(-test_marginal[row])[: args.budget]] = True
        oracle_gain = float(np.mean(base_loss - ops.squared_error(test_batch, oracle_selected)))
        record = {
            "between_candidate_variance": between,
            "within_episode_variance": within,
            "between_share": between / max(between + within, 1e-12),
            "candidate_mean_stability_neural": correlation_neural,
            "candidate_mean_stability_ridge": correlation_ridge,
            "prior_policy_gain": prior_gain,
            "oracle_static_gain": oracle_gain,
            "prior_over_oracle": prior_gain / oracle_gain if oracle_gain > 0 else float("nan"),
        }
        report["targets"][str(target)] = record
        print(json.dumps({"target": target, **record}), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "candidate_identity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
