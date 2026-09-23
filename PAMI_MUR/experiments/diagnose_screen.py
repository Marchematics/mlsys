#!/usr/bin/env python3
"""Where does response-aware routing lose with a strong backbone?

For one protocol cell this measures, at every greedy state:
  * the recall of the cached shortlist (does it contain the oracle-best
    candidate?),
  * the utility recovery of the shortlist alone (top-q by cached score, no
    reranking) versus the reranked decision,
  * the correlation between the predicted and true marginals.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KBS = ROOT / "KBS_MUR"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KBS / "src"))
sys.path.insert(0, str(KBS / "scripts"))

from mur.synthetic_experiment import score_state  # noqa: E402

from fast_router import train_ranked_response_model_fast, train_utility_model_fast  # noqa: E402
from neural_expert import SubsetExpertOps, finetune_expert, load_checkpoint  # noqa: E402
from pami_traffic import build_candidate_pool, load_traffic_data, make_neural_batch  # noqa: E402
from run_strong_backbone import build_full_state_data, sample_pair_dataset_fast  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/staeformer")
    parser.add_argument("--dataset", default="METRLA")
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--q-values", default="2,4,8,16")
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--calibration-episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=2000)
    parser.add_argument("--router-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    data = load_traffic_data(args.data_root / args.dataset)
    candidates = build_candidate_pool(
        data.correlation[args.target], target_sensor=args.target, candidate_count=16, seed=20260920
    )
    model, _ = load_checkpoint(args.expert_root / f"seed{args.seed}" / "expert.pt", device=args.device)
    model = copy.deepcopy(model)
    rng = np.random.default_rng(args.seed + 2_000)
    batches = []
    for starts, count in (
        (data.train_times, args.train_episodes),
        (data.validation_times, args.calibration_episodes),
        (data.test_times, args.test_episodes),
    ):
        times = rng.choice(starts, size=min(count, len(starts)), replace=False)
        batches.append(make_neural_batch(data, times, target_sensor=args.target, candidate_sensors=candidates))
    batches = tuple(batches)
    finetune_expert(
        model, batches[0], epochs=args.finetune_epochs, batch_size=256, learning_rate=3e-4,
        budget=args.budget, seed=args.seed + 7_777, device=args.device, validation_batch=batches[1],
    )
    ops = SubsetExpertOps(model, device=args.device)
    budget = args.budget
    static_pairs = sample_pair_dataset_fast(
        batches[0], ops, max_budget=budget, states_per_episode=4, seed=args.seed + 4_000, static_only=True
    )
    cached_pairs = sample_pair_dataset_fast(
        batches[0], ops, max_budget=budget, states_per_episode=4, seed=args.seed + 5_000, static_only=False
    )
    static_model = train_utility_model_fast(
        static_pairs, max_budget=budget, seed=args.seed + 6_000, device=args.device, epochs=args.router_epochs
    )
    cached_model = train_utility_model_fast(
        cached_pairs, max_budget=budget, seed=args.seed + 7_000, device=args.device, epochs=args.router_epochs
    )
    full_data = build_full_state_data(
        batches[0], ops, max_budget=budget, states_per_episode=4, seed=args.seed + 9_000
    )
    ranked_model = train_ranked_response_model_fast(
        full_data, max_budget=budget, seed=args.seed + 10_000, device=args.device,
        epochs=args.router_epochs, batch_size=128, beta=1.0,
    )

    test = batches[2]
    q_values = [int(item) for item in args.q_values.split(",") if item.strip()]
    rows = np.arange(test.episodes)
    report = {
        "dataset": args.dataset,
        "target": args.target,
        "seed": args.seed,
        "episodes": test.episodes,
        "budget": budget,
        "steps": [],
    }
    selected = np.zeros((test.episodes, test.candidate_count), dtype=bool)
    for step in range(budget):
        screen = score_state(cached_model, test, selected, device=args.device, pack_fn=ops.pack)
        true = ops.candidate_marginals(test, selected)
        eligible = ~selected
        masked_screen = np.where(eligible, screen, -np.inf)
        masked_true = np.where(eligible, true, -np.inf)
        oracle_choice = np.argmax(masked_true, axis=1)
        oracle_value = masked_true[rows, oracle_choice]
        step_record = {"step": step, "recall": {}, "screen_value_ratio": {}, "chosen_value_ratio": {}}
        order = np.argsort(-masked_screen, axis=1, kind="stable")
        for q in q_values:
            shortlist = order[:, :q]
            hit = np.any(shortlist == oracle_choice[:, None], axis=1)
            step_record["recall"][str(q)] = float(np.mean(hit))
            screen_choice = shortlist[:, 0]
            step_record["screen_value_ratio"][str(q)] = float(
                np.mean(masked_true[rows, screen_choice] / np.maximum(oracle_value, 1e-9))
            )
            qmask = np.zeros_like(selected)
            qmask[rows[:, None], shortlist] = True
            responses = ops.response_summary(test, selected, candidate_mask=qmask)

            def response_fn(batch_, selected_, responses=responses):
                return responses

            ranked = score_state(
                ranked_model, test, selected, device=args.device, pack_fn=ops.pack, response_fn=response_fn
            )
            ranked_choice = np.argmax(np.where(qmask, ranked, -np.inf), axis=1)
            step_record["chosen_value_ratio"][str(q)] = float(
                np.mean(masked_true[rows, ranked_choice] / np.maximum(oracle_value, 1e-9))
            )
        # cached-model accuracy on true marginals at this state
        finite = np.isfinite(masked_true) & np.isfinite(masked_screen)
        if finite.sum() > 1:
            a = masked_screen[finite]
            b = masked_true[finite]
            step_record["screen_true_correlation"] = float(np.corrcoef(a, b)[0, 1])
        best = np.argmax(masked_screen, axis=1)
        selected[rows, best] = True
        report["steps"].append(step_record)
        print(json.dumps(step_record), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "screen_diagnosis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
