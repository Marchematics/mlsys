#!/usr/bin/env python3
"""Measure whether observable traffic context features predict utility labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_traffic_full_router import load_protocol  # noqa: E402
from mur.synthetic_experiment import (  # noqa: E402
    predict_pair_dataset,
    sample_pair_dataset,
    train_utility_model,
)
from mur.traffic import fit_subset_expert, observed_candidate_marginals, pack_selected  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--gate-result", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg, (train, _, test), _ = load_protocol(args.data_path, args.gate_result, args.seed)
    expert = fit_subset_expert(
        train,
        seed=args.seed + 3_000,
        repeats=int(cfg["expert_repeats"]),
        ridge_penalty=float(cfg["ridge_penalty"]),
    )
    marginal_fn = lambda batch, selected: observed_candidate_marginals(batch, selected, expert)
    diagnostics = {}
    for name, static_only, offset in (("static", True, 4_000), ("mur", False, 5_000)):
        pairs = sample_pair_dataset(
            train,
            max_budget=int(cfg["budget"]),
            states_per_episode=4,
            seed=args.seed + offset,
            static_only=static_only,
            marginal_fn=marginal_fn,
            pack_fn=pack_selected,
        )
        model = train_utility_model(
            pairs,
            max_budget=int(cfg["budget"]),
            seed=args.seed + offset + 2_000,
            device=args.device,
            epochs=args.epochs,
        )
        test_pairs = sample_pair_dataset(
            test,
            max_budget=int(cfg["budget"]),
            states_per_episode=2,
            seed=args.seed + offset + 4_000,
            static_only=static_only,
            marginal_fn=marginal_fn,
            pack_fn=pack_selected,
        )
        train_prediction = predict_pair_dataset(model, pairs, device=args.device)
        test_prediction = predict_pair_dataset(model, test_pairs, device=args.device)
        diagnostics[name] = {
            "train_mae": float(np.mean(np.abs(train_prediction - pairs.target))),
            "test_mae": float(np.mean(np.abs(test_prediction - test_pairs.target))),
            "test_target_std": float(np.std(test_pairs.target)),
            "test_prediction_std": float(np.std(test_prediction)),
            "test_correlation": float(np.corrcoef(test_prediction, test_pairs.target)[0, 1]),
        }
    result = {"run_id": "R075_traffic_label_diagnostic", "seed": args.seed, "diagnostics": diagnostics}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
