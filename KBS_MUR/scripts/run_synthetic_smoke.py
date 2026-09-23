#!/usr/bin/env python3
"""Run R002 synthetic oracle-headroom and redundancy-contract smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mur.synthetic import (  # noqa: E402
    SyntheticConfig,
    generate_synthetic_batch,
    observed_candidate_marginals,
    squared_error,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--redundancy", type=float, default=0.5)
    parser.add_argument("--harmful", type=float, default=0.25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = SyntheticConfig(
        redundancy_ratio=args.redundancy,
        harmful_fraction=args.harmful,
    )
    batch = generate_synthetic_batch(args.episodes, config, seed=args.seed)
    empty = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    pool_all = np.ones_like(empty)
    selected = empty.copy()
    marginal_history = []
    choice_history = []
    for _ in range(config.budget):
        marginal = observed_candidate_marginals(batch, selected)
        marginal_history.append(marginal.copy())
        choice = np.argmax(marginal, axis=1)
        choice_history.append(choice.copy())
        best = marginal[np.arange(batch.episodes), choice]
        active = best > 0.0
        selected[np.arange(batch.episodes)[active], choice[active]] = True
    base_loss = squared_error(batch, empty)
    pool_loss = squared_error(batch, pool_all)
    oracle_loss = squared_error(batch, selected)
    first = marginal_history[0]
    mixed = (np.max(first, axis=1) > 0.0) & (np.min(first, axis=1) <= 0.0)

    duplicate_ratios = []
    second = marginal_history[1]
    for row in range(batch.episodes):
        first_selected = int(choice_history[0][row])
        group = batch.candidate_group[row, first_selected]
        duplicates = np.flatnonzero(
            (batch.candidate_group[row] == group)
            & (np.arange(batch.candidate_count) != first_selected)
            & ~batch.harmful[row]
        )
        if duplicates.size:
            standalone = first[row, duplicates]
            after = second[row, duplicates]
            valid = standalone > 1e-12
            if np.any(valid):
                duplicate_ratios.extend((after[valid] / standalone[valid]).tolist())

    result = {
        "run_id": "R002",
        "config": {
            "episodes": args.episodes,
            "seed": args.seed,
            **config.__dict__,
        },
        "metrics": {
            "base_loss": float(np.mean(base_loss)),
            "pool_all_loss": float(np.mean(pool_loss)),
            "oracle_greedy_loss": float(np.mean(oracle_loss)),
            "oracle_gain": float(np.mean(base_loss - oracle_loss)),
            "pool_all_gain": float(np.mean(base_loss - pool_loss)),
            "mixed_sign_state_fraction": float(np.mean(mixed)),
            "positive_standalone_fraction": float(np.mean(first > 0.0)),
            "median_duplicate_marginal_ratio_after_first": float(np.median(duplicate_ratios)),
            "mean_selected": float(np.mean(np.sum(selected, axis=1))),
        },
        "provenance": {
            "script": str(Path(__file__).relative_to(ROOT)),
            "script_sha256": _sha256(Path(__file__)),
            "synthetic_module_sha256": _sha256(ROOT / "src" / "mur" / "synthetic.py"),
        },
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
