"""Analyse the test-time adaptation sweep (R141).

The theory says the headroom a router can realise is the part of the anchor's
residual that stays predictable. Per-target fine-tuning is precisely the
operation that absorbs that residual, so sweeping the number of adaptation
epochs gives a controlled one-parameter dose-response for the paper's central
prediction: as the anchor adapts, the structural opportunity persists, the
identifiable share collapses, and at some point selecting stops beating pooling.

Usage
-----
python experiments/analyze_adaptation_sweep.py \
    --run-root results/raw/R141_adapt_sweep_20260923_1500 \
    --out results/derived/R141_adaptation_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Policies that are trained on the target's own training episodes. ``pool_all``
# and the oracle levels are the references the trained routers must beat.
TRAINED = ["ranked_response_q4", "ranked_response_q8", "cached_mur", "static_utility", "relevance", "mmr"]
REFERENCES = ["pool_all", "random", "oracle_static", "oracle_greedy"]
PRIMARY = "ranked_response_q4"


def bootstrap_ci(values: np.ndarray, *, draws: int = 10000, seed: int = 20260923):
    """Paired target-level bootstrap CI for the mean."""

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, values.size, size=(draws, values.size))
    means = values[picks].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "n": int(values.size),
    }


def read_level(level_dir: Path) -> dict:
    """Read one adaptation level: per-cell gains plus the finetune record."""

    cells = []
    for cell in sorted(level_dir.glob("target*_seed*")):
        record = json.loads((cell / "result.json").read_text())
        metrics = record.get("metrics") or {}
        finetune = record.get("finetune") or {}
        row = {
            "target": int(record["target_sensor"]),
            "seed": int(record["seed"]),
            "best_validation_mse": finetune.get("best_validation_mse"),
            "final_train_mse": finetune.get("final_train_mse"),
            "expert_rows": record.get("expert_rows"),
        }
        for policy, value in (record.get("metrics") or {}).items():
            if isinstance(value, dict):
                gain = value.get("mean_prediction_gain")
                if isinstance(gain, (int, float)):
                    row[policy] = float(gain)
                    row[f"{policy}__negative_transfer"] = float(value.get("negative_transfer_rate", float("nan")))
            elif isinstance(value, (int, float)):
                row[policy] = float(value)
        cells.append(row)
    return {"cells": cells, "path": str(level_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()

    levels = {}
    for level_dir in sorted(args.run_root.glob("ft*")):
        if not level_dir.is_dir():
            continue
        epochs = int(level_dir.name[2:])
        levels[epochs] = read_level(level_dir)
    if not levels:
        raise SystemExit(f"no ft* level directories under {args.run_root}")

    # discover the policy names actually present
    sample = next(iter(levels.values()))["cells"][0]
    skip = {"target", "seed", "best_validation_mse", "final_train_mse", "expert_rows"}
    policies = [k for k in sample if k not in skip and not k.endswith("__negative_transfer")]
    print("policies found:", policies)

    summary = {
        "run_root": str(args.run_root),
        "levels": sorted(levels),
        "policies": policies,
        "by_level": {},
    }

    # Cells present in EVERY level. Levels can finish at different times, and
    # comparing a partial level against a complete one confounds adaptation with
    # the composition of the target sample, so every headline number is reported
    # on the matched subset as well as on whatever each level happened to run.
    key = lambda c: (c["target"], c["seed"])
    common = set.intersection(*[{key(c) for c in levels[e]["cells"]} for e in levels])
    summary["matched_cells"] = len(common)
    summary["matched_keys"] = sorted(f"t{t}_s{s}" for t, s in common)

    for epochs in sorted(levels):
        cells = levels[epochs]["cells"]
        entry = {"cells": len(cells), "policy": {}, "validation_mse": None, "policy_matched": {}}
        mse = [c["best_validation_mse"] for c in cells if isinstance(c.get("best_validation_mse"), (int, float))]
        if mse:
            entry["validation_mse"] = float(np.mean(mse))
        for policy in policies:
            values = np.asarray([c[policy] for c in cells if policy in c], dtype=float)
            if values.size == 0:
                continue
            ci = bootstrap_ci(values, draws=args.draws)
            entry["policy"][policy] = ci
            mv = np.asarray([c[policy] for c in cells if policy in c and key(c) in common], dtype=float)
            if mv.size:
                entry["policy_matched"][policy] = bootstrap_ci(mv, draws=args.draws)
        # paired deltas against the strongest reference
        for target in ("full", "matched"):
            tag = "_vs_pool_all" if target == "full" else "_vs_pool_all_matched"
            pool_key = "policy" if target == "full" else "policy_matched"
            if "pool_all" not in entry[pool_key]:
                continue
            for policy in TRAINED:
                if policy not in entry[pool_key]:
                    continue
                deltas = np.asarray(
                    [c[policy] - c["pool_all"] for c in cells
                     if policy in c and "pool_all" in c and (target == "full" or key(c) in common)],
                    dtype=float,
                )
                entry[pool_key][policy][tag] = bootstrap_ci(deltas, draws=args.draws)
        if "oracle_static" in entry["policy"]:
            for tag, subset in (("", cells), ("_matched", [c for c in cells if key(c) in common])):
                if not subset:
                    continue
                oracle = np.asarray([c["oracle_static"] for c in subset], dtype=float)
                pool = np.asarray([c["pool_all"] for c in subset if "pool_all" in c], dtype=float)
                learned = np.asarray([c.get(PRIMARY, np.nan) for c in subset], dtype=float)
                entry[f"structural_gap_oracle_minus_pool{tag}"] = float(np.nanmean(oracle - pool))
                if np.isfinite(learned).any():
                    entry[f"realized_share{tag}"] = float(
                        np.nanmean(learned / np.where(np.abs(oracle) > 1e-9, oracle, np.nan))
                    )
        summary["by_level"][str(epochs)] = entry

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "adaptation_sweep.json").write_text(json.dumps(summary, indent=1))

    # a flat per-cell table for reuse
    with (args.out / "cells.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["finetune_epochs", "target", "seed", "validation_mse", *policies])
        for epochs in sorted(levels):
            for cell in levels[epochs]["cells"]:
                writer.writerow(
                    [epochs, cell["target"], cell["seed"], cell.get("best_validation_mse"),
                     *[cell.get(p, "") for p in policies]]
                )

    # console report
    print(f"\nmatched cells across all levels: {len(common)}")
    print(f"{'epochs':>7} {'val_mse':>9} {'rmur':>9} {'pool':>9} {'random':>9} "
          f"{'oracle':>9} {'share':>7} {'d_pool(matched)':>26}")
    for epochs in sorted(levels):
        e = summary["by_level"][str(epochs)]
        p = e["policy_matched"] or e["policy"]
        rmur = p.get(PRIMARY, {}).get("mean", float("nan"))
        pool = p.get("pool_all", {}).get("mean", float("nan"))
        rand = p.get("random", {}).get("mean", float("nan"))
        oracle = p.get("oracle_static", {}).get("mean", float("nan"))
        share = e.get("realized_share_matched", e.get("realized_share", float("nan")))
        delta = p.get(PRIMARY, {}).get("_vs_pool_all_matched", {})
        vm = e["validation_mse"]
        print(f"{epochs:>7} {vm if vm is not None else float('nan'):>9.4f} "
              f"{rmur:>9.4f} {pool:>9.4f} {rand:>9.4f} {oracle:>9.4f} {share:>7.3f} "
              f"{delta.get('mean', float('nan')):>+13.4f}  [{delta.get('low', float('nan')):+.4f},{delta.get('high', float('nan')):+.4f}]")
    print(f"\nwrote {args.out / 'adaptation_sweep.json'}")


if __name__ == "__main__":
    main()
