"""External validity: does the pilot rule hold outside traffic forecasting? (R157)

The deployment procedure was validated on six traffic deployments (R154-R156).
The demonstration-selection family is a second domain with the same decision
object --- pick a small context set for a frozen predictor --- so the same
question can be asked there at no cost, because those runs record the class of
every evaluation episode.

The pilot/deployment split is by *class* rather than by sensor: half the classes
are the pilot, the other half the deployment. That is a harder split than the
traffic one, because different classes are genuinely different sub-problems
rather than different draws from the same population.

Usage
-----
python experiments/analyze_pilot_external.py --out results/derived/R157_pilot_external
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = ROOT / "results/raw/R110_demo_selection_20260921_0408"
BENCHMARKS = ["cifar10", "cifar100", "svhn", "eurosat", "dtd"]
SEEDS = (101, 202, 303)
ROUTER = "ranked_response_q4"
BASELINE = "pool_all"
# The decision a practitioner actually faces in this family is which context
# rule to use, so several contrasts are scored: some are obvious (the router is
# far below pooling) and some are marginal, and the pilot question is only
# interesting for the marginal ones.
CONTRASTS = {
    "router_minus_pool": ("ranked_response_q4", "pool_all"),
    "relevance_minus_pool": ("relevance", "pool_all"),
    "router_minus_relevance": ("ranked_response_q4", "relevance"),
    "mmr_minus_pool": ("mmr", "pool_all"),
}


def class_differences(benchmark: str, left: str, right: str) -> np.ndarray:
    """Per-class mean of ``left - right``, averaged over seeds."""

    per_class: dict[int, list[float]] = {}
    for seed in SEEDS:
        path = DEMO_ROOT / benchmark / f"seed{seed}" / "episode_gains.npz"
        if not path.exists():
            continue
        data = np.load(path)
        if left not in data or right not in data:
            continue
        delta = data[left] - data[right]
        classes = data["class_index"]
        for klass in np.unique(classes):
            per_class.setdefault(int(klass), []).append(float(delta[classes == klass].mean()))
    return np.asarray(
        [np.mean(v) for _, v in sorted(per_class.items())], dtype=float
    )


def verdict(values: np.ndarray, *, draws: int = 10000, seed: int = 0) -> str:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return "UNRESOLVED"
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    low, high = np.quantile(means, 0.025), np.quantile(means, 0.975)
    if low > 0:
        return "AHEAD"
    if high < 0:
        return "BELOW"
    return "UNRESOLVED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=400)
    args = parser.parse_args()

    summary = {"source": str(DEMO_ROOT), "contrasts": {}}
    rng = np.random.default_rng(20260927)
    for name, (left, right) in CONTRASTS.items():
        summary["contrasts"][name] = _analyse(left, right, rng, args.splits)
    summary["marginal_contrasts"] = [
        name for name, entry in summary["contrasts"].items()
        if abs(entry["pooled"]["mean_difference"]) < 0.05
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pilot_external.json").write_text(json.dumps(summary, indent=1))

    for name, entry in summary["contrasts"].items():
        pooled = entry["pooled"]
        print(f"\n=== {name}  (pooled Δ={pooled['mean_difference']:+.4f}, verdict {pooled['verdict']})")
        print(f"{'benchmark':10s} {'classes':>8s} {'mean Δ':>10s} {'full verdict':>13s} "
              f"{'split agreement':>16s} {'sign agreement':>15s}")
        for benchmark in BENCHMARKS:
            e = entry["benchmarks"].get(benchmark)
            if not e:
                continue
            print(f"{benchmark:10s} {e['classes']:>8d} {e['mean_difference']:>+10.4f} "
                  f"{e['full_verdict']:>13s} {e['verdict_agreement']:>16.3f} {e['sign_agreement']:>15.3f}")
        print(f"benchmark-level verdict agreement: {pooled['benchmark_verdict_agreement']}")
    print(f"\nmarginal contrasts (|Δ| < .05): {summary['marginal_contrasts']}")
    print(f"\nwrote {args.out / 'pilot_external.json'}")


def _analyse(left: str, right: str, rng, splits: int) -> dict:
    """Run the class-split pilot test for one contrast."""

    entry = {"left": left, "right": right, "benchmarks": {}}
    pooled_values = []
    agree = total = 0
    for benchmark in BENCHMARKS:
        values = class_differences(benchmark, left, right)
        if values.size < 4:
            continue
        full = verdict(values, seed=1)
        item = {
            "classes": int(values.size),
            "mean_difference": float(values.mean()),
            "full_verdict": full,
        }
        # half/half class splits, repeated
        records = []
        half = values.size // 2
        for _ in range(splits):
            order = rng.permutation(values.size)
            first, second = order[:half], order[half:]
            p = verdict(values[first], draws=1500, seed=int(rng.integers(1 << 30)))
            d = verdict(values[second], draws=1500, seed=int(rng.integers(1 << 30)))
            records.append((p, d))
        pilot_verdicts = [r[0] for r in records]
        deploy_verdicts = [r[1] for r in records]
        item["pilot_verdict_mode"] = max(set(pilot_verdicts), key=pilot_verdicts.count)
        item["deployment_verdict_mode"] = max(set(deploy_verdicts), key=deploy_verdicts.count)
        item["verdict_agreement"] = float(np.mean([p == d for p, d in records]))
        sign_hits = 0
        for _ in range(splits):
            order = rng.permutation(values.size)
            sign_hits += np.sign(values[order[:half]].mean()) == np.sign(values[order[half:]].mean())
        item["sign_agreement"] = float(sign_hits / splits)
        entry["benchmarks"][benchmark] = item
        pooled_values.append(values)
        agree += int(item["pilot_verdict_mode"] == item["deployment_verdict_mode"])
        total += 1
    all_values = np.concatenate(pooled_values) if pooled_values else np.zeros(0)
    entry["pooled"] = {
        "classes": int(all_values.size),
        "mean_difference": float(all_values.mean()) if all_values.size else float("nan"),
        "verdict": verdict(all_values, seed=2) if all_values.size else "UNRESOLVED",
        "benchmark_verdict_agreement": f"{agree}/{total}",
    }
    return entry


if __name__ == "__main__":
    main()
