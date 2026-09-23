"""Shared loaders for the multi-target R-MUR runs (R080b).

Every figure that uses these loaders reads the immutable per-target result
files under ``results/raw/R080b_*`` and aggregates them in memory. No derived
number is stored in this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

SOURCES = {
    101: ROOT / "results/raw/R080b_multitarget_rmur_s1_20260920_2340",
    202: ROOT / "results/raw/R080b_multitarget_rmur_s202_diag_20260921",
    303: ROOT / "results/raw/R080b_multitarget_rmur_s303_diag_20260921",
}

DATASETS = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]
LABELS = ["METR-LA", "PEMS-BAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]

POLICIES = [
    ("relevance", "Relevance"),
    ("mmr", "MMR"),
    ("static_utility", "Standalone Utility"),
    ("cached_mur", "Cached-only router"),
    ("ranked_response_q4", "R-MUR"),
    # The table includes the trivial pool-everything policy because it is the
    # strongest competitor; the figure must rank the same set or the two
    # disagree about what R-MUR's average rank is measured against.
    ("pool_all", "Pool all 16"),
]

# Content-based subset-selection baselines, saved by their own run set.
SUBSET_SOURCES = sorted((ROOT / "results/raw").glob("R202_subset_baselines_s3_*"))
SUBSET_POLICIES = [
    ("facility_location", "Facility location"),
    ("facility_location_relevance", "Facility loc. + relevance"),
    ("dpp_greedy", "DPP greedy"),
    ("kcenter", "k-center"),
]
FIXED_SET_SOURCES = sorted((ROOT / "results/raw").glob("R206_fixed_set_baselines_v2_s3_*"))
FIXED_SET_POLICIES = [
    ("forward_val", "Greedy validation set"),
    ("adaptive_k", "Adaptive-$k$ relevance"),
    ("ids_cluster", "Cluster-based selection"),
    ("lookahead_val", "Two-step lookahead set"),
    ("grad_influence", "Gradient influence"),
]
ALL_POLICIES = POLICIES + SUBSET_POLICIES + FIXED_SET_POLICIES

# Policies kept for the target-level contrasts.
CONTRAST_POLICIES = ["ranked_response_q4", "static_utility", "cached_mur", "oracle_greedy"]


def _complete_seeds() -> list[int]:
    """Seeds that finished every target of every dataset.

    The figures must aggregate the same runs as the result tables, so partial
    seeds (still running) are excluded rather than mixed in.
    """

    shared: set[int] | None = None
    for dataset in DATASETS:
        counts: dict[int, int] = {}
        for seed, root in SOURCES.items():
            data_dir = root / dataset
            if not data_dir.exists():
                continue
            counts[seed] = sum(
                1 for target_dir in data_dir.glob("target*_seed*") if (target_dir / "result.json").exists()
            )
        complete = {seed for seed, count in counts.items() if count >= 32}
        shared = complete if shared is None else (shared & complete)
    return sorted(shared or set())


def _iter_results(dataset: str):
    for seed in _complete_seeds():
        data_dir = SOURCES[seed] / dataset
        if not data_dir.exists():
            continue
        for target_dir in sorted(data_dir.glob("target*_seed*")):
            result_path = target_dir / "result.json"
            if not result_path.exists():
                continue
            yield seed, json.loads(result_path.read_text(encoding="utf-8"))


def load_target_level_subset(dataset: str, policy: str) -> np.ndarray:
    """Target-level means for a policy of the content-based or fixed-set runs."""

    root = None
    for candidate in SUBSET_SOURCES + FIXED_SET_SOURCES:
        dataset_dir = candidate / dataset
        if not dataset_dir.exists():
            continue
        probe = sorted(dataset_dir.glob("target*_seed*/result.json"))
        if not probe:
            continue
        # Select the run set that actually stores this policy: the two baseline
        # families live in separate run directories.
        if policy in json.loads(probe[0].read_text(encoding="utf-8")).get("metrics", {}):
            root = dataset_dir
            break
    if root is None:
        return np.empty(0)
    per_target: dict[int, list[float]] = {}
    for target_dir in sorted(root.glob("target*_seed*")):
        path = target_dir / "result.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record["seed"]) not in set(_complete_seeds()):
            continue
        target = int(record["target_sensor"])
        per_target.setdefault(target, []).append(float(record["metrics"][policy]["mean_prediction_gain"]))
    return np.asarray([float(np.mean(per_target[t])) for t in sorted(per_target)], dtype=float)


def load_target_level(dataset: str, policies=None) -> dict[str, np.ndarray]:
    """Target-level means for every requested policy.

    Returns ``{"targets": ..., policy: array}`` where each array is aligned to
    the sorted target indices and each entry averages the seeds that completed
    for that target.
    """

    keys = list(policies) if policies is not None else [key for key, _ in POLICIES]
    per_target: dict[int, dict[str, list[float]]] = {}
    for _seed, result in _iter_results(dataset):
        target = int(result["target_sensor"])
        bucket = per_target.setdefault(target, {key: [] for key in keys})
        for key in keys:
            bucket[key].append(float(result["metrics"][key]["mean_prediction_gain"]))
    targets = sorted(per_target)
    out: dict[str, np.ndarray] = {"targets": np.asarray(targets, dtype=int)}
    for key in keys:
        out[key] = np.asarray([np.mean(per_target[t][key]) for t in targets], dtype=float)
    if "ranked_response_q4" in out and "static_utility" in out:
        out["delta_static"] = out["ranked_response_q4"] - out["static_utility"]
    if "ranked_response_q4" in out and "cached_mur" in out:
        out["delta_cached"] = out["ranked_response_q4"] - out["cached_mur"]
    return out


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(values.size))
