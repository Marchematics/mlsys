"""Temporary timing probe for the fixed-set baseline sweep (deleted after use)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_response_router_multitarget import (  # noqa: E402
    build_candidate_pool,
    prepare_dataset,
    target_set,
)
from mur.traffic import fit_subset_expert, make_batch, squared_error  # noqa: E402


def mark(label: str, started: float) -> float:
    now = time.time()
    print(f"{label}: {now - started:.1f}s", flush=True)
    return now


t0 = time.time()
prep = prepare_dataset(ROOT.parent / "data" / "staeformer" / "METRLA")
t0 = mark("prepare_dataset", t0)

targets = target_set(prep["num_nodes"], 2)
t0 = mark("target_set", t0)

target = int(targets[0])
candidates = build_candidate_pool(
    prep["corr"][target], target_sensor=target, candidate_count=16, seed=20260920
)
t0 = mark("candidate_pool", t0)

rng = np.random.default_rng(2101)
times = rng.choice(prep["train_starts"], size=1000, replace=False)
batch = make_batch(
    prep["values"],
    times,
    target_sensor=target,
    candidate_sensors=candidates,
    history_length=12,
    horizon=12,
)
t0 = mark(f"make_batch (episodes={batch.episodes})", t0)

expert = fit_subset_expert(batch, seed=3101, repeats=3, ridge_penalty=10.0)
t0 = mark("fit_subset_expert", t0)

mask = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
started = time.time()
for index in range(batch.candidate_count):
    trial = mask.copy()
    trial[:, index] = True
    squared_error(batch, trial, expert).mean()
print(f"16 evaluations: {time.time() - started:.2f}s", flush=True)

started = time.time()
squared_error(batch, mask, expert)
print(f"one evaluation: {time.time() - started:.3f}s", flush=True)
