"""Timing probe: cost of one one-shot baseline cell (delete after use)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mur.traffic import fit_subset_expert, make_batch, squared_error  # noqa: E402
from run_response_router_multitarget import (  # noqa: E402
    build_candidate_pool,
    prepare_dataset,
    target_set,
)
from run_traffic_ranked_response_router_sweep import (  # noqa: E402
    build_full_state_data,
    select_oneshot_response,
    train_ranked_response_model,
)


def mark(label: str, started: float) -> float:
    now = time.time()
    print(f"{label}: {now - started:.1f}s", flush=True)
    return now


t0 = time.time()
prepared = prepare_dataset(ROOT.parent / "data" / "staeformer" / "METRLA")
t0 = mark("prepare_dataset", t0)

target = int(target_set(prepared["num_nodes"], 8)[0])
candidates = build_candidate_pool(
    prepared["corr"][target], target_sensor=target, candidate_count=16, seed=20260920
)
rng = np.random.default_rng(2101)
batches = []
for starts, count in (
    (prepared["train_starts"], 4000),
    (prepared["validation_starts"], 1000),
    (prepared["test_starts"], 2000),
):
    times = rng.choice(starts, size=min(count, len(starts)), replace=False)
    batches.append(
        make_batch(
            prepared["values"],
            times,
            target_sensor=target,
            candidate_sensors=candidates,
            history_length=12,
            horizon=12,
        )
    )
train, _val, test = batches
t0 = mark("batches", t0)

expert = fit_subset_expert(train, seed=3101, repeats=3, ridge_penalty=10.0)
t0 = mark("expert", t0)

data = build_full_state_data(train, expert, max_budget=4, states_per_episode=4, seed=9000 + 101)
t0 = mark("build_full_state_data", t0)

model = train_ranked_response_model(data, max_budget=4, seed=10000 + 101, device="cpu", epochs=20, beta=1.0)
t0 = mark("train_ranked_response_model (20 epochs)", t0)

selected = select_oneshot_response(model, model, test, expert, budget=4, device="cpu")
t0 = mark("select_oneshot_response", t0)

base = squared_error(test, np.zeros((test.episodes, test.candidate_count), dtype=bool), expert)
loss = squared_error(test, selected, expert)
print(f"oneshot gain on this cell: {float(np.mean(base - loss)):+.4f}", flush=True)
