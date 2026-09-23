"""R112 ablation: R-MUR with a forced budget (new module; R110/frozen code untouched).

The frozen ``select_ranked_response_mur`` accepts a candidate only when its
predicted marginal utility is above zero (``active = best > 0.0``).  That rule is
harm-safe when extra context can hurt, but it also conflates *ranking quality*
with *stopping*: a reranker that ranks correctly but scores conservatively selects
too few contexts.  R110 measured exactly that failure mode on CIFAR-100 and DTD.

This module provides the identical two-stage greedy loop with the acceptance rule
removed, so exactly ``budget`` candidates are added.  The scoring path (cached
screen at the current state, top-q shortlist, compact response summaries, ranked
response model) is unchanged; only ``active`` is dropped.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for _path in (HERE,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from frozen_helpers import score_state  # noqa: E402


def select_ranked_response_mur_forced(
    screen_model,
    response_model,
    batch,
    ops,
    *,
    budget: int,
    q: int,
    device: str,
    return_trace: bool = False,
):
    """Forced-budget variant of the frozen R-MUR greedy selection.

    Identical to ``select_ranked_response_mur`` except that every step adds the
    arg-max candidate of the shortlist, so the selected set always has size
    ``budget``.
    """

    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    candidate_count = batch.candidate_count
    q = max(1, min(int(q), candidate_count))
    trace = {key: [] for key in ("choice", "topq", "response_scores")}
    expert_rows = 0
    rows = np.arange(batch.episodes)
    for _ in range(budget):
        before = ops.rows
        screen = score_state(screen_model, batch, selected, device=device, pack_fn=ops.pack)
        topq = np.argsort(-screen, axis=1, kind="stable")[:, :q]
        qmask = np.zeros_like(selected, dtype=bool)
        qmask[rows[:, None], topq] = True
        responses = ops.response_summary(batch, selected, candidate_mask=qmask)

        def response_fn(batch_, selected_):
            return responses

        response_scores = score_state(
            response_model,
            batch,
            selected,
            device=device,
            pack_fn=ops.pack,
            response_fn=response_fn,
        )
        expert_rows += ops.rows - before
        masked = np.where(qmask, response_scores, -np.inf)
        choice = np.argmax(masked, axis=1)
        # Forced budget: no "> 0" acceptance test.
        selected[rows, choice] = True
        if return_trace:
            trace["choice"].append(choice.astype(np.int64))
            trace["topq"].append(topq.astype(np.int64))
            trace["response_scores"].append(response_scores.astype(np.float32))
    if return_trace:
        for key in trace:
            trace[key] = np.stack(trace[key], axis=0)
        return selected, trace, expert_rows
    return selected, expert_rows
