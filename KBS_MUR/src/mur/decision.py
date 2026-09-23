"""Decision objects and sequential routing without task-identity assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


Array = np.ndarray
ScoreFunction = Callable[[tuple[int, ...], Array], Array]


def observed_marginal_utility(
    loss_before: Array | float,
    loss_after: Array | float,
    *,
    cost: Array | float = 1.0,
    cost_weight: float = 0.0,
) -> Array:
    """Return the episode-level before/after loss difference minus context cost."""

    if cost_weight < 0.0:
        raise ValueError("cost_weight must be nonnegative")
    before = np.asarray(loss_before, dtype=float)
    after = np.asarray(loss_after, dtype=float)
    context_cost = np.asarray(cost, dtype=float)
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise ValueError("losses must be finite")
    if not np.all(np.isfinite(context_cost)) or np.any(context_cost < 0.0):
        raise ValueError("cost must be finite and nonnegative")
    return before - after - cost_weight * context_cost


def squared_loss_conditional_utility(
    prediction_before: Array | float,
    prediction_after: Array | float,
    conditional_mean: Array | float,
    *,
    cost: Array | float = 1.0,
    cost_weight: float = 0.0,
) -> Array:
    """Evaluate ``2*d*(mu-f_A)-d^2-lambda*c`` for squared loss."""

    before = np.asarray(prediction_before, dtype=float)
    after = np.asarray(prediction_after, dtype=float)
    mean = np.asarray(conditional_mean, dtype=float)
    delta = after - before
    return 2.0 * delta * (mean - before) - np.square(delta) - cost_weight * np.asarray(
        cost, dtype=float
    )


@dataclass(frozen=True)
class RoutingStep:
    """One auditable state transition in sequential routing."""

    selected_before: tuple[int, ...]
    remaining_indices: tuple[int, ...]
    predicted_utilities: tuple[float, ...]
    chosen_index: int | None
    stop_reason: str | None


@dataclass(frozen=True)
class RoutingTrace:
    """Final selected set and the complete sequence of routing decisions."""

    selected_indices: tuple[int, ...]
    steps: tuple[RoutingStep, ...]


def route_sequentially(
    candidate_count: int,
    score_fn: ScoreFunction,
    *,
    budget: int,
    threshold: float = 0.0,
) -> RoutingTrace:
    """Greedily add the best predicted marginal utility while it exceeds threshold.

    Scores are recomputed after every selection, which is the semantic
    difference between MUR and a static utility ranking. Ties are broken by the
    lowest original candidate index for deterministic audits.
    """

    if candidate_count < 0 or budget < 0:
        raise ValueError("candidate_count and budget must be nonnegative")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    selected: list[int] = []
    steps: list[RoutingStep] = []
    while len(selected) < min(budget, candidate_count):
        remaining = np.asarray(
            [index for index in range(candidate_count) if index not in selected], dtype=int
        )
        scores = np.asarray(score_fn(tuple(selected), remaining), dtype=float)
        if scores.shape != remaining.shape or not np.all(np.isfinite(scores)):
            raise ValueError("score_fn must return one finite score per remaining candidate")
        best_position = int(np.argmax(scores))
        best_index = int(remaining[best_position])
        best_score = float(scores[best_position])
        if best_score <= threshold:
            steps.append(
                RoutingStep(
                    tuple(selected),
                    tuple(int(i) for i in remaining),
                    tuple(float(v) for v in scores),
                    None,
                    "no_score_above_threshold",
                )
            )
            break
        steps.append(
            RoutingStep(
                tuple(selected),
                tuple(int(i) for i in remaining),
                tuple(float(v) for v in scores),
                best_index,
                None,
            )
        )
        selected.append(best_index)
    if not steps and budget == 0:
        steps.append(RoutingStep((), tuple(range(candidate_count)), (), None, "zero_budget"))
    return RoutingTrace(tuple(selected), tuple(steps))


def one_step_regret(true_utilities: Sequence[float], predicted_utilities: Sequence[float]) -> float:
    """Return true best marginal minus the marginal chosen by predicted score."""

    truth = np.asarray(true_utilities, dtype=float)
    prediction = np.asarray(predicted_utilities, dtype=float)
    if truth.ndim != 1 or truth.size == 0 or prediction.shape != truth.shape:
        raise ValueError("utility vectors must be nonempty and have equal shape")
    chosen = int(np.argmax(prediction))
    return float(np.max(truth) - truth[chosen])


def certified_positive(predicted_utility: float, epsilon: float) -> bool:
    """Whether uniform error epsilon certifies a strictly positive true marginal."""

    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")
    return predicted_utility > epsilon


def certified_no_positive_remaining(max_predicted_utility: float, epsilon: float) -> bool:
    """Whether uniform error epsilon certifies every remaining marginal is nonpositive."""

    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")
    return max_predicted_utility <= -epsilon

