import numpy as np
import pytest

from mur.decision import (
    certified_no_positive_remaining,
    certified_positive,
    observed_marginal_utility,
    one_step_regret,
    route_sequentially,
    squared_loss_conditional_utility,
)


def test_loss_difference_and_squared_loss_identity_agree_in_expectation() -> None:
    before = 0.2
    after = 0.5
    conditional_mean = 0.8
    direct = squared_loss_conditional_utility(before, after, conditional_mean)
    expected = (before - conditional_mean) ** 2 - (after - conditional_mean) ** 2
    assert direct == pytest.approx(expected)
    assert observed_marginal_utility(0.7, 0.4, cost=2.0, cost_weight=0.1) == pytest.approx(0.1)


def test_sequential_router_recomputes_scores_after_selection() -> None:
    def scores(selected: tuple[int, ...], remaining: np.ndarray) -> np.ndarray:
        table = {
            (): np.asarray([0.8, 0.75, 0.5]),
            (0,): np.asarray([0.05, 0.45]),
        }
        return table[selected]

    trace = route_sequentially(3, scores, budget=2)
    assert trace.selected_indices == (0, 2)
    assert trace.steps[1].selected_before == (0,)


def test_sequential_router_stops_when_no_score_clears_threshold() -> None:
    trace = route_sequentially(
        2, lambda selected, remaining: np.asarray([-0.1, -0.2]), budget=2, threshold=0.0
    )
    assert trace.selected_indices == ()
    assert trace.steps[-1].stop_reason == "no_score_above_threshold"


def test_uniform_error_certificates_and_regret_bound() -> None:
    truth = np.asarray([0.6, 0.4, 0.1])
    prediction = np.asarray([0.42, 0.5, 0.1])
    epsilon = float(np.max(np.abs(truth - prediction)))
    assert one_step_regret(truth, prediction) <= 2.0 * epsilon + 1e-12
    assert certified_positive(0.21, 0.2)
    assert not certified_positive(0.2, 0.2)
    assert certified_no_positive_remaining(-0.2, 0.2)
    assert not certified_no_positive_remaining(0.0, 0.2)

