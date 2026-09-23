import numpy as np

from mur.traffic import fit_subset_expert, make_batch, observed_candidate_marginals


def test_traffic_windows_use_history_before_target() -> None:
    values = np.arange(80, dtype=np.float32).reshape(40, 2)
    batch = make_batch(
        values,
        np.asarray([10, 20]),
        target_sensor=0,
        candidate_sensors=np.asarray([1]),
        history_length=3,
        horizon=2,
    )
    np.testing.assert_array_equal(batch.anchor_history[0], values[7:10, 0])
    np.testing.assert_array_equal(batch.candidate_history[0, 0], values[7:10, 1])
    np.testing.assert_array_equal(batch.target_future[0], values[10:12, 0])


def test_subset_expert_produces_finite_candidate_marginals() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(size=(120, 4)).astype(np.float32)
    batch = make_batch(
        values,
        np.arange(12, 90),
        target_sensor=0,
        candidate_sensors=np.asarray([1, 2]),
        history_length=4,
        horizon=3,
    )
    expert = fit_subset_expert(batch, seed=4, repeats=3, ridge_penalty=1.0)
    selected = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    marginal = observed_candidate_marginals(batch, selected, expert)
    assert marginal.shape == selected.shape
    assert np.all(np.isfinite(marginal))
