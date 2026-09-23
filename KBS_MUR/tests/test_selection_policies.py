import numpy as np

from mur.synthetic import SyntheticConfig, generate_synthetic_batch, observed_candidate_marginals
from mur.synthetic_experiment import select_mmr, select_oracle_static, summarize_policy


def test_oracle_static_is_fixed_standalone_order() -> None:
    batch = generate_synthetic_batch(
        24, SyntheticConfig(candidate_count=8, group_count=8, budget=2, redundancy_ratio=0.5), seed=12
    )
    empty = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    standalone = observed_candidate_marginals(batch, empty)
    selected = select_oracle_static(batch, budget=2)
    expected = np.zeros_like(selected)
    order = np.argsort(-standalone, axis=1, kind="stable")[:, :2]
    expected[np.arange(batch.episodes)[:, None], order] = standalone[
        np.arange(batch.episodes)[:, None], order
    ] > 0.0
    np.testing.assert_array_equal(selected, expected)


def test_mmr_has_no_duplicate_groups_and_reports_rate() -> None:
    batch = generate_synthetic_batch(
        24, SyntheticConfig(candidate_count=16, group_count=16, budget=4, redundancy_ratio=0.75), seed=14
    )
    selected = select_mmr(batch, budget=4, gamma=0.5)
    for row in range(batch.episodes):
        groups = batch.candidate_group[row, selected[row]]
        assert len(groups) == len(np.unique(groups))
    metrics = summarize_policy(batch, selected, selected)
    assert metrics["duplicate_selection_rate"] == 0.0
