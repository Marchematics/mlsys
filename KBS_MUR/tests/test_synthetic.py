import numpy as np

from mur.synthetic import (
    SyntheticConfig,
    expert_prediction,
    generate_synthetic_batch,
    observed_candidate_marginals,
    squared_error,
)


def test_duplicates_have_positive_standalone_and_zero_post_selection_marginal() -> None:
    batch = generate_synthetic_batch(
        64,
        SyntheticConfig(candidate_count=8, group_count=8, redundancy_ratio=0.75),
        seed=4,
    )
    empty = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    standalone = observed_candidate_marginals(batch, empty)
    assert np.all(standalone > 0.0)
    found = False
    for row in range(batch.episodes):
        groups, counts = np.unique(batch.candidate_group[row], return_counts=True)
        duplicate_group = int(groups[np.argmax(counts)])
        columns = np.flatnonzero(batch.candidate_group[row] == duplicate_group)
        if columns.size < 2:
            continue
        selected = empty.copy()
        selected[row, columns[0]] = True
        marginal = observed_candidate_marginals(batch, selected)
        assert abs(marginal[row, columns[1]]) < 1e-10
        found = True
        break
    assert found


def test_oracle_single_candidate_improves_frozen_expert() -> None:
    batch = generate_synthetic_batch(32, SyntheticConfig(), seed=8)
    empty = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    marginal = observed_candidate_marginals(batch, empty)
    choice = np.argmax(marginal, axis=1)
    selected = empty.copy()
    selected[np.arange(batch.episodes), choice] = True
    assert np.mean(squared_error(batch, selected)) < np.mean(squared_error(batch, empty))
    assert expert_prediction(batch, empty).shape == batch.target.shape

