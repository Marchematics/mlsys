import numpy as np

from mur.electricity import ElectricityData, observed_candidate_marginals, sample_context_batch, squared_error


def _toy_data() -> ElectricityData:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(6 * 7 * 96, 8)).astype(np.float32)
    slot_features = np.column_stack(
        [np.ones(7 * 96), np.sin(np.arange(7 * 96) / 17.0), np.cos(np.arange(7 * 96) / 17.0)]
    ).astype(np.float32)
    return ElectricityData(
        values=values,
        train_clients=np.arange(4),
        validation_clients=np.arange(4, 6),
        test_clients=np.arange(6, 8),
        slot_features=slot_features,
        normalization_mean=0.0,
        normalization_sd=1.0,
    )


def test_electricity_context_batch_has_ground_truth_and_set_marginals() -> None:
    data = _toy_data()
    batch = sample_context_batch(
        data,
        clients=data.test_clients,
        week_start=0,
        week_stop=6,
        episodes=12,
        candidate_count=4,
        context_points=8,
        ridge_penalty=1.0,
        rng=np.random.default_rng(11),
    )
    empty = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
    marginal = observed_candidate_marginals(batch, empty)
    assert batch.query_embedding.shape == (12, 3 * data.slot_features.shape[1] + 3)
    assert batch.candidate_embedding.shape[:2] == (12, 4)
    assert np.isfinite(batch.target).all()
    assert np.isfinite(marginal).all()
    assert np.all(batch.candidate_week < batch.anchor_week[:, None])
    assert np.all(batch.candidate_week != batch.anchor_week[:, None])
    first = empty.copy()
    first[:, 0] = True
    assert np.allclose(
        marginal[:, 0],
        squared_error(batch, empty) - squared_error(batch, first),
    )
