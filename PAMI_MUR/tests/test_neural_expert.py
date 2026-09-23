"""Unit tests for the PAMI subset-capable experts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "experiments"))

from neural_expert import (  # noqa: E402
    SubsetDeepSets,
    SubsetExpertOps,
    SubsetSTTransformer,
    build_backbone,
    pack_selected,
    sample_subset_masks,
)
from pami_traffic import NeuralTrafficBatch, build_candidate_pool  # noqa: E402


def make_batch(episodes: int = 5, candidates: int = 6, history: int = 4, horizon: int = 3, nodes: int = 21):
    rng = np.random.default_rng(0)
    return NeuralTrafficBatch(
        anchor_history=rng.normal(size=(episodes, history)).astype(np.float32),
        candidate_history=rng.normal(size=(episodes, candidates, history)).astype(np.float32),
        target_future=rng.normal(size=(episodes, horizon)).astype(np.float32),
        target_sensor=3,
        candidate_sensors=rng.choice(nodes, size=candidates, replace=False).astype(np.int64),
    )


@pytest.mark.parametrize("backbone", [SubsetSTTransformer, SubsetDeepSets])
def test_shapes_and_empty_subset(backbone):
    torch.manual_seed(0)
    model = backbone(num_nodes=21, history_length=4, horizon=3, d_model=16, num_layers=2 if backbone is SubsetSTTransformer else None) if False else backbone(
        num_nodes=21,
        history_length=4,
        horizon=3,
        **({"d_model": 16, "num_layers": 2, "nhead": 2, "feed_forward_dim": 32} if backbone is SubsetSTTransformer else {"d_model": 16, "hidden_dim": 32}),
    )
    ops = SubsetExpertOps(model, device="cpu")
    batch = make_batch()
    for size in (0, 1, 6):
        mask = np.zeros((batch.episodes, batch.candidate_count), dtype=bool)
        if size:
            mask[:, :size] = True
        prediction = ops.predict(batch, mask)
        assert prediction.shape == (batch.episodes, batch.horizon)
        assert np.isfinite(prediction).all()


def test_unselected_candidates_do_not_change_prediction():
    """Subset capability: extra unselected candidates must be inert."""

    torch.manual_seed(0)
    model = SubsetSTTransformer(
        num_nodes=21, history_length=4, horizon=3, d_model=16, num_layers=2, nhead=2, feed_forward_dim=32
    )
    ops = SubsetExpertOps(model, device="cpu")
    batch = make_batch(episodes=4, candidates=6)
    mask = np.zeros((batch.episodes, 6), dtype=bool)
    mask[:, 0] = True
    first = ops.predict(batch, mask)
    # Append two extra candidates that stay unselected.
    extended = NeuralTrafficBatch(
        anchor_history=batch.anchor_history,
        candidate_history=np.concatenate([batch.candidate_history, np.random.default_rng(1).normal(size=(4, 2, 4)).astype(np.float32)], axis=1),
        target_future=batch.target_future,
        target_sensor=batch.target_sensor,
        candidate_sensors=np.concatenate([batch.candidate_sensors, np.array([9, 10])]),
    )
    wider = np.zeros((4, 8), dtype=bool)
    wider[:, 0] = True
    second = ops.predict(extended, wider)
    np.testing.assert_allclose(first, second, atol=1e-5)


def test_candidate_order_does_not_change_prediction():
    torch.manual_seed(0)
    model = SubsetSTTransformer(
        num_nodes=21, history_length=4, horizon=3, d_model=16, num_layers=2, nhead=2, feed_forward_dim=32
    )
    ops = SubsetExpertOps(model, device="cpu")
    batch = make_batch(episodes=4, candidates=6)
    mask = np.zeros((4, 6), dtype=bool)
    mask[:, [0, 3]] = True
    base = ops.predict(batch, mask)
    order = np.array([3, 0, 5, 4, 2, 1])
    permuted = NeuralTrafficBatch(
        anchor_history=batch.anchor_history,
        candidate_history=batch.candidate_history[:, order],
        target_future=batch.target_future,
        target_sensor=batch.target_sensor,
        candidate_sensors=batch.candidate_sensors[order],
    )
    prediction = ops.predict(permuted, mask[:, order])
    np.testing.assert_allclose(base, prediction, atol=1e-5)


def test_marginals_match_brute_force():
    torch.manual_seed(0)
    model = SubsetSTTransformer(
        num_nodes=21, history_length=4, horizon=3, d_model=16, num_layers=2, nhead=2, feed_forward_dim=32
    )
    ops = SubsetExpertOps(model, device="cpu")
    batch = make_batch(episodes=6, candidates=5)
    selected = np.zeros((6, 5), dtype=bool)
    selected[0, 1] = True
    selected[2, 0] = True
    marginals = ops.candidate_marginals(batch, selected)
    assert marginals.shape == (6, 5)
    assert np.all(np.isneginf(marginals[selected]))
    base = ops.squared_error(batch, selected)
    for candidate in range(5):
        for row in range(6):
            if selected[row, candidate]:
                continue
            augmented = selected.copy()
            augmented[row, candidate] = True
            expected = base[row] - ops.squared_error(batch, augmented)[row]
            assert marginals[row, candidate] == pytest.approx(expected, abs=1e-6)


def test_response_summary_matches_squared_loss_identity():
    """m(j|A) = mean_h [2 d_h (y_h - f_h) - d_h^2] for the squared loss."""

    torch.manual_seed(0)
    model = SubsetSTTransformer(
        num_nodes=21, history_length=4, horizon=3, d_model=16, num_layers=2, nhead=2, feed_forward_dim=32
    )
    ops = SubsetExpertOps(model, device="cpu")
    batch = make_batch(episodes=5, candidates=5)
    selected = np.zeros((5, 5), dtype=bool)
    selected[1, 2] = True
    summary = ops.response_summary(batch, selected)
    marginals = ops.candidate_marginals(batch, selected)
    base = ops.predict(batch, selected)
    for candidate in range(5):
        for row in range(5):
            if selected[row, candidate]:
                assert np.allclose(summary[row, candidate], 0.0)
                continue
            augmented = selected.copy()
            augmented[row, candidate] = True
            after = ops.predict(batch, augmented)[row]
            delta = after - base[row]
            identity = np.mean(2.0 * delta * (batch.target_future[row] - base[row]) - delta ** 2)
            assert marginals[row, candidate] == pytest.approx(identity, abs=1e-6)
            assert summary[row, candidate, 2] == pytest.approx(float(np.mean(delta)), abs=1e-5)
            assert summary[row, candidate, 4] == pytest.approx(float(np.mean(np.abs(delta))), abs=1e-5)


def test_candidate_mask_zeros_features():
    torch.manual_seed(0)
    model = SubsetDeepSets(num_nodes=21, history_length=4, horizon=3, d_model=16, hidden_dim=32)
    ops = SubsetExpertOps(model, device="cpu")
    batch = make_batch(episodes=4, candidates=5)
    selected = np.zeros((4, 5), dtype=bool)
    candidate_mask = np.zeros((4, 5), dtype=bool)
    candidate_mask[:, 2] = True
    summary = ops.response_summary(batch, selected, candidate_mask=candidate_mask)
    assert np.allclose(summary[:, [0, 1, 3, 4]], 0.0)
    assert not np.allclose(summary[:, 2], 0.0)


def test_pack_selected_matches_protocol():
    batch = make_batch(episodes=3, candidates=5, history=4)
    selected = np.zeros((3, 5), dtype=bool)
    selected[0, 3] = True
    selected[1, [0, 4]] = True
    values, valid = pack_selected(batch, selected, max_budget=3)
    assert values.shape == (3, 3, 4)
    assert valid.tolist() == [[True, False, False], [True, True, False], [False, False, False]]
    np.testing.assert_allclose(values[0, 0], batch.candidate_history[0, 3])
    np.testing.assert_allclose(values[1, 1], batch.candidate_history[1, 4])


def test_subset_mask_mixture_covers_sizes():
    rng = np.random.default_rng(0)
    masks = sample_subset_masks(rng, 500, 16, budget=4)
    sizes = masks.sum(axis=1)
    assert sizes.min() == 0
    assert sizes.max() == 16
    assert (sizes <= 4).mean() > 0.6
    assert ((sizes > 4) & (sizes < 16)).mean() > 0.05


def test_deepsets_and_transformer_registered():
    for name in ("subset_transformer", "subset_deepsets"):
        model = build_backbone(name, 10, 4, 2, d_model=8, **({"num_layers": 1, "nhead": 2} if name == "subset_transformer" else {}))
        assert model is not None


def test_pair_marginals_match_candidate_marginals():
    torch.manual_seed(0)
    model = SubsetSTTransformer(
        num_nodes=21, history_length=4, horizon=3, d_model=16, num_layers=2, nhead=2, feed_forward_dim=32
    )
    ops = SubsetExpertOps(model, device="cpu")
    batch = make_batch(episodes=6, candidates=5)
    selected = np.zeros((6, 5), dtype=bool)
    selected[0, 1] = True
    selected[3, 4] = True
    columns = np.array([2, 0, 3, 1, 2, 0])
    fast = ops.pair_marginals(batch, selected, columns)
    reference = ops.candidate_marginals(batch, selected)
    for row, column in enumerate(columns):
        if selected[row, column]:
            continue
        assert fast[row] == pytest.approx(reference[row, column], abs=1e-6)


def test_joint_marginals_and_responses_match_separate_paths():
    torch.manual_seed(0)
    model = SubsetSTTransformer(
        num_nodes=21, history_length=4, horizon=3, d_model=16, num_layers=2, nhead=2, feed_forward_dim=32
    )
    ops = SubsetExpertOps(model, device="cpu")
    batch = make_batch(episodes=5, candidates=5)
    selected = np.zeros((5, 5), dtype=bool)
    selected[2, 3] = True
    joint_marginal, joint_response = ops.candidate_marginals_and_responses(batch, selected)
    reference_marginal = ops.candidate_marginals(batch, selected)
    reference_response = ops.response_summary(batch, selected)
    finite = np.isfinite(reference_marginal)
    np.testing.assert_allclose(joint_marginal[finite], reference_marginal[finite], atol=1e-6)
    np.testing.assert_allclose(joint_response, reference_response, atol=1e-5)


def test_candidate_pool_structure():
    correlation = np.linspace(-1, 1, 40)
    pool = build_candidate_pool(correlation, target_sensor=0, candidate_count=12, seed=3)
    assert pool.shape == (12,)
    assert len(set(pool.tolist())) == 12
    assert 0 not in pool
