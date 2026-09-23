"""Tests for the learned subset-selection baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from neural_expert import SubsetDeepSets, SubsetExpertOps  # noqa: E402
from pami_traffic import NeuralTrafficBatch  # noqa: E402
from policy_selector import LearnedPolicySelector, select_learned_policy, train_policy_selector  # noqa: E402


def make_batch(episodes: int = 12, candidates: int = 5, history: int = 4, horizon: int = 3):
    rng = np.random.default_rng(0)
    return NeuralTrafficBatch(
        anchor_history=rng.normal(size=(episodes, history)).astype(np.float32),
        candidate_history=rng.normal(size=(episodes, candidates, history)).astype(np.float32),
        target_future=rng.normal(size=(episodes, horizon)).astype(np.float32),
        target_sensor=1,
        candidate_sensors=rng.choice(15, size=candidates, replace=False).astype(np.int64),
    )


def test_policy_scores_every_candidate():
    torch.manual_seed(0)
    model = LearnedPolicySelector(embedding_dim=4, hidden_dim=16, set_dim=8)
    batch = make_batch()
    packed, valid = SubsetExpertOps(
        SubsetDeepSets(num_nodes=15, history_length=4, horizon=3, d_model=8, hidden_dim=16), device="cpu"
    ).pack(batch, np.zeros((batch.episodes, batch.candidate_count), dtype=bool), max_budget=3)
    logits = model(
        torch.from_numpy(batch.query_embedding).float(),
        torch.from_numpy(packed).float(),
        torch.from_numpy(valid),
        torch.from_numpy(batch.candidate_embedding).float(),
        3,
    )
    assert logits.shape == (batch.episodes, batch.candidate_count)
    assert torch.isfinite(logits).all()


def test_greedy_selection_respects_budget():
    torch.manual_seed(0)
    batch = make_batch()
    expert = SubsetExpertOps(
        SubsetDeepSets(num_nodes=15, history_length=4, horizon=3, d_model=8, hidden_dim=16), device="cpu"
    )
    model = LearnedPolicySelector(embedding_dim=4, hidden_dim=16, set_dim=8)
    selected = select_learned_policy(model, batch, expert, budget=3, device="cpu")
    assert selected.shape == (batch.episodes, batch.candidate_count)
    assert np.all(selected.sum(axis=1) == 3)


def test_training_improves_or_preserves_reward():
    torch.manual_seed(0)
    batch = make_batch(episodes=24, candidates=5)
    expert = SubsetExpertOps(
        SubsetDeepSets(num_nodes=15, history_length=4, horizon=3, d_model=8, hidden_dim=16), device="cpu"
    )
    model = LearnedPolicySelector(embedding_dim=4, hidden_dim=16, set_dim=8)

    def mean_reward():
        selected = select_learned_policy(model, batch, expert, budget=2, device="cpu")
        return float(np.mean(expert.squared_error(batch, selected)))

    before = mean_reward()
    info = train_policy_selector(
        model, batch, expert, budget=2, epochs=8, batch_size=8, seed=1, device="cpu"
    )
    after = mean_reward()
    assert np.isfinite(after)
    assert len(info["mean_reward_history"]) == 8
    assert after >= before - 1e-6
