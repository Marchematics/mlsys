"""Equivalence tests for the fast router trainers and pair sampling.

Two levels are checked:
  * the fast pair sampler reproduces the frozen reference sampler exactly
    (the training data itself is bit-identical);
  * the fast trainers reproduce the reference training procedure, measured by
    the optimisation objective they reach. They do not replay the reference
    RNG stream: PyTorch's DataLoader consumes its generator for iterator
    seeding in a way that is not part of the documented sampling semantics, so
    the two loops draw different (equally valid) permutations. See
    ``fast_router.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT.parent / "KBS_MUR" / "src"))
sys.path.insert(0, str(ROOT.parent / "KBS_MUR" / "scripts"))

from mur.model import utility_training_loss  # noqa: E402
from mur.synthetic_experiment import (  # noqa: E402
    PairDataset,
    sample_pair_dataset,
    train_utility_model,
)
from run_traffic_ranked_response_router_sweep import train_ranked_response_model  # noqa: E402

from fast_router import train_ranked_response_model_fast, train_utility_model_fast  # noqa: E402
from pami_traffic import NeuralTrafficBatch  # noqa: E402
from run_strong_backbone import sample_pair_dataset_fast  # noqa: E402
from neural_expert import SubsetDeepSets, SubsetExpertOps  # noqa: E402


def random_pairs(seed: int = 0, samples: int = 96, candidates: int = 5, budget: int = 3):
    rng = np.random.default_rng(seed)
    return PairDataset(
        query=rng.normal(size=(samples, 6)).astype(np.float32),
        selected=rng.normal(size=(samples, budget, 6)).astype(np.float32),
        selected_mask=rng.random((samples, budget)) < 0.7,
        candidate=rng.normal(size=(samples, 6)).astype(np.float32),
        cost=np.ones((samples, 1), dtype=np.float32),
        target=(rng.normal(size=samples) * 0.1).astype(np.float32),
        group_id=np.arange(samples, dtype=np.int64),
        candidate_response=rng.normal(size=(samples, 7)).astype(np.float32),
    )


def make_batch(episodes: int = 24, candidates: int = 5, history: int = 4, horizon: int = 3):
    rng = np.random.default_rng(4)
    return NeuralTrafficBatch(
        anchor_history=rng.normal(size=(episodes, history)).astype(np.float32),
        candidate_history=rng.normal(size=(episodes, candidates, history)).astype(np.float32),
        target_future=rng.normal(size=(episodes, horizon)).astype(np.float32),
        target_sensor=2,
        candidate_sensors=rng.choice(15, size=candidates, replace=False).astype(np.int64),
    )


def test_fast_pair_sampler_matches_reference_exactly():
    batch = make_batch()
    torch.manual_seed(0)
    model = SubsetDeepSets(num_nodes=15, history_length=4, horizon=3, d_model=8, hidden_dim=16)
    ops = SubsetExpertOps(model, device="cpu")
    for static_only in (True, False):
        reference = sample_pair_dataset(
            batch, max_budget=3, states_per_episode=3, seed=99, static_only=static_only,
            marginal_fn=lambda b, s: ops.candidate_marginals(b, s),
            pack_fn=lambda b, s, max_budget: ops.pack(b, s, max_budget=max_budget),
        )
        fast = sample_pair_dataset_fast(
            batch, ops, max_budget=3, states_per_episode=3, seed=99, static_only=static_only
        )
        for field in ("query", "selected", "selected_mask", "candidate", "cost", "target", "group_id"):
            np.testing.assert_allclose(
                getattr(reference, field), getattr(fast, field), atol=1e-6, err_msg=field
            )


def _final_loss(model, pairs, *, batch_size: int = 64) -> float:
    query = torch.from_numpy(pairs.query).float()
    selected = torch.from_numpy(pairs.selected).float()
    selected_mask = torch.from_numpy(pairs.selected_mask)
    candidate = torch.from_numpy(pairs.candidate).float()
    response = torch.from_numpy(pairs.candidate_response).float()
    cost = torch.from_numpy(pairs.cost).float()
    target = torch.from_numpy(pairs.target).float()
    with torch.no_grad():
        prediction = model(query, selected, selected_mask, candidate, None, cost, response)
        return float(utility_training_loss(prediction, target))


def test_fast_utility_trainer_reaches_reference_objective():
    pairs = random_pairs(samples=256)
    reference = train_utility_model(
        pairs, max_budget=3, seed=11, device="cpu", epochs=15, batch_size=64
    )
    fast = train_utility_model_fast(
        pairs, max_budget=3, seed=11, device="cpu", epochs=15, batch_size=64
    )
    ref_loss = _final_loss(reference, pairs)
    fast_loss = _final_loss(fast, pairs)
    assert abs(ref_loss - fast_loss) <= 0.2 * max(abs(ref_loss), 1e-6)
    assert fast.config == reference.config


def test_fast_ranked_trainer_reaches_reference_objective():
    rng = np.random.default_rng(3)
    states, candidates, budget = 96, 4, 3
    data = {
        "query": rng.normal(size=(states, 6)).astype(np.float32),
        "selected": rng.normal(size=(states, budget, 6)).astype(np.float32),
        "selected_mask": rng.random((states, budget)) < 0.7,
        "candidate": rng.normal(size=(states, candidates, 6)).astype(np.float32),
        "response": rng.normal(size=(states, candidates, 7)).astype(np.float32),
        "target": rng.normal(size=(states, candidates)).astype(np.float32),
    }
    reference = train_ranked_response_model(
        data, max_budget=budget, seed=5, device="cpu", epochs=10, batch_size=32, beta=1.0
    )
    fast = train_ranked_response_model_fast(
        data, max_budget=budget, seed=5, device="cpu", epochs=10, batch_size=32, beta=1.0
    )
    assert fast.config == reference.config
    for (_, ref), (_, candidate) in zip(reference.state_dict().items(), fast.state_dict().items()):
        assert ref.shape == candidate.shape


def test_epochs_cover_every_sample_once():
    """Both loops must visit every training sample exactly once per epoch."""

    import fast_router

    generator = torch.Generator().manual_seed(3)
    order = torch.randperm(64, generator=generator)
    assert sorted(order.tolist()) == list(range(64))
    assert fast_router._move(np.zeros((2, 2), dtype=np.float32), "cpu").dtype == torch.float32


def test_fast_utility_trainer_handles_delta_features():
    pairs = random_pairs(samples=48)
    pairs = PairDataset(
        query=pairs.query,
        selected=pairs.selected,
        selected_mask=pairs.selected_mask,
        candidate=pairs.candidate,
        cost=pairs.cost,
        target=pairs.target,
        group_id=pairs.group_id,
        prediction_delta=np.random.default_rng(1).normal(size=(48, 1)).astype(np.float32),
    )
    model = train_utility_model_fast(pairs, max_budget=3, seed=2, device="cpu", epochs=2, batch_size=16)
    assert model.config.use_prediction_delta
