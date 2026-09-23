import pytest
import torch

from mur.model import MarginalUtilityRouter, RouterConfig, utility_training_loss


def test_set_encoder_is_permutation_invariant_and_handles_empty_set() -> None:
    torch.manual_seed(2)
    model = MarginalUtilityRouter(RouterConfig(embedding_dim=4, hidden_dim=8, set_dim=6))
    selected = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    encoded = model.encode_set(selected, mask)
    permuted = model.encode_set(selected[:, [1, 0, 2]], mask[:, [1, 0, 2]])
    assert torch.allclose(encoded[0], permuted[0], atol=1e-6)
    assert torch.allclose(encoded[1], torch.zeros(4))


def test_router_forward_and_training_loss() -> None:
    torch.manual_seed(3)
    cfg = RouterConfig(
        embedding_dim=5,
        prediction_dim=2,
        hidden_dim=10,
        set_dim=7,
        use_prediction_delta=True,
    )
    model = MarginalUtilityRouter(cfg)
    batch = 6
    output = model(
        torch.randn(batch, 5),
        torch.randn(batch, 3, 5),
        torch.tensor([[True, True, False]] * batch),
        torch.randn(batch, 5),
        torch.randn(batch, 2),
        torch.ones(batch, 1),
    )
    target = torch.linspace(-0.5, 0.5, batch)
    loss = utility_training_loss(
        output, target, group_ids=torch.tensor([0, 0, 0, 1, 1, 1]), ranking_weight=0.2
    )
    assert output.shape == (batch,)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_ranking_loss_requires_group_ids() -> None:
    with pytest.raises(ValueError):
        utility_training_loss(torch.zeros(2), torch.ones(2), ranking_weight=1.0)


def test_mur_light_rejects_exact_prediction_delta() -> None:
    model = MarginalUtilityRouter(RouterConfig(embedding_dim=3))
    with pytest.raises(ValueError, match="must be None"):
        model(
            torch.zeros(1, 3),
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, dtype=torch.bool),
            torch.zeros(1, 3),
            torch.zeros(1, 1),
            torch.ones(1, 1),
        )

