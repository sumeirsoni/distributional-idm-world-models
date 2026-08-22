
import numpy as np
import pytest
import torch
from torch import nn

from distributional_idm.evaluation.planning import (
    latent_planning_success,
    posthoc_cycle_validity,
)
from distributional_idm.evaluation.probes import latent_health, ridge_probe_r2
from distributional_idm.models.idm import DeterministicIDM
from distributional_idm.world_model import (
    ArmConfig,
    Encoder,
    RandomFeatureTarget,
    WorldModel,
    vicreg_coverage,
)


def test_encoder_accepts_flat_and_column_states() -> None:
    encoder = Encoder()
    flat = torch.randn(7)
    column = flat.unsqueeze(-1)
    assert torch.allclose(encoder(flat), encoder(column))


def test_idm_gradients_reach_encoder() -> None:
    model = WorldModel(ArmConfig(name="det", idm_kind="deterministic"))
    states = torch.randn(32)
    next_states = states + 0.25
    actions = torch.where(torch.rand(32) < 0.5, -torch.ones(32), torch.ones(32)) * 0.5
    loss = model.idm_loss(actions, states, next_states)
    assert loss is not None
    loss.backward()
    total_grad = sum(
        float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None
    )
    assert total_grad > 0


def test_vicreg_coverage_penalizes_collapse_and_correlation() -> None:
    collapsed = torch.zeros(64, 4)
    variance_hinge, _ = vicreg_coverage(collapsed)
    assert float(variance_hinge) == pytest.approx(1.0)

    generator = torch.Generator().manual_seed(0)
    independent = torch.randn(4096, 8, generator=generator)
    _, cov_penalty = vicreg_coverage(independent)
    correlated = independent.repeat(1, 2)
    _, cov_penalty_doubled = vicreg_coverage(correlated)
    assert cov_penalty_doubled > 10 * cov_penalty


def test_random_feature_target_is_deterministic() -> None:
    states = torch.randn(16)
    first = RandomFeatureTarget(seed=3)(states)
    second = RandomFeatureTarget(seed=3)(states)
    third = RandomFeatureTarget(seed=4)(states)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_ridge_probe_recovers_linear_target() -> None:
    generator = np.random.default_rng(0)
    latents = generator.normal(size=(500, 8))
    targets = latents @ np.linspace(-2, 2, 8) + 3.0
    r2 = ridge_probe_r2(latents[:400], targets[:400], latents[400:], targets[400:])
    assert r2 > 0.999


def test_latent_health_detects_rank() -> None:
    generator = np.random.default_rng(1)
    full_rank = generator.normal(size=(300, 8))
    one_dim = np.zeros((300, 8))
    one_dim[:, 0] = generator.normal(size=300)
    assert latent_health(full_rank)["effective_rank"] > 6
    assert latent_health(one_dim)["effective_rank"] < 1.1


def test_latent_planning_with_perfect_model_succeeds() -> None:
    class IdentityEncoder(nn.Module):
        def forward(self, states: torch.Tensor) -> torch.Tensor:
            if states.dim() == 1:
                states = states.unsqueeze(-1)
            return states.repeat(1, 4)

    class PredictorWithDisplacement(nn.Module):
        def forward(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            out = z.clone()
            out[:, :1] = out[:, :1] + actions.unsqueeze(-1) ** 2
            return out

    encoder = IdentityEncoder()
    states = torch.tensor([0.0, 2.0, -3.0])
    next_states = states + 1.0
    result = latent_planning_success(encoder, PredictorWithDisplacement(), states, next_states)
    assert result["planning_success"] == pytest.approx(1.0)


def test_posthoc_cycle_validity_prefers_modal_actions() -> None:
    class FlatEncoder(nn.Module):
        def forward(self, states: torch.Tensor) -> torch.Tensor:
            return states.unsqueeze(-1) * 0.0

    head = DeterministicIDM(2)
    with torch.no_grad():
        for module in head.net.modules():
            if isinstance(module, nn.Linear):
                module.bias.zero_()
        head.net[-1].weight.zero_()
    states = torch.zeros(50)
    next_states = states + 1.0
    error_zero_actions = posthoc_cycle_validity(head, FlatEncoder(), states, next_states, 4)
    assert error_zero_actions == pytest.approx(1.0)
