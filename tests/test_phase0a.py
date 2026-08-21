import math

import numpy as np
import pytest
import torch

from distributional_idm.envs.quadratic import (
    QuadraticConfig,
    generate_dataset,
    make_same_mean_pair,
)
from distributional_idm.evaluation.metrics import (
    known_forward_cycle_error_quadratic,
    mode_coverage_from_samples,
    sample_mdn,
)
from distributional_idm.models.idm import (
    MDNIDM,
    DeterministicIDM,
    FixedVarianceGaussianIDM,
    GaussianIDM,
)
from distributional_idm.objectives.losses import (
    fixed_gaussian_nll_loss,
    mse_loss,
)
from distributional_idm.training import (
    TrainConfig,
    fit_standardizer,
    tensors_from,
    train_model,
)

TINY = {
    "n_train_episodes": 8,
    "n_val_episodes": 2,
    "n_test_episodes": 2,
    "episode_length": 50,
}


def test_dataset_shapes_and_splits() -> None:
    splits = generate_dataset(QuadraticConfig(**TINY, seed=0))
    assert set(splits) == {"train", "val", "test"}
    total = sum(len(splits[name]) for name in splits)
    assert total == (8 + 2 + 2) * 50
    train_ids = set(splits["train"].episode_ids.tolist())
    test_ids = set(splits["test"].episode_ids.tolist())
    assert train_ids.isdisjoint(test_ids)


def test_quadratic_dynamics_and_e10_support() -> None:
    data = generate_dataset(QuadraticConfig(name="e10", nonnegative_actions=True, **TINY, seed=1))["train"]
    displacement = data.next_states_clean - data.states
    np.testing.assert_allclose(displacement, data.actions**2, atol=1e-9)
    assert (data.actions > 0).all()

    symmetric = generate_dataset(QuadraticConfig(name="e1", **TINY, seed=1))["train"]
    assert (symmetric.actions < 0).any() and (symmetric.actions > 0).any()


def test_symmetric_conditional_has_zero_empirical_mean() -> None:
    data = generate_dataset(
        QuadraticConfig(magnitudes=(1.0,), n_train_episodes=200, n_val_episodes=20, n_test_episodes=20, episode_length=100, seed=3)
    )["train"]
    displacement = data.next_states_clean - data.states
    mask = np.isclose(displacement, 1.0)
    assert abs(data.actions[mask].mean()) < 0.05


def test_mse_head_learns_conditional_mean_not_modes() -> None:
    config = QuadraticConfig(
        magnitudes=(1.0,), n_train_episodes=120, n_val_episodes=20, n_test_episodes=40, episode_length=100, seed=0
    )
    splits = generate_dataset(config)
    conditioning = "endpoint"
    standardizer = fit_standardizer(splits["train"], conditioning)
    tensors = {
        name: tensors_from(splits[name], conditioning, standardizer) for name in ("train", "val", "test")
    }
    model = DeterministicIDM(tensors["train"]["features"].shape[-1])
    train_model(model, tensors["train"], tensors["val"], TrainConfig(epochs=150), seed=0)
    with torch.no_grad():
        predictions = model.predict(tensors["test"]["features"])["mean"]
    assert float(predictions.abs().mean()) < 0.35


def test_mdn_captures_both_modes() -> None:
    torch.manual_seed(0)
    rows = 256
    features = torch.randn(rows, 4)
    actions = torch.where(torch.rand(rows) < 0.5, -torch.ones(rows), torch.ones(rows))
    model = MDNIDM(input_dim=4, n_components=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(800):
        logits, means, raw_log_var = model(features)
        from distributional_idm.objectives.losses import mdn_nll_loss

        loss = mdn_nll_loss(logits, means, raw_log_var, actions, variance_floor=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        outputs = model.predict(features)
        samples = sample_mdn(outputs["logits"], outputs["means"], outputs["stds"], 16, torch.Generator().manual_seed(7))
        low = -torch.ones(rows, 1)
        high = torch.ones(rows, 1)
        tol = torch.full((rows, 1), 0.05)
        coverage = mode_coverage_from_samples(samples, low, high, tol)
    assert coverage > 0.9


def test_fixed_variance_gradient_matches_mse_direction() -> None:
    features = torch.randn(64, 3)
    actions = torch.randn(64)
    det_model = DeterministicIDM(3)
    fixed_model = FixedVarianceGaussianIDM(3)
    fixed_model.net.load_state_dict(det_model.net.state_dict())

    mse_grad = torch.autograd.grad(mse_loss(det_model(features), actions), list(det_model.parameters()))
    fixed_grad = torch.autograd.grad(
        fixed_gaussian_nll_loss(fixed_model(features), actions, std=0.5), list(fixed_model.parameters())
    )
    for g_mse, g_fixed in zip(mse_grad, fixed_grad):
        cosine = torch.nn.functional.cosine_similarity(g_mse.flatten(), g_fixed.flatten(), dim=0)
        assert cosine > 0.999


def test_gaussian_variance_respects_floor() -> None:
    model = GaussianIDM(3, variance_floor=1e-2)
    features = torch.randn(16, 3)
    with torch.no_grad():
        outputs = model.predict(features)
    assert float(outputs["std"].min()) >= math.sqrt(1e-2) - 1e-6


def test_cycle_metric_matches_manual_computation() -> None:
    states = torch.tensor([0.0, 1.0])
    candidates = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    next_clean = torch.tensor([1.0, 2.25])
    expected = (abs(0 + 1 - 1) + abs(0 + 1 - 1) + abs(1 + 0.25 - 2.25) + abs(1 + 4 - 2.25)) / 4
    result = known_forward_cycle_error_quadratic(states, candidates, next_clean)
    assert abs(result - expected / 1) < 1e-6


def test_same_mean_pair_requires_noise_and_matches() -> None:
    with pytest.raises(ValueError):
        make_same_mean_pair(noise_std=0.0)
    _, _, diag = make_same_mean_pair(noise_std=0.15, mean_tolerance=5e-3)
    assert abs(diag["mean_a"] - diag["mean_b"]) <= 5e-3
    assert diag["variance_b"] > diag["variance_a"]
