"""Representation probes and latent-health statistics."""

from __future__ import annotations

import numpy as np
import torch


def ridge_probe_r2(
    latents_train: np.ndarray,
    targets_train: np.ndarray,
    latents_test: np.ndarray,
    targets_test: np.ndarray,
    ridge: float = 1e-3,
) -> float:
    """Closed-form ridge regression R2 of a scalar target from latents.

    Targets are standardized with train statistics before fitting so that
    drifting state scales do not distort the comparison. Non-finite latents
    are zeroed, which drives R2 toward the intercept-only baseline.
    """

    latents_train = np.nan_to_num(latents_train, nan=0.0, posinf=0.0, neginf=0.0)
    latents_test = np.nan_to_num(latents_test, nan=0.0, posinf=0.0, neginf=0.0)

    mean = targets_train.mean()
    std = max(float(targets_train.std()), 1e-6)
    y_train = (targets_train - mean) / std
    y_test = (targets_test - mean) / std

    x_train = np.concatenate([latents_train, np.ones((len(latents_train), 1))], axis=1)
    x_test = np.concatenate([latents_test, np.ones((len(latents_test), 1))], axis=1)
    gram = x_train.T @ x_train + ridge * len(x_train) * np.eye(x_train.shape[1])
    weights = np.linalg.solve(gram, x_train.T @ y_train)

    predictions = x_test @ weights
    ss_res = float(((y_test - predictions) ** 2).sum())
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


@torch.no_grad()
def encode_dataset(encoder: torch.nn.Module, states: torch.Tensor) -> np.ndarray:
    encoder.eval()
    return encoder(states).cpu().numpy()


def latent_health(latents: np.ndarray) -> dict[str, float]:
    """Variance, effective rank, and singular-value decay of a latent set."""

    latents = np.nan_to_num(latents, nan=0.0, posinf=1e6, neginf=-1e6)
    centered = latents - latents.mean(axis=0)
    singular_values = np.linalg.svd(centered / math_sqrt(len(centered)), compute_uv=False)
    energy = singular_values**2
    distribution = energy / (energy.sum() + 1e-12)
    entropy = -(distribution[distribution > 0] * np.log(distribution[distribution > 0] + 1e-12)).sum()
    effective_rank = float(np.exp(entropy))
    return {
        "mean_dim_variance": float(latents.var(axis=0).mean()),
        "effective_rank": effective_rank,
        "top_singular_value": float(singular_values[0]),
    }


def math_sqrt(value: int) -> float:
    return value**0.5
