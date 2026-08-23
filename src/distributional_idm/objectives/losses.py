"""Objective functions for Phase 0A action heads.

Signs follow the convention that training minimizes the returned scalar.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predictions, targets)


def gaussian_nll_loss(
    mean: torch.Tensor,
    raw_log_variance: torch.Tensor,
    targets: torch.Tensor,
    variance_floor: float,
) -> torch.Tensor:
    log_var = raw_log_variance.clamp_min(math.log(variance_floor))
    return 0.5 * ((targets - mean) ** 2 / log_var.exp() + log_var).mean()


def beta_nll_loss(
    mean: torch.Tensor,
    raw_log_variance: torch.Tensor,
    targets: torch.Tensor,
    variance_floor: float,
    beta: float = 0.5,
) -> torch.Tensor:
    """PRISM-faithful beta-NLL: detached sg((sigma^2)^beta) times the
    per-sample diagonal-Gaussian NLL (Seitzer et al.; PRISM uses beta=0.5)."""

    log_var = raw_log_variance.clamp_min(math.log(variance_floor))
    per_sample = 0.5 * (targets - mean) ** 2 / log_var.exp() + log_var
    weight = (log_var.exp().detach()) ** beta
    return (weight * per_sample).mean()


def clamped_gaussian_nll_loss(
    mean: torch.Tensor,
    raw_log_variance: torch.Tensor,
    targets: torch.Tensor,
    variance_floor: float,
    clamp_value: float = 10.0,
) -> torch.Tensor:
    log_var = raw_log_variance.clamp_min(math.log(variance_floor))
    per_sample = 0.5 * (targets - mean) ** 2 / log_var.exp() + log_var
    return per_sample.clamp_max(clamp_value).mean()


def fixed_gaussian_nll_loss(
    mean: torch.Tensor,
    targets: torch.Tensor,
    std: float,
) -> torch.Tensor:
    return (0.5 * (targets - mean) ** 2 / std**2).mean()


def mdn_nll_loss(
    logits: torch.Tensor,
    means: torch.Tensor,
    raw_log_variance: torch.Tensor,
    targets: torch.Tensor,
    variance_floor: float,
) -> torch.Tensor:
    log_var = raw_log_variance.clamp_min(math.log(variance_floor))
    var = log_var.exp()
    component_nll = 0.5 * ((targets.unsqueeze(-1) - means) ** 2 / var + log_var)
    log_mix = torch.logsumexp(F.log_softmax(logits, dim=-1) - component_nll, dim=-1)
    return -log_mix.mean()
