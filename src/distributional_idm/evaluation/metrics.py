"""Density, mode, and cycle metrics for Phase 0A.

All functions take torch tensors on any device unless noted otherwise.
Mode locations for E1 are analytic: the valid actions for a clean
displacement d are +-sqrt(d), so callers pass per-row mode arrays.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_nll(
    mean: torch.Tensor,
    std: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    nll = 0.5 * (targets - mean) ** 2 / std**2 + std.log()
    return float(nll.mean().detach())


def mdn_nll(
    logits: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    log_probs = -0.5 * (targets.unsqueeze(-1) - means) ** 2 / stds**2 - stds.log()
    log_mix = torch.logsumexp(F.log_softmax(logits, dim=-1) + log_probs, dim=-1)
    return float(-log_mix.mean().detach())


def mdn_component_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def component_band_mass(
    mean: torch.Tensor,
    std: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Probability each Gaussian component assigns to [lower, upper]."""

    std = std.clamp_min(1e-8)
    return normal_cdf((upper - mean) / std) - normal_cdf((lower - mean) / std)


def _column(v: torch.Tensor) -> torch.Tensor:
    """(n,) or (n,1) -> (n,1); idempotent, so callers may pass either."""

    return v.reshape(-1, 1)


def invalid_between_mode_mass(
    logits: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
    modes_low: torch.Tensor,
    modes_high: torch.Tensor,
    tol: torch.Tensor,
    second_band: bool = True,
) -> float:
    """Mean probability mass outside the valid-mode tolerance bands."""

    weights = mdn_component_probabilities(logits)
    low_band = component_band_mass(
        means, stds, _column(modes_low) - _column(tol), _column(modes_low) + _column(tol)
    )
    high_band = (
        component_band_mass(
            means, stds, _column(modes_high) - _column(tol), _column(modes_high) + _column(tol)
        )
        if second_band
        else torch.zeros_like(low_band)
    )
    inside = (weights * (low_band + high_band)).sum(dim=-1)
    return float((1.0 - inside).mean().detach())


def heteroscedastic_invalid_mass(
    mean: torch.Tensor,
    std: torch.Tensor,
    modes_low: torch.Tensor,
    modes_high: torch.Tensor,
    tol: torch.Tensor,
    second_band: bool = True,
) -> float:
    low_band = component_band_mass(
        _column(mean), _column(std), _column(modes_low) - _column(tol), _column(modes_low) + _column(tol)
    )
    high_band = (
        component_band_mass(
            _column(mean), _column(std), _column(modes_high) - _column(tol), _column(modes_high) + _column(tol)
        )
        if second_band
        else torch.zeros_like(low_band)
    )
    inside = low_band + high_band
    return float((1.0 - inside).mean().detach())


def sample_mdn(
    logits: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
    n_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample (n_rows, n_samples) actions from a batch of 1D mixtures."""

    weights = mdn_component_probabilities(logits)
    n_rows, k = weights.shape
    flat_weights = weights.reshape(-1, k)
    counts = torch.multinomial(
        flat_weights.detach().cpu(), num_samples=n_samples, replacement=True, generator=generator
    )
    counts = counts.to(means.device)

    rows = torch.arange(n_rows, device=means.device).unsqueeze(-1).expand(-1, n_samples)
    comp_means = means[rows, counts]
    comp_stds = stds[rows, counts]
    noise = torch.randn(
        n_rows, n_samples, device=means.device, dtype=means.dtype, generator=generator
    )
    return comp_means + comp_stds * noise


def mode_coverage_from_samples(
    samples: torch.Tensor,
    modes_low: torch.Tensor,
    modes_high: torch.Tensor,
    tol: torch.Tensor,
    second_band: bool = True,
) -> float:
    """Fraction of rows with at least one sample inside the mode bands."""

    low, high, t = _column(modes_low), _column(modes_high), _column(tol)
    near_low = (samples >= low - t) & (samples <= low + t)
    covered = near_low.any(dim=-1)
    if second_band:
        near_high = (samples >= high - t) & (samples <= high + t)
        covered = covered & near_high.any(dim=-1)
    return float(covered.float().mean().detach())


def mode_precision_from_samples(
    samples: torch.Tensor,
    modes_low: torch.Tensor,
    modes_high: torch.Tensor,
    tol: torch.Tensor,
    second_band: bool = True,
) -> float:
    low, high, t = _column(modes_low), _column(modes_high), _column(tol)
    near_low = (samples >= low - t) & (samples <= low + t)
    near_mode = near_low
    if second_band:
        near_high = (samples >= high - t) & (samples <= high + t)
        near_mode = near_low | near_high
    return float(near_mode.float().mean().detach())


def known_forward_cycle_error_quadratic(
    states: torch.Tensor,
    candidate_actions: torch.Tensor,
    next_states_clean: torch.Tensor,
) -> float:
    """|F(s, a) - s'| with F(s, a) = s + a^2; candidates shape (n, k)."""

    predicted = states.unsqueeze(-1) + candidate_actions**2
    target = next_states_clean.unsqueeze(-1)
    return float((predicted - target).abs().mean().detach())
