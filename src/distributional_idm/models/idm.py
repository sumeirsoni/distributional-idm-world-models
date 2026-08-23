"""Action-head models for Phase 0A.

Every model consumes a feature vector built by the caller (endpoint pairs,
displacements, or state-only inputs) and produces parameters for one of the
registered objectives.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(current, hidden_dim))
        layers.append(nn.SiLU())
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class DeterministicIDM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, depth: int = 2) -> None:
        super().__init__()
        self.net = _mlp(input_dim, hidden_dim, 1, depth)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    def loss(self, batch: dict[str, torch.Tensor], **_: object) -> torch.Tensor:
        from distributional_idm.objectives.losses import mse_loss

        return mse_loss(self(batch["features"]), batch["actions"])

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"mean": self(features)}


class GaussianIDM(nn.Module):
    """Heteroscedastic diagonal Gaussian with a variance floor."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        depth: int = 2,
        variance_floor: float = 1e-4,
        loss_variant: str = "nll",
    ) -> None:
        super().__init__()
        self.variance_floor = variance_floor
        self.loss_variant = loss_variant
        self.net = _mlp(input_dim, hidden_dim, 2, depth)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(features)
        return out[..., 0], out[..., 1]

    def loss(self, batch: dict[str, torch.Tensor], **_: object) -> torch.Tensor:
        from distributional_idm.objectives.losses import gaussian_nll_loss

        mean, raw_log_var = self(batch["features"])
        return gaussian_nll_loss(mean, raw_log_var, batch["actions"], self.variance_floor)

    def sample_reparam(self, features: torch.Tensor, n_samples: int, generator=None) -> torch.Tensor:
        mean, raw_log_var = self(features)
        log_var = raw_log_var.clamp_min(torch.log(torch.tensor(self.variance_floor)))
        return _reparameterized_gaussian_sample(mean, log_var.exp().sqrt(), n_samples, generator)

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        mean, raw_log_var = self(features)
        log_var = raw_log_var.clamp_min(torch.log(torch.tensor(self.variance_floor)))
        return {"mean": mean, "std": log_var.exp().sqrt()}


class FixedVarianceGaussianIDM(nn.Module):
    """Scalar fixed-variance Gaussian; its optimum and gradient direction
    match globally scaled MSE."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, depth: int = 2, std: float = 0.5) -> None:
        super().__init__()
        self.std = std
        self.net = _mlp(input_dim, hidden_dim, 1, depth)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    def loss(self, batch: dict[str, torch.Tensor], **_: object) -> torch.Tensor:
        from distributional_idm.objectives.losses import fixed_gaussian_nll_loss

        return fixed_gaussian_nll_loss(self(batch["features"]), batch["actions"], self.std)

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"mean": self(features), "std": torch.full_like(self(features), self.std)}


def _reparameterized_gaussian_sample(
    mean: torch.Tensor, std: torch.Tensor, n_samples: int, generator=None
) -> torch.Tensor:
    noise = torch.randn(
        mean.shape[0], n_samples, device=mean.device, dtype=mean.dtype, generator=generator
    )
    return mean.unsqueeze(-1) + std.unsqueeze(-1) * noise


class MDNIDM(nn.Module):
    """Mixture density network over one-dimensional actions."""

    def __init__(
        self,
        input_dim: int,
        n_components: int = 4,
        hidden_dim: int = 64,
        depth: int = 2,
        variance_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.variance_floor = variance_floor
        self.trunk = _mlp(input_dim, hidden_dim, hidden_dim, depth)
        self.head = nn.Linear(hidden_dim, 3 * n_components)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        out = self.head(hidden)
        k = self.n_components
        logits = out[..., :k]
        means = out[..., k : 2 * k]
        raw_log_var = out[..., 2 * k :]
        return logits, means, raw_log_var

    def loss(self, batch: dict[str, torch.Tensor], **_: object) -> torch.Tensor:
        from distributional_idm.objectives.losses import mdn_nll_loss

        logits, means, raw_log_var = self(batch["features"])
        return mdn_nll_loss(logits, means, raw_log_var, batch["actions"], self.variance_floor)

    def sample_reparam(self, features: torch.Tensor, n_samples: int, generator=None) -> torch.Tensor:
        logits, means, raw_log_var = self(features)
        log_var = raw_log_var.clamp_min(torch.log(torch.tensor(self.variance_floor)))
        stds = log_var.exp().sqrt()
        n = features.shape[0]
        gumbel = F.gumbel_softmax(
            logits.unsqueeze(1).expand(-1, n_samples, -1), tau=1.0, hard=True, dim=-1
        )
        eps = torch.randn(
            n, n_samples, device=features.device, dtype=features.dtype, generator=generator
        )
        comp_mean = (gumbel * means.unsqueeze(1)).sum(-1)
        comp_std = (gumbel * stds.unsqueeze(1)).sum(-1)
        return comp_mean + comp_std * eps

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        logits, means, raw_log_var = self(features)
        log_var = raw_log_var.clamp_min(torch.log(torch.tensor(self.variance_floor)))
        return {
            "logits": logits,
            "means": means,
            "stds": log_var.exp().sqrt(),
        }


class SigmaOnlyGaussianIDM(nn.Module):
    """Heteroscedastic NLL with the mean pinned to zero: isolates the
    uncertainty channel of the regularizer (prereg amendment 8)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, depth: int = 2, variance_floor: float = 1e-4):
        super().__init__()
        self.variance_floor = variance_floor
        self.net = _mlp(input_dim, hidden_dim, 1, depth)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    def loss(self, batch: dict[str, torch.Tensor], **_: object) -> torch.Tensor:
        log_var = self(batch["features"]).clamp_min(math.log(self.variance_floor))
        return (0.5 * batch["actions"] ** 2 / log_var.exp() + log_var).mean()

    def sample_reparam(self, features: torch.Tensor, n_samples: int, generator=None) -> torch.Tensor:
        noise = torch.randn(
            features.shape[0], n_samples, device=features.device, dtype=features.dtype,
            generator=generator,
        )
        std = self(features).clamp_min(math.log(self.variance_floor)).exp().sqrt()
        return std.unsqueeze(-1) * noise

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self(features)
        std = raw.clamp_min(math.log(self.variance_floor)).exp().sqrt()
        return {"mean": torch.zeros_like(std), "std": std}
