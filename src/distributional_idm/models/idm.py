"""Action-head models for Phase 0A.

Every model consumes a feature vector built by the caller (endpoint pairs,
displacements, or state-only inputs) and produces parameters for one of the
registered objectives.
"""

from __future__ import annotations

import torch
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
    ) -> None:
        super().__init__()
        self.variance_floor = variance_floor
        self.net = _mlp(input_dim, hidden_dim, 2, depth)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(features)
        return out[..., 0], out[..., 1]

    def loss(self, batch: dict[str, torch.Tensor], **_: object) -> torch.Tensor:
        from distributional_idm.objectives.losses import gaussian_nll_loss

        mean, raw_log_var = self(batch["features"])
        return gaussian_nll_loss(mean, raw_log_var, batch["actions"], self.variance_floor)

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

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        logits, means, raw_log_var = self(features)
        log_var = raw_log_var.clamp_min(torch.log(torch.tensor(self.variance_floor)))
        return {
            "logits": logits,
            "means": means,
            "stds": log_var.exp().sqrt(),
        }
