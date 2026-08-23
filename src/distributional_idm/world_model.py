"""Minimal JEPA-style world model backbone and Phase 1A training arms.

Backbone: online encoder f(s) -> z, EMA target encoder, forward predictor
P(z_t, a_t) -> z_{t+1}. World-model loss is latent MSE against detached
target encodings. Training uses a fixed update budget with no early
stopping because validation world-model loss rewards collapse.

Gradient routing (registered): action heads read ONLINE representations and
both endpoints stay attached, so IDM gradients reach the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from distributional_idm.models.flow import FlowIDM
from distributional_idm.models.idm import (
    MDNIDM,
    DeterministicIDM,
    GaussianIDM,
    SigmaOnlyGaussianIDM,
)
from distributional_idm.objectives.losses import (
    gaussian_nll_loss,
    mdn_nll_loss,
    mse_loss,
)

LATENT_DIM = 16


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(current, hidden_dim))
        layers.append(nn.SiLU())
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, state_dim: int = 1, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.net = _mlp(state_dim, 64, latent_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() == 1:
            states = states.unsqueeze(-1)
        return self.net(states)


class ForwardPredictor(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.net = _mlp(latent_dim + 1, 64, latent_dim)

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, actions.unsqueeze(-1)], dim=-1))


@dataclass
class ArmConfig:
    name: str
    idm_kind: str | None = None
    idm_conditioning: str = "endpoints"
    aux_task: bool = False
    coverage: bool = False
    cycle: bool = False
    cycle_weight: float = 1.0
    idm_weight: float = 1.0
    coverage_weight: float = 1.0


def vicreg_coverage(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance hinge and off-diagonal covariance penalty, per-dim normalized."""

    centered = latents - latents.mean(dim=0)
    std = centered.std(dim=0)
    variance_hinge = torch.relu(1.0 - std).mean()
    cov = (centered.T @ centered) / max(1, latents.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    covariance_penalty = (off_diag**2).sum() / latents.shape[1]
    return variance_hinge, covariance_penalty


class RandomFeatureTarget:
    """Fixed random linear map of the state; non-action auxiliary target."""

    def __init__(self, state_dim: int = 1, output_dim: int = 8, seed: int = 1234) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(state_dim, output_dim, generator=generator)
        self.bias = torch.randn(output_dim, generator=generator)

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() == 1:
            states = states.unsqueeze(-1)
        return states @ self.weight.to(states.device) + self.bias.to(states.device)


class WorldModel(nn.Module):
    """Online encoder, forward predictor, and optional heads for one arm."""

    def __init__(self, config: ArmConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder()
        self.predictor = ForwardPredictor()
        if config.idm_kind == "deterministic":
            self.idm: nn.Module | None = DeterministicIDM(2 * LATENT_DIM)
        elif config.idm_kind == "gaussian":
            self.idm = GaussianIDM(2 * LATENT_DIM)
        elif config.idm_kind == "mdn":
            self.idm = MDNIDM(2 * LATENT_DIM, n_components=2)
        elif config.idm_kind == "flow":
            self.idm = FlowIDM(2 * LATENT_DIM, n_bins=24)
        elif config.idm_kind == "sigma_gaussian":
            self.idm = SigmaOnlyGaussianIDM(2 * LATENT_DIM)
        elif config.idm_kind is None:
            self.idm = None
        else:
            raise ValueError(config.idm_kind)
        if config.aux_task:
            self.aux_head = _mlp(LATENT_DIM, 64, 8)
        else:
            self.aux_head = None

    def idm_features(
        self,
        batch_actions: torch.Tensor,
        batch_states: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """Conditioning features for the attached action head.

        endpoints: [f(s_t), f(s_{t+1})] - gradients reach the encoder only.
        predictor: [f(s_t), P(f(s_t), a_t)] - the head trains on the
        planner's own input distribution; gradients reach encoder and
        predictor (prereg amendment 4 ablation).
        """

        z_t = self.encoder(batch_states)
        if self.config.idm_conditioning == "predictor":
            z_next = self.predictor(z_t, batch_actions)
        elif self.config.idm_conditioning == "endpoints":
            z_next = self.encoder(next_states)
        else:
            raise ValueError(self.config.idm_conditioning)
        return torch.cat([z_t, z_next], dim=-1)

    def idm_loss(
        self,
        batch_actions: torch.Tensor,
        batch_states: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Both endpoint occurrences encoded online and attached."""

        if self.idm is None:
            return None
        features = self.idm_features(batch_actions, batch_states, next_states)
        if isinstance(self.idm, DeterministicIDM):
            return mse_loss(self.idm(features), batch_actions)
        if isinstance(self.idm, SigmaOnlyGaussianIDM):
            return self.idm.loss({"features": features, "actions": batch_actions})
        if isinstance(self.idm, GaussianIDM):
            mean, raw_log_var = self.idm(features)
            return gaussian_nll_loss(mean, raw_log_var, batch_actions, 1e-4)
        if isinstance(self.idm, FlowIDM):
            return self.idm.flow_nll_tensor(features, batch_actions).mean()
        logits, means, raw_log_var = self.idm(features)
        return mdn_nll_loss(logits, means, raw_log_var, batch_actions, 1e-3)

    def aux_loss(
        self,
        batch_states: torch.Tensor,
        feature_target: RandomFeatureTarget,
    ) -> torch.Tensor | None:
        if self.aux_head is None:
            return None
        z = self.encoder(batch_states)
        return mse_loss(self.aux_head(z), feature_target(batch_states))

    def coverage_loss(self, batch_states: torch.Tensor) -> torch.Tensor | None:
        if not self.config.coverage:
            return None
        variance_hinge, covariance_penalty = vicreg_coverage(self.encoder(batch_states))
        return variance_hinge + covariance_penalty

    def cycle_loss(
        self,
        batch_states: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_next_states: torch.Tensor,
        z_target_next: torch.Tensor,
        n_samples: int = 4,
    ) -> torch.Tensor | None:
        """Reparameterized samples from the head are pushed through the
        forward predictor and regressed toward the detached target encoding
        of the actual next state (prereg amendment 7)."""

        if self.idm is None or not self.config.cycle:
            return None
        features = self.idm_features(batch_actions, batch_states, batch_next_states)
        sampled = self.idm.sample_reparam(features, n_samples)
        z_t = features[:, :LATENT_DIM]
        total = 0.0
        for k in range(n_samples):
            z_hat = self.predictor(z_t, sampled[:, k])
            total = total + ((z_hat - z_target_next) ** 2).sum(-1).mean()
        return total / n_samples
