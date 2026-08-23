"""Phase 1A training: backbone updates, EMA targets, arm losses.

No early stopping and no validation-based stopping: validation world-model
loss decreases under collapse, so stopping on it favors the failure the
no-IDM arm is supposed to exhibit. Every arm gets an identical fixed budget.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from distributional_idm.world_model import RandomFeatureTarget, WorldModel


@dataclass
class BackboneConfig:
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 1e-3
    ema_momentum: float = 0.99
    max_grad_norm: float = 1.0
    restarts: int = 3
    device: str = "cpu"


def train_backbone_with_restarts(
    arm_config,
    tensors_train: dict[str, torch.Tensor],
    tensors_val: dict[str, torch.Tensor],
    config: BackboneConfig,
    seed: int,
    feature_target: RandomFeatureTarget | None = None,
) -> WorldModel:
    """Train several backbone initializations and keep the one with the best
    validation latent-planning success (amendment 6). Validation planning is
    used because validation world-model loss rewards collapse.
    """

    from distributional_idm.evaluation.planning import latent_planning_success
    from distributional_idm.world_model import ArmConfig

    assert isinstance(arm_config, ArmConfig)
    best_model = None
    best_score = -float("inf")
    for restart in range(config.restarts):
        torch.manual_seed(seed)
        candidate = WorldModel(arm_config)
        train_backbone(
            candidate,
            tensors_train,
            config,
            seed * 1000 + restart,
            feature_target if arm_config.aux_task else None,
        )
        candidate.eval()
        score = latent_planning_success(
            candidate.encoder,
            candidate.predictor,
            tensors_val["states"],
            tensors_val["next_clean"],
        )["planning_success"]
        if score > best_score:
            best_score = score
            best_model = candidate
    return best_model


def train_backbone(
    world_model: WorldModel,
    tensors: dict[str, torch.Tensor],
    config: BackboneConfig,
    seed: int,
    feature_target: RandomFeatureTarget | None = None,
) -> WorldModel:
    torch.manual_seed(seed)
    target_encoder = copy.deepcopy(world_model.encoder)
    for param in target_encoder.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.Adam(
        list(world_model.encoder.parameters())
        + list(world_model.predictor.parameters())
        + list(world_model.idm.parameters() if world_model.idm is not None else [])
        + list(world_model.aux_head.parameters() if world_model.aux_head is not None else []),
        lr=config.learning_rate,
    )
    n = tensors["states"].shape[0]
    generator = torch.Generator().manual_seed(seed)

    for _ in range(config.epochs):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, config.batch_size):
            index = permutation[start : start + config.batch_size]
            states = tensors["states"][index]
            actions = tensors["actions"][index]
            next_states = tensors["next_clean"][index]

            z = world_model.encoder(states)
            with torch.no_grad():
                z_target = target_encoder(next_states)
            wm_loss = ((world_model.predictor(z, actions) - z_target) ** 2).mean()

            total = wm_loss
            idm_loss = world_model.idm_loss(actions, states, next_states)
            if idm_loss is not None:
                total = total + world_model.config.idm_weight * idm_loss
            aux_loss = (
                world_model.aux_loss(states, feature_target)
                if feature_target is not None
                else None
            )
            if aux_loss is not None:
                total = total + world_model.config.idm_weight * aux_loss
            coverage_loss = world_model.coverage_loss(states)
            if coverage_loss is not None:
                total = total + world_model.config.coverage_weight * coverage_loss
            cycle_loss = world_model.cycle_loss(states, actions, next_states, z_target)
            if cycle_loss is not None:
                total = total + world_model.config.cycle_weight * cycle_loss

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                config.max_grad_norm,
            )
            optimizer.step()
            with torch.no_grad():
                for online, target in zip(
                    world_model.encoder.parameters(), target_encoder.parameters()
                ):
                    target.mul_(config.ema_momentum).add_(
                        online.detach(), alpha=1 - config.ema_momentum
                    )
    return world_model


def train_frozen_head(
    head_template: nn.Module,
    encoder: nn.Module,
    tensors: dict[str, torch.Tensor],
    config: BackboneConfig,
    seed: int,
) -> nn.Module:
    """Train only an action head on detached latents of a frozen checkpoint."""

    torch.manual_seed(seed)
    head = copy.deepcopy(head_template)
    optimizer = torch.optim.Adam(head.parameters(), lr=config.learning_rate)
    n = tensors["states"].shape[0]
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        features = torch.cat(
            [encoder(tensors["states"]), encoder(tensors["next_clean"])], dim=-1
        )
    actions = tensors["actions"]

    for _ in range(config.epochs):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, config.batch_size):
            index = permutation[start : start + config.batch_size]
            batch = {"features": features[index], "actions": actions[index]}
            loss = head.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return head
