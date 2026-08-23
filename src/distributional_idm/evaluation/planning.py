"""One-step latent planning and post-hoc cycle-validity evaluation.

Latent planner: encode s_t, score candidate actions by latent distance of
the learned predictor to the encoded next state, execute with the true
transition, and count success when the achieved displacement matches the
desired one within tolerance. The representation is in the loop; the true
transition only executes the selected action. Goals are one-step because a
one-step predictor cannot reach multi-step displacements (|a^2| <= 1 while
drift accumulates).
"""

from __future__ import annotations

import torch

ACTION_GRID = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
               0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SUCCESS_TOLERANCE = 0.05


@torch.no_grad()
def latent_planning_success(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    states: torch.Tensor,
    goals: torch.Tensor,
    tolerance: float = SUCCESS_TOLERANCE,
) -> dict[str, float]:
    """Select actions in latent space; score displacement error under true F."""

    encoder.eval()
    predictor.eval()
    candidates = torch.tensor(ACTION_GRID, device=states.device).unsqueeze(0)
    z_t = encoder(states)
    z_g = encoder(goals).detach()

    repeated_z = z_t.unsqueeze(1).expand(-1, len(ACTION_GRID), -1)
    predicted = predictor(repeated_z.reshape(-1, repeated_z.shape[-1]),
                          candidates.expand(len(states), -1).reshape(-1))
    predicted = predicted.reshape(len(states), len(ACTION_GRID), -1)
    distances = ((predicted - z_g.unsqueeze(1)) ** 2).sum(-1)
    chosen = candidates.expand(len(states), -1).gather(1, distances.argmin(dim=1, keepdim=True)).squeeze(-1)

    desired = goals - states
    achieved = chosen**2
    errors = (achieved - desired).abs()
    success = (errors <= tolerance).float().mean()
    result = {
        "planning_success": float(success),
        "mean_displacement_error": float(errors.mean()),
        "chosen_action_abs_mean": float(chosen.abs().mean()),
    }
    for level in torch.unique(desired.round(decimals=3)):
        mask = torch.isclose(desired, level, atol=0.01)
        if int(mask.sum()) >= 20:
            key = f"planning_success_disp_{float(level):.3f}"
            result[key] = float((errors[mask] <= tolerance).float().mean())
    return result


@torch.no_grad()
def posthoc_cycle_validity(
    head: torch.nn.Module,
    encoder: torch.nn.Module,
    states: torch.Tensor,
    next_states_clean: torch.Tensor,
    n_samples: int,
    generator: torch.Generator | None = None,
) -> float:
    """Known-forward cycle error of actions sampled from a post-hoc head
    trained identically on every arm's frozen latents."""

    from distributional_idm.evaluation.metrics import sample_mdn

    encoder.eval()
    features = torch.cat([encoder(states), encoder(next_states_clean)], dim=-1)
    if hasattr(head, "sample"):
        actions = head.sample(features, n_samples, generator)
    else:
        outputs = head.predict(features)
        if "logits" in outputs:
            actions = sample_mdn(outputs["logits"], outputs["means"], outputs["stds"],
                                 n_samples, generator)
        else:
            mean = outputs["mean"]
            std = outputs.get("std")
            if std is not None:
                noise = torch.randn(len(mean), n_samples, generator=generator)
                actions = mean.unsqueeze(-1) + std.unsqueeze(-1) * noise
            else:
                actions = mean.unsqueeze(-1).expand(-1, n_samples)
    predicted = states.unsqueeze(-1) + actions**2
    target = next_states_clean.unsqueeze(-1)
    return float((predicted - target).abs().mean())


@torch.no_grad()
def posthoc_head_nll(
    head: torch.nn.Module,
    encoder: torch.nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states_clean: torch.Tensor,
) -> float | None:
    """Held-out NLL of the post-hoc head on each arm's latents.

    Deterministic heads define no likelihood; callers receive None.
    """

    from distributional_idm.evaluation.metrics import gaussian_nll, mdn_nll

    encoder.eval()
    features = torch.cat([encoder(states), encoder(next_states_clean)], dim=-1)
    if hasattr(head, "nll"):
        return head.nll(features, actions)
    outputs = head.predict(features)
    if "logits" in outputs:
        return mdn_nll(outputs["logits"], outputs["means"], outputs["stds"], actions)
    std = outputs.get("std")
    if std is None:
        return None
    return gaussian_nll(outputs["mean"], std, actions)
