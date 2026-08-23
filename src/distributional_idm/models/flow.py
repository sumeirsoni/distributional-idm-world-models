"""Conditional monotone-spline flow head over 1-D actions.

A feature-conditioned cubic-Hermite spline on [-B, B] with Fritsch-Carlson-
clamped node slopes (monotonicity guaranteed). Exact log-density via
autograd; inversion via bisection on the forward map. Identity outside.
No mixture bookkeeping anywhere.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

BOUND = 3.0
MIN_SLOPE = 0.05
N_BISECTION_STEPS = 40


def _spline_edges(n_bins: int, device, dtype=torch.float32):
    edges = torch.linspace(-BOUND, BOUND, n_bins + 1, device=device, dtype=dtype)
    return edges


def _build_spline(raw: torch.Tensor, n_bins: int):
    """raw (n, n_bins+1) -> per-row node positions y_edges (n, n_bins+1) and
    node slopes (n, n_bins+1) clamped for monotonicity."""

    heights = F.softplus(raw) + MIN_SLOPE
    heights = heights / heights.sum(-1, keepdim=True) * 2 * BOUND
    y_edges = -BOUND + torch.cumsum(heights, dim=-1)
    y_edges = torch.cat([torch.full_like(y_edges[..., :1], -BOUND), y_edges], dim=-1)
    return y_edges


def _node_slopes(raw: torch.Tensor, y_edges: torch.Tensor, x_edges: torch.Tensor):
    n_bins = x_edges.shape[-1] - 1
    raw_slopes = F.softplus(raw) + MIN_SLOPE
    chords = (y_edges[..., 1:] - y_edges[..., :-1]) / (
        x_edges[..., 1:] - x_edges[..., :-1]
    )
    slopes = []
    for j in range(n_bins + 1):
        cap = torch.full_like(raw_slopes[..., j], float("inf"))
        if j > 0:
            cap = torch.minimum(cap, 3.0 * chords[..., j - 1])
        if j < n_bins:
            cap = torch.minimum(cap, 3.0 * chords[..., j])
        slopes.append(torch.clamp(raw_slopes[..., j], max=cap))
    return torch.stack(slopes, dim=-1)


def _hermite_eval(x: torch.Tensor, y_edges: torch.Tensor, slopes: torch.Tensor,
                  x_edges: torch.Tensor):
    """Evaluate spline and dy/dx at scalar-per-row inputs x (n,)."""

    k = x_edges.shape[-1] - 1
    rows = torch.arange(x.shape[0], device=x.device)
    idx = torch.clip(torch.searchsorted(x_edges, x.unsqueeze(-1), right=True) - 1, 0, k - 1)
    idx = idx.squeeze(-1)

    x0 = x_edges[rows, idx]
    x1 = x_edges[rows, idx + 1]
    w = x1 - x0
    y0 = y_edges[rows, idx]
    y1 = y_edges[rows, idx + 1]
    s0 = slopes[rows, idx]
    s1 = slopes[rows, idx + 1]
    t = (x - x0) / w

    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    y = h00 * y0 + h10 * w * s0 + h01 * y1 + h11 * w * s1

    dy_dt = (
        (6 * t**2 - 6 * t) * y0
        + (3 * t**2 - 4 * t + 1) * w * s0
        + (-6 * t**2 + 6 * t) * y1
        + (3 * t**2 - 2 * t) * w * s1
    )
    dy_dx = dy_dt / w
    return y, dy_dx


def _hermite_invert(y: torch.Tensor, y_edges: torch.Tensor, slopes: torch.Tensor,
                    x_edges: torch.Tensor):
    """Bisection inverse of the monotone spline; y (n,) -> x (n,)."""

    lo = torch.full_like(y, -BOUND)
    hi = torch.full_like(y, BOUND)
    for _ in range(N_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        y_mid, _ = _hermite_eval(mid, y_edges, slopes, x_edges)
        go_right = y_mid < y
        lo = torch.where(go_right, mid, lo)
        hi = torch.where(go_right, hi, mid)
    return 0.5 * (lo + hi)


class MonotoneSplineFlow(nn.Module):
    """Feature-conditioned monotone spline transform u ~ N(0,1) <-> action."""

    def __init__(self, input_dim: int, n_bins: int = 24, hidden_dim: int = 64, depth: int = 2):
        super().__init__()
        self.n_bins = n_bins
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(nn.SiLU())
            current = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.height_head = nn.Linear(hidden_dim, n_bins)
        self.slope_head = nn.Linear(hidden_dim, n_bins + 1)

    def params(self, features: torch.Tensor):
        hidden = self.trunk(features)
        heights_raw = self.height_head(hidden)
        y_edges = _build_spline(heights_raw, self.n_bins)
        x_edges = _spline_edges(self.n_bins, features.device, features.dtype)
        x_edges = x_edges.unsqueeze(0).expand(features.shape[0], -1)
        slopes = _node_slopes(self.slope_head(hidden), y_edges, x_edges)
        return x_edges, y_edges, slopes

    def log_prob(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x_edges, y_edges, slopes = self.params(features)
        y, dydx = _hermite_eval(actions, y_edges, slopes, x_edges)
        inside = ((actions >= -BOUND) & (actions <= BOUND)).float()
        base = -0.5 * y.detach() ** 2 - 0.5 * math.log(2 * math.pi)
        log_det = torch.log(dydx.abs().clamp_min(1e-12))
        outside_base = -0.5 * actions**2 - 0.5 * math.log(2 * math.pi)
        return inside * (base + log_det) + (1 - inside) * outside_base

    def sample(self, features: torch.Tensor, n_samples: int, generator=None) -> torch.Tensor:
        with torch.no_grad():
            x_edges, y_edges, slopes = self.params(features)
            u = torch.randn(
                features.shape[0], n_samples, generator=generator, device=features.device
            )
            flat_u = u.reshape(-1)
            reps = lambda t: t.repeat_interleave(n_samples, dim=0)
            x = _hermite_invert(flat_u, reps(y_edges), reps(slopes), reps(x_edges))
            return x.reshape(features.shape[0], n_samples)

    def sample_reparam(self, features: torch.Tensor, n_samples: int, generator=None) -> torch.Tensor:
        """Differentiable sampling: bisection locates the root, then three
        Newton steps carry gradients through the spline parameters."""

        x_edges, y_edges, slopes = self.params(features)
        with torch.no_grad():
            u = torch.randn(
                features.shape[0], n_samples, generator=generator, device=features.device
            )
            flat_u = u.reshape(-1)
            reps = lambda t: t.repeat_interleave(n_samples, dim=0)
            x = _hermite_invert(flat_u, reps(y_edges), reps(slopes), reps(x_edges))
        flat_edges_x = reps(x_edges)
        flat_edges_y = reps(y_edges)
        flat_slopes = reps(slopes)
        for _ in range(3):
            residual, dydx = _hermite_eval(x, flat_edges_y, flat_slopes, flat_edges_x)
            x = x - (residual - flat_u) / dydx.clamp_min(1e-6)
        return x.reshape(features.shape[0], n_samples)


class FlowIDM(nn.Module):
    """Endpoint-conditioned monotone-flow head: exact NLL, bisection sampling."""

    def __init__(self, input_dim: int, n_bins: int = 24, hidden_dim: int = 64, depth: int = 2):
        super().__init__()
        self.flow = MonotoneSplineFlow(input_dim, n_bins=n_bins, hidden_dim=hidden_dim, depth=depth)

    def loss(self, batch: dict[str, torch.Tensor], **_: object) -> torch.Tensor:
        return -self.flow.log_prob(batch["features"], batch["actions"]).mean()

    def flow_nll_tensor(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return -self.flow.log_prob(features, actions)

    def nll(self, features: torch.Tensor, actions: torch.Tensor) -> float:
        with torch.no_grad():
            return float(-self.flow.log_prob(features, actions).mean().detach())

    def sample_reparam(self, features: torch.Tensor, n_samples: int, generator=None) -> torch.Tensor:
        return self.flow.sample_reparam(features, n_samples, generator)

    @torch.no_grad()
    def sample(self, features: torch.Tensor, n_samples: int, generator=None) -> torch.Tensor:
        return self.flow.sample(features, n_samples, generator)

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        n = features.shape[0]
        grid = torch.linspace(-2.9, 2.9, 257, device=features.device)
        x_edges, y_edges, slopes = self.flow.params(features)
        reps = lambda t: t.repeat_interleave(257, dim=0)
        y, dydx = _hermite_eval(
            grid.repeat(n), reps(y_edges), reps(slopes), reps(x_edges)
        )
        log_prob = (
            -0.5 * y**2 - 0.5 * math.log(2 * math.pi) + torch.log(dydx.abs().clamp_min(1e-12))
        ).reshape(n, -1)
        best = log_prob.argmax(dim=-1)
        mode = grid[best]
        probs = torch.softmax(log_prob, dim=-1)
        mean = probs.mul(grid.unsqueeze(0).expand_as(probs)).sum(dim=-1)
        return {"mean": mean, "mode": mode}
