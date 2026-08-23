"""Quadratic-ambiguity environment (E1) and its one-to-one control (E10).

Transition: s_{t+1} = s_t + a_t^2 plus optional Gaussian transition noise;
observations report s_{t+1} plus optional Gaussian observation noise.

With support {+-m}, the conditional law p(a | displacement) is two-point, so
two noise-free conditionals sharing a mean are identical. Mean-matched pairs
with different modal structure therefore require observation noise; see
``make_same_mean_pair``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuadraticConfig:
    name: str = "e1_symmetric"
    magnitudes: tuple[float, ...] = (0.5, 1.0)
    positive_probabilities: tuple[float, ...] | None = None
    forced_positive_magnitudes: tuple[float, ...] = ()
    observation_noise_std: float = 0.0
    transition_noise_std: float = 0.0
    n_train_episodes: int = 500
    n_val_episodes: int = 50
    n_test_episodes: int = 50
    episode_length: int = 200
    start_range: float = 5.0
    nonnegative_actions: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.positive_probabilities is None:
            object.__setattr__(
                self, "positive_probabilities", (0.5,) * len(self.magnitudes)
            )
        if len(self.positive_probabilities) != len(self.magnitudes):
            raise ValueError("need one sign probability per magnitude")
        if any(p < 0.0 or p > 1.0 for p in self.positive_probabilities):
            raise ValueError("sign probabilities must lie in [0, 1]")
        if not self.magnitudes:
            raise ValueError("need at least one magnitude")


@dataclass(frozen=True)
class TransitionDataset:
    states: np.ndarray
    actions: np.ndarray
    next_states_observed: np.ndarray
    next_states_clean: np.ndarray
    episode_ids: np.ndarray
    meta: dict

    def __len__(self) -> int:
        return self.states.shape[0]

    def episode_split(self, episode_ids: set[int]) -> TransitionDataset:
        mask = np.isin(self.episode_ids, list(episode_ids))
        return TransitionDataset(
            states=self.states[mask],
            actions=self.actions[mask],
            next_states_observed=self.next_states_observed[mask],
            next_states_clean=self.next_states_clean[mask],
            episode_ids=self.episode_ids[mask],
            meta=self.meta,
        )


def generate_dataset(config: QuadraticConfig) -> dict[str, TransitionDataset]:
    rng = np.random.default_rng(config.seed)
    chunks: dict[str, TransitionDataset] = {}
    offset = 0
    for split_name, n_episodes in (
        ("train", config.n_train_episodes),
        ("val", config.n_val_episodes),
        ("test", config.n_test_episodes),
    ):
        chunks[split_name] = _generate_split(config, rng, split_name, offset, n_episodes)
        offset += n_episodes
    return chunks


def _generate_split(
    config: QuadraticConfig,
    rng: np.random.Generator,
    split_name: str,
    episode_offset: int,
    n_episodes: int,
) -> TransitionDataset:
    if n_episodes == 0:
        empty = np.empty(0, dtype=np.float64)
        ids_empty = np.empty(0, dtype=np.int64)
        return TransitionDataset(
            states=empty,
            actions=empty,
            next_states_observed=empty,
            next_states_clean=empty,
            episode_ids=ids_empty,
            meta={"env": "quadratic", "split": split_name, **vars(config)},
        )

    magnitudes = np.asarray(config.magnitudes, dtype=np.float64)
    sign_probs = np.asarray(config.positive_probabilities, dtype=np.float64)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_observed: list[np.ndarray] = []
    next_clean: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []

    for episode in range(n_episodes):
        s = rng.uniform(-config.start_range, config.start_range)
        s_traj = np.empty(config.episode_length)
        a_traj = np.empty(config.episode_length)
        sn_obs = np.empty(config.episode_length)
        sn_clean = np.empty(config.episode_length)
        for t in range(config.episode_length):
            mag_idx = int(rng.integers(len(magnitudes)))
            magnitude = magnitudes[mag_idx]
            if config.nonnegative_actions or magnitude in config.forced_positive_magnitudes:
                a = magnitude
            else:
                sign = 1.0 if rng.uniform() < sign_probs[mag_idx] else -1.0
                a = sign * magnitude
            s_next = s + a**2
            if config.transition_noise_std > 0.0:
                s_next += rng.normal(0.0, config.transition_noise_std)
            s_obs = s_next
            if config.observation_noise_std > 0.0:
                s_obs += rng.normal(0.0, config.observation_noise_std)
            s_traj[t] = s
            a_traj[t] = a
            sn_obs[t] = s_obs
            sn_clean[t] = s_next
            s = s_next

        ids = np.full(config.episode_length, episode_offset + episode, dtype=np.int64)
        states.append(s_traj)
        actions.append(a_traj)
        next_observed.append(sn_obs)
        next_clean.append(sn_clean)
        episode_ids.append(ids)

    return TransitionDataset(
        states=np.concatenate(states),
        actions=np.concatenate(actions),
        next_states_observed=np.concatenate(next_observed),
        next_states_clean=np.concatenate(next_clean),
        episode_ids=np.concatenate(episode_ids),
        meta={"env": "quadratic", "split": split_name, **vars(config)},
    )


def conditional_law_table(
    dataset: TransitionDataset,
    displacement_rtol: float = 1e-6,
) -> dict[float, dict[str, float]]:
    """Empirical conditional law of the action given the clean displacement."""

    displacement = dataset.next_states_clean - dataset.states
    table: dict[float, dict[str, float]] = {}
    for level in np.unique(np.round(displacement, 9)):
        mask = np.abs(displacement - level) <= displacement_rtol * max(1.0, abs(level))
        subset = dataset.actions[mask]
        positive = subset[subset > 0]
        table[float(level)] = {
            "mean": float(subset.mean()),
            "std": float(subset.std()),
            "n": int(mask.sum()),
            "mean_positive": float(positive.mean()) if positive.size else float("nan"),
            "weight_positive": float(positive.size / subset.size),
        }
    return table


def make_same_mean_pair(
    noise_std: float,
    base_weight: float = 0.7,
    displacement_probe: float = 1.0,
    mean_tolerance: float = 1e-3,
    seed: int = 0,
) -> tuple[QuadraticConfig, QuadraticConfig, dict]:
    """Two configs whose observation-noise conditionals share a mean at the
    probe displacement while their modal structure differs: one supports a
    single magnitude pair, the other two."""

    if noise_std <= 0.0:
        raise ValueError("mean-matched structures require observation noise")

    def posterior_mean(weights: tuple[float, ...], mags: tuple[float, ...]) -> float:
        mags_arr = np.asarray(mags)
        weights_arr = np.asarray(weights)
        squared = mags_arr**2
        lik = np.exp(-0.5 * ((displacement_probe - squared) / noise_std) ** 2)
        lik /= lik.sum()
        signed_mean = (2.0 * weights_arr - 1.0) * mags_arr
        return float((lik * signed_mean).sum())

    def posterior_var(weights: tuple[float, ...], mags: tuple[float, ...]) -> float:
        mags_arr = np.asarray(mags)
        weights_arr = np.asarray(weights)
        squared = mags_arr**2
        lik = np.exp(-0.5 * ((displacement_probe - squared) / noise_std) ** 2)
        lik /= lik.sum()
        points = np.array([(2.0 * w - 1.0) * m for w, m in zip(weights_arr, mags_arr)])
        mean = float(points.mean())
        return float(((points - mean) ** 2).sum() / len(points))

    mags_b = (0.5, 1.0)
    best: tuple[float, float] | None = None
    best_gap = -np.inf
    target_mean = posterior_mean((base_weight,), (1.0,))
    for w0 in np.linspace(0.02, 0.98, 49):
        for w1 in np.linspace(0.02, 0.98, 49):
            mean_b = posterior_mean((float(w0), float(w1)), mags_b)
            variance_gap = posterior_var((float(w0), float(w1)), mags_b) - posterior_var(
                (base_weight,), (1.0,)
            )
            if abs(mean_b - target_mean) <= mean_tolerance and variance_gap > best_gap:
                best_gap = variance_gap
                best = (float(w0), float(w1))
    if best is None:
        raise RuntimeError("no mean-matched configuration found; widen the grid")

    cfg_a = QuadraticConfig(
        name="same_mean_single",
        magnitudes=(1.0,),
        positive_probabilities=(base_weight,),
        observation_noise_std=noise_std,
        seed=seed,
    )
    cfg_b = QuadraticConfig(
        name="same_mean_multi",
        magnitudes=mags_b,
        positive_probabilities=best,
        observation_noise_std=noise_std,
        seed=seed + 1,
    )
    diagnostics = {
        "probe_displacement": displacement_probe,
        "noise_std": noise_std,
        "mean_a": posterior_mean((base_weight,), (1.0,)),
        "mean_b": posterior_mean(best, mags_b),
        "variance_a": posterior_var((base_weight,), (1.0,)),
        "variance_b": posterior_var(best, mags_b),
        "weights_b": list(best),
    }
    return cfg_a, cfg_b, diagnostics
