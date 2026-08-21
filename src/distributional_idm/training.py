"""Feature construction and a compact torch training loop."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from distributional_idm.envs.quadratic import TransitionDataset

CONDITIONINGS = ("endpoint", "displacement", "state_only", "state_displacement")


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> Standardizer:
        return cls(features.mean(axis=0), np.clip(features.std(axis=0), 1e-6, None))

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.std


def build_features(
    dataset: TransitionDataset,
    conditioning: str,
    standardizer: Standardizer | None = None,
    use_observed_next_state: bool = True,
) -> tuple[np.ndarray, Standardizer | None]:
    next_states = (
        dataset.next_states_observed if use_observed_next_state else dataset.next_states_clean
    )
    displacement = next_states - dataset.states
    if conditioning == "endpoint":
        raw = np.stack([dataset.states, next_states], axis=-1)
    elif conditioning == "displacement":
        raw = displacement[:, None]
    elif conditioning == "state_displacement":
        raw = np.stack([dataset.states, displacement], axis=-1)
    elif conditioning == "state_only":
        raw = dataset.states[:, None]
    else:
        raise ValueError(f"unknown conditioning {conditioning}")
    if standardizer is None:
        return raw.astype(np.float32), None
    return standardizer.transform(raw).astype(np.float32), None


def fit_standardizer(dataset: TransitionDataset, conditioning: str) -> Standardizer:
    raw, _ = build_features(dataset, conditioning)
    return Standardizer.fit(raw)


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 1024
    learning_rate: float = 3e-3
    patience: int = 8
    device: str = "cpu"


def tensors_from(
    dataset: TransitionDataset,
    conditioning: str,
    standardizer: Standardizer,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    features, _ = build_features(dataset, conditioning, standardizer)
    return {
        "features": torch.from_numpy(features).to(device),
        "actions": torch.from_numpy(dataset.actions.astype(np.float32)).to(device),
        "states": torch.from_numpy(dataset.states.astype(np.float32)).to(device),
        "next_clean": torch.from_numpy(dataset.next_states_clean.astype(np.float32)).to(device),
    }


def train_model(
    model: nn.Module,
    train_tensors: dict[str, torch.Tensor],
    val_tensors: dict[str, torch.Tensor],
    config: TrainConfig,
    seed: int,
) -> list[float]:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    n = train_tensors["features"].shape[0]
    best_val = math.inf
    best_state: dict | None = None
    history: list[float] = []
    patience_left = config.patience
    generator = torch.Generator().manual_seed(seed)

    for _ in range(config.epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, config.batch_size):
            index = permutation[start : start + config.batch_size]
            batch = {
                key: value[index] for key, value in train_tensors.items() if key in {"features", "actions"}
            }
            loss = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(model.loss(val_tensors))
        history.append(val_loss)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def paired_t_statistic(differences: np.ndarray) -> tuple[float, float]:
    """Mean, t statistic, and two-sided p value from a normal approximation."""

    n = differences.size
    mean = float(differences.mean())
    if n < 2:
        return mean, float("nan"), float("nan")
    sd = float(differences.std(ddof=1))
    if sd == 0.0:
        return mean, float("inf") if mean != 0 else 0.0, 0.0 if mean != 0 else 1.0
    t = mean / (sd / math.sqrt(n))
    from math import erfc, sqrt

    p = float(erfc(abs(t) / sqrt(2)))
    return mean, t, p
