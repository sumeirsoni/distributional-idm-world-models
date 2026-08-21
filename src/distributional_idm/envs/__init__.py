"""Controlled synthetic research environments."""

from distributional_idm.envs.quadratic import (
    QuadraticConfig,
    TransitionDataset,
    conditional_law_table,
    generate_dataset,
    make_same_mean_pair,
)

__all__ = [
    "QuadraticConfig",
    "TransitionDataset",
    "conditional_law_table",
    "generate_dataset",
    "make_same_mean_pair",
]
