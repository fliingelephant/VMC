"""Weight-based null vector for sample-space QGT."""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from vmc import config  # noqa: F401

__all__ = ["WeightConfig", "compute_weight_projection"]


@dataclass(frozen=True)
class WeightConfig:
    """Configuration for weight-based null removal."""
    pass


def compute_weight_projection(cfg: WeightConfig, samples, sampler=None):
    """Compute the null vector for sample-space QGT projection.

    Returns:
        null: (n_samples,) all-ones vector (default).
    """
    n_samples = samples.shape[0]
    return jnp.ones(n_samples, dtype=jnp.complex128)
