"""General-purpose utility functions."""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp

__all__ = [
    "occupancy_to_spin",
    "spin_to_occupancy",
]


def occupancy_to_spin(occupancies: jax.Array) -> jax.Array:
    """Convert 0/1 occupancy variables into -1/+1 spins."""
    return (2 * occupancies - 1).astype(jnp.int32)


def spin_to_occupancy(spins: jax.Array) -> jax.Array:
    """Convert -1/+1 spins into 0/1 occupancy variables."""
    return ((spins + 1) // 2).astype(jnp.int32)
