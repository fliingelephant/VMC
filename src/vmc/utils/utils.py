"""General-purpose utility functions."""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax.typing import DTypeLike

__all__ = [
    "occupancy_to_spin",
    "random_tensor",
    "spin_to_occupancy",
    "_metropolis_hastings_accept",
    "_tree_add_scaled",
]


def random_tensor(
    rngs,
    shape: tuple[int, ...],
    dtype: "DTypeLike",
) -> jax.Array:
    """Create a random tensor with proper complex dtype handling.

    Args:
        rngs: Flax NNX random key generator.
        shape: Shape of the tensor.
        dtype: Target dtype (can be real or complex).

    Returns:
        Random tensor with the specified shape and dtype.
    """
    dtype = jnp.dtype(dtype)
    is_complex = jnp.issubdtype(dtype, jnp.complexfloating)
    if is_complex:
        real_dtype = jnp.real(jnp.zeros((), dtype=dtype)).dtype
        complex_unit = jnp.array(1j, dtype=dtype)
        key_re, key_im = rngs.params(), rngs.params()
        return 0.5 * (
            jax.random.uniform(key_re, shape, dtype=real_dtype)
            + complex_unit * jax.random.uniform(key_im, shape, dtype=real_dtype)
        )
    return jax.random.uniform(rngs.params(), shape, dtype=dtype)


def occupancy_to_spin(occupancies: jax.Array) -> jax.Array:
    """Convert 0/1 occupancy variables into -1/+1 spins."""
    return (2 * occupancies - 1).astype(jnp.int32)


def spin_to_occupancy(spins: jax.Array) -> jax.Array:
    """Convert -1/+1 spins into 0/1 occupancy variables."""
    return ((spins + 1) // 2).astype(jnp.int32)


def _tree_add_scaled(base, delta, scale: float):
    return jax.tree.map(
        lambda x, y: x + jnp.asarray(scale, dtype=y.dtype) * y,
        base,
        delta,
    )

def _metropolis_ratio(current_prob: jax.Array, proposed_prob: jax.Array) -> jax.Array:
    """Return Metropolis ratio p(x') / p(x) with proper handling of zero probabilities."""
    return jnp.where(
        current_prob > 0.0,
        proposed_prob / current_prob,
        jnp.where(proposed_prob > 0.0, jnp.inf, 0.0),
    )

def _hastings_ratio(
    forward_prob: jax.Array,
    backward_prob: jax.Array,
) -> jax.Array:
    """Return q(x|x') / q(x'|x) with proper handling of zero probabilities."""
    return jnp.where(
        forward_prob > 0.0,
        backward_prob / forward_prob,
        jnp.where(backward_prob > 0.0, jnp.inf, 0.0),
    )

def _metropolis_hastings_accept(
    key: jax.Array,
    current_prob: jax.Array,
    proposed_prob: jax.Array,
    *,
    proposal_ratio: jax.Array | float = 1.0,  # q(x|x') / q(x'|x)
) -> tuple[jax.Array, jax.Array]:
    key, accept_key = jax.random.split(key)
    return key, jax.random.uniform(accept_key) < jnp.minimum(1.0, _metropolis_ratio(current_prob, proposed_prob) * proposal_ratio)
