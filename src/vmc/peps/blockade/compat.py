"""Compatibility surfaces for blockade PEPS."""
from __future__ import annotations

from typing import Any

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp

from vmc.peps.common.contraction import _contract_bottom

__all__ = ["blockade_apply"]


def _peps_apply_occupancy(
    tensors: list[list[jax.Array]],
    config: jax.Array,
    shape: tuple[int, int],
    strategy: Any,
) -> jax.Array:
    """Compute PEPS amplitude for occupancy configuration."""
    n_rows, n_cols = shape
    boundary = tuple(
        jnp.ones((1, 1, 1), dtype=jnp.asarray(tensors[0][0]).dtype)
        for _ in range(n_cols)
    )
    for row in range(n_rows):
        mpo = tuple(
            jnp.transpose(tensors[row][c][config[row, c]], (2, 3, 0, 1))
            for c in range(n_cols)
        )
        boundary = strategy.apply(boundary, mpo)
    return _contract_bottom(boundary)


def blockade_apply(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    shape: tuple[int, int],
    config: Any,
    strategy: Any,
) -> jax.Array:
    """Compute blockade-PEPS amplitude for a given sample."""
    n_config = sample.reshape(shape)
    invalid_h = jnp.any(n_config[:, 1:] * n_config[:, :-1])
    invalid_v = jnp.any(n_config[1:, :] * n_config[:-1, :])
    invalid = invalid_h | invalid_v
    dtype = jnp.asarray(tensors[0][0]).dtype

    def _compute_amp(_):
        from vmc.peps.blockade.model import assemble_tensors

        eff_tensors = assemble_tensors(tensors, n_config, config)
        return _peps_apply_occupancy(eff_tensors, n_config, shape, strategy)

    return jax.lax.cond(
        invalid,
        lambda _: jnp.zeros((), dtype=dtype),
        _compute_amp,
        operand=None,
    )
