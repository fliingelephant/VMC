"""Compatibility surfaces for blockade PEPS."""
from __future__ import annotations

from typing import Any

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp

from vmc.peps.common.contraction import _contract_bottom

__all__ = ["blockade_apply"]


def blockade_apply(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    peps_config: Any,
    strategy: Any,
) -> jax.Array:
    """Compute blockade-PEPS amplitude for a given sample."""
    from vmc.peps.blockade.model import _build_row_mpo

    shape = peps_config.shape
    n_rows, n_cols = shape
    n_config = sample.reshape(shape)
    invalid_h = jnp.any(n_config[:, 1:] * n_config[:, :-1])
    invalid_v = jnp.any(n_config[1:, :] * n_config[:-1, :])
    invalid = invalid_h | invalid_v
    dtype = jnp.asarray(tensors[0][0]).dtype

    def _compute_amp(_):
        boundary = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
        for row in range(n_rows):
            boundary = strategy.apply(
                boundary,
                _build_row_mpo(tensors, n_config, peps_config, row),
            )
        return _contract_bottom(boundary)

    return jax.lax.cond(
        invalid,
        lambda _: jnp.zeros((), dtype=dtype),
        _compute_amp,
        operand=None,
    )
