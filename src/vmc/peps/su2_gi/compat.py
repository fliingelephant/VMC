"""Compatibility surfaces for pure-gauge SU(2) GI-PEPS."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from vmc.peps.common.contraction import _contract_bottom
from vmc.peps.su2_gi.contraction import build_row_mpo

__all__ = ["build_row_mpo", "su2_gi_apply"]


def su2_gi_apply(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    shape: tuple[int, int],
    tables: Any,
    strategy: Any,
) -> jax.Array:
    """Compute a pure-gauge SU(2) GI-PEPS amplitude."""
    n_cols = shape[1]
    boundary = tuple(
        jnp.ones((1, 1, 1), dtype=jnp.asarray(tensors[0][0]).dtype)
        for _ in range(n_cols)
    )
    for row in range(shape[0]):
        boundary = strategy.apply(
            boundary,
            build_row_mpo(tensors, sample, shape, tables, row=row),
        )
    return _contract_bottom(boundary)
