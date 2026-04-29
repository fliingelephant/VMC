"""Common PEPS contraction primitives."""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

from typing import Any

import jax
import jax.numpy as jnp

from vmc.peps.common.strategy import ContractionStrategy

__all__ = [
    "_build_row_mpo",
    "_contract_bottom",
    "_contract_2row_2col",
    "_contract_2row_1col",
    "_forward_with_cache",
    "_apply_mpo_from_below",
    "_compute_right_envs",
    "_contract_1row_1col",
]

def _build_row_mpo(tensors, row_indices, row, n_cols):
    """Build row-MPO for PEPS contraction."""
    return tuple(
        jnp.transpose(jnp.asarray(tensors[row][col])[row_indices[col]], (2, 3, 0, 1))
        for col in range(n_cols)
    )


def _contract_bottom(mps):
    """Contract bottom boundary of boundary state to get scalar amplitude."""
    state = jnp.array([1.0], dtype=mps[0].dtype)
    for site in mps:
        state = jnp.tensordot(state, site[:, 0, :], axes=[[0], [0]])
    return state.squeeze()


def _forward_with_cache(
    tensors: Any,
    spins: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
) -> tuple[jax.Array, list[tuple]]:
    """Forward pass that caches the top boundary before each row."""
    n_rows, n_cols = shape
    dtype = jnp.asarray(tensors[0][0]).dtype

    top_envs = [None] * n_rows
    boundary = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))

    for row in range(n_rows):
        top_envs[row] = boundary
        mpo = _build_row_mpo(tensors, spins[row], row, n_cols)
        boundary = strategy.apply(boundary, mpo)

    return _contract_bottom(boundary), top_envs


def _apply_mpo_from_below(
    bottom_mps: tuple,
    mpo: tuple,
    strategy: ContractionStrategy,
) -> tuple:
    """Apply MPO to boundary boundary state from below (for backward sweep)."""
    return strategy.apply(bottom_mps, tuple(jnp.transpose(w, (0, 1, 3, 2)) for w in mpo))


def _compute_right_envs(
    top_env: tuple,
    mpo_row: tuple,
    bottom_env: tuple,
    dtype,
) -> list[jax.Array]:
    """Compute right environments using direct einsum (no transfers)."""
    n_cols = len(mpo_row)
    right_envs = [None] * n_cols
    right_envs[n_cols - 1] = jnp.ones((1, 1, 1), dtype=dtype)
    for c in range(n_cols - 2, -1, -1):
        # Direct einsum: top @ mpo @ bot @ right_env -> new_right_env
        # top: (a, u, b), mpo: (c, d, u, v), bot: (e, v, f), right_env: (b, d, f) -> (a, c, e)
        right_envs[c] = jnp.einsum(
            "aub,cduv,evf,bdf->ace",
            top_env[c + 1], mpo_row[c + 1], bottom_env[c + 1], right_envs[c + 1],
            optimize=[(0, 3), (0, 2), (0, 1)],
        )
    return right_envs


def _contract_1row_1col(left_env, top, mpo, bottom, right_env):
    """Contract a one-row, one-column window to a scalar amplitude."""
    return jnp.einsum(
        "ace,aub,cduv,evf,bdf->",
        left_env,
        top,
        mpo,
        bottom,
        right_env,
        optimize=[(0, 1), (1, 2), (1, 2), (0, 1)],
    )


def _contract_2row_2col(left_env, top_env, mpo0_c, mpo1_c, mpo0_c1, mpo1_c1, bottom_env, right_env, c):
    """Contract 2-row, 2-column window for amplitude."""
    return jnp.einsum(
        "alxe,aub,lruv,xyvw,ewf,bgc,rsgh,ythi,fij,cstj->",
        left_env, top_env[c], mpo0_c, mpo1_c, bottom_env[c],
        top_env[c + 1], mpo0_c1, mpo1_c1, bottom_env[c + 1], right_env,
        optimize=[(0, 1), (0, 8), (0, 7), (0, 6), (0, 4), (0, 4), (1, 2), (1, 2), (0, 1)],
    )


def _contract_2row_1col(left_env, top, mpo0, mpo1, bottom, right_env):
    """Contract 2-row, 1-column for amplitude (last column)."""
    return jnp.einsum(
        "alxe,aub,lruv,xyvw,ewf,bryf->",
        left_env, top, mpo0, mpo1, bottom, right_env,
        optimize=[(0, 1), (0, 4), (1, 2), (1, 2), (0, 1)],
    )
