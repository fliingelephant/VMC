"""Small-o helpers for tensor network states."""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp

from VMC.models.peps import SimplePEPS, _compute_all_gradients, _forward_with_cache
from VMC.utils.utils import spin_to_occupancy

__all__ = [
    "small_o_row_mps",
    "small_o_row_peps",
]


def _small_o_row_mps_from_indices(
    tensors: list[jax.Array],
    indices: jax.Array,
    bond_dim: int,
    phys_dim: int = 2,
) -> tuple[jax.Array, jax.Array]:
    n_sites = len(tensors)
    left_envs = [jnp.ones((1,), dtype=tensors[0].dtype)]
    for site in range(n_sites):
        mat = tensors[site][indices[site]]
        left_envs.append(jnp.einsum("i,ij->j", left_envs[-1], mat))

    right_envs: list[jax.Array] = [None] * (n_sites + 1)
    right_envs[n_sites] = jnp.ones((1,), dtype=tensors[0].dtype)
    right = right_envs[n_sites]
    for site in range(n_sites - 1, -1, -1):
        mat = tensors[site][indices[site]]
        right = jnp.einsum("ij,j->i", mat, right)
        right_envs[site] = right

    amp = left_envs[-1][0]
    inv_amp = 1.0 / amp
    o_parts = []
    p_parts = []
    for site in range(n_sites):
        left = left_envs[site]
        right = right_envs[site + 1]
        o_site = (left[:, None] * right[None, :]) * inv_amp
        o_parts.append(o_site.reshape(-1))
        left_dim = 1 if site == 0 else bond_dim
        right_dim = 1 if site == n_sites - 1 else bond_dim
        params_per_phys = left_dim * right_dim
        p_parts.append(jnp.full((params_per_phys,), indices[site], dtype=jnp.int8))
    o_row = jnp.concatenate(o_parts)
    p_row = jnp.concatenate(p_parts)
    return o_row, p_row


def small_o_row_mps(
    tensors: list,
    sample: jax.Array,
    bond_dim: int,
    phys_dim: int = 2,
) -> tuple[jax.Array, jax.Array]:
    """Build small-o row for a single MPS sample (spins in {-1, +1})."""
    indices = spin_to_occupancy(sample)
    tensors_list = [jnp.asarray(t) for t in tensors]
    return _small_o_row_mps_from_indices(tensors_list, indices, bond_dim, phys_dim)


def _small_o_row_peps_from_indices(
    tensors: list[list[jax.Array]],
    occ_sample: jax.Array,
    shape: tuple[int, int],
    bond_dim: int,
    strategy,
) -> tuple[jax.Array, jax.Array]:
    spins = occ_sample.reshape(shape)
    amp, top_envs = _forward_with_cache(tensors, spins, shape, strategy)
    env_grads = _compute_all_gradients(tensors, spins, shape, strategy, top_envs)
    inv_amp = 1.0 / amp
    o_parts = []
    p_parts = []
    n_rows, n_cols = shape
    for r in range(n_rows):
        for c in range(n_cols):
            o_parts.append(env_grads[r][c].reshape(-1) * inv_amp)
            up = 1 if r == 0 else bond_dim
            down = 1 if r == shape[0] - 1 else bond_dim
            left = 1 if c == 0 else bond_dim
            right = 1 if c == shape[1] - 1 else bond_dim
            params_per_phys = up * down * left * right
            p_parts.append(jnp.full((params_per_phys,), spins[r, c], dtype=jnp.int8))
    o_row = jnp.concatenate(o_parts)
    p_row = jnp.concatenate(p_parts)
    return o_row, p_row


def small_o_row_peps(
    model: SimplePEPS,
    sample: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Build small-o row for a single PEPS sample (spins in {-1, +1})."""
    occ_sample = spin_to_occupancy(sample)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    return _small_o_row_peps_from_indices(
        tensors, occ_sample, model.shape, model.bond_dim, model.strategy
    )
