"""Compatibility surfaces for gauge-invariant PEPS."""
from __future__ import annotations

import functools
from typing import Any

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp

from vmc.peps.common.contraction import _build_row_mpo, _contract_bottom, _forward_with_cache
from vmc.peps.common.energy import _compute_all_gradients

__all__ = ["gi_apply"]


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _peps_apply_occupancy(
    tensors: Any,
    sample: jax.Array,
    shape: tuple[int, int],
    strategy: Any,
) -> jax.Array:
    spins = sample.reshape(shape)
    boundary = tuple(
        jnp.ones((1, 1, 1), dtype=jnp.asarray(tensors[0][0]).dtype)
        for _ in range(shape[1])
    )
    for row in range(shape[0]):
        mpo = _build_row_mpo(tensors, spins[row], row, shape[1])
        boundary = strategy.apply(boundary, mpo)
    return _contract_bottom(boundary)


def _peps_apply_occupancy_fwd(tensors, sample, shape, strategy):
    spins = sample.reshape(shape)
    amp, top_envs = _forward_with_cache(tensors, spins, shape, strategy)
    return amp, (tensors, spins, top_envs)


def _peps_apply_occupancy_bwd(shape, strategy, residuals, g):
    tensors, spins, top_envs = residuals
    n_rows, n_cols = shape
    env_grads = _compute_all_gradients(tensors, spins, shape, strategy, top_envs)
    grad_leaves = []
    for r in range(n_rows):
        for c in range(n_cols):
            grad_full = jnp.zeros_like(jnp.asarray(tensors[r][c]))
            grad_leaves.append(grad_full.at[spins[r, c]].set(g * env_grads[r][c]))
    return (
        jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(tensors), grad_leaves),
        None,
    )


_peps_apply_occupancy.defvjp(_peps_apply_occupancy_fwd, _peps_apply_occupancy_bwd)


def gi_apply(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    shape: tuple[int, int],
    config: Any,
    strategy: Any,
) -> jax.Array:
    """Compute GI-PEPS amplitude for a given sample."""
    n_rows, n_cols = shape
    num_sites = n_rows * n_cols
    num_h = n_rows * (n_cols - 1)
    sites = sample[:num_sites].reshape((n_rows, n_cols))
    h_links = sample[num_sites : num_sites + num_h].reshape((n_rows, n_cols - 1))
    v_links = sample[num_sites + num_h :].reshape((n_rows - 1, n_cols))

    n = jnp.asarray(config.N, dtype=h_links.dtype)
    nl = jnp.pad(h_links, ((0, 0), (1, 0)), constant_values=0)
    nr = jnp.pad(h_links, ((0, 0), (0, 1)), constant_values=0)
    nu = jnp.pad(v_links, ((1, 0), (0, 0)), constant_values=0)
    nd = jnp.pad(v_links, ((0, 1), (0, 0)), constant_values=0)
    div = (nl + nd - nu - nr) % n
    charge_of_site = jnp.asarray(config.charge_of_site, dtype=sites.dtype)
    charge = charge_of_site[sites]
    valid = (div + charge) % n == jnp.asarray(config.Qx, dtype=div.dtype)
    invalid = jnp.any(~valid)
    dtype = jnp.asarray(tensors[0][0]).dtype

    def _compute_amp(_):
        from vmc.peps.gi.model import assemble_tensors

        eff_tensors = assemble_tensors(tensors, h_links, v_links, config)
        spins = sites.reshape(-1)
        return _peps_apply_occupancy(eff_tensors, spins, shape, strategy)

    return jax.lax.cond(
        invalid,
        lambda _: jnp.zeros((), dtype=dtype),
        _compute_amp,
        operand=None,
    )
