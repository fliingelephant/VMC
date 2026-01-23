"""Core wavefunction evaluation APIs.

The only evaluation entrypoints are `_value`, `_grad`, and `_value_and_grad`.
"""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from plum import dispatch

from vmc.models.mps import MPS
from vmc.models.peps import (
    PEPS,
    _compute_all_gradients,
    _forward_with_cache,
)
from vmc.utils.utils import spin_to_occupancy

__all__ = [
    "_value",
    "_grad",
    "_value_and_grad",
]


@dispatch
def _value(
    model: MPS,
    sample: jax.Array,
) -> jax.Array:
    """Compute amplitude for MPS sample(s). Auto-vmaps if sample.ndim == 2.

    Samples are flattened: shape (n_sites,) or (batch, n_sites).
    """
    sample = jnp.asarray(sample)
    tensors = [jnp.asarray(t) for t in model.tensors]
    if sample.ndim == 2:
        return MPS.apply(tensors, sample)
    return MPS.apply(tensors, sample[None, :])[0]


@dispatch
def _value(
    model: PEPS,
    sample: jax.Array,
) -> jax.Array:
    """Compute amplitude for PEPS sample(s). Auto-vmaps if sample.ndim == 2.

    Samples are flattened: shape (n_sites,) or (batch, n_sites).
    """
    sample = jnp.asarray(sample)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    if sample.ndim == 2:
        return jax.vmap(
            lambda s: PEPS.apply(tensors, s, model.shape, model.strategy)
        )(sample)
    return PEPS.apply(tensors, sample, model.shape, model.strategy)


def _grad(
    model: MPS | PEPS,
    sample: jax.Array,
    *,
    full_gradient: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """Compute amplitude gradient for sample(s). Auto-vmaps if sample.ndim == 2.

    Samples are flattened: shape (n_sites,) or (batch, n_sites).
    """
    _, grad_row, p_row = _value_and_grad(
        model, sample, full_gradient=full_gradient
    )
    return grad_row, p_row


@dispatch
def _value_and_grad(
    model: MPS,
    sample: jax.Array,
    *,
    full_gradient: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Compute amplitude and gradient for MPS sample(s). Auto-vmaps if sample.ndim == 2.

    Samples are flattened: shape (n_sites,) or (batch, n_sites).
    """
    sample = jnp.asarray(sample)
    if sample.ndim == 2:
        return jax.vmap(lambda s: _value_and_grad(model, s, full_gradient=full_gradient))(sample)

    tensors = [jnp.asarray(t) for t in model.tensors]
    n_sites = model.n_sites
    bond_dim = model.bond_dim
    phys_dim = model.phys_dim
    indices = spin_to_occupancy(sample)

    if full_gradient:
        tensors_flat = jnp.concatenate([t.ravel() for t in tensors])

        def amplitude_full(flat_params: jax.Array) -> jax.Array:
            state = jnp.ones((1,), dtype=flat_params.dtype)
            offset = 0
            for site in range(n_sites):
                left_dim, right_dim = MPS.site_dims(site, n_sites, bond_dim)
                size = phys_dim * left_dim * right_dim
                tensor = flat_params[offset : offset + size].reshape(
                    phys_dim, left_dim, right_dim
                )
                mat = tensor[indices[site]]
                state = jnp.einsum("i,ij->j", state, mat)
                offset += size
            return state[0]

        amp, grad_row = jax.value_and_grad(amplitude_full, holomorphic=True)(
            tensors_flat
        )
        return amp, grad_row, None

    left_envs = [jnp.ones((1,), dtype=tensors[0].dtype)]
    for site in range(n_sites):
        mat = tensors[site][indices[site]]
        left_envs.append(jnp.einsum("i,ij->j", left_envs[-1], mat))

    right_envs = [None] * (n_sites + 1)
    right_envs[n_sites] = jnp.ones((1,), dtype=tensors[0].dtype)
    right = right_envs[n_sites]
    for site in range(n_sites - 1, -1, -1):
        mat = tensors[site][indices[site]]
        right = jnp.einsum("ij,j->i", mat, right)
        right_envs[site] = right

    amp = left_envs[-1][0]
    grad_parts, p_parts = [], []
    for site in range(n_sites):
        left = left_envs[site]
        right = right_envs[site + 1]
        grad_site = left[:, None] * right[None, :]
        grad_parts.append(grad_site.reshape(-1))
        left_dim, right_dim = MPS.site_dims(site, n_sites, bond_dim)
        params_per_phys = left_dim * right_dim
        p_parts.append(jnp.full((params_per_phys,), indices[site], dtype=jnp.int8))

    return amp, jnp.concatenate(grad_parts), jnp.concatenate(p_parts)


@dispatch
def _value_and_grad(
    model: PEPS,
    sample: jax.Array,
    *,
    full_gradient: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Compute amplitude and gradient for PEPS sample(s). Auto-vmaps if sample.ndim == 2.

    Samples are flattened: shape (n_sites,) or (batch, n_sites).
    """
    sample = jnp.asarray(sample)
    if sample.ndim == 2:
        return jax.vmap(
            lambda s: _value_and_grad(model, s, full_gradient=full_gradient)
        )(sample)

    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    shape = model.shape
    bond_dim = model.bond_dim
    n_rows, n_cols = shape

    if full_gradient:
        amp, grad = jax.value_and_grad(PEPS.apply, holomorphic=True)(
            tensors, sample, shape, model.strategy
        )
        grad_flat, _ = ravel_pytree(grad)
        return amp, grad_flat, None

    spins = spin_to_occupancy(sample).reshape(shape)
    amp, top_envs = _forward_with_cache(tensors, spins, shape, model.strategy)
    env_grads = _compute_all_gradients(tensors, spins, shape, model.strategy, top_envs)

    grad_parts, p_parts = [], []
    for r in range(n_rows):
        for c in range(n_cols):
            grad_parts.append(env_grads[r][c].reshape(-1))
            up, down, left, right = PEPS.site_dims(
                r, c, n_rows, n_cols, bond_dim
            )
            params_per_phys = up * down * left * right
            p_parts.append(jnp.full((params_per_phys,), spins[r, c], dtype=jnp.int8))

    return amp, jnp.concatenate(grad_parts), jnp.concatenate(p_parts)
