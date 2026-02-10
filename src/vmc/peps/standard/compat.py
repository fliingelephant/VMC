"""Compatibility surfaces for standard PEPS."""
from __future__ import annotations

import functools
from typing import Any

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from vmc.operators.local_terms import LocalHamiltonian, bucket_terms
from vmc.peps.common.contraction import _forward_with_cache
from vmc.peps.common.energy import (
    _compute_all_env_grads_and_energy,
    _compute_all_gradients,
)
from vmc.peps.common.strategy import ContractionStrategy
from vmc.utils.utils import spin_to_occupancy

__all__ = [
    "peps_apply",
    "local_estimate",
    "_value",
    "_grad",
    "_value_and_grad",
]


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def peps_apply(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
) -> jax.Array:
    spins = spin_to_occupancy(sample).reshape(shape)
    amp, _ = _forward_with_cache(tensors, spins, shape, strategy)
    return amp


def _peps_apply_fwd(tensors, sample, shape, strategy):
    spins = spin_to_occupancy(sample).reshape(shape)
    amp, top_envs = _forward_with_cache(tensors, spins, shape, strategy)
    return amp, (tensors, spins, top_envs)


def _peps_apply_bwd(shape, strategy, residuals, g):
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


peps_apply.defvjp(_peps_apply_fwd, _peps_apply_bwd)


def _value(
    model: Any,
    sample: jax.Array,
) -> jax.Array:
    """Compute amplitude for standard PEPS sample(s)."""
    sample = jnp.asarray(sample)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    if sample.ndim == 2:
        return jax.vmap(
            lambda s: model.apply(tensors, s, model.shape, model.strategy)
        )(sample)
    return model.apply(tensors, sample, model.shape, model.strategy)


def _grad(
    model: Any,
    sample: jax.Array,
    *,
    full_gradient: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """Compute amplitude gradient for standard PEPS sample(s)."""
    _, grad_row, p_row = _value_and_grad(
        model, sample, full_gradient=full_gradient
    )
    return grad_row, p_row


def _value_and_grad(
    model: Any,
    sample: jax.Array,
    *,
    full_gradient: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Compute amplitude and gradient for standard PEPS sample(s)."""
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
        amp, grad = jax.value_and_grad(model.apply, holomorphic=True)(
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
            up, down, left, right = model.site_dims(
                r, c, n_rows, n_cols, bond_dim
            )
            params_per_phys = up * down * left * right
            p_parts.append(jnp.full((params_per_phys,), spins[r, c], dtype=jnp.int8))

    return amp, jnp.concatenate(grad_parts), jnp.concatenate(p_parts)


def local_estimate(
    model: Any,
    samples: jax.Array,
    operator: LocalHamiltonian,
    amps: jax.Array,
) -> jax.Array:
    """Compute local energy estimates for PEPS from local operator terms."""
    samples = jnp.asarray(samples)
    amps = jnp.asarray(amps)
    shape = model.shape
    diagonal_terms, one_site_terms, horizontal_terms, vertical_terms, _ = bucket_terms(
        operator.terms, shape
    )
    has_diag = bool(diagonal_terms)
    has_one_site = any(term_list for row in one_site_terms for term_list in row)
    has_horizontal = any(term_list for row in horizontal_terms for term_list in row)
    has_vertical = any(term_list for row in vertical_terms for term_list in row)
    has_offdiag = has_one_site or has_horizontal or has_vertical

    if not has_diag and not has_offdiag:
        return jnp.zeros((samples.shape[0],), dtype=amps.dtype)

    if has_diag and not has_offdiag:
        phys_dim = model.phys_dim

        def diag_only(sample):
            spins = spin_to_occupancy(sample).reshape(shape)
            total = jnp.zeros((), dtype=amps.dtype)
            for term in diagonal_terms:
                idx = jnp.asarray(0, dtype=jnp.int32)
                for row, col in term.sites:
                    idx = idx * phys_dim + spins[row, col]
                total = total + term.diag[idx]
            return total

        return jax.vmap(diag_only)(samples)

    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]

    def per_sample(sample, amp):
        occupancy = spin_to_occupancy(sample)
        spins = occupancy.reshape(shape)
        _, top_envs = _forward_with_cache(tensors, spins, shape, model.strategy)
        _, energy, _ = _compute_all_env_grads_and_energy(
            tensors,
            spins,
            amp,
            shape,
            model.strategy,
            top_envs,
            diagonal_terms=diagonal_terms,
            one_site_terms=one_site_terms,
            horizontal_terms=horizontal_terms,
            vertical_terms=vertical_terms,
            collect_grads=False,
        )
        return energy

    return jax.vmap(per_sample, in_axes=(0, 0))(samples, amps)
