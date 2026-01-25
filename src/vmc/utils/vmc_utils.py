"""Variational Monte Carlo utility functions.

This module provides helper functions for VMC calculations, including
sample manipulation and Jacobian computation.
"""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from plum import dispatch

from vmc.core.eval import _value
from vmc.models.peps import (
    PEPS,
    _build_row_mpo,
    _compute_all_env_grads_and_energy,
)
from vmc.operators.local_terms import LocalHamiltonian, bucket_terms
from vmc.utils.utils import spin_to_occupancy

__all__ = [
    "flatten_samples",
    "batched_eval",
    "build_dense_jac",
    "local_estimate",
    "model_params",
]


def flatten_samples(samples: jax.Array) -> jax.Array:
    """Flatten all leading dimensions, keep the site dimension intact."""
    samples = jnp.asarray(samples)
    return samples.reshape(-1, samples.shape[-1])


def batched_eval(
    eval_fn: Callable[[jax.Array], jax.Array],
    samples: jax.Array,
    *,
    batch_size: int,
) -> jax.Array:
    """Evaluate eval_fn in fixed-size chunks of batch_size using jax.lax.scan."""
    n_samples = int(samples.shape[0])
    trailing_shape = samples.shape[1:]
    pad = (-n_samples) % batch_size
    if pad:
        padding = jnp.zeros((pad, *trailing_shape), dtype=samples.dtype)
        samples = jnp.concatenate([samples, padding], axis=0)
    num_batches = samples.shape[0] // batch_size
    batches = samples.reshape(num_batches, batch_size, *trailing_shape)

    def scan_fn(_, batch):
        return None, eval_fn(batch)

    _, output_batches = jax.lax.scan(scan_fn, None, batches)
    output_shape = output_batches.shape
    outputs = output_batches.reshape(
        output_shape[0] * output_shape[1], *output_shape[2:]
    )[:n_samples]
    return outputs


def model_params(model) -> dict[str, Any]:
    """Extract model parameters as plain arrays."""
    _, params, _ = nnx.split(model, nnx.Param, ...)
    return params.to_pure_dict()


@functools.partial(jax.jit, static_argnames=("apply_fun", "holomorphic"))
def build_dense_jac(
    apply_fun: Callable,
    params: dict,
    model_state: dict,
    samples: jax.Array,
    *,
    holomorphic: bool = True,
) -> jax.Array:
    """Compute dense, centered Jacobian for NetKet-compatible apply_funs.

    The Jacobian is not rescaled by ``1 / sqrt(n_samples)``.
    """
    jac_fun = jax.jacrev(
        lambda p, x: apply_fun({"params": p, **model_state}, x),
        holomorphic=holomorphic,
    )
    jac_tree = jax.vmap(jac_fun, in_axes=(None, 0))(params, samples)
    jac_tree = jax.tree_util.tree_map(
        lambda x: x - jnp.mean(x, axis=0, keepdims=True),
        jac_tree,
    )
    leaves = [
        leaf.reshape(samples.shape[0], -1)
        for leaf in jax.tree_util.tree_leaves(jac_tree)
    ]
    return jnp.concatenate(leaves, axis=1)


@dispatch
def local_estimate(
    model: PEPS,
    samples: jax.Array,
    operator: LocalHamiltonian,
    amps: jax.Array,
) -> jax.Array:
    """Compute local energy estimates for PEPS from local operator terms."""
    samples = jnp.asarray(samples)
    amps = jnp.asarray(amps)
    shape = model.shape
    n_rows, n_cols = shape
    diagonal_terms, one_site_terms, horizontal_terms, vertical_terms = bucket_terms(
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
    dtype = tensors[0][0].dtype

    def per_sample(sample, amp):
        spins = spin_to_occupancy(sample).reshape(shape)
        row_mpos = [
            _build_row_mpo(tensors, spins[row], row, n_cols)
            for row in range(n_rows)
        ]
        boundary = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
        top_envs = []
        for row in range(n_rows):
            top_envs.append(boundary)
            boundary = model.strategy.apply(boundary, row_mpos[row])
        _, energy, _ = _compute_all_env_grads_and_energy(
            tensors,
            spins,
            amp,
            shape,
            model.strategy,
            top_envs,
            row_mpos=row_mpos,
            diagonal_terms=diagonal_terms,
            one_site_terms=one_site_terms,
            horizontal_terms=horizontal_terms,
            vertical_terms=vertical_terms,
            collect_grads=False,
        )
        return energy

    return jax.vmap(per_sample, in_axes=(0, 0))(samples, amps)


@dispatch
def local_estimate(
    model: object,
    samples: jax.Array,
    operator: object,
    amps: jax.Array,
) -> jax.Array:
    """Compute local energy estimates for samples.

    Args:
        model: Variational model (MPS/PEPS).
        samples: Spin configurations with shape (n_samples, n_sites).
        operator: Operator providing ``get_conn_padded``.
        amps: Pre-computed amplitudes for samples.

    Returns:
        Local energy estimates with shape (n_samples,).
    """
    samples = jnp.asarray(samples)
    sigma_p, mels = operator.get_conn_padded(samples)
    sigma_p = jnp.asarray(sigma_p)
    mels = jnp.asarray(mels)

    if sigma_p.ndim != 3:
        sigma_p = sigma_p.reshape(samples.shape[0], -1, samples.shape[-1])
        mels = mels.reshape(sigma_p.shape[:2])

    flat_sigma_p = sigma_p.reshape(-1, sigma_p.shape[-1])
    amps_sigma_p = jax.vmap(_value, in_axes=(None, 0))(model, flat_sigma_p).reshape(
        sigma_p.shape[:-1]
    )
    return jnp.sum(mels * (amps_sigma_p / amps[:, None]), axis=-1)
