"""Variational Monte Carlo utility functions.

This module provides helper functions for VMC calculations, including
sample manipulation and Jacobian computation.
"""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from plum import dispatch

from VMC.models.mps import SimpleMPS
from VMC.models.peps import SimplePEPS, make_peps_amplitude
from VMC.utils.smallo import (
    _small_o_row_mps_from_indices,
    _small_o_row_peps_from_indices,
)
from VMC.utils.utils import spin_to_occupancy

if TYPE_CHECKING:
    from netket.vqs import MCState

__all__ = [
    "flatten_samples",
    "batched_eval",
    "build_dense_jac",
    "build_dense_jac_from_state",
    "get_apply_fun",
]


def flatten_samples(samples: jax.Array) -> jax.Array:
    """Flatten all leading dimensions, keep the site dimension intact.

    Args:
        samples: Sample array with shape (..., n_sites).

    Returns:
        Flattened samples with shape (n_samples, n_sites).
    """
    samples = jnp.asarray(samples)
    return samples.reshape(-1, samples.shape[-1])


def batched_eval(
    eval_fn: Callable[[jax.Array], jax.Array],
    samples: jax.Array,
    *,
    batch_size: int,
) -> jax.Array:
    """Evaluate eval_fn in fixed-size chunks of batch_size using jax.lax.scan.

    This pads samples to a multiple of batch_size, scans over chunks, and
    trims the result back to the original sample count.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if samples.ndim == 0:
        raise ValueError("samples must have a batch dimension.")
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


def get_apply_fun(state: "MCState") -> tuple[Any, dict, dict, dict]:
    """Extract apply function and related data from a variational state.

    This function provides a stable interface to access the internal
    apply function from NetKet's MCState. If NetKet's internal API
    changes, only this function needs to be updated.

    Args:
        state: NetKet variational state.

    Returns:
        Tuple of (apply_fun, params, model_state, training_kwargs).
    """
    return (
        state._apply_fun,
        state.parameters,
        state.model_state,
        getattr(state, "training_kwargs", {}),
    )


@functools.partial(jax.jit, static_argnames=("apply_fun", "holomorphic"))
def _build_dense_jac_apply(
    apply_fun,
    params: dict,
    model_state: dict,
    samples: jax.Array,
    *,
    holomorphic: bool = True,
) -> jax.Array:
    """Compute dense, centered Jacobian flattened over parameter leaves.

    This function is JIT-compiled with `apply_fun` as a static argument,
    so it will be recompiled once per unique model but cached thereafter.

    Args:
        apply_fun: Functional form of the wavefunction (static, triggers recompile).
        params: Parameter pytree.
        model_state: Model state dict (e.g., batch statistics).
        samples: Sample configurations with shape (n_samples, n_sites).
        holomorphic: Whether the function is holomorphic (static).

    Returns:
        Dense Jacobian matrix with shape (n_samples, n_params).
    """
    jac_fun = jax.jacrev(
        lambda p, x: apply_fun({"params": p, **model_state}, x),
        holomorphic=holomorphic,
    )
    jac_tree = jax.vmap(jac_fun, in_axes=(None, 0))(params, samples)
    jac_tree = jax.tree_util.tree_map(
        lambda x: (x - jnp.mean(x, axis=0, keepdims=True)) / jnp.sqrt(samples.shape[0]),
        jac_tree,
    )
    leaves = [
        leaf.reshape(samples.shape[0], -1)
        for leaf in jax.tree_util.tree_leaves(jac_tree)
    ]
    return jnp.concatenate(leaves, axis=1)


def _build_full_jac_mps(model: SimpleMPS, samples: jax.Array) -> jax.Array:
    """Compute full (uncentered) Jacobian for MPS samples."""
    n_sites = model.n_sites
    bond_dim = model.bond_dim
    phys_dim = model.phys_dim
    tensors_list = [jnp.asarray(t) for t in model.tensors]
    tensors_flat = jnp.concatenate([t.ravel() for t in tensors_list])
    occupancy = spin_to_occupancy(samples)

    def log_amplitude_full(tensors_flat: jax.Array, sample_occ: jax.Array) -> jax.Array:
        state = jnp.ones((1,), dtype=tensors_flat.dtype)
        offset = 0
        for site in range(n_sites):
            left_dim = 1 if site == 0 else bond_dim
            right_dim = 1 if site == n_sites - 1 else bond_dim
            size = phys_dim * left_dim * right_dim
            tensor = tensors_flat[offset : offset + size].reshape(
                phys_dim, left_dim, right_dim
            )
            mat = tensor[sample_occ[site]]
            state = jnp.einsum("i,ij->j", state, mat)
            offset += size
        return jnp.log(state[0])

    jac_fn = jax.jacrev(log_amplitude_full, holomorphic=True)
    return jax.vmap(jac_fn, in_axes=(None, 0))(tensors_flat, occupancy)


def _build_full_jac_peps(model: SimplePEPS, samples: jax.Array) -> jax.Array:
    """Compute full (uncentered) Jacobian for PEPS samples."""
    shape = model.shape
    amp_fn = make_peps_amplitude(shape, model.strategy)

    def tensors_to_flat(tensors) -> jax.Array:
        leaves = []
        for row in tensors:
            for tensor in row:
                leaves.append(jnp.asarray(tensor).ravel())
        return jnp.concatenate(leaves)

    def flat_to_tensors(flat: jax.Array, template):
        result = []
        offset = 0
        for row in template:
            row_result = []
            for tensor in row:
                t = jnp.asarray(tensor)
                size = t.size
                row_result.append(flat[offset : offset + size].reshape(t.shape))
                offset += size
            result.append(row_result)
        return result

    tensors_flat = tensors_to_flat(model.tensors)

    def log_amplitude_full(flat_params: jax.Array, sample: jax.Array) -> jax.Array:
        tensors_nested = flat_to_tensors(flat_params, model.tensors)
        amp = amp_fn(tensors_nested, sample)
        return jnp.log(amp)

    jac_fn = jax.jacrev(log_amplitude_full, holomorphic=True)
    return jax.vmap(jac_fn, in_axes=(None, 0))(tensors_flat, samples)


@functools.partial(jax.jit, static_argnames=("bond_dim", "phys_dim"))
def _build_small_o_jac_mps(
    tensors: list[jax.Array],
    samples: jax.Array,
    bond_dim: int,
    phys_dim: int,
) -> tuple[jax.Array, jax.Array]:
    """Build small-o Jacobian for MPS without forming the full Jacobian."""
    occupancy = spin_to_occupancy(samples)
    tensors_list = [jnp.asarray(t) for t in tensors]

    def row_fn(indices: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _small_o_row_mps_from_indices(tensors_list, indices, bond_dim, phys_dim)

    o, p = jax.vmap(row_fn)(occupancy)
    return o, p


def _build_small_o_jac_peps(
    model: SimplePEPS,
    samples: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Build small-o Jacobian for PEPS without forming the full Jacobian."""
    shape = model.shape
    bond_dim = model.bond_dim
    occupancy = spin_to_occupancy(samples)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]

    def row_fn(occ_sample: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _small_o_row_peps_from_indices(
            tensors, occ_sample, shape, bond_dim, model.strategy
        )

    o, p = jax.vmap(row_fn)(occupancy)
    return o, p


@dispatch
def build_dense_jac(
    apply_fun: Callable,
    params: dict,
    model_state: dict,
    samples: jax.Array,
    *,
    holomorphic: bool = True,
) -> jax.Array:
    """Compute dense, centered Jacobian for NetKet-compatible apply_funs."""
    return _build_dense_jac_apply(
        apply_fun,
        params,
        model_state,
        samples,
        holomorphic=holomorphic,
    )


@dispatch
def build_dense_jac(
    model: SimpleMPS,
    samples: jax.Array,
    *,
    full_gradient: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute full or small-o Jacobian for MPS samples (uncentered).

    Returns (o, p) when full_gradient is False, otherwise dense O.
    """
    samples = flatten_samples(samples)
    if full_gradient:
        return _build_full_jac_mps(model, samples)
    tensors = [jnp.asarray(t) for t in model.tensors]
    return _build_small_o_jac_mps(
        tensors,
        samples,
        bond_dim=model.bond_dim,
        phys_dim=model.phys_dim,
    )


@dispatch
def build_dense_jac(
    model: SimplePEPS,
    samples: jax.Array,
    *,
    full_gradient: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute full or small-o Jacobian for PEPS samples (uncentered).

    Returns (o, p) when full_gradient is False, otherwise dense O.
    """
    samples = flatten_samples(samples)
    if full_gradient:
        return _build_full_jac_peps(model, samples)
    return _build_small_o_jac_peps(model, samples)


def build_dense_jac_from_state(
    state: "MCState",
    samples: jax.Array | None = None,
    *,
    holomorphic: bool = True,
) -> jax.Array:
    """Compute dense Jacobian directly from a variational state.

    Convenience wrapper around build_dense_jac that extracts
    the necessary components from the state.

    Args:
        state: NetKet variational state.
        samples: Optional samples; uses state.samples if not provided.
        holomorphic: Whether the function is holomorphic.

    Returns:
        Dense Jacobian matrix with shape (n_samples, n_params).
    """
    if samples is None:
        samples = flatten_samples(state.samples)
    else:
        samples = flatten_samples(samples)

    apply_fun, params, model_state, _ = get_apply_fun(state)
    return build_dense_jac(apply_fun, params, model_state, samples, holomorphic=holomorphic)
