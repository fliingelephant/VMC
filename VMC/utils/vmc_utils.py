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
    """Extract apply function and related data from a variational state."""
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
    """Compute dense, centered Jacobian flattened over parameter leaves."""
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
        apply_fun, params, model_state, samples, holomorphic=holomorphic,
    )


def build_dense_jac_from_state(
    state: "MCState",
    samples: jax.Array | None = None,
    *,
    holomorphic: bool = True,
) -> jax.Array:
    """Compute dense Jacobian directly from a variational state."""
    if samples is None:
        samples = flatten_samples(state.samples)
    else:
        samples = flatten_samples(samples)

    apply_fun, params, model_state, _ = get_apply_fun(state)
    return build_dense_jac(apply_fun, params, model_state, samples, holomorphic=holomorphic)
