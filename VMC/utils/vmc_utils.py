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

from VMC.core.eval import _value

if TYPE_CHECKING:
    from netket.vqs import MCState

__all__ = [
    "flatten_samples",
    "batched_eval",
    "build_dense_jac",
    "build_dense_jac_from_state",
    "get_apply_fun",
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


def get_apply_fun(state: "MCState") -> tuple[Any, dict, dict, dict]:
    """Extract apply function and related data from a variational state."""
    return (
        state._apply_fun,
        state.parameters,
        state.model_state,
        getattr(state, "training_kwargs", {}),
    )


def model_params(model) -> dict[str, Any]:
    """Extract model parameters as plain arrays."""
    tensors = jax.tree_util.tree_map(jnp.asarray, model.tensors)
    return {"tensors": tensors}


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


def local_estimate(model, samples: jax.Array, operator) -> jax.Array:
    """Compute local energy estimates for samples.

    Args:
        model: Variational model (MPS/PEPS).
        samples: Spin configurations with shape (n_samples, n_sites).
        operator: Operator providing ``get_conn_padded``.

    Returns:
        Local energy estimates with shape (n_samples,).
    """
    samples = jnp.asarray(samples)
    if not hasattr(operator, "get_conn_padded"):
        raise TypeError("operator must provide get_conn_padded")

    sigma_p, mels = operator.get_conn_padded(samples)
    sigma_p = jnp.asarray(sigma_p)
    mels = jnp.asarray(mels)

    if sigma_p.ndim != 3:
        sigma_p = sigma_p.reshape(samples.shape[0], -1, samples.shape[-1])
        mels = mels.reshape(sigma_p.shape[:2])

    amps_sigma = jax.vmap(lambda s: _value(model, s))(samples)
    flat_sigma_p = sigma_p.reshape(-1, sigma_p.shape[-1])
    amps_sigma_p = jax.vmap(lambda s: _value(model, s))(flat_sigma_p).reshape(
        sigma_p.shape[:-1]
    )
    return jnp.sum(mels * (amps_sigma_p / amps_sigma[:, None]), axis=-1)
