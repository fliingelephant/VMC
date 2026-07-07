"""Kernel data contracts shared by all PEPS families.

The MC sampler drives every family through the same cache-turnover cycle:
``transition`` consumes a :class:`Cache` and emits a :class:`Context`;
``estimate`` consumes the context and emits the next cache plus
:class:`LocalEstimates`.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from plum import dispatch

__all__ = [
    "Cache",
    "Context",
    "LocalEstimates",
    "build_mc_kernels",
    "_assemble_log_derivatives",
    "_broadcast_coeffs",
]


@dispatch.abstract
def build_mc_kernels(
    model: Any,
    operator: object,
    *,
    full_gradient: bool = False,
    observables: tuple = (),
) -> tuple[Callable, Callable, Callable]:
    """Build init_cache/transition/estimate kernels for one PEPS family."""


class Cache(NamedTuple):
    """Persistent cache across sweeps."""

    bottom_envs: Any
    coeffs: jax.Array | None = None


def _broadcast_coeffs(
    dynamic_coeffs: jax.Array | None, n_samples: int
) -> jax.Array | None:
    """Broadcast per-sweep dynamic coefficients across chains (None passthrough)."""
    if dynamic_coeffs is None:
        return None
    return jnp.broadcast_to(dynamic_coeffs, (n_samples, dynamic_coeffs.shape[0]))


class Context(NamedTuple):
    """Transient transition output consumed by estimate()."""

    amp: jax.Array
    top_envs: Any
    coeffs: jax.Array | None = None


class LocalEstimates(NamedTuple):
    """Per-sweep local quantities."""

    local_log_derivatives: jax.Array
    local_estimate: jax.Array
    active_slice_indices: jax.Array | None
    amp: jax.Array | None = None


def _assemble_log_derivatives(
    tensors: Any,
    params_per_site: Any,
    total_active_params: int,
    shape: tuple[int, int],
    env_grads: list[list[jax.Array]],
    config_state: jax.Array,
    amp: jax.Array,
    *,
    full_gradient: bool,
) -> tuple[jax.Array, jax.Array | None]:
    n_rows, n_cols = shape
    config_2d = config_state.reshape(shape)

    if full_gradient:
        grad_parts = []
        for row in range(n_rows):
            for col in range(n_cols):
                grad_full = jnp.zeros_like(tensors[row][col])
                grad_full = grad_full.at[config_2d[row, col]].set(env_grads[row][col])
                grad_parts.append(grad_full.reshape(-1))
        return jnp.concatenate(grad_parts) / amp, None

    grad_parts = [
        env_grads[row][col].reshape(-1)
        for row in range(n_rows)
        for col in range(n_cols)
    ]
    active_slice_indices = jnp.repeat(
        config_state.astype(jnp.int16),
        params_per_site,
        axis=0,
        total_repeat_length=total_active_params,
    )
    return jnp.concatenate(grad_parts) / amp, active_slice_indices
