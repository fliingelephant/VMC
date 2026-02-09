"""Generic rollout builder for MCMC transition/estimate workflows."""
from __future__ import annotations

import logging
import math
from typing import Any, Callable

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax

__all__ = [
    "make_mc_sampler",
    "_collect_steps",
    "_sample_counts",
    "_trim_samples",
]

logger = logging.getLogger(__name__)


def make_mc_sampler(
    transition: Callable[[Any, jax.Array, jax.Array, Any], tuple[jax.Array, jax.Array, Any]],
    estimate: Callable[[Any, jax.Array, Any], tuple[Any, Any]],
) -> Callable:
    """Build a pure-JAX MC sampler from transition/estimate kernels."""

    def mc_sweep(
        tensors: Any,
        config_state: jax.Array,
        key: jax.Array,
        cache: Any,
    ) -> tuple[jax.Array, jax.Array, Any, Any]:
        config_next, key_next, context = transition(tensors, config_state, key, cache)
        cache_next, local_estimates = estimate(tensors, config_next, context)
        return config_next, key_next, cache_next, local_estimates

    batched_mc_sweep = jax.vmap(mc_sweep, in_axes=(None, 0, 0, 0))

    def mc_sampler(
        tensors: Any,
        config_states: jax.Array,
        chain_keys: jax.Array,
        cache: Any,
        *,
        n_steps: int,
    ) -> tuple[tuple[jax.Array, jax.Array, Any], tuple[jax.Array, Any]]:
        def step(carry, _):
            config_carry, key_carry, cache_carry = carry
            config_next, key_next, cache_next, estimates = batched_mc_sweep(
                tensors,
                config_carry,
                key_carry,
                cache_carry,
            )
            return (config_next, key_next, cache_next), (config_next, estimates)

        return jax.lax.scan(
            step,
            (config_states, chain_keys, cache),
            xs=None,
            length=n_steps,
        )
    return mc_sampler


def _collect_steps(step_fn, carry, count: int, desc: str):
    if logger.isEnabledFor(logging.INFO):
        logger.info("%s: sampling %d steps", desc, count)
    result = jax.lax.scan(step_fn, carry, xs=None, length=count)
    if logger.isEnabledFor(logging.INFO):
        logger.info("%s: done", desc)
    return result


def _sample_counts(n_samples: int, n_chains: int) -> tuple[int, int, int, int]:
    chain_length = math.ceil(n_samples / n_chains)
    return n_samples, n_chains, chain_length, chain_length * n_chains


def _trim_samples(samples: jax.Array, total_samples: int, num_samples: int) -> jax.Array:
    return samples.reshape((total_samples,) + samples.shape[2:])[:num_samples]
