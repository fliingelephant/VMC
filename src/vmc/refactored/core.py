"""Generic rollout builder for MCMC transition/estimate workflows."""
from __future__ import annotations

from typing import Any, Callable

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax

__all__ = ["make_mc_sampler"]


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
    ) -> tuple[tuple[jax.Array, jax.Array, Any], Any]:
        def step(carry, _):
            config_carry, key_carry, cache_carry = carry
            config_next, key_next, cache_next, estimates = batched_mc_sweep(
                tensors,
                config_carry,
                key_carry,
                cache_carry,
            )
            return (config_next, key_next, cache_next), estimates

        return jax.lax.scan(
            step,
            (config_states, chain_keys, cache),
            xs=None,
            length=n_steps,
        )
    return mc_sampler
