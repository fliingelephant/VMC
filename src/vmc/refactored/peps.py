"""PEPS kernels for the refactored rollout core."""
from __future__ import annotations

from typing import Any, Callable, NamedTuple

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp

from vmc.models.peps import (
    PEPS,
    _apply_mpo_from_below,
    _build_row_mpo,
    _compute_all_env_grads_and_energy,
    _compute_right_envs,
    _contract_bottom,
    _metropolis_ratio,
)
from vmc.operators.local_terms import LocalHamiltonian, bucket_terms
from vmc.utils.smallo import params_per_site as params_per_site_fn

__all__ = [
    "Cache",
    "Context",
    "LocalEstimates",
    "build_mc_kernels",
]


class Cache(NamedTuple):
    """Persistent cache across sweeps."""

    bottom_envs: Any


class Context(NamedTuple):
    """Transient transition output consumed by estimate()."""

    amp: jax.Array
    top_envs: Any


class LocalEstimates(NamedTuple):
    """Per-sweep local quantities."""

    local_log_derivatives: jax.Array
    local_estimate: jax.Array
    active_slice_indices: jax.Array | None
    amp: jax.Array | None = None


def _assemble_log_derivatives(
    tensors: Any,
    params_per_site: jax.Array,
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
        config_state.astype(jnp.int8),
        params_per_site,
        axis=0,
        total_repeat_length=total_active_params,
    )
    return jnp.concatenate(grad_parts) / amp, active_slice_indices


def build_mc_kernels(
    model: PEPS,
    operator: LocalHamiltonian,
    *,
    full_gradient: bool = False,
) -> tuple[Callable, Callable, Callable]:
    """Build PEPS init_cache/transition/estimate kernels.

    The returned kernels are intentionally not jitted. For tVMC, jit the outer
    entrypoint that calls the sampler and donate chain state buffers there, e.g.
    donate `(config_states, chain_keys, cache)`.
    """
    shape = model.shape
    n_rows, n_cols = shape
    n_sites = int(n_rows * n_cols)
    strategy = model.strategy
    params_per_site = tuple(int(p) for p in params_per_site_fn(model))
    params_per_site_repeats = jnp.asarray(params_per_site, dtype=jnp.int32)
    total_active_params = int(sum(params_per_site))
    diagonal_terms, one_site_terms, horizontal_terms, vertical_terms, _ = bucket_terms(
        operator.terms, shape
    )

    def init_cache(tensors: Any, config_states: jax.Array) -> Cache:
        config_states_flat = config_states.reshape(config_states.shape[0], n_sites)

        def build_one(config_state: jax.Array):
            indices = config_state.reshape(shape)
            dtype = tensors[0][0].dtype
            envs = [None] * n_rows
            env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
            for row in range(n_rows - 1, -1, -1):
                envs[row] = env
                mpo = _build_row_mpo(tensors, indices[row], row, n_cols)
                env = _apply_mpo_from_below(env, mpo, strategy)
            return tuple(envs)

        return Cache(bottom_envs=jax.vmap(build_one)(config_states_flat))

    def transition(
        tensors: Any,
        config_state: jax.Array,
        key: jax.Array,
        cache: Cache,
    ) -> tuple[jax.Array, jax.Array, Context]:
        indices = config_state.reshape(shape)
        dtype = tensors[0][0].dtype
        top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
        top_envs_cache = [None] * n_rows

        for row in range(n_rows):
            top_envs_cache[row] = top_env
            bottom_env = cache.bottom_envs[row]
            mpo_row = _build_row_mpo(tensors, indices[row], row, n_cols)
            right_envs = _compute_right_envs(top_env, mpo_row, bottom_env, dtype)
            left_env = jnp.ones((1, 1, 1), dtype=dtype)
            # Reinitialize per row (instead of carrying across rows) because
            # top_env is replaced by strategy.apply(...) at row end, which may
            # truncate and change the effective boundary representation.
            amp_cur = jnp.einsum(
                "ace,aub,cduv,evf,bdf->",
                left_env,
                top_env[0],
                mpo_row[0],
                bottom_env[0],
                right_envs[0],
                optimize=[(0, 1), (1, 2), (1, 2), (0, 1)],
            )
            updated_row = []

            for col in range(n_cols):
                site_tensor = tensors[row][col]
                phys_dim = int(site_tensor.shape[0])
                cur_idx = indices[row, col]
                if phys_dim == 1:
                    flip_idx = cur_idx
                elif phys_dim == 2:
                    flip_idx = 1 - cur_idx
                else:
                    key, flip_key = jax.random.split(key)
                    delta = jax.random.randint(
                        flip_key, (), 1, phys_dim, dtype=jnp.int32
                    )
                    flip_idx = (cur_idx + delta) % phys_dim

                mpo_cur = mpo_row[col]
                mpo_flip = jnp.transpose(site_tensor[flip_idx], (2, 3, 0, 1))
                amp_flip = jnp.einsum(
                    "ace,aub,cduv,evf,bdf->",
                    left_env,
                    top_env[col],
                    mpo_flip,
                    bottom_env[col],
                    right_envs[col],
                    optimize=[(0, 1), (1, 2), (1, 2), (0, 1)],
                )
                ratio = _metropolis_ratio(jnp.abs(amp_cur) ** 2, jnp.abs(amp_flip) ** 2)
                key, accept_key = jax.random.split(key)
                accept = jax.random.uniform(accept_key) < jnp.minimum(1.0, ratio)
                new_idx = jnp.where(accept, flip_idx, cur_idx)
                indices = indices.at[row, col].set(new_idx)
                amp_cur = jnp.where(accept, amp_flip, amp_cur)

                mpo_sel = jnp.where(accept, mpo_flip, mpo_cur)
                updated_row.append(mpo_sel)
                left_env = jnp.einsum(
                    "ace,aub,cduv,evf->bdf",
                    left_env,
                    top_env[col],
                    mpo_sel,
                    bottom_env[col],
                    optimize=[(0, 1), (0, 2), (0, 1)],
                )

            top_env = strategy.apply(top_env, tuple(updated_row))

        amp = _contract_bottom(top_env)
        return indices.reshape(-1), key, Context(amp=amp, top_envs=tuple(top_envs_cache))

    def estimate(
        tensors: Any,
        config_state_next: jax.Array,
        context: Context,
    ) -> tuple[Cache, LocalEstimates]:
        indices = config_state_next.reshape(shape)
        env_grads, local_estimate, bottom_envs_next = _compute_all_env_grads_and_energy(
            tensors,
            indices,
            context.amp,
            shape,
            strategy,
            context.top_envs,
            diagonal_terms=diagonal_terms,
            one_site_terms=one_site_terms,
            horizontal_terms=horizontal_terms,
            vertical_terms=vertical_terms,
            collect_grads=True,
        )
        local_log_derivatives, active_slice_indices = _assemble_log_derivatives(
            tensors,
            params_per_site_repeats,
            total_active_params,
            shape,
            env_grads,
            config_state_next,
            context.amp,
            full_gradient=full_gradient,
        )
        return Cache(bottom_envs=tuple(bottom_envs_next)), LocalEstimates(
            local_log_derivatives=local_log_derivatives,
            local_estimate=local_estimate,
            active_slice_indices=active_slice_indices,
            amp=context.amp,
        )

    return init_cache, transition, estimate
