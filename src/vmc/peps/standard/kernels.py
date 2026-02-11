"""PEPS kernel bundle for the canonical sampling core."""
from __future__ import annotations

from typing import Any, Callable, NamedTuple

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from plum import dispatch

from vmc.peps.common.contraction import (
    _apply_mpo_from_below,
    _build_row_mpo,
    _compute_right_envs,
    _contract_bottom,
)
from vmc.peps.common.energy import _compute_all_env_grads_and_energy
from vmc.peps.standard.model import PEPS
from vmc.operators.local_terms import (
    BucketedOperators,
    LocalHamiltonian,
    bucket_operators,
)
from vmc.operators.time_dependent import TimeDependentHamiltonian
from vmc.utils.smallo import params_per_site as params_per_site_fn
from vmc.utils.utils import _metropolis_hastings_accept

__all__ = [
    "Cache",
    "Context",
    "LocalEstimates",
    "build_mc_kernels",
]


class Cache(NamedTuple):
    """Persistent cache across sweeps."""

    bottom_envs: Any
    coeffs: jax.Array | None = None


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


@dispatch
def _bucketed_terms_for_standard_operator(
    model: PEPS,
    operator: LocalHamiltonian,
) -> BucketedOperators:
    return bucket_operators(
        operator.terms,
        model.shape,
        eval_span=type(model).eval_span,
    )


@_bucketed_terms_for_standard_operator.dispatch
def _bucketed_terms_for_standard_operator(
    model: PEPS,
    operator: TimeDependentHamiltonian,
) -> BucketedOperators:
    base = operator.base
    if not isinstance(base, LocalHamiltonian):
        raise NotImplementedError(
            "TimeDependentHamiltonian for standard PEPS requires a LocalHamiltonian base."
        )
    return bucket_operators(
        base.terms,
        model.shape,
        eval_span=type(model).eval_span,
    )


@dispatch
def build_mc_kernels(
    model: PEPS,
    operator: object,
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
    params_per_site = tuple(int(p) for p in params_per_site_fn(model))
    params_per_site_repeats = jnp.asarray(params_per_site, dtype=jnp.int32)
    total_active_params = int(sum(params_per_site))
    terms = _bucketed_terms_for_standard_operator(model, operator)

    def init_cache(
        tensors: Any,
        samples: jax.Array,
        coeffs: jax.Array | None = None,
    ) -> Cache:
        samples_flat = samples.reshape(-1, n_sites)

        def build_one_bottom_envs(sample: jax.Array):
            sample = sample.reshape(shape)
            dtype = tensors[0][0].dtype
            envs = [None] * n_rows
            env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
            for row in range(n_rows - 1, -1, -1):
                envs[row] = env
                mpo = _build_row_mpo(tensors, sample[row], row, n_cols)
                env = _apply_mpo_from_below(env, mpo, model.strategy)
            return tuple(envs)

        coeffs_batch = None
        if coeffs is not None:
            coeffs_batch = jnp.broadcast_to(
                coeffs,
                (samples_flat.shape[0], coeffs.shape[0]),
            )
        return Cache(
            bottom_envs=jax.vmap(build_one_bottom_envs)(samples_flat),
            coeffs=coeffs_batch,
        )

    def transition(
        tensors: Any,
        sample: jax.Array,
        key: jax.Array,
        cache: Cache,
    ) -> tuple[jax.Array, jax.Array, Context]:
        """Transition kernel for a single Markov chain."""
        sample = sample.reshape(shape)
        dtype = tensors[0][0].dtype
        phys_dim = model.phys_dim
        top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
        top_envs_cache = [None] * n_rows

        def propose_index(
            key: jax.Array, idx: jax.Array, phys_dim: int
        ) -> tuple[jax.Array, jax.Array]:
            if phys_dim == 1:
                return key, idx
            if phys_dim == 2:
                return key, 1 - idx
            key, propose_key = jax.random.split(key)
            delta = jax.random.randint(propose_key, (), 1, phys_dim, dtype=jnp.int32)
            return key, (idx + delta) % phys_dim

        for row in range(n_rows):
            top_envs_cache[row] = top_env
            bottom_env = cache.bottom_envs[row]
            mpo_row = _build_row_mpo(tensors, sample[row], row, n_cols)
            right_envs = _compute_right_envs(top_env, mpo_row, bottom_env, dtype)
            left_env = jnp.ones((1, 1, 1), dtype=dtype)
            current_amplitude = jnp.einsum(
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
                current_idx = sample[row, col]
                key, proposed_idx = propose_index(key, current_idx, phys_dim)
                proposed_mpo = jnp.transpose(site_tensor[proposed_idx], (2, 3, 0, 1))
                proposed_amplitude = jnp.einsum(
                    "ace,aub,cduv,evf,bdf->",
                    left_env,
                    top_env[col],
                    proposed_mpo,
                    bottom_env[col],
                    right_envs[col],
                    optimize=[(0, 1), (1, 2), (1, 2), (0, 1)],
                )
                key, accept = _metropolis_hastings_accept(
                    key,
                    jnp.abs(current_amplitude) ** 2,
                    jnp.abs(proposed_amplitude) ** 2,
                )
                sample = sample.at[row, col].set(
                    jnp.where(accept, proposed_idx, current_idx)
                )
                current_amplitude = jnp.where(
                    accept, proposed_amplitude, current_amplitude
                )
                updated_mpo = jnp.where(accept, proposed_mpo, mpo_row[col])
                updated_row.append(updated_mpo)
                left_env = jnp.einsum(
                    "ace,aub,cduv,evf->bdf",
                    left_env,
                    top_env[col],
                    updated_mpo,
                    bottom_env[col],
                    optimize=[(0, 1), (0, 2), (0, 1)],
                )

            top_env = model.strategy.apply(top_env, tuple(updated_row))

        return sample.reshape(-1), key, Context(
            amp=_contract_bottom(top_env),
            top_envs=tuple(top_envs_cache),
            coeffs=cache.coeffs,
        )

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
            model.strategy,
            context.top_envs,
            terms=terms,
            coeffs=context.coeffs,
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
        return Cache(
            bottom_envs=tuple(bottom_envs_next),
            coeffs=context.coeffs,
        ), LocalEstimates(
            local_log_derivatives=local_log_derivatives,
            local_estimate=local_estimate,
            active_slice_indices=active_slice_indices,
            amp=context.amp,
        )

    return init_cache, transition, estimate
