"""Blockade-PEPS kernel dispatch extension for the generic MC sampler."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from vmc.operators.local_terms import merge_operators
from vmc.peps.blockade import model as blockade_model
from vmc.peps.blockade.model import BlockadePEPS
from vmc.peps.common.contraction import _apply_mpo_from_below
from vmc.peps.common.energy import _estimate_sweep
from vmc.peps.standard.kernels import Cache, Context, LocalEstimates, build_mc_kernels

__all__ = ["build_mc_kernels"]


@build_mc_kernels.dispatch
def build_mc_kernels(
    model: BlockadePEPS,
    operator: object,
    *,
    full_gradient: bool = False,
    observables: tuple = (),
) -> tuple[Any, Any, Any]:
    """Build blockade-PEPS init_cache/transition/estimate kernels."""
    peps_config = model.config
    shape = peps_config.shape
    n_rows, n_cols = shape
    strategy = model.strategy

    all_operators = (operator,) + observables
    terms, coeff_structure = merge_operators(
        all_operators, shape, eval_span=type(model).eval_span,
    )
    has_time_dep = any(s is not None for s in coeff_structure.schedules)

    def init_cache(
        tensors: Any,
        config_states: jax.Array,
        t: float | jax.Array | None = None,
    ) -> Cache:
        config_states_flat = config_states.reshape(config_states.shape[0], n_rows * n_cols)

        def build_one(config_state: jax.Array):
            indices = BlockadePEPS.unflatten_sample(config_state, shape)
            dtype = tensors[0][0].dtype
            envs = [None] * n_rows
            env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
            for row in range(n_rows - 1, -1, -1):
                envs[row] = env
                row_mpo = blockade_model._build_row_mpo(tensors, indices, peps_config, row)
                env = _apply_mpo_from_below(env, row_mpo, strategy)
            return tuple(envs)

        coeffs_batch = None
        if has_time_dep:
            coeffs = coeff_structure.build_coeffs(t)
            coeffs_batch = jnp.broadcast_to(
                coeffs,
                (config_states_flat.shape[0], coeffs.shape[0]),
            )
        return Cache(
            bottom_envs=jax.vmap(build_one)(config_states_flat),
            coeffs=coeffs_batch,
        )

    def transition(
        tensors: Any,
        config_state: jax.Array,
        key: jax.Array,
        cache: Cache,
    ) -> tuple[jax.Array, jax.Array, Context]:
        config_state_next, key_next, amp, top_envs = blockade_model.transition(
            tensors,
            config_state,
            key,
            cache.bottom_envs,
            peps_config,
            strategy,
        )
        return config_state_next, key_next, Context(amp=amp, top_envs=top_envs, coeffs=cache.coeffs)

    def estimate(
        tensors: Any,
        config_state_next: jax.Array,
        context: Context,
    ) -> tuple[Cache, LocalEstimates]:
        config_state = BlockadePEPS.unflatten_sample(config_state_next, shape)

        def build_row_mpo(
            tensors,
            sample,
            row,
        ):
            return blockade_model._build_row_mpo(tensors, sample, peps_config, row)

        env_grads, local_energy, envs_next = _estimate_sweep(
            tensors,
            config_state,
            context.amp,
            context.top_envs,
            strategy=strategy,
            terms=terms,
            build_row_mpo=build_row_mpo,
            eval_term=blockade_model._eval_blockade_term,
            env_config=peps_config,
            coeffs=context.coeffs,
            collect_grads=True,
        )
        indices = config_state_next.reshape(shape)
        grad_parts = []
        p_parts = []
        for row in range(n_rows):
            for col in range(n_cols):
                k_l = indices[row, col - 1] if col > 0 else 0
                k_u = indices[row - 1, col] if row > 0 else 0
                cfg_idx_n0 = k_l * (2 if row > 0 else 1) + k_u
                cfg_idx = jnp.where(indices[row, col] == 0, cfg_idx_n0, 0)
                env_grad = env_grads[row][col]
                if full_gradient:
                    grad_full = jnp.zeros_like(jnp.asarray(tensors[row][col]))
                    grad_parts.append(
                        grad_full.at[indices[row, col], cfg_idx].set(env_grad).reshape(-1)
                    )
                else:
                    grad_parts.append(env_grad.reshape(-1))
                    combined_idx = indices[row, col] * jnp.asarray(tensors[row][col]).shape[1] + cfg_idx
                    p_parts.append(
                        jnp.full((env_grad.size,), combined_idx, dtype=jnp.int16)
                    )
        local_log_derivatives = jnp.concatenate(grad_parts) / context.amp
        active_slice_indices = None if full_gradient else jnp.concatenate(p_parts)
        return Cache(bottom_envs=tuple(envs_next), coeffs=context.coeffs), LocalEstimates(
            local_log_derivatives=local_log_derivatives,
            local_estimate=local_energy,
            active_slice_indices=active_slice_indices,
            amp=context.amp,
        )

    return init_cache, transition, estimate
