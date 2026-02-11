"""GI-PEPS kernel dispatch extension for the generic MC sampler."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from vmc.operators.local_terms import bucket_operators
from vmc.operators.time_dependent import TimeDependentHamiltonian
from vmc.peps.gi import model as gi_model
from vmc.peps.gi.local_terms import GILocalHamiltonian
from vmc.peps.gi.model import GIPEPS, _link_value_or_zero, _site_cfg_index
from vmc.peps.standard.kernels import Cache, Context, LocalEstimates, build_mc_kernels

__all__ = ["build_mc_kernels"]


@build_mc_kernels.dispatch
def build_mc_kernels(
    model: GIPEPS,
    operator: GILocalHamiltonian,
    *,
    full_gradient: bool = False,
) -> tuple[Any, Any, Any]:
    """Build GI-PEPS init_cache/transition/estimate kernels."""
    shape = model.shape
    n_rows, n_cols = shape
    config = model.config
    strategy = model.strategy
    charge_of_site = jnp.asarray(model.charge_of_site, dtype=jnp.int32)
    charge_to_indices = model.charge_to_indices
    charge_deg = model.charge_deg
    bucketed_terms = bucket_operators(
        operator.terms,
        shape,
        eval_span=lambda op: type(model).eval_span(op),
    )

    def init_cache(
        tensors: Any,
        config_states: jax.Array,
        coeffs: jax.Array | None = None,
    ) -> Cache:
        del coeffs
        def build_one(config_state: jax.Array):
            sites, h_links, v_links = GIPEPS.unflatten_sample(config_state, shape)
            dtype = tensors[0][0].dtype
            envs = [None] * n_rows
            env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
            for row in range(n_rows - 1, -1, -1):
                envs[row] = env
                row_mpo = gi_model._build_row_mpo_gi(
                    tensors, sites, h_links, v_links, config, row, n_cols
                )
                env = gi_model._apply_mpo_from_below(env, row_mpo, strategy)
            return tuple(envs)

        return Cache(bottom_envs=jax.vmap(build_one)(config_states))

    def transition(
        tensors: Any,
        config_state: jax.Array,
        key: jax.Array,
        cache: Cache,
    ) -> tuple[jax.Array, jax.Array, Context]:
        config_state_next, key_next, amp, top_envs = gi_model.transition(
            tensors,
            config_state,
            key,
            cache.bottom_envs,
            shape,
            config,
            strategy,
            charge_of_site,
            charge_to_indices,
            charge_deg,
        )
        return config_state_next, key_next, Context(amp=amp, top_envs=top_envs)

    def estimate(
        tensors: Any,
        config_state_next: jax.Array,
        context: Context,
    ) -> tuple[Cache, LocalEstimates]:
        env_grads, local_energy, envs_next = gi_model.estimate(
            tensors,
            config_state_next,
            context.amp,
            operator,
            shape,
            config,
            strategy,
            context.top_envs,
            terms=bucketed_terms,
        )
        sites, h_links, v_links = GIPEPS.unflatten_sample(config_state_next, shape)
        if full_gradient:
            grad_parts = []
            for r in range(n_rows):
                for c in range(n_cols):
                    k_l = _link_value_or_zero(h_links, v_links, r, c, direction="left")
                    k_r = _link_value_or_zero(h_links, v_links, r, c, direction="right")
                    k_u = _link_value_or_zero(h_links, v_links, r, c, direction="up")
                    k_d = _link_value_or_zero(h_links, v_links, r, c, direction="down")
                    cfg_idx = _site_cfg_index(
                        config, k_l=k_l, k_u=k_u, k_r=k_r, k_d=k_d, r=r, c=c
                    )
                    grad_full = jnp.zeros_like(jnp.asarray(tensors[r][c]))
                    grad_full = grad_full.at[sites[r, c], cfg_idx].set(env_grads[r][c])
                    grad_parts.append(grad_full.reshape(-1))
            local_log_derivatives = jnp.concatenate(grad_parts) / context.amp
            active_slice_indices = None
        else:
            grad_parts = []
            p_parts = []
            for r in range(n_rows):
                for c in range(n_cols):
                    grad_parts.append(env_grads[r][c].reshape(-1))
                    params_per_site = env_grads[r][c].size
                    k_l = _link_value_or_zero(h_links, v_links, r, c, direction="left")
                    k_r = _link_value_or_zero(h_links, v_links, r, c, direction="right")
                    k_u = _link_value_or_zero(h_links, v_links, r, c, direction="up")
                    k_d = _link_value_or_zero(h_links, v_links, r, c, direction="down")
                    cfg_idx = _site_cfg_index(
                        config, k_l=k_l, k_u=k_u, k_r=k_r, k_d=k_d, r=r, c=c
                    )
                    combined_idx = sites[r, c] * jnp.asarray(tensors[r][c]).shape[1] + cfg_idx
                    p_parts.append(jnp.full((params_per_site,), combined_idx, dtype=jnp.int16))
            local_log_derivatives = jnp.concatenate(grad_parts) / context.amp
            active_slice_indices = jnp.concatenate(p_parts)
        return Cache(bottom_envs=tuple(envs_next)), LocalEstimates(
            local_log_derivatives=local_log_derivatives,
            local_estimate=local_energy,
            active_slice_indices=active_slice_indices,
            amp=context.amp,
        )

    return init_cache, transition, estimate


@build_mc_kernels.dispatch
def build_mc_kernels(
    model: GIPEPS,
    operator: TimeDependentHamiltonian,
    *,
    full_gradient: bool = False,
) -> tuple[Any, Any, Any]:
    del model, operator, full_gradient
    raise NotImplementedError("TimeDependentHamiltonian is not implemented for GIPEPS.")
