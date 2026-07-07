"""GI-PEPS kernel dispatch extension for the generic MC sampler."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from vmc.operators.local_terms import merge_operators
from vmc.operators.time_dependent import TimeDependentHamiltonian
from vmc.peps.gi import model as gi_model
from vmc.peps.gi.local_terms import HorizontalHiggsLinkTerm, VerticalHiggsLinkTerm
from vmc.peps.gi.model import GIPEPS, _link_value_or_zero, _site_cfg_index
from vmc.peps.common.kernels import Cache, Context, LocalEstimates, _broadcast_coeffs
from vmc.peps.common.kernels import build_mc_kernels

__all__ = ["build_mc_kernels"]


def _validate_terms(model: GIPEPS, operators: tuple[object, ...]) -> None:
    has_higgs_terms = any(
        isinstance(term, (HorizontalHiggsLinkTerm, VerticalHiggsLinkTerm))
        for operator in operators
        for term in (
            operator.base.terms
            if isinstance(operator, TimeDependentHamiltonian)
            else operator.terms
        )
    )
    if not has_higgs_terms:
        return
    if not model.config.is_binary_occupancy_matter:
        raise ValueError(
            "Higgs link terms require binary-occupancy Z2 GIPEPS "
            "(N=2, phys_dim=2, charge_of_site=(0, 1))."
        )
    if model.config.conserve_particle_number:
        raise ValueError(
            "Higgs link terms require parity-sector GI updates "
            "(conserve_particle_number=False)."
        )


@build_mc_kernels.dispatch
def build_mc_kernels(
    model: GIPEPS,
    operator: object,
    *,
    full_gradient: bool = False,
    observables: tuple = (),
) -> tuple[Any, Any, Any]:
    """Build GI-PEPS init_cache/transition/estimate kernels."""
    shape = model.shape
    n_rows, n_cols = shape
    config = model.config
    strategy = model.strategy
    charge_of_site = jnp.asarray(model.charge_of_site, dtype=jnp.int32)
    charge_to_indices = jnp.asarray(model.charge_to_indices, dtype=jnp.int32)
    charge_deg = jnp.asarray(model.charge_deg, dtype=jnp.int32)
    mask_per_charge = (
        None
        if config.mask_per_charge is None
        else jnp.asarray(config.mask_per_charge, dtype=jnp.bool_)
    )
    nc_per_site = tuple(sd // model.phys_dim for sd in model.sliced_dims)
    all_operators = (operator,) + observables
    _validate_terms(model, all_operators)
    bucketed_terms, coeff_structure = merge_operators(
        all_operators,
        shape,
        eval_span=type(model).eval_span,
    )
    has_time_dep = any(s is not None for s in coeff_structure.schedules)
    static_coeffs = None if has_time_dep else coeff_structure.build_coeffs()

    def init_cache(
        tensors: Any,
        config_states: jax.Array,
        t: float | jax.Array | None = None,
    ) -> Cache:
        def build_one(config_state: jax.Array):
            sites, h_links, v_links = GIPEPS.unflatten_sample(config_state, shape)
            dtype = tensors[0][0].dtype
            envs = [None] * n_rows
            env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
            for row in range(n_rows - 1, -1, -1):
                envs[row] = env
                row_mpo = gi_model._build_row_mpo_gi(
                    tensors, sites, h_links, v_links, config, row, n_cols, mask_per_charge
                )
                env = gi_model._apply_mpo_from_below(env, row_mpo, strategy)
            return tuple(envs)

        return Cache(
            bottom_envs=jax.vmap(build_one)(config_states),
            coeffs=_broadcast_coeffs(
                None if not has_time_dep else coeff_structure.build_coeffs(t),
                config_states.shape[0],
            ),
        )

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
            mask_per_charge,
            charge_of_site,
            charge_to_indices,
            charge_deg,
        )
        return config_state_next, key_next, Context(
            amp=amp,
            top_envs=top_envs,
            coeffs=cache.coeffs,
        )

    def estimate(
        tensors: Any,
        config_state_next: jax.Array,
        context: Context,
    ) -> tuple[Cache, LocalEstimates]:
        env_grads, local_energy, envs_next = gi_model.estimate(
            tensors,
            config_state_next,
            context.amp,
            config,
            strategy,
            context.top_envs,
            mask_per_charge,
            terms=bucketed_terms,
            coeffs=static_coeffs if context.coeffs is None else context.coeffs,
        )
        sites, h_links, v_links = GIPEPS.unflatten_sample(config_state_next, shape)
        grad_parts = []
        p_parts = [] if not full_gradient else None
        for r in range(n_rows):
            for c in range(n_cols):
                site = sites[r, c]
                env_grad = env_grads[r][c]
                k_l = _link_value_or_zero(h_links, v_links, r, c, direction="left")
                k_r = _link_value_or_zero(h_links, v_links, r, c, direction="right")
                k_u = _link_value_or_zero(h_links, v_links, r, c, direction="up")
                k_d = _link_value_or_zero(h_links, v_links, r, c, direction="down")
                cfg_idx = _site_cfg_index(
                    config, k_l=k_l, k_u=k_u, k_r=k_r, k_d=k_d, r=r, c=c
                )
                if full_gradient:
                    grad_full = jnp.zeros_like(jnp.asarray(tensors[r][c]))
                    grad_full = grad_full.at[site, cfg_idx].set(env_grad)
                    grad_parts.append(grad_full.reshape(-1))
                    continue
                grad_parts.append(env_grad.reshape(-1))
                combined_idx = site * nc_per_site[r * n_cols + c] + cfg_idx
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
