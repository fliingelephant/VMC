"""GI-PEPS sequential sampler with integrated energy/gradients."""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from plum import dispatch

from vmc.experimental.lgt.gi_local_terms import GILocalHamiltonian
from vmc.experimental.lgt.gi_peps import (
    GIPEPS,
    _link_value_or_zero,
    _site_cfg_index,
)
from vmc.models.peps import bottom_envs, grads_and_energy, sweep
from vmc.samplers.sequential import (
    _collect_steps,
    _sample_counts,
    _trim_samples,
)


@functools.partial(
    jax.jit,
    static_argnames=("n_samples", "n_chains", "burn_in", "full_gradient"),
)
@dispatch
def sequential_sample_with_gradients(
    model: GIPEPS,
    operator: GILocalHamiltonian,
    *,
    n_samples: int = 1,
    n_chains: int = 1,
    key: jax.Array,
    initial_configuration: jax.Array,
    burn_in: int = 0,
    full_gradient: bool = True,
):
    """Sequential sampling for GI-PEPS with per-sample gradient recording."""
    num_samples, num_chains, num_burn_in, chain_length, total_samples = _sample_counts(
        n_samples, n_chains, burn_in
    )
    shape = model.shape
    n_rows, n_cols = shape
    config = model.config
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]

    key, chain_key = jax.random.split(key)
    samples_flat = initial_configuration
    chain_keys = jax.random.split(chain_key, num_chains)
    envs = jax.vmap(lambda s: bottom_envs(model, s))(samples_flat)

    def flatten_full_gradients(env_grads, sample, amp):
        sites, h_links, v_links = GIPEPS.unflatten_sample(sample, shape)
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
                grad_site = jnp.zeros_like(tensors[r][c])
                grad_site = grad_site.at[sites[r, c], cfg_idx].set(env_grads[r][c])
                grad_parts.append(grad_site.reshape(-1))
        return jnp.concatenate(grad_parts) / amp, jnp.zeros((0,), dtype=jnp.int8)

    def flatten_sliced_gradients(env_grads, sample, amp):
        sites, h_links, v_links = GIPEPS.unflatten_sample(sample, shape)
        grad_parts = []
        p_parts = []
        for r in range(n_rows):
            for c in range(n_cols):
                grad_parts.append(env_grads[r][c].reshape(-1))
                params_per_site = env_grads[r][c].size
                # Compute cfg_idx for this site's link configuration
                k_l = _link_value_or_zero(h_links, v_links, r, c, direction="left")
                k_r = _link_value_or_zero(h_links, v_links, r, c, direction="right")
                k_u = _link_value_or_zero(h_links, v_links, r, c, direction="up")
                k_d = _link_value_or_zero(h_links, v_links, r, c, direction="down")
                cfg_idx = _site_cfg_index(
                    config, k_l=k_l, k_u=k_u, k_r=k_r, k_d=k_d, r=r, c=c
                )
                # Encode combined index: phys_idx * Nc + cfg_idx
                nc = tensors[r][c].shape[1]
                combined_idx = sites[r, c] * nc + cfg_idx
                p_parts.append(
                    jnp.full((params_per_site,), combined_idx, dtype=jnp.int16)
                )
        return jnp.concatenate(grad_parts) / amp, jnp.concatenate(p_parts)

    flatten_grads = flatten_full_gradients if full_gradient else flatten_sliced_gradients

    def mc_sweep(sample, key, envs):
        """Single MC sweep: Metropolis sweep + gradient/energy + flatten (single vmap)."""
        sample, key, amp = sweep(model, sample, key, envs)
        envs = bottom_envs(model, sample)
        env_grads, local_energy = grads_and_energy(model, sample, amp, operator, envs)
        grad_row, p_row = flatten_grads(env_grads, sample, amp)
        return sample, key, envs, grad_row, p_row, amp, local_energy

    def burn_step(carry, _):
        samples, chain_keys, envs = carry
        samples, chain_keys, envs, _, _, _, _ = jax.vmap(mc_sweep)(
            samples, chain_keys, envs
        )
        return (samples, chain_keys, envs), None

    (samples_flat, chain_keys, envs), _ = jax.lax.scan(
        burn_step,
        (samples_flat, chain_keys, envs),
        xs=None,
        length=num_burn_in,
    )

    def sample_step(carry, _):
        samples, chain_keys, envs = carry
        samples, chain_keys, envs, grad_row, p_row, amp, local_energy = jax.vmap(
            mc_sweep
        )(samples, chain_keys, envs)
        return (samples, chain_keys, envs), (
            samples,
            grad_row,
            p_row,
            amp,
            local_energy,
        )

    (final_samples, _, _), (samples_out, grads, p_rows, amps, energies) = _collect_steps(
        sample_step,
        (samples_flat, chain_keys, envs),
        chain_length,
        "Sequential sampling",
    )

    samples_out = _trim_samples(samples_out, total_samples, num_samples)
    grads = _trim_samples(grads, total_samples, num_samples)
    amps = _trim_samples(amps, total_samples, num_samples)
    energies = _trim_samples(energies, total_samples, num_samples)
    p = None if full_gradient else _trim_samples(p_rows, total_samples, num_samples)

    return (
        samples_out,
        grads,
        p,
        key,
        final_samples,
        amps,
        energies,
    )
