"""Sequential sampler dispatches for BlockadePEPS.

These dispatches enable BlockadePEPS to work with the unified sampler API.
"""
from __future__ import annotations

import functools
import logging

import jax
import jax.numpy as jnp
from plum import dispatch

from vmc.experimental.rydberg.blockade_peps import BlockadePEPS
from vmc.models.peps import bottom_envs, grads_and_energy, sweep
from vmc.operators.local_terms import LocalHamiltonian
from vmc.samplers.sequential import (
    _collect_steps,
    _sample_counts,
    _trim_samples,
)
from vmc.utils.smallo import params_per_site as params_per_site_fn

__all__ = [
    "sequential_sample",
    "sequential_sample_with_gradients",
]

logger = logging.getLogger(__name__)


@functools.partial(jax.jit, static_argnames=("n_samples", "n_chains", "burn_in"))
@dispatch
def sequential_sample(
    model: BlockadePEPS,
    *,
    n_samples: int = 1,
    n_chains: int = 1,
    key: jax.Array,
    initial_configuration: jax.Array,
    burn_in: int = 0,
):
    """Sequential sampling for BlockadePEPS using 2-row Metropolis sweeps."""
    num_samples, num_chains, num_burn_in, chain_length, total_samples = _sample_counts(
        n_samples, n_chains, burn_in
    )
    shape = model.shape
    n_rows, n_cols = shape
    n_sites = int(n_rows * n_cols)

    key, chain_key = jax.random.split(key)
    # Convert to flat samples for unified API
    samples_flat = initial_configuration.reshape(num_chains, n_sites)
    chain_keys = jax.random.split(chain_key, num_chains)
    envs = jax.vmap(lambda s: bottom_envs(model, s))(samples_flat)

    def sweep_once(sample, key, envs):
        sample, key, _ = sweep(model, sample, key, envs)
        envs = bottom_envs(model, sample)
        return sample, key, envs

    def burn_step(carry, _):
        samples, chain_keys, envs = carry
        samples, chain_keys, envs = jax.vmap(sweep_once)(samples, chain_keys, envs)
        return (samples, chain_keys, envs), None

    (samples_flat, chain_keys, envs), _ = jax.lax.scan(
        burn_step,
        (samples_flat, chain_keys, envs),
        xs=None,
        length=num_burn_in,
    )

    def sample_step(carry, _):
        samples, chain_keys, envs = carry
        samples, chain_keys, envs = jax.vmap(sweep_once)(samples, chain_keys, envs)
        return (samples, chain_keys, envs), samples

    (_, _, _), samples_out = _collect_steps(
        sample_step,
        (samples_flat, chain_keys, envs),
        chain_length,
        "Sequential sampling (BlockadePEPS)",
    )
    samples_out = _trim_samples(samples_out, total_samples, num_samples)
    return samples_out


@functools.partial(
    jax.jit,
    static_argnames=("n_samples", "n_chains", "burn_in", "full_gradient"),
)
@dispatch
def sequential_sample_with_gradients(
    model: BlockadePEPS,
    operator: LocalHamiltonian,
    *,
    n_samples: int = 1,
    n_chains: int = 1,
    key: jax.Array,
    initial_configuration: jax.Array,
    burn_in: int = 0,
    full_gradient: bool = False,
) -> (
    tuple[
        jax.Array,
        jax.Array,
        jax.Array | None,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]
):
    """Sequential sampling for BlockadePEPS with per-sample gradient recording.

    Total samples are ``n_samples`` across chains (burn-in sweeps are not recorded).

    Args:
        operator: Local Hamiltonian used for on-the-fly local energy evaluation.
        initial_configuration: Initial chain configs, shape (n_chains, n_rows, n_cols),
            in occupancy format (0 or 1 for blockade).

    Returns:
        samples, grads, p, key, final_configurations, amps, local_energies
    """
    num_samples, num_chains, num_burn_in, chain_length, total_samples = _sample_counts(
        n_samples, n_chains, burn_in
    )
    shape = model.shape
    n_rows, n_cols = shape
    n_sites = int(n_rows * n_cols)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    params_per_site = tuple(int(p) for p in params_per_site_fn(model))

    key, chain_key = jax.random.split(key)
    # Convert to flat samples for unified API
    samples_flat = initial_configuration.reshape(num_chains, n_sites)
    chain_keys = jax.random.split(chain_key, num_chains)
    envs = jax.vmap(lambda s: bottom_envs(model, s))(samples_flat)

    def flatten_full_gradients(env_grads, sample, amp):
        indices = sample.reshape(shape)
        grad_parts = []
        for row in range(n_rows):
            for col in range(n_cols):
                grad_full = jnp.zeros_like(tensors[row][col])
                grad_full = grad_full.at[indices[row, col]].set(env_grads[row][col])
                grad_parts.append(grad_full.ravel())
        return jnp.concatenate(grad_parts) / amp, jnp.zeros((0,), dtype=jnp.int8)

    def flatten_sliced_gradients(env_grads, sample, amp):
        grad_parts = [
            env_grads[row][col].reshape(-1)
            for row in range(n_rows)
            for col in range(n_cols)
        ]
        p_parts = [
            jnp.full((params_per_site[site],), sample[site], dtype=jnp.int8)
            for site in range(n_sites)
        ]
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

    (final_samples, _, _), (samples_out, grads, p, amps, local_energies) = _collect_steps(
        sample_step,
        (samples_flat, chain_keys, envs),
        chain_length,
        "Sequential sampling (BlockadePEPS)",
    )

    samples_out = _trim_samples(samples_out, total_samples, num_samples)
    grads = _trim_samples(grads, total_samples, num_samples)
    amps = _trim_samples(amps, total_samples, num_samples)
    local_energies = _trim_samples(local_energies, total_samples, num_samples)
    if full_gradient:
        p = None
    else:
        p = _trim_samples(p, total_samples, num_samples)

    return samples_out, grads, p, key, final_samples, amps, local_energies
