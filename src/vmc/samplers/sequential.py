"""Sequential Metropolis samplers for MPS/PEPS."""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import functools
import logging
import math

import jax
import jax.numpy as jnp
from plum import dispatch

from vmc.models.mps import MPS
from vmc.models.peps import (
    PEPS,
    _metropolis_ratio,
    bottom_envs,
    grads_and_energy,
    sweep,
)
from vmc.operators import LocalHamiltonian, bucket_terms
from vmc.utils.smallo import params_per_site as params_per_site_fn
from vmc.utils.vmc_utils import local_estimate

__all__ = [
    "sequential_sample",
    "sequential_sample_with_gradients",
]
logger = logging.getLogger(__name__)

def _collect_steps(step_fn, carry, count: int, desc: str):
    if logger.isEnabledFor(logging.INFO):
        logger.info("%s: sampling %d steps", desc, count)
    result = jax.lax.scan(step_fn, carry, xs=None, length=count)
    if logger.isEnabledFor(logging.INFO):
        logger.info("%s: done", desc)
    return result


def _sample_counts(n_samples: int, n_chains: int, burn_in: int) -> tuple[int, int, int, int, int]:
    chain_length = math.ceil(n_samples / n_chains)
    return n_samples, n_chains, burn_in, chain_length, chain_length * n_chains


def _trim_samples(samples: jax.Array, total_samples: int, num_samples: int) -> jax.Array:
    return samples.reshape((total_samples,) + samples.shape[2:])[:num_samples]


def _collect_samples(
    sweep_batched,
    state: jax.Array,
    chain_keys: jax.Array,
    *,
    num_burn_in: int,
    chain_length: int,
    sample_view,
):
    # Burn-in sweeps
    def burn_step(carry, _):
        state, key = carry
        state, key = sweep_batched(state, key)
        return (state, key), None

    (state, chain_keys), _ = jax.lax.scan(
        burn_step, (state, chain_keys), xs=None, length=num_burn_in
    )

    # Collect samples
    def sample_step(carry, _):
        state, chain_keys = carry
        state, chain_keys = sweep_batched(state, chain_keys)
        return (state, chain_keys), sample_view(state)

    (_, _), samples = _collect_steps(
        sample_step,
        (state, chain_keys),
        chain_length,
        "Sequential sampling",
    )
    return state, samples


def _run_burn_in_with_envs(
    sweep_with_envs,
    cache_envs_fn,
    state: jax.Array,
    chain_keys: jax.Array,
    cached_envs,
    *,
    num_burn_in: int,
):
    def step(carry, _):
        state, chain_keys, cached_envs = carry
        state, chain_keys, _, _ = sweep_with_envs(
            state, chain_keys, cached_envs, False
        )
        cached_envs = jax.vmap(cache_envs_fn)(state)
        return (state, chain_keys, cached_envs), None

    (state, chain_keys, cached_envs), _ = jax.lax.scan(
        step, (state, chain_keys, cached_envs), xs=None, length=num_burn_in
    )
    return state, chain_keys, cached_envs


def _pad_mps_tensors(tensors: list[jax.Array], bond_dim: int) -> jax.Array:
    """Pad MPS tensors to a uniform bond dimension for JAX scans."""
    phys_dim = int(tensors[0].shape[0])
    def pad_tensor(tensor):
        block = jnp.zeros((phys_dim, bond_dim, bond_dim), dtype=tensor.dtype)
        return block.at[:, :tensor.shape[1], :tensor.shape[2]].set(tensor)
    return jnp.stack([pad_tensor(t) for t in tensors], axis=0)


@functools.partial(jax.jit, static_argnames=("n_sites",))
def _mps_right_envs(
    tensors: jax.Array,
    indices: jax.Array,
    *,
    n_sites: int,
):
    bond_dim = tensors.shape[-1]
    right_end = jnp.zeros((bond_dim,), dtype=tensors.dtype).at[0].set(1.0)

    def right_step(carry, site):
        tensor = tensors[site, indices[site]]
        right_env = jnp.einsum("ij,j->i", tensor, carry)
        return right_env, right_env

    _, right_envs_rev = jax.lax.scan(
        right_step, right_end, jnp.arange(n_sites - 1, -1, -1)
    )
    right_envs = jnp.flip(right_envs_rev, axis=0)
    return jnp.concatenate([right_envs, right_end[None, :]], axis=0)


@functools.partial(jax.jit, static_argnames=("n_sites", "collect_left_envs"))
def _sequential_mps_sweep_with_envs(
    tensors: jax.Array,
    indices: jax.Array,
    right_envs: jax.Array,
    *,
    key: jax.Array,
    n_sites: int,
    collect_left_envs: bool,
):
    """Run a sequential Metropolis sweep with fixed site order."""
    phys_dim = int(tensors.shape[1])
    left_env0 = right_envs[-1]
    site_ids = jnp.arange(n_sites)

    def sweep_step(carry, site):
        indices, left_env, key = carry
        left_env_before = left_env
        right_env = right_envs[site + 1]
        cur_idx = indices[site]
        if phys_dim == 1:
            flip_idx = cur_idx
        elif phys_dim == 2:
            flip_idx = 1 - cur_idx
        else:
            key, flip_key = jax.random.split(key)
            delta = jax.random.randint(flip_key, (), 1, phys_dim, dtype=jnp.int32)
            flip_idx = (cur_idx + delta) % phys_dim
        tensor_cur = tensors[site, cur_idx]
        tensor_flip = tensors[site, flip_idx]
        amp_cur = jnp.einsum("i,ij,j->", left_env, tensor_cur, right_env, optimize=[(0, 1), (0, 1)])
        amp_flip = jnp.einsum("i,ij,j->", left_env, tensor_flip, right_env, optimize=[(0, 1), (0, 1)])
        prob_cur = jnp.abs(amp_cur) ** 2
        prob_flip = jnp.abs(amp_flip) ** 2
        ratio = _metropolis_ratio(prob_cur, prob_flip)

        key, accept_key = jax.random.split(key)
        accept = jax.random.uniform(accept_key) < jnp.minimum(1.0, ratio)
        new_idx = jnp.where(accept, flip_idx, cur_idx)
        indices = indices.at[site].set(new_idx)

        tensor_sel = jnp.where(accept, tensor_flip, tensor_cur)
        left_env = jnp.einsum("i,ij->j", left_env, tensor_sel)
        return (indices, left_env, key), left_env_before

    if collect_left_envs:
        (indices, left_env, key), left_envs = jax.lax.scan(
            sweep_step, (indices, left_env0, key), site_ids
        )
    else:
        def sweep_step_no_collect(carry, site):
            carry, _ = sweep_step(carry, site)
            return carry, None

        (indices, left_env, key), _ = jax.lax.scan(
            sweep_step_no_collect, (indices, left_env0, key), site_ids
        )
        left_envs = ()
    return indices, key, left_envs, left_env


@functools.partial(jax.jit, static_argnames=("n_sites",))
def _sequential_mps_sweep(
    tensors: jax.Array,
    indices: jax.Array,
    *,
    key: jax.Array,
    n_sites: int,
):
    right_envs = _mps_right_envs(tensors, indices, n_sites=n_sites)
    indices, key, _, _ = _sequential_mps_sweep_with_envs(
        tensors,
        indices,
        right_envs,
        key=key,
        n_sites=n_sites,
        collect_left_envs=False,
    )
    return indices, key


@functools.partial(jax.jit, static_argnames=("n_samples", "n_chains", "burn_in"))
@dispatch
def sequential_sample(
    model: MPS,
    *,
    n_samples: int = 1,
    n_chains: int = 1,
    key: jax.Array,
    initial_configuration: jax.Array,
    burn_in: int = 0,
):
    """Sequential sampling for MPS using Metropolis sweeps."""
    num_samples, num_chains, num_burn_in, chain_length, total_samples = _sample_counts(
        n_samples, n_chains, burn_in
    )
    tensors = [jnp.asarray(t) for t in model.tensors]
    n_sites = len(tensors)
    tensors_padded = _pad_mps_tensors(tensors, model.bond_dim)

    key, chain_key = jax.random.split(key)
    indices = initial_configuration
    chain_keys = jax.random.split(chain_key, num_chains)

    def sweep_once(indices, key):
        return _sequential_mps_sweep(
            tensors_padded,
            indices,
            key=key,
            n_sites=n_sites,
        )

    sweep_batched = jax.vmap(sweep_once, in_axes=(0, 0))
    _, samples = _collect_samples(
        sweep_batched,
        indices,
        chain_keys,
        num_burn_in=num_burn_in,
        chain_length=chain_length,
        sample_view=lambda x: x,
    )
    # TODO: properly distribute n_samples, avoid sampling overhead.
    samples = _trim_samples(samples, total_samples, num_samples)
    return samples


@functools.partial(jax.jit, static_argnames=("n_samples", "n_chains", "burn_in"))
@dispatch
def sequential_sample(
    model: PEPS,
    *,
    n_samples: int = 1,
    n_chains: int = 1,
    key: jax.Array,
    initial_configuration: jax.Array,
    burn_in: int = 0,
):
    """Sequential sampling for PEPS using Metropolis sweeps."""
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
        sample, key, _, _ = sweep(model, sample, key, envs)
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
        "Sequential sampling",
    )
    # TODO: properly distribute n_samples, avoid sampling overhead.
    samples_out = _trim_samples(samples_out, total_samples, num_samples)
    return samples_out


@functools.partial(
    jax.jit,
    static_argnames=("n_samples", "n_chains", "burn_in", "full_gradient"),
)
@dispatch
def sequential_sample_with_gradients(
    model: MPS,
    operator: object,
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
    """Sequential sampling for MPS with per-sample gradient recording.

    Total samples are ``n_samples`` across chains (burn-in sweeps are not recorded).

    Args:
        operator: Operator used to compute local energies after sampling.
        initial_configuration: Initial chain configs, shape (n_chains, n_sites),
            in occupancy format (0..phys_dim-1).

    Returns:
        samples, grads, p, key, final_configurations, amps, local_energies
    """
    num_samples, num_chains, num_burn_in, chain_length, total_samples = _sample_counts(
        n_samples, n_chains, burn_in
    )
    tensors = [jnp.asarray(t) for t in model.tensors]
    n_sites = len(tensors)
    bond_dim = model.bond_dim
    phys_dim = model.phys_dim
    tensors_padded = _pad_mps_tensors(tensors, bond_dim)
    params_per_site = tuple(int(p) for p in params_per_site_fn(model))

    key, chain_key = jax.random.split(key)
    indices = initial_configuration
    chain_keys = jax.random.split(chain_key, num_chains)
    right_envs = jax.vmap(
        lambda idx: _mps_right_envs(tensors_padded, idx, n_sites=n_sites)
    )(indices)

    def sweep_with_envs(indices, chain_keys, right_envs, collect_left_envs):
        def sweep_single(idx, key, right_env):
            return _sequential_mps_sweep_with_envs(
                tensors_padded,
                idx,
                right_env,
                key=key,
                n_sites=n_sites,
                collect_left_envs=collect_left_envs,
            )

        return jax.vmap(sweep_single, in_axes=(0, 0, 0))(
            indices, chain_keys, right_envs
        )

    indices, chain_keys, right_envs = _run_burn_in_with_envs(
        sweep_with_envs,
        lambda idx: _mps_right_envs(tensors_padded, idx, n_sites=n_sites),
        indices,
        chain_keys,
        right_envs,
        num_burn_in=num_burn_in,
    )

    def compute_site_grad(site, left_envs, right_envs):
        left_dim, right_dim = MPS.site_dims(site, n_sites, bond_dim)
        left = left_envs[site][:left_dim]
        right = right_envs[site + 1][:right_dim]
        return left[:, None] * right[None, :]

    def flatten_full_gradients(indices, left_envs, right_envs):
        grad_parts = []
        for site in range(n_sites):
            grad_site = compute_site_grad(site, left_envs, right_envs)
            left_dim, right_dim = MPS.site_dims(site, n_sites, bond_dim)
            grad_full = jnp.zeros((phys_dim, left_dim, right_dim), dtype=tensors[0].dtype)
            grad_full = grad_full.at[indices[site]].set(grad_site)
            grad_parts.append(grad_full.ravel())
        return jnp.concatenate(grad_parts)

    def flatten_sliced_gradients(indices, left_envs, right_envs):
        grad_parts = [
            compute_site_grad(site, left_envs, right_envs).reshape(-1)
            for site in range(n_sites)
        ]
        p_parts = [
            jnp.full((params_per_site[site],), indices[site], dtype=jnp.int8)
            for site in range(n_sites)
        ]
        return jnp.concatenate(grad_parts), jnp.concatenate(p_parts)

    def sample_step(carry, _):
        indices, chain_keys, right_envs = carry
        indices, chain_keys, left_envs, left_env = sweep_with_envs(
            indices, chain_keys, right_envs, True
        )
        right_envs_next = jax.vmap(
            lambda idx: _mps_right_envs(tensors_padded, idx, n_sites=n_sites)
        )(indices)
        amp = left_env[:, 0]
        if full_gradient:
            grad_row = jax.vmap(flatten_full_gradients)(
                indices, left_envs, right_envs_next
            )
            p_row = jnp.zeros((amp.shape[0], 0), dtype=jnp.int8)
        else:
            grad_row, p_row = jax.vmap(flatten_sliced_gradients)(
                indices, left_envs, right_envs_next
            )
        grad_row = grad_row / amp[:, None]
        return (indices, chain_keys, right_envs_next), (
            indices,
            grad_row,
            p_row,
            amp,
        )

    (final_configurations, _, _), (samples, grads, p, amps) = _collect_steps(
        sample_step,
        (indices, chain_keys, right_envs),
        chain_length,
        "Sequential sampling",
    )

    # TODO: properly distribute n_samples, avoid sampling overhead.
    samples = _trim_samples(samples, total_samples, num_samples)
    grads = _trim_samples(grads, total_samples, num_samples)
    amps = _trim_samples(amps, total_samples, num_samples)
    if full_gradient:
        p = None
    else:
        p = _trim_samples(p, total_samples, num_samples)

    local_energies = local_estimate(model, samples, operator, amps)
    return samples, grads, p, key, final_configurations, amps, local_energies


@functools.partial(
    jax.jit,
    static_argnames=("n_samples", "n_chains", "burn_in", "full_gradient"),
)
@dispatch
def sequential_sample_with_gradients(
    model: PEPS,
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
    """Sequential sampling for PEPS with per-sample gradient recording.

    Total samples are ``n_samples`` across chains (burn-in sweeps are not recorded).

    Args:
        operator: Local Hamiltonian used for on-the-fly local energy evaluation.
        initial_configuration: Initial chain configs, shape (n_chains, n_rows, n_cols),
            in occupancy format (0..phys_dim-1).

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
        sample, key, amp, top_envs = sweep(model, sample, key, envs)
        env_grads, local_energy, envs = grads_and_energy(model, sample, amp, operator, top_envs)
        grad_row, p_row = flatten_grads(env_grads, sample, amp)
        return sample, key, envs, grad_row, p_row, amp, local_energy

    def burn_step(carry, _):
        samples, chain_keys, envs = carry
        samples, chain_keys, _, _ = jax.vmap(
            lambda sample, key, env: sweep(model, sample, key, env)
        )(samples, chain_keys, envs)
        envs = jax.vmap(lambda sample: bottom_envs(model, sample))(samples)
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
        "Sequential sampling",
    )

    # TODO: properly distribute n_samples, avoid sampling overhead.
    samples_out = _trim_samples(samples_out, total_samples, num_samples)
    grads = _trim_samples(grads, total_samples, num_samples)
    amps = _trim_samples(amps, total_samples, num_samples)
    local_energies = _trim_samples(local_energies, total_samples, num_samples)
    if full_gradient:
        p = None
    else:
        p = _trim_samples(p, total_samples, num_samples)

    return samples_out, grads, p, key, final_samples, amps, local_energies
