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
    ContractionStrategy,
    PEPS,
    _apply_mpo_from_below,
    _build_row_mpo,
    _compute_all_env_grads_and_energy,
    _compute_right_envs,
    _contract_bottom,
    _contract_column_transfer,
    _contract_left_partial,
    _contract_right_partial,
)
from vmc.operators import LocalHamiltonian, bucket_terms
from vmc.utils.smallo import params_per_site as params_per_site_fn
from vmc.utils.utils import occupancy_to_spin, spin_to_occupancy
from vmc.utils.vmc_utils import local_estimate

__all__ = [
    "sequential_sample",
    "sequential_sample_with_gradients",
    "peps_sequential_sweep",
]
logger = logging.getLogger(__name__)

def _run_sweeps(sweep_fn, state: jax.Array, key: jax.Array, count: int):
    def step(carry, _):
        state, key = carry
        state, key = sweep_fn(state, key)
        return (state, key), None

    (state, key), _ = jax.lax.scan(step, (state, key), xs=None, length=count)
    return state, key


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


def _random_occupancy(key: jax.Array, n_chains: int, shape: tuple[int, ...]) -> jax.Array:
    return jax.vmap(lambda k: jax.random.bernoulli(k, 0.5, shape=shape).astype(jnp.int32))(
        jax.random.split(key, n_chains)
    )


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
    state, chain_keys = _run_sweeps(
        sweep_batched, state, chain_keys, num_burn_in
    )

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


def _metropolis_ratio(weight_cur: jax.Array, weight_flip: jax.Array) -> jax.Array:
    """Compute Metropolis acceptance ratio with proper handling of zero weights."""
    return jnp.where(
        weight_cur > 0.0,
        weight_flip / weight_cur,
        jnp.where(weight_flip > 0.0, jnp.inf, 0.0),
    )


def _pad_mps_tensors(tensors: list[jax.Array], bond_dim: int) -> jax.Array:
    """Pad MPS tensors to a uniform bond dimension for JAX scans."""
    def pad_tensor(tensor):
        block = jnp.zeros((2, bond_dim, bond_dim), dtype=tensor.dtype)
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
    left_env0 = right_envs[-1]
    site_ids = jnp.arange(n_sites)

    def sweep_step(carry, site):
        indices, left_env, key = carry
        left_env_before = left_env
        right_env = right_envs[site + 1]
        cur_idx = indices[site]
        flip_idx = 1 - cur_idx
        tensor_cur = tensors[site, cur_idx]
        tensor_flip = tensors[site, flip_idx]
        amp_cur = jnp.einsum("i,ij,j->", left_env, tensor_cur, right_env)
        amp_flip = jnp.einsum("i,ij,j->", left_env, tensor_flip, right_env)
        weight_cur = jnp.abs(amp_cur) ** 2
        weight_flip = jnp.abs(amp_flip) ** 2
        ratio = _metropolis_ratio(weight_cur, weight_flip)

        key, subkey = jax.random.split(key)
        accept = jax.random.uniform(subkey) < jnp.minimum(1.0, ratio)
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
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]

    key, chain_key = jax.random.split(key)
    spins = initial_configuration
    chain_keys = jax.random.split(chain_key, num_chains)
    bottom_envs = jax.vmap(
        lambda s: _peps_bottom_envs(tensors, s, shape, model.strategy)
    )(spins)

    def sweep_with_envs(spins, chain_keys, bottom_envs, collect_top_envs):
        def sweep_single(s, key, envs):
            return _peps_sequential_sweep_with_envs(
                tensors,
                s,
                shape,
                model.strategy,
                key,
                envs,
                collect_top_envs,
                collect_top_envs,
            )

        return jax.vmap(sweep_single, in_axes=(0, 0, 0))(
            spins, chain_keys, bottom_envs
        )

    spins, chain_keys, bottom_envs = _run_burn_in_with_envs(
        sweep_with_envs,
        lambda s: _peps_bottom_envs(tensors, s, shape, model.strategy),
        spins,
        chain_keys,
        bottom_envs,
        num_burn_in=num_burn_in,
    )

    def sample_step(carry, _):
        spins, chain_keys, bottom_envs = carry
        spins, chain_keys, _, _ = sweep_with_envs(
            spins, chain_keys, bottom_envs, False
        )
        bottom_envs = jax.vmap(
            lambda s: _peps_bottom_envs(tensors, s, shape, model.strategy)
        )(spins)
        return (spins, chain_keys, bottom_envs), spins.reshape(num_chains, n_sites)

    (_, _, _), samples = _collect_steps(
        sample_step,
        (spins, chain_keys, bottom_envs),
        chain_length,
        "Sequential sampling",
    )
    # TODO: properly distribute n_samples, avoid sampling overhead.
    samples = _trim_samples(samples, total_samples, num_samples)
    return samples


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

    local_energies = local_estimate(
        model, occupancy_to_spin(samples), operator, amps
    )
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
    bond_dim = model.bond_dim
    params_per_site = tuple(int(p) for p in params_per_site_fn(model))

    key, chain_key = jax.random.split(key)
    spins = initial_configuration
    chain_keys = jax.random.split(chain_key, num_chains)
    bottom_envs = jax.vmap(
        lambda s: _peps_bottom_envs(tensors, s, shape, model.strategy)
    )(spins)

    def sweep_with_envs(spins, chain_keys, bottom_envs, collect_top_envs):
        def sweep_single(s, key, envs):
            return _peps_sequential_sweep_with_envs(
                tensors,
                s,
                shape,
                model.strategy,
                key,
                envs,
                collect_top_envs,
                collect_top_envs,
            )

        return jax.vmap(sweep_single, in_axes=(0, 0, 0))(
            spins, chain_keys, bottom_envs
        )

    spins, chain_keys, bottom_envs = _run_burn_in_with_envs(
        sweep_with_envs,
        lambda s: _peps_bottom_envs(tensors, s, shape, model.strategy),
        spins,
        chain_keys,
        bottom_envs,
        num_burn_in=num_burn_in,
    )

    def flatten_full_gradients(env_grads, spins):
        grad_parts = []
        for row in range(n_rows):
            for col in range(n_cols):
                grad_full = jnp.zeros_like(tensors[row][col])
                phys_idx = spins[row, col]
                grad_full = grad_full.at[phys_idx].set(env_grads[row][col])
                grad_parts.append(grad_full.ravel())
        return jnp.concatenate(grad_parts)

    def flatten_sliced_gradients(env_grads, spins):
        grad_parts = [
            env_grads[row][col].reshape(-1)
            for row in range(n_rows)
            for col in range(n_cols)
        ]
        p_parts = [
            jnp.full((params_per_site[site],), spins.reshape(-1)[site], dtype=jnp.int8)
            for site in range(n_sites)
        ]
        return jnp.concatenate(grad_parts), jnp.concatenate(p_parts)

    diagonal_terms, one_site_terms, horizontal_terms, vertical_terms = bucket_terms(
        operator.terms, shape
    )

    def sample_step(carry, _):
        spins, chain_keys, bottom_envs = carry
        spins, chain_keys, aux, amp = sweep_with_envs(
            spins, chain_keys, bottom_envs, True
        )
        top_envs, row_mpos = aux
        env_grads, local_energy, bottom_envs_next = jax.vmap(
            lambda s, a, t, m: _compute_all_env_grads_and_energy(
                tensors,
                s,
                a,
                shape,
                model.strategy,
                t,
                row_mpos=m,
                diagonal_terms=diagonal_terms,
                one_site_terms=one_site_terms,
                horizontal_terms=horizontal_terms,
                vertical_terms=vertical_terms,
            )
        )(spins, amp, top_envs, row_mpos)
        if full_gradient:
            grad_row = jax.vmap(flatten_full_gradients)(env_grads, spins)
            p_row = jnp.zeros((amp.shape[0], 0), dtype=jnp.int8)
        else:
            grad_row, p_row = jax.vmap(flatten_sliced_gradients)(env_grads, spins)
        grad_row = grad_row / amp[:, None]
        return (spins, chain_keys, bottom_envs_next), (
            spins.reshape(num_chains, n_sites),
            grad_row,
            p_row,
            amp,
            local_energy,
        )

    (final_spins, _, _), (samples, grads, p, amps, local_energies) = _collect_steps(
        sample_step,
        (spins, chain_keys, bottom_envs),
        chain_length,
        "Sequential sampling",
    )

    # TODO: properly distribute n_samples, avoid sampling overhead.
    samples = _trim_samples(samples, total_samples, num_samples)
    grads = _trim_samples(grads, total_samples, num_samples)
    amps = _trim_samples(amps, total_samples, num_samples)
    local_energies = _trim_samples(local_energies, total_samples, num_samples)
    if full_gradient:
        p = None
    else:
        p = _trim_samples(p, total_samples, num_samples)

    return samples, grads, p, key, final_spins, amps, local_energies


@functools.partial(jax.jit, static_argnames=("shape", "strategy"))
def _peps_bottom_envs(
    tensors: list[list[jax.Array]],
    spins: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
) -> list[tuple]:
    n_rows, n_cols = shape
    dtype = tensors[0][0].dtype
    bottom_envs = [None] * n_rows
    bottom_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    for row in range(n_rows - 1, -1, -1):
        bottom_envs[row] = bottom_env
        # TODO: Reuse the most-bottom MPO/boundary across sweeps to avoid rebuilding
        # it when only upper rows change; needs cache plumbing in the sampler.
        mpo_row = _build_row_mpo(tensors, spins[row], row, n_cols)
        bottom_env = _apply_mpo_from_below(bottom_env, mpo_row, strategy)
    return bottom_envs


@functools.partial(
    jax.jit, static_argnames=("shape", "strategy", "collect_top_envs", "return_amp")
)
def _peps_sequential_sweep_with_envs(
    tensors: list[list[jax.Array]],
    spins: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
    key: jax.Array,
    bottom_envs: list[tuple],
    collect_top_envs: bool,
    return_amp: bool,
) -> tuple[jax.Array, jax.Array, tuple, jax.Array]:
    """Run one sequential Metropolis sweep over PEPS sites.

    Returns updated spins, key, (top_envs, row_mpos) when requested, and amplitude
    if ``return_amp`` is True.
    """
    n_rows, n_cols = shape
    dtype = tensors[0][0].dtype

    top_envs = [] if collect_top_envs else ()
    row_mpos = [] if collect_top_envs else ()
    top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    for row in range(n_rows):
        if collect_top_envs:
            top_envs.append(top_env)
        bottom_env = bottom_envs[row]
        # TODO: Reuse row MPOs between the sweep and gradient contraction to avoid
        # rebuilding per-row MPOs after flips; would require caching updated rows.
        mpo_row = _build_row_mpo(tensors, spins[row], row, n_cols)

        transfers = [
            _contract_column_transfer(top_env[col], mpo_row[col], bottom_env[col])
            for col in range(n_cols)
        ]
        right_envs = _compute_right_envs(transfers, dtype)
        left_env = jnp.ones((1, 1, 1), dtype=dtype)
        updated_row = []  # Track updated MPOs to avoid rebuilding the row.
        for col in range(n_cols):
            site_tensor = tensors[row][col]
            cur_idx = spins[row, col]
            flip_idx = 1 - cur_idx
            mpo_flip = jnp.transpose(site_tensor[flip_idx], (2, 3, 0, 1))
            transfer_cur = transfers[col]
            transfer_flip = _contract_column_transfer(
                top_env[col], mpo_flip, bottom_env[col]
            )
            amp_cur = jnp.einsum(
                "ace,abcdef,bdf->", left_env, transfer_cur, right_envs[col]
            )
            amp_flip = jnp.einsum(
                "ace,abcdef,bdf->", left_env, transfer_flip, right_envs[col]
            )
            weight_cur = jnp.abs(amp_cur) ** 2
            weight_flip = jnp.abs(amp_flip) ** 2
            ratio = _metropolis_ratio(weight_cur, weight_flip)

            key, subkey = jax.random.split(key)
            accept = jax.random.uniform(subkey) < jnp.minimum(1.0, ratio)
            new_idx = jnp.where(accept, flip_idx, cur_idx)
            spins = spins.at[row, col].set(new_idx)

            def accept_branch(_):
                return mpo_flip, transfer_flip

            def reject_branch(_):
                return mpo_row[col], transfer_cur

            mpo_sel, transfer = jax.lax.cond(
                accept, accept_branch, reject_branch, operand=None
            )
            updated_row.append(mpo_sel)
            left_env = _contract_left_partial(left_env, transfer)

        if collect_top_envs:
            row_mpos.append(tuple(updated_row))
        # Update top boundary with the updated row (reuse environments in sweep).
        top_env = strategy.apply(top_env, tuple(updated_row))

    amp = _contract_bottom(top_env) if return_amp else jnp.zeros((), dtype=dtype)
    aux = (top_envs, row_mpos) if collect_top_envs else ()
    return spins, key, aux, amp


def peps_sequential_sweep(
    tensors: list[list[jax.Array]],
    spins: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
    key: jax.Array,
):
    bottom_envs = _peps_bottom_envs(tensors, spins, shape, strategy)
    spins, key, _, _ = _peps_sequential_sweep_with_envs(
        tensors, spins, shape, strategy, key, bottom_envs, False, False
    )
    return spins, key
