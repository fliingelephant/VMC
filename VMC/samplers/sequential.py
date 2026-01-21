"""Sequential Metropolis samplers for MPS/PEPS."""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import functools
import math

import jax
import jax.numpy as jnp
from plum import dispatch
from tqdm import tqdm

from VMC.models.mps import MPS
from VMC.models.peps import (
    ContractionStrategy,
    PEPS,
    _apply_mpo_from_below,
    _build_row_mpo,
    _compute_all_gradients,
    _compute_single_gradient,
    _contract_bottom,
    _contract_column_transfer,
    _contract_left_partial,
    _contract_right_partial,
)
from VMC.utils.smallo import mps_site_dims, peps_site_dims
from VMC.utils.utils import occupancy_to_spin, spin_to_occupancy

__all__ = [
    "sequential_sample",
    "sequential_sample_with_gradients",
    "peps_sequential_sweep",
]


def _run_sweeps(sweep_fn, state: jax.Array, key: jax.Array, count: int):
    def step(carry, _):
        state, key = carry
        state, key = sweep_fn(state, key)
        return (state, key), None

    (state, key), _ = jax.lax.scan(step, (state, key), xs=None, length=count)
    return state, key


def _collect_steps(step_fn, carry, count: int, progress_interval: int | None, desc: str):
    if not progress_interval or progress_interval <= 0:
        return jax.lax.scan(step_fn, carry, xs=None, length=count)

    outputs_list = []
    for _ in tqdm(range(count), total=count, miniters=progress_interval, desc=desc):
        carry, output = step_fn(carry, None)
        outputs_list.append(output)
    outputs = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *outputs_list)
    return carry, outputs


def _sample_counts(n_samples: int, n_chains: int, burn_in: int) -> tuple[int, int, int, int, int]:
    num_samples = int(n_samples)
    num_chains = int(n_chains)
    num_burn_in = int(burn_in)
    chain_length = int(math.ceil(num_samples / num_chains))
    total_samples = chain_length * num_chains
    return num_samples, num_chains, num_burn_in, chain_length, total_samples


def _random_occupancy(key: jax.Array, n_chains: int, shape: tuple[int, ...]) -> jax.Array:
    init_keys = jax.random.split(key, n_chains)
    return jax.vmap(
        lambda k: jax.random.bernoulli(k, 0.5, shape=shape).astype(jnp.int32)
    )(init_keys)


def _trim_samples(samples: jax.Array, total_samples: int, num_samples: int) -> jax.Array:
    reshaped = samples.reshape((total_samples,) + samples.shape[2:])
    return reshaped[:num_samples]


def _collect_samples(
    sweep_batched,
    state: jax.Array,
    chain_keys: jax.Array,
    *,
    num_burn_in: int,
    chain_length: int,
    progress_interval: int | None,
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
        progress_interval,
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


def _pad_mps_tensors(
    tensors: list[jax.Array],
    bond_dim: int,
) -> jax.Array:
    """Pad MPS tensors to a uniform bond dimension for JAX scans."""
    padded = []
    for tensor in tensors:
        block = jnp.zeros((2, bond_dim, bond_dim), dtype=tensor.dtype)
        block = block.at[:, :tensor.shape[1], :tensor.shape[2]].set(tensor)
        padded.append(block)
    return jnp.stack(padded, axis=0)


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
    burn_in: int = 0,
    progress_interval: int | None = None,
    return_key: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Sequential sampling for MPS using Metropolis sweeps.

    Records one sample per sweep. ``n_samples`` counts recorded sweeps across
    chains; burn-in sweeps are not recorded. ``progress_interval`` counts samples.
    """
    num_samples, num_chains, num_burn_in, chain_length, total_samples = _sample_counts(
        n_samples, n_chains, burn_in
    )
    tensors = [jnp.asarray(t) for t in model.tensors]
    n_sites = len(tensors)
    tensors_padded = _pad_mps_tensors(tensors, model.bond_dim)

    key, init_key = jax.random.split(key)
    key, chain_key = jax.random.split(key)
    indices = _random_occupancy(init_key, num_chains, (n_sites,))
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
        progress_interval=progress_interval,
        sample_view=lambda x: x,
    )
    # TODO: properly distribute n_samples, avoid sampling overhead.
    samples = _trim_samples(samples, total_samples, num_samples)
    spins_batch = occupancy_to_spin(samples)
    if return_key:
        return spins_batch, key
    return spins_batch


@dispatch
def sequential_sample(
    model: PEPS,
    *,
    n_samples: int = 1,
    n_chains: int = 1,
    key: jax.Array,
    burn_in: int = 0,
    progress_interval: int | None = None,
    return_key: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Sequential sampling for PEPS using Metropolis sweeps.

    Records one sample per sweep. ``n_samples`` counts recorded sweeps across
    chains; burn-in sweeps are not recorded. ``progress_interval`` counts samples.
    """
    num_samples, num_chains, num_burn_in, chain_length, total_samples = _sample_counts(
        n_samples, n_chains, burn_in
    )
    shape = model.shape
    n_rows, n_cols = shape
    n_sites = int(n_rows * n_cols)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]

    key, init_key = jax.random.split(key)
    key, chain_key = jax.random.split(key)
    spins = _random_occupancy(init_key, num_chains, shape)
    chain_keys = jax.random.split(chain_key, num_chains)

    def sweep_once(spins, key):
        return peps_sequential_sweep(
            tensors,
            spins,
            shape,
            model.strategy,
            key,
        )

    sweep_batched = jax.vmap(sweep_once, in_axes=(0, 0))
    _, samples = _collect_samples(
        sweep_batched,
        spins,
        chain_keys,
        num_burn_in=num_burn_in,
        chain_length=chain_length,
        progress_interval=progress_interval,
        sample_view=lambda x: x.reshape(num_chains, n_sites),
    )
    # TODO: properly distribute n_samples, avoid sampling overhead.
    samples = _trim_samples(samples, total_samples, num_samples)
    samples = occupancy_to_spin(samples)
    if return_key:
        return samples, key
    return samples


@dispatch
def sequential_sample_with_gradients(
    model: MPS,
    *,
    n_samples: int = 1,
    n_chains: int = 1,
    key: jax.Array,
    initial_configuration: jax.Array | None = None,
    burn_in: int = 0,
    progress_interval: int | None = None,
    full_gradient: bool = False,
    return_prob: bool = False,
) -> (
    tuple[jax.Array, jax.Array, jax.Array | None, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array | None, jax.Array, jax.Array, dict[str, jax.Array]]
):
    """Sequential sampling for MPS with per-sample gradient recording.

    Total samples are ``n_samples`` across chains (burn-in sweeps are not recorded).

    Args:
        initial_configuration: Optional initial chain configs, shape (n_chains, n_sites),
            in spin format (-1/+1). If None, random initialization is used.

    Returns:
        samples, grads, p, key, final_configurations[, info]
    """
    num_samples, num_chains, num_burn_in, chain_length, total_samples = _sample_counts(
        n_samples, n_chains, burn_in
    )
    tensors = [jnp.asarray(t) for t in model.tensors]
    n_sites = len(tensors)
    bond_dim = model.bond_dim
    phys_dim = model.phys_dim
    tensors_padded = _pad_mps_tensors(tensors, bond_dim)

    key, chain_key = jax.random.split(key)
    if initial_configuration is not None:
        indices = spin_to_occupancy(initial_configuration)
    else:
        key, init_key = jax.random.split(key)
        indices = _random_occupancy(init_key, num_chains, (n_sites,))
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

    def flatten_full_gradients(indices, left_envs, right_envs):
        grad_parts = []
        for site in range(n_sites):
            left_dim, right_dim = mps_site_dims(site, n_sites, bond_dim)
            left = left_envs[site][:left_dim]
            right = right_envs[site + 1][:right_dim]
            grad_site = left[:, None] * right[None, :]
            grad_full = jnp.zeros(
                (phys_dim, left_dim, right_dim), dtype=tensors[0].dtype
            )
            grad_full = grad_full.at[indices[site]].set(grad_site)
            grad_parts.append(grad_full.ravel())
        return jnp.concatenate(grad_parts)

    def flatten_sliced_gradients(indices, left_envs, right_envs):
        grad_parts = []
        p_parts = []
        for site in range(n_sites):
            left_dim, right_dim = mps_site_dims(site, n_sites, bond_dim)
            left = left_envs[site][:left_dim]
            right = right_envs[site + 1][:right_dim]
            grad_site = left[:, None] * right[None, :]
            grad_parts.append(grad_site.reshape(-1))
            params_per_phys = left_dim * right_dim
            p_parts.append(jnp.full((params_per_phys,), indices[site], dtype=jnp.int8))
        return jnp.concatenate(grad_parts), jnp.concatenate(p_parts)

    if full_gradient:
        def flatten_fn(indices, left_envs, right_envs):
            return flatten_full_gradients(indices, left_envs, right_envs), jnp.zeros((0,), dtype=jnp.int8)
    else:
        flatten_fn = flatten_sliced_gradients

    def sample_step(carry, _):
        indices, chain_keys, right_envs = carry
        indices, chain_keys, left_envs, left_env = sweep_with_envs(
            indices, chain_keys, right_envs, True
        )
        right_envs_next = jax.vmap(
            lambda idx: _mps_right_envs(tensors_padded, idx, n_sites=n_sites)
        )(indices)
        amp = left_env[:, 0]
        grad_row, p_row = jax.vmap(flatten_fn)(
            indices, left_envs, right_envs_next
        )
        grad_row = grad_row / amp[:, None]
        return (indices, chain_keys, right_envs_next), (
            indices,
            grad_row,
            p_row,
            jnp.abs(amp) ** 2,
        )

    (final_configurations, _, _), (samples, grads, p, probs) = _collect_steps(
        sample_step,
        (indices, chain_keys, right_envs),
        chain_length,
        progress_interval,
        "Sequential sampling",
    )

    # TODO: properly distribute n_samples, avoid sampling overhead.
    samples = _trim_samples(samples, total_samples, num_samples)
    samples = occupancy_to_spin(samples)
    grads = _trim_samples(grads, total_samples, num_samples)
    probs = _trim_samples(probs, total_samples, num_samples)
    if full_gradient:
        p = None
    else:
        p = _trim_samples(p, total_samples, num_samples)
    final_configurations = occupancy_to_spin(final_configurations)

    if return_prob:
        return samples, grads, p, key, final_configurations, {"prob": probs}
    return samples, grads, p, key, final_configurations


@dispatch
def sequential_sample_with_gradients(
    model: PEPS,
    *,
    n_samples: int = 1,
    n_chains: int = 1,
    key: jax.Array,
    initial_configuration: jax.Array | None = None,
    burn_in: int = 0,
    progress_interval: int | None = None,
    full_gradient: bool = False,
    return_prob: bool = False,
) -> (
    tuple[jax.Array, jax.Array, jax.Array | None, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array | None, jax.Array, jax.Array, dict[str, jax.Array]]
):
    """Sequential sampling for PEPS with per-sample gradient recording.

    Total samples are ``n_samples`` across chains (burn-in sweeps are not recorded).

    Args:
        initial_configuration: Optional initial chain configs, shape (n_chains, n_rows, n_cols),
            in spin format (-1/+1). If None, random initialization is used.

    Returns:
        samples, grads, p, key, final_configurations[, info]
    """
    num_samples, num_chains, num_burn_in, chain_length, total_samples = _sample_counts(
        n_samples, n_chains, burn_in
    )
    shape = model.shape
    n_rows, n_cols = shape
    n_sites = int(n_rows * n_cols)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    bond_dim = model.bond_dim

    key, chain_key = jax.random.split(key)
    if initial_configuration is not None:
        spins = spin_to_occupancy(initial_configuration)
    else:
        key, init_key = jax.random.split(key)
        spins = _random_occupancy(init_key, num_chains, shape)
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
        grad_parts = []
        p_parts = []
        for row in range(n_rows):
            for col in range(n_cols):
                grad_parts.append(env_grads[row][col].reshape(-1))
                up, down, left, right = peps_site_dims(row, col, n_rows, n_cols, bond_dim)
                params_per_phys = up * down * left * right
                p_parts.append(
                    jnp.full((params_per_phys,), spins[row, col], dtype=jnp.int8)
                )
        return jnp.concatenate(grad_parts), jnp.concatenate(p_parts)

    if full_gradient:
        def flatten_fn(env_grads, spins):
            return flatten_full_gradients(env_grads, spins), jnp.zeros((0,), dtype=jnp.int8)
    else:
        flatten_fn = flatten_sliced_gradients

    def sample_step(carry, _):
        spins, chain_keys, bottom_envs, steps_left = carry
        spins, chain_keys, top_envs, top_env = sweep_with_envs(
            spins, chain_keys, bottom_envs, True
        )
        amp = jax.vmap(_contract_bottom)(top_env)
        # Skip bottom-env caching on the last sweep.
        cache_envs = steps_left > 1

        def cached_branch(_):
            return jax.vmap(
                lambda s, t: _compute_all_gradients(
                    tensors,
                    s,
                    shape,
                    model.strategy,
                    t,
                    cache_bottom_envs=True,
                )
            )(spins, top_envs)

        def final_branch(_):
            env_grads = jax.vmap(
                lambda s, t: _compute_all_gradients(
                    tensors,
                    s,
                    shape,
                    model.strategy,
                    t,
                    cache_bottom_envs=False,
                )
            )(spins, top_envs)
            return env_grads, bottom_envs

        env_grads, bottom_envs_next = jax.lax.cond(
            cache_envs, cached_branch, final_branch, operand=None
        )
        grad_row, p_row = jax.vmap(flatten_fn)(env_grads, spins)
        grad_row = grad_row / amp[:, None]
        return (spins, chain_keys, bottom_envs_next, steps_left - 1), (
            spins.reshape(num_chains, n_sites),
            grad_row,
            p_row,
            jnp.abs(amp) ** 2,
        )

    steps_left = jnp.asarray(chain_length)
    (final_spins, _, _, _), (samples, grads, p, probs) = _collect_steps(
        sample_step,
        (spins, chain_keys, bottom_envs, steps_left),
        chain_length,
        progress_interval,
        "Sequential sampling",
    )

    # TODO: properly distribute n_samples, avoid sampling overhead.
    samples = _trim_samples(samples, total_samples, num_samples)
    samples = occupancy_to_spin(samples)
    grads = _trim_samples(grads, total_samples, num_samples)
    probs = _trim_samples(probs, total_samples, num_samples)
    if full_gradient:
        p = None
    else:
        p = _trim_samples(p, total_samples, num_samples)
    final_configurations = occupancy_to_spin(final_spins)

    if return_prob:
        return samples, grads, p, key, final_configurations, {"prob": probs}
    return samples, grads, p, key, final_configurations


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


@functools.partial(jax.jit, static_argnames=("shape", "strategy", "collect_top_envs"))
def _peps_sequential_sweep_with_envs(
    tensors: list[list[jax.Array]],
    spins: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
    key: jax.Array,
    bottom_envs: list[tuple],
    collect_top_envs: bool,
) -> tuple[jax.Array, jax.Array, list[tuple], tuple]:
    """Run one sequential Metropolis sweep over PEPS sites."""
    n_rows, n_cols = shape
    dtype = tensors[0][0].dtype

    top_envs = [] if collect_top_envs else None
    top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    for row in range(n_rows):
        if collect_top_envs:
            top_envs.append(top_env)
        bottom_env = bottom_envs[row]
        # TODO: Reuse row MPOs between the sweep and gradient contraction to avoid
        # rebuilding per-row MPOs after flips; would require caching updated rows.
        mpo_row = _build_row_mpo(tensors, spins[row], row, n_cols)

        transfers = []
        for col in range(n_cols):
            transfer = _contract_column_transfer(
                top_env[col], mpo_row[col], bottom_env[col]
            )
            transfers.append(transfer)

        right_envs = [None] * n_cols
        right_envs[n_cols - 1] = env = jnp.ones((1, 1, 1), dtype=dtype)
        for col in range(n_cols - 2, -1, -1):
            env = _contract_right_partial(transfers[col + 1], env)
            right_envs[col] = env

        left_env = jnp.ones((1, 1, 1), dtype=dtype)
        updated_row = []  # Track updated MPOs to avoid rebuilding the row.
        for col in range(n_cols):
            env_grad = _compute_single_gradient(
                left_env,
                right_envs[col],
                top_env[col],
                bottom_env[col],
                mpo_row[col].shape,
            )
            site_tensor = tensors[row][col]

            def amp_for_phys(site_tensor_phys: jax.Array) -> jax.Array:
                mpo = jnp.transpose(site_tensor_phys, (2, 3, 0, 1))
                return jnp.einsum("udlr,lrud->", env_grad, mpo)

            amps = jax.vmap(amp_for_phys, in_axes=0)(site_tensor)
            weights = jnp.abs(amps) ** 2
            cur_idx = spins[row, col]
            flip_idx = 1 - cur_idx
            weight_cur = weights[cur_idx]
            weight_flip = weights[flip_idx]
            ratio = _metropolis_ratio(weight_cur, weight_flip)

            key, subkey = jax.random.split(key)
            accept = jax.random.uniform(subkey) < jnp.minimum(1.0, ratio)
            new_idx = jnp.where(accept, flip_idx, cur_idx)
            spins = spins.at[row, col].set(new_idx)

            mpo_sel = jnp.transpose(site_tensor[new_idx], (2, 3, 0, 1))
            updated_row.append(mpo_sel)
            transfer = _contract_column_transfer(
                top_env[col], mpo_sel, bottom_env[col]
            )
            left_env = _contract_left_partial(left_env, transfer)

        # Update top boundary with the updated row (reuse environments in sweep).
        top_env = strategy.apply(top_env, tuple(updated_row))

    return spins, key, top_envs if collect_top_envs else (), top_env


def peps_sequential_sweep(
    tensors: list[list[jax.Array]],
    spins: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
    key: jax.Array,
):
    bottom_envs = _peps_bottom_envs(tensors, spins, shape, strategy)
    spins, key, _, _ = _peps_sequential_sweep_with_envs(
        tensors, spins, shape, strategy, key, bottom_envs, False
    )
    return spins, key
