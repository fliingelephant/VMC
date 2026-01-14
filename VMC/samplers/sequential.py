"""Sequential Metropolis samplers for MPS/PEPS."""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import functools
import logging

import jax
import jax.numpy as jnp
from plum import dispatch

from VMC.models.mps import SimpleMPS
from VMC.models.peps import (
    ContractionStrategy,
    SimplePEPS,
    _apply_mpo_from_below,
    _build_row_mpo_static,
    _compute_single_gradient,
    _contract_column_transfer,
    _contract_left_partial,
    _contract_right_partial,
)
from VMC.core.eval import _value_and_grad

__all__ = [
    "sequential_sample",
    "sequential_sample_with_gradients",
    "peps_sequential_sweep",
]

logger = logging.getLogger(__name__)


def _validate_sampling_params(
    n_samples: int,
    n_sweeps: int,
    burn_in: int,
) -> tuple[int, int, int]:
    """Validate and convert sampling parameters."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if n_sweeps <= 0:
        raise ValueError("n_sweeps must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    return int(n_samples), int(n_sweeps), int(burn_in)


def _maybe_log_progress(
    current: int,
    total: int,
    interval: int | None,
) -> None:
    """Log progress every interval samples when enabled."""
    if interval is None or interval <= 0:
        return
    if current % interval != 0 and current != total:
        return
    if logger.isEnabledFor(logging.INFO):
        logger.info("Sequential PEPS sampling: %d/%d samples", current, total)


def _pad_mps_tensors(
    tensors: list[jax.Array],
    bond_dim: int,
) -> jax.Array:
    """Pad MPS tensors to a uniform bond dimension for JAX scans."""
    padded = []
    for tensor in tensors:
        left_dim = tensor.shape[1]
        right_dim = tensor.shape[2]
        block = jnp.zeros((2, bond_dim, bond_dim), dtype=tensor.dtype)
        block = block.at[:, :left_dim, :right_dim].set(tensor)
        padded.append(block)
    return jnp.stack(padded, axis=0)


@functools.partial(jax.jit, static_argnames=("n_sites",))
def _sequential_mps_sweep(
    tensors: jax.Array,
    indices: jax.Array,
    *,
    key: jax.Array,
    n_sites: int,
) -> tuple[jax.Array, jax.Array]:
    """Run a sequential Metropolis sweep with fixed site order."""
    bond_dim = tensors.shape[-1]
    right_end = jnp.zeros((bond_dim,), dtype=tensors.dtype)
    right_end = right_end.at[0].set(1.0)

    site_ids_rev = jnp.arange(n_sites - 1, -1, -1)

    def right_step(carry, site):
        tensor = tensors[site, indices[site]]
        right_env = jnp.einsum("ij,j->i", tensor, carry)
        return right_env, right_env

    _, right_envs_rev = jax.lax.scan(right_step, right_end, site_ids_rev)
    right_envs = jnp.flip(right_envs_rev, axis=0)
    right_envs = jnp.concatenate([right_envs, right_end[None, :]], axis=0)

    left_env0 = right_end
    site_ids = jnp.arange(n_sites)

    def sweep_step(carry, site):
        indices, left_env, key = carry
        right_env = right_envs[site + 1]
        cur_idx = indices[site]
        flip_idx = 1 - cur_idx
        tensor_cur = tensors[site, cur_idx]
        tensor_flip = tensors[site, flip_idx]
        amp_cur = jnp.einsum("i,ij,j->", left_env, tensor_cur, right_env)
        amp_flip = jnp.einsum("i,ij,j->", left_env, tensor_flip, right_env)
        weight_cur = jnp.abs(amp_cur) ** 2
        weight_flip = jnp.abs(amp_flip) ** 2
        ratio = jnp.where(
            weight_cur > 0.0,
            weight_flip / weight_cur,
            jnp.where(weight_flip > 0.0, jnp.inf, 0.0),
        )

        key, subkey = jax.random.split(key)
        accept = jax.random.uniform(subkey) < jnp.minimum(1.0, ratio)
        new_idx = jnp.where(accept, flip_idx, cur_idx)
        indices = indices.at[site].set(new_idx)

        tensor_sel = jnp.where(accept, tensor_flip, tensor_cur)
        left_env = jnp.einsum("i,ij->j", left_env, tensor_sel)
        return (indices, left_env, key), None

    (indices, _, key), _ = jax.lax.scan(
        sweep_step, (indices, left_env0, key), site_ids
    )
    return indices, key


@dispatch
def sequential_sample(
    model: SimpleMPS,
    *,
    n_samples: int,
    key: jax.Array,
    n_sweeps: int = 1,
    burn_in: int = 0,
    return_key: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Sequential sampling for MPS using Metropolis sweeps."""
    num_samples, num_sweeps, num_burn_in = _validate_sampling_params(
        n_samples, n_sweeps, burn_in
    )
    tensors = [jnp.asarray(t) for t in model.tensors]
    n_sites = len(tensors)
    tensors_padded = _pad_mps_tensors(tensors, model.bond_dim)

    key, subkey = jax.random.split(key)
    indices = jax.random.bernoulli(subkey, 0.5, shape=(n_sites,)).astype(jnp.int32)

    def sweep_step(carry, _):
        indices, key = carry
        indices, key = _sequential_mps_sweep(
            tensors_padded,
            indices,
            key=key,
            n_sites=n_sites,
        )
        return (indices, key), None

    def run_sweeps(
        indices: jax.Array,
        key: jax.Array,
        count: int,
    ) -> tuple[jax.Array, jax.Array]:
        (indices, key), _ = jax.lax.scan(
            sweep_step,
            (indices, key),
            xs=None,
            length=count,
        )
        return indices, key

    indices, key = run_sweeps(indices, key, num_burn_in)

    def sample_step(carry, _):
        indices, key = carry
        indices, key = run_sweeps(indices, key, num_sweeps)
        return (indices, key), indices

    (_, key), spins_batch = jax.lax.scan(
        sample_step,
        (indices, key),
        xs=None,
        length=num_samples,
    )

    spins_batch = 2 * spins_batch - 1
    spins_batch = spins_batch.astype(jnp.int32)
    if return_key:
        return spins_batch, key
    return spins_batch


@dispatch
def sequential_sample(
    model: SimplePEPS,
    *,
    n_samples: int,
    key: jax.Array,
    n_sweeps: int = 1,
    burn_in: int = 0,
    progress_interval: int | None = None,
    return_key: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Sequential sampling for PEPS using Metropolis sweeps."""
    num_samples, num_sweeps, num_burn_in = _validate_sampling_params(
        n_samples, n_sweeps, burn_in
    )
    shape = model.shape
    n_sites = int(shape[0] * shape[1])
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]

    key, subkey = jax.random.split(key)
    spins = jax.random.bernoulli(subkey, 0.5, shape=shape).astype(jnp.int32)

    def sweep_step(carry, _):
        spins, key = carry
        spins, key = peps_sequential_sweep(
            tensors,
            spins,
            shape,
            model.strategy,
            key,
        )
        return (spins, key), None

    def run_sweeps(
        spins: jax.Array,
        key: jax.Array,
        count: int,
    ) -> tuple[jax.Array, jax.Array]:
        (spins, key), _ = jax.lax.scan(
            sweep_step,
            (spins, key),
            xs=None,
            length=count,
        )
        return spins, key

    spins, key = run_sweeps(spins, key, num_burn_in)

    if progress_interval is None or progress_interval <= 0:
        def sample_step(carry, _):
            spins, key = carry
            spins, key = run_sweeps(spins, key, num_sweeps)
            return (spins, key), spins.reshape(n_sites)

        (_, key), spins_batch = jax.lax.scan(
            sample_step,
            (spins, key),
            xs=None,
            length=num_samples,
        )
        samples = (2 * spins_batch - 1).astype(jnp.int32)
        if return_key:
            return samples, key
        return samples

    samples_list = []
    for idx in range(num_samples):
        spins, key = run_sweeps(spins, key, num_sweeps)
        samples_list.append(spins.reshape(n_sites))
        current = idx + 1
        if current % progress_interval == 0 or current == num_samples:
            jax.block_until_ready(spins)
            _maybe_log_progress(current, num_samples, progress_interval)

    samples = jnp.stack(samples_list, axis=0)
    samples = (2 * samples - 1).astype(jnp.int32)
    if return_key:
        return samples, key
    return samples


@dispatch
def sequential_sample_with_gradients(
    model: SimpleMPS,
    *,
    n_samples: int,
    key: jax.Array,
    n_sweeps: int = 1,
    burn_in: int = 0,
    full_gradient: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array]:
    """Sequential sampling for MPS with per-sample gradient recording."""
    num_samples, num_sweeps, num_burn_in = _validate_sampling_params(
        n_samples, n_sweeps, burn_in
    )
    tensors = [jnp.asarray(t) for t in model.tensors]
    n_sites = len(tensors)
    tensors_padded = _pad_mps_tensors(tensors, model.bond_dim)

    key, subkey = jax.random.split(key)
    indices = jax.random.bernoulli(subkey, 0.5, shape=(n_sites,)).astype(jnp.int32)

    def sweep_step(carry, _):
        indices, key = carry
        indices, key = _sequential_mps_sweep(
            tensors_padded,
            indices,
            key=key,
            n_sites=n_sites,
        )
        return (indices, key), None

    def run_sweeps(
        indices: jax.Array,
        key: jax.Array,
        count: int,
    ) -> tuple[jax.Array, jax.Array]:
        (indices, key), _ = jax.lax.scan(
            sweep_step,
            (indices, key),
            xs=None,
            length=count,
        )
        return indices, key

    indices, key = run_sweeps(indices, key, num_burn_in)
    samples = []
    grads = []
    p_rows = []
    for _ in range(num_samples):
        indices, key = run_sweeps(indices, key, num_sweeps)
        sample = (2 * indices - 1).astype(jnp.int32)
        samples.append(sample)
        amp, grad_row, p_row = _value_and_grad(
            model, sample, full_gradient=full_gradient
        )
        grads.append(grad_row / amp)
        if not full_gradient:
            p_rows.append(p_row)

    samples = jnp.stack(samples, axis=0)
    grads = jnp.stack(grads, axis=0)
    if full_gradient:
        return samples, grads, None, key
    p = jnp.stack(p_rows, axis=0)
    return samples, grads, p, key


@dispatch
def sequential_sample_with_gradients(
    model: SimplePEPS,
    *,
    n_samples: int,
    key: jax.Array,
    n_sweeps: int = 1,
    burn_in: int = 0,
    progress_interval: int | None = None,
    full_gradient: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array]:
    """Sequential sampling for PEPS with per-sample gradient recording."""
    num_samples, num_sweeps, num_burn_in = _validate_sampling_params(
        n_samples, n_sweeps, burn_in
    )
    shape = model.shape
    n_sites = int(shape[0] * shape[1])
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]

    key, subkey = jax.random.split(key)
    spins = jax.random.bernoulli(subkey, 0.5, shape=shape).astype(jnp.int32)

    def sweep_step(carry, _):
        spins, key = carry
        spins, key = peps_sequential_sweep(
            tensors,
            spins,
            shape,
            model.strategy,
            key,
        )
        return (spins, key), None

    def run_sweeps(
        spins: jax.Array,
        key: jax.Array,
        count: int,
    ) -> tuple[jax.Array, jax.Array]:
        (spins, key), _ = jax.lax.scan(
            sweep_step,
            (spins, key),
            xs=None,
            length=count,
        )
        return spins, key

    spins, key = run_sweeps(spins, key, num_burn_in)

    samples = []
    grads = []
    p_rows = []
    for idx in range(num_samples):
        spins, key = run_sweeps(spins, key, num_sweeps)
        sample = (2 * spins.reshape(n_sites) - 1).astype(jnp.int32)
        samples.append(sample)
        amp, grad_row, p_row = _value_and_grad(
            model, sample, full_gradient=full_gradient
        )
        grads.append(grad_row / amp)
        if not full_gradient:
            p_rows.append(p_row)

        if progress_interval is not None and progress_interval > 0:
            current = idx + 1
            if current % progress_interval == 0 or current == num_samples:
                jax.block_until_ready(spins)
                _maybe_log_progress(current, num_samples, progress_interval)

    samples = jnp.stack(samples, axis=0)
    grads = jnp.stack(grads, axis=0)
    if full_gradient:
        return samples, grads, None, key
    p = jnp.stack(p_rows, axis=0)
    return samples, grads, p, key


def _peps_boundary_mps(n_cols: int, dtype: jnp.dtype) -> tuple[jax.Array, ...]:
    return tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))


@functools.partial(jax.jit, static_argnames=("shape", "strategy"))
def peps_sequential_sweep(
    tensors: list[list[jax.Array]],
    spins: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Run one sequential Metropolis sweep over PEPS sites."""
    n_rows, n_cols = shape
    dtype = tensors[0][0].dtype

    bottom_envs = [None] * n_rows
    bottom_env = _peps_boundary_mps(n_cols, dtype)
    for row in range(n_rows - 1, -1, -1):
        bottom_envs[row] = bottom_env
        mpo_row = _build_row_mpo_static(tensors, spins[row], row, n_cols)
        bottom_env = _apply_mpo_from_below(bottom_env, mpo_row, strategy)

    top_env = _peps_boundary_mps(n_cols, dtype)
    for row in range(n_rows):
        bottom_env = bottom_envs[row]
        mpo_row = _build_row_mpo_static(tensors, spins[row], row, n_cols)

        transfers = []
        for col in range(n_cols):
            transfer = _contract_column_transfer(
                top_env[col], mpo_row[col], bottom_env[col]
            )
            transfers.append(transfer)

        right_envs = [None] * n_cols
        right_envs[n_cols - 1] = jnp.ones((1, 1, 1), dtype=dtype)
        env = right_envs[n_cols - 1]
        for col in range(n_cols - 2, -1, -1):
            env = _contract_right_partial(transfers[col + 1], env)
            right_envs[col] = env

        left_env = jnp.ones((1, 1, 1), dtype=dtype)
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
            weights = jnp.maximum(weights, 0.0)
            cur_idx = spins[row, col]
            flip_idx = 1 - cur_idx
            weight_cur = weights[cur_idx]
            weight_flip = weights[flip_idx]
            ratio = jnp.where(
                weight_cur > 0.0,
                weight_flip / weight_cur,
                jnp.where(weight_flip > 0.0, jnp.inf, 0.0),
            )

            key, subkey = jax.random.split(key)
            accept = jax.random.uniform(subkey) < jnp.minimum(1.0, ratio)
            new_idx = jnp.where(accept, flip_idx, cur_idx)
            spins = spins.at[row, col].set(new_idx)

            mpo_sel = jnp.transpose(site_tensor[new_idx], (2, 3, 0, 1))
            transfer = _contract_column_transfer(
                top_env[col], mpo_sel, bottom_env[col]
            )
            left_env = _contract_left_partial(left_env, transfer)

        # Update top boundary with the updated row (reuse environments in sweep).
        mpo_row = _build_row_mpo_static(tensors, spins[row], row, n_cols)
        top_env = strategy.apply(top_env, mpo_row)

    return spins, key
