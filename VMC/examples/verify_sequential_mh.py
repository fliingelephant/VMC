#!/usr/bin/env python3
"""Verify sequential Metropolis-Hastings sampling correctness.

Tests MPS and PEPS sequential samplers against:
- FullSum (exact) for 3x3 systems
- NetKet MetropolisLocal for 5x5 and 8x8 systems

Parameters per user specification:
- bond_dim = 3
- chi = 9 (bond_dim^2, ZipUp truncation)
- n_samples = 4096
- Progress logging every 1000 samples

Also reports runtime and acceptance ratios for efficiency comparison.
Set VMC_LOG_LEVEL=INFO to see progress output.
"""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import functools
import logging
import time

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from flax import nnx

from VMC.examples.real_time import build_heisenberg_square
from VMC.models.mps import MPS
from VMC.models.peps import ContractionStrategy, PEPS, ZipUp
from VMC.utils.vmc_utils import get_apply_fun

logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================
BOND_DIM = 3
CHI = BOND_DIM ** 2  # 9
N_SAMPLES = 4096
BURN_IN = 200
PROGRESS_INTERVAL = 1000
SEED = 42


# ==============================================================================
# Progress Logging
# ==============================================================================
def _log_progress(current: int, total: int) -> None:
    """Log sampling progress every PROGRESS_INTERVAL samples."""
    if current > 0 and current % PROGRESS_INTERVAL == 0:
        logger.info(f"  Progress: {current}/{total} samples...")


# ==============================================================================
# MPS Sweep with Acceptance Tracking
# ==============================================================================
@functools.partial(jax.jit, static_argnames=("n_sites",))
def _sequential_mps_sweep_with_accept(
    tensors: jax.Array,
    indices: jax.Array,
    *,
    key: jax.Array,
    n_sites: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run a sequential Metropolis sweep and count acceptances."""
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
        indices, left_env, key, n_accept = carry
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
        n_accept = n_accept + accept.astype(jnp.int32)

        tensor_sel = jnp.where(accept, tensor_flip, tensor_cur)
        left_env = jnp.einsum("i,ij->j", left_env, tensor_sel)
        return (indices, left_env, key, n_accept), None

    (indices, _, key, n_accept), _ = jax.lax.scan(
        sweep_step, (indices, left_env0, key, jnp.int32(0)), site_ids
    )
    return indices, key, n_accept


# ==============================================================================
# Sampling with Progress
# ==============================================================================
def _sequential_mps_with_progress(
    model: MPS,
    n_samples: int,
    burn_in: int,
    key: jax.Array,
) -> tuple[jax.Array, float]:
    """Sample from MPS with progress logging. Returns samples and acceptance ratio."""
    all_samples = []
    total_accept = 0
    total_proposals = 0

    # Initial configuration
    tensors = [jnp.asarray(t) for t in model.tensors]
    n_sites = len(tensors)
    from VMC.samplers.sequential import _pad_mps_tensors
    tensors_padded = _pad_mps_tensors(tensors, model.bond_dim)

    key, subkey = jax.random.split(key)
    indices = jax.random.bernoulli(subkey, 0.5, shape=(n_sites,)).astype(jnp.int32)

    # Burn-in (don't count acceptances)
    for _ in range(burn_in):
        indices, key, _ = _sequential_mps_sweep_with_accept(
            tensors_padded, indices, key=key, n_sites=n_sites
        )

    # Sampling with progress (one sweep per recorded sample)
    for sample_idx in range(n_samples):
        indices, key, n_accept = _sequential_mps_sweep_with_accept(
            tensors_padded, indices, key=key, n_sites=n_sites
        )
        total_accept += int(n_accept)
        total_proposals += n_sites
        spins = 2 * indices - 1
        all_samples.append(spins)
        _log_progress(sample_idx + 1, n_samples)

    acceptance_ratio = total_accept / total_proposals if total_proposals > 0 else 0.0
    return jnp.stack(all_samples, axis=0).astype(jnp.int32), acceptance_ratio


@functools.partial(jax.jit, static_argnames=("shape", "strategy"))
def _peps_sequential_sweep_with_accept(
    tensors: list[list[jax.Array]],
    spins: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, int]:
    """Run one sequential Metropolis sweep over PEPS sites with acceptance count."""
    from VMC.models.peps import (
        _apply_mpo_from_below,
        _build_row_mpo,
        _compute_single_gradient,
        _contract_column_transfer,
        _contract_left_partial,
        _contract_right_partial,
    )

    n_rows, n_cols = shape
    dtype = tensors[0][0].dtype
    n_sites = n_rows * n_cols

    def _peps_boundary_mps(n_cols: int, dtype: jnp.dtype) -> tuple[jax.Array, ...]:
        return tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))

    bottom_envs = [None] * n_rows
    bottom_env = _peps_boundary_mps(n_cols, dtype)
    for row in range(n_rows - 1, -1, -1):
        bottom_envs[row] = bottom_env
        mpo_row = _build_row_mpo(tensors, spins[row], row, n_cols)
        bottom_env = _apply_mpo_from_below(bottom_env, mpo_row, strategy)

    top_env = _peps_boundary_mps(n_cols, dtype)
    total_accept = 0
    for row in range(n_rows):
        bottom_env = bottom_envs[row]
        mpo_row = _build_row_mpo(tensors, spins[row], row, n_cols)

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
            total_accept = total_accept + accept.astype(jnp.int32)

            mpo_sel = jnp.transpose(site_tensor[new_idx], (2, 3, 0, 1))
            transfer = _contract_column_transfer(top_env[col], mpo_sel, bottom_env[col])
            left_env = _contract_left_partial(left_env, transfer)

        mpo_row = _build_row_mpo(tensors, spins[row], row, n_cols)
        top_env = strategy.apply(top_env, mpo_row)

    return spins, key, total_accept


def _sequential_peps_with_progress(
    model: PEPS,
    n_samples: int,
    burn_in: int,
    key: jax.Array,
) -> tuple[jax.Array, float]:
    """Sample from PEPS with progress logging. Returns samples and acceptance ratio."""
    shape = model.shape
    n_sites = shape[0] * shape[1]
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]

    key, subkey = jax.random.split(key)
    spins = jax.random.bernoulli(subkey, 0.5, shape=shape).astype(jnp.int32)

    total_accept = 0
    total_proposals = 0

    # Burn-in (don't count acceptances)
    for _ in range(burn_in):
        spins, key, _ = _peps_sequential_sweep_with_accept(
            tensors, spins, shape, model.strategy, key
        )

    # Sampling with progress (one sweep per recorded sample)
    all_samples = []
    for sample_idx in range(n_samples):
        spins, key, n_accept = _peps_sequential_sweep_with_accept(
            tensors, spins, shape, model.strategy, key
        )
        total_accept += int(n_accept)
        total_proposals += n_sites
        all_samples.append(spins.reshape(n_sites))
        _log_progress(sample_idx + 1, n_samples)

    samples = jnp.stack(all_samples, axis=0)
    samples = 2 * samples - 1
    acceptance_ratio = total_accept / total_proposals if total_proposals > 0 else 0.0
    return samples.astype(jnp.int32), acceptance_ratio


# ==============================================================================
# Baseline Methods
# ==============================================================================
def _fullsum_energy(hi, hamiltonian, model) -> complex:
    """Compute exact energy using FullSum."""
    fullsum_state = nk.vqs.FullSumState(hi, model)
    return complex(fullsum_state.expect(hamiltonian).mean)


def _compute_exact_distribution(model, n_sites: int) -> np.ndarray:
    """Compute exact probability distribution for small systems."""
    n_states = 2 ** n_sites
    all_configs = []
    for i in range(n_states):
        bits = [(i >> (n_sites - 1 - k)) & 1 for k in range(n_sites)]
        spins = [2 * b - 1 for b in bits]
        all_configs.append(spins)
    all_configs = jnp.array(all_configs, dtype=jnp.int32)

    log_amps = model(all_configs)
    log_probs = 2.0 * jnp.real(log_amps)
    probs = jnp.exp(log_probs - jnp.max(log_probs))
    probs = probs / jnp.sum(probs)
    return np.asarray(probs)


def _empirical_distribution(samples: jax.Array, n_sites: int) -> np.ndarray:
    """Compute empirical distribution from samples."""
    n_states = 2 ** n_sites
    samples_np = np.asarray(samples)
    counts = np.zeros(n_states, dtype=np.int32)
    for sample in samples_np:
        bits = (sample + 1) // 2
        idx = 0
        for bit in bits:
            idx = (idx << 1) | int(bit)
        counts[idx] += 1
    return counts / float(samples_np.shape[0])


def _total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute total variation distance between two distributions."""
    return 0.5 * np.sum(np.abs(p - q))


def _energy_from_samples(
    vstate: nk.vqs.MCState,
    hamiltonian,
    samples: jax.Array,
) -> tuple[complex, float]:
    """Compute energy statistics from samples."""
    n_chains = int(vstate.sampler.n_chains)
    n_sites = int(vstate.hilbert.size)
    chain_length = int(samples.shape[0] // n_chains)
    vstate._samples = samples.reshape(n_chains, chain_length, n_sites)

    apply_fun, params, model_state, kwargs = get_apply_fun(vstate)
    flat = vstate._samples.reshape(-1, n_sites)
    logpsi = jax.vmap(
        lambda s: apply_fun({"params": params, **model_state}, s, **kwargs)
    )(flat)
    vstate._logpsi = logpsi.reshape(n_chains, chain_length)

    stats = vstate.expect(hamiltonian)
    return complex(stats.mean), float(jnp.sqrt(stats.variance))


def _netket_mh_energy(
    hi,
    hamiltonian,
    model,
    n_samples: int,
    seed: int,
) -> tuple[complex, float, float]:
    """Compute energy using NetKet MetropolisLocal sampler. Returns (energy, std, acceptance)."""
    sweep_size = int(hi.size)
    sampler = nk.sampler.MetropolisLocal(
        hi, n_chains=1, sweep_size=sweep_size, reset_chains=False
    )
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=BURN_IN,
        sampler_seed=seed,
    )
    stats = vstate.expect(hamiltonian)
    acceptance = 0.0
    if hasattr(vstate.sampler_state, "acceptance"):
        acceptance = float(vstate.sampler_state.acceptance)
    return complex(stats.mean), float(jnp.sqrt(stats.variance)), acceptance


# ==============================================================================
# Test Cases
# ==============================================================================
def test_mps_fullsum(length: int) -> bool:
    """Test MPS sequential sampling against FullSum."""
    logger.info(f"\n--- MPS {length}x{length} (FullSum baseline) ---")
    logger.info(f"  bond_dim={BOND_DIM}, n_samples={N_SAMPLES}, burn_in={BURN_IN}")

    hi, hamiltonian, _ = build_heisenberg_square(length, pbc=False)
    n_sites = int(hi.size)

    model = MPS(
        rngs=nnx.Rngs(SEED),
        n_sites=n_sites,
        bond_dim=BOND_DIM,
    )

    # Sequential sampling
    t0 = time.perf_counter()
    samples, seq_accept = _sequential_mps_with_progress(
        model, N_SAMPLES, BURN_IN, jax.random.key(SEED)
    )
    t_sample = time.perf_counter() - t0

    # Build vstate for energy computation
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=1)
    vstate = nk.vqs.MCState(sampler, model, n_samples=N_SAMPLES, n_discard_per_chain=0)

    seq_energy, seq_std = _energy_from_samples(vstate, hamiltonian, samples)
    exact_energy = _fullsum_energy(hi, hamiltonian, model)

    # Distribution comparison
    exact_dist = _compute_exact_distribution(model, n_sites)
    empirical_dist = _empirical_distribution(samples, n_sites)
    tvd = _total_variation_distance(exact_dist, empirical_dist)
    max_diff = np.max(np.abs(exact_dist - empirical_dist))

    energy_diff = abs(seq_energy - exact_energy)

    logger.info(f"  Sequential runtime:    {t_sample:.2f}s")
    logger.info(f"  Sequential acceptance: {seq_accept:.4f}")
    logger.info(f"  Sequential energy: {seq_energy.real:.6f} +/- {seq_std:.6f}")
    logger.info(f"  FullSum energy:    {exact_energy.real:.6f}")
    logger.info(f"  Energy difference: {energy_diff:.6e}")
    logger.info(f"  TVD:               {tvd:.4f}")
    logger.info(f"  Max prob diff:     {max_diff:.4f}")

    # Pass criteria
    passed = energy_diff < 3 * seq_std and tvd < 0.15
    status = "PASS" if passed else "FAIL"
    logger.info(f"  Status: {status}")
    return passed


def test_peps_fullsum(length: int) -> bool:
    """Test PEPS sequential sampling against FullSum."""
    logger.info(f"\n--- PEPS {length}x{length} (FullSum baseline) ---")
    logger.info(
        f"  bond_dim={BOND_DIM}, chi={CHI}, n_samples={N_SAMPLES}, burn_in={BURN_IN}"
    )

    hi, hamiltonian, _ = build_heisenberg_square(length, pbc=False)
    shape = (length, length)

    model = PEPS(
        rngs=nnx.Rngs(SEED),
        shape=shape,
        bond_dim=BOND_DIM,
        contraction_strategy=ZipUp(truncate_bond_dimension=CHI),
    )

    # Sequential sampling
    t0 = time.perf_counter()
    samples, seq_accept = _sequential_peps_with_progress(
        model, N_SAMPLES, BURN_IN, jax.random.key(SEED)
    )
    t_sample = time.perf_counter() - t0

    # Build vstate for energy computation
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=1)
    vstate = nk.vqs.MCState(sampler, model, n_samples=N_SAMPLES, n_discard_per_chain=0)

    seq_energy, seq_std = _energy_from_samples(vstate, hamiltonian, samples)
    exact_energy = _fullsum_energy(hi, hamiltonian, model)

    # Distribution comparison
    exact_dist = _compute_exact_distribution(model, n_sites)
    empirical_dist = _empirical_distribution(samples, n_sites)
    tvd = _total_variation_distance(exact_dist, empirical_dist)
    max_diff = np.max(np.abs(exact_dist - empirical_dist))

    energy_diff = abs(seq_energy - exact_energy)

    logger.info(f"  Sequential runtime:    {t_sample:.2f}s")
    logger.info(f"  Sequential acceptance: {seq_accept:.4f}")
    logger.info(f"  Sequential energy: {seq_energy.real:.6f} +/- {seq_std:.6f}")
    logger.info(f"  FullSum energy:    {exact_energy.real:.6f}")
    logger.info(f"  Energy difference: {energy_diff:.6e}")
    logger.info(f"  TVD:               {tvd:.4f}")
    logger.info(f"  Max prob diff:     {max_diff:.4f}")

    # Pass criteria (slightly relaxed for PEPS due to truncation)
    passed = energy_diff < 3 * seq_std and tvd < 0.20
    status = "PASS" if passed else "FAIL"
    logger.info(f"  Status: {status}")
    return passed


def test_mps_netket(length: int) -> bool:
    """Test MPS sequential sampling against NetKet MetropolisLocal."""
    logger.info(f"\n--- MPS {length}x{length} (NetKet MH baseline) ---")
    logger.info(f"  bond_dim={BOND_DIM}, n_samples={N_SAMPLES}, burn_in={BURN_IN}")

    hi, hamiltonian, _ = build_heisenberg_square(length, pbc=False)
    n_sites = int(hi.size)

    model = MPS(
        rngs=nnx.Rngs(SEED),
        n_sites=n_sites,
        bond_dim=BOND_DIM,
    )

    # Sequential sampling
    t0 = time.perf_counter()
    samples, seq_accept = _sequential_mps_with_progress(
        model, N_SAMPLES, BURN_IN, jax.random.key(SEED)
    )
    t_seq = time.perf_counter() - t0

    # Build vstate for energy computation
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=1)
    vstate = nk.vqs.MCState(sampler, model, n_samples=N_SAMPLES, n_discard_per_chain=0)

    seq_energy, seq_std = _energy_from_samples(vstate, hamiltonian, samples)

    # NetKet baseline
    t0 = time.perf_counter()
    nk_energy, nk_std, nk_accept = _netket_mh_energy(
        hi, hamiltonian, model, N_SAMPLES, SEED + 100
    )
    t_nk = time.perf_counter() - t0

    energy_diff = abs(seq_energy - nk_energy)
    combined_std = np.sqrt(seq_std**2 + nk_std**2)

    logger.info(f"  Sequential runtime:    {t_seq:.2f}s")
    logger.info(f"  NetKet runtime:        {t_nk:.2f}s")
    logger.info(f"  Runtime speedup:       {t_nk/t_seq:.2f}x" if t_seq > 0 else "  Runtime speedup: N/A")
    logger.info(f"  Sequential acceptance: {seq_accept:.4f}")
    logger.info(f"  NetKet acceptance:     {nk_accept:.4f}")
    logger.info(f"  Sequential energy: {seq_energy.real:.6f} +/- {seq_std:.6f}")
    logger.info(f"  NetKet energy:     {nk_energy.real:.6f} +/- {nk_std:.6f}")
    logger.info(f"  Energy difference: {energy_diff:.6e}")
    logger.info(f"  Combined std:      {combined_std:.6f}")

    # Pass criteria
    passed = energy_diff < 3 * combined_std
    status = "PASS" if passed else "FAIL"
    logger.info(f"  Status: {status}")
    return passed


def test_peps_netket(length: int) -> bool:
    """Test PEPS sequential sampling against reduced NetKet MH baseline."""
    # Use fewer samples for NetKet because it's O(N) slower per sample for PEPS
    nk_samples = 256  # Reduced for tractability
    nk_burn_in = 50   # Reduced burn-in

    logger.info(f"\n--- PEPS {length}x{length} (NetKet MH baseline - reduced) ---")
    logger.info(f"  bond_dim={BOND_DIM}, chi={CHI}")
    logger.info(f"  Sequential: n_samples={N_SAMPLES}, burn_in={BURN_IN}")
    logger.info(
        "  NetKet:     n_samples=%d, burn_in=%d (reduced - PEPS contraction is expensive)",
        nk_samples,
        nk_burn_in,
    )

    hi, hamiltonian, _ = build_heisenberg_square(length, pbc=False)
    n_sites = int(hi.size)
    shape = (length, length)

    model = PEPS(
        rngs=nnx.Rngs(SEED),
        shape=shape,
        bond_dim=BOND_DIM,
        contraction_strategy=ZipUp(truncate_bond_dimension=CHI),
    )

    # Sequential sampling (full)
    t0 = time.perf_counter()
    samples, seq_accept = _sequential_peps_with_progress(
        model, N_SAMPLES, BURN_IN, jax.random.key(SEED)
    )
    t_seq = time.perf_counter() - t0

    # Build vstate for energy computation
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=1)
    vstate = nk.vqs.MCState(sampler, model, n_samples=N_SAMPLES, n_discard_per_chain=0)

    seq_energy, seq_std = _energy_from_samples(vstate, hamiltonian, samples)

    # NetKet baseline (reduced samples)
    logger.info(f"  Running reduced NetKet MH baseline...")
    t0 = time.perf_counter()
    sweep_size = int(hi.size)
    nk_sampler = nk.sampler.MetropolisLocal(
        hi, n_chains=1, sweep_size=sweep_size, reset_chains=False
    )
    nk_vstate = nk.vqs.MCState(
        nk_sampler,
        model,
        n_samples=nk_samples,
        n_discard_per_chain=nk_burn_in,
        sampler_seed=SEED + 100,
    )
    nk_stats = nk_vstate.expect(hamiltonian)
    nk_energy = complex(nk_stats.mean)
    nk_std = float(jnp.sqrt(nk_stats.variance))
    nk_accept = float(nk_vstate.sampler_state.acceptance) if hasattr(nk_vstate.sampler_state, "acceptance") else 0.0
    t_nk = time.perf_counter() - t0

    energy_diff = abs(seq_energy - nk_energy)
    combined_std = np.sqrt(seq_std**2 + nk_std**2)

    logger.info(f"  Sequential runtime:    {t_seq:.2f}s ({N_SAMPLES} samples)")
    logger.info(f"  NetKet runtime:        {t_nk:.2f}s ({nk_samples} samples)")
    logger.info(f"  Estimated NetKet full: {t_nk * N_SAMPLES / nk_samples:.1f}s (extrapolated)")
    logger.info(f"  Sequential acceptance: {seq_accept:.4f}")
    logger.info(f"  NetKet acceptance:     {nk_accept:.4f}")
    logger.info(f"  Sequential energy: {seq_energy.real:.6f} +/- {seq_std:.6f}")
    logger.info(f"  NetKet energy:     {nk_energy.real:.6f} +/- {nk_std:.6f}")
    logger.info(f"  Energy difference: {energy_diff:.6e}")
    logger.info(f"  Combined std:      {combined_std:.6f}")

    # Pass criteria (relaxed due to fewer NetKet samples)
    passed = energy_diff < 4 * combined_std
    status = "PASS" if passed else "FAIL"
    logger.info(f"  Status: {status}")
    return passed


# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    logger.info("=" * 60)
    logger.info("Sequential Metropolis-Hastings Sampling Verification")
    logger.info("=" * 60)
    logger.info(f"Parameters:")
    logger.info(f"  bond_dim = {BOND_DIM}")
    logger.info(f"  chi = {CHI} (ZipUp truncation)")
    logger.info(f"  n_samples = {N_SAMPLES}")
    logger.info(f"  burn_in = {BURN_IN}")

    results = []

    # 3x3 FullSum tests
    results.append(("MPS 3x3 FullSum", test_mps_fullsum(3)))
    results.append(("PEPS 3x3 FullSum", test_peps_fullsum(3)))

    # 5x5 NetKet tests
    results.append(("MPS 5x5 NetKet", test_mps_netket(5)))
    results.append(("PEPS 5x5 NetKet", test_peps_netket(5)))

    # 8x8 NetKet tests
    results.append(("MPS 8x8 NetKet", test_mps_netket(8)))
    results.append(("PEPS 8x8 NetKet", test_peps_netket(8)))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {name}: {status}")
        all_passed = all_passed and passed

    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("All tests PASSED!")
    else:
        logger.info("Some tests FAILED!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
