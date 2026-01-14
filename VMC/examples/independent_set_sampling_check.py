"""Correctness checks for independent-set samplers."""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import logging
import math
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from VMC.models.mps import SimpleMPS
from VMC.models.peps import SimplePEPS, ZipUp
from VMC.utils import (
    DiscardBlockedSampler,
    IndependentSetSampler,
    all_config_batches,
    build_neighbor_arrays,
    config_codes,
    enumerate_all_configs,
    enumerate_independent_sets_grid,
    grid_edges,
    independent_set_violations,
    occupancy_to_spin,
)
from VMC.core.eval import _value
from VMC.utils.vmc_utils import batched_eval

logger = logging.getLogger(__name__)

N_SAMPLES = 4096
BATCH_SIZE = 256
FULL_HILBERT_BATCH_SIZE = 1 << 18
FULL_HILBERT_SHAPE = (3, 3)
SEEDS = (0, 1, 2)

MPS_BOND_DIM = 4
PEPS_BOND_DIM = 2
PEPS_TRUNCATE_BOND_DIM = 2 * PEPS_BOND_DIM**2
DTYPE = jnp.complex128
METRICS: list[dict[str, float | int | str]] = []

SWEEP_FACTOR_MPS_FULL = 200
SWEEP_FACTOR_PEPS_FULL = 200
SWEEP_FACTOR_MPS_FULLSUM = 40
SWEEP_FACTOR_PEPS_FULLSUM = 80
SWEEP_FACTOR_MPS_COMPARE = 20
SWEEP_FACTOR_PEPS_COMPARE = 40

TFIM_COUPLING = 1.0
TFIM_FIELD = 0.5

RUN_MPS = False
RUN_PEPS = True


def get_mh_metrics() -> list[dict[str, float | int | str]]:
    """Return collected MH statistical metrics."""
    return list(METRICS)


def reset_mh_metrics() -> None:
    """Clear collected MH statistical metrics."""
    METRICS.clear()


def weighted_mean(log_prob: jax.Array, values: jax.Array) -> jax.Array:
    """Compute weighted mean of values using log-probabilities."""
    log_norm = jax.scipy.special.logsumexp(log_prob)
    weights = jnp.exp(log_prob - log_norm)
    return jnp.sum(weights * values)


def sample_mean_and_error(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Compute sample mean and naive standard error."""
    values = jnp.real(values)
    mean = jnp.mean(values)
    err = jnp.std(values, ddof=1) / math.sqrt(values.shape[0])
    return mean, err


def heisenberg_local_energy(
    model,
    spins: jax.Array,
    edges: Sequence[tuple[int, int]],
) -> jax.Array:
    """Compute Heisenberg local energy for a batch of spin configurations."""
    log_amp = batched_eval(
        lambda batch: jnp.log(jax.vmap(lambda s: _value(model, s))(batch)),
        spins,
        batch_size=BATCH_SIZE,
    )
    energy = jnp.zeros((spins.shape[0],), dtype=log_amp.dtype)
    edge_idx = jnp.asarray(edges, dtype=jnp.int32)

    def scan_fn(energy_acc, edge):
        i = edge[0]
        j = edge[1]
        s_i = spins[:, i]
        s_j = spins[:, j]
        energy_acc = energy_acc + 0.25 * s_i * s_j
        flipped = spins.at[:, i].set(-s_i).at[:, j].set(-s_j)
        log_amp_flipped = batched_eval(
            lambda batch: jnp.log(jax.vmap(lambda s: _value(model, s))(batch)),
            flipped,
            batch_size=BATCH_SIZE,
        )
        ratio = jnp.exp(log_amp_flipped - log_amp)
        energy_acc = energy_acc + 0.5 * jnp.where(s_i != s_j, ratio, 0.0)
        return energy_acc, None

    energy, _ = jax.lax.scan(scan_fn, energy, edge_idx)
    return energy


def tfim_local_energy(
    model,
    spins: jax.Array,
    edges: Sequence[tuple[int, int]],
    *,
    coupling: float,
    field: float,
) -> jax.Array:
    """Compute transverse-field Ising local energy for a batch of spins."""
    log_amp = batched_eval(
        lambda batch: jnp.log(jax.vmap(lambda s: _value(model, s))(batch)),
        spins,
        batch_size=BATCH_SIZE,
    )
    energy = jnp.zeros((spins.shape[0],), dtype=log_amp.dtype)
    if edges:
        edge_idx = jnp.asarray(edges, dtype=jnp.int32)
        s_i = spins[:, edge_idx[:, 0]]
        s_j = spins[:, edge_idx[:, 1]]
        energy = energy + 0.25 * coupling * jnp.sum(s_i * s_j, axis=1)

    site_idx = jnp.arange(spins.shape[1], dtype=jnp.int32)

    def scan_fn(energy_acc, site):
        s_i = spins[:, site]
        flipped = spins.at[:, site].set(-s_i)
        log_amp_flipped = batched_eval(
            lambda batch: jnp.log(jax.vmap(lambda s: _value(model, s))(batch)),
            flipped,
            batch_size=BATCH_SIZE,
        )
        ratio = jnp.exp(log_amp_flipped - log_amp)
        energy_acc = energy_acc + 0.5 * field * ratio
        return energy_acc, None

    energy, _ = jax.lax.scan(scan_fn, energy, site_idx)
    return energy


def compare_full_hilbert_enumeration(
    num_sites: int,
    edges: Sequence[tuple[int, int]],
    independent_samples: jax.Array,
    *,
    batch_size: int,
) -> None:
    """Compare DP independent-set enumeration to a full Hilbert scan."""
    neighbors, mask = build_neighbor_arrays(num_sites, edges)
    codes_dp = jnp.sort(config_codes(independent_samples))
    allowed_codes = []
    for occupancies, codes in all_config_batches(num_sites, batch_size=batch_size):
        violations = independent_set_violations(occupancies, neighbors, mask)
        if bool(jnp.any(~violations)):
            allowed_codes.append(codes[~violations])
    if allowed_codes:
        codes_full = jnp.sort(jnp.concatenate(allowed_codes))
    else:
        codes_full = jnp.zeros((0,), dtype=jnp.uint32)

    total = 1 << num_sites
    blocked = total - int(codes_full.shape[0])
    if codes_full.shape != codes_dp.shape or not bool(jnp.all(codes_full == codes_dp)):
        mismatch = int(jnp.sum(codes_full[: codes_dp.shape[0]] != codes_dp))
        raise AssertionError(
            "Full Hilbert independent-set enumeration mismatch: "
            f"dp={int(codes_dp.shape[0])} full={int(codes_full.shape[0])} "
            f"blocked={blocked} mismatched_prefix={mismatch}"
        )
    logger.info(
        "Full Hilbert enumeration matched independent sets: total=%d allowed=%d "
        "blocked=%d",
        total,
        int(codes_full.shape[0]),
        blocked,
    )


def compare_full_hilbert_amplitudes(
    log_prob,
    all_configs: jax.Array,
    independent_samples: jax.Array,
    neighbors: jax.Array,
    mask: jax.Array,
    *,
    label: str,
) -> None:
    """Compare log-probabilities for independent sets via full Hilbert scan."""
    log_prob_all = batched_eval(log_prob, all_configs, batch_size=BATCH_SIZE)
    violations = independent_set_violations(all_configs, neighbors, mask)
    allowed_configs = all_configs[~violations]
    allowed_log_prob = log_prob_all[~violations]

    codes_full = config_codes(allowed_configs)
    order_full = jnp.argsort(codes_full)
    codes_full = codes_full[order_full]
    allowed_log_prob = allowed_log_prob[order_full]

    log_prob_ind = batched_eval(
        log_prob, independent_samples, batch_size=BATCH_SIZE
    )
    codes_ind = config_codes(independent_samples)
    order_ind = jnp.argsort(codes_ind)
    codes_ind = codes_ind[order_ind]
    log_prob_ind = log_prob_ind[order_ind]

    if codes_full.shape != codes_ind.shape or not bool(jnp.all(codes_full == codes_ind)):
        raise AssertionError(f"{label} mismatch in independent-set codes.")

    max_diff = float(jnp.max(jnp.abs(allowed_log_prob - log_prob_ind)))
    logger.info("%s full Hilbert amplitude max diff=%.3e", label, max_diff)


def flippable_sites_for_config(
    occupancies: jax.Array, neighbors: jax.Array, mask: jax.Array
) -> list[int]:
    """Return flippable site indices for a single configuration."""
    safe_neighbors = jnp.where(mask, neighbors, 0)
    neighbor_occ = jnp.take(occupancies, safe_neighbors, axis=-1)
    blocked = jnp.sum(neighbor_occ * mask, axis=-1)
    flippable = (occupancies == 1) | ((occupancies == 0) & (blocked == 0))
    return [int(i) for i in jnp.where(flippable)[0]]


def build_mh_transition_matrix(
    log_prob,
    allowed_configs: jax.Array,
    neighbors: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Construct the MH transition matrix on the allowed configuration space."""
    log_prob = batched_eval(log_prob, allowed_configs, batch_size=BATCH_SIZE)
    log_prob = jnp.real(log_prob)
    log_prob = log_prob - jax.scipy.special.logsumexp(log_prob)
    target_prob = jnp.exp(log_prob)

    codes = config_codes(allowed_configs)
    codes_list = [int(code) for code in codes]
    code_to_index = {code: idx for idx, code in enumerate(codes_list)}

    flippable_sites = []
    flippable_counts = []
    for config in allowed_configs:
        sites = flippable_sites_for_config(config, neighbors, mask)
        flippable_sites.append(sites)
        flippable_counts.append(len(sites))

    n_states = len(codes_list)
    transition = jnp.zeros((n_states, n_states), dtype=jnp.float64)

    for idx, config in enumerate(allowed_configs):
        row_sum = 0.0
        n_flip = flippable_counts[idx]
        if n_flip <= 0:
            raise AssertionError(
                "No flippable moves available for a valid configuration."
            )
        code = codes_list[idx]
        for site in flippable_sites[idx]:
            next_code = code ^ (1 << site)
            next_idx = code_to_index[next_code]
            n_flip_next = flippable_counts[next_idx]
            ratio = float(target_prob[next_idx]) / float(target_prob[idx])
            ratio *= float(n_flip) / float(n_flip_next)
            accept = 1.0 if ratio >= 1.0 else ratio
            prob = accept / float(n_flip)
            transition = transition.at[idx, next_idx].add(prob)
            row_sum += prob
        transition = transition.at[idx, idx].set(1.0 - row_sum)
    return transition, target_prob


def compare_mh_stationary_distribution(
    log_prob,
    allowed_configs: jax.Array,
    neighbors: jax.Array,
    mask: jax.Array,
    *,
    label: str,
    tol: float = 1e-10,
) -> None:
    """Verify MH stationary distribution matches the full-sum target."""
    transition, target_prob = build_mh_transition_matrix(
        log_prob, allowed_configs, neighbors, mask
    )
    stationary = target_prob @ transition
    max_diff = float(jnp.max(jnp.abs(stationary - target_prob)))
    logger.info("%s MH stationary max diff=%.3e", label, max_diff)
    if max_diff > tol:
        raise AssertionError(
            f"{label} MH stationary mismatch: max_diff={max_diff:.3e} tol={tol:.3e}"
        )


def sample_with_progress(
    sampler,
    log_prob,
    *,
    key: jax.Array,
    n_samples: int,
    n_sweeps: int,
    label: str,
    log_interval: int,
) -> tuple[jax.Array, jax.Array]:
    """Sample in batches with progress logging."""
    import time

    all_samples = []
    n_batches = (n_samples + log_interval - 1) // log_interval
    keys = jax.random.split(key, n_batches)
    start_time = time.time()
    acceptance = jnp.asarray(0.0)

    for i, batch_key in enumerate(keys):
        batch_size = min(log_interval, n_samples - i * log_interval)
        if batch_size <= 0:
            break
        batch_samples, _, _, acceptance = sampler.sample(
            log_prob, n_samples=batch_size, n_sweeps=n_sweeps, key=batch_key
        )
        all_samples.append(batch_samples)
        elapsed = time.time() - start_time
        completed = min((i + 1) * log_interval, n_samples)
        logger.info(
            "%s: %d/%d samples (%.1fs elapsed, acceptance=%.3f)",
            label, completed, n_samples, elapsed, float(acceptance),
        )

    return jnp.concatenate(all_samples, axis=0), acceptance


def compare_mh_statistical_distribution(
    log_prob,
    allowed_configs: jax.Array,
    neighbors: jax.Array,
    mask: jax.Array,
    *,
    sampler: IndependentSetSampler,
    key: jax.Array,
    n_samples: int,
    n_sweeps: int,
    label: str,
    alpha: float = 1e-3,
    log_interval: int = 1000,
) -> dict[str, float | int | str]:
    """Compare empirical MH samples to the exact target distribution."""
    from scipy import stats as scipy_stats

    log_prob_vals = batched_eval(log_prob, allowed_configs, batch_size=BATCH_SIZE)
    log_prob_vals = jnp.real(log_prob_vals)
    log_prob_vals = log_prob_vals - jax.scipy.special.logsumexp(log_prob_vals)
    target_prob = jnp.exp(log_prob_vals)

    samples, _ = sample_with_progress(
        sampler,
        log_prob,
        key=key,
        n_samples=n_samples,
        n_sweeps=n_sweeps,
        label=f"{label} MH",
        log_interval=log_interval,
    )
    verify_no_violations(samples, neighbors, mask, f"{label} MH samples")

    allowed_codes = config_codes(allowed_configs)
    order = jnp.argsort(allowed_codes)
    allowed_codes = allowed_codes[order]
    target_prob = target_prob[order]

    sample_codes = config_codes(samples)
    indices = jnp.searchsorted(allowed_codes, sample_codes)
    if not bool(jnp.all(allowed_codes[indices] == sample_codes)):
        raise AssertionError(f"{label} contains samples outside allowed configs.")

    counts = jnp.bincount(indices, length=allowed_codes.shape[0]).astype(jnp.float64)
    expected = target_prob * float(n_samples)

    mask_big = expected >= 5.0
    if not bool(jnp.all(mask_big)):
        expected_other = jnp.sum(expected[~mask_big])
        observed_other = jnp.sum(counts[~mask_big])
        expected = jnp.concatenate([expected[mask_big], expected_other[None]])
        counts = jnp.concatenate([counts[mask_big], observed_other[None]])

    chi2 = float(jnp.sum((counts - expected) ** 2 / expected))
    df = int(expected.shape[0] - 1)
    p_value = float(scipy_stats.chi2.sf(chi2, df))
    min_expected = float(jnp.min(expected))
    logger.info(
        "%s MH chi2=%.3f df=%d p=%.3e min_expected=%.2f",
        label, chi2, df, p_value, min_expected,
    )
    metric = {
        "label": label,
        "chi2": chi2,
        "df": df,
        "p_value": p_value,
        "min_expected": min_expected,
        "n_samples": int(n_samples),
        "n_sweeps": int(n_sweeps),
    }
    METRICS.append(metric)
    if p_value < alpha:
        raise AssertionError(
            f"{label} MH statistical mismatch: p={p_value:.3e} < alpha={alpha:.1e}"
        )
    return metric


def weighted_stats(log_prob: jax.Array, samples: jax.Array) -> dict[str, jax.Array]:
    """Compute weighted density statistics from log-probabilities."""
    occupancies = samples.astype(jnp.float64)
    density = jnp.mean(occupancies, axis=1)
    log_norm = jax.scipy.special.logsumexp(log_prob)
    weights = jnp.exp(log_prob - log_norm)
    mean_density = jnp.sum(weights * density)
    mean_density_sq = jnp.sum(weights * density * density)
    return {
        "mean_density": mean_density,
        "mean_density_sq": mean_density_sq,
    }


def sample_stats(samples: jax.Array) -> dict[str, jax.Array]:
    """Compute sample mean statistics and naive standard errors."""
    occupancies = samples.astype(jnp.float64)
    density = jnp.mean(occupancies, axis=1)
    mean_density = jnp.mean(density)
    mean_density_sq = jnp.mean(density * density)
    err_density = jnp.std(density, ddof=1) / math.sqrt(samples.shape[0])
    err_density_sq = jnp.std(density * density, ddof=1) / math.sqrt(samples.shape[0])
    return {
        "mean_density": mean_density,
        "mean_density_sq": mean_density_sq,
        "err_density": err_density,
        "err_density_sq": err_density_sq,
    }


def assert_match(
    label: str, estimate: jax.Array, reference: jax.Array, err: jax.Array
) -> None:
    diff = float(jnp.abs(estimate - reference))
    tol = float(5.0 * err + 1e-3)
    if diff > tol:
        raise AssertionError(
            f"{label} mismatch: diff={diff:.3e} tol={tol:.3e} "
            f"estimate={float(estimate):.6f} reference={float(reference):.6f}"
        )


def verify_no_violations(
    samples: jax.Array, neighbors: jax.Array, mask: jax.Array, label: str
) -> None:
    violations = independent_set_violations(samples, neighbors, mask)
    if bool(jnp.any(violations)):
        raise AssertionError(f"{label} produced blocked configurations.")


def run_fullsum_check(
    *,
    model_name: str,
    log_prob,
    sampler: IndependentSetSampler,
    full_samples: jax.Array,
    neighbors: jax.Array,
    mask: jax.Array,
    seed: int,
    key: jax.Array,
    n_sweeps: int,
) -> None:
    log_prob_full = batched_eval(log_prob, full_samples, batch_size=BATCH_SIZE)
    exact = weighted_stats(log_prob_full, full_samples)

    samples, _, _, acceptance = sampler.sample(
        log_prob, n_samples=N_SAMPLES, n_sweeps=n_sweeps, key=key
    )
    verify_no_violations(samples, neighbors, mask, f"{model_name} seed={seed} fullsum")
    stats = sample_stats(samples)

    logger.info(
        "%s seed=%d fullsum acceptance=%.3f density=%.6f exact=%.6f",
        model_name,
        seed,
        float(acceptance),
        float(stats["mean_density"]),
        float(exact["mean_density"]),
    )
    assert_match(
        f"{model_name} seed={seed} density",
        stats["mean_density"],
        exact["mean_density"],
        stats["err_density"],
    )
    assert_match(
        f"{model_name} seed={seed} density_sq",
        stats["mean_density_sq"],
        exact["mean_density_sq"],
        stats["err_density_sq"],
    )


def run_energy_check(
    *,
    model_name: str,
    model,
    log_prob,
    sampler: IndependentSetSampler,
    full_samples: jax.Array,
    edges: Sequence[tuple[int, int]],
    neighbors: jax.Array,
    mask: jax.Array,
    seed: int,
    key: jax.Array,
    n_sweeps: int,
) -> None:
    log_prob_full = batched_eval(log_prob, full_samples, batch_size=BATCH_SIZE)
    full_spins = occupancy_to_spin(full_samples)
    local_energy_full = heisenberg_local_energy(model, full_spins, edges)
    exact_energy = weighted_mean(log_prob_full, local_energy_full)
    local_energy_full_tfim = tfim_local_energy(
        model, full_spins, edges, coupling=TFIM_COUPLING, field=TFIM_FIELD
    )
    exact_energy_tfim = weighted_mean(log_prob_full, local_energy_full_tfim)

    samples, _, _, acceptance = sampler.sample(
        log_prob, n_samples=N_SAMPLES, n_sweeps=n_sweeps, key=key
    )
    verify_no_violations(samples, neighbors, mask, f"{model_name} seed={seed} energy")
    sample_spins = occupancy_to_spin(samples)
    local_energy_samples = heisenberg_local_energy(model, sample_spins, edges)
    mean_energy, err_energy = sample_mean_and_error(local_energy_samples)
    local_energy_samples_tfim = tfim_local_energy(
        model,
        sample_spins,
        edges,
        coupling=TFIM_COUPLING,
        field=TFIM_FIELD,
    )
    mean_energy_tfim, err_energy_tfim = sample_mean_and_error(
        local_energy_samples_tfim
    )

    logger.info(
        "%s seed=%d Heisenberg acceptance=%.3f energy=%.6f exact=%.6f",
        model_name,
        seed,
        float(acceptance),
        float(mean_energy),
        float(jnp.real(exact_energy)),
    )
    logger.info(
        "%s seed=%d TFIM(J=%.2f,h=%.2f) acceptance=%.3f energy=%.6f exact=%.6f",
        model_name,
        seed,
        TFIM_COUPLING,
        TFIM_FIELD,
        float(acceptance),
        float(mean_energy_tfim),
        float(jnp.real(exact_energy_tfim)),
    )
    assert_match(
        f"{model_name} seed={seed} Heisenberg energy",
        mean_energy,
        jnp.real(exact_energy),
        err_energy,
    )
    assert_match(
        f"{model_name} seed={seed} TFIM energy",
        mean_energy_tfim,
        jnp.real(exact_energy_tfim),
        err_energy_tfim,
    )


def run_sampler_comparison(
    *,
    model_name: str,
    log_prob,
    sampler_a: IndependentSetSampler,
    sampler_b: DiscardBlockedSampler,
    neighbors: jax.Array,
    mask: jax.Array,
    seed: int,
    key: jax.Array,
    n_sweeps: int,
    log_interval: int = 1000,
) -> None:
    key_a, key_b = jax.random.split(key, 2)

    samples_a, acceptance_a = sample_with_progress(
        sampler_a,
        log_prob,
        key=key_a,
        n_samples=N_SAMPLES,
        n_sweeps=n_sweeps,
        label=f"{model_name} seed={seed} sampler_a",
        log_interval=log_interval,
    )
    samples_b, acceptance_b = sample_with_progress(
        sampler_b,
        log_prob,
        key=key_b,
        n_samples=N_SAMPLES,
        n_sweeps=n_sweeps,
        label=f"{model_name} seed={seed} sampler_b",
        log_interval=log_interval,
    )

    verify_no_violations(samples_a, neighbors, mask, f"{model_name} seed={seed} a")
    verify_no_violations(samples_b, neighbors, mask, f"{model_name} seed={seed} b")
    stats_a = sample_stats(samples_a)
    stats_b = sample_stats(samples_b)
    err_density = math.sqrt(
        float(stats_a["err_density"] ** 2 + stats_b["err_density"] ** 2)
    )
    err_density_sq = math.sqrt(
        float(stats_a["err_density_sq"] ** 2 + stats_b["err_density_sq"] ** 2)
    )

    logger.info(
        "%s seed=%d compare acceptance=(%.3f, %.3f) density=(%.6f, %.6f)",
        model_name,
        seed,
        float(acceptance_a),
        float(acceptance_b),
        float(stats_a["mean_density"]),
        float(stats_b["mean_density"]),
    )
    assert_match(
        f"{model_name} seed={seed} density compare",
        stats_a["mean_density"],
        stats_b["mean_density"],
        err_density,
    )
    assert_match(
        f"{model_name} seed={seed} density_sq compare",
        stats_a["mean_density_sq"],
        stats_b["mean_density_sq"],
        err_density_sq,
    )


def main() -> None:
    shape_full = FULL_HILBERT_SHAPE
    shape_fullsum = (5, 5)
    compare_shapes = [(8, 8), (10, 10)]

    edges_full = grid_edges(*shape_full)
    edges_fullsum = grid_edges(*shape_fullsum)

    full_samples = enumerate_independent_sets_grid(*shape_fullsum)
    full_samples_full = enumerate_independent_sets_grid(*shape_full)
    all_full_configs = enumerate_all_configs(shape_full[0] * shape_full[1])
    compare_full_hilbert_enumeration(
        shape_full[0] * shape_full[1],
        edges_full,
        full_samples_full,
        batch_size=FULL_HILBERT_BATCH_SIZE,
    )
    neighbors_full, mask_full = build_neighbor_arrays(
        shape_full[0] * shape_full[1], edges_full
    )
    neighbors_fullsum, mask_fullsum = build_neighbor_arrays(
        shape_fullsum[0] * shape_fullsum[1], edges_fullsum
    )

    sampler_full = IndependentSetSampler(shape_full[0] * shape_full[1], edges_full)
    sampler_fullsum = IndependentSetSampler(
        shape_fullsum[0] * shape_fullsum[1], edges_fullsum
    )

    compare_configs = []
    for shape in compare_shapes:
        edges = grid_edges(*shape)
        n_sites = shape[0] * shape[1]
        neighbors, mask = build_neighbor_arrays(n_sites, edges)
        compare_configs.append(
            {
                "label": f"{shape[0]}x{shape[1]}",
                "shape": shape,
                "neighbors": neighbors,
                "mask": mask,
                "sampler": IndependentSetSampler(n_sites, edges),
                "baseline": DiscardBlockedSampler(n_sites, edges),
                "n_sites": n_sites,
            }
        )

    n_sites_full = shape_full[0] * shape_full[1]
    n_sites_fullsum = shape_fullsum[0] * shape_fullsum[1]
    n_sweeps_full = SWEEP_FACTOR_MPS_FULL * n_sites_full
    n_sweeps_full_peps = SWEEP_FACTOR_PEPS_FULL * n_sites_full
    n_sweeps_fullsum_mps = SWEEP_FACTOR_MPS_FULLSUM * n_sites_fullsum
    n_sweeps_fullsum_peps = SWEEP_FACTOR_PEPS_FULLSUM * n_sites_fullsum

    logger.info("Running independent-set sampler checks...")

    if RUN_MPS:
        for seed in SEEDS:
            key_full, key_compare = jax.random.split(jax.random.key(seed + 100), 2)
            key_full, key_energy = jax.random.split(key_full, 2)
            key_hilbert = jax.random.key(seed + 50)
            mps_full = SimpleMPS(
                rngs=nnx.Rngs(seed),
                n_sites=n_sites_full,
                bond_dim=MPS_BOND_DIM,
                dtype=DTYPE,
            )
            log_prob_full = lambda occ: 2.0 * jnp.real(
                jnp.log(
                    jax.vmap(lambda s: _value(mps_full, s))(
                        occupancy_to_spin(occ)
                    )
                )
            )
            compare_full_hilbert_amplitudes(
                log_prob_full,
                all_full_configs,
                full_samples_full,
                neighbors_full,
                mask_full,
                label=f"MPS {shape_full[0]}x{shape_full[1]} seed={seed}",
            )
            compare_mh_stationary_distribution(
                log_prob_full,
                full_samples_full,
                neighbors_full,
                mask_full,
                label=f"MPS {shape_full[0]}x{shape_full[1]} seed={seed}",
            )
            compare_mh_statistical_distribution(
                log_prob_full,
                full_samples_full,
                neighbors_full,
                mask_full,
                sampler=sampler_full,
                key=key_hilbert,
                n_samples=N_SAMPLES,
                n_sweeps=n_sweeps_full,
                label=f"MPS {shape_full[0]}x{shape_full[1]} seed={seed}",
            )
            run_energy_check(
                model_name=f"MPS {shape_full[0]}x{shape_full[1]}",
                model=mps_full,
                log_prob=log_prob_full,
                sampler=sampler_full,
                full_samples=full_samples_full,
                edges=edges_full,
                neighbors=neighbors_full,
                mask=mask_full,
                seed=seed,
                key=key_energy,
                n_sweeps=n_sweeps_full,
            )
            mps_fullsum = SimpleMPS(
                rngs=nnx.Rngs(seed),
                n_sites=n_sites_fullsum,
                bond_dim=MPS_BOND_DIM,
                dtype=DTYPE,
            )
            log_prob_fullsum = lambda occ: 2.0 * jnp.real(
                jnp.log(
                    jax.vmap(lambda s: _value(mps_fullsum, s))(
                        occupancy_to_spin(occ)
                    )
                )
            )
            run_fullsum_check(
                model_name=f"MPS {shape_fullsum[0]}x{shape_fullsum[1]}",
                log_prob=log_prob_fullsum,
                sampler=sampler_fullsum,
                full_samples=full_samples,
                neighbors=neighbors_fullsum,
                mask=mask_fullsum,
                seed=seed,
                key=key_full,
                n_sweeps=n_sweeps_fullsum_mps,
            )

            compare_keys = jax.random.split(key_compare, len(compare_configs))
            for cfg, cfg_key in zip(compare_configs, compare_keys):
                mps_compare = SimpleMPS(
                    rngs=nnx.Rngs(seed),
                    n_sites=cfg["n_sites"],
                    bond_dim=MPS_BOND_DIM,
                    dtype=DTYPE,
                )
                log_prob_compare = lambda occ: 2.0 * jnp.real(
                    jnp.log(
                        jax.vmap(lambda s: _value(mps_compare, s))(
                            occupancy_to_spin(occ)
                        )
                    )
                )
                run_sampler_comparison(
                    model_name=f"MPS {cfg['label']}",
                    log_prob=log_prob_compare,
                    sampler_a=cfg["sampler"],
                    sampler_b=cfg["baseline"],
                    neighbors=cfg["neighbors"],
                    mask=cfg["mask"],
                    seed=seed,
                    key=cfg_key,
                    n_sweeps=SWEEP_FACTOR_MPS_COMPARE * cfg["n_sites"],
                )

    if RUN_PEPS:
        for seed in SEEDS:
            key_full, key_compare = jax.random.split(jax.random.key(seed + 200), 2)
            key_full, key_energy = jax.random.split(key_full, 2)
            key_hilbert = jax.random.key(seed + 150)
            peps_full = SimplePEPS(
                rngs=nnx.Rngs(seed),
                shape=shape_full,
                bond_dim=PEPS_BOND_DIM,
                contraction_strategy=ZipUp(
                    truncate_bond_dimension=PEPS_TRUNCATE_BOND_DIM
                ),
                dtype=DTYPE,
            )
            log_prob_full = lambda occ: 2.0 * jnp.real(
                jnp.log(
                    jax.vmap(lambda s: _value(peps_full, s))(
                        occupancy_to_spin(occ)
                    )
                )
            )
            compare_full_hilbert_amplitudes(
                log_prob_full,
                all_full_configs,
                full_samples_full,
                neighbors_full,
                mask_full,
                label=f"PEPS {shape_full[0]}x{shape_full[1]} seed={seed}",
            )
            compare_mh_statistical_distribution(
                log_prob_full,
                full_samples_full,
                neighbors_full,
                mask_full,
                sampler=sampler_full,
                key=key_hilbert,
                n_samples=N_SAMPLES,
                n_sweeps=n_sweeps_full_peps,
                label=f"PEPS {shape_full[0]}x{shape_full[1]} seed={seed}",
            )
            run_energy_check(
                model_name=f"PEPS {shape_full[0]}x{shape_full[1]}",
                model=peps_full,
                log_prob=log_prob_full,
                sampler=sampler_full,
                full_samples=full_samples_full,
                edges=edges_full,
                neighbors=neighbors_full,
                mask=mask_full,
                seed=seed,
                key=key_energy,
                n_sweeps=n_sweeps_full_peps,
            )
            peps_fullsum = SimplePEPS(
                rngs=nnx.Rngs(seed),
                shape=shape_fullsum,
                bond_dim=PEPS_BOND_DIM,
                contraction_strategy=ZipUp(
                    truncate_bond_dimension=PEPS_TRUNCATE_BOND_DIM
                ),
                dtype=DTYPE,
            )
            log_prob_fullsum = lambda occ: 2.0 * jnp.real(
                jnp.log(
                    jax.vmap(lambda s: _value(peps_fullsum, s))(
                        occupancy_to_spin(occ)
                    )
                )
            )
            run_fullsum_check(
                model_name=f"PEPS {shape_fullsum[0]}x{shape_fullsum[1]}",
                log_prob=log_prob_fullsum,
                sampler=sampler_fullsum,
                full_samples=full_samples,
                neighbors=neighbors_fullsum,
                mask=mask_fullsum,
                seed=seed,
                key=key_full,
                n_sweeps=n_sweeps_fullsum_peps,
            )

            compare_keys = jax.random.split(key_compare, len(compare_configs))
            for cfg, cfg_key in zip(compare_configs, compare_keys):
                peps_compare = SimplePEPS(
                    rngs=nnx.Rngs(seed),
                    shape=cfg["shape"],
                    bond_dim=PEPS_BOND_DIM,
                    contraction_strategy=ZipUp(
                        truncate_bond_dimension=PEPS_TRUNCATE_BOND_DIM
                    ),
                    dtype=DTYPE,
                )
                log_prob_compare = lambda occ: 2.0 * jnp.real(
                    jnp.log(
                        jax.vmap(lambda s: _value(peps_compare, s))(
                            occupancy_to_spin(occ)
                        )
                    )
                )
                run_sampler_comparison(
                    model_name=f"PEPS {cfg['label']}",
                    log_prob=log_prob_compare,
                    sampler_a=cfg["sampler"],
                    sampler_b=cfg["baseline"],
                    neighbors=cfg["neighbors"],
                    mask=cfg["mask"],
                    seed=seed,
                    key=cfg_key,
                    n_sweeps=SWEEP_FACTOR_PEPS_COMPARE * cfg["n_sites"],
                )


if __name__ == "__main__":
    main()
