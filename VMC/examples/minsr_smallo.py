"""MinSR with small-o trick for memory-efficient tensor network optimization.

This module implements the small-o trick from Wu 2025 (Section III.C) for
memory-efficient minSR (minimal Stochastic Reconfiguration) with PEPS/MPS.

The small-o trick exploits tensor network locality:
    O[x](s)_{p,lrdu} = 0 if p != s(x)

Instead of storing the full O matrix (N_s x N_p), we store only the non-zero
physical slices o[x](s)_{lrdu} = O[x](s)_{s(x),lrdu}, reducing memory by factor d.

References:
    Wu 2025, "Real-Time Dynamics in Two Dimensions with Tensor Network States
    via Time-Dependent Variational Monte Carlo", Section III.C and Supplementary S-1.
"""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import functools
import logging
import time
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from flax import nnx

from VMC.models.mps import SimpleMPS
from VMC.models.peps import SimplePEPS, ZipUp
from VMC.samplers.sequential import sequential_sample
from VMC.utils import build_dense_jac

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
VERIFY_TAG = "[verify]"


def _log_verify(message: str, *args) -> None:
    logger.info("%s " + message, VERIFY_TAG, *args)

__all__ = [
    # OO† building (summation orderings)
    "build_OOdag_phys_ordering",
    "build_OOdag_site_ordering",
    # minSR solve
    "build_minsr_matrix",
    "solve_minsr_smallo",
    "recover_updates_smallo",
    # Verification and demo
    "verify_smallo_correctness_mps",
    "verify_smallo_correctness_peps",
    "minsr_smallo_mps_demo",
    "minsr_smallo_peps_demo",
]


# =============================================================================
# OO† Building (Summation Orderings)
# =============================================================================


@functools.partial(jax.jit, static_argnames=("phys_dim",))
def build_OOdag_phys_ordering(
    o: jax.Array,
    p: jax.Array,
    phys_dim: int,
) -> jax.Array:
    """Build OO† by looping over physical indices (standard small-o trick).

    This is the standard ordering from Wu 2025 supplementary S-1:
        OO† = Σ_k õ(k) @ õ(k)†  where õ(k) = o * (p == k)

    Args:
        o: (n_samples, n_params_reduced) uncentered small-o Jacobian.
        p: (n_samples, n_params_reduced) int8 physical index tensor.
        phys_dim: Physical dimension (number of physical indices).

    Returns:
        OOdag: (n_samples, n_samples) minSR Gram matrix.
    """
    OOdag = jnp.zeros((o.shape[0], o.shape[0]), dtype=o.dtype)
    for k in range(phys_dim):
        o_tilde = jnp.where(p == k, o, 0)
        OOdag = OOdag + o_tilde @ o_tilde.conj().T
    return OOdag


@functools.partial(
    jax.jit, static_argnames=("n_sites", "params_per_site", "phys_dim")
)
def _build_OOdag_site_ordering(
    o: jax.Array,
    p: jax.Array,
    n_sites: int,
    params_per_site: tuple[int, ...],
    phys_dim: int,
) -> jax.Array:
    """Build OO† by looping over sites first, then physical indices.

    Alternative ordering mentioned in smallo.pdf. May have different
    cache locality characteristics.

    Args:
        o: (n_samples, n_params_reduced) uncentered small-o Jacobian.
        p: (n_samples, n_params_reduced) int8 physical index tensor.
        n_sites: Number of sites.
        params_per_site: Parameters per site (may vary for open boundaries).
        phys_dim: Physical dimension.

    Returns:
        OOdag: (n_samples, n_samples) minSR Gram matrix.
    """
    OOdag = jnp.zeros((o.shape[0], o.shape[0]), dtype=o.dtype)
    offset = 0
    for site in range(n_sites):
        n_params = params_per_site[site]
        o_site = o[:, offset : offset + n_params]
        p_site = p[:, offset : offset + n_params]
        for k in range(phys_dim):
            mask = (p_site == k)
            o_masked = jnp.where(mask, o_site, 0)
            OOdag = OOdag + o_masked @ o_masked.conj().T
        offset += n_params
    return OOdag


def build_OOdag_site_ordering(
    o: jax.Array,
    p: jax.Array,
    n_sites: int,
    params_per_site: list[int] | tuple[int, ...],
    phys_dim: int,
) -> jax.Array:
    """Wrapper ensuring params_per_site is static for JIT compilation."""
    if not isinstance(params_per_site, tuple):
        params_per_site = tuple(params_per_site)
    return _build_OOdag_site_ordering(o, p, n_sites, params_per_site, phys_dim)


# =============================================================================
# minSR Solve Functions
# =============================================================================


@jax.jit
def build_minsr_matrix(OOdag: jax.Array) -> jax.Array:
    """Compute centered minSR matrix from OO†.

    The minSR matrix T requires centering:
        T_ss' = OO† - <O>O† - O<O>† + <O><O>†

    From Wu 2025 supplementary Listing 2.

    Args:
        OOdag: (n_samples, n_samples) raw OO† Gram matrix.

    Returns:
        T: (n_samples, n_samples) centered minSR matrix.
    """
    OOavg_c = jnp.mean(OOdag, axis=1)
    return OOdag - OOavg_c.conj() - OOavg_c[:, None] + jnp.mean(OOavg_c)


@functools.partial(jax.jit, static_argnames=("phys_dim", "params_per_site"))
def _recover_updates_smallo(
    o: jax.Array,
    p: jax.Array,
    y: jax.Array,
    phys_dim: int,
    params_per_site: tuple[int, ...] | None = None,
) -> jax.Array:
    """Recover parameter updates in FULL parameter space from minSR solution using small-o.

    Computes updates = O_centered† @ y by looping over physical indices.
    Returns updates in the full parameter space (with physical index dimension).

    Args:
        o: (n_samples, n_params_reduced) UNCENTERED small-o Jacobian.
        p: (n_samples, n_params_reduced) int8 physical index tensor.
        y: (n_samples,) solution from minSR linear system.
        phys_dim: Physical dimension.
        params_per_site: Params per physical slice for each site.
            Required for proper layout mapping. If None, assumes uniform.

    Returns:
        updates: (n_params_full,) parameter updates in full space.
                 Layout: for each site, phys_dim consecutive blocks of params_per_phys.
    """
    n_samples = o.shape[0]
    n_params_reduced = o.shape[1]
    y_sum = jnp.sum(y)

    # If params_per_site not provided, assume uniform (won't work for open BC)
    if params_per_site is None:
        n_params_full = n_params_reduced * phys_dim
        updates = jnp.zeros(n_params_full, dtype=o.dtype)
        for k in range(phys_dim):
            mask = (p == k).astype(o.dtype)
            o_k = o * mask
            o_k_mean = jnp.sum(o_k, axis=0) / n_samples
            updates_k = o_k.conj().T @ y - o_k_mean.conj() * y_sum
            updates = updates.at[k::phys_dim].set(updates_k)
        return updates

    # With params_per_site, properly map reduced to full space
    n_sites = len(params_per_site)
    n_params_full = sum(phys_dim * pps for pps in params_per_site)
    updates = jnp.zeros(n_params_full, dtype=o.dtype)

    offset_reduced = 0
    offset_full = 0
    for site in range(n_sites):
        pps = params_per_site[site]  # params per physical slice at this site

        # Extract site slice from reduced arrays
        o_site = o[:, offset_reduced:offset_reduced + pps]
        p_site = p[:, offset_reduced:offset_reduced + pps]

        for k in range(phys_dim):
            # Mask for samples using physical index k at this site
            mask = (p_site == k).astype(o.dtype)
            o_k = o_site * mask
            o_k_mean = jnp.sum(o_k, axis=0) / n_samples

            # Centered contribution for physical index k at this site
            updates_k = o_k.conj().T @ y - o_k_mean.conj() * y_sum

            # Place in full updates: site block starts at offset_full,
            # physical index k block starts at offset_full + k * pps
            start = offset_full + k * pps
            updates = updates.at[start:start + pps].set(updates_k)

        offset_reduced += pps
        offset_full += phys_dim * pps

    return updates


def recover_updates_smallo(
    o: jax.Array,
    p: jax.Array,
    y: jax.Array,
    phys_dim: int,
    params_per_site: list[int] | tuple[int, ...] | None = None,
) -> jax.Array:
    """Wrapper ensuring params_per_site is static for JIT compilation."""
    if params_per_site is not None and not isinstance(params_per_site, tuple):
        params_per_site = tuple(params_per_site)
    return _recover_updates_smallo(o, p, y, phys_dim, params_per_site)


def solve_minsr_smallo(
    o: jax.Array,
    p: jax.Array,
    dv: jax.Array,
    phys_dim: int,
    diag_shift: float,
    ordering: str = "physical",
    n_sites: int | None = None,
    params_per_site: list[int] | tuple[int, ...] | None = None,
) -> tuple[jax.Array, dict]:
    """Solve minSR using small-o trick.

    Steps:
        1. Build OOdag via specified ordering
        2. Center to get minSR matrix T
        3. Solve (T + diag_shift*I) @ y = dv
        4. Recover updates = O† @ y via small-o

    Args:
        o: (n_samples, n_params_reduced) uncentered small-o Jacobian.
        p: (n_samples, n_params_reduced) int8 physical index tensor.
        dv: (n_samples,) gradient vector.
        phys_dim: Physical dimension.
        diag_shift: Regularization parameter.
        ordering: "physical" or "site" summation ordering.
        n_sites: Required if ordering="site".
        params_per_site: Required if ordering="site".

    Returns:
        updates: (n_params_reduced,) parameter updates.
        metrics: Dict with diagnostic information.
    """
    # Build OOdag
    if ordering == "physical":
        OOdag = build_OOdag_phys_ordering(o, p, phys_dim)
    elif ordering == "site":
        if n_sites is None or params_per_site is None:
            raise ValueError("n_sites and params_per_site required for site ordering")
        OOdag = build_OOdag_site_ordering(o, p, n_sites, params_per_site, phys_dim)
    else:
        raise ValueError(f"Unknown ordering: {ordering}")

    # Center to get minSR matrix
    T = build_minsr_matrix(OOdag)

    # Regularize and solve
    n_samples = o.shape[0]
    T_reg = T + diag_shift * jnp.eye(n_samples, dtype=T.dtype)
    y = jsp.linalg.solve(T_reg, dv, assume_a="pos")

    # Recover updates
    updates = recover_updates_smallo(o, p, y, phys_dim)

    # Compute metrics
    residual = T @ y - dv
    metrics = {
        "residual_norm": float(jnp.linalg.norm(residual)),
        "dv_norm": float(jnp.linalg.norm(dv)),
        "ordering": ordering,
    }

    return updates, metrics


# =============================================================================
# Verification and Demo Functions
# =============================================================================


def verify_smallo_correctness_mps(
    model: SimpleMPS,
    samples: jax.Array,
    diag_shift: float = 1e-8,
) -> dict:
    """Verify small-o trick produces same results as full O matrix for MPS.

    The verification computes OOdag in two ways:
    1. Full O: compute full Jacobian O (uncentered), then OOdag_full = O @ O†
    2. Small-o: compute reduced Jacobian o (uncentered), then OOdag via phys_index loop

    Since O[s,α]=0 when α's physical index doesn't match s(x), both should be equal.
    The centering is then done on the OOdag matrix via build_minsr_matrix.

    Args:
        model: SimpleMPS model.
        samples: Spin configurations (n_samples, n_sites) in {-1, +1}.
        diag_shift: Regularization parameter.

    Returns:
        Dict with comparison metrics.
    """
    n_samples = samples.shape[0]
    n_sites = model.n_sites
    bond_dim = model.bond_dim
    phys_dim = model.phys_dim

    _log_verify("Building full Jacobian O (uncentered)...")
    O_raw = build_dense_jac(model, samples, full_gradient=True)
    _log_verify("  O_raw shape: %s", O_raw.shape)

    # Build small-o (also uncentered)
    _log_verify("Building small-o Jacobian (uncentered)...")
    o, p = build_dense_jac(model, samples, full_gradient=False)
    _log_verify("  o shape: %s, p shape: %s", o.shape, p.shape)

    # Compute OOdag_full directly from uncentered O
    _log_verify("Computing OOdag from full O (uncentered)...")
    OOdag_full = O_raw @ O_raw.conj().T
    _log_verify("  OOdag_full norm: %.6e", float(jnp.linalg.norm(OOdag_full)))

    # Compute OOdag from small-o (also uncentered)
    _log_verify("Computing OOdag from small-o (physical ordering, uncentered)...")
    OOdag_smallo = build_OOdag_phys_ordering(o, p, phys_dim)
    _log_verify("  OOdag_smallo norm: %.6e", float(jnp.linalg.norm(OOdag_smallo)))

    # Compare OOdag (should match exactly since both uncentered)
    _log_verify("Comparing OOdag (should match since both uncentered)...")
    OOdag_diff = jnp.linalg.norm(OOdag_full - OOdag_smallo)
    OOdag_norm = jnp.linalg.norm(OOdag_full)
    OOdag_error = float(OOdag_diff / (OOdag_norm + 1e-30))
    _log_verify("  OOdag absolute diff: %.6e", float(OOdag_diff))
    _log_verify("  OOdag relative error: %.6e", OOdag_error)

    # Compare minSR matrices (centered via build_minsr_matrix)
    _log_verify("Comparing minSR matrices (centered)...")
    T_full = build_minsr_matrix(OOdag_full)
    T_smallo = build_minsr_matrix(OOdag_smallo)
    T_diff = jnp.linalg.norm(T_full - T_smallo)
    T_norm = jnp.linalg.norm(T_full)
    T_error = float(T_diff / (T_norm + 1e-30))
    _log_verify("  minSR matrix relative error: %.6e", T_error)

    # Compare solutions with a random dv
    _log_verify("Comparing minSR solutions...")
    key = jax.random.key(42)
    dv = jax.random.normal(key, (n_samples,), dtype=jnp.complex128)
    dv = dv / jnp.sqrt(n_samples)

    # Compute params_per_site for MPS
    params_per_site = []
    for site in range(n_sites):
        left_dim = 1 if site == 0 else bond_dim
        right_dim = 1 if site == n_sites - 1 else bond_dim
        params_per_site.append(left_dim * right_dim)

    # Full solve: use centered T
    T_full_reg = T_full + diag_shift * jnp.eye(n_samples, dtype=T_full.dtype)
    y_full = jsp.linalg.solve(T_full_reg, dv, assume_a="pos")
    # For update recovery with full O, we need centered O
    O_centered = O_raw - jnp.mean(O_raw, axis=0, keepdims=True)
    updates_full = O_centered.conj().T @ y_full

    # Small-o: solve and recover updates in full space
    T_smallo_reg = T_smallo + diag_shift * jnp.eye(n_samples, dtype=T_smallo.dtype)
    y_smallo = jsp.linalg.solve(T_smallo_reg, dv, assume_a="pos")
    updates_smallo = recover_updates_smallo(o, p, y_smallo, phys_dim, params_per_site)

    # Compare y vectors (should match since T matrices match)
    y_diff = jnp.linalg.norm(y_full - y_smallo)
    y_norm = jnp.linalg.norm(y_full)
    y_error = float(y_diff / (y_norm + 1e-30))
    _log_verify("  y vector relative error: %.6e", y_error)

    # Compare updates directly in full space
    updates_diff = jnp.linalg.norm(updates_full - updates_smallo)
    updates_norm = jnp.linalg.norm(updates_full)
    updates_error = float(updates_diff / (updates_norm + 1e-30))
    _log_verify("  Updates relative error: %.6e", updates_error)
    _log_verify("  Full updates norm: %.6e", float(jnp.linalg.norm(updates_full)))
    _log_verify("  Small-o updates norm: %.6e", float(jnp.linalg.norm(updates_smallo)))

    results = {
        "OOdag_error": OOdag_error,
        "T_error": T_error,
        "y_error": y_error,
        "updates_error": updates_error,
        "memory_full": O_raw.nbytes,
        "memory_smallo": o.nbytes + p.nbytes,
        "memory_ratio": O_raw.nbytes / (o.nbytes + p.nbytes),
    }
    _log_verify("  Memory ratio (full/small-o): %.2fx", results["memory_ratio"])

    return results


def minsr_smallo_mps_demo(
    length: int = 4,
    bond_dim: int = 2,
    n_samples: int = 512,
    diag_shift: float = 1e-8,
    seed: int = 0,
    use_sequential_sampling: bool = True,
):
    """Demo comparing standard minSR vs small-o minSR for MPS.

    Args:
        length: Number of sites.
        bond_dim: Bond dimension (D=2 default).
        n_samples: Number of samples.
        diag_shift: Regularization parameter.
        seed: Random seed.
        use_sequential_sampling: Use sequential sampling (reuses environments).
    """
    logger.info("=" * 60)
    logger.info("MinSR Small-o Demo (MPS)")
    logger.info("=" * 60)
    logger.info("Parameters: length=%d, bond_dim=%d, n_samples=%d", length, bond_dim, n_samples)

    # Create model
    model = SimpleMPS(rngs=nnx.Rngs(seed), n_sites=length, bond_dim=bond_dim)

    # Generate samples
    key = jax.random.key(seed + 1)
    if use_sequential_sampling:
        logger.info("Using sequential sampling (environment reuse)...")
        samples = sequential_sample(model, n_samples=n_samples, key=key)
    else:
        logger.info("Using random sampling...")
        bits = jax.random.bernoulli(key, 0.5, shape=(n_samples, length))
        samples = (2 * bits - 1).astype(jnp.int32)  # {-1, +1}

    # Run verification
    results = verify_smallo_correctness_mps(model, samples, diag_shift)

    logger.info("\n" + "=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    logger.info("OOdag relative error: %.6e", results["OOdag_error"])
    logger.info("minSR matrix relative error: %.6e", results["T_error"])
    logger.info("Memory savings: %.2fx", results["memory_ratio"])

    return results


def verify_smallo_correctness_peps(
    model: SimplePEPS,
    samples: jax.Array,
    diag_shift: float = 1e-8,
) -> dict:
    """Verify small-o trick produces same results as full O matrix for PEPS.

    Args:
        model: SimplePEPS model.
        samples: Spin configurations (n_samples, n_sites) in {-1, +1}.
        diag_shift: Regularization parameter.

    Returns:
        Dict with comparison metrics.
    """
    n_samples = samples.shape[0]
    shape = model.shape
    bond_dim = model.bond_dim
    phys_dim = 2

    _log_verify("Building full Jacobian O for PEPS (uncentered)...")
    O_raw = build_dense_jac(model, samples, full_gradient=True)
    _log_verify("  O_raw shape: %s", O_raw.shape)

    # Build small-o (also uncentered)
    _log_verify("Building small-o Jacobian for PEPS (uncentered)...")
    o, p = build_dense_jac(model, samples, full_gradient=False)
    _log_verify("  o shape: %s, p shape: %s", o.shape, p.shape)

    # Compute OOdag_full directly from uncentered O
    _log_verify("Computing OOdag from full O (uncentered)...")
    OOdag_full = O_raw @ O_raw.conj().T
    _log_verify("  OOdag_full norm: %.6e", float(jnp.linalg.norm(OOdag_full)))

    # Compute OOdag from small-o (also uncentered)
    _log_verify("Computing OOdag from small-o (physical ordering, uncentered)...")
    OOdag_smallo = build_OOdag_phys_ordering(o, p, phys_dim)
    _log_verify("  OOdag_smallo norm: %.6e", float(jnp.linalg.norm(OOdag_smallo)))

    # Compare OOdag
    _log_verify("Comparing OOdag (should match since both uncentered)...")
    OOdag_diff = jnp.linalg.norm(OOdag_full - OOdag_smallo)
    OOdag_norm = jnp.linalg.norm(OOdag_full)
    OOdag_error = float(OOdag_diff / (OOdag_norm + 1e-30))
    _log_verify("  OOdag absolute diff: %.6e", float(OOdag_diff))
    _log_verify("  OOdag relative error: %.6e", OOdag_error)

    # Compare minSR matrices (centered via build_minsr_matrix)
    _log_verify("Comparing minSR matrices (centered)...")
    T_full = build_minsr_matrix(OOdag_full)
    T_smallo = build_minsr_matrix(OOdag_smallo)
    T_diff = jnp.linalg.norm(T_full - T_smallo)
    T_norm = jnp.linalg.norm(T_full)
    T_error = float(T_diff / (T_norm + 1e-30))
    _log_verify("  minSR matrix relative error: %.6e", T_error)

    # Compare solutions with a random dv
    _log_verify("Comparing minSR solutions...")
    key = jax.random.key(42)
    dv = jax.random.normal(key, (n_samples,), dtype=jnp.complex128)
    dv = dv / jnp.sqrt(n_samples)

    # Compute params_per_site for PEPS
    params_per_site = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            up = 1 if r == 0 else bond_dim
            down = 1 if r == shape[0] - 1 else bond_dim
            left = 1 if c == 0 else bond_dim
            right = 1 if c == shape[1] - 1 else bond_dim
            params_per_site.append(up * down * left * right)

    # Full solve
    T_full_reg = T_full + diag_shift * jnp.eye(n_samples, dtype=T_full.dtype)
    y_full = jsp.linalg.solve(T_full_reg, dv, assume_a="pos")
    O_centered = O_raw - jnp.mean(O_raw, axis=0, keepdims=True)
    updates_full = O_centered.conj().T @ y_full

    # Small-o: solve and recover updates in full space
    T_smallo_reg = T_smallo + diag_shift * jnp.eye(n_samples, dtype=T_smallo.dtype)
    y_smallo = jsp.linalg.solve(T_smallo_reg, dv, assume_a="pos")
    updates_smallo = recover_updates_smallo(o, p, y_smallo, phys_dim, params_per_site)

    # Compare y vectors
    y_diff = jnp.linalg.norm(y_full - y_smallo)
    y_norm = jnp.linalg.norm(y_full)
    y_error = float(y_diff / (y_norm + 1e-30))
    _log_verify("  y vector relative error: %.6e", y_error)

    # Compare updates directly in full space
    updates_diff = jnp.linalg.norm(updates_full - updates_smallo)
    updates_norm = jnp.linalg.norm(updates_full)
    updates_error = float(updates_diff / (updates_norm + 1e-30))
    _log_verify("  Updates relative error: %.6e", updates_error)
    _log_verify("  Full updates norm: %.6e", float(jnp.linalg.norm(updates_full)))
    _log_verify("  Small-o updates norm: %.6e", float(jnp.linalg.norm(updates_smallo)))

    results = {
        "OOdag_error": OOdag_error,
        "T_error": T_error,
        "y_error": y_error,
        "updates_error": updates_error,
        "memory_full": O_raw.nbytes,
        "memory_smallo": o.nbytes + p.nbytes,
        "memory_ratio": O_raw.nbytes / (o.nbytes + p.nbytes),
    }
    _log_verify("  Memory ratio (full/small-o): %.2fx", results["memory_ratio"])

    return results


def minsr_smallo_peps_demo(
    length: int = 4,
    bond_dim: int = 2,
    n_samples: int = 128,
    diag_shift: float = 1e-8,
    truncate_bond_dimension: int = 4,
    seed: int = 0,
    use_sequential_sampling: bool = True,
    n_sweeps: int = 1,
    progress_interval: int = 1000,
):
    """Demo comparing standard minSR vs small-o minSR for PEPS.

    Args:
        length: Grid side length (length x length lattice).
        bond_dim: Bond dimension (D=2 default).
        n_samples: Number of samples.
        diag_shift: Regularization parameter.
        truncate_bond_dimension: Truncation dimension for ZipUp (D^2=4 default).
        seed: Random seed.
        use_sequential_sampling: Use sequential Metropolis-Hastings sampling
            (reuses environments).
        n_sweeps: Number of sequential MH sweeps per sample.
        progress_interval: Progress logging interval for sequential sampling.
    """
    logger.info("=" * 60)
    logger.info("MinSR Small-o Demo (PEPS)")
    logger.info("=" * 60)
    logger.info(
        "Parameters: length=%d, bond_dim=%d, truncate=%d, n_samples=%d",
        length, bond_dim, truncate_bond_dimension, n_samples
    )

    # Create model with ZipUp strategy
    model = SimplePEPS(
        rngs=nnx.Rngs(seed),
        shape=(length, length),
        bond_dim=bond_dim,
        contraction_strategy=ZipUp(truncate_bond_dimension),
    )

    # Generate samples
    key = jax.random.key(seed + 1)
    n_sites = length * length
    if use_sequential_sampling:
        logger.info(
            "Using sequential Metropolis-Hastings sampling (environment reuse, %d sweeps)...",
            n_sweeps,
        )
        samples = sequential_sample(
            model,
            n_samples=n_samples,
            key=key,
            n_sweeps=n_sweeps,
            progress_interval=progress_interval,
        )
    else:
        logger.info("Using random sampling...")
        bits = jax.random.bernoulli(key, 0.5, shape=(n_samples, n_sites))
        samples = (2 * bits - 1).astype(jnp.int32)  # {-1, +1}

    # Run verification against full O matrix
    logger.info("\n--- Verification against full O matrix ---")
    verify_results = verify_smallo_correctness_peps(model, samples, diag_shift)

    # Build small-o for ordering comparison
    logger.info("\n--- Comparing summation orderings ---")
    o, p = build_dense_jac(model, samples, full_gradient=False)
    logger.info("  o shape: %s, p shape: %s", o.shape, p.shape)

    # Build OOdag with different orderings
    logger.info("Building OOdag with physical ordering...")
    t0 = time.perf_counter()
    OOdag_phys = build_OOdag_phys_ordering(o, p, phys_dim=2)
    jax.block_until_ready(OOdag_phys)
    time_phys = time.perf_counter() - t0
    logger.info("  Physical ordering time: %.3fs", time_phys)

    # Compute params_per_site for site ordering
    params_per_site = []
    for r in range(length):
        for c in range(length):
            up = 1 if r == 0 else bond_dim
            down = 1 if r == length - 1 else bond_dim
            left = 1 if c == 0 else bond_dim
            right = 1 if c == length - 1 else bond_dim
            params_per_site.append(up * down * left * right)

    logger.info("Building OOdag with site ordering...")
    t0 = time.perf_counter()
    OOdag_site = build_OOdag_site_ordering(o, p, n_sites, params_per_site, phys_dim=2)
    jax.block_until_ready(OOdag_site)
    time_site = time.perf_counter() - t0
    logger.info("  Site ordering time: %.3fs", time_site)

    # Compare orderings
    ordering_diff = jnp.linalg.norm(OOdag_phys - OOdag_site) / jnp.linalg.norm(OOdag_phys)
    logger.info("  Ordering difference (should be ~0): %.6e", float(ordering_diff))

    # Solve minSR
    key2 = jax.random.key(seed + 2)
    dv = jax.random.normal(key2, (n_samples,), dtype=jnp.complex128)
    dv = dv / jnp.sqrt(n_samples)

    logger.info("Solving minSR with physical ordering...")
    updates_phys, metrics_phys = solve_minsr_smallo(o, p, dv, 2, diag_shift, ordering="physical")
    logger.info("  Residual norm: %.6e", metrics_phys["residual_norm"])

    logger.info("Solving minSR with site ordering...")
    updates_site, metrics_site = solve_minsr_smallo(
        o, p, dv, 2, diag_shift, ordering="site",
        n_sites=n_sites, params_per_site=params_per_site
    )
    logger.info("  Residual norm: %.6e", metrics_site["residual_norm"])

    # Compare updates
    updates_diff = jnp.linalg.norm(updates_phys - updates_site) / jnp.linalg.norm(updates_phys)
    logger.info("  Updates difference (should be ~0): %.6e", float(updates_diff))

    results = {
        "OOdag_error": verify_results["OOdag_error"],
        "T_error": verify_results["T_error"],
        "y_error": verify_results["y_error"],
        "updates_error": verify_results["updates_error"],
        "ordering_diff": float(ordering_diff),
        "updates_ordering_diff": float(updates_diff),
        "residual_phys": metrics_phys["residual_norm"],
        "residual_site": metrics_site["residual_norm"],
        "ordering_time_phys": time_phys,
        "ordering_time_site": time_site,
        "memory_full": verify_results["memory_full"],
        "memory_smallo": verify_results["memory_smallo"],
        "memory_ratio": verify_results["memory_ratio"],
    }

    logger.info("\n" + "=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    logger.info("OOdag relative error: %.6e", results["OOdag_error"])
    logger.info("minSR matrix relative error: %.6e", results["T_error"])
    logger.info("y vector relative error: %.6e", results["y_error"])
    logger.info("Updates relative error: %.6e", results["updates_error"])
    logger.info("Physical vs Site ordering difference: %.6e", results["ordering_diff"])
    logger.info("Memory savings: %.2fx", results["memory_ratio"])

    return results


def main():
    """Run all demos."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    logger.info("\n" + "=" * 70)
    logger.info("MinSR Small-o Trick Demonstration")
    logger.info("=" * 70)

    # MPS demo
    logger.info("\n### MPS Demo ###\n")
    minsr_smallo_mps_demo(length=8, bond_dim=2, n_samples=256)

    # PEPS demo
    logger.info("\n### PEPS Demo ###\n")
    minsr_smallo_peps_demo(length=3, bond_dim=2, n_samples=64, truncate_bond_dimension=4)


if __name__ == "__main__":
    main()
