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

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import netket as nk
from flax import nnx

from VMC.core import _value_and_grad_batch
from VMC.models.mps import SimpleMPS
from VMC.models.peps import SimplePEPS, ZipUp
from VMC.samplers.sequential import sequential_sample_with_gradients
from VMC.utils.smallo import params_per_site
from VMC.utils.vmc_utils import flatten_samples, get_apply_fun

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
def build_minsr_matrix(OOdag: jax.Array, q: jax.Array) -> jax.Array:
    """Compute the reduced minSR matrix with explicit Q projection.

    With Q spanning the subspace orthogonal to the null vector, the reduced
    system is:
        T = Q† OO† Q = (Q† O)(Q† O)†

    Args:
        OOdag: (n_samples, n_samples) raw OO† Gram matrix.
        q: (n_samples, n_samples - 1) basis for the null-free subspace.

    Returns:
        T: (n_samples - 1, n_samples - 1) projected minSR matrix.
    """
    return q.conj().T @ OOdag @ q


def _nullspace_basis(
    n_samples: int,
    dtype: jnp.dtype,
) -> jax.Array:
    """Build an orthonormal basis for the subspace orthogonal to the all-ones vector."""
    ones = jnp.ones((n_samples, 1), dtype=dtype)
    q_full, _ = jnp.linalg.qr(ones, mode="complete")
    return q_full[:, 1:]


@functools.partial(jax.jit, static_argnames=("phys_dim", "params_per_site"))
def _apply_O_to_updates(
    o: jax.Array,
    p: jax.Array,
    updates: jax.Array,
    phys_dim: int,
    params_per_site: tuple[int, ...] | None = None,
) -> jax.Array:
    """Apply the full O matrix (implicit via small-o) to a full update vector."""
    n_samples = o.shape[0]
    o_updates = jnp.zeros((n_samples,), dtype=o.dtype)

    if params_per_site is None:
        for k in range(phys_dim):
            updates_k = updates[k::phys_dim]
            mask = (p == k).astype(o.dtype)
            o_updates = o_updates + jnp.sum(o * mask * updates_k[None, :], axis=1)
        return o_updates

    n_sites = len(params_per_site)
    offset_reduced = 0
    offset_full = 0
    for site in range(n_sites):
        pps = params_per_site[site]
        o_site = o[:, offset_reduced:offset_reduced + pps]
        p_site = p[:, offset_reduced:offset_reduced + pps]

        for k in range(phys_dim):
            updates_k = updates[offset_full + k * pps: offset_full + (k + 1) * pps]
            mask = (p_site == k).astype(o.dtype)
            o_updates = o_updates + jnp.sum(
                o_site * mask * updates_k[None, :], axis=1
            )

        offset_reduced += pps
        offset_full += phys_dim * pps

    return o_updates


def apply_O_to_updates(
    o: jax.Array,
    p: jax.Array,
    updates: jax.Array,
    phys_dim: int,
    params_per_site: list[int] | tuple[int, ...] | None = None,
) -> jax.Array:
    """Wrapper ensuring params_per_site is static for JIT compilation."""
    if params_per_site is not None and not isinstance(params_per_site, tuple):
        params_per_site = tuple(params_per_site)
    return _apply_O_to_updates(o, p, updates, phys_dim, params_per_site)


@functools.partial(jax.jit, static_argnames=("phys_dim", "params_per_site"))
def _recover_updates_smallo(
    o: jax.Array,
    p: jax.Array,
    y: jax.Array,
    phys_dim: int,
    params_per_site: tuple[int, ...] | None = None,
) -> jax.Array:
    """Recover parameter updates in FULL parameter space from minSR solution using small-o.

    Computes updates = O† @ y with y projected to the null-free subspace,
    returning updates in the full parameter space (with physical index dimension).

    Args:
        o: (n_samples, n_params_reduced) UNCENTERED small-o Jacobian.
        p: (n_samples, n_params_reduced) int8 physical index tensor.
        y: (n_samples,) solution from minSR linear system (already projected).
        phys_dim: Physical dimension.
        params_per_site: Params per physical slice for each site.
            Required for proper layout mapping. If None, assumes uniform.

    Returns:
        updates: (n_params_full,) parameter updates in full space.
                 Layout: for each site, phys_dim consecutive blocks of params_per_phys.
    """
    n_params_reduced = o.shape[1]

    # If params_per_site not provided, assume uniform (won't work for open BC)
    if params_per_site is None:
        n_params_full = n_params_reduced * phys_dim
        updates = jnp.zeros(n_params_full, dtype=o.dtype)
        for k in range(phys_dim):
            mask = (p == k).astype(o.dtype)
            o_k = o * mask
            updates_k = o_k.conj().T @ y
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

            # Contribution for physical index k at this site
            updates_k = o_k.conj().T @ y

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
        2. Project to get minSR matrix T using Q† O
        3. Solve (T + diag_shift*I) @ y = Q† dv
        4. Recover updates = O† Q y via small-o

    Args:
        o: (n_samples, n_params_reduced) uncentered small-o Jacobian.
        p: (n_samples, n_params_reduced) int8 physical index tensor.
        dv: (n_samples,) gradient vector.
        phys_dim: Physical dimension.
        diag_shift: Regularization parameter.
        ordering: "physical" or "site" summation ordering.
        n_sites: Required if ordering="site".
        params_per_site: Required for full-parameter updates and site ordering.

    Returns:
        updates: (n_params_full,) parameter updates in full space.
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

    # Project to get minSR matrix
    n_samples = o.shape[0]
    q = _nullspace_basis(n_samples, OOdag.dtype)
    T = build_minsr_matrix(OOdag, q)

    # Regularize and solve
    dv_red = q.conj().T @ dv
    T_reg = T + diag_shift * jnp.eye(T.shape[0], dtype=T.dtype)
    y_red = jsp.linalg.solve(T_reg, dv_red, assume_a="pos")
    y = q @ y_red

    # Recover updates
    updates = recover_updates_smallo(o, p, y, phys_dim, params_per_site)

    # Compute SR-form residuals using the projected force E = O† Q Q† dv.
    dv_proj = q @ dv_red
    E_param = recover_updates_smallo(o, p, dv_proj, phys_dim, params_per_site)
    o_updates = apply_O_to_updates(o, p, updates, phys_dim, params_per_site)
    oo_updates = recover_updates_smallo(o, p, o_updates, phys_dim, params_per_site)
    residual = oo_updates - E_param
    residual_reg = residual + diag_shift * updates
    E_norm_sq = jnp.vdot(E_param, E_param).real
    residual_norm = jnp.vdot(residual, residual).real / E_norm_sq
    residual_reg_norm = jnp.vdot(residual_reg, residual_reg).real / E_norm_sq
    metrics = {
        "residual_sr": float(residual_norm),
        "residual_sr_reg": float(residual_reg_norm),
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
    *,
    o: jax.Array | None = None,
    p: jax.Array | None = None,
) -> dict:
    """Verify small-o trick produces same results as full O matrix for MPS.

    The verification computes OOdag in two ways:
    1. Full O: construct full Jacobian O from small-o (uncentered), then OOdag_full = O @ O†
    2. Small-o: compute reduced Jacobian o (uncentered), then OOdag via phys_index loop

    Since O[s,α]=0 when α's physical index doesn't match s(x), both should be equal.
    The projection is then applied to OOdag via build_minsr_matrix.

    Args:
        model: SimpleMPS model.
        samples: Spin configurations (n_samples, n_sites) in {-1, +1}.
        diag_shift: Regularization parameter.
        o: Optional precomputed small-o Jacobian (grads / amp).
        p: Optional precomputed physical index tensor.

    Returns:
        Dict with comparison metrics.
    """
    n_samples = samples.shape[0]
    phys_dim = model.phys_dim

    samples_flat = flatten_samples(samples)

    # Build small-o (also uncentered)
    if o is None or p is None:
        _log_verify("Building small-o Jacobian (uncentered)...")
        amps, grads, p = _value_and_grad_batch(model, samples_flat, full_gradient=False)
        o = grads / amps[:, None]
    else:
        _log_verify("Using precomputed small-o Jacobian.")
    _log_verify("  o shape: %s, p shape: %s", o.shape, p.shape)

    _log_verify("Constructing full Jacobian O from small-o (uncentered)...")
    pps = params_per_site(model)
    blocks = []
    offset_reduced = 0
    for site_pps in pps:
        o_site = o[:, offset_reduced:offset_reduced + site_pps]
        p_site = p[:, offset_reduced:offset_reduced + site_pps]
        one_hot = jax.nn.one_hot(p_site, phys_dim, dtype=o.dtype)
        block = one_hot * o_site[:, :, None]
        block = jnp.transpose(block, (0, 2, 1)).reshape(
            n_samples, phys_dim * site_pps
        )
        blocks.append(block)
        offset_reduced += site_pps
    O_raw = jnp.concatenate(blocks, axis=1)
    _log_verify("  O_raw shape: %s", O_raw.shape)

    # Compute OOdag_full directly from uncentered O
    _log_verify("Computing OOdag from constructed O (uncentered)...")
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

    # Compare minSR matrices (projected via build_minsr_matrix)
    _log_verify("Comparing minSR matrices (projected)...")
    q = _nullspace_basis(n_samples, OOdag_full.dtype)
    T_full = build_minsr_matrix(OOdag_full, q)
    T_smallo = build_minsr_matrix(OOdag_smallo, q)
    T_diff = jnp.linalg.norm(T_full - T_smallo)
    T_norm = jnp.linalg.norm(T_full)
    T_error = float(T_diff / (T_norm + 1e-30))
    _log_verify("  minSR matrix relative error: %.6e", T_error)

    # Verify null vector projection: Q† @ ones should be ~0
    _log_verify("Verifying null vector projection...")
    ones = jnp.ones((n_samples,), dtype=q.dtype)
    null_check = float(jnp.linalg.norm(q.conj().T @ ones))
    _log_verify("  ||Q† @ ones|| (should be ~0): %.6e", null_check)

    T_full_reg = T_full + diag_shift * jnp.eye(T_full.shape[0], dtype=T_full.dtype)
    T_smallo_reg = T_smallo + diag_shift * jnp.eye(
        T_smallo.shape[0], dtype=T_smallo.dtype
    )

    # Compare solutions with a random dv
    _log_verify("Comparing minSR solutions...")
    key = jax.random.key(42)
    dv = jax.random.normal(key, (n_samples,), dtype=jnp.complex128)
    dv = dv / jnp.sqrt(n_samples)
    dv_red = q.conj().T @ dv
    dv_proj = q @ dv_red

    # Full solve: use projected T
    y_full_red = jsp.linalg.solve(T_full_reg, dv_red, assume_a="pos")
    y_full = q @ y_full_red
    updates_full = O_raw.conj().T @ y_full

    # Small-o: solve and recover updates in full space
    y_smallo_red = jsp.linalg.solve(T_smallo_reg, dv_red, assume_a="pos")
    y_smallo = q @ y_smallo_red
    updates_smallo = recover_updates_smallo(o, p, y_smallo, phys_dim, pps)

    # SR-form residual using the projected force E = O† Q Q† dv
    E_param = O_raw.conj().T @ dv_proj
    o_updates = O_raw @ updates_smallo
    oo_updates = O_raw.conj().T @ o_updates
    residual = oo_updates - E_param
    residual_reg = residual + diag_shift * updates_smallo
    E_norm_sq = jnp.vdot(E_param, E_param).real
    residual_norm = jnp.vdot(residual, residual).real / E_norm_sq
    residual_reg_norm = jnp.vdot(residual_reg, residual_reg).real / E_norm_sq
    _log_verify("  SR residual ||O†O u - E||^2/||E||^2: %.6e", residual_norm)
    _log_verify(
        "  SR residual ||(O†O+λI) u - E||^2/||E||^2: %.6e", residual_reg_norm
    )

    # Compare updates
    updates_error = float(jnp.linalg.norm(updates_full - updates_smallo) / (jnp.linalg.norm(updates_full) + 1e-30))
    _log_verify("  Updates relative error: %.6e", updates_error)

    # Compare updates with realistic forces from local energies.
    _log_verify("Comparing minSR solutions with local energies...")
    graph = nk.graph.Chain(length=model.n_sites, pbc=False)
    hi = nk.hilbert.Spin(s=1 / 2, N=model.n_sites)
    hamiltonians = {
        "Heisenberg": nk.operator.Heisenberg(hi, graph, dtype=jnp.complex128),
        "TFIM": nk.operator.Ising(hi, graph, h=1.0, J=1.0, dtype=jnp.complex128),
    }
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=1)
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=0,
    )
    vstate._samples = samples.reshape(1, n_samples, model.n_sites)
    apply_fun, params, model_state, kwargs = get_apply_fun(vstate)
    flat = vstate._samples.reshape(-1, model.n_sites)
    logpsi = jax.vmap(
        lambda s: apply_fun({"params": params, **model_state}, s, **kwargs)
    )(flat)
    vstate._logpsi = logpsi.reshape(1, n_samples)

    realistic_errors: dict[str, float] = {}
    for name, hamiltonian in hamiltonians.items():
        local_energies = vstate.local_estimators(hamiltonian).reshape(-1)
        dv = (local_energies - jnp.mean(local_energies)) / jnp.sqrt(n_samples)
        dv_red = q.conj().T @ dv
        y_full_red = jsp.linalg.solve(T_full_reg, dv_red, assume_a="pos")
        y_full = q @ y_full_red
        updates_full = O_raw.conj().T @ y_full
        y_smallo_red = jsp.linalg.solve(T_smallo_reg, dv_red, assume_a="pos")
        y_smallo = q @ y_smallo_red
        updates_smallo = recover_updates_smallo(o, p, y_smallo, phys_dim, pps)
        err = float(
            jnp.linalg.norm(updates_full - updates_smallo)
            / (jnp.linalg.norm(updates_full) + 1e-30)
        )
        realistic_errors[name] = err
        _log_verify("  %s updates relative error: %.6e", name, err)

    results = {
        "OOdag_error": OOdag_error,
        "T_error": T_error,
        "null_check": null_check,
        "residual_sr": float(residual_norm),
        "residual_sr_reg": float(residual_reg_norm),
        "updates_error": updates_error,
        "updates_error_heisenberg": realistic_errors.get("Heisenberg"),
        "updates_error_tfim": realistic_errors.get("TFIM"),
        "memory_full": O_raw.nbytes,
        "memory_smallo": o.nbytes + p.nbytes,
        "memory_ratio": O_raw.nbytes / (o.nbytes + p.nbytes),
    }
    _log_verify("  Memory: full=%d bytes, small-o=%d bytes, ratio=%.2fx",
                results["memory_full"], results["memory_smallo"], results["memory_ratio"])

    return results


def minsr_smallo_mps_demo(
    length: int = 4,
    bond_dim: int = 2,
    n_samples: int = 512,
    diag_shift: float = 1e-8,
    seed: int = 0,
    progress_interval: int = 1000,
):
    """Demo comparing standard minSR vs small-o minSR for MPS.

    Args:
        length: Number of sites.
        bond_dim: Bond dimension (D=2 default).
        n_samples: Number of samples.
        diag_shift: Regularization parameter.
        seed: Random seed.
        progress_interval: Progress logging interval for sampling.
    """
    logger.info("=" * 60)
    logger.info("MinSR Small-o Demo (MPS)")
    logger.info("=" * 60)
    logger.info("Parameters: length=%d, bond_dim=%d, n_samples=%d", length, bond_dim, n_samples)

    model = SimpleMPS(rngs=nnx.Rngs(seed), n_sites=length, bond_dim=bond_dim)
    key = jax.random.key(seed + 1)

    logger.info("Using sequential sampling with gradients (combined)...")
    samples, o, p, _ = sequential_sample_with_gradients(
        model,
        n_samples=n_samples,
        key=key,
        progress_interval=progress_interval,
        full_gradient=False,
    )

    # Run verification with pre-computed small-o
    results = verify_smallo_correctness_mps(model, samples, diag_shift, o=o, p=p)

    logger.info("\n" + "=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    logger.info("OOdag relative error: %.6e", results["OOdag_error"])
    logger.info("minSR matrix relative error: %.6e", results["T_error"])
    logger.info("Null vector check: %.6e", results["null_check"])
    logger.info("SR residual ||O†O u - E||^2/||E||^2: %.6e", results["residual_sr"])
    logger.info(
        "SR residual ||(O†O+λI) u - E||^2/||E||^2: %.6e",
        results["residual_sr_reg"],
    )
    if results.get("updates_error_tfim") is not None:
        logger.info(
            "TFIM updates relative error: %.6e", results["updates_error_tfim"]
        )
    if results.get("updates_error_heisenberg") is not None:
        logger.info(
            "Heisenberg updates relative error: %.6e",
            results["updates_error_heisenberg"],
        )
    logger.info("Memory savings: %.2fx", results["memory_ratio"])

    return results


def verify_smallo_correctness_peps(
    model: SimplePEPS,
    samples: jax.Array,
    diag_shift: float = 1e-8,
    *,
    o: jax.Array | None = None,
    p: jax.Array | None = None,
) -> dict:
    """Verify small-o trick produces same results as full O matrix for PEPS.

    Args:
        model: SimplePEPS model.
        samples: Spin configurations (n_samples, n_sites) in {-1, +1}.
        diag_shift: Regularization parameter.
        o: Optional precomputed small-o Jacobian (grads / amp).
        p: Optional precomputed physical index tensor.

    Returns:
        Dict with comparison metrics.
    """
    n_samples = samples.shape[0]
    phys_dim = 2

    samples_flat = flatten_samples(samples)

    # Build small-o (also uncentered)
    if o is None or p is None:
        _log_verify("Building small-o Jacobian for PEPS (uncentered)...")
        amps, grads, p = _value_and_grad_batch(model, samples_flat, full_gradient=False)
        o = grads / amps[:, None]
    else:
        _log_verify("Using precomputed small-o Jacobian.")
    _log_verify("  o shape: %s, p shape: %s", o.shape, p.shape)

    _log_verify("Constructing full Jacobian O from small-o (uncentered)...")
    pps = params_per_site(model)
    blocks = []
    offset_reduced = 0
    for site_pps in pps:
        o_site = o[:, offset_reduced:offset_reduced + site_pps]
        p_site = p[:, offset_reduced:offset_reduced + site_pps]
        one_hot = jax.nn.one_hot(p_site, phys_dim, dtype=o.dtype)
        block = one_hot * o_site[:, :, None]
        block = jnp.transpose(block, (0, 2, 1)).reshape(
            n_samples, phys_dim * site_pps
        )
        blocks.append(block)
        offset_reduced += site_pps
    O_raw = jnp.concatenate(blocks, axis=1)
    _log_verify("  O_raw shape: %s", O_raw.shape)

    # Compute OOdag_full directly from uncentered O
    _log_verify("Computing OOdag from constructed O (uncentered)...")
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

    # Compare minSR matrices (projected via build_minsr_matrix)
    _log_verify("Comparing minSR matrices (projected)...")
    q = _nullspace_basis(n_samples, OOdag_full.dtype)
    T_full = build_minsr_matrix(OOdag_full, q)
    T_smallo = build_minsr_matrix(OOdag_smallo, q)
    T_diff = jnp.linalg.norm(T_full - T_smallo)
    T_norm = jnp.linalg.norm(T_full)
    T_error = float(T_diff / (T_norm + 1e-30))
    _log_verify("  minSR matrix relative error: %.6e", T_error)

    # Verify null vector projection: Q† @ ones should be ~0
    _log_verify("Verifying null vector projection...")
    ones = jnp.ones((n_samples,), dtype=q.dtype)
    null_check = float(jnp.linalg.norm(q.conj().T @ ones))
    _log_verify("  ||Q† @ ones|| (should be ~0): %.6e", null_check)

    T_full_reg = T_full + diag_shift * jnp.eye(T_full.shape[0], dtype=T_full.dtype)
    T_smallo_reg = T_smallo + diag_shift * jnp.eye(
        T_smallo.shape[0], dtype=T_smallo.dtype
    )

    # Compare solutions with a random dv
    _log_verify("Comparing minSR solutions...")
    key = jax.random.key(42)
    dv = jax.random.normal(key, (n_samples,), dtype=jnp.complex128)
    dv = dv / jnp.sqrt(n_samples)
    dv_red = q.conj().T @ dv
    dv_proj = q @ dv_red

    # Full solve
    y_full_red = jsp.linalg.solve(T_full_reg, dv_red, assume_a="pos")
    y_full = q @ y_full_red
    updates_full = O_raw.conj().T @ y_full

    # Small-o: solve and recover updates in full space
    y_smallo_red = jsp.linalg.solve(T_smallo_reg, dv_red, assume_a="pos")
    y_smallo = q @ y_smallo_red
    updates_smallo = recover_updates_smallo(o, p, y_smallo, phys_dim, pps)

    # SR-form residual using the projected force E = O† Q Q† dv
    E_param = O_raw.conj().T @ dv_proj
    o_updates = O_raw @ updates_smallo
    oo_updates = O_raw.conj().T @ o_updates
    residual = oo_updates - E_param
    residual_reg = residual + diag_shift * updates_smallo
    E_norm_sq = jnp.vdot(E_param, E_param).real
    residual_norm = jnp.vdot(residual, residual).real / E_norm_sq
    residual_reg_norm = jnp.vdot(residual_reg, residual_reg).real / E_norm_sq
    _log_verify("  SR residual ||O†O u - E||^2/||E||^2: %.6e", residual_norm)
    _log_verify(
        "  SR residual ||(O†O+λI) u - E||^2/||E||^2: %.6e", residual_reg_norm
    )

    # Compare updates
    updates_error = float(jnp.linalg.norm(updates_full - updates_smallo) / (jnp.linalg.norm(updates_full) + 1e-30))
    _log_verify("  Updates relative error: %.6e", updates_error)

    # Compare updates with realistic forces from local energies.
    _log_verify("Comparing minSR solutions with local energies...")
    n_rows, n_cols = model.shape
    graph = nk.graph.Square(length=n_rows, pbc=False)
    hi = nk.hilbert.Spin(s=1 / 2, N=n_rows * n_cols)
    hamiltonians = {
        "Heisenberg": nk.operator.Heisenberg(hi, graph, dtype=jnp.complex128),
        "TFIM": nk.operator.Ising(hi, graph, h=1.0, J=1.0, dtype=jnp.complex128),
    }
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=1)
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=0,
    )
    vstate._samples = samples.reshape(1, n_samples, n_rows * n_cols)
    apply_fun, params, model_state, kwargs = get_apply_fun(vstate)
    flat = vstate._samples.reshape(-1, n_rows * n_cols)
    logpsi = jax.vmap(
        lambda s: apply_fun({"params": params, **model_state}, s, **kwargs)
    )(flat)
    vstate._logpsi = logpsi.reshape(1, n_samples)

    realistic_errors: dict[str, float] = {}
    for name, hamiltonian in hamiltonians.items():
        local_energies = vstate.local_estimators(hamiltonian).reshape(-1)
        dv = (local_energies - jnp.mean(local_energies)) / jnp.sqrt(n_samples)
        dv_red = q.conj().T @ dv
        y_full_red = jsp.linalg.solve(T_full_reg, dv_red, assume_a="pos")
        y_full = q @ y_full_red
        updates_full = O_raw.conj().T @ y_full
        y_smallo_red = jsp.linalg.solve(T_smallo_reg, dv_red, assume_a="pos")
        y_smallo = q @ y_smallo_red
        updates_smallo = recover_updates_smallo(o, p, y_smallo, phys_dim, pps)
        err = float(
            jnp.linalg.norm(updates_full - updates_smallo)
            / (jnp.linalg.norm(updates_full) + 1e-30)
        )
        realistic_errors[name] = err
        _log_verify("  %s updates relative error: %.6e", name, err)

    results = {
        "OOdag_error": OOdag_error,
        "T_error": T_error,
        "null_check": null_check,
        "residual_sr": float(residual_norm),
        "residual_sr_reg": float(residual_reg_norm),
        "updates_error": updates_error,
        "updates_error_heisenberg": realistic_errors.get("Heisenberg"),
        "updates_error_tfim": realistic_errors.get("TFIM"),
        "memory_full": O_raw.nbytes,
        "memory_smallo": o.nbytes + p.nbytes,
        "memory_ratio": O_raw.nbytes / (o.nbytes + p.nbytes),
    }
    _log_verify("  Memory: full=%d bytes, small-o=%d bytes, ratio=%.2fx",
                results["memory_full"], results["memory_smallo"], results["memory_ratio"])

    return results


def minsr_smallo_peps_demo(
    length: int = 4,
    bond_dim: int = 2,
    n_samples: int = 128,
    diag_shift: float = 1e-8,
    truncate_bond_dimension: int = 4,
    seed: int = 0,
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

    model = SimplePEPS(
        rngs=nnx.Rngs(seed),
        shape=(length, length),
        bond_dim=bond_dim,
        contraction_strategy=ZipUp(truncate_bond_dimension),
    )

    key = jax.random.key(seed + 1)
    n_sites = length * length

    logger.info(
        "Using sequential sampling with gradients (combined, %d sweeps)...",
        n_sweeps,
    )
    samples, o, p, _ = sequential_sample_with_gradients(
        model,
        n_samples=n_samples,
        key=key,
        n_sweeps=n_sweeps,
        progress_interval=progress_interval,
        full_gradient=False,
    )

    logger.info("  o shape: %s, p shape: %s", o.shape, p.shape)

    # Run verification against full O matrix
    logger.info("\n--- Verification against full O matrix ---")
    verify_results = verify_smallo_correctness_peps(model, samples, diag_shift, o=o, p=p)

    # Build OOdag with different orderings
    logger.info("Building OOdag with physical ordering...")
    t0 = time.perf_counter()
    OOdag_phys = build_OOdag_phys_ordering(o, p, phys_dim=2)
    jax.block_until_ready(OOdag_phys)
    time_phys = time.perf_counter() - t0
    logger.info("  Physical ordering time: %.3fs", time_phys)

    pps = params_per_site(model)

    logger.info("Building OOdag with site ordering...")
    t0 = time.perf_counter()
    OOdag_site = build_OOdag_site_ordering(o, p, n_sites, pps, phys_dim=2)
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
    updates_phys, metrics_phys = solve_minsr_smallo(
        o, p, dv, 2, diag_shift, ordering="physical", params_per_site=pps
    )
    logger.info(
        "  SR residual ||(O†O+λI) u - E||^2/||E||^2: %.6e",
        metrics_phys["residual_sr_reg"],
    )

    logger.info("Solving minSR with site ordering...")
    updates_site, metrics_site = solve_minsr_smallo(
        o, p, dv, 2, diag_shift, ordering="site",
        n_sites=n_sites, params_per_site=pps
    )
    logger.info(
        "  SR residual ||(O†O+λI) u - E||^2/||E||^2: %.6e",
        metrics_site["residual_sr_reg"],
    )

    # Compare updates
    updates_diff = jnp.linalg.norm(updates_phys - updates_site) / jnp.linalg.norm(updates_phys)
    logger.info("  Updates difference (should be ~0): %.6e", float(updates_diff))

    results = {
        "OOdag_error": verify_results["OOdag_error"],
        "T_error": verify_results["T_error"],
        "residual_sr": verify_results["residual_sr"],
        "residual_sr_reg": verify_results["residual_sr_reg"],
        "updates_error": verify_results["updates_error"],
        "updates_error_heisenberg": verify_results.get("updates_error_heisenberg"),
        "updates_error_tfim": verify_results.get("updates_error_tfim"),
        "ordering_diff": float(ordering_diff),
        "updates_ordering_diff": float(updates_diff),
        "residual_phys": metrics_phys["residual_sr_reg"],
        "residual_site": metrics_site["residual_sr_reg"],
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
    logger.info("SR residual ||O†O u - E||^2/||E||^2: %.6e", results["residual_sr"])
    logger.info(
        "SR residual ||(O†O+λI) u - E||^2/||E||^2: %.6e",
        results["residual_sr_reg"],
    )
    if results.get("updates_error_tfim") is not None:
        logger.info(
            "TFIM updates relative error: %.6e", results["updates_error_tfim"]
        )
    if results.get("updates_error_heisenberg") is not None:
        logger.info(
            "Heisenberg updates relative error: %.6e",
            results["updates_error_heisenberg"],
        )
    logger.info("Updates relative error: %.6e", results["updates_error"])
    logger.info("Memory savings: %.2fx", results["memory_ratio"])

    return results


def main():
    """Run all demos."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    logger.info("\n" + "=" * 70)
    logger.info("MinSR Small-o Trick Demonstration")
    logger.info("=" * 70)

    # MPS demo with much larger parameters for better verification
    logger.info("\n### MPS Demo ###\n")
    minsr_smallo_mps_demo(length=32, bond_dim=6, n_samples=2048)

    # PEPS demo with much larger parameters (truncate_bond_dimension = bond_dim^2)
    logger.info("\n### PEPS Demo ###\n")
    minsr_smallo_peps_demo(length=5, bond_dim=4, n_samples=512, truncate_bond_dimension=16)


if __name__ == "__main__":
    main()
