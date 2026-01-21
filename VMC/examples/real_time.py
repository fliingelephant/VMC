from __future__ import annotations

import functools
import logging

from VMC import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import netket as nk
from flax import nnx

from VMC.drivers import DynamicsDriver, RealTimeUnit, TimeUnit
from VMC.models.mps import MPS
from VMC.models.peps import PEPS
from VMC.preconditioners import SRPreconditioner
from VMC.samplers.sequential import sequential_sample_with_gradients
from VMC.utils.utils import occupancy_to_spin, spin_to_occupancy
from VMC.utils.vmc_utils import model_params

logger = logging.getLogger(__name__)


def build_heisenberg_square(length: int, *, pbc: bool = False):
    graph = nk.graph.Square(length=length, pbc=pbc)
    hi = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes)
    H = nk.operator.Heisenberg(hi, graph, dtype=jnp.complex128)
    return hi, H, (length, length)


def build_total_sz(hi):
    ops = [nk.operator.spin.sigmaz(hi, i) for i in range(hi.size)]
    return sum(ops)


def normalized_wavefunction(model, basis: jax.Array, baseline_eps: float = 0.0) -> jax.Array:
    """Return the variational wavefunction psi(basis) normalized to 1."""
    logpsi = jax.vmap(model)(basis)
    psi = jnp.exp(logpsi.reshape(-1))
    if baseline_eps != 0.0:
        psi = psi + baseline_eps
    norm = jnp.linalg.norm(psi)
    return psi / norm


def dense_exact_dynamics(psi0, hi, H, dt, n_steps):
    """Exact dense propagation for small Hilbert spaces."""
    H_dense = jnp.asarray(H.to_dense())
    U_dt = jsp.linalg.expm(-1j * dt * H_dense)
    norm0 = jnp.conj(psi0) @ psi0
    energy0 = jnp.conj(psi0) @ (H_dense @ psi0) / norm0
    logger.info("Exact Energy at t=0: %s", complex(energy0))

    def step(_, state):
        state = U_dt @ state
        return state / jnp.linalg.norm(state)

    psi = jax.lax.fori_loop(0, n_steps, step, psi0)
    norm = jnp.conj(psi) @ psi
    energy = jnp.conj(psi) @ (H_dense @ psi) / norm
    sz_dense = jnp.asarray(build_total_sz(hi).to_dense())
    total_sz = jnp.conj(psi) @ (sz_dense @ psi) / norm
    logger.info("Exact Energy at t=%.6f: %s", n_steps * dt, complex(energy))

    return psi, energy, total_sz


def random_bitstring(key, n_sites: int):
    bits = jax.random.bernoulli(key, p=0.5, shape=(n_sites,))
    return occupancy_to_spin(bits)


def reset_product_state_mps(model: MPS, spins: jnp.ndarray):
    eps = 1e-3
    for site, t in enumerate(model.tensors):
        phys_idx = 1 if spins[site] == 1 else 0
        arr = jnp.ones_like(t[...]) * eps
        arr = arr.at[phys_idx, 0, 0].set(1.0)
        t.value = arr


def reset_product_state_peps(model: PEPS, spins: jnp.ndarray):
    new_rows = []
    idx = 0
    for row in model.tensors:
        new_row = []
        for t in row:
            eps = 1e-3
            arr = jnp.ones_like(t[...]) * eps
            phys_idx = 1 if spins[idx] == 1 else 0
            arr = arr.at[phys_idx, 0, 0, 0, 0].set(1.0)
            idx += 1
            t.value = arr
            new_row.append(t)
        new_rows.append(new_row)
    model.tensors = new_rows


def align_phase(reference: jax.Array, target: jax.Array):
    """Remove global phase between two state vectors."""
    overlap = jnp.conj(reference) @ target
    phase = overlap / (jnp.abs(overlap) + 1e-12)
    return target * jnp.conj(phase), overlap


def _bits_from_state(state_row: jax.Array) -> str:
    bits = spin_to_occupancy(state_row)
    return "".join(str(int(b)) for b in bits)


def _fmt_c(z: jax.Array) -> str:
    return f"{float(jnp.real(z)):+.3e}{float(jnp.imag(z)):+.3e}j"


def _log_wavefunction_table(
    title: str, basis: jax.Array, psi_ref: jax.Array, psi_target: jax.Array
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    lines = [
        title,
        f"{'state':>6} {'psi_ref':>24} {'psi_target':>24} "
        f"{'delta':>24} {'|delta|':>10}",
    ]
    for st, pref, ptarg in zip(basis, psi_ref, psi_target):
        delta = ptarg - pref
        lines.append(
            f"{_bits_from_state(st):>6} "
            f"{_fmt_c(pref):>24} {_fmt_c(ptarg):>24} {_fmt_c(delta):>24} "
            f"{float(jnp.abs(delta)) :>10.3e}"
        )
    logger.debug("\n".join(lines))


def run_case(
    name: str,
    model,
    hi,
    H,
    *,
    T: float = 0.01,
    n_steps: int = 1,
    time_unit: TimeUnit = RealTimeUnit(),
    n_samples: int = 256,
    diag_shift: float = 1e-2,
    return_states: bool = False,
    spins: jax.Array | None = None,
    print_table: bool = True,
    use_full_sum: bool = False,
):
    dt = T / n_steps
    if spins is None:
        spins = random_bitstring(jax.random.key(0), hi.size)
    if isinstance(model, MPS):
        reset_product_state_mps(model, spins)
    elif isinstance(model, PEPS):
        reset_product_state_peps(model, spins)

    basis = jnp.asarray(hi.all_states(), dtype=jnp.int32)
    psi0 = normalized_wavefunction(model, basis, baseline_eps=0.0)
    H_dense = jnp.asarray(H.to_dense())
    energy_t0_dense = jnp.conj(psi0) @ (H_dense @ psi0)
    psi_exact, energy_exact, sz_exact = dense_exact_dynamics(psi0, hi, H, dt, n_steps)

    fullsum_state = nk.vqs.FullSumState(hi, model)
    fullsum_state.parameters = model_params(model)
    psi_fullsum = fullsum_state.to_array()
    psi_fullsum_aligned, overlap_fullsum = align_phase(psi0, psi_fullsum)
    fullsum_fidelity = float(jnp.abs(overlap_fullsum) ** 2)
    fullsum_max = float(jnp.max(jnp.abs(psi_fullsum_aligned - psi0)))

    if print_table:
        _log_wavefunction_table(
            f"{name} t=0 amplitude table (psi_init vs dense)",
            basis,
            psi_fullsum,
            psi0,
        )

    sampler = functools.partial(
        sequential_sample_with_gradients,
        n_samples=n_samples,
        burn_in=0,
        full_gradient=False,
    )
    preconditioner = SRPreconditioner(diag_shift=diag_shift)
    driver = DynamicsDriver(
        model,
        H,
        sampler=sampler,
        preconditioner=preconditioner,
        dt=dt,
        time_unit=time_unit,
        sampler_key=jax.random.key(0),
    )

    if use_full_sum:
        energy_t0_fullsum = fullsum_state.expect(H).mean
        logger.info("TN Energy (FullSum) at t=0: %s", complex(energy_t0_fullsum))

    logger.info("\n%s", "=" * 60)
    logger.info(name)
    logger.info("=" * 60)
    logger.info("%s |<psi0|psi_fullsum>|^2: %.6f", name, fullsum_fidelity)
    logger.info("%s psi0 vs fullsum max|Δ|: %.3e", name, fullsum_max)
    logger.info("TN Energy at t=0 (dense): %s", complex(energy_t0_dense))

    driver.run(T, show_progress=True)
    stats = driver.energy
    energy_t_final = stats.mean if stats is not None else jnp.nan

    fullsum_state.parameters = model_params(model)
    energy_var = fullsum_state.expect(H).mean
    sz_op = build_total_sz(hi)
    sz_var = fullsum_state.expect(sz_op).mean

    logger.info("TN Energy at t=%.6f: %s", T, complex(energy_t_final))
    logger.info("TN Energy (FullSum) at t=%.6f: %s", T, complex(energy_var))

    psi_var = normalized_wavefunction(model, basis, baseline_eps=0.0)
    psi_var_aligned, overlap = align_phase(psi_exact, psi_var)
    amp_l2 = jnp.linalg.norm(psi_var_aligned - psi_exact)
    amp_max = jnp.max(jnp.abs(psi_var_aligned - psi_exact))
    fidelity = float(jnp.abs(overlap) ** 2)
    norm_exact = float(jnp.linalg.norm(psi_exact))
    norm_var = float(jnp.linalg.norm(psi_var_aligned))
    norm_init = float(jnp.linalg.norm(psi0))

    logger.info("%s |<psi_exact|psi_var>|^2: %.6f", name, fidelity)
    logger.info("%s max|Δ|: %.3e", name, float(amp_max))
    logger.info("%s ||psi_exact||: %.6f", name, norm_exact)
    logger.info("%s ||psi_var||: %.6f", name, norm_var)
    logger.info("%s ||psi_init||: %.6f", name, norm_init)
    logger.info("%s amp L2 error: %.6e", name, float(amp_l2))
    logger.info("%s total_sz exact: %s", name, complex(sz_exact))
    logger.info("%s total_sz var: %s", name, complex(sz_var))

    if return_states:
        return {
            "psi_exact": psi_exact,
            "psi_var": psi_var_aligned,
            "energy_exact": energy_exact,
            "sz_exact": sz_exact,
        }
    return None
