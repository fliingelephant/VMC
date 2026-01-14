from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

from VMC import config  # noqa: F401 - JAX config must be imported first

import logging

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import netket as nk
import netket.experimental as nkx
import netket.experimental.dynamics as dyn
from flax import nnx

from VMC.drivers.custom_driver import (
    CustomTDVP_SR,
    ImaginaryTime,
    PropagationType,
    RealTime,
)
from VMC.models.mps import MPS
from VMC.models.peps import PEPS
from VMC.preconditioners import solve_cholesky
from VMC.utils.vmc_utils import get_apply_fun
from VMC.utils.utils import occupancy_to_spin


logger = logging.getLogger(__name__)

def build_heisenberg_square(length: int, *, pbc: bool = False):
    graph = nk.graph.Square(length=length, pbc=pbc)
    hi = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes)
    H = nk.operator.Heisenberg(hi, graph, dtype=jnp.complex128)
    return hi, H, (length, length)


def build_total_sz(hi):
    ops = [nk.operator.spin.sigmaz(hi, i) for i in range(hi.size)]
    return sum(ops)


def normalized_wavefunction(vstate, basis: jax.Array, baseline_eps: float = 0.0) -> jax.Array:
    """Return the variational wavefunction psi(basis) normalized to 1.

    Args:
        vstate: Variational state.
        basis: Hilbert space basis states.
        baseline_eps: Uniform floor added before normalization.

    Returns:
        Normalized wavefunction amplitudes.
    """
    apply_fun, params, model_state, kwargs = get_apply_fun(vstate)
    logpsi = jax.vmap(
        lambda s: apply_fun({"params": params, **model_state}, s, **kwargs)
    )(basis)
    psi = jnp.exp(logpsi.reshape(-1))
    if baseline_eps != 0.0:
        psi = psi + baseline_eps
    norm = jnp.linalg.norm(psi)
    return psi / norm


def dense_exact_dynamics(psi0, hi, H, dt, n_steps):
    """
    Exact dense propagation with small steps: psi_{k+1} = exp(-i dt H) psi_k.
    Only feasible for small Hilbert spaces (here 2x2 spins -> dim 16).
    """
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
    bits = jax.random.bernoulli(key, p=0.5, shape=(n_sites,)).astype(jnp.int32)
    spins = occupancy_to_spin(bits)
    return spins


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
        new_rows.append(nnx.List(new_row))
    model.tensors = nnx.List(new_rows)


def align_phase(reference: jax.Array, target: jax.Array):
    """Remove global phase between two state vectors."""
    overlap = jnp.conj(reference) @ target
    phase = overlap / (jnp.abs(overlap) + 1e-12)
    return target * jnp.conj(phase), overlap


def _bits_from_state(state_row: jax.Array) -> str:
    """Map spins {-1, +1} to bitstring '0/1'."""
    bits = ((state_row + 1) // 2).astype(int)
    return "".join(str(int(b)) for b in bits)


def _fmt_c(z: jax.Array) -> str:
    """Format complex numbers with fixed width real/imag parts."""
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


class TDVPBackend(abc.ABC):
    """Abstract backend for building TDVP drivers."""

    @abc.abstractmethod
    def build_driver(
        self,
        H,
        vstate,
        dt: float,
        *,
        propagation: PropagationType,
        diag_shift: float,
        use_ntk: bool,
        solver,
    ) -> Any:
        """Build and return a TDVP driver instance."""


@dataclass(frozen=True)
class CustomTDVPBackend(TDVPBackend):
    """Backend wrapping the custom TDVP implementation."""

    def build_driver(
        self,
        H,
        vstate,
        dt: float,
        *,
        propagation: PropagationType,
        diag_shift: float,
        use_ntk: bool,
        solver,
    ) -> CustomTDVP_SR:
        return CustomTDVP_SR(
            H,
            vstate,
            dt=dt,
            propagation=propagation,
            diag_shift=diag_shift,
            use_ntk=use_ntk,
            solver=solver,
        )


@dataclass(frozen=True)
class NetketTDVPBackend(TDVPBackend):
    """Backend wrapping NetKet's TDVP implementation."""

    def build_driver(
        self,
        H,
        vstate,
        dt: float,
        *,
        propagation: PropagationType,
        diag_shift: float,
        use_ntk: bool,
        solver,
    ) -> nkx.TDVP:
        if use_ntk:
            raise ValueError("NetKet TDVP uses standard SR; set use_ntk=False.")
        propagation_type = _propagation_to_netket(propagation)
        return nkx.TDVP(
            H,
            vstate,
            dyn.RK4(dt=dt),
            propagation_type=propagation_type,
            qgt=nk.optimizer.qgt.QGTJacobianDense(
                diag_shift=diag_shift,
                holomorphic=True,
            ),
            linear_solver=solver,
        )


_CUSTOM_BACKEND = CustomTDVPBackend()
_NETKET_BACKEND = NetketTDVPBackend()
_REAL_TIME = RealTime()
_IMAGINARY_TIME = ImaginaryTime()


def _normalize_backend(backend: str | TDVPBackend) -> TDVPBackend:
    if isinstance(backend, TDVPBackend):
        return backend
    if isinstance(backend, str):
        name = backend.lower()
        if name == "custom":
            return _CUSTOM_BACKEND
        if name == "netket":
            return _NETKET_BACKEND
    raise TypeError(f"Unknown TDVP backend: {backend!r}")


def _normalize_propagation(propagation: str | PropagationType) -> PropagationType:
    if isinstance(propagation, PropagationType):
        return propagation
    if isinstance(propagation, str):
        name = propagation.lower()
        if name == "real":
            return _REAL_TIME
        if name == "imag":
            return _IMAGINARY_TIME
    raise TypeError(f"Unknown propagation: {propagation!r}")


def _propagation_to_netket(propagation: PropagationType) -> str:
    if isinstance(propagation, RealTime):
        return "real"
    if isinstance(propagation, ImaginaryTime):
        return "imag"
    raise TypeError(f"Unsupported propagation type: {type(propagation)}")


def _build_tdvp_driver(
    backend: TDVPBackend,
    H,
    vstate,
    dt: float,
    *,
    propagation: PropagationType,
    diag_shift: float,
    use_ntk: bool,
    solver,
):
    return backend.build_driver(
        H,
        vstate,
        dt,
        propagation=propagation,
        diag_shift=diag_shift,
        use_ntk=use_ntk,
        solver=solver,
    )


def run_case(
    name: str,
    model,
    hi,
    H,
    sampler=None,
    T=0.01,
    n_steps=1,
    driver_backend: str | TDVPBackend = "custom",
    propagation_type: str | PropagationType = "real",
    use_ntk: bool | None = None,
    solver=None,
    diag_shift=1e-2,
    n_samples=256,
    return_states: bool = False,
    spins: jax.Array | None = None,
    print_table: bool = True,
    use_full_sum: bool = False,
):
    """
    if spins is None:
        spins = random_bitstring(jax.random.key(0), hi.size)
    if isinstance(model, MPS):
        reset_product_state_mps(model, spins)
    elif isinstance(model, PEPS):
        reset_product_state_peps(model, spins)
    """
    dt = T / n_steps
    backend = _normalize_backend(driver_backend)
    propagation = _normalize_propagation(propagation_type)
    if use_ntk is None:
        use_ntk = isinstance(backend, CustomTDVPBackend)
    if solver is None:
        solver = (
            solve_cholesky
            if isinstance(backend, CustomTDVPBackend)
            else nk.optimizer.solver.cholesky
        )

    if use_full_sum:
        vstate = nk.vqs.FullSumState(hi, model)
    else:
        if sampler is None:
            raise ValueError("sampler must be provided when use_full_sum=False.")
        vstate = nk.vqs.MCState(
            sampler, model, n_samples=n_samples, n_discard_per_chain=10
        )
    basis = jnp.asarray(hi.all_states(), dtype=jnp.int32)  # full sampling basis
    psi0 = normalized_wavefunction(vstate, basis, baseline_eps=0.0)
    if use_full_sum:
        psi_fullsum = vstate.to_array()
    else:
        psi_fullsum = nk.vqs.FullSumState(hi, model).to_array()
    psi_fullsum_aligned, overlap_fullsum = align_phase(psi0, psi_fullsum)
    fullsum_fidelity = float(jnp.abs(overlap_fullsum) ** 2)
    fullsum_max = float(jnp.max(jnp.abs(psi_fullsum_aligned - psi0)))
    psi_exact, energy_exact, sz_exact = dense_exact_dynamics(psi0, hi, H, dt, n_steps)

    tdvp = _build_tdvp_driver(
        backend,
        H,
        vstate,
        dt,
        propagation=propagation,
        diag_shift=diag_shift,
        use_ntk=use_ntk,
        solver=solver,
    )

    logger.info("\n%s", "=" * 60)
    logger.info(name)
    logger.info("=" * 60)
    logger.info("%s |<psi0|psi_fullsum>|^2: %.6f", name, fullsum_fidelity)
    logger.info("%s psi0 vs fullsum max|Δ|: %.3e", name, fullsum_max)

    if print_table:
        _log_wavefunction_table(
            f"{name} t=0 amplitude table (psi_init vs dense)",
            basis,
            psi_fullsum,
            psi0,
        )

    energy_t0 = vstate.expect(H).mean
    energy_t0_dense = psi0.conj() @ (H.to_dense() @ psi0)
    energy_t0_fullsum = nk.vqs.FullSumState(hi, model).expect(H).mean
    logger.info("TN Energy at t=0: %s", complex(energy_t0))
    logger.info("TN Energy at t=0 (dense): %s", complex(energy_t0_dense))
    logger.info("TN Energy (FullSum) at t=0: %s", complex(energy_t0_fullsum))
    tdvp.run(T, show_progress=True)
    energy_t_final = vstate.expect(H).mean
    logger.info("TN Energy at t=%.6f: %s", T, complex(energy_t_final))
    energy_var = vstate.expect(H).mean
    sz_op = build_total_sz(hi)
    sz_var = vstate.expect(sz_op).mean
    psi_var = normalized_wavefunction(vstate, basis, baseline_eps=0.0)
    psi_var_aligned, overlap = align_phase(psi_exact, psi_var)
    amp_l2 = jnp.linalg.norm(psi_var_aligned - psi_exact)
    amp_max = jnp.max(jnp.abs(psi_var_aligned - psi_exact))
    fidelity = float(jnp.abs(overlap) ** 2)
    norm_exact = float(jnp.linalg.norm(psi_exact))
    norm_var = float(jnp.linalg.norm(psi_var_aligned))
    norm_init = float(jnp.linalg.norm(psi0))

    logger.info("%s RK4 (T=%.3f) energy: %s", name, T, complex(energy_var))
    logger.info("%s RK4 (T=%.3f) <Sz>:   %s", name, T, complex(sz_var))
    logger.info("%s dense expm energy:     %s", name, complex(energy_exact))
    logger.info("%s dense expm <Sz>:       %s", name, complex(sz_exact))
    logger.info("%s |<psi_exact|psi_var>|^2: %.6f", name, fidelity)
    logger.info(
        "%s norms: ||psi_init||=%.6f, ||psi_exact||=%.6f, ||psi_var||=%.6f",
        name,
        norm_init,
        norm_exact,
        norm_var,
    )
    logger.info(
        "%s amplitude mismatch: ||Δψ||₂=%.3e, max|Δψ|=%.3e",
        name,
        float(amp_l2),
        float(amp_max),
    )
    if print_table:
        _log_wavefunction_table(
            f"{name} amplitude table (aligned phases):",
            basis,
            psi_exact,
            psi_var_aligned,
        )

    if return_states:
        return psi_exact, psi_var_aligned, basis
    return None, None, None


def main():
    length = 3
    shape = (length, length)
    n_sites = length * length
    mps_bond = 16
    peps_bond = 4
    # Use zip-up PEPS contraction with a moderate truncation dimension.
    truncate_bond_dimension = peps_bond * peps_bond
    T = 0.02
    n_steps = 10  # dt = T / n_steps = 0.002

    hi, H, shape = build_heisenberg_square(length, pbc=False)
    sampler_kwargs = {"n_chains": 8}
    spins = random_bitstring(jax.random.key(0), hi.size)

    # 1. MPS real-time RK4 with custom TDVP (SR, no NTK)
    mps_model_custom = MPS(rngs=nnx.Rngs(0), n_sites=n_sites, bond_dim=mps_bond)
    _, psi_mps_custom, basis = run_case(
        "Real-time MPS (RK4, custom TDVP SR)",
        mps_model_custom,
        hi,
        H,
        nk.sampler.MetropolisLocal(hi, **sampler_kwargs),
        T=T,
        n_steps=n_steps,
        driver_backend="custom",
        use_ntk=False,
        solver=solve_cholesky,
        diag_shift=1e-4,
        n_samples=4096,
        return_states=True,
        spins=spins,
    )

    # 2. MPS real-time RK4 with NetKet TDVP (standard SR)
    mps_model_netket = MPS(rngs=nnx.Rngs(1), n_sites=n_sites, bond_dim=mps_bond)
    _, psi_mps_netket, _ = run_case(
        "Real-time MPS (RK4, NetKet TDVP SR)",
        mps_model_netket,
        hi,
        H,
        nk.sampler.MetropolisLocal(hi, **sampler_kwargs),
        T=T,
        n_steps=n_steps,
        driver_backend="netket",
        solver=nk.optimizer.solver.cholesky,
        diag_shift=1e-4,
        n_samples=4096,
        return_states=True,
        spins=spins,
        print_table=False,
    )

    # 3. MPS real-time RK4 with NetKet TDVP (FullSumState)
    mps_model_fullsum = MPS(rngs=nnx.Rngs(2), n_sites=n_sites, bond_dim=mps_bond)
    run_case(
        "Real-time MPS (RK4, NetKet TDVP FullSum)",
        mps_model_fullsum,
        hi,
        H,
        None,
        T=T,
        n_steps=n_steps,
        driver_backend="netket",
        solver=nk.optimizer.solver.cholesky,
        diag_shift=1e-4,
        use_full_sum=True,
        spins=spins,
        print_table=False,
    )

    cross_overlap = jnp.vdot(psi_mps_custom, psi_mps_netket)
    cross_fidelity = float(jnp.abs(cross_overlap) ** 2)
    logger.info("\n%s", "=" * 60)
    logger.info("Custom TDVP vs NetKet TDVP fidelity (final variational states)")
    logger.info("=" * 60)
    logger.info("|<psi_custom|psi_netket>|^2 = %.6f", cross_fidelity)

    logger.info("\n%s", "=" * 60)
    logger.info("MPS vs PEPS fidelity (final variational states)")
    logger.info("=" * 60)
if __name__ == "__main__":
    main()
