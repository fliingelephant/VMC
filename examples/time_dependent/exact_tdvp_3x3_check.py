"""Benchmark 3x3 time-dependent PEPS TDVP against exact evolution.

Hamiltonian:
    H(t) = jzz * sum_<ij> sigmaz_i sigmaz_j - hx(t) * sum_i sigmax_i
    hx(t) = hx0 + hx_slope * t

The exact reference evolves the full 9-qubit state vector (dimension 512) with
midpoint piecewise-constant propagation and configurable sub-stepping per TDVP
step. The PEPS trajectory uses TDVPDriver.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from flax import nnx

from vmc import config  # noqa: F401 - JAX config must be imported first
from vmc.drivers import RK4, RealTimeUnit, TDVPDriver
from vmc.operators import (
    AffineSchedule,
    DiagonalTerm,
    LocalHamiltonian,
    OneSiteTerm,
    TimeDependentHamiltonian,
)
from vmc.peps import NoTruncation, PEPS
from vmc.peps.standard.compat import _value
from vmc.preconditioners import (
    DirectSolve,
    SRPreconditioner,
    solve_cg,
    solve_cholesky,
    solve_svd,
)


@dataclass(frozen=True)
class RunConfig:
    """Benchmark configuration."""

    shape: tuple[int, int] = (3, 3)
    steps: int = 10
    dt: float = 0.02
    jzz: float = -1.0
    hx0: float = 0.4
    hx_slope: float = -0.15
    peps_hx_slope: float | None = None
    seed: int = 0
    n_samples: int = 4096
    n_chains: int = 64
    solver: str = "svd"
    bond_dim: int = 1
    diag_shift: float = 1e-8
    exact_substeps: int = 8
    csv: str | None = None


def _site_index(row: int, col: int, n_cols: int) -> int:
    return row * n_cols + col


def _nearest_neighbor_edges(shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    n_rows, n_cols = shape
    edges = []
    for row in range(n_rows):
        for col in range(n_cols):
            if col + 1 < n_cols:
                edges.append(
                    (_site_index(row, col, n_cols), _site_index(row, col + 1, n_cols))
                )
            if row + 1 < n_rows:
                edges.append(
                    (_site_index(row, col, n_cols), _site_index(row + 1, col, n_cols))
                )
    return tuple(edges)


def _all_occupancy_states(n_sites: int) -> jax.Array:
    basis = jnp.arange(1 << n_sites, dtype=jnp.uint32)
    bit_positions = jnp.arange(n_sites, dtype=jnp.uint32)
    return ((basis[:, None] >> bit_positions[None, :]) & 1).astype(jnp.int32)


def _mz_basis_values(states: jax.Array) -> jax.Array:
    return jnp.mean(1.0 - 2.0 * states.astype(jnp.float64), axis=1)


def _average_mz(psi: jax.Array, mz_basis: jax.Array) -> float:
    probabilities = jnp.abs(psi) ** 2
    return float(jnp.real(jnp.sum(probabilities * mz_basis)))


def _build_exact_hamiltonian_parts(
    shape: tuple[int, int], *, jzz: float
) -> tuple[jax.Array, jax.Array]:
    n_sites = shape[0] * shape[1]
    dim = 1 << n_sites
    basis = np.arange(dim, dtype=np.uint32)
    zz_diag = np.zeros((dim,), dtype=np.float64)
    for left, right in _nearest_neighbor_edges(shape):
        z_left = 1.0 - 2.0 * ((basis >> left) & 1).astype(np.float64)
        z_right = 1.0 - 2.0 * ((basis >> right) & 1).astype(np.float64)
        zz_diag = zz_diag + jzz * z_left * z_right

    h_static = np.diag(zz_diag.astype(np.complex128))
    h_drive = np.zeros((dim, dim), dtype=np.complex128)
    indices = np.arange(dim, dtype=np.int64)
    for site in range(n_sites):
        flipped = indices ^ (1 << site)
        h_drive[indices, flipped] += -1.0

    return jnp.asarray(h_static), jnp.asarray(h_drive)


def _build_vmc_time_dependent_hamiltonian(
    shape: tuple[int, int], *, jzz: float, hx0: float, hx_slope: float
) -> TimeDependentHamiltonian:
    sigmax = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    sigmaz_sigmaz_diag = jnp.array([1.0, -1.0, -1.0, 1.0], dtype=jnp.complex128)
    terms = []
    offsets = []
    slopes = []
    n_rows, n_cols = shape
    for row in range(n_rows):
        for col in range(n_cols):
            terms.append(OneSiteTerm(row=row, col=col, op=-sigmax))
            offsets.append(hx0)
            slopes.append(hx_slope)
            if col + 1 < n_cols:
                terms.append(
                    DiagonalTerm(
                        sites=((row, col), (row, col + 1)),
                        diag=jzz * sigmaz_sigmaz_diag,
                    )
                )
                offsets.append(1.0)
                slopes.append(0.0)
            if row + 1 < n_rows:
                terms.append(
                    DiagonalTerm(
                        sites=((row, col), (row + 1, col)),
                        diag=jzz * sigmaz_sigmaz_diag,
                    )
                )
                offsets.append(1.0)
                slopes.append(0.0)
    return TimeDependentHamiltonian(
        base=LocalHamiltonian(shape=shape, terms=tuple(terms)),
        schedule=AffineSchedule(
            offset=jnp.asarray(offsets, dtype=jnp.float64),
            slope=jnp.asarray(slopes, dtype=jnp.float64),
        ),
    )


def _normalized_peps_state(model: PEPS, states: jax.Array) -> jax.Array:
    amplitudes = _value(model, states)
    return amplitudes / jnp.linalg.norm(amplitudes)


def _run_exact_trajectory(
    cfg: RunConfig,
    initial_state: jax.Array,
    mz_basis: jax.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_static, h_drive = _build_exact_hamiltonian_parts(cfg.shape, jzz=cfg.jzz)
    psi = jnp.asarray(initial_state)
    dt_sub = cfg.dt / cfg.exact_substeps
    t = 0.0
    times = [t]
    mz_values = [_average_mz(psi, mz_basis)]
    psi_values = [np.asarray(psi)]
    for _ in range(cfg.steps):
        for substep in range(cfg.exact_substeps):
            t_mid = t + (substep + 0.5) * dt_sub
            hx_mid = cfg.hx0 + cfg.hx_slope * t_mid
            h_mid = h_static + jnp.asarray(hx_mid, dtype=h_static.real.dtype) * h_drive
            propagator = jsp_linalg.expm(-1j * dt_sub * h_mid)
            psi = propagator @ psi
        psi = psi / jnp.linalg.norm(psi)
        t = t + cfg.dt
        times.append(t)
        mz_values.append(_average_mz(psi, mz_basis))
        psi_values.append(np.asarray(psi))
    return np.asarray(times), np.asarray(mz_values), np.asarray(psi_values)


def _run_vmc_trajectory(
    cfg: RunConfig,
    model: PEPS,
    states: jax.Array,
    mz_basis: jax.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    operator = _build_vmc_time_dependent_hamiltonian(
        cfg.shape,
        jzz=cfg.jzz,
        hx0=cfg.hx0,
        hx_slope=cfg.hx_slope if cfg.peps_hx_slope is None else cfg.peps_hx_slope,
    )
    driver = TDVPDriver(
        model,
        operator,
        preconditioner=SRPreconditioner(
            diag_shift=cfg.diag_shift,
            strategy=DirectSolve(
                solver={
                    "cholesky": solve_cholesky,
                    "svd": solve_svd,
                    "cg": solve_cg,
                }[cfg.solver]
            ),
        ),
        dt=cfg.dt,
        t0=0.0,
        time_unit=RealTimeUnit(),
        integrator=RK4(),
        sampler_key=jax.random.key(cfg.seed + 11),
        n_samples=cfg.n_samples,
        n_chains=cfg.n_chains,
        full_gradient=True,
    )
    psi = _normalized_peps_state(driver.model, states)
    times = [0.0]
    mz_values = [_average_mz(psi, mz_basis)]
    psi_values = [np.asarray(psi)]
    for _ in range(cfg.steps):
        driver.run(cfg.dt)
        psi = _normalized_peps_state(driver.model, states)
        times.append(float(driver.t))
        mz_values.append(_average_mz(psi, mz_basis))
        psi_values.append(np.asarray(psi))
    return np.asarray(times), np.asarray(mz_values), np.asarray(psi_values)


def _parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Benchmark 3x3 time-dependent PEPS TDVP against exact evolution."
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--jzz", type=float, default=-1.0)
    parser.add_argument("--hx0", type=float, default=0.4)
    parser.add_argument("--hx-slope", type=float, default=-0.15)
    parser.add_argument("--peps-hx-slope", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--n-chains", type=int, default=64)
    parser.add_argument(
        "--solver",
        type=str,
        choices=("cholesky", "svd", "cg"),
        default="svd",
    )
    parser.add_argument("--bond-dim", type=int, default=1)
    parser.add_argument("--diag-shift", type=float, default=1e-8)
    parser.add_argument("--exact-substeps", type=int, default=8)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()
    if args.exact_substeps < 1:
        raise ValueError("--exact-substeps must be >= 1")
    return RunConfig(
        steps=args.steps,
        dt=args.dt,
        jzz=args.jzz,
        hx0=args.hx0,
        hx_slope=args.hx_slope,
        peps_hx_slope=args.peps_hx_slope,
        seed=args.seed,
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        solver=args.solver,
        bond_dim=args.bond_dim,
        diag_shift=args.diag_shift,
        exact_substeps=args.exact_substeps,
        csv=args.csv,
    )


def main() -> None:
    cfg = _parse_args()
    n_sites = cfg.shape[0] * cfg.shape[1]
    states = _all_occupancy_states(n_sites)
    mz_basis = _mz_basis_values(states)
    model = PEPS(
        rngs=nnx.Rngs(cfg.seed),
        shape=cfg.shape,
        bond_dim=cfg.bond_dim,
        phys_dim=2,
        contraction_strategy=NoTruncation(),
    )
    psi0 = _normalized_peps_state(model, states)
    exact_t, exact_mz, exact_psi = _run_exact_trajectory(cfg, psi0, mz_basis)
    vmc_t, vmc_mz, vmc_psi = _run_vmc_trajectory(cfg, model, states, mz_basis)
    if not np.allclose(exact_t, vmc_t):
        raise ValueError("Time grids differ between exact and VMC trajectories.")

    diff_mz = vmc_mz - exact_mz
    overlap = np.sum(np.conjugate(exact_psi) * vmc_psi, axis=1)
    fidelity = np.abs(overlap) ** 2
    rmse = float(np.sqrt(np.mean(diff_mz**2)))
    max_abs = float(np.max(np.abs(diff_mz)))

    print("time\tmz_exact\tmz_vmc\tdiff_mz\tfidelity")
    for t, m_ref, m_vmc, dmz, fid in zip(
        exact_t, exact_mz, vmc_mz, diff_mz, fidelity, strict=True
    ):
        print(f"{t:.6f}\t{m_ref:.8f}\t{m_vmc:.8f}\t{dmz:+.3e}\t{fid:.8f}")
    print(
        f"\nRMSE(mz)={rmse:.3e}, max_abs(mz)={max_abs:.3e}, "
        f"final_fidelity={float(fidelity[-1]):.8f}"
    )

    if cfg.csv:
        out = Path(cfg.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        arr = np.stack([exact_t, exact_mz, vmc_mz, diff_mz, fidelity], axis=1)
        np.savetxt(
            out,
            arr,
            delimiter=",",
            header="time,mz_exact,mz_vmc,diff_mz,fidelity",
            comments="",
        )
        print(f"saved {out}")


if __name__ == "__main__":
    main()
