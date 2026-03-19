"""PEPS tVMC quench for the 2D transverse-field Ising model.

This script follows the smooth quench used for tensor-network simulations in
Schmitt et al. (2022):

    epsilon(t) = t / tau_q - 4 t^3 / (27 tau_q^3)
    J(t) = 1 + epsilon(t)
    g(t) = g_c (1 - epsilon(t))

with t in [-3 tau_q / 2, 3 tau_q / 2] on an odd-L open square lattice.

The initial state is the exact |+x>^N product state embedded into a PEPS of
bond dimension D, without additional gauge-randomizing bond transformations.
The script records raw dynamical observables only: energy, m_x, and the
center-spin correlation profile Czz(R) along lattice axes.
"""

from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.drivers import RK4, RealTimeUnit, TDVPDriver
from vmc.operators import (
    CubicSchedule,
    DiagonalOperator,
    LocalHamiltonian,
    OneSiteOperator,
    TimeDependentHamiltonian,
)
from vmc.peps import PEPS, Variational
from vmc.preconditioners import (
    DirectSolve,
    MetricsConfig,
    SRPreconditioner,
    solve_cg,
    solve_cholesky,
    solve_svd,
)


GC = 3.04438
SIGMA_X = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
SIGMA_ZZ_DIAG = jnp.asarray([1.0, -1.0, -1.0, 1.0], dtype=jnp.complex128)
PLUS_X = jnp.asarray([1.0, 1.0], dtype=jnp.complex128) / jnp.sqrt(2.0)


@dataclass(frozen=True)
class RunConfig:
    """Runtime configuration for a single Schmitt-style quench."""

    L: int = 7
    tau_q: float = 0.8
    dt: float = 0.01
    bond_dim: int = 4
    boundary_dim: int = 16
    seed: int = 0
    n_samples: int = 10240
    n_chains: int = 1024
    solver: str = "cholesky"
    diag_shift: float = 1e-8
    gc: float = GC
    csv: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.L, self.L)


def smooth_epsilon(t: float | jax.Array, tau_q: float) -> jax.Array:
    """Return the smooth cubic Schmitt ramp parameter epsilon(t)."""
    t = jnp.asarray(t, dtype=jnp.float64)
    tau_q = jnp.asarray(tau_q, dtype=jnp.float64)
    return t / tau_q - 4.0 * t**3 / (27.0 * tau_q**3)


def smooth_time_window(tau_q: float) -> tuple[float, float]:
    """Return the Schmitt smooth-ramp time window."""
    return (-1.5 * tau_q, 1.5 * tau_q)


def center_site(shape: tuple[int, int]) -> tuple[int, int]:
    """Return the unique central site for an odd-by-odd lattice."""
    if shape[0] % 2 == 0 or shape[1] % 2 == 0:
        raise ValueError("Center-spin correlations require an odd-by-odd lattice.")
    return (shape[0] // 2, shape[1] // 2)


def center_axis_distances(shape: tuple[int, int]) -> tuple[int, ...]:
    """Return paper-aligned axis distances from the central spin."""
    row, col = center_site(shape)
    max_distance = min(row, shape[0] - 1 - row, col, shape[1] - 1 - col)
    return tuple(range(1, max_distance + 1))


def build_center_czz_observables(
    shape: tuple[int, int],
) -> tuple[LocalHamiltonian, ...]:
    """Build Czz(R) observables relative to the central spin along lattice axes."""
    row, col = center_site(shape)
    observables = []
    for distance in center_axis_distances(shape):
        sites = (
            ((row, col), (row, col - distance)),
            ((row, col), (row, col + distance)),
            ((row, col), (row - distance, col)),
            ((row, col), (row + distance, col)),
        )
        diag = SIGMA_ZZ_DIAG / len(sites)
        observables.append(
            LocalHamiltonian(
                shape=shape,
                terms=tuple(DiagonalOperator(sites=pair, diag=diag) for pair in sites),
            )
        )
    return tuple(observables)


def build_mx_observable(shape: tuple[int, int]) -> LocalHamiltonian:
    """Build average transverse magnetization."""
    coeff = 1.0 / (shape[0] * shape[1])
    return LocalHamiltonian(
        shape=shape,
        terms=tuple(
            OneSiteOperator(row=row, col=col, op=coeff * SIGMA_X)
            for row in range(shape[0])
            for col in range(shape[1])
        ),
    )


def build_plus_x_product_peps(
    shape: tuple[int, int],
    bond_dim: int,
    boundary_dim: int,
    seed: int,
) -> PEPS:
    """Build the exact |+x>^N product state as a PEPS."""
    model = PEPS(
        rngs=nnx.Rngs(seed),
        shape=shape,
        bond_dim=bond_dim,
        phys_dim=2,
        contraction_strategy=Variational(boundary_dim),
        dtype=jnp.complex128,
    )
    n_rows, n_cols = shape
    for row in range(n_rows):
        for col in range(n_cols):
            up, down, left, right = model.site_dims(
                row, col, n_rows, n_cols, bond_dim
            )
            tensor = jnp.zeros((2, up, down, left, right), dtype=jnp.complex128)
            model.tensors[row][col][...] = tensor.at[:, 0, 0, 0, 0].set(PLUS_X)
    return model


def build_schmitt_smooth_tfim(
    shape: tuple[int, int],
    tau_q: float,
    gc: float,
) -> TimeDependentHamiltonian:
    """Build the smooth-ramp Schmitt TFIM Hamiltonian."""
    terms = []
    constant = []
    linear = []
    quadratic = []
    cubic = []
    j_cubic = -4.0 / (27.0 * tau_q**3)
    g_cubic = 4.0 * gc / (27.0 * tau_q**3)

    for row in range(shape[0]):
        for col in range(shape[1]):
            terms.append(OneSiteOperator(row=row, col=col, op=-SIGMA_X))
            constant.append(gc)
            linear.append(-gc / tau_q)
            quadratic.append(0.0)
            cubic.append(g_cubic)

            if col + 1 < shape[1]:
                terms.append(
                    DiagonalOperator(
                        sites=((row, col), (row, col + 1)),
                        diag=-SIGMA_ZZ_DIAG,
                    )
                )
                constant.append(1.0)
                linear.append(1.0 / tau_q)
                quadratic.append(0.0)
                cubic.append(j_cubic)

            if row + 1 < shape[0]:
                terms.append(
                    DiagonalOperator(
                        sites=((row, col), (row + 1, col)),
                        diag=-SIGMA_ZZ_DIAG,
                    )
                )
                constant.append(1.0)
                linear.append(1.0 / tau_q)
                quadratic.append(0.0)
                cubic.append(j_cubic)

    return TimeDependentHamiltonian(
        base=LocalHamiltonian(shape=shape, terms=tuple(terms)),
        schedule=CubicSchedule(
            constant=jnp.asarray(constant, dtype=jnp.float64),
            linear=jnp.asarray(linear, dtype=jnp.float64),
            quadratic=jnp.asarray(quadratic, dtype=jnp.float64),
            cubic=jnp.asarray(cubic, dtype=jnp.float64),
        ),
    )


def solver_from_name(name: str):
    """Return the configured linear solver."""
    return {
        "cg": solve_cg,
        "cholesky": solve_cholesky,
        "svd": solve_svd,
    }[name]


def measurement_row(
    t: float,
    tau_q: float,
    gc: float,
    energy_per_site: float,
    energy_err: float,
    mx: float,
    mx_err: float,
    czz_values: tuple[float, ...],
    czz_errs: tuple[float, ...],
    distances: tuple[int, ...],
) -> dict[str, float]:
    """Assemble one output row."""
    epsilon = float(smooth_epsilon(t, tau_q))
    row = {
        "time": float(t),
        "epsilon": epsilon,
        "J": 1.0 + epsilon,
        "g": gc * (1.0 - epsilon),
        "energy": energy_per_site,
        "energy_err": energy_err,
        "mx": mx,
        "mx_err": mx_err,
    }
    for distance, value, err in zip(distances, czz_values, czz_errs, strict=True):
        row[f"czz_r{distance}"] = value
        row[f"czz_r{distance}_err"] = err
    return row


def build_driver(cfg: RunConfig) -> tuple[TDVPDriver, tuple[int, ...]]:
    """Build the PEPS model and TDVP driver for one quench."""
    model = build_plus_x_product_peps(
        cfg.shape,
        cfg.bond_dim,
        cfg.boundary_dim,
        cfg.seed,
    )
    distances = center_axis_distances(cfg.shape)
    observables = (
        build_mx_observable(cfg.shape),
        *build_center_czz_observables(cfg.shape),
    )
    t0, _ = smooth_time_window(cfg.tau_q)
    driver = TDVPDriver(
        model,
        build_schmitt_smooth_tfim(cfg.shape, cfg.tau_q, cfg.gc),
        observables=observables,
        preconditioner=SRPreconditioner(
            diag_shift=cfg.diag_shift,
            strategy=DirectSolve(solver=solver_from_name(cfg.solver)),
            metrics_config=MetricsConfig(
                record_FS_norm=True,
                record_TDVP_residual=True,
                record_SR_solve_residual=True,
                record_step_wall_time=True,
            ),
        ),
        dt=cfg.dt,
        t0=t0,
        time_unit=RealTimeUnit(),
        integrator=RK4(),
        sampler_key=jax.random.key(cfg.seed + 17),
        n_samples=cfg.n_samples,
        n_chains=cfg.n_chains,
        full_gradient=False,
    )
    return driver, distances


def _stat_err(stat) -> float:
    eom = stat.error_of_mean
    return float(eom.real) if jnp.isfinite(eom) else 0.0


def run_single_quench(
    cfg: RunConfig,
    csv_path: Path | None = None,
    tau_q_column: float | None = None,
) -> list[dict[str, float]]:
    """Run one Schmitt-style smooth quench and return the logged trajectory.

    If csv_path is given, rows are written incrementally (crash-safe).
    """
    t0, t1 = smooth_time_window(cfg.tau_q)
    total_time = t1 - t0
    steps = int(round(total_time / cfg.dt))
    if not jnp.isclose(total_time, steps * cfg.dt):
        raise ValueError("The smooth-ramp window must be an integer multiple of dt.")

    print(
        f"[schmitt] L={cfg.L}  tau_q={cfg.tau_q:.4f}  D={cfg.bond_dim}  "
        f"D'={cfg.boundary_dim}  Ns={cfg.n_samples}  chains={cfg.n_chains}  "
        f"solver={cfg.solver}  dt={cfg.dt:.4f}  steps={steps}",
        flush=True,
    )
    print("[schmitt] Building driver...", flush=True)
    driver, distances = build_driver(cfg)
    print(f"[schmitt] Driver built. Starting time evolution ({steps} steps)...", flush=True)
    n_sites = cfg.L * cfg.L
    rows = []
    csv_writer = None
    csv_fh = None

    print(
        "[schmitt] step time wall_time E/N E/N_err mx mx_err TDVP_res SR_res",
        flush=True,
    )
    for step_i in range(steps):
        driver.run(cfg.dt)
        e_stats = driver.energy
        mx_stats = driver.observable_stats[0]
        czz_stats = driver.observable_stats[1:]
        metrics = driver.metrics
        wall = metrics.get("step_wall_time", float("nan"))
        tdvp_res = metrics.get("TDVP_residual", float("nan"))
        sr_res = metrics.get("SR_solve_residual", float("nan"))
        print(
            f"[schmitt] {step_i + 1:4d}/{steps} {float(driver.t):8.4f} "
            f"{wall:7.2f}s  E/N={float(e_stats.mean.real) / n_sites:.8f} "
            f"+/- {_stat_err(e_stats) / n_sites:.6f}  "
            f"mx={float(mx_stats.mean.real):.6f} +/- {_stat_err(mx_stats):.6f}  "
            f"TDVP_res={tdvp_res:.2e}  SR_res={sr_res:.2e}",
            flush=True,
        )
        row = measurement_row(
            float(driver.t),
            cfg.tau_q,
            cfg.gc,
            float(e_stats.mean.real) / n_sites,
            _stat_err(e_stats) / n_sites,
            float(mx_stats.mean.real),
            _stat_err(mx_stats),
            tuple(float(s.mean.real) for s in czz_stats),
            tuple(_stat_err(s) for s in czz_stats),
            distances,
        )
        if tau_q_column is not None:
            row = {"tau_q": tau_q_column, **row}
        rows.append(row)

        if csv_path is not None:
            if csv_writer is None:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                csv_fh = csv_path.open("w", newline="")
                csv_writer = csv.DictWriter(csv_fh, fieldnames=list(row.keys()))
                csv_writer.writeheader()
            csv_writer.writerow(row)
            csv_fh.flush()

    if csv_fh is not None:
        csv_fh.close()
        print(f"[schmitt] Saved {csv_path}", flush=True)
    return rows


def run_tauq_sweep(
    cfg: RunConfig, tau_qs: tuple[float, ...], csv_path: Path | None = None,
) -> list[dict[str, float]]:
    """Run multiple quenches and concatenate the logged rows."""
    all_rows = []
    for tau_q in tau_qs:
        sub_cfg = replace(cfg, tau_q=tau_q)
        all_rows.extend(
            run_single_quench(sub_cfg, csv_path=csv_path, tau_q_column=tau_q)
        )
    return all_rows


def parse_args() -> tuple[RunConfig, tuple[float, ...]]:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a PEPS Schmitt-style smooth TFIM quench on an odd OBC lattice."
    )
    parser.add_argument("--L", type=int, default=7)
    parser.add_argument("--tau-q", type=float, nargs="+", default=[0.8])
    parser.add_argument("--dt", type=float, default=RunConfig.dt)
    parser.add_argument("--bond-dim", type=int, default=RunConfig.bond_dim)
    parser.add_argument("--boundary-dim", type=int, default=RunConfig.boundary_dim)
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument("--n-samples", type=int, default=RunConfig.n_samples)
    parser.add_argument("--n-chains", type=int, default=RunConfig.n_chains)
    parser.add_argument(
        "--solver",
        choices=("cg", "cholesky", "svd"),
        default="cholesky",
    )
    parser.add_argument("--diag-shift", type=float, default=RunConfig.diag_shift)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    cfg = RunConfig(
        L=args.L,
        tau_q=float(args.tau_q[0]),
        dt=args.dt,
        bond_dim=args.bond_dim,
        boundary_dim=args.boundary_dim,
        seed=args.seed,
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        solver=args.solver,
        diag_shift=args.diag_shift,
        csv=args.csv,
    )
    center_site(cfg.shape)
    if cfg.dt <= 0.0:
        raise ValueError("--dt must be positive.")
    if cfg.bond_dim < 1:
        raise ValueError("--bond-dim must be at least 1.")
    if cfg.boundary_dim < 1:
        raise ValueError("--boundary-dim must be at least 1.")
    if any(tau_q <= 0.0 for tau_q in args.tau_q):
        raise ValueError("--tau-q values must be positive.")
    return cfg, tuple(float(tau_q) for tau_q in args.tau_q)


def main() -> None:
    """Run one or more smooth Schmitt quenches and print the logged rows."""
    cfg, tau_qs = parse_args()
    csv_path = Path(cfg.csv) if cfg.csv else None
    if len(tau_qs) == 1:
        rows = run_single_quench(cfg, csv_path=csv_path)
    else:
        rows = run_tauq_sweep(cfg, tau_qs, csv_path=csv_path)
    headers = tuple(rows[0].keys())
    print("\t".join(headers))
    for row in rows:
        print("\t".join(f"{float(row[key]):.10f}" for key in headers))


if __name__ == "__main__":
    main()
