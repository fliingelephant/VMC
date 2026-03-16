"""Fixed-step SR benchmark for pure Z2 lattice gauge theory.

This example mirrors the benchmark-script structure used by the ground-state
examples, but targets the gauge-invariant PEPS implementation for pure Z2 LGT.
It runs one fixed-step imaginary-time SR trajectory and records:

- total energy
- mean plaquette value
- mean horizontal-link Z expectation
- mean vertical-link Z expectation
"""

from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver
from vmc.operators import PlaquetteOperator
from vmc.peps import ZipUp
from vmc.peps.gi import GILocalHamiltonian, GIPEPS, GIPEPSConfig
from vmc.peps.gi.local_terms import LinkDiagonalTerm, build_electric_terms
from vmc.preconditioners import (
    DirectSolve,
    MetricsConfig,
    SRPreconditioner,
    solve_cholesky,
)
from vmc.qgt import ParameterSpace


L = 3
SHAPE = (L, L)
H_COUPLING = 1.0
G_COUPLING = 0.2

BOND_DIM = 2
BOUNDARY_DIM = 3 * BOND_DIM

N_SAMPLES = 1024
N_CHAINS = 64
SEED = 42

SR_FIXED_STEPS = 100
SR_FIXED_DT = 0.01
SR_DIAG_SHIFT = 1e-8

SR_METRICS_CONFIG = MetricsConfig(
    record_FS_norm=True,
    record_TDVP_residual=True,
    record_SR_solve_residual=True,
    record_step_wall_time=True,
)


def build_z2_hamiltonian(
    shape: tuple[int, int],
    h: float,
    g: float,
) -> GILocalHamiltonian:
    """Build the pure Z2 gauge Hamiltonian."""
    n_rows, n_cols = shape
    plaquette_terms = tuple(
        PlaquetteOperator(row=r, col=c, coeff=-h)
        for r in range(n_rows - 1)
        for c in range(n_cols - 1)
    )
    electric_terms = build_electric_terms(shape, coeff=g, N=2)
    return GILocalHamiltonian(shape=shape, terms=electric_terms + plaquette_terms)


def build_mean_plaquette_observable(shape: tuple[int, int]) -> GILocalHamiltonian:
    """Build the average plaquette operator.

    ``PlaquetteOperator`` evaluates ``coeff * (P + P†)``. For Z2, ``P = P†``,
    so the average plaquette value is obtained with a coefficient of
    ``1 / (2 * n_plaquettes)``.
    """
    n_rows, n_cols = shape
    n_plaquettes = (n_rows - 1) * (n_cols - 1)
    coeff = jnp.asarray(0.5 / n_plaquettes, dtype=jnp.complex128)
    terms = tuple(
        PlaquetteOperator(row=r, col=c, coeff=coeff)
        for r in range(n_rows - 1)
        for c in range(n_cols - 1)
    )
    return GILocalHamiltonian(shape=shape, terms=terms)


def build_mean_link_z_observable(
    shape: tuple[int, int],
    *,
    orientation: str,
) -> GILocalHamiltonian:
    """Build the average Z expectation on horizontal or vertical links."""
    n_rows, n_cols = shape
    if orientation == "h":
        count = n_rows * (n_cols - 1)
        positions = tuple(
            (r, c)
            for r in range(n_rows)
            for c in range(n_cols - 1)
        )
    elif orientation == "v":
        count = (n_rows - 1) * n_cols
        positions = tuple(
            (r, c)
            for r in range(n_rows - 1)
            for c in range(n_cols)
        )
    else:
        raise ValueError(f"Unsupported orientation: {orientation!r}")

    diag = jnp.asarray([1.0, -1.0], dtype=jnp.complex128) / count
    terms = tuple(
        LinkDiagonalTerm(
            sites=(position,),
            diag=diag,
            orientation=orientation,
        )
        for position in positions
    )
    return GILocalHamiltonian(shape=shape, terms=terms)


def build_problem() -> tuple[GILocalHamiltonian, tuple[GILocalHamiltonian, ...]]:
    """Build the Hamiltonian and benchmark observables."""
    return (
        build_z2_hamiltonian(SHAPE, H_COUPLING, G_COUPLING),
        (
            build_mean_plaquette_observable(SHAPE),
            build_mean_link_z_observable(SHAPE, orientation="h"),
            build_mean_link_z_observable(SHAPE, orientation="v"),
        ),
    )


def build_model(seed: int) -> GIPEPS:
    """Build a fresh GIPEPS model."""
    return GIPEPS(
        rngs=nnx.Rngs(seed),
        config=GIPEPSConfig(
            shape=SHAPE,
            N=2,
            phys_dim=1,
            Qx=0,
            degeneracy_per_charge=(BOND_DIM, BOND_DIM),
            charge_of_site=(0,),
        ),
        contraction_strategy=ZipUp(truncate_bond_dimension=BOUNDARY_DIM),
    )


def append_series(series: dict[str, list], **values) -> None:
    """Append one row into a columnar series dict."""
    for key, value in values.items():
        series.setdefault(key, []).append(value)


def benchmark_output_dir() -> Path:
    """Build the output directory for the current benchmark settings."""
    g_token = format(G_COUPLING, ".3f").replace(".", "p")
    return (
        Path(__file__).resolve().parent
        / f"z2_pure_gauge_benchmark_{L}x{L}_g{g_token}_ns{N_SAMPLES}_{SR_FIXED_STEPS}"
    )


def save_run(
    output_path: Path,
    *,
    config_data: dict,
    series: dict[str, list],
) -> None:
    """Write one benchmark run to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "final_step": series["step"][-1],
        "final_energy_mean": series["energy_mean"][-1],
        "final_energy_error": series["energy_error"][-1],
        "final_plaquette_mean": series["plaquette_mean"][-1],
        "final_plaquette_error": series["plaquette_error"][-1],
        "final_z_h_mean": series["z_h_mean"][-1],
        "final_z_h_error": series["z_h_error"][-1],
        "final_z_v_mean": series["z_v_mean"][-1],
        "final_z_v_error": series["z_v_error"][-1],
        "final_imaginary_time": series["imaginary_time"][-1],
    }
    output_path.write_text(
        json.dumps(
            {
                "problem": {
                    "gauge_group": "Z2",
                    "shape": SHAPE,
                    "h": H_COUPLING,
                    "g": G_COUPLING,
                    "bond_dim": BOND_DIM,
                    "boundary_method": "ZipUp",
                    "boundary_dimension": BOUNDARY_DIM,
                    "n_samples": N_SAMPLES,
                    "n_chains": N_CHAINS,
                    "seed": SEED,
                    "Qx": 0,
                },
                "config": config_data,
                "series": series,
                "summary": summary,
            },
            indent=2,
        )
    )
    print(f"Saved {output_path}", flush=True)


def run_sr(
    hamiltonian: GILocalHamiltonian,
    observables: tuple[GILocalHamiltonian, ...],
    output_path: Path,
    *,
    n_steps: int,
    dt: float,
) -> None:
    """Run fixed-step SR and save the trajectory."""
    label = output_path.stem
    driver = TDVPDriver(
        build_model(SEED),
        hamiltonian,
        observables=observables,
        preconditioner=SRPreconditioner(
            space=ParameterSpace(),
            strategy=DirectSolve(solver=solve_cholesky),
            diag_shift=SR_DIAG_SHIFT,
            metrics_config=SR_METRICS_CONFIG,
        ),
        dt=dt,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(SEED),
        n_samples=N_SAMPLES,
        n_chains=N_CHAINS,
        full_gradient=False,
    )
    series: dict[str, list] = {}
    print(
        (
            f"[{label}] step t dt wall_time energy energy_err energy_var "
            "plaquette plaquette_err z_h z_h_err z_v z_v_err "
            "applied_FS_step_norm_squared FS_norm_squared TDVP_residual "
            "SR_solve_residual"
        ),
        flush=True,
    )

    for step in range(1, n_steps + 1):
        driver.run(dt)
        metrics = driver.metrics
        energy = driver.energy
        plaquette, z_h, z_v = driver.observable_stats
        fs_norm_squared = float(metrics["FS_norm_squared"])
        row = {
            "step": step,
            "imaginary_time": float(driver.t),
            "dt": dt,
            "step_wall_time": float(metrics["step_wall_time"]),
            "energy_mean": float(energy.mean.real),
            "energy_error": float(energy.error_of_mean.real),
            "energy_variance": float(energy.variance.real),
            "plaquette_mean": float(plaquette.mean.real),
            "plaquette_error": float(plaquette.error_of_mean.real),
            "z_h_mean": float(z_h.mean.real),
            "z_h_error": float(z_h.error_of_mean.real),
            "z_v_mean": float(z_v.mean.real),
            "z_v_error": float(z_v.error_of_mean.real),
            "applied_FS_step_norm_squared": dt**2 * fs_norm_squared,
            "FS_norm_squared": fs_norm_squared,
            "TDVP_residual": float(metrics["TDVP_residual"]),
            "SR_solve_residual": float(metrics["SR_solve_residual"]),
        }
        append_series(series, **row)
        print(
            (
                f"[{label}] {row['step']:3d} {row['imaginary_time']:.6f} "
                f"{row['dt']:.6f} {row['step_wall_time']:.3f} "
                f"{row['energy_mean']:.10f} {row['energy_error']:.6f} "
                f"{row['energy_variance']:.6f} {row['plaquette_mean']:.10f} "
                f"{row['plaquette_error']:.6f} {row['z_h_mean']:.10f} "
                f"{row['z_h_error']:.6f} {row['z_v_mean']:.10f} "
                f"{row['z_v_error']:.6f} "
                f"{row['applied_FS_step_norm_squared']:.6e} "
                f"{row['FS_norm_squared']:.6e} "
                f"{row['TDVP_residual']:.6e} "
                f"{row['SR_solve_residual']:.6e}"
            ),
            flush=True,
        )

    save_run(
        output_path,
        config_data={
            "method": label,
            "diag_shift": SR_DIAG_SHIFT,
            "dt": dt,
            "n_steps": n_steps,
        },
        series=series,
    )


def main() -> None:
    """Run the fixed-step SR benchmark."""
    hamiltonian, observables = build_problem()
    output_dir = benchmark_output_dir()
    print(
        (
            f"Benchmarking pure Z2 gauge theory on {SHAPE}, "
            f"h={H_COUPLING:.3f}, g={G_COUPLING:.3f}, "
            f"Dk={BOND_DIM}, Dc={BOUNDARY_DIM}, nsamples={N_SAMPLES}"
        ),
        flush=True,
    )
    run_sr(
        hamiltonian,
        observables,
        output_dir / "sr_fixed.json",
        n_steps=SR_FIXED_STEPS,
        dt=SR_FIXED_DT,
    )


if __name__ == "__main__":
    main()
