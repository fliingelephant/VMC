"""Shared helpers for Z2 hard-core-boson ground-state example scripts."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver
from vmc.operators import PlaquetteOperator
from vmc.peps import GIPEPS, GIPEPSConfig, GILocalHamiltonian, Variational
from vmc.peps.gi.local_terms import (
    HorizontalMatterHoppingTerm,
    LinkDiagonalTerm,
    MatterMassTerm,
    VerticalMatterHoppingTerm,
    build_electric_terms,
)
from vmc.preconditioners import (
    DirectSolve,
    MetricsConfig,
    SRPreconditioner,
    solve_cholesky,
)
from vmc.qgt import ParameterSpace


CHARGE_OF_SITE = (0, 1)
QX = 0

DEFAULT_H = 1.0
DEFAULT_G = 0.33
DEFAULT_J = 0.5
DEFAULT_M = 0.0

DEFAULT_BOUNDARY_SWEEPS = 2
DEFAULT_DT = 0.01
DEFAULT_DIAG_SHIFT = 1e-4
DEFAULT_LOG_EVERY = 50
DEFAULT_SAVE_EVERY = 50

SR_METRICS_CONFIG = MetricsConfig(
    record_FS_norm=True,
    record_TDVP_residual=True,
    record_SR_solve_residual=True,
    record_step_wall_time=True,
)


def half_filling(shape: tuple[int, int]) -> int:
    """Return the half-filling hard-core boson number for a lattice."""
    return shape[0] * shape[1] // 2


def format_token(value: int | float) -> str:
    """Format numeric values for filesystem-safe run-directory names."""
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return format(float(value), ".17g").replace("-", "m").replace(".", "p")


def coupling_suffix(*, h: float, g: float, J: float, m: float) -> str:
    """Format the Hamiltonian couplings for a run-directory suffix."""
    return (
        f"h{format_token(h)}_g{format_token(g)}_"
        f"J{format_token(J)}_m{format_token(m)}"
    )


def build_z2_hardcore_boson_hamiltonian(
    shape: tuple[int, int],
    *,
    h: float,
    g: float,
    J: float,
    m: float,
) -> GILocalHamiltonian:
    """Build the Z2 gauge theory with hard-core bosons Hamiltonian."""
    n_rows, n_cols = shape
    plaquette_terms = tuple(
        PlaquetteOperator(row=row, col=col, coeff=-h)
        for row in range(n_rows - 1)
        for col in range(n_cols - 1)
    )
    electric_terms = build_electric_terms(shape, coeff=g, N=2)
    mass_terms = tuple(
        MatterMassTerm(
            row=row,
            col=col,
            coeff=m,
            charge_of_site=CHARGE_OF_SITE,
        )
        for row in range(n_rows)
        for col in range(n_cols)
    )
    horizontal_hops = tuple(
        HorizontalMatterHoppingTerm(row=row, col=col, coeff=J)
        for row in range(n_rows)
        for col in range(n_cols - 1)
    )
    vertical_hops = tuple(
        VerticalMatterHoppingTerm(row=row, col=col, coeff=J)
        for row in range(n_rows - 1)
        for col in range(n_cols)
    )
    return GILocalHamiltonian(
        shape=shape,
        terms=electric_terms + plaquette_terms + mass_terms + horizontal_hops + vertical_hops,
    )


def build_model(
    shape: tuple[int, int],
    *,
    particle_number: int,
    bond_dim_per_charge: int,
    boundary_dim: int,
    boundary_sweeps: int,
    seed: int,
) -> GIPEPS:
    """Build a GI-PEPS model for Z2 hard-core bosons."""
    return GIPEPS(
        rngs=nnx.Rngs(seed),
        config=GIPEPSConfig(
            shape=shape,
            N=2,
            phys_dim=2,
            Qx=QX,
            degeneracy_per_charge=(bond_dim_per_charge, bond_dim_per_charge),
            charge_of_site=CHARGE_OF_SITE,
            particle_number=particle_number,
        ),
        contraction_strategy=Variational(boundary_dim, n_sweeps=boundary_sweeps),
    )


def build_ground_state_driver(
    *,
    shape: tuple[int, int],
    h: float,
    g: float,
    J: float,
    m: float,
    particle_number: int,
    bond_dim_per_charge: int,
    boundary_dim: int,
    boundary_sweeps: int,
    seed: int,
    n_samples: int,
    n_chains: int,
    dt: float,
    diag_shift: float,
    observables: tuple[Any, ...] = (),
) -> TDVPDriver:
    """Build a ground-state TDVP driver for one Z2 hard-core-boson point."""
    return TDVPDriver(
        build_model(
            shape,
            particle_number=particle_number,
            bond_dim_per_charge=bond_dim_per_charge,
            boundary_dim=boundary_dim,
            boundary_sweeps=boundary_sweeps,
            seed=seed,
        ),
        build_z2_hardcore_boson_hamiltonian(shape, h=h, g=g, J=J, m=m),
        observables=observables,
        preconditioner=SRPreconditioner(
            space=ParameterSpace(),
            strategy=DirectSolve(solver=solve_cholesky),
            diag_shift=diag_shift,
            metrics_config=SR_METRICS_CONFIG,
        ),
        dt=dt,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(seed),
        n_samples=n_samples,
        n_chains=n_chains,
        full_gradient=False,
    )


def latest_paths(run_dir: Path) -> tuple[Path, Path]:
    """Return the checkpoint and metadata paths for one run directory."""
    return run_dir / "latest.npz", run_dir / "latest.json"


def prepare_run_dir(run_dir: Path, *, resume: bool) -> Path:
    """Prepare one overwritten or resumable run directory."""
    npz_path, json_path = latest_paths(run_dir)
    if resume:
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing checkpoint in {run_dir}")
        return run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_latest_json(run_dir: Path) -> dict[str, Any]:
    """Load the human-readable latest metadata for one run directory."""
    _, json_path = latest_paths(run_dir)
    return json.loads(json_path.read_text())


def validate_problem(run_dir: Path, problem: dict[str, Any]) -> None:
    """Ensure a resumed run uses the same parameter point."""
    latest = load_latest_json(run_dir)
    if latest["problem"] != problem:
        raise ValueError(
            f"Run configuration does not match existing checkpoint in {run_dir}."
        )


def maybe_resume(
    run_dir: Path,
    *,
    problem: dict[str, Any],
    driver: TDVPDriver,
    resume: bool,
    label: str,
) -> None:
    """Resume one driver from an existing checkpoint if requested."""
    if not resume:
        return
    _, json_path = latest_paths(run_dir)
    if json_path.exists():
        validate_problem(run_dir, problem)
    restore_latest(run_dir, driver)
    print(
        f"[{label}] resumed at step={driver.step_count} tau={driver.t:.6f}",
        flush=True,
    )


def save_latest(
    run_dir: Path,
    *,
    driver: TDVPDriver,
    problem: dict[str, Any],
    latest_metrics: dict[str, Any],
) -> None:
    """Save the latest exact-resume state and human-readable metadata."""
    npz_path, json_path = latest_paths(run_dir)
    tensor_arrays = {
        f"tensor_{row}_{col}": np.asarray(tensor)
        for row, tensors in driver._tensors.items()
        for col, tensor in tensors.items()
    }
    npz_tmp_path = npz_path.with_name(f"{npz_path.name}.tmp")
    with npz_tmp_path.open("wb") as handle:
        np.savez(
            handle,
            step_count=np.asarray(driver.step_count, dtype=np.int64),
            imaginary_time=np.asarray(float(driver.t), dtype=np.float64),
            sampler_key=np.asarray(jax.random.key_data(driver._sampler_key)),
            sampler_key_impl=np.asarray(str(jax.random.key_impl(driver._sampler_key))),
            sampler_configuration=np.asarray(driver._sampler_configuration),
            **tensor_arrays,
        )
    npz_tmp_path.replace(npz_path)

    json_tmp_path = json_path.with_name(f"{json_path.name}.tmp")
    clean_latest_metrics = {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in latest_metrics.items()
    }
    json_tmp_path.write_text(
        json.dumps(
            {
                "problem": problem,
                "progress": {
                    "completed_steps": int(driver.step_count),
                    "imaginary_time": float(driver.t),
                },
                "latest_metrics": clean_latest_metrics,
            },
            indent=2,
            allow_nan=False,
        )
    )
    json_tmp_path.replace(json_path)


def restore_latest(run_dir: Path, driver: TDVPDriver) -> None:
    """Restore one exact-resume checkpoint into a fresh driver."""
    npz_path, _ = latest_paths(run_dir)
    with np.load(npz_path, allow_pickle=False) as data:
        saved_config = jnp.asarray(data["sampler_configuration"])
        if int(saved_config.shape[0]) != driver.n_chains:
            raise ValueError(
                f"Checkpoint n_chains={saved_config.shape[0]} does not match driver.n_chains={driver.n_chains}."
            )
        driver._tensors = {
            row: {
                col: jnp.asarray(data[f"tensor_{row}_{col}"])
                for col in tensors
            }
            for row, tensors in driver._tensors.items()
        }
        driver._sampler_configuration = saved_config
        driver._sampler_key = jax.random.wrap_key_data(
            jnp.asarray(data["sampler_key"], dtype=jnp.uint32),
            impl=data["sampler_key_impl"].item(),
        )
        driver.step_count = int(data["step_count"])
        driver.t = float(data["imaginary_time"])


def ground_state_metrics(
    driver: TDVPDriver,
    *,
    energy_scale: float,
    step_wall_time: float,
) -> dict[str, float]:
    """Extract the common ground-state metrics for one completed step."""
    energy = driver.energy
    metrics = driver.metrics
    fs_norm_squared = float(metrics["FS_norm_squared"])
    return {
        "completed_steps": int(driver.step_count),
        "imaginary_time": float(driver.t),
        "step_wall_time": float(step_wall_time),
        "energy_mean": float(energy.mean.real) / energy_scale,
        "energy_error": float(energy.error_of_mean.real) / energy_scale,
        "energy_variance": float(energy.variance.real) / energy_scale**2,
        "applied_FS_step_norm_squared": float(driver.dt**2 * fs_norm_squared),
        "FS_norm_squared": fs_norm_squared,
        "TDVP_residual": float(metrics["TDVP_residual"]),
        "SR_solve_residual": float(metrics["SR_solve_residual"]),
    }


def run_ground_state_steps(
    *,
    label: str,
    driver: TDVPDriver,
    run_dir: Path,
    problem: dict[str, Any],
    n_steps: int,
    log_every: int,
    save_every: int,
    energy_scale: float,
    update_row: Callable[[TDVPDriver, dict[str, float]], None] | None = None,
    format_extra: Callable[[dict[str, float]], str] | None = None,
) -> None:
    """Run one ground-state trajectory to the requested target step count."""
    remaining_steps = max(0, n_steps - driver.step_count)
    if remaining_steps == 0:
        return
    extra_header = "" if format_extra is None else format_extra({})
    if extra_header:
        extra_header = f" {extra_header}"
    print(
        "[{label}] step tau dt wall_time energy_per_site energy_err energy_var"
        f"{extra_header} applied_FS_step_norm_squared FS_norm_squared "
        "TDVP_residual SR_solve_residual".format(label=label),
        flush=True,
    )
    for local_step in range(1, remaining_steps + 1):
        driver.run(driver.dt)
        row = ground_state_metrics(
            driver,
            energy_scale=energy_scale,
            step_wall_time=float(driver.metrics["step_wall_time"]),
        )
        if update_row is not None:
            update_row(driver, row)
        extra_values = "" if format_extra is None else format_extra(row)
        if driver.step_count % log_every == 0 or local_step == remaining_steps:
            print(
                (
                    f"[{label}] {row['completed_steps']:4d} {row['imaginary_time']:.6f} "
                    f"{driver.dt:.6f} {row['step_wall_time']:.6f} "
                    f"{row['energy_mean']:.10f} {row['energy_error']:.6e} "
                    f"{row['energy_variance']:.6e}"
                    f"{'' if not extra_values else ' ' + extra_values} "
                    f"{row['applied_FS_step_norm_squared']:.6e} "
                    f"{row['FS_norm_squared']:.6e} "
                    f"{row['TDVP_residual']:.6e} "
                    f"{row['SR_solve_residual']:.6e}"
                ),
                flush=True,
            )
        if driver.step_count % save_every == 0 or local_step == remaining_steps:
            save_latest(run_dir, driver=driver, problem=problem, latest_metrics=row)
            print(f"Saved {run_dir / 'latest.json'}", flush=True)


def _site_in_box(row: int, col: int, row0: int, row1: int, col0: int, col1: int) -> bool:
    return row0 <= row < row1 and col0 <= col < col1


def build_central_bulk_observable(
    shape: tuple[int, int],
    *,
    h: float,
    g: float,
    J: float,
    m: float,
    bulk_size: int,
) -> GILocalHamiltonian:
    """Build the central-bulk energy observable used in the finite-size analysis."""
    n_rows, n_cols = shape
    if bulk_size > min(shape):
        raise ValueError(f"bulk_size={bulk_size} exceeds lattice shape {shape}.")
    row0 = (n_rows - bulk_size) // 2
    col0 = (n_cols - bulk_size) // 2
    row1 = row0 + bulk_size
    col1 = col0 + bulk_size

    terms = []
    for term in build_z2_hardcore_boson_hamiltonian(shape, h=h, g=g, J=J, m=m).terms:
        if isinstance(term, MatterMassTerm):
            row, col = term.sites[0]
            if _site_in_box(row, col, row0, row1, col0, col1):
                terms.append(term)
        elif isinstance(term, HorizontalMatterHoppingTerm):
            if (
                _site_in_box(term.row, term.col, row0, row1, col0, col1)
                and _site_in_box(term.row, term.col + 1, row0, row1, col0, col1)
            ):
                terms.append(term)
        elif isinstance(term, VerticalMatterHoppingTerm):
            if (
                _site_in_box(term.row, term.col, row0, row1, col0, col1)
                and _site_in_box(term.row + 1, term.col, row0, row1, col0, col1)
            ):
                terms.append(term)
        elif isinstance(term, PlaquetteOperator):
            if (
                _site_in_box(term.row, term.col, row0, row1, col0, col1)
                and _site_in_box(term.row + 1, term.col + 1, row0, row1, col0, col1)
            ):
                terms.append(term)
        elif isinstance(term, LinkDiagonalTerm):
            if term.orientation == "h":
                keep = (
                    _site_in_box(term.sites[0][0], term.sites[0][1], row0, row1, col0, col1)
                    and _site_in_box(
                        term.sites[0][0], term.sites[0][1] + 1, row0, row1, col0, col1
                    )
                )
            else:
                keep = (
                    _site_in_box(term.sites[0][0], term.sites[0][1], row0, row1, col0, col1)
                    and _site_in_box(
                        term.sites[0][0] + 1, term.sites[0][1], row0, row1, col0, col1
                    )
                )
            if keep:
                terms.append(term)
        else:
            raise TypeError(f"Unsupported term type: {type(term)!r}")
    return GILocalHamiltonian(shape=shape, terms=tuple(terms))
