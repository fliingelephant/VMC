"""Resumable 10x10 Z2 vison propagation benchmark with GI-PEPS.

This example is designed for larger runs where checkpointing and resuming are
more important than exact comparison. It supports:

1. ``ground-state``: optimize a deconfined ground state and checkpoint it.
2. ``real-time``: start from a ground state or resume a real-time checkpoint.

Both subcommands overwrite a stable ``latest`` state file and update the JSON
log every ``save_every`` steps, so long runs can be continued safely.
"""

from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from netket import stats as nkstats

from vmc.drivers import ImaginaryTimeUnit, RK4, RealTimeUnit, TDVPDriver
from vmc.operators import PlaquetteOperator
from vmc.peps import GIPEPS, GIPEPSConfig, GILocalHamiltonian, Variational
from vmc.peps.gi.local_terms import build_electric_terms
from vmc.preconditioners import MetricsConfig, SRPreconditioner


EXAMPLE_DIR = Path(__file__).resolve().parent

DEFAULT_L = 10
DEFAULT_H = 1.0
DEFAULT_G = 0.1
DEFAULT_BOND_DIM = 4
DEFAULT_N_SAMPLES = 10240
DEFAULT_N_CHAINS = 1024
DEFAULT_N_STEPS_GS = 400
DEFAULT_DT_GS = 0.005
DEFAULT_GS_DIAG_SHIFT = 1e-4
DEFAULT_T = 18.0
DEFAULT_DT_RT = 0.005
DEFAULT_RT_DIAG_SHIFT = 1e-8
DEFAULT_SAVE_EVERY = 50
DEFAULT_SEED = 42

SR_METRICS_CONFIG = MetricsConfig(
    record_FS_norm=True,
    record_TDVP_residual=True,
    record_SR_solve_residual=True,
    record_step_wall_time=True,
)


def append_series(series: dict[str, list], **values) -> None:
    for key, value in values.items():
        series.setdefault(key, []).append(value)


def _token(value: float, digits: int = 3) -> str:
    return format(value, f".{digits}f").replace(".", "p")


def _run_stem(*, L: int, g: float, bond_dim: int) -> str:
    return f"z2_vison_propagation_L{L}_g{_token(g)}_Dk{bond_dim}"


def _default_ground_state_state_path(*, L: int, g: float, bond_dim: int) -> Path:
    return EXAMPLE_DIR / f"{_run_stem(L=L, g=g, bond_dim=bond_dim)}_ground_state_latest.npz"


def _default_real_time_state_path(*, L: int, g: float, bond_dim: int) -> Path:
    return EXAMPLE_DIR / f"{_run_stem(L=L, g=g, bond_dim=bond_dim)}_real_time_latest.npz"


def _normalize_state_path(path: Path) -> Path:
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def _matching_json_path(state_path: Path) -> Path:
    return _normalize_state_path(state_path).with_suffix(".json")


def build_z2_hamiltonian(
    shape: tuple[int, int],
    *,
    h: float,
    g: float,
) -> GILocalHamiltonian:
    n_rows, n_cols = shape
    plaquette_terms = tuple(
        PlaquetteOperator(row=row, col=col)
        for row in range(n_rows - 1)
        for col in range(n_cols - 1)
    )
    electric_terms = build_electric_terms(shape, N=2)
    return GILocalHamiltonian(
        shape=shape,
        terms=electric_terms + plaquette_terms,
        coeffs=(jnp.asarray(g),) * len(electric_terms)
        + (jnp.asarray(-h),) * len(plaquette_terms),
    )


def selected_open_plaquettes(shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    center = (shape[0] - 2) // 2
    return ((0, 0), (0, 1), (center, center))


def open_to_internal_plaquette(
    shape: tuple[int, int],
    row: int,
    col: int,
) -> tuple[int, int]:
    return shape[0] - 2 - row, col


def build_selected_plaquette_observables(
    shape: tuple[int, int],
    plaquettes: tuple[tuple[int, int], ...] | None = None,
) -> tuple[GILocalHamiltonian, ...]:
    if plaquettes is None:
        plaquettes = selected_open_plaquettes(shape)
    return tuple(
        GILocalHamiltonian(
            shape=shape,
            terms=(PlaquetteOperator(*open_to_internal_plaquette(shape, row, col)),),
            coeffs=(jnp.asarray(0.5),),
        )
        for row, col in plaquettes
    )


def build_model(
    shape: tuple[int, int],
    *,
    bond_dim: int,
    seed: int,
) -> GIPEPS:
    return GIPEPS(
        rngs=nnx.Rngs(seed),
        config=GIPEPSConfig(
            shape=shape,
            N=2,
            phys_dim=1,
            Qx=0,
            degeneracy_per_charge=(bond_dim, bond_dim),
            charge_of_site=(0,),
        ),
        contraction_strategy=Variational(truncate_bond_dimension=3 * bond_dim),
    )


def _site_independent_directions(
    shape: tuple[int, int],
    row: int,
    col: int,
) -> tuple[str, ...]:
    n_rows, n_cols = shape
    active = {
        "left": col > 0,
        "right": col < n_cols - 1,
        "up": row > 0,
        "down": row < n_rows - 1,
    }
    dependent = next(
        direction
        for direction in ("right", "down", "up", "left")
        if active[direction]
    )
    return tuple(
        direction
        for direction in ("left", "up", "down", "right")
        if active[direction] and direction != dependent
    )


def _z2_phase_for_direction(
    shape: tuple[int, int],
    row: int,
    col: int,
    direction: str,
) -> jax.Array:
    directions = _site_independent_directions(shape, row, col)
    if direction not in directions:
        raise ValueError(
            f"Direction {direction!r} is not independent at site {(row, col)}."
        )
    n_configs = 1 << len(directions)
    cfg_indices = jnp.arange(n_configs, dtype=jnp.int32)
    digit_index = directions.index(direction)
    divisor = 1 << (len(directions) - digit_index - 1)
    values = (cfg_indices // divisor) % 2
    return (1 - 2 * values).astype(jnp.complex128)


def create_bottom_left_vison(model: GIPEPS) -> GIPEPS:
    n_rows, n_cols = model.shape
    if n_rows < 2 or n_cols < 2:
        raise ValueError("The bottom-left vison construction requires L >= 2.")
    site_row = n_rows - 2
    site_col = 0
    phase = _z2_phase_for_direction(model.shape, site_row, site_col, "down")
    graphdef, params, model_state = nnx.split(model, nnx.Param, ...)
    tensors = nnx.to_pure_dict(params)["tensors"]
    tensors = {
        row: {col: jnp.asarray(tensor) for col, tensor in row_dict.items()}
        for row, row_dict in tensors.items()
    }
    tensors[site_row][site_col] = (
        tensors[site_row][site_col] * phase[None, :, None, None, None, None]
    )
    return nnx.merge(graphdef, {"tensors": tensors}, model_state)


def save_model_state(model: GIPEPS, metadata: dict, output_path: Path) -> None:
    output_path = _normalize_state_path(output_path)
    _, params, model_state = nnx.split(model, nnx.Param, ...)
    if nnx.to_pure_dict(model_state):
        raise ValueError("Expected an empty non-parameter GIPEPS state.")
    tensors = nnx.to_pure_dict(params)["tensors"]
    arrays = {"metadata_json": np.asarray(json.dumps(metadata))}
    for row, row_dict in tensors.items():
        for col, tensor in row_dict.items():
            arrays[f"tensor_{row}_{col}"] = np.asarray(tensor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    print(f"Saved {output_path}", flush=True)


def load_model_state(input_path: Path) -> tuple[GIPEPS, dict]:
    input_path = _normalize_state_path(input_path)
    with np.load(input_path) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        shape = tuple(int(x) for x in metadata["shape"])
        bond_dim = int(metadata["bond_dim"])
        model = build_model(shape, bond_dim=bond_dim, seed=int(metadata["seed"]))
        graphdef, _, model_state = nnx.split(model, nnx.Param, ...)
        tensors = {
            row: {
                col: jnp.asarray(data[f"tensor_{row}_{col}"])
                for col in range(shape[1])
            }
            for row in range(shape[0])
        }
    return nnx.merge(graphdef, {"tensors": tensors}, model_state), metadata


def _measure_driver(driver: TDVPDriver) -> tuple[object, tuple[object, ...]]:
    config_states = driver._sampler_configuration.reshape(driver.n_chains, -1)
    _, (key, config_states), (local_estimates, _) = driver._time_derivative(
        driver._tensors,
        driver.t,
        (driver._sampler_key, config_states),
    )
    driver._sampler_key = key
    driver._sampler_configuration = config_states
    energy = nkstats.statistics(local_estimates[:, 0])
    observables = tuple(
        nkstats.statistics(local_estimates[:, idx])
        for idx in range(1, local_estimates.shape[1])
    )
    return energy, observables


def _load_existing_series(json_path: Path) -> dict[str, list]:
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text()).get("series", {})


def _validate_resume_series(
    series: dict[str, list],
    *,
    step: int,
    time: float,
    time_key: str,
) -> None:
    if not series:
        return
    if int(series["step"][-1]) != step:
        raise ValueError(
            f"Series last step {series['step'][-1]} does not match state step {step}."
        )
    if abs(float(series[time_key][-1]) - time) > 1e-12 * max(1.0, abs(time)):
        raise ValueError(
            f"Series last {time_key} {series[time_key][-1]} does not match state time {time}."
        )


def _save_run_json(result: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {output_path}", flush=True)


def run_ground_state(
    model: GIPEPS,
    hamiltonian: GILocalHamiltonian,
    *,
    n_samples: int,
    n_chains: int,
    n_steps: int,
    dt: float,
    diag_shift: float,
    seed: int,
    initial_step: int,
    initial_time: float,
    series: dict[str, list],
    checkpoint_every: int,
    checkpoint_fn,
) -> None:
    driver = TDVPDriver(
        model,
        hamiltonian,
        preconditioner=SRPreconditioner(
            diag_shift=diag_shift,
            metrics_config=SR_METRICS_CONFIG,
        ),
        dt=dt,
        t0=initial_time,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(seed),
        n_samples=n_samples,
        n_chains=n_chains,
        full_gradient=False,
    )
    print(
        "[ground_state] step t dt wall_time energy energy_err energy_var "
        "applied_FS_step_norm_squared FS_norm_squared TDVP_residual "
        "SR_solve_residual",
        flush=True,
    )
    final_step = initial_step + n_steps
    for step in range(initial_step + 1, final_step + 1):
        driver.run(dt)
        metrics = driver.metrics
        energy = driver.energy
        fs_norm_squared = float(metrics["FS_norm_squared"])
        row = {
            "step": step,
            "imaginary_time": float(driver.t),
            "dt": dt,
            "step_wall_time": float(metrics["step_wall_time"]),
            "energy_mean": float(energy.mean.real),
            "energy_error": float(energy.error_of_mean.real),
            "energy_variance": float(energy.variance.real),
            "applied_FS_step_norm_squared": dt**2 * fs_norm_squared,
            "FS_norm_squared": fs_norm_squared,
            "TDVP_residual": float(metrics["TDVP_residual"]),
            "SR_solve_residual": float(metrics["SR_solve_residual"]),
        }
        append_series(series, **row)
        print(
            (
                f"[ground_state] {row['step']:4d} {row['imaginary_time']:.6f} "
                f"{row['dt']:.6f} {row['step_wall_time']:.3f} "
                f"{row['energy_mean']:.10f} {row['energy_error']:.6f} "
                f"{row['energy_variance']:.6f} "
                f"{row['applied_FS_step_norm_squared']:.6e} "
                f"{row['FS_norm_squared']:.6e} "
                f"{row['TDVP_residual']:.6e} "
                f"{row['SR_solve_residual']:.6e}"
            ),
            flush=True,
        )
        if step % checkpoint_every == 0 or step == final_step:
            checkpoint_fn(driver.model, step, float(driver.t), series)


def run_real_time(
    model: GIPEPS,
    hamiltonian: GILocalHamiltonian,
    *,
    n_samples: int,
    n_chains: int,
    T: float,
    dt: float,
    diag_shift: float,
    seed: int,
    initial_step: int,
    initial_time: float,
    series: dict[str, list],
    checkpoint_every: int,
    checkpoint_fn,
) -> None:
    observables = build_selected_plaquette_observables(model.shape)
    driver = TDVPDriver(
        model,
        hamiltonian,
        observables=observables,
        preconditioner=SRPreconditioner(
            diag_shift=diag_shift,
            metrics_config=SR_METRICS_CONFIG,
        ),
        dt=dt,
        t0=initial_time,
        time_unit=RealTimeUnit(),
        integrator=RK4(),
        sampler_key=jax.random.key(seed),
        n_samples=n_samples,
        n_chains=n_chains,
        full_gradient=False,
    )
    remaining_time = T - initial_time
    if remaining_time < -1e-12 * max(1.0, abs(T), abs(initial_time)):
        raise ValueError(
            f"Target final time T={T} is smaller than current checkpoint time {initial_time}."
        )
    n_steps = int(round(remaining_time / dt))
    if abs(remaining_time - n_steps * dt) > 1e-12 * max(1.0, abs(remaining_time), abs(dt)):
        raise ValueError(
            f"Remaining time {remaining_time} must be an integer multiple of dt={dt}."
        )

    print(
        "[real_time] step t dt wall_time energy energy_err energy_var "
        "drift_percent p00 p00_err p01 p01_err pcc pcc_err "
        "applied_FS_step_norm_squared FS_norm_squared TDVP_residual "
        "SR_solve_residual",
        flush=True,
    )

    if not series:
        energy, observable_stats = _measure_driver(driver)
        reference_energy = float(energy.mean.real)
        selected_means = [float(stat.mean.real) for stat in observable_stats]
        selected_errors = [
            float(stat.error_of_mean.real) for stat in observable_stats
        ]
        append_series(
            series,
            step=initial_step,
            time=initial_time,
            energy_mean=reference_energy,
            energy_error=float(energy.error_of_mean.real),
            energy_variance=float(energy.variance.real),
            energy_drift_percent=0.0,
            selected_plaquette_mean=selected_means,
            selected_plaquette_error=selected_errors,
            step_wall_time=None,
            applied_FS_step_norm_squared=None,
            FS_norm_squared=None,
            TDVP_residual=None,
            SR_solve_residual=None,
        )
        checkpoint_fn(driver.model, initial_step, initial_time, series)
    else:
        reference_energy = float(series["energy_mean"][0])

    final_step = initial_step + n_steps
    for step in range(initial_step + 1, final_step + 1):
        driver.run(dt)
        metrics = driver.metrics
        energy = driver.energy
        energy_mean = float(energy.mean.real)
        selected_means = [float(stat.mean.real) for stat in driver.observable_stats]
        selected_errors = [
            float(stat.error_of_mean.real) for stat in driver.observable_stats
        ]
        fs_norm_squared = float(metrics["FS_norm_squared"])
        row = {
            "step": step,
            "time": float(driver.t),
            "energy_mean": energy_mean,
            "energy_error": float(energy.error_of_mean.real),
            "energy_variance": float(energy.variance.real),
            "energy_drift_percent": abs(energy_mean - reference_energy)
            / abs(reference_energy)
            * 100.0,
            "selected_plaquette_mean": selected_means,
            "selected_plaquette_error": selected_errors,
            "step_wall_time": float(metrics["step_wall_time"]),
            "applied_FS_step_norm_squared": dt**2 * fs_norm_squared,
            "FS_norm_squared": fs_norm_squared,
            "TDVP_residual": float(metrics["TDVP_residual"]),
            "SR_solve_residual": float(metrics["SR_solve_residual"]),
        }
        append_series(series, **row)
        print(
            (
                f"[real_time] {row['step']:4d} {row['time']:.6f} {dt:.6f} "
                f"{row['step_wall_time']:.3f} {row['energy_mean']:.10f} "
                f"{row['energy_error']:.6f} {row['energy_variance']:.6f} "
                f"{row['energy_drift_percent']:.6f} "
                f"{selected_means[0]:.10f} {selected_errors[0]:.6f} "
                f"{selected_means[1]:.10f} {selected_errors[1]:.6f} "
                f"{selected_means[2]:.10f} {selected_errors[2]:.6f} "
                f"{row['applied_FS_step_norm_squared']:.6e} "
                f"{row['FS_norm_squared']:.6e} "
                f"{row['TDVP_residual']:.6e} "
                f"{row['SR_solve_residual']:.6e}"
            ),
            flush=True,
        )
        if step % checkpoint_every == 0 or step == final_step:
            checkpoint_fn(driver.model, step, float(driver.t), series)


def _build_problem_metadata(
    *,
    shape: tuple[int, int],
    h: float,
    g: float,
    bond_dim: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "gauge_group": "Z2",
        "shape": list(shape),
        "L": shape[0],
        "h": h,
        "g": g,
        "bond_dim": bond_dim,
        "boundary_method": "Variational",
        "boundary_dimension": 3 * bond_dim,
        "seed": seed,
    }


def _ground_state_result(
    *,
    problem: dict[str, Any],
    state_path: Path,
    n_samples: int,
    n_chains: int,
    dt: float,
    diag_shift: float,
    checkpoint_every: int,
    resume_state: Path | None,
    series: dict[str, list],
) -> dict[str, Any]:
    return {
        "problem": problem,
        "stage": "ground_state",
        "state_path": str(state_path),
        "config": {
            "n_samples": n_samples,
            "n_chains": n_chains,
            "dt": dt,
            "diag_shift": diag_shift,
            "checkpoint_every": checkpoint_every,
            "resume_state": None if resume_state is None else str(resume_state),
        },
        "series": series,
        "summary": {
            "final_step": series["step"][-1] if series else 0,
            "final_imaginary_time": series["imaginary_time"][-1] if series else 0.0,
            "final_energy_mean": series["energy_mean"][-1] if series else None,
            "final_energy_error": series["energy_error"][-1] if series else None,
            "final_energy_variance": series["energy_variance"][-1] if series else None,
        },
    }


def _real_time_result(
    *,
    problem: dict[str, Any],
    state_path: Path,
    n_samples: int,
    n_chains: int,
    T: float,
    dt: float,
    diag_shift: float,
    checkpoint_every: int,
    input_state: Path,
    series: dict[str, list],
) -> dict[str, Any]:
    shape = tuple(problem["shape"])
    selected_open = [list(site) for site in selected_open_plaquettes(shape)]
    measured_internal = [
        list(open_to_internal_plaquette(shape, row, col))
        for row, col in selected_open_plaquettes(shape)
    ]
    return {
        "problem": problem,
        "stage": "real_time",
        "state_path": str(state_path),
        "config": {
            "n_samples": n_samples,
            "n_chains": n_chains,
            "T": T,
            "dt": dt,
            "diag_shift": diag_shift,
            "checkpoint_every": checkpoint_every,
            "input_state": str(input_state),
        },
        "selected_open_plaquettes": selected_open,
        "measured_internal_plaquettes": measured_internal,
        "vison": {
            "operator": "sigma_z",
            "orientation": "v",
            "link_row": shape[0] - 2,
            "link_col": 0,
            "plaquettes": [[shape[0] - 2, 0]],
        },
        "series": series,
        "summary": {
            "final_step": series["step"][-1] if series else 0,
            "final_time": series["time"][-1] if series else 0.0,
            "final_energy_mean": series["energy_mean"][-1] if series else None,
            "final_energy_error": series["energy_error"][-1] if series else None,
            "final_energy_variance": series["energy_variance"][-1] if series else None,
            "final_energy_drift_percent": (
                series["energy_drift_percent"][-1] if series else None
            ),
        },
    }


def _run_ground_state_command(args: argparse.Namespace) -> None:
    resume_state = (
        None if args.resume_state is None else _normalize_state_path(args.resume_state)
    )
    if resume_state is not None:
        model, metadata = load_model_state(resume_state)
        if metadata.get("stage") != "ground_state":
            raise ValueError("Ground-state resume requires a ground-state checkpoint.")
        shape = tuple(int(x) for x in metadata["shape"])
        problem = _build_problem_metadata(
            shape=shape,
            h=float(metadata["h"]),
            g=float(metadata["g"]),
            bond_dim=int(metadata["bond_dim"]),
            seed=int(metadata["seed"]),
        )
        state_path = _normalize_state_path(args.state_output or resume_state)
        json_path = args.json_output or _matching_json_path(state_path)
        initial_step = int(metadata.get("step", 0))
        initial_time = float(metadata.get("time", 0.0))
        series = _load_existing_series(json_path)
        _validate_resume_series(
            series,
            step=initial_step,
            time=initial_time,
            time_key="imaginary_time",
        )
        seed = args.seed if args.seed is not None else int(metadata["seed"])
    else:
        shape = (args.L, args.L)
        model = build_model(shape, bond_dim=args.bond_dim, seed=args.seed)
        problem = _build_problem_metadata(
            shape=shape,
            h=args.h,
            g=args.g,
            bond_dim=args.bond_dim,
            seed=args.seed,
        )
        state_path = _normalize_state_path(
            args.state_output
            or _default_ground_state_state_path(
                L=args.L,
                g=args.g,
                bond_dim=args.bond_dim,
            )
        )
        json_path = args.json_output or _matching_json_path(state_path)
        initial_step = 0
        initial_time = 0.0
        series = {}
        seed = args.seed

    hamiltonian = build_z2_hamiltonian(
        shape,
        h=float(problem["h"]),
        g=float(problem["g"]),
    )

    def checkpoint(model_state: GIPEPS, step: int, time: float, series_data: dict[str, list]) -> None:
        save_model_state(
            model_state,
            {
                **problem,
                "stage": "ground_state",
                "step": step,
                "time": time,
            },
            state_path,
        )
        _save_run_json(
            _ground_state_result(
                problem=problem,
                state_path=state_path,
                n_samples=args.n_samples,
                n_chains=args.n_chains,
                dt=args.dt,
                diag_shift=args.diag_shift,
                checkpoint_every=args.save_every,
                resume_state=resume_state,
                series=series_data,
            ),
            json_path,
        )

    run_ground_state(
        model,
        hamiltonian,
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        n_steps=args.n_steps,
        dt=args.dt,
        diag_shift=args.diag_shift,
        seed=seed,
        initial_step=initial_step,
        initial_time=initial_time,
        series=series,
        checkpoint_every=args.save_every,
        checkpoint_fn=checkpoint,
    )


def _run_real_time_command(args: argparse.Namespace) -> None:
    input_state = _normalize_state_path(args.state)
    model, metadata = load_model_state(input_state)
    shape = tuple(int(x) for x in metadata["shape"])
    problem = _build_problem_metadata(
        shape=shape,
        h=float(metadata["h"]),
        g=float(metadata["g"]),
        bond_dim=int(metadata["bond_dim"]),
        seed=int(metadata["seed"]),
    )
    stage = metadata.get("stage")
    if stage == "ground_state":
        model = create_bottom_left_vison(model)
        initial_step = 0
        initial_time = 0.0
        state_path = _normalize_state_path(args.state_output or _default_real_time_state_path(
            L=problem["L"],
            g=float(problem["g"]),
            bond_dim=int(problem["bond_dim"]),
        ))
        json_path = args.json_output or _matching_json_path(state_path)
        series: dict[str, list] = {}
    elif stage == "real_time":
        initial_step = int(metadata.get("step", 0))
        initial_time = float(metadata.get("time", 0.0))
        state_path = _normalize_state_path(args.state_output or input_state)
        json_path = args.json_output or _matching_json_path(state_path)
        series = _load_existing_series(json_path)
        _validate_resume_series(
            series,
            step=initial_step,
            time=initial_time,
            time_key="time",
        )
    else:
        raise ValueError("Real-time run requires a ground-state or real-time checkpoint.")

    hamiltonian = build_z2_hamiltonian(
        shape,
        h=float(problem["h"]),
        g=float(problem["g"]),
    )
    seed = args.seed if args.seed is not None else int(problem["seed"]) + 1

    def checkpoint(model_state: GIPEPS, step: int, time: float, series_data: dict[str, list]) -> None:
        save_model_state(
            model_state,
            {
                **problem,
                "stage": "real_time",
                "step": step,
                "time": time,
            },
            state_path,
        )
        _save_run_json(
            _real_time_result(
                problem=problem,
                state_path=state_path,
                n_samples=args.n_samples,
                n_chains=args.n_chains,
                T=args.T,
                dt=args.dt,
                diag_shift=args.diag_shift,
                checkpoint_every=args.save_every,
                input_state=input_state,
                series=series_data,
            ),
            json_path,
        )

    run_real_time(
        model,
        hamiltonian,
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        T=args.T,
        dt=args.dt,
        diag_shift=args.diag_shift,
        seed=seed,
        initial_step=initial_step,
        initial_time=initial_time,
        series=series,
        checkpoint_every=args.save_every,
        checkpoint_fn=checkpoint,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable 10x10 Z2 vison propagation benchmark with checkpointed latest state/json outputs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ground_state = subparsers.add_parser(
        "ground-state",
        help="Optimize and checkpoint the 10x10 Z2 ground state.",
    )
    ground_state.add_argument("--L", type=int, default=DEFAULT_L)
    ground_state.add_argument("--h", type=float, default=DEFAULT_H)
    ground_state.add_argument("--g", type=float, default=DEFAULT_G)
    ground_state.add_argument("--bond-dim", type=int, default=DEFAULT_BOND_DIM)
    ground_state.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    ground_state.add_argument("--n-chains", type=int, default=DEFAULT_N_CHAINS)
    ground_state.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS_GS)
    ground_state.add_argument("--dt", type=float, default=DEFAULT_DT_GS)
    ground_state.add_argument("--diag-shift", type=float, default=DEFAULT_GS_DIAG_SHIFT)
    ground_state.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    ground_state.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ground_state.add_argument("--resume-state", type=Path, default=None)
    ground_state.add_argument("--state-output", type=Path, default=None)
    ground_state.add_argument("--json-output", type=Path, default=None)
    ground_state.set_defaults(run=_run_ground_state_command)

    real_time = subparsers.add_parser(
        "real-time",
        help="Start or resume 10x10 real-time vison dynamics from a checkpoint.",
    )
    real_time.add_argument("--state", type=Path, required=True)
    real_time.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    real_time.add_argument("--n-chains", type=int, default=DEFAULT_N_CHAINS)
    real_time.add_argument("--T", type=float, default=DEFAULT_T)
    real_time.add_argument("--dt", type=float, default=DEFAULT_DT_RT)
    real_time.add_argument("--diag-shift", type=float, default=DEFAULT_RT_DIAG_SHIFT)
    real_time.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    real_time.add_argument("--seed", type=int, default=None)
    real_time.add_argument("--state-output", type=Path, default=None)
    real_time.add_argument("--json-output", type=Path, default=None)
    real_time.set_defaults(run=_run_real_time_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
