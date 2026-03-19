"""Resumable Z2 vison confinement dynamics with Higgs terms.

This example targets the Z2 gauge theory with Higgs field discussed in
Wu and Nys (2026), comparing deconfined and Higgs-regime vison dynamics.

The paper writes the Hamiltonian as

    H = -sum_i sigma_i^z - sum_p B_p - J sum_l sigma^-_l X_l sigma^+_l - g sum_l Z_l .

The GIPEPS implementation in this repo uses a binary occupancy basis
`n in {0, 1}` with charge `(0, 1)` on matter sites and the diagonal Z2
gauge term `2 - 2 Z_l`. In that basis:

    sigma^z = 1 - 2 n
    -sigma^z = 2 n - 1
    -g Z_l = 0.5 * g * (2 - 2 Z_l) - g

so we implement the paper Hamiltonian up to additive constants by using:

    + 2 * n
    + 0.5 * g * (2 - 2 Z_l)
    - J * sigma_x X sigma_x
    - h * B_p

The script supports two stages:

1. ``ground-state``: optimize a parity-sector Z2 Higgs ground state with exact
   checkpoint/resume support.
2. ``real-time``: create an interior vison pair on a saved ground state or
   resume a previous real-time checkpoint.

Real-time resumption uses a target final time ``--T-final``. If a checkpoint is
already at time ``t_ckpt``, the script advances only the remaining interval
``T_final - t_ckpt``.
"""

from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import argparse
import json
import logging
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from netket import stats as nkstats

from vmc.drivers import ImaginaryTimeUnit, RK4, RealTimeUnit, TDVPDriver
from vmc.operators import PlaquetteOperator
from vmc.peps import GIPEPS, GIPEPSConfig, GILocalHamiltonian, Variational
from vmc.peps.gi.local_terms import (
    HorizontalHiggsLinkTerm,
    MatterMassTerm,
    VerticalHiggsLinkTerm,
    build_electric_terms,
)
from vmc.preconditioners import MetricsConfig, SRPreconditioner
from vmc.qgt import ParameterSpace, SampleSpace


logger = logging.getLogger(__name__)

EXAMPLE_DIR = Path(__file__).resolve().parent
CHARGE_OF_SITE = (0, 1)

DEFAULT_L = 8
DEFAULT_H = 1.0
DEFAULT_G = 0.1
DEFAULT_J = 0.1
DEFAULT_SIGMA_Z_FIELD = 1.0
DEFAULT_BOND_DIM = 2
DEFAULT_BOUNDARY_SWEEPS = 2
DEFAULT_N_SAMPLES = 4096
DEFAULT_N_CHAINS = 512
DEFAULT_N_STEPS_GS = 200
DEFAULT_DT_GS = 0.005
DEFAULT_GS_DIAG_SHIFT = 1e-4
DEFAULT_T_FINAL = 8.0
DEFAULT_DT_RT = 0.005
DEFAULT_RT_DIAG_SHIFT = 1e-8
DEFAULT_LOG_EVERY = 10
DEFAULT_SAVE_EVERY = 20
DEFAULT_SEED = 42
DEFAULT_SOLVER_SPACE = "minsr"

SR_METRICS_CONFIG = MetricsConfig(
    record_FS_norm=True,
    record_TDVP_residual=True,
    record_SR_solve_residual=True,
    record_step_wall_time=True,
)


def _space_from_name(space_name: str) -> ParameterSpace | SampleSpace:
    """Convert one CLI space name into the corresponding QGT space object."""
    if space_name == "sr":
        return ParameterSpace()
    if space_name == "minsr":
        return SampleSpace()
    raise ValueError(f"Unsupported solver space {space_name!r}.")


def _checkpoint_solver_space(metadata: dict[str, Any]) -> str:
    """Return the solver space encoded in a checkpoint.

    This script requires the checkpoint to store the solver space explicitly so
    that resumes never depend on hidden legacy defaults.
    """
    if "solver_space" not in metadata:
        raise ValueError(
            "Checkpoint is missing 'solver_space'. "
            "Remove the checkpoint and start a fresh run with this script version."
        )
    return str(metadata["solver_space"])


class CheckpointState(NamedTuple):
    """Loaded model plus exact sampler state."""

    model: GIPEPS
    metadata: dict[str, Any]
    sampler_configuration: jax.Array
    sampler_key: jax.Array
    step_count: int
    time: float


def append_series(series: dict[str, list], **values) -> None:
    """Append one row into a columnar series dict."""
    for key, value in values.items():
        series.setdefault(key, []).append(value)


def _replace_or_append_snapshot(
    snapshots: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> None:
    """Replace the trailing snapshot if it has the same step, else append."""
    if snapshots and int(snapshots[-1]["step"]) == int(snapshot["step"]):
        snapshots[-1] = snapshot
    else:
        snapshots.append(snapshot)


def _token(value: int | float, digits: int = 3) -> str:
    """Format numeric values for filesystem-safe path components."""
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return format(float(value), f".{digits}f").replace(".", "p").replace("-", "m")


def _problem_stem(*, L: int, g: float, J: float, bond_dim: int) -> str:
    """Return the base path stem for one Higgs-vison problem."""
    return (
        f"z2_vison_higgs_confinement_L{L}_"
        f"g{_token(g)}_J{_token(J)}_Dk{bond_dim}"
    )


def _vison_stem(*, orientation: str, row: int, col: int) -> str:
    """Return the vison-link path suffix."""
    return f"vison_{orientation}_r{row}_c{col}"


def _normalize_state_path(path: Path) -> Path:
    """Ensure checkpoint files use the .npz suffix."""
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def _matching_json_path(state_path: Path) -> Path:
    """Return the JSON metadata path corresponding to one checkpoint."""
    return _normalize_state_path(state_path).with_suffix(".json")


def _default_ground_state_state_path(
    *,
    L: int,
    g: float,
    J: float,
    bond_dim: int,
) -> Path:
    """Return the default ground-state checkpoint path."""
    stem = _problem_stem(L=L, g=g, J=J, bond_dim=bond_dim)
    return EXAMPLE_DIR / f"{stem}_ground_state_latest.npz"


def _default_real_time_state_path(
    ground_state_path: Path,
    *,
    orientation: str,
    row: int,
    col: int,
) -> Path:
    """Return the default real-time checkpoint path derived from the ground state."""
    state_path = _normalize_state_path(ground_state_path)
    stem = state_path.stem.removesuffix("_ground_state_latest")
    return state_path.with_name(
        f"{stem}_real_time_{_vison_stem(orientation=orientation, row=row, col=col)}_latest.npz"
    )


def build_problem_metadata(
    *,
    shape: tuple[int, int],
    h: float,
    g: float,
    J: float,
    sigma_z_field: float,
    bond_dim: int,
    boundary_dim: int,
    boundary_sweeps: int,
    seed: int,
    solver_space: str,
) -> dict[str, Any]:
    """Return the serializable problem metadata stored in checkpoints."""
    return {
        "model": "z2_vison_higgs_confinement",
        "gauge_group": "Z2",
        "matter_sector": "parity_only",
        "shape": list(shape),
        "L": int(shape[0]),
        "h": float(h),
        "g": float(g),
        "J": float(J),
        "sigma_z_field": float(sigma_z_field),
        "bond_dim": int(bond_dim),
        "boundary_dim": int(boundary_dim),
        "boundary_sweeps": int(boundary_sweeps),
        "seed": int(seed),
        "solver_space": str(solver_space),
        "paper_mapping": {
            "matter_term": "-sigma_z -> 2*n - 1",
            "electric_term": "-g*Z -> 0.5*g*(2-2Z) - g",
        },
    }


def build_model(
    shape: tuple[int, int],
    *,
    bond_dim: int,
    boundary_dim: int,
    boundary_sweeps: int,
    seed: int,
) -> GIPEPS:
    """Build the parity-sector Z2 GIPEPS used by the Higgs example."""
    return GIPEPS(
        rngs=nnx.Rngs(seed),
        config=GIPEPSConfig(
            shape=shape,
            N=2,
            phys_dim=2,
            Qx=0,
            degeneracy_per_charge=(bond_dim, bond_dim),
            charge_of_site=CHARGE_OF_SITE,
            conserve_particle_number=False,
            particle_number=None,
        ),
        contraction_strategy=Variational(boundary_dim, n_sweeps=boundary_sweeps),
    )


def build_z2_higgs_hamiltonian(
    shape: tuple[int, int],
    *,
    h: float,
    g: float,
    J: float,
    sigma_z_field: float,
) -> GILocalHamiltonian:
    """Build the Z2 gauge-theory Hamiltonian with Higgs-link terms.

    The paper couplings are implemented up to additive constants, as explained in
    the module docstring.
    """
    n_rows, n_cols = shape
    plaquette_terms = tuple(
        PlaquetteOperator(row=row, col=col)
        for row in range(n_rows - 1)
        for col in range(n_cols - 1)
    )
    electric_terms = build_electric_terms(shape, N=2)
    matter_terms = tuple(
        MatterMassTerm(row=row, col=col, charge_of_site=CHARGE_OF_SITE)
        for row in range(n_rows)
        for col in range(n_cols)
    )
    horizontal_higgs_terms = tuple(
        HorizontalHiggsLinkTerm(row=row, col=col)
        for row in range(n_rows)
        for col in range(n_cols - 1)
    )
    vertical_higgs_terms = tuple(
        VerticalHiggsLinkTerm(row=row, col=col)
        for row in range(n_rows - 1)
        for col in range(n_cols)
    )
    terms = (
        electric_terms
        + plaquette_terms
        + matter_terms
        + horizontal_higgs_terms
        + vertical_higgs_terms
    )
    electric_coeff = jnp.asarray(0.5 * g)
    plaquette_coeff = jnp.asarray(-h)
    matter_coeff = jnp.asarray(2.0 * sigma_z_field)
    higgs_coeff = jnp.asarray(-J)
    coeffs = (
        (electric_coeff,) * len(electric_terms)
        + (plaquette_coeff,) * len(plaquette_terms)
        + (matter_coeff,) * len(matter_terms)
        + (higgs_coeff,) * len(horizontal_higgs_terms)
        + (higgs_coeff,) * len(vertical_higgs_terms)
    )
    return GILocalHamiltonian(shape=shape, terms=terms, coeffs=coeffs)


def build_all_plaquette_observables(
    shape: tuple[int, int],
) -> tuple[GILocalHamiltonian, ...]:
    """Build one observable per plaquette for map snapshots.

    ``PlaquetteOperator`` evaluates ``P + P†``. For Z2, ``P = P†``, so a
    coefficient of ``0.5`` yields the plaquette expectation value itself.
    """
    n_rows, n_cols = shape
    return tuple(
        GILocalHamiltonian(
            shape=shape,
            terms=(PlaquetteOperator(row=row, col=col),),
            coeffs=(jnp.asarray(0.5),),
        )
        for row in range(n_rows - 1)
        for col in range(n_cols - 1)
    )


def _site_independent_directions(
    shape: tuple[int, int],
    row: int,
    col: int,
) -> tuple[str, ...]:
    """Return the locally independent link directions on one GIPEPS site."""
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
    """Return the sigma_z phase on the site's Nc slices for one link direction."""
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


def default_vison_link(
    shape: tuple[int, int],
    orientation: str,
) -> tuple[int, int]:
    """Return a central interior link for creating the default vison pair."""
    n_rows, n_cols = shape
    if min(shape) < 4:
        raise ValueError("Interior vison-pair insertion requires L >= 4.")
    if orientation == "v":
        return (n_rows - 2) // 2, n_cols // 2
    if orientation == "h":
        return n_rows // 2, (n_cols - 2) // 2
    raise ValueError(f"Unsupported orientation {orientation!r}.")


def vison_pair_plaquettes(
    shape: tuple[int, int],
    *,
    orientation: str,
    row: int,
    col: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the two plaquettes excited by one interior link flip.

    Plaquette coordinates use the internal ``PlaquetteOperator`` convention:
    row-major with rows counted from the top.
    """
    n_rows, n_cols = shape
    if orientation == "v":
        if not (0 <= row < n_rows - 1 and 0 < col < n_cols - 1):
            raise ValueError(
                "Vertical vison-pair link must satisfy 0 <= row < L-1 and 0 < col < L-1."
            )
        return (row, col - 1), (row, col)
    if orientation == "h":
        if not (0 < row < n_rows - 1 and 0 <= col < n_cols - 1):
            raise ValueError(
                "Horizontal vison-pair link must satisfy 0 < row < L-1 and 0 <= col < L-1."
            )
        return (row - 1, col), (row, col)
    raise ValueError(f"Unsupported orientation {orientation!r}.")


def create_interior_vison_pair(
    model: GIPEPS,
    *,
    orientation: str,
    row: int,
    col: int,
) -> GIPEPS:
    """Act with sigma_z on one interior link to create a vison pair."""
    vison_pair_plaquettes(model.shape, orientation=orientation, row=row, col=col)
    graphdef, params, model_state = nnx.split(model, nnx.Param, ...)
    tensors = nnx.to_pure_dict(params)["tensors"]
    tensors = {
        site_row: {
            site_col: jnp.asarray(tensor)
            for site_col, tensor in row_dict.items()
        }
        for site_row, row_dict in tensors.items()
    }
    if orientation == "v":
        phase = _z2_phase_for_direction(model.shape, row, col, "down")
        tensors[row][col] = tensors[row][col] * phase[None, :, None, None, None, None]
    elif orientation == "h":
        phase = _z2_phase_for_direction(model.shape, row, col + 1, "left")
        tensors[row][col + 1] = (
            tensors[row][col + 1] * phase[None, :, None, None, None, None]
        )
    else:
        raise ValueError(f"Unsupported orientation {orientation!r}.")
    return nnx.merge(graphdef, {"tensors": tensors}, model_state)


def save_driver_checkpoint(
    driver: TDVPDriver,
    metadata: dict[str, Any],
    output_path: Path,
) -> None:
    """Save one exact-resume checkpoint including sampler state and time."""
    output_path = _normalize_state_path(output_path)
    _, params, model_state = nnx.split(driver.model, nnx.Param, ...)
    if nnx.to_pure_dict(model_state):
        raise ValueError("Expected an empty non-parameter GIPEPS state.")
    tensors = nnx.to_pure_dict(params)["tensors"]
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "sampler_configuration": np.asarray(driver._sampler_configuration),
        "sampler_key": np.asarray(jax.random.key_data(driver._sampler_key)),
        "sampler_key_impl": np.asarray(str(jax.random.key_impl(driver._sampler_key))),
        "step_count": np.asarray(driver.step_count, dtype=np.int64),
        "time": np.asarray(float(driver.t), dtype=np.float64),
    }
    for row, row_dict in tensors.items():
        for col, tensor in row_dict.items():
            arrays[f"tensor_{row}_{col}"] = np.asarray(tensor)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez(handle, **arrays)
    tmp_path.replace(output_path)
    logger.info("Saved %s", output_path)


def load_checkpoint_state(input_path: Path) -> CheckpointState:
    """Load one exact-resume checkpoint."""
    input_path = _normalize_state_path(input_path)
    with np.load(input_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        shape = tuple(int(x) for x in metadata["shape"])
        model = build_model(
            shape,
            bond_dim=int(metadata["bond_dim"]),
            boundary_dim=int(metadata["boundary_dim"]),
            boundary_sweeps=int(metadata["boundary_sweeps"]),
            seed=int(metadata["seed"]),
        )
        graphdef, _, model_state = nnx.split(model, nnx.Param, ...)
        tensors = {
            row: {
                col: jnp.asarray(data[f"tensor_{row}_{col}"])
                for col in range(shape[1])
            }
            for row in range(shape[0])
        }
        sampler_configuration = jnp.asarray(data["sampler_configuration"])
        sampler_key = jax.random.wrap_key_data(
            jnp.asarray(data["sampler_key"], dtype=jnp.uint32),
            impl=data["sampler_key_impl"].item(),
        )
        step_count = int(data["step_count"])
        time = float(data["time"])
    return CheckpointState(
        model=nnx.merge(graphdef, {"tensors": tensors}, model_state),
        metadata=metadata,
        sampler_configuration=sampler_configuration,
        sampler_key=sampler_key,
        step_count=step_count,
        time=time,
    )


def seed_driver_from_checkpoint(
    driver: TDVPDriver,
    checkpoint: CheckpointState,
    *,
    step_count: int,
    time: float,
) -> None:
    """Seed a fresh driver with the sampler state from one checkpoint."""
    if int(checkpoint.sampler_configuration.shape[0]) != driver.n_chains:
        raise ValueError(
            "Checkpoint n_chains does not match the requested n_chains for resume."
        )
    driver._sampler_configuration = checkpoint.sampler_configuration
    driver._sampler_key = checkpoint.sampler_key
    driver.step_count = int(step_count)
    driver.t = float(time)


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON from disk if it exists, else return an empty dict."""
    return json.loads(path.read_text()) if path.exists() else {}


def _load_resume_json(output_path: Path, fallback_path: Path | None = None) -> dict[str, Any]:
    """Load JSON from the intended output path or a fallback path."""
    if output_path.exists():
        return _load_json(output_path)
    if fallback_path is not None and fallback_path.exists():
        return _load_json(fallback_path)
    return {}


def _save_json(result: dict[str, Any], output_path: Path) -> None:
    """Write one JSON result file atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    tmp_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    tmp_path.replace(output_path)
    logger.info("Saved %s", output_path)


def _validate_series(
    series: dict[str, list],
    *,
    step: int,
    time: float,
    time_key: str,
) -> None:
    """Validate that the resume JSON matches the checkpoint state."""
    if not series:
        return
    if int(series["step"][-1]) != int(step):
        raise ValueError(
            f"Series last step {series['step'][-1]} does not match checkpoint step {step}."
        )
    expected = float(series[time_key][-1])
    tolerance = 1e-12 * max(1.0, abs(expected), abs(time))
    if abs(expected - float(time)) > tolerance:
        raise ValueError(
            f"Series last {time_key} {expected} does not match checkpoint time {time}."
        )


def _measure_driver(
    driver: TDVPDriver,
) -> tuple[object, tuple[object, ...]]:
    """Measure the driver's current state without advancing physical time."""
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


def _plaquette_map_from_stats(
    observable_stats: tuple[object, ...],
    shape: tuple[int, int],
    *,
    attr: str,
) -> list[list[float]]:
    """Convert a flat plaquette-stat tuple into a 2D list."""
    n_rows, n_cols = shape
    return [
        [
            float(getattr(observable_stats[row * (n_cols - 1) + col], attr).real)
            for col in range(n_cols - 1)
        ]
        for row in range(n_rows - 1)
    ]


def _plaquette_snapshot(
    *,
    step: int,
    time: float,
    observable_stats: tuple[object, ...],
    shape: tuple[int, int],
) -> dict[str, Any]:
    """Build one serializable plaquette-map snapshot."""
    return {
        "step": int(step),
        "time": float(time),
        "mean": _plaquette_map_from_stats(observable_stats, shape, attr="mean"),
        "error": _plaquette_map_from_stats(observable_stats, shape, attr="error_of_mean"),
    }


def build_ground_state_driver(
    *,
    model: GIPEPS,
    hamiltonian: GILocalHamiltonian,
    space_name: str,
    n_samples: int,
    n_chains: int,
    dt: float,
    diag_shift: float,
    seed: int,
    t0: float,
) -> TDVPDriver:
    """Build the imaginary-time driver used for ground-state preparation."""
    return TDVPDriver(
        model,
        hamiltonian,
        preconditioner=SRPreconditioner(
            space=_space_from_name(space_name),
            diag_shift=diag_shift,
            metrics_config=SR_METRICS_CONFIG,
        ),
        dt=dt,
        t0=t0,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(seed),
        n_samples=n_samples,
        n_chains=n_chains,
        full_gradient=False,
    )


def build_real_time_driver(
    *,
    model: GIPEPS,
    hamiltonian: GILocalHamiltonian,
    space_name: str,
    n_samples: int,
    n_chains: int,
    dt: float,
    diag_shift: float,
    seed: int,
    t0: float,
) -> TDVPDriver:
    """Build the real-time driver used for vison-pair propagation."""
    return TDVPDriver(
        model,
        hamiltonian,
        observables=build_all_plaquette_observables(model.shape),
        preconditioner=SRPreconditioner(
            space=_space_from_name(space_name),
            diag_shift=diag_shift,
            metrics_config=SR_METRICS_CONFIG,
        ),
        dt=dt,
        t0=t0,
        time_unit=RealTimeUnit(),
        integrator=RK4(),
        sampler_key=jax.random.key(seed),
        n_samples=n_samples,
        n_chains=n_chains,
        full_gradient=False,
    )


def _ground_state_result(
    *,
    problem: dict[str, Any],
    state_path: Path,
    solver_space: str,
    n_samples: int,
    n_chains: int,
    n_steps: int,
    dt: float,
    diag_shift: float,
    log_every: int,
    save_every: int,
    resume_state: Path | None,
    series: dict[str, list],
) -> dict[str, Any]:
    """Build the serializable ground-state JSON payload."""
    summary = {
        "final_step": series["step"][-1] if series else 0,
        "final_imaginary_time": series["imaginary_time"][-1] if series else 0.0,
        "final_energy_mean": series["energy_mean"][-1] if series else None,
        "final_energy_error": series["energy_error"][-1] if series else None,
        "final_energy_variance": series["energy_variance"][-1] if series else None,
    }
    return {
        "problem": problem,
        "stage": "ground_state",
        "state_path": str(state_path),
        "config": {
            "solver_space": solver_space,
            "n_samples": n_samples,
            "n_chains": n_chains,
            "n_steps": n_steps,
            "dt": dt,
            "diag_shift": diag_shift,
            "log_every": log_every,
            "save_every": save_every,
            "resume_state": None if resume_state is None else str(resume_state),
            "n_steps_semantics": "additional_steps",
        },
        "series": series,
        "summary": summary,
    }


def _real_time_result(
    *,
    problem: dict[str, Any],
    state_path: Path,
    solver_space: str,
    n_samples: int,
    n_chains: int,
    T_final: float,
    dt: float,
    diag_shift: float,
    log_every: int,
    save_every: int,
    snapshot_every: int,
    input_state: Path,
    vison: dict[str, Any],
    reference_energy_mean: float,
    series: dict[str, list],
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the serializable real-time JSON payload."""
    summary = {
        "reference_energy_mean": reference_energy_mean,
        "final_step": series["step"][-1] if series else 0,
        "final_time": series["time"][-1] if series else 0.0,
        "final_energy_mean": series["energy_mean"][-1] if series else None,
        "final_energy_error": series["energy_error"][-1] if series else None,
        "final_energy_variance": series["energy_variance"][-1] if series else None,
        "final_energy_drift_percent": (
            series["energy_drift_percent"][-1] if series else None
        ),
    }
    return {
        "problem": problem,
        "stage": "real_time",
        "state_path": str(state_path),
        "input_state": str(input_state),
        "config": {
            "solver_space": solver_space,
            "n_samples": n_samples,
            "n_chains": n_chains,
            "T_final": T_final,
            "dt": dt,
            "diag_shift": diag_shift,
            "log_every": log_every,
            "save_every": save_every,
            "snapshot_every": snapshot_every,
            "time_resume_semantics": "advance_until_T_final",
        },
        "vison": vison,
        "plaquette_row_order": "internal_top_to_bottom",
        "plaquette_col_order": "left_to_right",
        "series": series,
        "plaquette_snapshots": snapshots,
        "summary": summary,
    }


def run_ground_state(
    *,
    driver: TDVPDriver,
    n_steps: int,
    log_every: int,
    save_every: int,
    series: dict[str, list],
    checkpoint_fn,
) -> None:
    """Run additional imaginary-time steps with periodic checkpointing."""
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")
    logger.info(
        "[ground_state] step tau dt wall_time energy energy_err energy_var "
        "applied_FS_step_norm_squared FS_norm_squared TDVP_residual SR_solve_residual"
    )
    for local_step in range(1, n_steps + 1):
        driver.run(driver.dt)
        metrics = driver.metrics
        energy = driver.energy
        fs_norm_squared = float(metrics["FS_norm_squared"])
        row = {
            "step": int(driver.step_count),
            "imaginary_time": float(driver.t),
            "dt": float(driver.dt),
            "step_wall_time": float(metrics["step_wall_time"]),
            "energy_mean": float(energy.mean.real),
            "energy_error": float(energy.error_of_mean.real),
            "energy_variance": float(energy.variance.real),
            "applied_FS_step_norm_squared": float(driver.dt**2 * fs_norm_squared),
            "FS_norm_squared": fs_norm_squared,
            "TDVP_residual": float(metrics["TDVP_residual"]),
            "SR_solve_residual": float(metrics["SR_solve_residual"]),
        }
        append_series(series, **row)
        if driver.step_count % log_every == 0 or local_step == n_steps:
            logger.info(
                "[ground_state] %4d %.6f %.6f %.3f %.10f %.6e %.6e %.6e %.6e %.6e %.6e",
                row["step"],
                row["imaginary_time"],
                row["dt"],
                row["step_wall_time"],
                row["energy_mean"],
                row["energy_error"],
                row["energy_variance"],
                row["applied_FS_step_norm_squared"],
                row["FS_norm_squared"],
                row["TDVP_residual"],
                row["SR_solve_residual"],
            )
        if driver.step_count % save_every == 0 or local_step == n_steps:
            checkpoint_fn(driver, series)


def run_real_time(
    *,
    driver: TDVPDriver,
    T_final: float,
    log_every: int,
    save_every: int,
    snapshot_every: int,
    series: dict[str, list],
    snapshots: list[dict[str, Any]],
    reference_energy_mean: float | None,
    checkpoint_fn,
) -> float:
    """Run real-time evolution until the requested target final time."""
    logger.info(
        "[real_time] step t dt wall_time energy energy_err energy_var drift_percent "
        "applied_FS_step_norm_squared FS_norm_squared TDVP_residual SR_solve_residual"
    )
    if not series:
        energy, observable_stats = _measure_driver(driver)
        reference_energy_mean = (
            float(energy.mean.real)
            if reference_energy_mean is None
            else float(reference_energy_mean)
        )
        append_series(
            series,
            step=int(driver.step_count),
            time=float(driver.t),
            energy_mean=float(energy.mean.real),
            energy_error=float(energy.error_of_mean.real),
            energy_variance=float(energy.variance.real),
            energy_drift_percent=0.0,
            step_wall_time=None,
            applied_FS_step_norm_squared=None,
            FS_norm_squared=None,
            TDVP_residual=None,
            SR_solve_residual=None,
        )
        _replace_or_append_snapshot(
            snapshots,
            _plaquette_snapshot(
                step=driver.step_count,
                time=driver.t,
                observable_stats=observable_stats,
                shape=driver.model.shape,
            ),
        )
        checkpoint_fn(driver, series, snapshots, reference_energy_mean)
    if reference_energy_mean is None:
        raise ValueError("reference_energy_mean must be defined before real-time stepping.")

    remaining_time = float(T_final) - float(driver.t)
    tolerance = 1e-12 * max(1.0, abs(T_final), abs(driver.t), abs(driver.dt))
    if remaining_time < -tolerance:
        raise ValueError(
            f"Target final time {T_final} is smaller than the checkpoint time {driver.t}."
        )
    n_steps = int(round(max(0.0, remaining_time) / driver.dt))
    if abs(max(0.0, remaining_time) - n_steps * driver.dt) > tolerance:
        raise ValueError(
            f"Remaining time {remaining_time} must be an integer multiple of dt={driver.dt}."
        )
    if n_steps == 0:
        return reference_energy_mean

    energy_denom = max(abs(reference_energy_mean), 1e-12)
    for local_step in range(1, n_steps + 1):
        driver.run(driver.dt)
        metrics = driver.metrics
        energy = driver.energy
        fs_norm_squared = float(metrics["FS_norm_squared"])
        row = {
            "step": int(driver.step_count),
            "time": float(driver.t),
            "energy_mean": float(energy.mean.real),
            "energy_error": float(energy.error_of_mean.real),
            "energy_variance": float(energy.variance.real),
            "energy_drift_percent": (
                abs(float(energy.mean.real) - reference_energy_mean) / energy_denom * 100.0
            ),
            "step_wall_time": float(metrics["step_wall_time"]),
            "applied_FS_step_norm_squared": float(driver.dt**2 * fs_norm_squared),
            "FS_norm_squared": fs_norm_squared,
            "TDVP_residual": float(metrics["TDVP_residual"]),
            "SR_solve_residual": float(metrics["SR_solve_residual"]),
        }
        append_series(series, **row)
        if driver.step_count % log_every == 0 or local_step == n_steps:
            logger.info(
                "[real_time] %4d %.6f %.6f %.3f %.10f %.6e %.6e %.6e %.6e %.6e %.6e %.6e",
                row["step"],
                row["time"],
                driver.dt,
                row["step_wall_time"],
                row["energy_mean"],
                row["energy_error"],
                row["energy_variance"],
                row["energy_drift_percent"],
                row["applied_FS_step_norm_squared"],
                row["FS_norm_squared"],
                row["TDVP_residual"],
                row["SR_solve_residual"],
            )
        should_checkpoint = driver.step_count % save_every == 0 or local_step == n_steps
        should_snapshot = (
            driver.step_count % snapshot_every == 0 or should_checkpoint
        )
        if should_snapshot:
            _replace_or_append_snapshot(
                snapshots,
                _plaquette_snapshot(
                    step=driver.step_count,
                    time=driver.t,
                    observable_stats=driver.observable_stats,
                    shape=driver.model.shape,
                ),
            )
        if should_checkpoint:
            checkpoint_fn(driver, series, snapshots, reference_energy_mean)
    return reference_energy_mean


def _run_ground_state_command(args: argparse.Namespace) -> None:
    """Handle the ``ground-state`` CLI subcommand."""
    if args.log_every <= 0 or args.save_every <= 0:
        raise ValueError("--log-every and --save-every must be positive.")
    resume_state = (
        None if args.resume_state is None else _normalize_state_path(args.resume_state)
    )
    if resume_state is not None:
        checkpoint = load_checkpoint_state(resume_state)
        if checkpoint.metadata.get("stage") != "ground_state":
            raise ValueError("Ground-state resume requires a ground-state checkpoint.")
        stored_space = _checkpoint_solver_space(checkpoint.metadata)
        space_name = args.space
        if space_name != stored_space:
            raise ValueError(
                "Ground-state resume requires the same solver space as the checkpoint. "
                f"Checkpoint uses {stored_space!r}."
            )
        problem = build_problem_metadata(
            shape=tuple(int(x) for x in checkpoint.metadata["shape"]),
            h=float(checkpoint.metadata["h"]),
            g=float(checkpoint.metadata["g"]),
            J=float(checkpoint.metadata["J"]),
            sigma_z_field=float(checkpoint.metadata["sigma_z_field"]),
            bond_dim=int(checkpoint.metadata["bond_dim"]),
            boundary_dim=int(checkpoint.metadata["boundary_dim"]),
            boundary_sweeps=int(checkpoint.metadata["boundary_sweeps"]),
            seed=int(checkpoint.metadata["seed"]),
            solver_space=space_name,
        )
        state_path = _normalize_state_path(args.state_output or resume_state)
        json_path = args.json_output or _matching_json_path(state_path)
        resume_json = _matching_json_path(resume_state)
        result = _load_resume_json(json_path, resume_json)
        series = result.get("series", {})
        _validate_series(
            series,
            step=checkpoint.step_count,
            time=checkpoint.time,
            time_key="imaginary_time",
        )
        logger.info(
            "Ground-state setup: L=%s h=%s g=%s J=%s Dk=%s space=%s n_samples=%s n_chains=%s",
            problem["L"],
            problem["h"],
            problem["g"],
            problem["J"],
            problem["bond_dim"],
            space_name,
            args.n_samples,
            args.n_chains,
        )
        driver = build_ground_state_driver(
            model=checkpoint.model,
            hamiltonian=build_z2_higgs_hamiltonian(
                tuple(int(x) for x in checkpoint.metadata["shape"]),
                h=float(checkpoint.metadata["h"]),
                g=float(checkpoint.metadata["g"]),
                J=float(checkpoint.metadata["J"]),
                sigma_z_field=float(checkpoint.metadata["sigma_z_field"]),
            ),
            space_name=space_name,
            n_samples=args.n_samples,
            n_chains=args.n_chains,
            dt=args.dt,
            diag_shift=args.diag_shift,
            seed=int(problem["seed"]),
            t0=checkpoint.time,
        )
        seed_driver_from_checkpoint(
            driver,
            checkpoint,
            step_count=checkpoint.step_count,
            time=checkpoint.time,
        )
    else:
        shape = (args.L, args.L)
        boundary_dim = args.boundary_dim or 3 * args.bond_dim
        space_name = args.space
        problem = build_problem_metadata(
            shape=shape,
            h=args.h,
            g=args.g,
            J=args.J,
            sigma_z_field=args.sigma_z_field,
            bond_dim=args.bond_dim,
            boundary_dim=boundary_dim,
            boundary_sweeps=args.boundary_sweeps,
            seed=args.seed,
            solver_space=space_name,
        )
        state_path = _normalize_state_path(
            args.state_output
            or _default_ground_state_state_path(
                L=args.L,
                g=args.g,
                J=args.J,
                bond_dim=args.bond_dim,
            )
        )
        json_path = args.json_output or _matching_json_path(state_path)
        series: dict[str, list] = {}
        logger.info(
            "Ground-state setup: L=%s h=%s g=%s J=%s Dk=%s space=%s n_samples=%s n_chains=%s",
            problem["L"],
            problem["h"],
            problem["g"],
            problem["J"],
            problem["bond_dim"],
            space_name,
            args.n_samples,
            args.n_chains,
        )
        driver = build_ground_state_driver(
            model=build_model(
                shape,
                bond_dim=args.bond_dim,
                boundary_dim=boundary_dim,
                boundary_sweeps=args.boundary_sweeps,
                seed=args.seed,
            ),
            hamiltonian=build_z2_higgs_hamiltonian(
                shape,
                h=args.h,
                g=args.g,
                J=args.J,
                sigma_z_field=args.sigma_z_field,
            ),
            space_name=space_name,
            n_samples=args.n_samples,
            n_chains=args.n_chains,
            dt=args.dt,
            diag_shift=args.diag_shift,
            seed=args.seed,
            t0=0.0,
        )

    logger.info(
        "Ground-state run: L=%s h=%s g=%s J=%s Dk=%s space=%s n_samples=%s n_chains=%s",
        problem["L"],
        problem["h"],
        problem["g"],
        problem["J"],
        problem["bond_dim"],
        space_name,
        args.n_samples,
        args.n_chains,
    )

    def checkpoint_fn(driver_state: TDVPDriver, series_data: dict[str, list]) -> None:
        save_driver_checkpoint(
            driver_state,
            {**problem, "stage": "ground_state"},
            state_path,
        )
        _save_json(
            _ground_state_result(
                problem=problem,
                state_path=state_path,
                solver_space=space_name,
                n_samples=args.n_samples,
                n_chains=args.n_chains,
                n_steps=args.n_steps,
                dt=args.dt,
                diag_shift=args.diag_shift,
                log_every=args.log_every,
                save_every=args.save_every,
                resume_state=resume_state,
                series=series_data,
            ),
            json_path,
        )

    run_ground_state(
        driver=driver,
        n_steps=args.n_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        series=series,
        checkpoint_fn=checkpoint_fn,
    )


def _run_real_time_command(args: argparse.Namespace) -> None:
    """Handle the ``real-time`` CLI subcommand."""
    if args.log_every <= 0 or args.save_every <= 0:
        raise ValueError("--log-every and --save-every must be positive.")
    snapshot_every = args.save_every if args.snapshot_every is None else args.snapshot_every
    if snapshot_every <= 0:
        raise ValueError("--snapshot-every must be positive.")

    input_state = _normalize_state_path(args.state)
    checkpoint = load_checkpoint_state(input_state)
    shape = tuple(int(x) for x in checkpoint.metadata["shape"])
    stage = checkpoint.metadata.get("stage")
    if stage == "ground_state":
        space_name = args.space
    elif stage == "real_time":
        stored_space = _checkpoint_solver_space(checkpoint.metadata)
        space_name = args.space
        if args.space is not None and args.space != stored_space:
            raise ValueError(
                "Resuming a real-time checkpoint requires the same solver space as the checkpoint. "
                f"Checkpoint uses {stored_space!r}."
            )
    else:
        raise ValueError("Real-time run requires a ground-state or real-time checkpoint.")
    problem = build_problem_metadata(
        shape=shape,
        h=float(checkpoint.metadata["h"]),
        g=float(checkpoint.metadata["g"]),
        J=float(checkpoint.metadata["J"]),
        sigma_z_field=float(checkpoint.metadata["sigma_z_field"]),
        bond_dim=int(checkpoint.metadata["bond_dim"]),
        boundary_dim=int(checkpoint.metadata["boundary_dim"]),
        boundary_sweeps=int(checkpoint.metadata["boundary_sweeps"]),
        seed=int(checkpoint.metadata["seed"]),
        solver_space=space_name,
    )
    hamiltonian = build_z2_higgs_hamiltonian(
        shape,
        h=float(problem["h"]),
        g=float(problem["g"]),
        J=float(problem["J"]),
        sigma_z_field=float(problem["sigma_z_field"]),
    )

    if stage == "ground_state":
        orientation = "v" if args.orientation is None else args.orientation
        default_row, default_col = default_vison_link(shape, orientation)
        link_row = default_row if args.link_row is None else args.link_row
        link_col = default_col if args.link_col is None else args.link_col
        plaquettes = vison_pair_plaquettes(
            shape,
            orientation=orientation,
            row=link_row,
            col=link_col,
        )
        model = create_interior_vison_pair(
            checkpoint.model,
            orientation=orientation,
            row=link_row,
            col=link_col,
        )
        driver = build_real_time_driver(
            model=model,
            hamiltonian=hamiltonian,
            space_name=space_name,
            n_samples=args.n_samples,
            n_chains=args.n_chains,
            dt=args.dt,
            diag_shift=args.diag_shift,
            seed=int(problem["seed"]) + 1,
            t0=0.0,
        )
        seed_driver_from_checkpoint(driver, checkpoint, step_count=0, time=0.0)
        state_path = _normalize_state_path(
            args.state_output
            or _default_real_time_state_path(
                input_state,
                orientation=orientation,
                row=link_row,
                col=link_col,
            )
        )
        if state_path == input_state:
            raise ValueError(
                "Real-time state-output must differ from the input ground-state checkpoint."
            )
        json_path = args.json_output or _matching_json_path(state_path)
        series: dict[str, list] = {}
        snapshots: list[dict[str, Any]] = []
        reference_energy_mean = None
        vison = {
            "operator": "sigma_z",
            "orientation": orientation,
            "link_row": int(link_row),
            "link_col": int(link_col),
            "excited_plaquettes": [list(site) for site in plaquettes],
        }
        logger.info(
            "Real-time setup: L=%s h=%s g=%s J=%s Dk=%s space=%s n_samples=%s n_chains=%s vison=%s(%s,%s) T_final=%s",
            problem["L"],
            problem["h"],
            problem["g"],
            problem["J"],
            problem["bond_dim"],
            space_name,
            args.n_samples,
            args.n_chains,
            vison["orientation"],
            vison["link_row"],
            vison["link_col"],
            args.T_final,
        )
    elif stage == "real_time":
        vison = checkpoint.metadata["vison"]
        if args.orientation is not None and args.orientation != vison["orientation"]:
            raise ValueError(
                "Resuming a real-time checkpoint requires the same vison orientation."
            )
        if args.link_row is not None and args.link_row != int(vison["link_row"]):
            raise ValueError(
                "Resuming a real-time checkpoint requires the same vison link row."
            )
        if args.link_col is not None and args.link_col != int(vison["link_col"]):
            raise ValueError(
                "Resuming a real-time checkpoint requires the same vison link col."
            )
        driver = build_real_time_driver(
            model=checkpoint.model,
            hamiltonian=hamiltonian,
            space_name=space_name,
            n_samples=args.n_samples,
            n_chains=args.n_chains,
            dt=args.dt,
            diag_shift=args.diag_shift,
            seed=int(problem["seed"]) + 1,
            t0=checkpoint.time,
        )
        seed_driver_from_checkpoint(
            driver,
            checkpoint,
            step_count=checkpoint.step_count,
            time=checkpoint.time,
        )
        state_path = _normalize_state_path(args.state_output or input_state)
        json_path = args.json_output or _matching_json_path(state_path)
        result = _load_resume_json(json_path, _matching_json_path(input_state))
        series = result.get("series", {})
        snapshots = result.get("plaquette_snapshots", [])
        _validate_series(
            series,
            step=checkpoint.step_count,
            time=checkpoint.time,
            time_key="time",
        )
        reference_energy_mean = result.get("summary", {}).get("reference_energy_mean")
        if reference_energy_mean is None:
            reference_energy_mean = checkpoint.metadata.get("reference_energy_mean")
        logger.info(
            "Real-time setup: L=%s h=%s g=%s J=%s Dk=%s space=%s n_samples=%s n_chains=%s vison=%s(%s,%s) T_final=%s",
            problem["L"],
            problem["h"],
            problem["g"],
            problem["J"],
            problem["bond_dim"],
            space_name,
            args.n_samples,
            args.n_chains,
            vison["orientation"],
            vison["link_row"],
            vison["link_col"],
            args.T_final,
        )

    logger.info(
        "Real-time run: L=%s h=%s g=%s J=%s Dk=%s space=%s n_samples=%s n_chains=%s "
        "vison=%s(%s,%s) T_final=%s",
        problem["L"],
        problem["h"],
        problem["g"],
        problem["J"],
        problem["bond_dim"],
        space_name,
        args.n_samples,
        args.n_chains,
        vison["orientation"],
        vison["link_row"],
        vison["link_col"],
        args.T_final,
    )

    def checkpoint_fn(
        driver_state: TDVPDriver,
        series_data: dict[str, list],
        snapshot_data: list[dict[str, Any]],
        reference_energy: float,
    ) -> None:
        save_driver_checkpoint(
            driver_state,
            {
                **problem,
                "stage": "real_time",
                "vison": vison,
                "reference_energy_mean": float(reference_energy),
            },
            state_path,
        )
        _save_json(
            _real_time_result(
                problem=problem,
                state_path=state_path,
                solver_space=space_name,
                n_samples=args.n_samples,
                n_chains=args.n_chains,
                T_final=args.T_final,
                dt=args.dt,
                diag_shift=args.diag_shift,
                log_every=args.log_every,
                save_every=args.save_every,
                snapshot_every=snapshot_every,
                input_state=input_state,
                vison=vison,
                reference_energy_mean=float(reference_energy),
                series=series_data,
                snapshots=snapshot_data,
            ),
            json_path,
        )

    run_real_time(
        driver=driver,
        T_final=args.T_final,
        log_every=args.log_every,
        save_every=args.save_every,
        snapshot_every=snapshot_every,
        series=series,
        snapshots=snapshots,
        reference_energy_mean=None if reference_energy_mean is None else float(reference_energy_mean),
        checkpoint_fn=checkpoint_fn,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Resumable Z2 vison confinement example with Higgs terms, "
            "ground-state checkpointing, and real-time target-final-time resume."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ground_state = subparsers.add_parser(
        "ground-state",
        help="Optimize or resume a parity-sector Z2 Higgs ground state.",
    )
    ground_state.add_argument("--L", type=int, default=DEFAULT_L)
    ground_state.add_argument("--h", type=float, default=DEFAULT_H)
    ground_state.add_argument("--g", type=float, default=DEFAULT_G)
    ground_state.add_argument("--J", type=float, default=DEFAULT_J)
    ground_state.add_argument(
        "--sigma-z-field",
        type=float,
        default=DEFAULT_SIGMA_Z_FIELD,
        help="Coefficient of the paper term -sum_i sigma_i^z.",
    )
    ground_state.add_argument("--bond-dim", type=int, default=DEFAULT_BOND_DIM)
    ground_state.add_argument(
        "--boundary-dim",
        type=int,
        default=None,
        help="Boundary MPS bond dimension. Defaults to 3 * bond-dim.",
    )
    ground_state.add_argument(
        "--boundary-sweeps",
        type=int,
        default=DEFAULT_BOUNDARY_SWEEPS,
    )
    ground_state.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    ground_state.add_argument("--n-chains", type=int, default=DEFAULT_N_CHAINS)
    ground_state.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS_GS,
        help="Additional imaginary-time steps to run.",
    )
    ground_state.add_argument("--dt", type=float, default=DEFAULT_DT_GS)
    ground_state.add_argument("--diag-shift", type=float, default=DEFAULT_GS_DIAG_SHIFT)
    ground_state.add_argument(
        "--space",
        choices=("sr", "minsr"),
        default=DEFAULT_SOLVER_SPACE,
        help="SR solve space. Defaults to minSR.",
    )
    ground_state.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    ground_state.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    ground_state.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ground_state.add_argument("--resume-state", type=Path, default=None)
    ground_state.add_argument("--state-output", type=Path, default=None)
    ground_state.add_argument("--json-output", type=Path, default=None)
    ground_state.set_defaults(run=_run_ground_state_command)

    real_time = subparsers.add_parser(
        "real-time",
        help="Create an interior vison pair from a ground state or resume a real-time run.",
    )
    real_time.add_argument("--state", type=Path, required=True)
    real_time.add_argument(
        "--orientation",
        choices=("v", "h"),
        default=None,
        help=(
            "For a ground-state input, create the vison pair on a vertical or "
            "horizontal interior link. Ignored when resuming a real-time checkpoint."
        ),
    )
    real_time.add_argument(
        "--link-row",
        type=int,
        default=None,
        help=(
            "Link row in site coordinates. For vertical links this is the upper-site row; "
            "for horizontal links this is the left-site row."
        ),
    )
    real_time.add_argument(
        "--link-col",
        type=int,
        default=None,
        help=(
            "Link col in site coordinates. For vertical links this is the upper-site col; "
            "for horizontal links this is the left-site col."
        ),
    )
    real_time.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    real_time.add_argument("--n-chains", type=int, default=DEFAULT_N_CHAINS)
    real_time.add_argument(
        "--T-final",
        "--T",
        dest="T_final",
        type=float,
        default=DEFAULT_T_FINAL,
        help=(
            "Target final physical time. When resuming, the script advances only "
            "the remaining interval T_final - t_checkpoint."
        ),
    )
    real_time.add_argument("--dt", type=float, default=DEFAULT_DT_RT)
    real_time.add_argument("--diag-shift", type=float, default=DEFAULT_RT_DIAG_SHIFT)
    real_time.add_argument(
        "--space",
        choices=("sr", "minsr"),
        default=DEFAULT_SOLVER_SPACE,
        help="SR solve space. Defaults to minSR.",
    )
    real_time.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    real_time.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    real_time.add_argument(
        "--snapshot-every",
        type=int,
        default=None,
        help="Plaquette-map snapshot cadence. Defaults to --save-every.",
    )
    real_time.add_argument("--state-output", type=Path, default=None)
    real_time.add_argument("--json-output", type=Path, default=None)
    real_time.set_defaults(run=_run_real_time_command)

    return parser


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = build_parser()
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
