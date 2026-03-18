"""Reproduce Fig. 5(a) vison propagation with GI-PEPS.

This example follows the 6x6 Z2 vison benchmark of Wu and Liu (2025) using
three decoupled subcommands:

1. ``ground-state`` optimizes the deconfined ground state and saves it.
2. ``real-time`` loads the saved state, inserts one boundary vison, and runs
   real-time TDVP.
3. ``plot`` reads the saved real-time JSON and overlays the upstream exact
   open-data trace without computing exact dynamics locally.

The selected plaquettes follow the open-data convention directly: row-major
indices on the 5x5 plaquette grid for the 6x6 lattice.
"""

from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import argparse
import io
import json
import ssl
import urllib.request
from pathlib import Path

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


FIG5A_PLAQUETTES = ((0, 0), (0, 1), (2, 2))
FIG5A_OPEN_DATA_COLUMNS = (3, 4, 15)
EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_EXACT_OPEN_DATA_URL = (
    "https://raw.githubusercontent.com/yantaow/open_data/main/"
    "wu2025accurate/Fig5/6x6_vison/vison_6x6_g0.1_h1.0_dt0.1_exact.out"
)
DEFAULT_EXACT_OPEN_DATA_CACHE = (
    EXAMPLE_DIR / "vison_6x6_g0.1_h1.0_dt0.1_exact.out"
)

DEFAULT_L = 6
DEFAULT_H = 1.0
DEFAULT_G = 0.1
DEFAULT_BOND_DIM = 3
DEFAULT_N_SAMPLES = 10240
DEFAULT_N_CHAINS = 1024
DEFAULT_N_STEPS_GS = 400
DEFAULT_DT_GS = 0.005
DEFAULT_GS_DIAG_SHIFT = 1e-4
DEFAULT_T = 18.0
DEFAULT_DT_RT = 0.01
DEFAULT_RT_DIAG_SHIFT = 1e-8
DEFAULT_SEED = 42

SR_METRICS_CONFIG = MetricsConfig(
    record_FS_norm=True,
    record_TDVP_residual=True,
    record_SR_solve_residual=True,
    record_step_wall_time=True,
)


def append_series(series: dict[str, list], **values) -> None:
    """Append one row into a columnar series dict."""
    for key, value in values.items():
        series.setdefault(key, []).append(value)


def _token(value: float, digits: int = 3) -> str:
    return format(value, f".{digits}f").replace(".", "p")


def _run_stem(*, L: int, g: float, bond_dim: int) -> str:
    return f"z2_vison_propagation_L{L}_g{_token(g)}_Dk{bond_dim}"


def _default_ground_state_state_path(*, L: int, g: float, bond_dim: int) -> Path:
    return EXAMPLE_DIR / f"{_run_stem(L=L, g=g, bond_dim=bond_dim)}_ground_state.npz"


def _default_ground_state_json_path(*, L: int, g: float, bond_dim: int) -> Path:
    return EXAMPLE_DIR / f"{_run_stem(L=L, g=g, bond_dim=bond_dim)}_ground_state.json"


def _default_real_time_json_path(state_path: Path, *, T: float) -> Path:
    stem = state_path.stem.removesuffix("_ground_state")
    return state_path.with_name(f"{stem}_real_time_T{_token(T)}.json")


def _default_plot_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_fig5a.png")


def _save_json(result: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {output_path}", flush=True)


def _download_text(url: str) -> str:
    """Download text with a verified SSL context."""
    verify_paths = ssl.get_default_verify_paths()
    for cafile in (
        verify_paths.cafile,
        verify_paths.openssl_cafile,
        "/etc/ssl/cert.pem",
        "/opt/homebrew/etc/openssl@3/cert.pem",
    ):
        if cafile and Path(cafile).exists():
            context = ssl.create_default_context(cafile=cafile)
            with urllib.request.urlopen(url, context=context) as response:
                return response.read().decode("utf-8")
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def _load_or_download_text(path: Path, url: str) -> str:
    """Load cached text if available, otherwise download and cache it."""
    if path.exists():
        return path.read_text()
    text = _download_text(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"Saved {path}", flush=True)
    return text


def _prefetch_exact_open_data() -> None:
    """Best-effort cache warmup for the exact open-data trace."""
    try:
        _load_or_download_text(
            DEFAULT_EXACT_OPEN_DATA_CACHE,
            DEFAULT_EXACT_OPEN_DATA_URL,
        )
    except Exception as exc:
        print(f"Warning: failed to cache exact open data: {exc}", flush=True)


def build_z2_hamiltonian(
    shape: tuple[int, int],
    *,
    h: float,
    g: float,
) -> GILocalHamiltonian:
    """Build the pure Z2 gauge Hamiltonian."""
    n_rows, n_cols = shape
    plaquette_terms = tuple(
        PlaquetteOperator(row=row, col=col, coeff=-h)
        for row in range(n_rows - 1)
        for col in range(n_cols - 1)
    )
    electric_terms = build_electric_terms(shape, coeff=g, N=2)
    return GILocalHamiltonian(shape=shape, terms=electric_terms + plaquette_terms)


def build_fig5a_plaquette_observables(
    shape: tuple[int, int],
    plaquettes: tuple[tuple[int, int], ...] = FIG5A_PLAQUETTES,
) -> tuple[GILocalHamiltonian, ...]:
    """Build the selected plaquette observables used in Fig. 5(a).

    The plaquette labels follow the Wu open-data convention directly: row-major
    plaquette-grid indices on the ``(L - 1) x (L - 1)`` plaquette lattice.

    ``PlaquetteOperator`` evaluates ``coeff * (P + P†)``. For Z2, ``P = P†``,
    so a coefficient of ``0.5`` yields the plaquette expectation value itself.
    """
    if shape[0] < 4 or shape[1] < 4:
        raise ValueError("Fig. 5(a) selected plaquettes require L >= 4.")
    return tuple(
        GILocalHamiltonian(
            shape=shape,
            terms=(PlaquetteOperator(row=row, col=col, coeff=0.5),),
        )
        for row, col in plaquettes
    )


def build_model(
    shape: tuple[int, int],
    *,
    bond_dim: int,
    seed: int,
) -> GIPEPS:
    """Build a pure-gauge GI-PEPS model."""
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
    """Return the independent local link directions on one GI-PEPS site."""
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


def create_bottom_left_vison(model: GIPEPS) -> GIPEPS:
    """Act with sigma_z on the bottom-left vertical boundary link."""
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


def save_ground_state(model: GIPEPS, metadata: dict, output_path: Path) -> None:
    """Save the optimized PEPS tensors and minimal rebuild metadata."""
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


def load_ground_state(input_path: Path) -> tuple[GIPEPS, dict]:
    """Load a saved optimized GI-PEPS state."""
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


def _measure_driver(
    driver: TDVPDriver,
) -> tuple[object, tuple[object, ...]]:
    """Measure the driver's current state without evolving time."""
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
) -> tuple[TDVPDriver, dict[str, list]]:
    """Run imaginary-time SR for the ground state."""
    driver = TDVPDriver(
        model,
        hamiltonian,
        preconditioner=SRPreconditioner(
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
    series: dict[str, list] = {}
    print(
        "[ground_state] step t dt wall_time energy energy_err energy_var "
        "applied_FS_step_norm_squared FS_norm_squared TDVP_residual "
        "SR_solve_residual",
        flush=True,
    )
    for step in range(1, n_steps + 1):
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
    return driver, series


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
) -> dict[str, list]:
    """Run real-time TDVP from the vison state."""
    observables = build_fig5a_plaquette_observables(model.shape)
    driver = TDVPDriver(
        model,
        hamiltonian,
        observables=observables,
        preconditioner=SRPreconditioner(
            diag_shift=diag_shift,
            metrics_config=SR_METRICS_CONFIG,
        ),
        dt=dt,
        time_unit=RealTimeUnit(),
        integrator=RK4(),
        sampler_key=jax.random.key(seed),
        n_samples=n_samples,
        n_chains=n_chains,
        full_gradient=False,
    )
    n_steps = int(round(T / dt))
    if abs(T - n_steps * dt) > 1e-12 * max(1.0, abs(T), abs(dt)):
        raise ValueError(f"T={T} must be an integer multiple of dt={dt}.")

    series = {
        "step": [],
        "time": [],
        "energy_mean": [],
        "energy_error": [],
        "energy_variance": [],
        "energy_drift_percent": [],
        "selected_plaquette_mean": [],
        "selected_plaquette_error": [],
        "step_wall_time": [],
        "applied_FS_step_norm_squared": [],
        "FS_norm_squared": [],
        "TDVP_residual": [],
        "SR_solve_residual": [],
    }
    print(
        "[real_time] step t dt wall_time energy energy_err energy_var "
        "drift_percent p00 p00_err p01 p01_err p22 p22_err "
        "applied_FS_step_norm_squared FS_norm_squared TDVP_residual "
        "SR_solve_residual",
        flush=True,
    )
    energy, observable_stats = _measure_driver(driver)
    reference_energy = float(energy.mean.real)
    selected_means = [float(stat.mean.real) for stat in observable_stats]
    selected_errors = [float(stat.error_of_mean.real) for stat in observable_stats]
    append_series(
        series,
        step=0,
        time=0.0,
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
    print(
        (
            f"[real_time] {0:4d} {0.0:.6f} {dt:.6f} {'-':>8} "
            f"{reference_energy:.10f} {float(energy.error_of_mean.real):.6f} "
            f"{float(energy.variance.real):.6f} {0.0:.6f} "
            f"{selected_means[0]:.10f} {selected_errors[0]:.6f} "
            f"{selected_means[1]:.10f} {selected_errors[1]:.6f} "
            f"{selected_means[2]:.10f} {selected_errors[2]:.6f} "
            f"{'-':>14} {'-':>14} {'-':>14} {'-':>14}"
        ),
        flush=True,
    )
    for step in range(1, n_steps + 1):
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
    return series


def load_exact_open_data(
    *,
    cache_path: Path = DEFAULT_EXACT_OPEN_DATA_CACHE,
    url: str = DEFAULT_EXACT_OPEN_DATA_URL,
) -> dict:
    """Load the upstream exact open-data trace used for Fig. 5(a)."""
    data = np.loadtxt(io.StringIO(_load_or_download_text(cache_path, url)))
    selected = [
        [float(row[column]) for column in FIG5A_OPEN_DATA_COLUMNS]
        for row in data
    ]
    return {
        "cache_path": str(cache_path),
        "source_url": url,
        "selected_plaquettes": [list(site) for site in FIG5A_PLAQUETTES],
        "time": data[:, 1].astype(float).tolist(),
        "selected_plaquette_mean": selected,
    }


def plot_fig5a(
    real_time_result: dict,
    exact_result: dict,
    output_path: Path,
) -> None:
    """Overlay GI-PEPS and upstream exact selected plaquette traces."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    peps_time = real_time_result["real_time"]["time"]
    peps_values = real_time_result["real_time"]["selected_plaquette_mean"]
    exact_time = exact_result["time"]
    exact_values = exact_result["selected_plaquette_mean"]

    problem = real_time_result["problem"]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.6), sharex=True)
    for axis, index, plaquette in zip(axes, range(3), FIG5A_PLAQUETTES, strict=True):
        axis.plot(
            peps_time,
            [row[index] for row in peps_values],
            color="#1f77b4",
            linewidth=1.6,
            label="GI-PEPS",
        )
        axis.plot(
            exact_time,
            [row[index] for row in exact_values],
            color="#ff7f0e",
            linewidth=1.4,
            label="exact open data",
        )
        axis.set_ylabel(rf"$\langle P_{{{plaquette[0]},{plaquette[1]}}}\rangle / 2$")
        axis.grid(alpha=0.25)
        if index == 0:
            axis.legend(frameon=False, loc="best")
    axes[0].set_title(
        rf"(a) ${problem['L']}\times {problem['L']}$ $Z_2$ gauge theory, $g={problem['g']}$"
    )
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}", flush=True)


def _run_ground_state_command(args: argparse.Namespace) -> None:
    _prefetch_exact_open_data()
    if args.L < 4:
        raise ValueError("Fig. 5(a) requires L >= 4.")
    shape = (args.L, args.L)
    hamiltonian = build_z2_hamiltonian(shape, h=args.h, g=args.g)
    driver, series = run_ground_state(
        build_model(shape, bond_dim=args.bond_dim, seed=args.seed),
        hamiltonian,
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        n_steps=args.n_steps,
        dt=args.dt,
        diag_shift=args.diag_shift,
        seed=args.seed,
    )
    state_path = args.state_output or _default_ground_state_state_path(
        L=args.L,
        g=args.g,
        bond_dim=args.bond_dim,
    )
    state_metadata = {
        "gauge_group": "Z2",
        "shape": list(shape),
        "L": args.L,
        "h": args.h,
        "g": args.g,
        "bond_dim": args.bond_dim,
        "boundary_method": "Variational",
        "boundary_dimension": 3 * args.bond_dim,
        "seed": args.seed,
    }
    save_ground_state(driver.model, state_metadata, state_path)
    output_path = args.output or _default_ground_state_json_path(
        L=args.L,
        g=args.g,
        bond_dim=args.bond_dim,
    )
    result = {
        "problem": state_metadata,
        "selected_plaquettes": [list(site) for site in FIG5A_PLAQUETTES],
        "selected_plaquette_columns": list(FIG5A_OPEN_DATA_COLUMNS),
        "ground_state": {
            "n_samples": args.n_samples,
            "n_chains": args.n_chains,
            "n_steps": args.n_steps,
            "dt": args.dt,
            "diag_shift": args.diag_shift,
            "state_path": str(state_path),
            "series": series,
        },
        "summary": {
            "final_step": series["step"][-1],
            "final_imaginary_time": series["imaginary_time"][-1],
            "final_energy_mean": series["energy_mean"][-1],
            "final_energy_error": series["energy_error"][-1],
            "final_energy_variance": series["energy_variance"][-1],
        },
    }
    _save_json(result, output_path)


def _run_real_time_command(args: argparse.Namespace) -> None:
    model, metadata = load_ground_state(args.state)
    shape = tuple(int(x) for x in metadata["shape"])
    hamiltonian = build_z2_hamiltonian(
        shape,
        h=float(metadata["h"]),
        g=float(metadata["g"]),
    )
    series = run_real_time(
        create_bottom_left_vison(model),
        hamiltonian,
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        T=args.T,
        dt=args.dt,
        diag_shift=args.diag_shift,
        seed=args.seed if args.seed is not None else int(metadata["seed"]) + 1,
    )
    output_path = args.output or _default_real_time_json_path(args.state, T=args.T)
    result = {
        "problem": metadata,
        "state_path": str(args.state),
        "selected_plaquettes": [list(site) for site in FIG5A_PLAQUETTES],
        "selected_plaquette_columns": list(FIG5A_OPEN_DATA_COLUMNS),
        "vison": {
            "operator": "sigma_z",
            "orientation": "v",
            "link_row": shape[0] - 2,
            "link_col": 0,
            "plaquettes": [[shape[0] - 2, 0]],
        },
        "real_time": {
            "n_samples": args.n_samples,
            "n_chains": args.n_chains,
            "T": args.T,
            "dt": args.dt,
            "diag_shift": args.diag_shift,
            "series_source": "GI-PEPS",
            **series,
        },
        "summary": {
            "final_real_time_energy_mean": series["energy_mean"][-1],
            "final_real_time_energy_error": series["energy_error"][-1],
            "final_real_time_energy_variance": series["energy_variance"][-1],
            "final_energy_drift_percent": series["energy_drift_percent"][-1],
        },
    }
    _save_json(result, output_path)


def _run_plot_command(args: argparse.Namespace) -> None:
    real_time_result = json.loads(args.input.read_text())
    exact_result = load_exact_open_data(cache_path=args.exact_cache, url=args.exact_url)
    output_path = args.output or _default_plot_path(args.input)
    plot_fig5a(real_time_result, exact_result, output_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Z2 vison propagation benchmark with decoupled ground-state, real-time, and plot flows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ground_state = subparsers.add_parser(
        "ground-state",
        help="Optimize and save the 6x6 Z2 ground state.",
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
    ground_state.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ground_state.add_argument("--state-output", type=Path, default=None)
    ground_state.add_argument("--output", type=Path, default=None)
    ground_state.set_defaults(run=_run_ground_state_command)

    real_time = subparsers.add_parser(
        "real-time",
        help="Load a saved ground state and run real-time vison dynamics.",
    )
    real_time.add_argument("--state", type=Path, required=True)
    real_time.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    real_time.add_argument("--n-chains", type=int, default=DEFAULT_N_CHAINS)
    real_time.add_argument("--T", type=float, default=DEFAULT_T)
    real_time.add_argument("--dt", type=float, default=DEFAULT_DT_RT)
    real_time.add_argument("--diag-shift", type=float, default=DEFAULT_RT_DIAG_SHIFT)
    real_time.add_argument("--seed", type=int, default=None)
    real_time.add_argument("--output", type=Path, default=None)
    real_time.set_defaults(run=_run_real_time_command)

    plot = subparsers.add_parser(
        "plot",
        help="Plot saved GI-PEPS real-time JSON against the upstream exact open data.",
    )
    plot.add_argument("--input", type=Path, required=True)
    plot.add_argument("--exact-cache", type=Path, default=DEFAULT_EXACT_OPEN_DATA_CACHE)
    plot.add_argument("--exact-url", default=DEFAULT_EXACT_OPEN_DATA_URL)
    plot.add_argument("--output", type=Path, default=None)
    plot.set_defaults(run=_run_plot_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
