"""Reproduce Fig. 5(a) vison propagation with GI-PEPS.

This script follows the real-time setup of Wu and Liu (2025) for the pure Z2
gauge theory:

1. Optimize the deconfined ground state with imaginary-time SR.
2. Create a single vison by acting with sigma_z on the bottom-left vertical
   boundary link.
3. Evolve the vison state in real time with TDVP and record plaquette values.

This script targets the 6x6 benchmark of Wu and Liu (2025) and records the
three selected plaquettes shown in Fig. 5(a): (0, 0), (0, 1), and (2, 2).
"""

from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import argparse
import json
import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from netket import stats as nkstats
from flax import nnx

from vmc.drivers import ImaginaryTimeUnit, RK4, RealTimeUnit, TDVPDriver
from vmc.operators import PlaquetteOperator
from vmc.peps import GIPEPS, GIPEPSConfig, GILocalHamiltonian, ZipUp
from vmc.peps.gi.local_terms import build_electric_terms
from vmc.preconditioners import SRPreconditioner


FIG5A_PLAQUETTES = ((0, 0), (0, 1), (2, 2))

DEFAULT_L = 6
DEFAULT_H = 1.0
DEFAULT_G = 0.1
DEFAULT_BOND_DIM = 3
DEFAULT_N_SAMPLES_GS = 2048
DEFAULT_N_STEPS_GS = 100
DEFAULT_DT_GS = 0.01
DEFAULT_GS_DIAG_SHIFT = 1e-4
DEFAULT_N_SAMPLES_RT = 2048
DEFAULT_T = 18.0
DEFAULT_DT_RT = 0.005
DEFAULT_RT_DIAG_SHIFT = 1e-8
DEFAULT_SEED = 42

logger = logging.getLogger(__name__)


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

    The paper labels plaquettes by physics coordinates ``(x, y)`` with origin
    at the bottom-left corner. Internally we convert them to array coordinates
    whose origin is at the top-left corner.

    ``PlaquetteOperator`` evaluates ``coeff * (P + P†)``. For Z2, ``P = P†``,
    so a coefficient of ``0.5`` yields the plaquette expectation value itself.
    """
    if shape[0] < 4 or shape[1] < 4:
        raise ValueError("Fig. 5(a) selected plaquettes require L >= 4.")
    return tuple(
        GILocalHamiltonian(
            shape=shape,
            terms=(
                PlaquetteOperator(
                    row=shape[0] - 2 - y,
                    col=x,
                    coeff=0.5,
                ),
            ),
        )
        for x, y in plaquettes
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
        contraction_strategy=ZipUp(truncate_bond_dimension=3 * bond_dim),
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
    """Act with sigma_z on the bottom-left vertical boundary link.

    This is the single-vison construction used in the 2025 Wu-Liu paper.
    The bottom-left boundary link touches only one plaquette, so the excitation
    is a single vison rather than an internal-link vison pair.
    """
    n_rows, n_cols = model.shape
    if n_rows < 2 or n_cols < 2:
        raise ValueError("The bottom-left vison construction requires L >= 2.")
    site_row = n_rows - 2
    site_col = 0
    phase = _z2_phase_for_direction(model.shape, site_row, site_col, "down")
    graphdef, params, model_state = nnx.split(model, nnx.Param, ...)
    tensors = nnx.to_pure_dict(params)["tensors"]
    tensors = {
        r: {c: jnp.array(tensor) for c, tensor in row_dict.items()}
        for r, row_dict in tensors.items()
    }
    tensors[site_row][site_col] = (
        tensors[site_row][site_col] * phase[None, :, None, None, None, None]
    )
    return nnx.merge(graphdef, {"tensors": tensors}, model_state)


def _default_output_path(
    *,
    L: int,
    g: float,
    bond_dim: int,
    T: float,
) -> Path:
    """Build the default JSON output path for one vison run."""
    g_token = format(g, ".3f").replace(".", "p")
    t_token = format(T, ".3f").replace(".", "p")
    return (
        Path(__file__).resolve().parent
        / f"z2_vison_propagation_L{L}_g{g_token}_Dk{bond_dim}_T{t_token}.json"
    )


def _save_result(result: dict, output_path: Path) -> None:
    """Write one run to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    logger.info("Saved %s", output_path)


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
    n_steps: int,
    dt: float,
    diag_shift: float,
    seed: int,
) -> TDVPDriver:
    """Run imaginary-time SR for the ground state."""
    driver = TDVPDriver(
        model,
        hamiltonian,
        preconditioner=SRPreconditioner(diag_shift=diag_shift),
        dt=dt,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(seed),
        n_samples=n_samples,
        n_chains=min(64, n_samples),
        full_gradient=False,
    )
    print(
        "[ground_state] step tau dt wall_time energy energy_err energy_var",
        flush=True,
    )
    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        driver.run(dt)
        wall_time = time.perf_counter() - t0
        energy = driver.energy
        energy_mean = float(energy.mean.real)
        energy_error = float(energy.error_of_mean.real)
        energy_variance = float(energy.variance.real)
        print(
            (
                f"[ground_state] {step:4d} {driver.t:.6f} {dt:.6f} "
                f"{wall_time:.3f} {energy_mean:.10f} {energy_error:.6f} "
                f"{energy_variance:.6f}"
            ),
            flush=True,
        )
        logger.info(
            "[ground_state] step=%4d tau=%.3f E=%.10f err=%.6f var=%.6f",
            step,
            driver.t,
            energy_mean,
            energy_error,
            energy_variance,
        )
    return driver


def run_real_time(
    model: GIPEPS,
    hamiltonian: GILocalHamiltonian,
    *,
    n_samples: int,
    T: float,
    dt: float,
    diag_shift: float,
    seed: int,
) -> tuple[TDVPDriver, dict[str, list]]:
    """Run real-time TDVP from the vison state."""
    observables = build_fig5a_plaquette_observables(model.shape)
    driver = TDVPDriver(
        model,
        hamiltonian,
        observables=observables,
        preconditioner=SRPreconditioner(diag_shift=diag_shift),
        dt=dt,
        time_unit=RealTimeUnit(),
        integrator=RK4(),
        sampler_key=jax.random.key(seed),
        n_samples=n_samples,
        n_chains=min(64, n_samples),
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
    }
    print(
        "[real_time] step t dt wall_time energy energy_err energy_var "
        "drift_percent p00 p00_err p01 p01_err p22 p22_err",
        flush=True,
    )
    energy, observable_stats = _measure_driver(driver)
    reference_energy = float(energy.mean.real)
    selected_means = [float(stat.mean.real) for stat in observable_stats]
    selected_errors = [float(stat.error_of_mean.real) for stat in observable_stats]
    series["step"].append(0)
    series["time"].append(0.0)
    series["energy_mean"].append(reference_energy)
    series["energy_error"].append(float(energy.error_of_mean.real))
    series["energy_variance"].append(float(energy.variance.real))
    series["energy_drift_percent"].append(0.0)
    series["selected_plaquette_mean"].append(selected_means)
    series["selected_plaquette_error"].append(selected_errors)
    print(
        (
            f"[real_time] {0:4d} {0.0:.6f} {dt:.6f} {0.0:.3f} "
            f"{reference_energy:.10f} {float(energy.error_of_mean.real):.6f} "
            f"{float(energy.variance.real):.6f} {0.0:.6f} "
            f"{selected_means[0]:.10f} {selected_errors[0]:.6f} "
            f"{selected_means[1]:.10f} {selected_errors[1]:.6f} "
            f"{selected_means[2]:.10f} {selected_errors[2]:.6f}"
        ),
        flush=True,
    )
    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        driver.run(dt)
        wall_time = time.perf_counter() - t0
        energy = driver.energy
        energy_mean = float(energy.mean.real)
        energy_error = float(energy.error_of_mean.real)
        energy_variance = float(energy.variance.real)
        drift = abs(energy_mean - reference_energy) / abs(reference_energy) * 100.0
        selected_means = [float(stat.mean.real) for stat in driver.observable_stats]
        selected_errors = [
            float(stat.error_of_mean.real) for stat in driver.observable_stats
        ]
        series["step"].append(step)
        series["time"].append(float(driver.t))
        series["energy_mean"].append(energy_mean)
        series["energy_error"].append(energy_error)
        series["energy_variance"].append(energy_variance)
        series["energy_drift_percent"].append(drift)
        series["selected_plaquette_mean"].append(selected_means)
        series["selected_plaquette_error"].append(selected_errors)
        print(
            (
                f"[real_time] {step:4d} {driver.t:.6f} {dt:.6f} "
                f"{wall_time:.3f} {energy_mean:.10f} {energy_error:.6f} "
                f"{energy_variance:.6f} {drift:.6f} "
                f"{selected_means[0]:.10f} {selected_errors[0]:.6f} "
                f"{selected_means[1]:.10f} {selected_errors[1]:.6f} "
                f"{selected_means[2]:.10f} {selected_errors[2]:.6f}"
            ),
            flush=True,
        )
        logger.info(
            "[rt] step=%4d t=%.3f E=%.10f err=%.6f drift=%.4f%%",
            step,
            driver.t,
            energy_mean,
            energy_error,
            drift,
        )
    return driver, series


def run_benchmark(
    *,
    L: int = DEFAULT_L,
    h: float = DEFAULT_H,
    g: float = DEFAULT_G,
    bond_dim: int = DEFAULT_BOND_DIM,
    n_samples_gs: int = DEFAULT_N_SAMPLES_GS,
    n_steps_gs: int = DEFAULT_N_STEPS_GS,
    dt_gs: float = DEFAULT_DT_GS,
    gs_diag_shift: float = DEFAULT_GS_DIAG_SHIFT,
    n_samples_rt: int = DEFAULT_N_SAMPLES_RT,
    T: float = DEFAULT_T,
    dt_rt: float = DEFAULT_DT_RT,
    rt_diag_shift: float = DEFAULT_RT_DIAG_SHIFT,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run the full vison benchmark and return one JSON-ready result dict."""
    if L < 4:
        raise ValueError("Fig. 5(a) requires L >= 4.")
    if n_steps_gs < 1:
        raise ValueError("n_steps_gs must be positive.")
    if dt_gs <= 0.0 or dt_rt <= 0.0:
        raise ValueError("Time steps must be positive.")
    if T <= 0.0:
        raise ValueError("T must be positive.")
    shape = (L, L)
    logger.info("=" * 72)
    logger.info("Z2 vison propagation")
    logger.info("=" * 72)
    logger.info("L=%d h=%.3f g=%.3f D_k=%d", L, h, g, bond_dim)
    logger.info("ground state: n_samples=%d n_steps=%d dt=%.4f", n_samples_gs, n_steps_gs, dt_gs)
    logger.info("real time:    n_samples=%d T=%.3f dt=%.4f", n_samples_rt, T, dt_rt)

    hamiltonian = build_z2_hamiltonian(shape, h=h, g=g)
    ground_state_driver = run_ground_state(
        build_model(shape, bond_dim=bond_dim, seed=seed),
        hamiltonian,
        n_samples=n_samples_gs,
        n_steps=n_steps_gs,
        dt=dt_gs,
        diag_shift=gs_diag_shift,
        seed=seed,
    )
    vison_model = create_bottom_left_vison(ground_state_driver.model)
    real_time_driver, real_time_series = run_real_time(
        vison_model,
        hamiltonian,
        n_samples=n_samples_rt,
        T=T,
        dt=dt_rt,
        diag_shift=rt_diag_shift,
        seed=seed + 1,
    )
    ground_state_energy = ground_state_driver.energy
    final_real_time_energy = real_time_driver.energy
    return {
        "problem": {
            "gauge_group": "Z2",
            "L": L,
            "shape": shape,
            "h": h,
            "g": g,
            "bond_dim": bond_dim,
            "boundary_method": "ZipUp",
            "boundary_dimension": 3 * bond_dim,
            "seed": seed,
        },
        "selected_plaquettes": [list(site) for site in FIG5A_PLAQUETTES],
        "ground_state": {
            "n_samples": n_samples_gs,
            "n_steps": n_steps_gs,
            "dt": dt_gs,
            "diag_shift": gs_diag_shift,
            "final_energy_mean": float(ground_state_energy.mean.real),
            "final_energy_error": float(ground_state_energy.error_of_mean.real),
            "final_energy_variance": float(ground_state_energy.variance.real),
        },
        "vison": {
            "operator": "sigma_z",
            "orientation": "v",
            "link_row": L - 2,
            "link_col": 0,
            "plaquettes": [[L - 2, 0]],
        },
        "real_time": {
            "n_samples": n_samples_rt,
            "T": T,
            "dt": dt_rt,
            "diag_shift": rt_diag_shift,
            **real_time_series,
        },
        "summary": {
            "final_real_time_energy_mean": float(final_real_time_energy.mean.real),
            "final_real_time_energy_error": float(final_real_time_energy.error_of_mean.real),
            "final_real_time_energy_variance": float(final_real_time_energy.variance.real),
            "final_energy_drift_percent": real_time_series["energy_drift_percent"][-1],
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PEPS side of the 2025 Fig. 5(a) Z2 vison benchmark.",
    )
    parser.add_argument("--L", type=int, default=DEFAULT_L)
    parser.add_argument("--h", type=float, default=DEFAULT_H)
    parser.add_argument("--g", type=float, default=DEFAULT_G)
    parser.add_argument("--bond-dim", type=int, default=DEFAULT_BOND_DIM)
    parser.add_argument("--n-samples-gs", type=int, default=DEFAULT_N_SAMPLES_GS)
    parser.add_argument("--n-steps-gs", type=int, default=DEFAULT_N_STEPS_GS)
    parser.add_argument("--dt-gs", type=float, default=DEFAULT_DT_GS)
    parser.add_argument("--gs-diag-shift", type=float, default=DEFAULT_GS_DIAG_SHIFT)
    parser.add_argument("--n-samples-rt", type=int, default=DEFAULT_N_SAMPLES_RT)
    parser.add_argument("--T", type=float, default=DEFAULT_T)
    parser.add_argument("--dt-rt", type=float, default=DEFAULT_DT_RT)
    parser.add_argument("--rt-diag-shift", type=float, default=DEFAULT_RT_DIAG_SHIFT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_benchmark(
        L=args.L,
        h=args.h,
        g=args.g,
        bond_dim=args.bond_dim,
        n_samples_gs=args.n_samples_gs,
        n_steps_gs=args.n_steps_gs,
        dt_gs=args.dt_gs,
        gs_diag_shift=args.gs_diag_shift,
        n_samples_rt=args.n_samples_rt,
        T=args.T,
        dt_rt=args.dt_rt,
        rt_diag_shift=args.rt_diag_shift,
        seed=args.seed,
    )
    output_path = args.output or _default_output_path(
        L=args.L,
        g=args.g,
        bond_dim=args.bond_dim,
        T=args.T,
    )
    _save_result(result, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
