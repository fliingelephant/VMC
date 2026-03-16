"""Benchmark AFM Heisenberg ground-state optimization with SR, Adam, and adaptive SR.

The script writes one JSON file per method so each run in ``main()`` can be
commented out independently.
"""

from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import netket as nk
from flax import nnx
from jax.flatten_util import ravel_pytree
from netket import stats as nkstats

from vmc.core import _sample_counts, _trim_samples, make_mc_sampler
from vmc.drivers import ImaginaryTimeUnit, TDVPDriver
from vmc.operators import (
    DiagonalOperator,
    HorizontalTwoSiteOperator,
    LocalHamiltonian,
    VerticalTwoSiteOperator,
)
from vmc.peps import PEPS, build_mc_kernels
from vmc.peps.common.strategy import Variational
from vmc.preconditioners import (
    DirectSolve,
    MetricsConfig,
    SRPreconditioner,
    solve_cholesky,
)
from vmc.preconditioners.preconditioners import _adjoint_matvec, _reorder_updates
from vmc.qgt import ParameterSpace, SlicedJacobian
from vmc.qgt.jacobian import SliceOrdering
from vmc.utils import _tree_add_scaled


L = 4
SHAPE = (L, L)
J = 1.0

BOND_DIM = 3
BOUNDARY_DIM = 9

N_SAMPLES = 8192
N_CHAINS = 128
SEED = 42

SR_FIXED_STEPS = 200
SR_FIXED_DT = 0.02
SR_DIAG_SHIFT = 1e-8

ADAM_STEPS = SR_FIXED_STEPS
ADAM_LEARNING_RATE = 0.02
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

SR_ADAPTIVE_TARGET_FS_NORM = SR_FIXED_DT
SR_ADAPTIVE_DT_MIN = 1e-4
SR_ADAPTIVE_DT_MAX = SR_FIXED_DT * 2
SR_ADAPTIVE_MAX_STEPS = 1000

SR_METRICS_CONFIG = MetricsConfig(
    record_FS_norm=True,
    record_TDVP_residual=True,
    record_SR_solve_residual=True,
    record_step_wall_time=True,
)


def build_heisenberg_2d(shape: tuple[int, int], coupling: float) -> LocalHamiltonian:
    """Build the 2D antiferromagnetic Heisenberg Hamiltonian."""
    szsz = 0.25 * coupling * jnp.array([1, -1, -1, 1], dtype=jnp.complex128)
    exchange = 0.5 * coupling * jnp.array(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=jnp.complex128,
    )
    terms = []
    for row in range(shape[0]):
        for col in range(shape[1]):
            if col + 1 < shape[1]:
                terms.append(DiagonalOperator(((row, col), (row, col + 1)), szsz))
                terms.append(HorizontalTwoSiteOperator(row=row, col=col, op=exchange))
            if row + 1 < shape[0]:
                terms.append(DiagonalOperator(((row, col), (row + 1, col)), szsz))
                terms.append(VerticalTwoSiteOperator(row=row, col=col, op=exchange))
    return LocalHamiltonian(shape=shape, terms=tuple(terms))


def build_nn_sxsx_observable(shape: tuple[int, int]) -> LocalHamiltonian:
    """Build average nearest-neighbor SxSx."""
    n_bonds = shape[0] * (shape[1] - 1) + (shape[0] - 1) * shape[1]
    sxsx = 0.25 * jnp.array(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=jnp.complex128,
    ) / n_bonds
    terms = []
    for row in range(shape[0]):
        for col in range(shape[1]):
            if col + 1 < shape[1]:
                terms.append(HorizontalTwoSiteOperator(row=row, col=col, op=sxsx))
            if row + 1 < shape[0]:
                terms.append(VerticalTwoSiteOperator(row=row, col=col, op=sxsx))
    return LocalHamiltonian(shape=shape, terms=tuple(terms))


def build_nn_sysy_observable(shape: tuple[int, int]) -> LocalHamiltonian:
    """Build average nearest-neighbor SySy."""
    n_bonds = shape[0] * (shape[1] - 1) + (shape[0] - 1) * shape[1]
    sysy = 0.25 * jnp.array(
        [
            [0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
        ],
        dtype=jnp.complex128,
    ) / n_bonds
    terms = []
    for row in range(shape[0]):
        for col in range(shape[1]):
            if col + 1 < shape[1]:
                terms.append(HorizontalTwoSiteOperator(row=row, col=col, op=sysy))
            if row + 1 < shape[0]:
                terms.append(VerticalTwoSiteOperator(row=row, col=col, op=sysy))
    return LocalHamiltonian(shape=shape, terms=tuple(terms))


def build_nn_szsz_observable(shape: tuple[int, int]) -> LocalHamiltonian:
    """Build average nearest-neighbor SzSz."""
    n_bonds = shape[0] * (shape[1] - 1) + (shape[0] - 1) * shape[1]
    szsz = 0.25 * jnp.array([1, -1, -1, 1], dtype=jnp.complex128) / n_bonds
    terms = []
    for row in range(shape[0]):
        for col in range(shape[1]):
            if col + 1 < shape[1]:
                terms.append(DiagonalOperator(((row, col), (row, col + 1)), szsz))
            if row + 1 < shape[0]:
                terms.append(DiagonalOperator(((row, col), (row + 1, col)), szsz))
    return LocalHamiltonian(shape=shape, terms=tuple(terms))


def build_problem() -> tuple[LocalHamiltonian, tuple[LocalHamiltonian, ...], float]:
    """Build the Hamiltonian, observables, and exact energy."""
    graph = nk.graph.Grid(extent=SHAPE, pbc=False)
    hilbert = nk.hilbert.Spin(s=0.5, N=SHAPE[0] * SHAPE[1])
    # NetKet's Heisenberg operator uses Pauli normalization, so the local
    # S.S Hamiltonian defined above matches J/4 with sign_rule disabled.
    exact_energy = float(
        nk.exact.lanczos_ed(
            nk.operator.Heisenberg(
                hilbert,
                graph,
                J=0.25 * J,
                sign_rule=False,
                dtype=jnp.complex128,
            ),
            k=1,
        )[0].real
    )
    return (
        build_heisenberg_2d(SHAPE, J),
        (
            build_nn_sxsx_observable(SHAPE),
            build_nn_sysy_observable(SHAPE),
            build_nn_szsz_observable(SHAPE),
        ),
        exact_energy,
    )


def build_model(seed: int) -> PEPS:
    """Build a fresh PEPS model."""
    return PEPS(
        rngs=nnx.Rngs(seed),
        shape=SHAPE,
        bond_dim=BOND_DIM,
        contraction_strategy=Variational(BOUNDARY_DIM),
    )


def append_series(series: dict[str, list], **values) -> None:
    """Append one row into a columnar series dict."""
    for key, value in values.items():
        series.setdefault(key, []).append(value)


def benchmark_output_dir() -> Path:
    """Build the default output folder for the current benchmark settings."""
    return (
        Path(__file__).resolve().parent
        / f"heisenberg_benchmark_{L}x{L}_J{J:.0f}_ns{N_SAMPLES}_{SR_FIXED_STEPS}"
    )


def save_run(
    output_path: Path,
    *,
    exact_energy: float,
    config_data: dict,
    series: dict[str, list],
) -> None:
    """Write one benchmark run to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "final_step": series["step"][-1],
        "final_energy_mean": series["energy_mean"][-1],
        "final_energy_error": series["energy_error"][-1],
        "final_nn_sxsx_mean": series["nn_sxsx_mean"][-1],
        "final_nn_sysy_mean": series["nn_sysy_mean"][-1],
        "final_nn_szsz_mean": series["nn_szsz_mean"][-1],
    }
    if "imaginary_time" in series:
        summary["final_imaginary_time"] = series["imaginary_time"][-1]
    output_path.write_text(
        json.dumps(
            {
                "problem": {
                    "shape": SHAPE,
                    "J": J,
                    "bond_dim": BOND_DIM,
                    "boundary_method": "Variational",
                    "boundary_dimension": BOUNDARY_DIM,
                    "n_samples": N_SAMPLES,
                    "n_chains": N_CHAINS,
                    "seed": SEED,
                    "exact_energy": exact_energy,
                },
                "config": config_data,
                "series": series,
                "summary": summary,
            },
            indent=2,
        )
    )
    print(f"Saved {output_path}", flush=True)


def plot_observables_vs_step(
    plt,
    outputs: dict[str, dict],
    output_path: Path,
) -> None:
    """Plot energy and nearest-neighbor correlators against optimization step."""
    colors = {
        "sr_fixed": "#1f4e79",
        "adam": "#b07a00",
        "sr_adaptive": "#7a3e9d",
        "exact": "#b22222",
    }
    exact_energy = outputs["sr_fixed"]["problem"]["exact_energy"]
    fig, axes = plt.subplots(4, 1, figsize=(9, 13.0), sharex=True)
    specs = (
        ("energy_mean", "energy_error", "Energy"),
        ("nn_sxsx_mean", "nn_sxsx_error", "nn_sxsx"),
        ("nn_sysy_mean", "nn_sysy_error", "nn_sysy"),
        ("nn_szsz_mean", "nn_szsz_error", "nn_szsz"),
    )
    for ax, (y_key, err_key, ylabel) in zip(axes, specs):
        for name, label in (
            ("sr_fixed", "SR fixed"),
            ("adam", "Adam"),
            ("sr_adaptive", "SR adaptive"),
        ):
            series = outputs[name]["series"]
            ax.errorbar(
                series["step"],
                series[y_key],
                yerr=series[err_key],
                label=label,
                color=colors[name],
                linewidth=1.4,
                elinewidth=0.8,
                capsize=2,
                alpha=0.95,
            )
        if y_key == "energy_mean":
            ax.axhline(
                exact_energy,
                color=colors["exact"],
                linestyle="--",
                linewidth=1.2,
                label=f"Exact E = {exact_energy:.6f}",
            )
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("Optimization step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}", flush=True)


def plot_observables_vs_imaginary_time(
    plt,
    outputs: dict[str, dict],
    output_path: Path,
) -> None:
    """Plot energy and nearest-neighbor correlators against imaginary time for SR runs."""
    colors = {"sr_fixed": "#1f4e79", "sr_adaptive": "#7a3e9d", "exact": "#b22222"}
    exact_energy = outputs["sr_fixed"]["problem"]["exact_energy"]
    fig, axes = plt.subplots(4, 1, figsize=(9, 13.0), sharex=True)
    specs = (
        ("energy_mean", "energy_error", "Energy"),
        ("nn_sxsx_mean", "nn_sxsx_error", "nn_sxsx"),
        ("nn_sysy_mean", "nn_sysy_error", "nn_sysy"),
        ("nn_szsz_mean", "nn_szsz_error", "nn_szsz"),
    )
    for ax, (y_key, err_key, ylabel) in zip(axes, specs):
        for name, label in (("sr_fixed", "SR fixed"), ("sr_adaptive", "SR adaptive")):
            series = outputs[name]["series"]
            ax.errorbar(
                series["imaginary_time"],
                series[y_key],
                yerr=series[err_key],
                label=label,
                color=colors[name],
                linewidth=1.4,
                elinewidth=0.8,
                capsize=2,
                alpha=0.95,
            )
        if y_key == "energy_mean":
            ax.axhline(
                exact_energy,
                color=colors["exact"],
                linestyle="--",
                linewidth=1.2,
                label=f"Exact E = {exact_energy:.6f}",
            )
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("Imaginary time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}", flush=True)


def write_figures(output_dir: Path) -> None:
    """Write benchmark figures next to the JSON outputs."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not available; skipping figures", flush=True)
        return

    outputs = {
        name: json.loads((output_dir / f"{name}.json").read_text())
        for name in ("sr_fixed", "adam", "sr_adaptive")
    }
    plot_observables_vs_step(
        plt,
        outputs,
        output_dir / "observables_vs_step_errorbars.png",
    )
    plot_observables_vs_imaginary_time(
        plt,
        outputs,
        output_dir / "observables_vs_imaginary_time_errorbars.png",
    )


def run_sr(
    hamiltonian: LocalHamiltonian,
    observables: tuple[LocalHamiltonian, ...],
    exact_energy: float,
    output_path: Path,
    *,
    n_steps: int | None = None,
    dt: float = SR_FIXED_DT,
    target_time: float | None = None,
) -> None:
    """Run fixed-step or adaptive-step SR."""
    adaptive = target_time is not None
    assert adaptive != (n_steps is not None), (
        "Provide exactly one of n_steps or target_time."
    )

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
            "nn_sxsx nn_sxsx_err nn_sysy nn_sysy_err nn_szsz nn_szsz_err "
            "applied_FS_step_norm_squared FS_norm_squared TDVP_residual "
            "SR_solve_residual"
        ),
        flush=True,
    )

    if adaptive:
        time_derivative = jax.jit(driver._time_derivative)
        key = driver._sampler_key
        config_states = driver._sampler_configuration.reshape(driver.n_chains, -1)
        tensors = driver._tensors
        t = float(driver.t)
        step = 0
        while step < SR_ADAPTIVE_MAX_STEPS and t < target_time:
            step += 1
            t0 = time.perf_counter()
            updates, (key, config_states), (local_estimates, metrics) = time_derivative(
                tensors,
                t,
                (key, config_states),
            )
            jax.block_until_ready(local_estimates)
            FS_norm_squared = float(metrics["FS_norm_squared"])
            step_dt = SR_ADAPTIVE_TARGET_FS_NORM / max(
                FS_norm_squared**0.5,
                1e-30,
            )
            step_dt = min(SR_ADAPTIVE_DT_MAX, max(SR_ADAPTIVE_DT_MIN, step_dt))
            step_dt = min(step_dt, target_time - t)
            tensors = _tree_add_scaled(tensors, updates, step_dt)
            t += step_dt
            wall_time = time.perf_counter() - t0

            energy = nkstats.statistics(local_estimates[:, 0])
            nn_sxsx = nkstats.statistics(local_estimates[:, 1])
            nn_sysy = nkstats.statistics(local_estimates[:, 2])
            nn_szsz = nkstats.statistics(local_estimates[:, 3])
            row = {
                "step": step,
                "imaginary_time": t,
                "dt": step_dt,
                "step_wall_time": wall_time,
                "energy_mean": float(energy.mean.real),
                "energy_error": float(energy.error_of_mean.real),
                "energy_variance": float(energy.variance.real),
                "nn_sxsx_mean": float(nn_sxsx.mean.real),
                "nn_sxsx_error": float(nn_sxsx.error_of_mean.real),
                "nn_sysy_mean": float(nn_sysy.mean.real),
                "nn_sysy_error": float(nn_sysy.error_of_mean.real),
                "nn_szsz_mean": float(nn_szsz.mean.real),
                "nn_szsz_error": float(nn_szsz.error_of_mean.real),
                "applied_FS_step_norm_squared": step_dt**2 * FS_norm_squared,
                "FS_norm_squared": FS_norm_squared,
                "TDVP_residual": float(metrics["TDVP_residual"]),
                "SR_solve_residual": float(metrics["SR_solve_residual"]),
            }
            append_series(series, **row)
            print(
                (
                    f"[{label}] {row['step']:3d} {row['imaginary_time']:.6f} "
                    f"{row['dt']:.6f} {row['step_wall_time']:.3f} "
                    f"{row['energy_mean']:.10f} {row['energy_error']:.6f} "
                    f"{row['energy_variance']:.6f} {row['nn_sxsx_mean']:.10f} "
                    f"{row['nn_sxsx_error']:.6f} {row['nn_sysy_mean']:.10f} "
                    f"{row['nn_sysy_error']:.6f} {row['nn_szsz_mean']:.10f} "
                    f"{row['nn_szsz_error']:.6f} "
                    f"{row['applied_FS_step_norm_squared']:.6e} "
                    f"{row['FS_norm_squared']:.6e} "
                    f"{row['TDVP_residual']:.6e} "
                    f"{row['SR_solve_residual']:.6e}"
                ),
                flush=True,
            )

        save_run(
            output_path,
            exact_energy=exact_energy,
            config_data={
                "method": label,
                "diag_shift": SR_DIAG_SHIFT,
                "target_time": target_time,
                "target_FS_step_norm": SR_ADAPTIVE_TARGET_FS_NORM,
                "dt_min": SR_ADAPTIVE_DT_MIN,
                "dt_max": SR_ADAPTIVE_DT_MAX,
                "max_steps": SR_ADAPTIVE_MAX_STEPS,
            },
            series=series,
        )
        return

    for step in range(1, n_steps + 1):
        driver.run(dt)
        metrics = driver.metrics
        energy = driver.energy
        nn_sxsx, nn_sysy, nn_szsz = driver.observable_stats
        FS_norm_squared = float(metrics["FS_norm_squared"])
        row = {
            "step": step,
            "imaginary_time": float(driver.t),
            "dt": dt,
            "step_wall_time": float(metrics["step_wall_time"]),
            "energy_mean": float(energy.mean.real),
            "energy_error": float(energy.error_of_mean.real),
            "energy_variance": float(energy.variance.real),
            "nn_sxsx_mean": float(nn_sxsx.mean.real),
            "nn_sxsx_error": float(nn_sxsx.error_of_mean.real),
            "nn_sysy_mean": float(nn_sysy.mean.real),
            "nn_sysy_error": float(nn_sysy.error_of_mean.real),
            "nn_szsz_mean": float(nn_szsz.mean.real),
            "nn_szsz_error": float(nn_szsz.error_of_mean.real),
            "applied_FS_step_norm_squared": dt**2 * FS_norm_squared,
            "FS_norm_squared": FS_norm_squared,
            "TDVP_residual": float(metrics["TDVP_residual"]),
            "SR_solve_residual": float(metrics["SR_solve_residual"]),
        }
        append_series(series, **row)
        print(
            (
                f"[{label}] {row['step']:3d} {row['imaginary_time']:.6f} "
                f"{row['dt']:.6f} {row['step_wall_time']:.3f} "
                f"{row['energy_mean']:.10f} {row['energy_error']:.6f} "
                f"{row['energy_variance']:.6f} {row['nn_sxsx_mean']:.10f} "
                f"{row['nn_sxsx_error']:.6f} {row['nn_sysy_mean']:.10f} "
                f"{row['nn_sysy_error']:.6f} {row['nn_szsz_mean']:.10f} "
                f"{row['nn_szsz_error']:.6f} "
                f"{row['applied_FS_step_norm_squared']:.6e} "
                f"{row['FS_norm_squared']:.6e} "
                f"{row['TDVP_residual']:.6e} "
                f"{row['SR_solve_residual']:.6e}"
            ),
            flush=True,
        )

    save_run(
        output_path,
        exact_energy=exact_energy,
        config_data={
            "method": label,
            "diag_shift": SR_DIAG_SHIFT,
            "dt": dt,
            "n_steps": n_steps,
        },
        series=series,
    )


def run_adam(
    hamiltonian: LocalHamiltonian,
    observables: tuple[LocalHamiltonian, ...],
    exact_energy: float,
    output_path: Path,
) -> None:
    """Run Adam on the bare parameter-space gradient."""
    label = output_path.stem
    model = build_model(SEED)
    init_cache, transition, estimate = build_mc_kernels(
        model,
        hamiltonian,
        observables=observables,
        full_gradient=False,
    )
    mc_sampler = make_mc_sampler(transition, estimate)
    _, num_chains, chain_length, total_samples = _sample_counts(
        N_SAMPLES, N_CHAINS
    )
    ordering = SliceOrdering()
    params_per_site_tuple = model.params_per_site
    sliced_dims_tuple = model.sliced_dims
    grad_factor = ImaginaryTimeUnit().grad_factor

    params = [[jnp.asarray(tensor) for tensor in row] for row in model.tensors]
    flat_params, unravel = ravel_pytree(params)
    m = jnp.zeros_like(flat_params)
    v = jnp.zeros(
        flat_params.shape,
        dtype=jnp.real(jnp.zeros((), dtype=flat_params.dtype)).dtype,
    )
    key = jax.random.key(SEED)
    key, init_key = jax.random.split(key)
    config_states = model.random_physical_configuration(
        init_key, n_samples=N_CHAINS
    ).reshape(N_CHAINS, -1)

    def raw_gradient_step(params, key, config_states):
        key, chain_key = jax.random.split(key)
        chain_keys = jax.random.split(chain_key, num_chains)
        cache = init_cache(params, config_states)
        (config_states, _, _), (samples_hist, estimates) = mc_sampler(
            params,
            config_states,
            chain_keys,
            cache,
            n_steps=chain_length,
        )
        samples = _trim_samples(samples_hist, total_samples, N_SAMPLES)
        o = _trim_samples(
            estimates.local_log_derivatives,
            total_samples,
            N_SAMPLES,
        )
        p = _trim_samples(
            estimates.active_slice_indices,
            total_samples,
            N_SAMPLES,
        )
        local_estimates = _trim_samples(
            estimates.local_estimate,
            total_samples,
            N_SAMPLES,
        )
        dv = grad_factor * (
            local_estimates[:, 0] - jnp.mean(local_estimates[:, 0])
        ) / samples.shape[0]
        grad_flat = _adjoint_matvec(
            SlicedJacobian(o, p, sliced_dims_tuple, ordering),
            dv,
        )
        grad_flat = _reorder_updates(
            ordering,
            grad_flat,
            params_per_site_tuple,
            sliced_dims_tuple,
        )
        return grad_flat, key, config_states, local_estimates

    raw_gradient_step = jax.jit(raw_gradient_step)
    series: dict[str, list] = {}
    print(
        (
            f"[{label}] step wall_time energy energy_err energy_var "
            "nn_sxsx nn_sxsx_err nn_sysy nn_sysy_err nn_szsz nn_szsz_err"
        ),
        flush=True,
    )
    for step in range(1, ADAM_STEPS + 1):
        t0 = time.perf_counter()
        grad_flat, key, config_states, local_estimates = raw_gradient_step(
            params,
            key,
            config_states,
        )
        jax.block_until_ready(local_estimates)
        m = ADAM_BETA1 * m + (1.0 - ADAM_BETA1) * grad_flat
        v = ADAM_BETA2 * v + (1.0 - ADAM_BETA2) * jnp.real(
            grad_flat.conj() * grad_flat
        )
        flat_params = flat_params + ADAM_LEARNING_RATE * (
            m / (1.0 - ADAM_BETA1**step)
        ) / (jnp.sqrt(v / (1.0 - ADAM_BETA2**step)) + ADAM_EPS)
        params = unravel(flat_params)
        wall_time = time.perf_counter() - t0

        energy = nkstats.statistics(local_estimates[:, 0])
        nn_sxsx = nkstats.statistics(local_estimates[:, 1])
        nn_sysy = nkstats.statistics(local_estimates[:, 2])
        nn_szsz = nkstats.statistics(local_estimates[:, 3])
        row = {
            "step": step,
            "step_wall_time": wall_time,
            "energy_mean": float(energy.mean.real),
            "energy_error": float(energy.error_of_mean.real),
            "energy_variance": float(energy.variance.real),
            "nn_sxsx_mean": float(nn_sxsx.mean.real),
            "nn_sxsx_error": float(nn_sxsx.error_of_mean.real),
            "nn_sysy_mean": float(nn_sysy.mean.real),
            "nn_sysy_error": float(nn_sysy.error_of_mean.real),
            "nn_szsz_mean": float(nn_szsz.mean.real),
            "nn_szsz_error": float(nn_szsz.error_of_mean.real),
        }
        append_series(series, **row)
        print(
            (
                f"[{label}] {row['step']:3d} {row['step_wall_time']:.3f} "
                f"{row['energy_mean']:.10f} {row['energy_error']:.6f} "
                f"{row['energy_variance']:.6f} {row['nn_sxsx_mean']:.10f} "
                f"{row['nn_sxsx_error']:.6f} {row['nn_sysy_mean']:.10f} "
                f"{row['nn_sysy_error']:.6f} {row['nn_szsz_mean']:.10f} "
                f"{row['nn_szsz_error']:.6f}"
            ),
            flush=True,
        )

    save_run(
        output_path,
        exact_energy=exact_energy,
        config_data={
            "method": label,
            "learning_rate": ADAM_LEARNING_RATE,
            "beta1": ADAM_BETA1,
            "beta2": ADAM_BETA2,
            "eps": ADAM_EPS,
            "n_steps": ADAM_STEPS,
        },
        series=series,
    )


def main() -> None:
    """Run the benchmark and write one JSON file per method."""
    hamiltonian, observables, exact_energy = build_problem()
    output_dir = benchmark_output_dir()
    print(
        (
            f"Benchmarking AFM Heisenberg on {SHAPE}, J={J:.3f}, "
            f"D={BOND_DIM}, Dc={BOUNDARY_DIM}, nsamples={N_SAMPLES}"
        ),
        flush=True,
    )
    print(f"Exact ground-state energy: {exact_energy:.10f}", flush=True)

    # Comment out any run below to execute methods separately.
    run_sr(
        hamiltonian,
        observables,
        exact_energy,
        output_dir / "sr_fixed.json",
        n_steps=SR_FIXED_STEPS,
        dt=SR_FIXED_DT,
    )
    run_adam(
        hamiltonian,
        observables,
        exact_energy,
        output_dir / "adam.json",
    )
    run_sr(
        hamiltonian,
        observables,
        exact_energy,
        output_dir / "sr_adaptive.json",
        target_time=SR_FIXED_STEPS * SR_FIXED_DT,
    )
    write_figures(output_dir)


if __name__ == "__main__":
    main()
