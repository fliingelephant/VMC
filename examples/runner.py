"""Shared run infrastructure for PEPS-VMC example scripts."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from vmc.operators.time_dependent import TimeDependentHamiltonian
from vmc.preconditioners import MetricsConfig, solve_cg, solve_cholesky, solve_svd

SOLVERS = {"cholesky": solve_cholesky, "svd": solve_svd, "cg": solve_cg}

DEFAULT_METRICS_CONFIG = MetricsConfig(
    record_FS_norm=True,
    record_TDVP_residual=True,
    record_SR_solve_residual=True,
    record_step_wall_time=True,
)


def resolve_solver(name: str):
    """Map a solver name to the corresponding function."""
    return SOLVERS[name]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add standard CLI arguments shared across example scripts."""
    parser.add_argument("--n-samples", type=int, default=10240)
    parser.add_argument("--n-chains", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--boundary-dim", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--diag-shift", type=float, default=1e-4)
    parser.add_argument("--solver", choices=("cholesky", "svd", "cg"), default="cholesky")
    parser.add_argument("--solver-space", choices=("sr", "minsr"), default="sr")
    parser.add_argument("--full-gradient", action="store_true")
    parser.add_argument("--gauge-removal", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--T-final", type=float, default=None, dest="T_final")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for checkpoints and series")


def save_checkpoint(run_dir, driver, step, *, series=None, **metadata):
    """Save driver state to run_dir/latest/ (orbax) + latest.json. Writes atomically."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "tensors": {
            str(row): {str(col): t for col, t in rd.items()}
            for row, rd in driver._tensors.items()
        },
        "sampler_key": driver._sampler_key,
        "sampler_configuration": driver._sampler_configuration,
    }
    ckptr = ocp.PyTreeCheckpointer()
    ckpt_path = run_dir / "latest"
    if ckpt_path.exists():
        import shutil
        shutil.rmtree(ckpt_path)
    ckptr.save(ckpt_path, state)

    json_data = {
        "step": int(step),
        "time": float(driver.t),
        "config": _extract_config(driver, metadata.pop("config", None)),
        **metadata,
    }
    if series is not None:
        json_data["series"] = series
    _atomic_write_json(run_dir / "latest.json", json_data)


def load_checkpoint(run_dir, driver):
    """Restore driver state from run_dir/latest/ (orbax) + latest.json. Returns metadata."""
    run_dir = Path(run_dir)

    target = {
        "tensors": {
            str(row): {str(col): t for col, t in rd.items()}
            for row, rd in driver._tensors.items()
        },
        "sampler_key": driver._sampler_key,
        "sampler_configuration": driver._sampler_configuration,
    }
    ckptr = ocp.PyTreeCheckpointer()
    restored = ckptr.restore(run_dir / "latest", item=target)

    saved_config = restored["sampler_configuration"]
    if saved_config.shape[0] != driver.n_chains:
        raise ValueError(
            f"Checkpoint n_chains={saved_config.shape[0]} != "
            f"driver n_chains={driver.n_chains}."
        )
    for row, rd in driver._tensors.items():
        for col in rd:
            driver._tensors[row][col] = restored["tensors"][str(row)][str(col)]
    driver._sampler_key = restored["sampler_key"]
    driver._sampler_configuration = saved_config

    with open(run_dir / "latest.json") as f:
        metadata = json.load(f)
    driver.step_count = metadata["step"]
    driver.t = metadata["time"]
    return metadata


def load_model_from_checkpoint(run_dir, model):
    """Load model tensor values from a runner checkpoint.

    Restores only the tensors (not sampler state) into ``model``.
    Returns the updated model and the checkpoint metadata dict.
    """
    from flax import nnx

    run_dir = Path(run_dir)
    graphdef, params, model_state = nnx.split(model, nnx.Param, ...)
    tensors = nnx.to_pure_dict(params)["tensors"]
    target = {
        "tensors": {
            str(row): {str(col): t for col, t in rd.items()}
            for row, rd in tensors.items()
        },
    }
    ckptr = ocp.PyTreeCheckpointer()
    restored = ckptr.restore(run_dir / "latest", item=target, partial_restore=True)
    loaded = {
        row: {
            col: restored["tensors"][str(row)][str(col)]
            for col in rd
        }
        for row, rd in tensors.items()
    }
    with open(run_dir / "latest.json") as f:
        metadata = json.load(f)
    return nnx.merge(graphdef, {"tensors": loaded}, model_state), metadata


def run(
    driver,
    *,
    n_steps: int | None = None,
    T_final: float | None = None,
    run_dir: str | Path,
    observable_names: tuple[str, ...] = (),
    log_every: int = 10,
    save_every: int = 50,
    resume: bool = False,
    extra_config: dict | None = None,
) -> None:
    """Run a TDVP trajectory with periodic printing and checkpointing."""
    run_dir = Path(run_dir)
    series: dict[str, list] = {}

    if resume and (run_dir / "latest").exists():
        metadata = load_checkpoint(run_dir, driver)
        series = metadata.get("series", {})

    start_step = driver.step_count

    if T_final is not None:
        remaining = T_final - driver.t
        if remaining <= 1e-12 * max(1.0, abs(T_final)):
            print(
                f"Already at t={driver.t:.6f} >= T_final={T_final:.6f}.",
                flush=True,
            )
            return
        total_new_steps = int(round(remaining / driver.dt))
    elif n_steps is not None:
        total_new_steps = n_steps
    else:
        raise ValueError("Provide exactly one of n_steps or T_final.")

    target_step = start_step + total_new_steps

    _print_config_table(
        driver,
        n_steps=total_new_steps,
        T_final=T_final,
        run_dir=str(run_dir),
        observable_names=observable_names,
        log_every=log_every,
        save_every=save_every,
        resume=resume and start_step > 0,
        extra_config=extra_config,
        start_step=start_step,
        target_step=target_step,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    _print_header(observable_names, driver)

    for _ in range(total_new_steps):
        driver.run(driver.dt)
        step = driver.step_count
        _append_series(series, driver, observable_names)
        if step % log_every == 0 or step == target_step:
            _print_step(driver, observable_names)
        if step % save_every == 0 or step == target_step:
            save_checkpoint(
                run_dir, driver, step,
                series=series,
                config=extra_config or {},
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: dict) -> None:
    text = json.dumps(data, indent=2, default=_json_default)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _extract_config(driver, extra):
    """Auto-extract a structured config dict from the driver."""
    model = driver.model
    cfg = {
        "model": type(model).__name__,
        "shape": list(getattr(model, "shape", [])),
        "dt": driver.dt,
        "n_samples": driver.n_samples,
        "n_chains": driver.n_chains,
        "full_gradient": driver.full_gradient,
    }
    prec = driver.preconditioner
    space = getattr(prec, "space", None)
    cfg["solver_space"] = "minsr" if space and "Sample" in type(space).__name__ else "sr"
    cfg["diag_shift"] = prec.diag_shift
    if extra:
        cfg["extra"] = extra
    return cfg


def _print_config_table(
    driver, *, n_steps, T_final, run_dir, observable_names,
    log_every, save_every, resume, extra_config, start_step, target_step,
):
    model = driver.model
    lines = []

    dev = jax.devices()[0]
    lines.append(("Device", f"{dev.platform} ({dev.device_kind})"))

    model_name = type(model).__name__
    shape = getattr(model, "shape", None)
    if shape:
        lines.append(("Model", f"{model_name} ({shape[0]}x{shape[1]})"))
    else:
        lines.append(("Model", model_name))

    cfg = getattr(model, "config", None)
    if cfg and hasattr(cfg, "degeneracy_per_charge"):
        dpc = cfg.degeneracy_per_charge
        lines.append(("Bond dim", f"{max(dpc) * len(dpc)} (D_k={max(dpc)}, N={cfg.N})"))
    elif hasattr(model, "Dmax"):
        lines.append(("Bond dim", str(model.Dmax)))
    elif hasattr(model, "bond_dim"):
        lines.append(("Bond dim", str(model.bond_dim)))

    n_params = sum(
        t.size for rd in driver._tensors.values() for t in rd.values()
    )
    lines.append(("Parameters", f"{n_params:,}"))

    strategy = getattr(model, "strategy", None)
    if strategy is not None:
        s_name = type(strategy).__name__
        bdim = getattr(strategy, "truncate_bond_dimension", None)
        if bdim is not None:
            s_name += f"(D'={bdim})"
        n_sw = getattr(strategy, "n_sweeps", None)
        if n_sw is not None:
            s_name += f"(sweeps={n_sw})"
        lines.append(("Strategy", s_name))

    lines.append(("dtype", str(getattr(model, "dtype", "?"))))

    op = driver.operator
    op_name = type(op).__name__
    if isinstance(op, TimeDependentHamiltonian):
        op_name += f"({type(op.schedule).__name__})"
    lines.append(("Hamiltonian", op_name))

    tu = type(driver.time_unit).__name__
    time_label = "imaginary" if "Imaginary" in tu else "real"
    lines.append((
        "Integrator",
        f"{type(driver.integrator).__name__} ({time_label} time)",
    ))
    lines.append(("dt", str(driver.dt)))
    lines.append(("t0", f"{driver.t:.6f}"))

    if T_final is not None:
        lines.append(("Target", f"T = {T_final:.6f} ({n_steps} steps)"))
    else:
        lines.append(("Target", f"{n_steps} steps -> step {target_step}"))

    lines.append(("Samples", f"{driver.n_samples} ({driver.n_chains} chains)"))
    lines.append(("Full gradient", str(driver.full_gradient)))

    prec = driver.preconditioner
    strat_name = type(prec.strategy).__name__
    solver_fn = getattr(prec.strategy, "solver", None)
    if solver_fn is not None:
        strat_name += f"({solver_fn.__name__})"
    space = getattr(prec, "space", None)
    sr_label = "minSR" if space and "Sample" in type(space).__name__ else "SR"
    lines.append(("Solver", f"{strat_name} . {sr_label}"))

    ordering = getattr(prec, "ordering", None)
    if ordering is not None:
        lines.append(("Ordering", type(ordering).__name__))

    lines.append(("Diag shift", f"{prec.diag_shift:.1e}"))

    gc = getattr(prec, "gauge_config", None)
    lines.append(("Gauge removal", "enabled" if gc is not None else "None"))

    if observable_names:
        lines.append(("Observables", ", ".join(observable_names)))

    w = max(len(label) for label, _ in lines) + 2
    bar = "-" * 50

    print(f"-- Run Configuration {bar}", flush=True)
    for label, value in lines:
        print(f"{label:<{w}}{value}", flush=True)

    print(f"-- Resume {bar}", flush=True)
    if resume:
        print(f"{'Checkpoint':<{w}}step {start_step}, t = {driver.t:.6f}", flush=True)
        if T_final is not None:
            print(
                f"{'Remaining':<{w}}{n_steps} steps "
                f"(t: {driver.t:.6f} -> {T_final:.6f})",
                flush=True,
            )
        else:
            print(
                f"{'Remaining':<{w}}{n_steps} steps -> step {target_step}",
                flush=True,
            )
    else:
        print(f"{'Starting fresh':<{w}}t = {driver.t:.6f}", flush=True)

    print(f"-- Output {bar}", flush=True)
    print(f"{'Run dir':<{w}}{run_dir}", flush=True)
    print(f"{'Log every':<{w}}{log_every}", flush=True)
    print(f"{'Save every':<{w}}{save_every}", flush=True)

    if extra_config:
        print(f"-- Problem {bar}", flush=True)
        for key, value in extra_config.items():
            print(f"{key:<{w}}{value}", flush=True)

    print(bar, flush=True)


def _metrics_config(driver):
    return getattr(
        getattr(driver, "preconditioner", None), "metrics_config", None
    )


def _print_header(observable_names, driver):
    mc = _metrics_config(driver)
    cols = ["step", "time", "energy", "error", "variance"]
    cols.extend(observable_names)
    if getattr(mc, "record_FS_norm", False):
        cols.append("FS_norm")
    if getattr(mc, "record_TDVP_residual", False):
        cols.append("TDVP_res")
    if getattr(mc, "record_SR_solve_residual", False):
        cols.append("SR_res")
    if getattr(mc, "record_step_wall_time", False):
        cols.append("wall")
    print("  ".join(f"{c:>14}" for c in cols), flush=True)


def _print_step(driver, observable_names):
    energy = driver.energy
    metrics = driver.metrics
    parts = [
        f"{driver.step_count:6d}",
        f"{float(driver.t):12.6f}",
        f"{float(energy.mean.real):18.10f}",
        f"+/- {float(energy.error_of_mean.real):10.2e}",
        f"{float(energy.variance.real):12.4e}",
    ]
    for i in range(len(observable_names)):
        stat = driver.observable_stats[i]
        parts.append(f"{float(stat.mean.real):14.6f}")
    mc = _metrics_config(driver)
    if getattr(mc, "record_FS_norm", False):
        parts.append(f"{float(metrics.get('FS_norm_squared', float('nan'))):12.4e}")
    if getattr(mc, "record_TDVP_residual", False):
        parts.append(f"{float(metrics.get('TDVP_residual', float('nan'))):12.4e}")
    if getattr(mc, "record_SR_solve_residual", False):
        parts.append(f"{float(metrics.get('SR_solve_residual', float('nan'))):12.4e}")
    if getattr(mc, "record_step_wall_time", False):
        parts.append(f"{float(metrics.get('step_wall_time', float('nan'))):6.1f}s")
    print("  ".join(parts), flush=True)


def _append_series(series, driver, observable_names):
    energy = driver.energy
    metrics = driver.metrics
    series.setdefault("step", []).append(int(driver.step_count))
    series.setdefault("time", []).append(float(driver.t))
    series.setdefault("energy_mean", []).append(float(energy.mean.real))
    series.setdefault("energy_error", []).append(float(energy.error_of_mean.real))
    series.setdefault("energy_variance", []).append(float(energy.variance.real))
    for i, name in enumerate(observable_names):
        stat = driver.observable_stats[i]
        series.setdefault(f"{name}_mean", []).append(float(stat.mean.real))
        series.setdefault(f"{name}_error", []).append(float(stat.error_of_mean.real))
    for key in ("FS_norm_squared", "TDVP_residual", "SR_solve_residual"):
        if key in metrics:
            series.setdefault(key, []).append(float(metrics[key]))
    if "step_wall_time" in metrics:
        series.setdefault("wall_time", []).append(float(metrics["step_wall_time"]))
