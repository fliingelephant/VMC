"""Simulation workflow infrastructure: run loop, checkpointing, logging.

Provides the outer loop that wraps a TDVPDriver with periodic logging,
checkpointing, and resume support. Inspired by NetKet's driver.run() and
PyTorch Lightning's Trainer, but tailored for PEPS-tVMC.

Usage::

    from vmc.workflow import run, add_common_args, ConsoleLog, JsonLog, CompositeLog

    run(driver, n_steps=200, run_dir="data/my_run",
        out=CompositeLog(ConsoleLog(), JsonLog("data/my_run/metrics.jsonl")))
"""
from __future__ import annotations

import abc
import argparse
import json
import logging
import math
import os
import platform
import socket
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from vmc.operators.time_dependent import TimeDependentHamiltonian
from vmc.preconditioners import MetricsConfig, solve_cg, solve_cholesky, solve_svd

logger = logging.getLogger(__name__)

SOLVERS = {"cholesky": solve_cholesky, "svd": solve_svd, "cg": solve_cg}

DEFAULT_METRICS_CONFIG = MetricsConfig(
    record_FS_norm=True,
    record_TDVP_residual=True,
    record_SR_solve_residual=True,
    record_step_wall_time=True,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class AbstractLog(abc.ABC):
    """Base class for simulation loggers.

    Loggers receive per-step metrics via ``__call__`` and flush buffered
    data to disk via ``flush``.
    """

    @abc.abstractmethod
    def __call__(self, step: int, item: dict) -> None:
        """Log one step of metrics."""

    @abc.abstractmethod
    def flush(self) -> None:
        """Flush any buffered data to disk."""


class ConsoleLog(AbstractLog):
    """Formatted per-step output to the terminal."""

    def __init__(self):
        self._header_printed = False

    def __call__(self, step: int, item: dict) -> None:
        if not self._header_printed:
            header = "  ".join(f"{k:>14}" for k in item)
            logger.info(header)
            self._header_printed = True
        parts = []
        for k, v in item.items():
            if isinstance(v, float):
                if abs(v) < 1e-2 or abs(v) > 1e4:
                    parts.append(f"{v:14.4e}")
                else:
                    parts.append(f"{v:14.6f}")
            elif isinstance(v, int):
                parts.append(f"{v:14d}")
            else:
                parts.append(f"{str(v):>14}")
        logger.info("  ".join(parts))

    def flush(self) -> None:
        pass


class JsonLog(AbstractLog):
    """Append JSON lines to a file, one object per step."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self._path, "a")

    def __call__(self, step: int, item: dict) -> None:
        self._f.write(json.dumps({"step": step, **item}, default=_json_default) + "\n")

    def flush(self) -> None:
        self._f.flush()


class CompositeLog(AbstractLog):
    """Combine multiple loggers."""

    def __init__(self, *loggers: AbstractLog):
        self._loggers = loggers

    def __call__(self, step: int, item: dict) -> None:
        for log in self._loggers:
            log(step, item)

    def flush(self) -> None:
        for log in self._loggers:
            log.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for checkpoints and series")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(run_dir, driver, step, *, series=None, runtime=None, **metadata):
    """Save driver state to run_dir/latest/ (orbax) + latest.json."""
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
    }
    if runtime is not None:
        json_data["runtime"] = runtime
    if series is not None:
        json_data["series"] = series
    json_data.update(metadata)
    _atomic_write_json(run_dir / "latest.json", json_data)


def load_checkpoint(run_dir, driver):
    """Restore driver state from run_dir/latest/ (orbax) + latest.json."""
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
    """Load model tensor values from a checkpoint (tensors only, no sampler state)."""
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


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

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
    out: AbstractLog | None = None,
) -> None:
    """Run a TDVP trajectory with periodic logging and checkpointing.

    Specify exactly one of ``n_steps`` or ``T_final``:

    - ``n_steps``: run this many steps (ground-state optimization).
    - ``T_final``: run until this absolute time (dynamics). On resume,
      computes remaining time from the checkpoint.

    Args:
        driver: TDVPDriver instance.
        n_steps: Number of steps to run.
        T_final: Target final time (absolute).
        run_dir: Output directory for checkpoints.
        observable_names: Names for driver observables (for logging keys).
        log_every: Log metrics every N steps.
        save_every: Save checkpoint every N steps.
        resume: If True, resume from existing checkpoint in run_dir.
        extra_config: Additional config to save in checkpoint JSON.
        out: Logger instance. Default: ConsoleLog().
    """
    if n_steps is not None and T_final is not None:
        raise TypeError("Specify n_steps or T_final, not both.")
    if n_steps is None and T_final is None:
        raise TypeError("Specify n_steps or T_final.")

    run_dir = Path(run_dir)
    series: dict[str, list] = {}

    if out is None:
        out = ConsoleLog()

    runtime = _collect_runtime()

    if resume and (run_dir / "latest").exists():
        metadata = load_checkpoint(run_dir, driver)
        series = metadata.get("series", {})
        runtime = metadata.get("runtime", runtime)

    start_step = driver.step_count

    if T_final is not None:
        remaining = T_final - driver.t
        n_exact = remaining / driver.dt
        n_rounded = round(n_exact)
        if abs(n_exact - n_rounded) > 0.01:
            raise ValueError(
                f"T_final={T_final} from t={driver.t:.6f} is not an integer "
                f"multiple of dt={driver.dt} (remaining/dt = {n_exact:.6f})."
            )
        if n_rounded <= 0:
            logger.info("Already at t=%.6f >= T_final=%.6f.", driver.t, T_final)
            return
        total_new_steps = n_rounded
    else:
        total_new_steps = n_steps

    target_step = start_step + total_new_steps

    _log_config_table(
        driver,
        n_steps=total_new_steps,
        t_end=T_final,
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

    for _ in range(total_new_steps):
        driver.run(driver.dt)
        step = driver.step_count
        item = _build_step_item(driver, observable_names)
        series.setdefault("step", []).append(int(step))
        _accumulate_series(series, item)
        if step % log_every == 0 or step == target_step:
            out(step, item)
        if step % save_every == 0 or step == target_step:
            runtime["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            save_checkpoint(
                run_dir, driver, step,
                series=series,
                runtime=runtime,
                config=extra_config or {},
            )
            out.flush()

    logger.info("Run complete: %d steps, t=%.6f", target_step, driver.t)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_runtime() -> dict:
    """Collect runtime environment metadata."""
    dev = jax.devices()[0]
    return {
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": f"{dev.platform} ({dev.device_kind})",
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
    }


def _build_step_item(driver, observable_names: tuple[str, ...]) -> dict:
    """Build a flat metrics dict for one step."""
    energy = driver.energy
    metrics = driver.metrics
    item = {
        "time": float(driver.t),
        "energy_mean": float(energy.mean.real),
        "energy_error": float(energy.error_of_mean.real),
        "energy_variance": float(energy.variance.real),
    }
    for i, name in enumerate(observable_names):
        stat = driver.observable_stats[i]
        item[f"{name}_mean"] = float(stat.mean.real)
        item[f"{name}_error"] = float(stat.error_of_mean.real)
    for key in ("FS_norm_squared", "TDVP_residual", "SR_solve_residual", "step_wall_time"):
        if key in metrics:
            item[key] = float(metrics[key])
    return item


def _accumulate_series(series: dict[str, list], item: dict) -> None:
    """Append one step's metrics to the columnar series dict."""
    for key, value in item.items():
        series.setdefault(key, []).append(value)


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


def _log_config_table(
    driver, *, n_steps, t_end, run_dir, observable_names,
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
    lines.append(("Integrator", f"{type(driver.integrator).__name__} ({time_label} time)"))
    lines.append(("dt", str(driver.dt)))
    lines.append(("t0", f"{driver.t:.6f}"))

    if t_end is not None:
        lines.append(("Target", f"t_end = {t_end:.6f} ({n_steps} steps)"))
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

    logger.info("-- Run Configuration %s", bar)
    for label, value in lines:
        logger.info(f"{label:<{w}}{value}")

    logger.info("-- Resume %s", bar)
    if resume:
        logger.info(f"{'Checkpoint':<{w}}step {start_step}, t = {driver.t:.6f}")
        if t_end is not None:
            logger.info(f"{'Remaining':<{w}}{n_steps} steps (t: {driver.t:.6f} -> {t_end:.6f})")
        else:
            logger.info(f"{'Remaining':<{w}}{n_steps} steps -> step {target_step}")
    else:
        logger.info(f"{'Starting fresh':<{w}}t = {driver.t:.6f}")

    logger.info("-- Output %s", bar)
    logger.info(f"{'Run dir':<{w}}{run_dir}")
    logger.info(f"{'Log every':<{w}}{log_every}")
    logger.info(f"{'Save every':<{w}}{save_every}")

    if extra_config:
        logger.info("-- Problem %s", bar)
        for key, value in extra_config.items():
            logger.info(f"{key:<{w}}{value}")

    logger.info(bar)
