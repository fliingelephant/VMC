"""Simulation workflow infrastructure: run loop, checkpointing, logging.

Provides the outer loop that wraps a TDVPDriver with periodic logging,
checkpointing, and resume support.

- Checkpointing via Orbax CheckpointManager (atomic, step-numbered, max_to_keep).
- Per-step metrics via pluggable AbstractLog (ConsoleLog, JsonLog, CompositeLog).
- Static config stored as Orbax metadata, validated on resume.

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
import platform
import socket
import time
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp

from vmc.operators.time_dependent import TimeDependentHamiltonian
from vmc.preconditioners import MetricsConfig, solve_cg, solve_cholesky, solve_svd

logger = logging.getLogger(__name__)

__all__ = [
    "AbstractLog", "ConsoleLog", "JsonLog", "CompositeLog",
    "add_common_args", "resolve_solver", "run",
    "load_model_from_checkpoint", "read_config", "DEFAULT_METRICS_CONFIG",
]

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
    """Base class for simulation loggers."""

    @abc.abstractmethod
    def __call__(self, step: int, item: dict) -> None: ...

    @abc.abstractmethod
    def flush(self) -> None: ...


class ConsoleLog(AbstractLog):
    """Formatted per-step output to the terminal."""

    def __init__(self):
        self._header_printed = False

    def __call__(self, step: int, item: dict) -> None:
        if not self._header_printed:
            logger.info("  ".join(f"{k:>14}" for k in item))
            self._header_printed = True
        parts = []
        for v in item.values():
            if isinstance(v, float):
                parts.append(f"{v:14.4e}" if abs(v) < 1e-2 or abs(v) > 1e4 else f"{v:14.6f}")
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
# Checkpointing (Orbax CheckpointManager)
# ---------------------------------------------------------------------------

_ITEM_NAMES = ("tensors", "sampler")


def read_config(run_dir: str | Path) -> dict:
    """Read run config from checkpoint metadata without loading tensors."""
    mgr = ocp.CheckpointManager(
        Path(run_dir), item_names=_ITEM_NAMES,
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    meta = mgr.metadata()
    return dict(meta.custom_metadata) if hasattr(meta, "custom_metadata") else {}


def _str_keys(tensors: dict) -> dict:
    """Convert int-keyed tensor dict to string keys for orbax."""
    return {str(r): {str(c): t for c, t in rd.items()} for r, rd in tensors.items()}


def load_model_from_checkpoint(run_dir, model):
    """Load model tensor values from a checkpoint (tensors only, no sampler state)."""
    from flax import nnx

    run_dir = Path(run_dir)
    mgr = ocp.CheckpointManager(
        run_dir, item_names=_ITEM_NAMES,
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    graphdef, params, model_state = nnx.split(model, nnx.Param, ...)
    target = _str_keys(nnx.to_pure_dict(params)["tensors"])
    restored = mgr.restore(mgr.latest_step(), args=ocp.args.Composite(
        tensors=ocp.args.StandardRestore(target),
    ))
    loaded = {
        row: {col: restored["tensors"][str(row)][str(col)] for col in rd}
        for row, rd in nnx.to_pure_dict(params)["tensors"].items()
    }
    meta = mgr.metadata()
    meta_dict = dict(meta.custom_metadata) if hasattr(meta, "custom_metadata") else {}
    return nnx.merge(graphdef, {"tensors": loaded}, model_state), meta_dict


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

    Checkpoints are managed by Orbax CheckpointManager (atomic, step-numbered).
    Per-step metrics are written by the ``out`` logger (default: ConsoleLog).
    """
    if n_steps is not None and T_final is not None:
        raise TypeError("Specify n_steps or T_final, not both.")
    if n_steps is None and T_final is None:
        raise TypeError("Specify n_steps or T_final.")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if out is None:
        out = CompositeLog(ConsoleLog(), JsonLog(run_dir / "metrics.jsonl"))

    runtime = _collect_runtime()

    # Build metadata for this run
    run_metadata = _extract_config(driver, extra_config)
    run_metadata["runtime"] = runtime

    mgr = ocp.CheckpointManager(
        run_dir,
        item_names=_ITEM_NAMES,
        metadata=run_metadata,
        options=ocp.CheckpointManagerOptions(max_to_keep=2, save_interval_steps=1),
    )

    if resume and mgr.latest_step() is not None:
        latest = mgr.latest_step()
        restored = mgr.restore(latest, args=ocp.args.Composite(
            tensors=ocp.args.StandardRestore(_str_keys(driver._tensors)),
            sampler=ocp.args.StandardRestore({
                "key": driver._sampler_key,
                "configuration": driver._sampler_configuration.reshape(driver.n_chains, -1),
                "t": jax.numpy.asarray(driver.t),
            }),
        ))
        saved_config = restored["sampler"]["configuration"]
        if saved_config.shape[0] != driver.n_chains:
            raise ValueError(
                f"Checkpoint n_chains={saved_config.shape[0]} != "
                f"driver n_chains={driver.n_chains}."
            )
        for row, rd in driver._tensors.items():
            for col in rd:
                driver._tensors[row][col] = restored["tensors"][str(row)][str(col)]
        driver._sampler_key = restored["sampler"]["key"]
        driver._sampler_configuration = saved_config
        driver.step_count = latest
        driver.t = float(restored["sampler"]["t"])
        saved_meta = mgr.metadata()
        saved_extra = getattr(saved_meta, "custom_metadata", {}).get("extra", {})
        if extra_config and saved_extra and extra_config != saved_extra:
            logger.warning(
                "Resume config differs from checkpoint. "
                "Saved: %s, Current: %s", saved_extra, extra_config,
            )

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
        runtime=runtime,
    )

    for _ in range(total_new_steps):
        driver.run(driver.dt)
        step = driver.step_count
        item = _build_step_item(driver, observable_names)
        if step % log_every == 0 or step == target_step:
            out(step, item)
        if step % save_every == 0 or step == target_step:
            mgr.save(step, args=ocp.args.Composite(
                tensors=ocp.args.StandardSave(_str_keys(driver._tensors)),
                sampler=ocp.args.StandardSave({
                    "key": driver._sampler_key,
                    "configuration": driver._sampler_configuration.reshape(driver.n_chains, -1),
                    "t": jax.numpy.asarray(driver.t),
                }),
            ))
            mgr.wait_until_finished()
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
        "step": int(driver.step_count),
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


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
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
    log_every, save_every, resume, extra_config, start_step, target_step, runtime,
):
    model = driver.model
    lines = []

    lines.append(("Device", runtime.get("platform", "?")))

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
