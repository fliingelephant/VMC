# Runner Module & Examples Refactor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `examples/runner.py` — shared run infrastructure — and refactor all example scripts to use it, consolidating duplicated run loops, checkpointing, printing, and CLI patterns across the codebase.

**Architecture:** One new module (`examples/runner.py`) providing `add_common_args`, `resolve_solver`, `run`, `save_checkpoint`, `load_checkpoint`, and `DEFAULT_METRICS_CONFIG`. Scripts keep their physics (model/operator construction) but delegate all run infrastructure to the runner. Subcommand-based scripts (z2_vison_propagation, z2_vison_propagation_L10, z2_vison_higgs_confinement) are consolidated into per-directory ground_state.py + dynamics.py pairs. The runner handles checkpointing, resume, config table, per-step printing, and series accumulation.

**Tech Stack:** JAX, Flax NNX, argparse, numpy, json. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-20-runner-module-design.md`

---

## Decision Points (resolved)

1. **Import mechanism:** Scripts use `sys.path.insert(0, ...)` to add the `examples/` directory, then `from runner import ...`. Follow the existing `__package__` guard pattern in z2_hardcore_boson scripts.
2. **MetricsConfig default:** All 4 flags enabled by default. Runner exports `DEFAULT_METRICS_CONFIG`.
3. **Solver mapping:** Runner exports `resolve_solver(name)` mapping `"cholesky"/"svd"/"cg"` to solver functions.
4. **Parameter scans:** Each point gets its own `run_dir`.
5. **Incompatible patterns:** Adam and adaptive-dt loops stay custom; heisenberg/ising keep all 3 methods.

---

## File Structure

```
CREATE:
  examples/runner.py                           — shared run infrastructure
  examples/lgt/z2_vison/ground_state.py        — consolidates propagation + L10 ground-state
  examples/lgt/z2_vison/dynamics.py            — consolidates propagation + L10 real-time
  examples/lgt/z2_vison/plot.py                — Fig 5a plotting (from z2_vison_propagation)
  examples/lgt/z2_vison_higgs/ground_state.py  — from z2_vison_higgs_confinement ground-state
  examples/lgt/z2_vison_higgs/dynamics.py      — from z2_vison_higgs_confinement real-time
  tests/test_runner.py                         — runner integration tests

REFACTOR IN-PLACE:
  examples/lgt/z2_pure_gauge.py                — use runner.run()
  examples/lgt/odd_z2_gauge.py                 — drop SimulationData, use runner.run()
  examples/lgt/z3_pure_gauge.py                — drop SimulationData, use runner.run()
  examples/lgt/z2_hardcore_boson/common.py     — remove infrastructure, keep physics
  examples/lgt/z2_hardcore_boson/benchmark_3x3.py   — use runner
  examples/lgt/z2_hardcore_boson/bond_dim_scan.py    — use runner
  examples/lgt/z2_hardcore_boson/energy_vs_J.py      — use runner
  examples/lgt/z2_hardcore_boson/finite_size_scaling.py — use runner
  examples/Schmitt_2022_TFIM2d/schmitt_tfim_quench.py  — use runner for single quench
  examples/ground_states/heisenberg.py         — use DEFAULT_METRICS_CONFIG, clean up
  examples/ground_states/ising.py              — same as heisenberg
  examples/time_dependent/exact_tdvp_3x3_check.py — use resolve_solver, clean up

DELETE:
  examples/lgt/observables.py                  — replaced by runner
  examples/lgt/z2_vison_propagation/__main__.py — consolidated into z2_vison/
  examples/lgt/z2_vison_propagation_L10/__main__.py — consolidated into z2_vison/
  examples/lgt/z2_vison_higgs_confinement/__main__.py — consolidated into z2_vison_higgs/

MOVE:
  examples/lgt/z2_vison_propagation/exact.py → examples/lgt/z2_vison/exact.py

LEAVE ALONE:
  examples/gauge_removal/main.py
```

---

### Task 1: Create `examples/runner.py`

**Files:**
- Create: `examples/runner.py`

This is the foundation. Everything else depends on it.

- [ ] **Step 1: Create runner.py**

```python
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


def save_checkpoint(run_dir, driver, step, *, series=None, **metadata):
    """Save driver state to run_dir/latest/. Writes atomically via orbax."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build state dict with string keys (orbax requires string dict keys)
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
    """Restore driver state from run_dir/latest/. Returns metadata."""
    run_dir = Path(run_dir)

    # Build target tree matching the save structure
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
    # Convert string keys back to int keys
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

    w = max(len(l) for l, _ in lines) + 2
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
```

- [ ] **Step 2: Commit**

```bash
git add examples/runner.py
git commit -m "Add examples/runner.py — shared run infrastructure"
```

---

### Task 2: Test runner

**Files:**
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write integration tests**

```python
"""Tests for examples/runner.py."""
from __future__ import annotations

from vmc import config  # noqa: F401

import json
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

# Make runner importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from runner import (  # noqa: E402
    DEFAULT_METRICS_CONFIG,
    add_common_args,
    load_checkpoint,
    resolve_solver,
    run,
    save_checkpoint,
)

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver  # noqa: E402
from vmc.operators import DiagonalOperator, LocalHamiltonian, OneSiteOperator  # noqa: E402
from vmc.peps import PEPS  # noqa: E402
from vmc.peps.common.strategy import Variational  # noqa: E402
from vmc.preconditioners import DirectSolve, SRPreconditioner, solve_cholesky  # noqa: E402


def _make_tiny_driver():
    shape = (2, 2)
    model = PEPS(
        rngs=nnx.Rngs(0), shape=shape, bond_dim=2,
        contraction_strategy=Variational(4), dtype=jnp.complex128,
    )
    sigma_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    zz_diag = jnp.array([1, -1, -1, 1], dtype=jnp.complex128)
    hamiltonian = LocalHamiltonian(
        shape=shape,
        terms=(
            OneSiteOperator(0, 0, sigma_x),
            DiagonalOperator(((0, 0), (0, 1)), zz_diag),
        ),
    )
    return TDVPDriver(
        model, hamiltonian,
        preconditioner=SRPreconditioner(
            diag_shift=1e-2,
            strategy=DirectSolve(solver=solve_cholesky),
            metrics_config=DEFAULT_METRICS_CONFIG,
        ),
        dt=0.01,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(42),
        n_samples=64, n_chains=8,
    )


def test_resolve_solver():
    assert resolve_solver("cholesky") is solve_cholesky


def test_add_common_args():
    import argparse
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(["--n-samples", "2048", "--solver", "svd", "--resume"])
    assert args.n_samples == 2048
    assert args.solver == "svd"
    assert args.resume is True
    assert args.dt == 0.01  # default


def test_checkpoint_round_trip():
    driver = _make_tiny_driver()
    driver.run(driver.dt)
    driver.run(driver.dt)
    step_before = driver.step_count
    time_before = driver.t
    tensors_before = jax.tree.map(lambda x: x.copy(), driver._tensors)

    with tempfile.TemporaryDirectory() as tmpdir:
        series = {"step": [1, 2], "energy_mean": [-0.5, -0.6]}
        save_checkpoint(tmpdir, driver, step_before, series=series, extra="test")
        assert (Path(tmpdir) / "latest").exists()  # orbax directory
        assert (Path(tmpdir) / "latest.json").exists()

        driver2 = _make_tiny_driver()
        metadata = load_checkpoint(tmpdir, driver2)
        assert driver2.step_count == step_before
        assert driver2.t == time_before
        assert metadata["series"]["energy_mean"] == [-0.5, -0.6]
        assert metadata["extra"] == "test"
        for row in driver2._tensors:
            for col in driver2._tensors[row]:
                jnp.testing.assert_array_equal(
                    driver2._tensors[row][col], tensors_before[row][col],
                )


def test_run_fresh():
    driver = _make_tiny_driver()
    with tempfile.TemporaryDirectory() as tmpdir:
        run(driver, n_steps=5, run_dir=tmpdir, log_every=1, save_every=5)
        assert (Path(tmpdir) / "latest").exists()
        with open(Path(tmpdir) / "latest.json") as f:
            data = json.load(f)
        assert data["step"] == 5
        assert len(data["series"]["step"]) == 5
        assert len(data["series"]["energy_mean"]) == 5


def test_run_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        driver = _make_tiny_driver()
        run(driver, n_steps=3, run_dir=tmpdir, log_every=1, save_every=3)
        with open(Path(tmpdir) / "latest.json") as f:
            data = json.load(f)
        assert data["step"] == 3

        driver2 = _make_tiny_driver()
        run(driver2, n_steps=2, run_dir=tmpdir, log_every=1, save_every=5, resume=True)
        with open(Path(tmpdir) / "latest.json") as f:
            data = json.load(f)
        assert data["step"] == 5
        assert len(data["series"]["step"]) == 5
        assert data["series"]["step"] == [1, 2, 3, 4, 5]
```

- [ ] **Step 2: Run tests**

Run: `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_runner.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_runner.py
git commit -m "Add runner integration tests"
```

---

### Task 3: Refactor `lgt/z2_pure_gauge.py`

**Files:**
- Modify: `examples/lgt/z2_pure_gauge.py`

This is the simplest runner-compatible script. It validates the runner with real physics.

**Changes:**
1. Add `sys.path` import for runner
2. Remove `append_series`, `save_run`, `SR_METRICS_CONFIG` — replaced by runner
3. Replace `run_sr()` with: build driver → `runner.run()`
4. Keep all physics functions unchanged: `build_z2_hamiltonian`, `build_mean_plaquette_observable`, `build_mean_link_z_observable`, `build_problem`, `build_model`
5. Use `DEFAULT_METRICS_CONFIG` from runner

The refactored `main()` should be roughly:

```python
def main() -> None:
    hamiltonian, observables = build_problem()
    driver = TDVPDriver(
        build_model(SEED),
        hamiltonian,
        observables=observables,
        preconditioner=SRPreconditioner(
            space=ParameterSpace(),
            strategy=DirectSolve(solver=solve_cholesky),
            diag_shift=SR_DIAG_SHIFT,
            metrics_config=DEFAULT_METRICS_CONFIG,
        ),
        dt=SR_FIXED_DT,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(SEED),
        n_samples=N_SAMPLES,
        n_chains=N_CHAINS,
    )
    run(
        driver,
        n_steps=SR_FIXED_STEPS,
        run_dir=benchmark_output_dir(),
        observable_names=("plaquette", "z_h", "z_v"),
        log_every=1,
        save_every=SR_FIXED_STEPS,
        extra_config={
            "gauge_group": "Z2", "L": L,
            "h": H_COUPLING, "g": G_COUPLING,
        },
    )
```

- [ ] **Step 1: Refactor the script**
- [ ] **Step 2: Test by running**: `JAX_PLATFORM_NAME=cpu uv run python examples/lgt/z2_pure_gauge.py` (verify config table prints, per-step output shows, no crash)
- [ ] **Step 3: Commit**

---

### Task 4: Refactor `z2_hardcore_boson/` scripts

**Files:**
- Modify: `examples/lgt/z2_hardcore_boson/common.py` — remove infrastructure
- Modify: `examples/lgt/z2_hardcore_boson/benchmark_3x3.py`
- Modify: `examples/lgt/z2_hardcore_boson/bond_dim_scan.py`
- Modify: `examples/lgt/z2_hardcore_boson/energy_vs_J.py`
- Modify: `examples/lgt/z2_hardcore_boson/finite_size_scaling.py`

**Changes to common.py — REMOVE:**
- `save_latest`, `restore_latest`, `maybe_resume`, `prepare_run_dir`, `latest_paths`
- `run_ground_state_steps`, `ground_state_metrics`
- `SR_METRICS_CONFIG`
- `DEFAULT_LOG_EVERY`, `DEFAULT_SAVE_EVERY`

**Changes to common.py — KEEP:**
- `build_z2_hardcore_boson_hamiltonian`, `build_model`, `build_central_bulk_observable`
- `half_filling`, `format_token`, `coupling_suffix`
- Constants: `CHARGE_OF_SITE`, `QX`, `DEFAULT_H`, `DEFAULT_G`, `DEFAULT_J`, `DEFAULT_M`
- `DEFAULT_BOUNDARY_SWEEPS`, `DEFAULT_DT`, `DEFAULT_DIAG_SHIFT`

**Pattern for each script (benchmark_3x3 as example):**

```python
"""3x3 quoted-ED benchmark for Z2 gauge fields coupled to hard-core bosons."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vmc import config  # noqa: F401

import argparse

import jax
from flax import nnx

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver
from vmc.preconditioners import DirectSolve, SRPreconditioner, solve_cholesky
from vmc.qgt import ParameterSpace

from runner import DEFAULT_METRICS_CONFIG, run

from common import (
    DEFAULT_BOUNDARY_SWEEPS, DEFAULT_DIAG_SHIFT, DEFAULT_DT,
    build_model, build_z2_hardcore_boson_hamiltonian,
)

# ... constants stay the same ...

def main() -> None:
    args = _parse_args()
    shape = SHAPE
    model = build_model(shape, ...)
    hamiltonian = build_z2_hardcore_boson_hamiltonian(shape, ...)
    driver = TDVPDriver(
        model, hamiltonian,
        preconditioner=SRPreconditioner(
            space=ParameterSpace(),
            strategy=DirectSolve(solver=solve_cholesky),
            diag_shift=args.diag_shift,
            metrics_config=DEFAULT_METRICS_CONFIG,
        ),
        dt=args.dt,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(args.seed),
        n_samples=args.n_samples,
        n_chains=args.n_chains,
    )
    run(
        driver,
        n_steps=args.n_steps,
        run_dir=_run_dir(),
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
        extra_config={
            "quoted_ed_energy_per_site": QUOTED_ED_ENERGY_PER_SITE,
            "particle_number": PARTICLE_NUMBER,
        },
    )
```

**Note on finite_size_scaling.py:** This script uses `build_central_bulk_observable` to create observables at different bulk sizes. These become named observables like `bulk16`, `bulk14`, `bulk12` passed to the runner via `observable_names`.

- [ ] **Step 1: Trim common.py**
- [ ] **Step 2: Refactor benchmark_3x3.py**
- [ ] **Step 3: Refactor bond_dim_scan.py**
- [ ] **Step 4: Refactor energy_vs_J.py**
- [ ] **Step 5: Refactor finite_size_scaling.py**
- [ ] **Step 6: Test**: `JAX_PLATFORM_NAME=cpu uv run python examples/lgt/z2_hardcore_boson/benchmark_3x3.py --n-steps 3 --n-samples 64 --n-chains 8 --log-every 1 --save-every 3`
- [ ] **Step 7: Commit**

---

### Task 5: Consolidate z2_vison scripts

**Files:**
- Create: `examples/lgt/z2_vison/ground_state.py`
- Create: `examples/lgt/z2_vison/dynamics.py`
- Create: `examples/lgt/z2_vison/plot.py`
- Move: `examples/lgt/z2_vison_propagation/exact.py` → `examples/lgt/z2_vison/exact.py`
- Delete: `examples/lgt/z2_vison_propagation/__main__.py`
- Delete: `examples/lgt/z2_vison_propagation_L10/__main__.py`

This consolidates the 6x6 and 10x10 vison propagation scripts into a single pair. The lattice size L is a CLI parameter. The runner handles checkpointing/resume.

**Physics to preserve from both scripts:**
- `build_z2_hamiltonian` (identical in both)
- `build_model` (identical in both)
- `create_bottom_left_vison` (identical in both)
- `_site_independent_directions`, `_z2_phase_for_direction` (identical)
- `build_selected_plaquette_observables` with `open_to_internal_plaquette` coordinate transform
- `save_model_state`, `load_model_state` (for ground-state → dynamics handoff)

**ground_state.py pattern:**

```python
#!/usr/bin/env python
"""Z2 pure gauge ground-state optimization."""
# sys.path setup, imports...
# Physics functions: build_z2_hamiltonian, build_model, save_model_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=0.1)
    add_common_args(parser)
    args = parser.parse_args()

    shape = (args.L, args.L)
    model = build_model(shape, bond_dim=args.bond_dim, seed=args.seed)
    hamiltonian = build_z2_hamiltonian(shape, h=args.h, g=args.g)
    driver = TDVPDriver(model, hamiltonian, ...)

    run_dir = f"data/z2_vison/L{args.L}_g{args.g}_Dk{args.bond_dim}"
    run(driver, n_steps=args.n_steps, run_dir=run_dir, ...)

    # Save model state for dynamics handoff
    save_model_state(driver.model, {...}, Path(run_dir) / "ground_state.npz")
```

**dynamics.py pattern:**

```python
#!/usr/bin/env python
"""Z2 vison propagation dynamics."""
# sys.path setup, imports...
# Physics: build_z2_hamiltonian, build_model, load_model_state,
#          create_bottom_left_vison, build_selected_plaquette_observables

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=0.1)
    add_common_args(parser)
    args = parser.parse_args()

    model, metadata = load_model_state(args.state)
    model = create_bottom_left_vison(model)
    shape = model.shape
    hamiltonian = build_z2_hamiltonian(shape, ...)
    observables = build_selected_plaquette_observables(shape)
    plaq_names = tuple(f"P_{r}{c}" for r, c in selected_open_plaquettes(shape))

    driver = TDVPDriver(
        model, hamiltonian, observables=observables,
        time_unit=RealTimeUnit(), integrator=RK4(), ...
    )

    run_dir = f"data/z2_vison/L{args.L}_g{args.g}_Dk{args.bond_dim}_rt"
    run(driver, T_final=args.T_final, run_dir=run_dir,
        observable_names=plaq_names, resume=args.resume, ...)
```

**plot.py:** Extract plot_fig5a and exact data download/cache from z2_vison_propagation. No runner dependency — reads the JSON output.

- [ ] **Step 1: Create z2_vison/ directory**
- [ ] **Step 2: Write ground_state.py** (merge physics from both propagation + L10)
- [ ] **Step 3: Write dynamics.py** (merge both real-time subcommands)
- [ ] **Step 4: Write plot.py** (extract from z2_vison_propagation)
- [ ] **Step 5: Move exact.py** to z2_vison/
- [ ] **Step 6: Test ground_state**: `JAX_PLATFORM_NAME=cpu uv run python examples/lgt/z2_vison/ground_state.py --L 4 --n-steps 3 --n-samples 64 --n-chains 8 --log-every 1 --save-every 3 --bond-dim 2`
- [ ] **Step 7: Delete old __main__.py files and empty directories**
- [ ] **Step 8: Commit**

---

### Task 6: Refactor z2_vison_higgs_confinement

**Files:**
- Create: `examples/lgt/z2_vison_higgs/ground_state.py`
- Create: `examples/lgt/z2_vison_higgs/dynamics.py`
- Delete: `examples/lgt/z2_vison_higgs_confinement/__main__.py`

Same consolidation pattern as Task 5 but for the Higgs variant. Read the existing `__main__.py` carefully — it uses Higgs-specific terms (parity-conserving GI terms, interior vison pair creation rather than boundary vison).

**Key physics to preserve:**
- The Higgs Hamiltonian with matter mass term (`2n`), electric term (`0.5g(2-2Z)`), hopping (`-J sigma_x X sigma_x`), and plaquette (`-h B_p`)
- Interior vison pair creation (different from boundary vison in z2_vison)
- `--T-final` mode for real-time (the spec's T_final maps directly to runner)
- Parity sector handling

- [ ] **Step 1: Read z2_vison_higgs_confinement/__main__.py thoroughly**
- [ ] **Step 2: Write ground_state.py**
- [ ] **Step 3: Write dynamics.py**
- [ ] **Step 4: Test ground_state**
- [ ] **Step 5: Delete old script and directory**
- [ ] **Step 6: Commit**

---

### Task 7: Refactor odd_z2_gauge.py and z3_pure_gauge.py

**Files:**
- Modify: `examples/lgt/odd_z2_gauge.py`
- Modify: `examples/lgt/z3_pure_gauge.py`
- Delete: `examples/lgt/observables.py`

**Changes (identical pattern for both):**
1. Remove `from .observables import SimulationData, format_step_log`
2. Remove `SimulationData` usage and `data.add_step` calls
3. Replace chunked `driver.run(k * dt)` loop with `runner.run()`
4. Replace `scan_g()` to call `runner.run()` per g value, each with its own run_dir
5. Add CLI via `add_common_args` (currently these use function parameters, not argparse)
6. Use `DEFAULT_METRICS_CONFIG`

**Refactored main() for odd_z2_gauge.py:**

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--g", type=float, default=0.5)
    parser.add_argument("--h", type=float, default=1.0)
    add_common_args(parser)
    args = parser.parse_args()

    shape = (args.L, args.L)
    model = GIPEPS(
        rngs=nnx.Rngs(args.seed),
        config=GIPEPSConfig(shape=shape, N=2, phys_dim=1, Qx=1, ...),
        contraction_strategy=ZipUp(truncate_bond_dimension=3 * args.bond_dim),
    )
    hamiltonian = build_odd_z2_hamiltonian(shape, h=args.h, g=args.g)
    driver = TDVPDriver(
        model, hamiltonian,
        preconditioner=SRPreconditioner(
            diag_shift=args.diag_shift, metrics_config=DEFAULT_METRICS_CONFIG,
        ),
        dt=args.dt, time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(args.seed),
        n_samples=args.n_samples, n_chains=args.n_chains,
        full_gradient=args.full_gradient,
    )
    run_dir = f"data/odd_z2/L{args.L}_g{args.g}"
    run(driver, n_steps=args.n_steps, run_dir=run_dir,
        log_every=args.log_every, save_every=args.save_every,
        extra_config={"L": args.L, "g": args.g, "h": args.h, "Qx": 1})
```

**scan_g() becomes:**
```python
def scan_g():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--g-values", type=float, nargs="+",
                        default=[0.3, 0.4, 0.5, 0.6, 0.64, 0.7, 0.8, 0.9])
    add_common_args(parser)
    args = parser.parse_args()
    for g in args.g_values:
        args.g = g
        main_with_args(args)  # each gets own run_dir
```

- [ ] **Step 1: Refactor odd_z2_gauge.py**
- [ ] **Step 2: Refactor z3_pure_gauge.py** (same pattern, N=3)
- [ ] **Step 3: Delete observables.py**
- [ ] **Step 4: Test**: `JAX_PLATFORM_NAME=cpu uv run python -m examples.lgt.odd_z2_gauge --L 4 --n-steps 5 --n-samples 64 --n-chains 8 --log-every 1 --save-every 5 --bond-dim 2`
- [ ] **Step 5: Commit**

---

### Task 8: Refactor schmitt_tfim_quench.py

**Files:**
- Modify: `examples/Schmitt_2022_TFIM2d/schmitt_tfim_quench.py`

**Changes:**
1. Remove `solver_from_name` — use `resolve_solver` from runner
2. Remove `RunConfig` dataclass — use argparse args directly
3. Remove `measurement_row` — runner handles series accumulation
4. Replace `run_single_quench()` loop with `runner.run()` using `T_final`
5. Keep all physics: `build_schmitt_smooth_tfim`, `build_plus_x_product_peps`, `build_mx_observable`, `build_center_czz_observables`
6. Keep `run_tauq_sweep` as outer loop calling runner per tau_q

**Refactored pattern:**

```python
def run_single_quench(args):
    t0, t1 = smooth_time_window(args.tau_q)
    model = build_plus_x_product_peps(...)
    distances = center_axis_distances(args.shape)
    obs_names = ("mx", *(f"czz_r{d}" for d in distances))
    observables = (build_mx_observable(...), *build_center_czz_observables(...))
    driver = TDVPDriver(
        model,
        build_schmitt_smooth_tfim(...),
        observables=observables,
        preconditioner=SRPreconditioner(
            diag_shift=args.diag_shift,
            strategy=DirectSolve(solver=resolve_solver(args.solver)),
            metrics_config=DEFAULT_METRICS_CONFIG,
        ),
        dt=args.dt, t0=t0,
        time_unit=RealTimeUnit(), integrator=RK4(),
        sampler_key=jax.random.key(args.seed + 17),
        n_samples=args.n_samples, n_chains=args.n_chains,
    )
    run_dir = f"data/tfim_quench/L{args.L}_tauq{args.tau_q}_D{args.bond_dim}"
    run(driver, T_final=t1, run_dir=run_dir,
        observable_names=obs_names,
        log_every=args.log_every, save_every=args.save_every,
        extra_config={"L": args.L, "tau_q": args.tau_q, "gc": GC})
```

- [ ] **Step 1: Refactor the script**
- [ ] **Step 2: Test**: `JAX_PLATFORM_NAME=cpu uv run python examples/Schmitt_2022_TFIM2d/schmitt_tfim_quench.py --L 3 --tau-q 0.8 --dt 0.1 --bond-dim 2 --boundary-dim 4 --n-samples 64 --n-chains 8 --log-every 1 --save-every 5`
- [ ] **Step 3: Commit**

---

### Task 9: Refactor heisenberg.py and ising.py

**Files:**
- Modify: `examples/ground_states/heisenberg.py`
- Modify: `examples/ground_states/ising.py`

These are demo scripts with 3 optimization methods each. Light refactoring:

**Changes:**
1. Remove duplicated `SR_METRICS_CONFIG` → use `DEFAULT_METRICS_CONFIG` from runner
2. Remove duplicated `append_series` → inline `series.setdefault(k, []).append(v)`
3. For `run_sr()` with `n_steps` (fixed-step): use `runner.run()` for the loop. Keep custom `save_run()` output format for the plot functions.
4. For `run_sr()` with `target_time` (adaptive): keep custom loop (variable dt, manual tensor updates)
5. For `run_adam()`: keep custom loop (no driver.run)
6. Keep all physics builders and plot functions unchanged

**SR fixed method can use runner.run():** Build driver, call `run()`, then read the series from the JSON to build the custom output format for `save_run()`. Alternatively, just clean up the existing loop since it's a demo script. Either approach is fine — prefer whichever is simpler.

Given these are demo/benchmark scripts, the simplest approach is: import `DEFAULT_METRICS_CONFIG` from runner, clean up duplicated helpers, but keep the custom loops for all 3 methods. The educational value of seeing the explicit loop outweighs the DRY benefit.

- [ ] **Step 1: Refactor heisenberg.py** (import DEFAULT_METRICS_CONFIG, remove duplicated helpers)
- [ ] **Step 2: Refactor ising.py** (same pattern)
- [ ] **Step 3: Run full test suite**: `JAX_PLATFORM_NAME=cpu uv run pytest -m "not slow" -v`
- [ ] **Step 4: Commit**

---

### Task 10: Clean up exact_tdvp_3x3_check.py

**Files:**
- Modify: `examples/time_dependent/exact_tdvp_3x3_check.py`

**Changes:**
1. Remove inline `solver_from_name` dict → use `resolve_solver` from runner
2. Import `DEFAULT_METRICS_CONFIG` if applicable
3. Clean up formatting/style

This is a standalone comparison script with a fundamentally different structure (dual exact vs PEPS trajectory). Keep the core logic, just clean imports.

- [ ] **Step 1: Refactor**
- [ ] **Step 2: Commit**

---

### Task 11: Delete obsolete files and final validation

**Files to delete:**
- `examples/lgt/observables.py` (if not deleted in Task 7)
- `examples/lgt/z2_vison_propagation/__main__.py`
- `examples/lgt/z2_vison_propagation_L10/__main__.py`
- `examples/lgt/z2_vison_higgs_confinement/__main__.py`
- Empty directories after deletion

**Validation:**
- [ ] **Step 1: Delete files**
- [ ] **Step 2: Run full test suite**: `JAX_PLATFORM_NAME=cpu uv run pytest -m "not slow" -v`
- [ ] **Step 3: Grep for broken imports**: `grep -r "from.*observables import\|from.*common import.*run_ground_state\|from.*common import.*save_latest\|from.*common import.*maybe_resume" examples/`
- [ ] **Step 4: Commit**

---

## Execution Notes

- **Tasks 3–10 are independent** and can run in parallel after Tasks 1–2 complete.
- **Task 11 depends on all others** (cleanup + validation).
- When refactoring, preserve exact physics (model construction, operator terms, observable definitions). Only replace run loop infrastructure.
- For `sys.path` imports: use `sys.path.insert(0, str(Path(__file__).resolve().parents[N]))` where N depends on directory depth. `examples/lgt/z2_pure_gauge.py` needs `parents[1]`, `examples/lgt/z2_hardcore_boson/benchmark_3x3.py` needs `parents[2]`.
- The `__package__` guard pattern from z2_hardcore_boson should be preserved for `common.py` imports since those scripts import from a sibling file.
