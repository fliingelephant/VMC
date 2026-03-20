# Runner Module Design

Shared run infrastructure for PEPS-VMC example scripts. One file (`examples/runner.py`) providing a universal run loop, checkpointing, resume, CLI argument helpers, and pre-flight config display.

## Motivation

The 17 example scripts duplicate: run loops, checkpointing, resume logic, per-step printing, CLI argument definitions, and output file management. The runner extracts the common 90% while leaving problem-specific logic (model construction, state preparation, output directory naming) in each script.

## API

```python
# examples/runner.py

def add_common_args(parser: argparse.ArgumentParser) -> None
def run(driver, *, n_steps=None, T_final=None, run_dir, ...) -> None
def save_checkpoint(run_dir, driver, step, *, series=None, **metadata) -> None
def load_checkpoint(run_dir, driver) -> dict
```

### `add_common_args(parser)`

Adds standard CLI arguments used by nearly every script:

| Argument | Type | Default | Maps to |
|---|---|---|---|
| `--n-samples` | int | 10240 | `TDVPDriver(n_samples=)` |
| `--n-chains` | int | 1024 | `TDVPDriver(n_chains=)` |
| `--dt` | float | 0.01 | `TDVPDriver(dt=)` |
| `--seed` | int | 42 | `jax.random.key(seed)` |
| `--bond-dim` | int | 4 | Model construction (script-interpreted) |
| `--boundary-dim` | int | 16 | `Variational(boundary_dim)` |
| `--diag-shift` | float | 1e-4 | `SRPreconditioner(diag_shift=)` |
| `--solver` | str | `"cholesky"` | `solve_cholesky`, `solve_svd`, or `solve_cg` |
| `--solver-space` | str | `"sr"` | `"sr"` → `ParameterSpace()`, `"minsr"` → `SampleSpace()` |
| `--full-gradient` | flag | False | `TDVPDriver(full_gradient=)` |
| `--gauge-removal` | flag | False | `GaugeConfig` if set, else `None` |
| `--log-every` | int | 10 | Print every N steps |
| `--save-every` | int | 50 | Checkpoint every N steps |
| `--resume` | flag | False | Load checkpoint and continue |
| `--n-steps` | int | — | Number of additional steps to run |
| `--T-final` | float | — | Absolute target time (real time) |

Notes:
- `--bond-dim` is interpreted by the script, not the runner. For GIPEPS it may map to `degeneracy_per_charge`, not a single bond dimension.
- `--solver` choices: `cholesky` (default), `svd`, `cg`. Maps to `solve_cholesky`, `solve_svd`, `solve_cg` from `vmc.preconditioners`.
- `--full-gradient` disables the small-o trick (SlicedJacobian). Significantly more expensive — show prominently in config table.

The script reads these from `args` to construct model + driver. The runner does not construct the model or driver — that stays in the script.

### `run(driver, *, ...)`

```python
def run(
    driver: TDVPDriver,
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
```

Exactly one of `n_steps` or `T_final` must be provided.

**Behavior:**

1. **Resume** — if `resume=True`, load checkpoint from `run_dir`, restore driver state (mutates driver in-place: tensors, sampler key/configuration, step_count, time)
2. **Print config table** — auto-extracted from driver + model + preconditioner + `extra_config` (see below)
3. **Create run_dir** — `Path(run_dir).mkdir(parents=True, exist_ok=True)`. Never deletes existing files.
4. **Loop** — call `driver.run(driver.dt)` per step, print progress, accumulate series, periodic checkpoint
5. **Final save** — checkpoint + series at end

The runner reads `driver.step_count` and `driver.t` as the authoritative step/time counters. It does not maintain independent counters.

**Stop condition:**
- `n_steps`: run this many additional steps from current position. On resume from step 200 with `n_steps=200`, runs steps 201–400.
- `T_final`: absolute target time. On resume from `t=12.0` with `T_final=18.0`, runs until `t≥18.0`. Works with any starting time (including negative, e.g. quench protocols starting at `t₀ = -1.5τ_q`).

### `save_checkpoint(run_dir, driver, step, *, series=None, **metadata)`

Saves driver state to `run_dir/`. Exposed for scripts needing custom loops (adaptive dt, Adam).

Both `latest.npz` and `latest.json` are written atomically: write to a temporary file, then `os.replace()` to the final path. This prevents corruption if the process is killed mid-write.

### `load_checkpoint(run_dir, driver) -> dict`

Restores driver state from `run_dir/`. **Mutates `driver` in-place:**
- `driver._tensors` — model parameters
- `driver._sampler_key` — PRNG key (reconstructed via `jax.random.wrap_key_data` from stored `sampler_key_impl`)
- `driver._sampler_configuration` — Markov chain states
- `driver.step_count` — step counter
- `driver.t` — current time

Returns metadata dict (step, time, series, config, extra metadata). Exposed for custom loops.

## Config Table

Printed by `run()` before the loop starts. Shows everything that could be accidentally misconfigured. Auto-extracted from the driver, model, and preconditioner — no manual specification needed for standard fields.

```
── Run Configuration ──────────────────────────
Device          cuda (NVIDIA A100)
Model           GIPEPS (10×10)
Bond dim        8 (D_k=2, N=2)
Parameters      12,416 (block-sparse)
Strategy        Variational(D'=16)
dtype           complex128
Hamiltonian     TimeDependentHamiltonian(CubicSchedule)
Integrator      RK4 (real time)
dt              0.01
t0              -1.200
Target          T = 1.200 (240 steps)
Samples         10240 (1024 chains)
Full gradient   False
Solver          DirectSolve(cholesky) · minSR
Ordering        SliceOrdering
Diag shift      1.0e-04
Gauge removal   None
Observables     mx, czz_r1, czz_r2
── Resume ─────────────────────────────────────
Starting fresh  t = -1.200
── Output ─────────────────────────────────────
Run dir         data/tfim_quench/L7_tauq0.8_D4
Log every       10
Save every      50
── Problem ────────────────────────────────────
L               7
tau_q           0.8
initial_state   product |+x>
───────────────────────────────────────────────
```

Imaginary-time resume example:

```
── Run Configuration ──────────────────────────
Device          cuda (NVIDIA A100)
Model           GIPEPS (10×10)
Bond dim        8 (D_k=2, N=2)
Parameters      12,416 (block-sparse)
Strategy        Variational(D'=16)
dtype           complex128
Hamiltonian     LocalHamiltonian
Integrator      Euler (imaginary time)
dt              0.01
Target          200 steps → step 400
Samples         10240 (1024 chains)
Full gradient   False
Solver          DirectSolve(cholesky) · minSR
Ordering        SliceOrdering
Diag shift      1.0e-04
Gauge removal   enabled
Observables     plaquette, link_z
── Resume ─────────────────────────────────────
Checkpoint      step 200, t = 2.000
Remaining       200 steps (t: 2.000 → 4.000)
── Output ─────────────────────────────────────
Run dir         data/z2_pure/L10_g0.3_D8
Log every       10
Save every      50
── Problem ────────────────────────────────────
L               10
g               0.3
───────────────────────────────────────────────
```

**Extraction sources:**
- Device: `jax.devices()[0]` — platform and device name
- Model type: `type(model).__name__`
- Shape: `model.shape`
- Bond dimension: `model.bond_dim` for PEPS; `model.dmax` and `model.degeneracy_per_charge` for GIPEPS; `model.Dmax` for BlockadePEPS
- Parameters: real number of free parameters, accounting for block-sparsity in GIPEPS (sum of non-zero elements across all tensors, not total tensor size)
- Contraction strategy: `model.strategy` (attribute name is `strategy`, not `contraction_strategy`)
  - For `Variational`: display `model.strategy.truncate_bond_dimension` as D'
  - For `ZipUp`, `DensityMatrix`, `NoTruncation`: display class name
- dtype: `model.dtype`
- GIPEPS extras: `model.config.N`, `model.config.degeneracy_per_charge`
- Hamiltonian type: `type(operator).__name__`, and schedule class if `TimeDependentHamiltonian`
- Driver: `driver.dt`, `driver.t` (as t0), `driver.n_samples`, `driver.n_chains`, `driver.full_gradient`, `driver.step_count`
- Time unit / integrator: extracted from driver internals
- Preconditioner: `driver.preconditioner.diag_shift`, `driver.preconditioner.space`, `driver.preconditioner.strategy`, `driver.preconditioner.gauge_config`, `driver.preconditioner.ordering`
- `extra_config` dict: problem-specific parameters (L, g, J, initial_state, vison_site, etc.)

**Missing metrics handling:** If a metric (FS_norm, TDVP_residual, SR_solve_residual) is not recorded (MetricsConfig flag is False), display "---" in the per-step print and omit from series.

## Per-Step Print

Uses `print()` (not `logging`) due to a logging-level bug that blocks output.

**Header** printed once before the loop:

```
step      time       energy              error      variance   plaq       link_z     FS_norm    TDVP_res   SR_res     wall
```

**Per step** (every `log_every` steps):

```
   10     0.100     -0.7639730142 ±   2.3e-04      5.1e-03    0.8234     0.1234     1.2e-03    3.4e-06    1.2e-12    1.2s
```

Fields:
- `step`: integer step count (from `driver.step_count`)
- `time`: imaginary or real time (from `driver.t`)
- `energy`: mean ± error of mean
- `variance`: energy variance
- Observable values: mean for each named observable
- `FS_norm`: Fubini-Study norm squared (from metrics, if recorded)
- `TDVP_res`: TDVP residual (from metrics, if recorded)
- `SR_res`: SR solve residual (from metrics, if recorded)
- `wall`: wall-clock time for the step (from metrics, if recorded)

All values flushed (`flush=True`). All floats are real-valued (`.real` on complex JAX values).

## Checkpoint Format

Two files in `run_dir/`, written atomically (tmp + `os.replace()`):

### `latest.npz`

NumPy compressed archive containing:
- Model tensor arrays (one key per tensor, e.g., `tensor_0_0`, `tensor_0_1`, ...)
- `sampler_key_impl`: raw PRNG key data (from `jax.random.key_data(key)`), needed for `jax.random.wrap_key_data()` reconstruction
- `sampler_configuration`: current Markov chain configurations

### `latest.json`

Human-readable metadata + accumulated series:

```json
{
  "step": 200,
  "time": 2.0,
  "config": {
    "model": "GIPEPS",
    "shape": [10, 10],
    "bond_dim": 8,
    "dt": 0.01,
    "n_samples": 10240,
    "n_chains": 1024,
    "solver_space": "sr",
    "diag_shift": 0.0001,
    "extra": {"L": 10, "g": 0.3}
  },
  "series": {
    "step": [1, 2, 3],
    "time": [0.01, 0.02, 0.03],
    "energy_mean": [-0.123, -0.456, -0.763],
    "energy_error": [0.01, 0.005, 0.002],
    "energy_variance": [0.1, 0.05, 0.005],
    "plaq_mean": [0.5, 0.7, 0.82],
    "plaq_error": [0.01, 0.005, 0.002],
    "link_z_mean": [0.1, 0.12, 0.123],
    "link_z_error": [0.01, 0.005, 0.002],
    "FS_norm_squared": [0.01, 0.005, 0.001],
    "TDVP_residual": [1e-4, 1e-5, 1e-6],
    "SR_solve_residual": [1e-10, 1e-11, 1e-12],
    "wall_time": [1.2, 1.1, 1.15]
  }
}
```

On resume, series is loaded and new steps are appended.

## Resume Semantics

### Imaginary Time

```bash
# First run: 200 steps
python ground_state.py --n-steps 200

# Not converged — run 200 more
python ground_state.py --n-steps 200 --resume

# Still not converged — 200 more
python ground_state.py --n-steps 200 --resume
```

`--n-steps` is the number of additional steps to run. On resume, the runner loads the checkpoint (e.g., step 200), then runs 200 more steps (201–400). The config table shows: `Target  200 steps → step 400`.

### Real Time

```bash
# Load ground state, run dynamics to T=12
python dynamics.py --state data/.../latest.npz --T-final 12

# Extend to T=18
python dynamics.py --T-final 18 --resume

# Extend further
python dynamics.py --T-final 24 --resume
```

`--T-final` is the absolute target time. On resume from `t=12.0`, runs until `t=18.0`.

### Validation

On resume, the following are checked against the checkpoint config:
- **Must match** (error if mismatch): model type, shape
- **Warn on mismatch** (continue): dt, solver_space, diag_shift, n_samples, n_chains

If `T_final ≤ driver.t` (for real time), print message and exit without error.

## Script Structure

Each physics problem has its own directory with separate scripts:

```
examples/
  runner.py                          # shared infrastructure
  ground_states/
    heisenberg.py                    # imaginary time only
    ising.py                         # imaginary time only
  lgt/
    z2_pure_gauge/
      ground_state.py                # imaginary time
    z2_hardcore_boson/
      ground_state.py                # imaginary time
    z2_vison/
      ground_state.py                # imaginary time
      dynamics.py                    # real time (loads ground state)
    z2_vison_higgs/
      ground_state.py                # imaginary time
      dynamics.py                    # real time
  Schmitt_2022_TFIM2d/
    quench.py                        # real time from product state
```

### Example: Imaginary-Time Script

```python
#!/usr/bin/env python
"""Z2 pure gauge ground-state optimization."""
from __future__ import annotations

from vmc import config  # noqa: F401

import argparse
import jax
from flax import nnx

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver
from vmc.gauge import GaugeConfig
from vmc.peps.gi import GIPEPS, GIPEPSConfig
from vmc.peps.common.strategy import Variational
from vmc.preconditioners import DirectSolve, MetricsConfig, SRPreconditioner, solve_cholesky
from vmc.qgt import ParameterSpace, SampleSpace

from examples.runner import add_common_args, run


def build_model(args):
    return GIPEPS(
        rngs=nnx.Rngs(args.seed),
        config=GIPEPSConfig(shape=(args.L, args.L), N=2, Qx=0, ...),
        contraction_strategy=Variational(args.boundary_dim),
    )


def build_hamiltonian(args):
    ...
    return operator, observables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--g", type=float, default=0.3)
    add_common_args(parser)
    args = parser.parse_args()

    model = build_model(args)
    operator, observables = build_hamiltonian(args)

    space = SampleSpace() if args.solver_space == "minsr" else ParameterSpace()
    driver = TDVPDriver(
        model, operator,
        observables=observables,
        preconditioner=SRPreconditioner(
            space=space,
            strategy=DirectSolve(solver=solve_cholesky),
            diag_shift=args.diag_shift,
            gauge_config=GaugeConfig() if args.gauge_removal else None,
            metrics_config=MetricsConfig(
                record_FS_norm=True,
                record_TDVP_residual=True,
                record_SR_solve_residual=True,
                record_step_wall_time=True,
            ),
        ),
        dt=args.dt,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(args.seed),
        n_samples=args.n_samples,
        n_chains=args.n_chains,
    )

    run_dir = f"data/z2_pure/L{args.L}_g{args.g}_D{args.bond_dim}"
    run(driver, n_steps=args.n_steps, run_dir=run_dir,
        observable_names=("plaquette", "link_z"),
        log_every=args.log_every, save_every=args.save_every,
        resume=args.resume,
        extra_config={"L": args.L, "g": args.g})


if __name__ == "__main__":
    main()
```

### Example: Real-Time Script

```python
#!/usr/bin/env python
"""Z2 vison propagation dynamics."""
from __future__ import annotations

from vmc import config  # noqa: F401

import argparse
import jax
from flax import nnx

from vmc.drivers import RealTimeUnit, RK4, TDVPDriver
from vmc.gauge import GaugeConfig
from vmc.peps.gi import GIPEPS, GIPEPSConfig
from vmc.peps.common.strategy import Variational
from vmc.preconditioners import DirectSolve, MetricsConfig, SRPreconditioner, solve_cholesky
from vmc.qgt import ParameterSpace, SampleSpace

from examples.runner import add_common_args, run, load_checkpoint


def load_ground_state(state_path, args):
    """Build model and load tensors from ground-state checkpoint."""
    model = build_model(args)
    # Load tensors from .npz into model
    ...
    return model


def insert_vison(model, site):
    """Apply sigma_z on a link to create a vison excitation."""
    ...


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, required=True,
                        help="Path to ground-state .npz")
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--g", type=float, default=0.1)
    add_common_args(parser)
    args = parser.parse_args()

    model = load_ground_state(args.state, args)
    insert_vison(model, site=(0, 0))

    operator, observables = build_hamiltonian(args)

    space = SampleSpace() if args.solver_space == "minsr" else ParameterSpace()
    driver = TDVPDriver(
        model, operator,
        observables=observables,
        preconditioner=SRPreconditioner(
            space=space,
            strategy=DirectSolve(solver=solve_cholesky),
            diag_shift=args.diag_shift,
            gauge_config=GaugeConfig() if args.gauge_removal else None,
            metrics_config=MetricsConfig(
                record_FS_norm=True,
                record_TDVP_residual=True,
                record_SR_solve_residual=True,
                record_step_wall_time=True,
            ),
        ),
        dt=args.dt,
        time_unit=RealTimeUnit(),
        integrator=RK4(),
        sampler_key=jax.random.key(args.seed),
        n_samples=args.n_samples,
        n_chains=args.n_chains,
    )

    run_dir = f"data/z2_vison_rt/L{args.L}_g{args.g}_D{args.bond_dim}"
    run(driver, T_final=args.T_final, run_dir=run_dir,
        observable_names=("P_00", "P_01", "P_22"),
        log_every=args.log_every, save_every=args.save_every,
        resume=args.resume,
        extra_config={"L": args.L, "g": args.g})


if __name__ == "__main__":
    main()
```

## Incompatible Patterns

These script patterns cannot use `run()` and should use `save_checkpoint`/`load_checkpoint` directly:

- **Adaptive time-stepping** — variable dt per step (e.g., ising.py SR adaptive)
- **Adam optimizer** — custom update rule, not using TDVPDriver
- **Multi-step `driver.run(k * dt)`** — some scripts call driver.run with multiples of dt
- **Plotting** — separate concern, stays in scripts or dedicated plot scripts

## Implementation Notes

- Use `print(..., flush=True)` everywhere, not `logging`, due to a logging-level bug that blocks output.
- The preconditioner is accessible via `driver.preconditioner`.
- Checkpoint files are written atomically (write to `.tmp`, then `os.replace()`).
- `run()` creates `run_dir` via `mkdir(parents=True, exist_ok=True)` but never deletes existing files.
- Series is stored in columnar format for efficient JSON I/O and easy numpy/pandas loading.
- All `float` values in the series are real (call `.real` on JAX complex values).
- PRNG key saved as `jax.random.key_data(key)` and restored via `jax.random.wrap_key_data()`.
- Wall-time recording via `MetricsConfig(record_step_wall_time=True)` forces JAX synchronization per step; this is expected and necessary for timing.
- Parameter count for config table: for GIPEPS, count actual non-zero elements in block-sparse tensors (charge-sector slices), not the full dense tensor size.
- State preparation (product states, vison insertion, etc.) is script-specific. Document in `extra_config` for visibility in the config table (e.g., `extra_config={"initial_state": "product |+x>", "vison_site": (0,0)}`).
