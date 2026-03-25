# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Variational Monte Carlo dynamics for projected entangled-pair states (PEPS), implemented with JAX primitives and NetKet interfaces. Under active development.

## Commands

```bash
# Install/sync dependencies
uv sync --frozen --group dev

# Run all tests (skip slow + physics models)
uv run pytest -m "not slow"

# Run all tests with coverage (matches CI)
uv run pytest -m "not slow" --cov=src/vmc --cov-branch --cov-report=term-missing

# Run a single test file or test
uv run pytest tests/test_sequential_sampling.py
uv run pytest tests/test_sequential_sampling.py::test_name -v

# Run with JAX on CPU (CI default)
JAX_PLATFORM_NAME=cpu uv run pytest -m "not slow"
```

## Architecture

### Source layout: `src/vmc/`

**Core framework** (`core/`): `make_mc_sampler(transition, estimate)` — generic MC rollout via `vmap(mc_sweep)` over chains + `lax.scan` over sweeps.

**Three PEPS model families** (`peps/`), each with `model.py`, `kernels.py`, `compat.py`:
- `standard/` — Canonical open-boundary PEPS (`PEPS` class)
- `blockade/` — Rydberg blockade-constrained PEPS (`BlockadePEPS`), directed gauge-canonical form with validity masking
- `gi/` — Gauge-invariant PEPS for Abelian LGT (`GIPEPS`), charge-sector-sliced tensors

**Shared PEPS infrastructure** (`peps/common/`):
- `contraction.py` — MPO-to-boundary contractions with top-boundary caching
- `energy.py` — Local energy and derivative computation (backward pass using cached envs)
- `strategy.py` — `ContractionStrategy` hierarchy: `NoTruncation`, `ZipUp`, `DensityMatrix`, `Variational`

**Operators** (`operators/`): `LocalHamiltonian` built from `OneSiteOperator`, `HorizontalTwoSiteOperator`, `VerticalTwoSiteOperator`, `PlaquetteOperator`, `DiagonalOperator`. Terms grouped by span via `BucketedOperators`. `TimeDependentHamiltonian` wraps base operators with time-varying coefficients.

**QGT** (`qgt/`): Lazy quantum geometric tensor (matvec without explicit S matrix). `Jacobian` (full) and `SlicedJacobian` (memory-efficient) with `SiteOrdering`/`SliceOrdering`.

**Preconditioners** (`preconditioners/`): `SRPreconditioner` for stochastic reconfiguration with `DirectSolve`/`QRSolve`.

**Drivers** (`drivers/`): `TDVPDriver` for real/imaginary-time evolution with `Euler`/`RK4` integrators.

**Gauge** (`gauge/`): Gauge removal/projection for numerical stability.

### Key design pattern: cache turnover

The MC sampler uses a cache-turnover pattern (see `REFACTOR.md`):
- `transition`: consumes `Cache(bottom_envs)`, sweeps top→bottom, emits `Context(amp, top_envs)`
- `estimate`: consumes `Context`, sweeps bottom→top, emits new `Cache` + `LocalEstimates`

`build_mc_kernels(model, operator)` dispatches via `plum` to produce model-specific `init_cache`, `transition`, `estimate` kernels. Static metadata (shape, strategy, bucketed terms) is closed over; dynamic data (tensors, configs, keys, cache) passed at runtime.

### Multi-dispatch pattern

```python
@build_mc_kernels.dispatch
def build_mc_kernels(model: PEPS, operator: LocalHamiltonian, ...): ...

@build_mc_kernels.dispatch
def build_mc_kernels(model: BlockadePEPS, operator: LocalHamiltonian, ...): ...
```

ABCs for single-type hierarchies; `plum @dispatch` for multi-type functions. Import dispatched functions directly (no aliases).

## Conventions

Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), with JAX-native patterns below taking precedence.

### Dispatching

- **Extending library functions** → `@library.func.dispatch`.
- **Adding overloads** → Define base with `@dispatch`, add overloads with `@base_func.dispatch`.
- **NO strings for dispatching.** Use typed objects.
- **NO aliased imports for dispatch.** Import the dispatched function directly (`from module import func`), never alias. Dispatch handles routing by type.
- **Minimize helper functions.** Inline short logic; only extract when reused 3+ times or significantly improves readability.

### Design

- **Einsum with `optimize=True`** over sequential tensordot. Use einsum with `optimize=True` or manually define contraction orders, instead of sequentially doing tensor/matrix operations.
- **Composition over inheritance**, no factory functions. Combine small strategy objects rather than deep class trees.
- **DRY (Don't Repeat Yourself).** Consolidate duplicated implementations into a single source of truth.
- **Least redundancy is top priority.** Reuse environments/transfers within a row or row-pair; evaluate gradients and energy together in the same pass; avoid rebuilding row/2-row transfers or per-site assembled tensors multiple times; only materialize extra boundaries when they reduce total compute.
- **Match theory first.** Verify Gauss-law conventions, term geometry (one-site/horizontal/vertical/plaquette), and sampling/energy formulas against the papers in `notes/`; if ambiguous, consult the notes and ask rather than guessing.
- **Think twice.** For complicated or important algorithms, think twice before implementing.
- **Uncertain correctness.** Implementation might be totally incorrect; for uncertain behavior, refer to notes or ask the user.
- **Unified eval API (core).** `_value`, `_grad`, and `_value_and_grad` are the only evaluation entrypoints; every other evaluation is a variant of these (plum-dispatched for MPS/PEPS). Avoid manual-dispatch name variants, `log_*` helpers, or `*_fn` wrappers.
- **Sampling gradients.** When a sampler records gradients, compute value+Jacobian for each proposal together and keep gradients only for accepted proposals.
- **Let it crash**: no defensive parameter checks; assume correct wiring and let errors surface.
- **Occupancy (0/1) internally**, spin (±1) only at NetKet API boundaries.

### Style

- **Julia-style defaults**: `def foo(x, y=10):` not `def foo(x, y=None): y = y or 10`
- **No intermediate variables**: return directly: `return expr` not `result = expr; return result`
- **No unused imports/variables.** Remove any defined but unreferenced code.

### JAX patterns

- **`jax.lax.scan`** for shape-uniform sequences; explicit loops for edge contractions where shapes differ.
- **Single `jax.vmap`** over entire pipeline for XLA fusion. Fuse multiple operations into one function and vmap that, rather than separate vmaps for each step.
- **No `jax.block_until_ready`** in hot paths — it breaks XLA fusion.
- Default dtype: `jnp.complex128` (64-bit enabled in `config.py`)
- All models are Flax NNX Modules with `list[list[nnx.Param]]` tensor storage

### Logging

- Use Python `logging` module, not `print()`.
- **Guard expensive debug computations** with `logger.isEnabledFor(logging.DEBUG)`.
