# PEPS tVMC Refactor Notes

## Goal

Keep a minimal and efficient PEPS tVMC path centered on transition/estimate
decomposition with maximal environment reuse.

## Current Design (Implemented)

- `src/vmc/refactored/core.py`
  - `make_mc_sampler(transition, estimate) -> mc_sampler`
  - generic/model-agnostic
  - core compute pattern: `vmap(mc_sweep)` over chains + `lax.scan` over sweeps
- `src/vmc/refactored/peps.py`
  - `build_mc_kernels(model, operator, full_gradient=...)`
  - model-specific kernels:
    - `init_cache(tensors, config_states) -> Cache`
    - `transition(tensors, config_state, key, cache) -> (config_next, key_next, Context)`
    - `estimate(tensors, config_next, context) -> (cache_next, LocalEstimates)`

## Core Contract

```python
mc_sampler(
    tensors,
    config_states,
    chain_keys,
    cache,
    *,
    n_steps,
) -> ((config_states_next, chain_keys_next, cache_next), local_estimates_history)
```

Notes:

- `make_mc_sampler` intentionally does not include `burn_in` or `n_samples` trimming.
- `burn_in` and trim (`[:n_samples]` after flatten) are wrapper-level policies.

## PEPS Cache Turnover Semantics

- Persistent cache: `Cache(bottom_envs=...)`
- `transition`:
  - consumes `bottom_envs`
  - sweeps top->bottom
  - emits `Context(amp, top_envs)`
- `estimate`:
  - consumes `Context`
  - sweeps bottom->top once to compute both:
    - local log-derivatives / local energy
    - `bottom_envs_next`

This is a cache-turnover pattern, not in-place cache mutation.

## Data Containers

Defined in `src/vmc/refactored/peps.py`:

- `Cache`
- `Context`
- `LocalEstimates`

Structure should remain fixed across scan steps. Tuple-structured env containers
are preferred at API boundaries.

## Closure Boundary (Efficiency-Critical)

Close over static metadata in `build_mc_kernels`:

- shape (`n_rows`, `n_cols`, `n_sites`)
- contraction strategy
- bucketed operator terms
- parameter metadata (`params_per_site`, etc.)
- `full_gradient`

Keep dynamic at runtime:

- `tensors` (variational parameters)
- `config_states`, `chain_keys`, `cache`

## JAX Placement Guidance

- `make_mc_sampler` remains pure JAX (no internal jit requirement).
- Put the main `jax.jit` on the outer RHS/time-step entrypoint in driver/sampler wrapper.
- Apply donation (`donate_argnums`) at that outer jitted boundary for large carry buffers.

## Benchmark/Correctness Status

- Refactored path reproduces `sequential_sample_with_gradients` outputs exactly
  (samples, gradients, `p`, amplitudes, local energies, key/final states) under
  matched key handling and burn-in semantics.
- Runtime is close to baseline and can be slightly faster or slightly slower
  depending on shape/chains/samples and wrapper structure.
- Therefore, refactor value is primarily cleaner separation + reusable kernels,
  with near-parity performance (not guaranteed universal speedup).

## Naming (Current)

- `build_mc_kernels`
- `make_mc_sampler`
- `transition`
- `estimate`
- `mc_sweep`
- `local_log_derivatives`
- `local_estimate`
- `active_slice_indices`
- `full_gradient`

## Non-Goals in This Pass

- Compatibility cleanup outside the main PEPS tVMC path (`eval.py`, legacy adapters).
- GI/Blockade unification implementation in the same patch set.
- Alternate no-gradient / no-energy specialized kernels.

