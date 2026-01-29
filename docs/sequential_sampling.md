# Sequential Sampling

This repo implements a fixed-order, single-spin Metropolis sampler for both MPS and PEPS.
The goal is to reuse environments aggressively while keeping the transition kernel simple.

## What was implemented

- **MPS**: `_sequential_mps_sweep` in `VMC/samplers/sequential.py` performs one sweep.
  `sequential_sample` wraps burn-in and sampling loops via plum dispatch.
  Recorded samples are taken once per sweep.
  `n_samples` controls how many sweeps are recorded; total samples are `n_samples`
  (burn-in sweeps are not recorded).
  - Precompute right environments for the current configuration.
  - Sweep sites left-to-right, propose a single-spin flip, and accept with
    `min(1, |psi_new|^2 / |psi_old|^2)`.
  - Update the left environment immediately after acceptance so later sites reuse the cache.
  - Tensors are padded to a uniform bond dimension so the sweep can run inside a JAX scan.

- **PEPS**: `sequential_sample` in `VMC/samplers/sequential.py` wraps burn-in and sampling
  loops via plum dispatch, using internal dispatched `sweep` and `bottom_envs` functions.
  Recorded samples are taken once per sweep.
  `n_samples` controls how many sweeps are recorded; total samples are `n_samples`
  (burn-in sweeps are not recorded).
  - Build bottom environments once per sweep.
  - For each row, build right environments once and reuse a left environment while moving left-to-right.
  - Propose a single-spin flip at each site and accept with the same ratio.
  - Update the top boundary after finishing each row.

Both samplers generate a **chain**: accepted flips are kept and the sweep continues on the updated
configuration.

## Verify script

`examples/verify_sequential_mh.py` compares sequential sampling against FullSum and NetKet
MetropolisLocal.

Defaults follow your request (3x3 FullSum; 5x5/8x8 MH) and use large sample counts.

Examples:

```bash
VMC_LOG_LEVEL=INFO ./.venv/bin/python examples/verify_sequential_mh.py
```

```bash
VMC_LOG_LEVEL=INFO ./.venv/bin/python -m VMC.examples.verify_sequential_mh
```

The script logs energy mean/std for sequential vs NetKet MH and reports the FullSum baseline
for the 3x3 case.

## Unit tests

Minimal distribution sanity checks live in `tests/test_sequential_sampling.py` and can be run with:

```bash
./.venv/bin/python -m unittest tests.test_sequential_sampling
```

These tests keep sizes small and use fixed RNG seeds to stay deterministic.
