# Sequential Sampling

This repo implements a fixed-order, single-spin Metropolis sampler for both MPS and PEPS.
The goal is to reuse environments aggressively while keeping the transition kernel simple.

## What was implemented

- **MPS**: `_sequential_mh_mps_sweep` in `examples/real_time_minimal.py` performs one sweep.
  - Precompute right environments for the current configuration.
  - Sweep sites left-to-right, propose a single-spin flip, and accept with
    `min(1, |psi_new|^2 / |psi_old|^2)`.
  - Update the left environment immediately after acceptance so later sites reuse the cache.
  - Tensors are padded to a uniform bond dimension so the sweep can run inside a JAX scan.

- **PEPS**: `peps_sequential_sweep` in `examples/real_time_minimal.py` performs one sweep.
  - Build bottom environments once per sweep.
  - For each row, build right environments once and reuse a left environment while moving left-to-right.
  - Propose a single-spin flip at each site and accept with the same ratio.
  - Update the top boundary after finishing each row.

Both samplers generate a **chain**: accepted flips are kept and the sweep continues on the updated
configuration.

## Verify script

`scripts/verify_sequential_sampling.py` compares sequential sampling against FullSum and NetKet
MetropolisLocal.

Defaults follow your request (3x3 FullSum; 5x5/8x8/10x10 MH) and use large sample counts.

Examples:

```bash
./.venv/bin/python scripts/verify_sequential_sampling.py --model peps
```

```bash
./.venv/bin/python scripts/verify_sequential_sampling.py --model peps --nsamples 8192 --sweeps 20 --burn-in 400
```

```bash
./.venv/bin/python scripts/verify_sequential_sampling.py --model mps --fullsum-size 3 --mh-sizes 5 8 10
```

The script prints energy mean/std for sequential vs NetKet MH and reports the FullSum baseline
for the 3x3 case.

## Timing benchmark

You can also time sequential sampling against random single-spin flips:

```bash
./.venv/bin/python scripts/verify_sequential_sampling.py --model peps --benchmark --benchmark-size 4
```

Notes:
- `--sweeps` is the number of sequential full-lattice sweeps between samples.
- The benchmark compares against `random_flip_sample` using `random_sweeps = sweeps * n_sites`
  so the random sampler attempts the same number of single-spin proposals.

## Unit tests

Minimal distribution sanity checks live in `tests/test_sequential_sampling.py` and can be run with:

```bash
./.venv/bin/python -m unittest tests.test_sequential_sampling
```

These tests keep sizes small and use fixed RNG seeds to stay deterministic.
