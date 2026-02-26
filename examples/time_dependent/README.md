# Time-Dependent 3x3 Exact Benchmark

This folder benchmarks time-dependent PEPS TDVP against exact Schrödinger
evolution on the full `2^9 = 512` dimensional Hilbert space (3x3 qubits).

Hamiltonian:

`H(t) = jzz * sum_<ij> sigmaz_i sigmaz_j - hx(t) * sum_i sigmax_i`,

with `hx(t) = hx0 + hx_slope * t`.

The script compares:

1. Exact trajectory from full-state propagation.
2. PEPS/VMC trajectory from `TDVPDriver` with `TimeDependentHamiltonian`.

Reported metrics per time point:

- `mz_exact`, `mz_vmc`, and `diff_mz`.
- State fidelity `|<psi_exact | psi_vmc>|^2`.

## Run

```bash
.venv/bin/python examples/time_dependent/exact_tdvp_3x3_check.py
```

Optional args:

```bash
.venv/bin/python examples/time_dependent/exact_tdvp_3x3_check.py \
  --steps 20 --dt 0.02 \
  --n-samples 4096 --n-chains 64 --bond-dim 1 \
  --csv examples/time_dependent/trajectory.csv
```

Force a PEPS-only constant schedule (for mismatch stress test):

```bash
.venv/bin/python examples/time_dependent/exact_tdvp_3x3_check.py \
  --hx-slope -0.15 --peps-hx-slope 0.0 --bond-dim 2
```

## Notes

- Exact evolution uses midpoint piecewise-constant propagation with
  `--exact-substeps` micro-steps per TDVP step (`expm` per micro-step).
- Initial state is built from a single random PEPS realization; exact evolution
  starts from the wavefunction reconstructed from those PEPS amplitudes.
- TDVP is variational and Monte Carlo based, so perfect agreement is not expected.
