# EXPERIENCE.md — Practitioner Knowledge for PEPS-tVMC

Operational wisdom accumulated from running simulations. Not systematic
rules (see CLAUDE.md for those), but the kind of knowledge that belongs
to a skilled practitioner (熟练工).

## Contraction Strategy

- **Always use Variational on GPU.** ZipUp involves SVD which is not well
  batched on GPU. Variational uses iterative sweeps that parallelize better.
- **ZipUp is fine on CPU** for small systems (L <= 6) where SVD cost is
  manageable.
- **Boundary dimension D' ~ 2D to 3D** is typical for Variational. Too small
  gives inaccurate contraction; too large wastes compute without improving
  accuracy.

## Bond Dimension

- **Z2 LGT converges well with D_k=2** for lattice sizes up to 32x32
  (Wu & Liu 2025, Table I).
- **For Z2 Higgs, D_k=2 is sufficient** for both deconfined and Higgs phases
  (Wu & Nys 2026).
- **Start with D_k=2, increase if energy variance is large.**
- **Standard PEPS (Heisenberg, TFIM):** D=3-4 for small lattices (L <= 8),
  D=6-8 for production (L >= 16). See Liu et al. 2021 for finite-size scaling.

## Solver Choice

- **Cholesky is default.** Faster and more parallelizable on GPU than SVD.
- **SVD is more robust** for ill-conditioned QGT. Use when Cholesky gives NaN.
- **CG (conjugate gradient)** for very large parameter counts where direct
  solve is too expensive.

## Solver Space (SR vs minSR)

- **For GIPEPS with large per-site parameters, use minSR** (`SampleSpace`).
  Avoids materializing the full Jacobian.
- **For standard PEPS, SR** (`ParameterSpace`) is fine — N_p is typically
  smaller than N_s.
- **Crossover:** when N_s > N_p - N_gv - 2, use minSR
  (Wu & Nys 2026, Sec. III.C).

## Sampling

- **n_samples=10240, n_chains=1024** is a good starting point for production
  on GPU.
- **For testing/debugging, use n_samples=64, n_chains=8.**
- **Sequential sampling** visits bonds in order along the lattice, reducing
  cost from O(N_site^2) to O(N_site) per sweep (Liu et al. 2021).

## Time Steps

- **Imaginary time:** dt=0.005 to 0.01 is typical for ground-state SR.
- **Real time:** dt=0.005 to 0.01 for RK4. Smaller dt gives better energy
  conservation but costs more steps.
- **For quenches with smooth ramps** (Schmitt protocol), dt=0.01 works well
  (Wu & Nys 2026, Fig. 5c).

## Convergence

- **FS_norm_squared should decrease** during imaginary-time optimization. If
  it plateaus, the state is near convergence or a local minimum.
- **Energy drift < 0.5%** over the full trajectory indicates stable real-time
  evolution.
- **TDVP residual 1e-9 to 1e-25** is normal and indicates the TDVP equation
  is being solved accurately (Wu & Nys 2026, Fig. 6).

## Gauge Removal

- **Always use gauge removal for real-time dynamics.** Without it, the QGT
  is ill-conditioned and the TDVP equation is unstable.
- **For imaginary-time (ground state), gauge removal is optional** but
  improves convergence speed.
- **minSR achieves gauge removal automatically** — the parameter index is
  contracted away, so gauge directions vanish.

## Diag Shift (Tikhonov Regularization)

- **1e-4 for ground state** is a safe starting point.
- **1e-6 to 1e-8 for real-time dynamics** — needs to be small to avoid
  biasing the time evolution, but large enough to regularize.
- **If solver gives NaN, increase diag_shift** or switch to SVD solver.
