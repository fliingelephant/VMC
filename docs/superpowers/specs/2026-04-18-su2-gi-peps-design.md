# SU(2) Gauge-Invariant PEPS — Design

Date: 2026-04-18
Status: Proposed
Scope: New module `src/vmc/peps/su2_gi/` for non-Abelian SU(2) lattice gauge theory with VMC. Pure gauge, ground-state + real-time, no touching of existing Abelian `src/vmc/peps/gi/`.

---

## 1. Goals and non-goals

### In scope

- **Pure SU(2) Yang–Mills** on an open-boundary square lattice with Kogut–Susskind Hamiltonian at truncation `j_max` (first target: `j_max = 1/2`, hardcore-gluon).
- **Block-sparse storage**: packed `(N_blocks, D, D, D, D)` per site, no padding, no mask. Every entry is a variational DoF.
- **Gauge canonical form (GCF)** for SU(2): link reduced tensors absorbed into vertex reduced tensors; link tensors are parameter-free identities on reduced indices.
- **MC sampling** over gauge-field configurations in the electric-basis irrep labeling, with plaquette-flip moves preserving Gauss's law.
- **Variational+QR boundary-MPS compression** with per-sector `vmap` — no SVD.
- **Ground state (SR / imaginary-time) and real-time dynamics (tVMC)** via the existing `TDVPDriver` + `Euler/RK4` + `SRPreconditioner`, unchanged.
- **Typed `GaugeGroup` protocol** with SU(2) as its first and only concrete instance. The protocol exists so that future additions (U(1), SU(3), eventual Abelian unification) are purely additive.

### Out of scope (deferred, additive later)

- Matter fields (fermionic/bosonic in fundamental irrep); static background charges `Q_x ≠ 0`.
- `j_max ≥ 1` with heterogeneous `D_j` per sector (current MVP uses uniform `D_j = D`).
- U(1), SU(3), abelian unification.
- Refactoring of existing `src/vmc/peps/gi/`.

### Non-goals

- A generic "non-Abelian framework first, specialize to SU(2) later" abstraction layer.
- A full symmetric-tensor library (TensorKit-style). The structured block-sparse layout gives us the needed algorithmic gain without a library dependency cliff.

---

## 2. Physics contract

### Hamiltonian (pure SU(2) YM, Kogut–Susskind)

Open-boundary square lattice `Λ = (n_rows, n_cols)`. Gauge links carry `|j, m_L, m_R⟩` with `j ≤ j_max`. Pure-gauge Hamiltonian:

```
Ĥ = g_E · Σ_links Ê²_link   −   g_B · Σ_plaq (Û□ + Û†□)
```

- Electric energy: `Ê² |j,m_L,m_R⟩ = j(j+1) |j,m_L,m_R⟩` (diagonal Casimir).
- Plaquette: `Û□ = tr(U_1 U_2 U_3† U_4†)` around a 2×2 plaquette; raises/lowers link irreps (sparse in the `(j,m_L,m_R)` basis with SU(2) CG coefficients as matrix elements).

### Gauss's law (pure gauge)

For each vertex `x`, the adjacent link irreps must fuse to a singlet:

```
j_left ⊗ j_up ⊗ j_right* ⊗ j_down*  ⊇  singlet.
```

Intertwiner multiplicity `ι` counts the number of distinct singlet fusions for a given 4-tuple. For `j_max = 1/2`:

| leg tuple (unordered)              | count | intertwiner mult. |
|------------------------------------|-------|-------------------|
| `(0,0,0,0)`                        | 1     | 1                 |
| `(½,½,0,0)` and 5 distinguishable permutations by position | 6     | 1 each            |
| `(½,½,½,½)`                        | 1     | 2                 |
| **total (bulk vertex, j_max=½)**   |       | **9 blocks**      |

Boundary vertices have fewer legs → fewer blocks. All lattice-position-dependent block tables are precomputed at model-build from the group + geometry.

### GCF (why tensors are parameter-economical)

Each link reduced tensor factorizes as `B_j^{a_l a_r} = δ_{a_l a_r}` after gauge fixing (Schur's lemma guarantees the gauge freedom is a per-irrep `D_j × D_j` matrix, absorbable into the adjacent `A`). Only vertex reduced tensors carry variational parameters; no link tensor ever appears explicitly in storage.

### Sampling basis

A sample `s` = irrep labels `{j_ℓ}` per edge. Magnetic multiplicity `(2j+1)` is summed analytically at evaluation time via CG factors folded into the MPO bricks (see §5.3). This is the standard symmetric-tensor "sample the charge, integrate the magnetic" convention.

### Proposal moves (detailed balance)

Plaquette-flip move: apply `Û□|s⟩`; the result is a superposition over several Gauss-law-compatible outcome configs. The proposal draws one outcome with CG-squared weight `|⟨s'|Û□|s⟩|²`. Metropolis accept using Hastings ratio `|Ψ(s')/Ψ(s)|² × (proposal ratio)`. Sequential (not random) sweeping over plaquettes, matching the Wu–Liu pattern in `_plaquette_sweep_row_pair`.

---

## 3. Module layout

```
src/vmc/peps/su2_gi/
  __init__.py
  group.py           # GaugeGroup protocol + SU2 impl + CG/6j tables
  block_table.py     # precompute allowed-block tables per lattice position
  model.py           # SU2GIPEPS nnx.Module, sample representation, init/sampling utilities
  block_ops.py       # block-aware einsums, QR, MPO application (Variational+QR, vmapped over block axis)
  contraction.py     # boundary-MPS build / apply / compress, re-using block_ops primitives
  kernels.py         # build_mc_kernels dispatch (init_cache, transition, estimate)
  local_terms.py     # PlaquetteSU2Term, LinkCasimirTerm
  compat.py          # flatten/unflatten sample, apply(), ...
```

**No shared code with `src/vmc/peps/gi/`.** Common primitives in `src/vmc/peps/common/` (e.g. `_apply_mpo_from_below`, `_compute_right_envs`) are reused where they operate at the "one site = one dense MPO brick" level — the block-sparse layer is resolved upstream in `block_ops.py` so that each site hands common a single dense brick.

**Kernel dispatch registration** follows the pattern in `src/vmc/peps/gi/kernels.py:43` — the driver registers the SU(2) kernel via `import vmc.peps.su2_gi.kernels  # noqa: F401` next to the existing GI registration in `src/vmc/drivers/tdvp.py:32`.

---

## 4. Data model

### 4.1 `GaugeGroup` protocol (`group.py`)

```python
class GaugeGroup(Protocol):
    name: str
    j_max: float | int
    def irreps(self) -> tuple[float, ...]: ...
    def dim(self, j: float) -> int: ...        # (2j+1) for SU(2); 1 for Abelian
    def dual(self, j: float) -> float: ...     # j for SU(2); −j for U(1)/Z_N
    def fuse(self, j1: float, j2: float) -> tuple[tuple[float, int], ...]:
        """(j_out, multiplicity) pairs."""
    def cg(self, j1: float, j2: float, j: float) -> jax.Array:
        """Clebsch–Gordan block of shape (dim j1, dim j2, dim j, mult)."""
    def casimir(self, j: float) -> float: ...  # j(j+1) for SU(2)
```

Concrete `SU2(j_max)` implements all six. CG coefficients are computed once at construction time via the standard recursion relation (Racah formula) and cached as a dict `{(j1,j2,j): jnp.array}`.

### 4.2 Block table (`block_table.py`)

For each lattice position `(r, c)` enumerate the allowed singlet-fusion tuples:

```python
allowed_blocks[(r, c)] = tuple of (
    (j_left, j_up, j_right, j_down),          # each ∈ group.irreps(); boundary legs fixed to j=0
    iota                                       # 0 ≤ iota < intertwiner_multiplicity(tuple)
)
```

Ordering is canonical (lexicographic on `(j_left, j_up, j_right, j_down, iota)`) so a block-id is a small integer. A static lookup table `block_id[(r, c)][j_l, j_u, j_r, j_d, iota] → int` is materialized as a `jnp.asarray` for `jnp.take` access during sampling.

### 4.3 `SU2GIPEPS` module (`model.py`)

```python
class SU2GIPEPSConfig(frozen dataclass):
    shape: tuple[int, int]
    group: GaugeGroup                # SU2(j_max)
    D: int                           # uniform reduced bond dim per sector
    Qx: tuple[tuple[int, ...], ...]  # target irrep per vertex; 0 (singlet) everywhere for MVP
    mps_sector_schedule: Mapping[float, int]  # {j: D_j^{MPS}} for boundary-MPS bonds
    dtype: Any = jnp.complex128

class SU2GIPEPS(nnx.Module):
    # Parameters
    tensors: list[list[nnx.Param]]   # tensors[r][c] shape (N_blocks[r,c], D, D, D, D)
    # Metadata (NOT nnx.Param)
    block_ids: jax.Array             # static block-id lookup tables
    cg_cache: dict                   # CG tensors per (j1,j2,j)
    mpo_bricks: jax.Array            # precomputed dense MPO per sample-irrep-tuple (see §5.3)
```

Shape rationale: each block has shape `(D, D, D, D)` because each virtual leg carries `D` reduced slots per irrep sector under the uniform-`D_j` convention. No phys dim (pure gauge). `N_blocks` is position-dependent: 9 for bulk, fewer on edges/corners.

Sample representation: `jnp.int32` tensors `h_links[(n_rows, n_cols-1)]` and `v_links[(n_rows-1, n_cols)]` storing irrep indices (not `j` values — indices into `group.irreps()`). Flattened via `SU2GIPEPS.flatten_sample(h_links, v_links)` → `jax.Array` for `Cache`/`Context` transport.

### 4.4 Parameter count and efficiency accounting

Bulk site, pure SU(2) `j_max=1/2`, uniform `D_j=D`:
- Block-sparse packed layout: `N_blocks × D^4 = 9 D^4` complex entries.
- Hypothetical full-dense unfolded leg of size `∑_j (2j+1) D_j = 3D`: `(3D)^4 = 81 D^4` complex entries — 9× wasted parameters, 81× wasted on `(3D)^4 / D^4` per block.

**Where the 9× win applies** (exact savings):

- **Variational parameter count**: 9× smaller (matches storage).
- **SR / QGT matrix dimension**: 9× smaller (gradient vector is 9× shorter per site because only the sample-selected block contributes a nonzero entry).
- **Gradient accumulation** in `estimate`: each sample writes into one of `N_blocks` slots; 9× less autodiff work than a full unfolded `(3D)^4` tensor.

**Where the 9× win does *not* apply** under the MVP path (§5.3):

- **Boundary-MPS contraction compute**: once a sample gathers its per-site block and tensors it with magnetic-index identities, each effective brick is dense `(3D)^4`. Boundary-MPS compression operates on these dense bricks with dense MPS bonds — no per-sector speedup. This is the MVP trade-off explicitly chosen in §5.3.

Net: MVP keeps the variational/SR parameter-count win (9×) at the cost of doing boundary-MPS contraction in the unfolded representation. A follow-up block-aware MPS (§5.3, deferred) would recover the compute win.

---

## 5. Execution pipeline

### 5.1 Sample → effective per-site MPO brick

Given a sample `s = {j_ℓ}`, a vertex tensor `A[r][c]` with shape `(N_blocks, D, D, D, D)` selects a single block via static lookup:

```python
tup = (s.j_left(r,c), s.j_up(r,c), s.j_right(r,c), s.j_down(r,c))
b_id = block_ids[r, c][tup]            # jnp.take
block = A[r, c][b_id]                  # shape (D, D, D, D)
```

**Magnetic multiplicity is folded in at this step**: the effective MPO brick that feeds `common/contraction.py` primitives is

```python
mpo_brick[r, c] = jnp.einsum(
    "lurd, L, U, R, D -> lLuUrRdD",    # schematic; L,U,R,D are magnetic indices of dim (2j+1)
    block, I_jl, I_ju, I_jr, I_jd,
)  # then reshape to (Dl*(2jl+1), Du*(2ju+1), Dr*(2jr+1), Dd*(2jd+1))
```

That is, the per-irrep block is tensored with the `(2j+1)`-dimensional identity on each leg's magnetic index. The reshape unfolds `(reduced × magnetic) → single leg dim`. For intertwiner multiplicity > 1 (the `(½)⊗⁴` case), we sum over `iota` with the intertwiner-space basis tensor (precomputed once from CG): `block = Σ_ι A[r,c][b_id,ι] · I^{(ι)}_{m_l m_u m_r m_d}`.

The resulting `mpo_brick[r, c]` has shape `(Dl_eff, Du_eff, Dr_eff, Dd_eff)` where `Dk_eff = D × (2j_k+1)`. From here onward, standard dense boundary-MPS primitives apply. **This is the critical efficiency choice**: we never build the full dense `A` with all irrep sectors folded in; we build only the brick for the sampled irreps.

### 5.2 Per-sample `vmap`

Everything above is `vmap`-friendly: `jnp.take(A[r, c], b_id(s), axis=0)` vectorizes over `s`, and the magnetic-identity einsums are static-shape in each sector. The cost of building `mpo_brick[r, c]` for one sample is `O(D^4 × (2j+1)^4)` times the small intertwiner sum — negligible next to the boundary-MPS contraction.

### 5.3 Boundary-MPS contraction (reuse `common/`)

Once each site has produced its dense MPO brick, the full pipeline in `src/vmc/peps/common/contraction.py:23` (`_build_row_mpo`), `:69` (`_compute_right_envs`), `:60` (`_apply_mpo_from_below`) lifts verbatim — they operate at the brick level and don't care that the brick came from a block-sparse reduction.

`Variational` strategy from `src/vmc/peps/common/strategy.py:174` (`_apply_mpo_variational`) is the compression strategy. Bond dim on each boundary-MPS bond is the sum `D_eff^{MPS} = ∑_j D_j^{MPS} (2j+1)` over the `mps_sector_schedule`. **All shapes are static at model-build.** QR-only, no SVD, no reallocation.

**Per-sector vs. merged MPS (explicit design choice).** Two options:

- **Option A (MVP — chosen).** Boundary-MPS in *unfolded* representation: single dense bond per site of width `D_eff^{MPS} = ∑_j (2j+1) D_j^{MPS}`, dense MPO bricks of shape `(D_eff)^4 = (3D)^4`. Reuses `common/strategy.py:Variational` without modification. **Saves storage/gradient by 9× (see §4.4), does NOT save boundary-MPS compute.**

- **Option B (deferred).** Block-aware boundary-MPS: each MPS bond carries irrep sectors with static dim `D_j^{MPS}`, MPO bricks stored as block-sparse, `Variational+QR` sweep vmapped over sector axis. Saves both storage *and* compute. Requires SU(2)-specific replacement of `_apply_mpo_variational` (cannot extend `common/strategy.py` without breaking backward-compat; the natural place is `su2_gi/contraction.py`).

The MVP commits to Option A for simplicity and maximum reuse; Option B is additive and semantically equivalent, slotted in when benchmarks demand it.

### 5.4 Plaquette operator (`local_terms.py`)

```python
@register
class PlaquetteSU2Term(TransitionOperator):
    row: int; col: int          # top-left corner
    # no data — CG structure lives in the group instance
```

Evaluation dispatches on this term (following the `_eval_term` pattern in `src/vmc/peps/gi/model.py:866`). At runtime:

1. Enumerate the finite set of outcome configurations `s → s'` with non-zero `⟨s'|Û□|s⟩`. For `j_max=1/2`, each plaquette flip can change each of the 4 border links between `j=0` and `j=1/2`; the outcome set is ≤ 16 configs (most with zero CG weight).
2. For each non-zero outcome, compute `⟨s'|Û□|s⟩` as a product of four CG coefficients (one per corner of the plaquette) times two `{6j}` factors (from re-coupling the internal fusion channels). Precomputed once at model-build and stored as `ampl_table[plaq_type, outcome_id]`.
3. Rebuild the four updated MPO bricks at `(r,c)`, `(r,c+1)`, `(r+1,c)`, `(r+1,c+1)` via §5.1 with the outcome irreps.
4. Evaluate via `_contract_2row_2col` (reused from `src/vmc/peps/common/contraction.py:90`), summed with the CG/6j weights.

Cost per plaquette term: `O(16 · D^4_eff · contraction_cost)` in the worst case; typically fewer outcomes are allowed.

### 5.5 Casimir / electric term (`LinkCasimirTerm`)

Purely diagonal in the sample: for each link ℓ with sampled irrep `j_ℓ`, add `g_E · j_ℓ (j_ℓ+1)`. Implemented as a `DiagonalOperator` subclass with `energy(h_links, v_links)` returning the sum. Zero contraction cost.

### 5.6 Kernels (`kernels.py`)

Mirrors `src/vmc/peps/gi/kernels.py` structure but without matter:

- **`init_cache`**: for each chain, build bottom envs by sweeping row-wise bottom→top. Uses `_build_row_mpo_su2` which delegates to §5.1. Boundary-MPS compression via `Variational` strategy (already QR-only).
- **`transition`**: plaquette-flip sweep over row pairs, mirroring `_plaquette_sweep_row_pair` in `src/vmc/peps/gi/model.py:1134`. For each plaquette, propose an outcome drawn from the CG-weighted distribution, Metropolis-accept on `|Ψ(s')/Ψ(s)|²`, update the row MPOs in place, maintain left envs.
- **`estimate`**: sweep rows top→bottom, compute diagonal energy (Casimir), transition-term energies (plaquette via §5.4), accumulate env-gradients `G = (1/Ψ) ∂Ψ/∂A_block[b_id]`. Gradient collection mirrors the existing GI pattern (`src/vmc/peps/gi/kernels.py:148-170`) but indexes into the `N_blocks` axis rather than the `Nc` axis.

### 5.7 Driver & integrator plumbing

**Zero changes to `TDVPDriver`**. The driver already supports `LocalHamiltonian`/`TimeDependentHamiltonian` operators built from transition + diagonal terms, the `SRPreconditioner`, `Euler`/`RK4` integrators, and `RealTimeUnit`/`ImaginaryTimeUnit` for real-vs-imaginary time. The only plumbing addition is the `noqa: F401` import of `vmc.peps.su2_gi.kernels` next to existing GI registration in `src/vmc/drivers/tdvp.py:32`.

**Ground state**: `TimeUnit=ImaginaryTimeUnit()`, `Integrator=Euler()` — plain SR step.
**Real-time**: `TimeUnit=RealTimeUnit()`, `Integrator=RK4()` — tVMC.

Both share the same sampling / `build_mc_kernels` / `SRPreconditioner` stack.

---

## 6. Efficiency contract (end-to-end)

**Invariants** — must hold throughout:

1. **No padding, no mask.** Every parameter in `A[r,c][b_id, :, :, :, :]` is a meaningful variational DoF.
2. **Static shapes.** Block count `N_blocks[r,c]`, MPS sector schedule, plaquette outcome counts are all compile-time constants derivable from `(group, shape, Qx, mps_sector_schedule)`.
3. **`vmap`-friendly.** Per-sample operations are gathers on static-shape axes + dense einsums. No dict-of-varying-shape pytrees.
4. **QR-only.** No SVD anywhere in hot paths. All compression goes through `_qr_compactwy`.

**Savings summary (MVP):**

| Axis | Savings vs. fully unfolded dense PEPS |
|------|----------------------------------------|
| Variational parameter count | 9× |
| SR / QGT matrix dimension | 9× |
| Gradient accumulation compute | 9× |
| Boundary-MPS contraction compute | 1× (Option A, MVP); 9×+ under Option B (deferred) |

**MVP simplifications (explicit):**

- **Uniform `D_j ≡ D`** across irrep sectors. Heterogeneous `D_j` would require pytree-of-varying-shape blocks and is deferred.
- **Unfolded boundary-MPS** (Option A in §5.3). Full block-aware MPS is Option B, deferred.
- **Target irrep `Q_x = 0` everywhere** (singlet). Nonzero static background charges are additive later.

---

## 7. Tests

Organize as `tests/test_su2_gi_*.py`.

### 7.1 Group primitives
- `test_su2_cg_unitarity`: `∑_{m1 m2} CG(j1,m1; j2,m2 | j,m) CG(j1,m1; j2,m2 | j',m') = δ_{jj'}δ_{mm'}` per fused `(j,j')`.
- `test_su2_fusion_counts`: for `j_max ∈ {1/2, 1}`, `fuse` matches hand-computed table.

### 7.2 Block tables
- `test_su2_bulk_block_count`: `j_max=1/2` bulk vertex gives 9 blocks including intertwiner multiplicities.
- `test_su2_boundary_block_count`: corner/edge vertex block counts match theory (boundary legs fixed to `j=0`).

### 7.3 End-to-end amplitude
- `test_su2_amplitude_gauge_invariance`: applying a vertex-local SU(2) rotation at `x` leaves `|Ψ(s)|²` unchanged (up to target irrep — singlet for pure-gauge). Enforced numerically on small lattices.
- `test_su2_amplitude_matches_exact_dense`: on a `2x2` lattice, a random `SU2GIPEPS` at `D=2` agrees with its fully-unfolded dense PEPS contraction.

### 7.4 Sampling
- `test_su2_plaquette_flip_preserves_gauss_law`: sequential plaquette sweeps never produce a Gauss-law-violating `s`.
- `test_su2_plaquette_detailed_balance`: on a tiny Hilbert space (e.g. `1x2`) Monte Carlo converges to `|Ψ|²` distribution against exact enumeration.

### 7.5 Ground state (imaginary time)
- `test_su2_pure_gauge_4x4_jmax_half_gs_vs_ed`: `4×4` lattice at `j_max=1/2`, optimize with SR (imaginary-time Euler) to 2k steps, compare to ED (Hilbert space is tractable at this size under `j_max=1/2`). Match within sampling error < `1e-3`.
- `test_su2_pure_gauge_convergence_in_D`: same lattice, sweep `D ∈ {2, 4}`, show energy decreases monotonically.

### 7.6 Real-time (tVMC)
- `test_su2_pure_gauge_energy_conservation`: RK4 real-time evolution of a glueball-like initial state; total energy conserved to `<0.5%` over `T=5` at `dt=0.005` on `4×4` lattice (mirroring the Abelian vison test in the Wu–Liu SM).

All tests run under `JAX_PLATFORM_NAME=cpu`, `pytest -m "not slow"`.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| **CG/6j correctness**: sign and normalization conventions for SU(2) vary between references | Cross-check CG via `test_su2_cg_unitarity` and against `sympy.physics.quantum.cg`; pin the Condon–Shortley convention in `group.py`. |
| **Intertwiner basis ambiguity** for `(½)^⊗4` (two singlets): choice of basis affects what the "intertwiner index" means | Fix orthonormal basis by diagonalizing in the tree-decomposition channel convention (s-channel $j_m=0$ vs $j_m=1$). Document in `group.py`. |
| **Plaquette outcome set grows** with `j_max` | For `j_max=1/2` capped at 16 outcomes per plaquette → manageable. Re-evaluate when lifting to `j_max ≥ 1`. |
| **Sample space ergodicity**: pure-plaquette moves may not reach all Gauss-law-compatible configs on open boundaries | Verified analytically for open BC + singlet `Q_x=0`: plaquettes generate the full Gauss-law-compatible space on a simply-connected lattice. |
| **Uniform `D_j` suboptimal at large `j_max`** | Known. Heterogeneous `D_j` is explicitly deferred to a follow-up design (expected: adopt sector-sliced tensors as a pytree of dense arrays, lose vmap-naive but recoverable via padded fused representation). |
| **GCF not formally justified** for non-Abelian in the Wu–Liu derivation | Include a self-contained derivation in `docs/su2_gi_gcf.md` referencing Schur's lemma at the reduced-index level. |

---

## 9. Benchmarks

`4×4` pure SU(2) YM at `j_max=1/2`, varying `g_E/g_B`, `D ∈ {2, 4}`, 1024 samples, 2000 SR steps:

| Target | Value | Reference |
|--------|-------|-----------|
| Ground-state energy at `g_E=g_B=1` | match ED within sampling error | ED of the hardcore-gluon model at `4×4` |
| `D=2 → D=4` energy decrease | monotone | — |
| Plaquette expectation `⟨U□⟩` | match ED | — |
| Real-time energy drift, `T=5, dt=0.005` | < `0.5%` | Wu–Liu SM Fig. S3 analog |

---

## 10. Implementation ordering

Follow-on implementation plan will cover this in detail. At a high level:

1. `group.py` (SU2 + CG + 6j) — testable in isolation.
2. `block_table.py` — testable against hand-computed counts.
3. `model.py` skeleton (`SU2GIPEPS`, sample flatten/unflatten, `random_physical_configuration`).
4. `block_ops.py` (§5.1 brick assembly, `vmap`-ed).
5. `contraction.py` using `common/` primitives.
6. `local_terms.py` (`LinkCasimirTerm`, `PlaquetteSU2Term`).
7. `kernels.py` (`init_cache`, `transition`, `estimate`).
8. Driver plumbing (one-liner `noqa: F401` import).
9. Tests §7.1–7.6 in order.
10. Benchmark §9.
