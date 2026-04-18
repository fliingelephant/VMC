# SU(2) Gauge-Invariant PEPS — Design

Date: 2026-04-18
Status: Proposed (revised to adopt block-aware boundary-MPS throughout)
Scope: New module `src/vmc/peps/su2_gi/` for non-Abelian SU(2) lattice gauge theory with VMC. Pure gauge, ground-state + real-time, no touching of existing Abelian `src/vmc/peps/gi/`.

---

## 1. Goals and non-goals

### In scope

- **Pure SU(2) Yang–Mills** on an open-boundary square lattice with Kogut–Susskind Hamiltonian at truncation `j_max` (first target: `j_max = 1/2`, hardcore-gluon).
- **Block-sparse storage** for both vertex tensors *and* boundary-MPS bonds: every object stored as a collection of dense blocks keyed by irrep-sector tuples. No padding, no mask.
- **Gauge canonical form (GCF)** for SU(2): link reduced tensors are parameter-free identities on reduced indices (bookkeeping only, not stored).
- **MC sampling** over gauge-field configurations in the electric-basis irrep labeling, with plaquette-flip Metropolis moves preserving Gauss's law.
- **Block-aware Variational+QR boundary-MPS compression** — per-sector QR vmapped over a static sector axis. No SVD.
- **Ground state (SR / imaginary-time) and real-time dynamics (tVMC)** via the existing `TDVPDriver` + `Euler/RK4` + `SRPreconditioner`, unchanged.
- **Typed `GaugeGroup` protocol** with SU(2) as its first and only concrete instance.

### Out of scope (deferred, additive later)

- Matter fields (bosonic or fermionic in fundamental irrep); static background charges `Q_x ≠ 0`.
- `j_max ≥ 1` with heterogeneous `D_j` per sector (MVP uses uniform `D_j = D`, uniform `D_j^{MPS} = χ`).
- U(1), SU(3), abelian unification.
- Refactoring of existing `src/vmc/peps/gi/`.

### Non-goals

- A generic "non-Abelian framework first, specialize to SU(2) later" abstraction layer.
- A full symmetric-tensor library (TensorKit-style). Structured block-sparse with static sector axes gives the algorithmic gain without a library dependency cliff.

---

## 2. Physics contract

### Hamiltonian (pure SU(2) YM, Kogut–Susskind)

Open-boundary square lattice `Λ = (n_rows, n_cols)`. Gauge links carry `|j, m_L, m_R⟩` with `j ≤ j_max`. Pure-gauge Hamiltonian:

```
Ĥ = g_E · Σ_links Ê²_link   −   g_B · Σ_plaq (Û□ + Û†□)
```

- Electric energy: `Ê² |j,m_L,m_R⟩ = j(j+1) |j,m_L,m_R⟩` (diagonal Casimir).
- Plaquette: `Û□ = tr(U_1 U_2 U_3† U_4†)` around a 2×2 plaquette; raises/lowers link irreps with SU(2) CG coefficients as matrix elements. Hermitian combination `Û□+Û†□` is used in the Hamiltonian.

### Gauss's law (pure gauge)

For each vertex `x`, the adjacent link irreps must fuse to a singlet (for MVP; extendable to target irrep `Q_x` later):

```
j_left ⊗ j_up ⊗ j_right* ⊗ j_down*  ⊇  singlet.
```

Intertwiner multiplicity `ι` counts the number of distinct singlet fusions for a given 4-tuple. For `j_max = 1/2`, bulk vertex:

| leg tuple (position-ordered)                      | count | intertwiner mult. |
|---------------------------------------------------|-------|-------------------|
| `(0,0,0,0)`                                       | 1     | 1                 |
| `(½,½,0,0)` and 5 other 2-out-of-4 leg positions  | 6     | 1 each            |
| `(½,½,½,½)`                                       | 1     | 2                 |
| **total**                                         |       | **9 blocks**      |

Boundary vertices: legs pointing outside the lattice are fixed to `j=0`; block count reduces accordingly (corner = 1, edge = 3 for `j_max=1/2`).

### GCF (why tensors are parameter-economical)

Reduced link tensor factorizes as `B^j_{a_l a_r} = δ_{a_l a_r}` after gauge fixing. Derivation: Schur's lemma forces the bond-gauge freedom $X$ to be `⊕_j X^j` with `X^j ∈ GL(D_j)`; choosing `X^j = (𝓑^j)^{-1/2}` absorbs the full variational content of `B` into the adjacent `A`'s. **Post-GCF, link "tensors" are pure bookkeeping** — they are never stored or indexed; the fact that neighboring `A`'s share the same virtual irrep `j` on a common bond suffices.

### Sampling basis

A sample `s` = integer-labeled irreps `{j_ℓ}` per edge (indices into `group.irreps()`). Magnetic indices `m` are *not* sampled — they are summed analytically via CG/intertwiner structure folded into the MPO bricks (see §5.1). This is the standard symmetric-tensor "sample the charge, integrate the magnetic" convention.

### Proposal moves (detailed balance)

Plaquette-flip move: propose `s → s'` where `s'` differs from `s` on the 4 border links of a plaquette, with probability proportional to `|⟨s'|Û□+Û†□|s⟩|²`. Since `Û□+Û†□` is Hermitian, `|⟨s'|·|s⟩| = |⟨s|·|s'⟩|` → proposal is **symmetric**, so Metropolis accept uses `min(1, |Ψ(s')/Ψ(s)|²)` with no Hastings correction. Sequential (not random) sweeping over plaquettes, mirroring `_plaquette_sweep_row_pair` in `src/vmc/peps/gi/model.py:1134`.

**Ergodicity.** For open BC with target singlet `Q_x=0`, plaquette operators generate the full Gauss-law-compatible subspace on a simply-connected lattice (the non-Abelian analog of the Abelian pure-plaquette-flip ergodicity used by Wu–Liu). Sketch: any Gauss-law-satisfying link-irrep config can be reached from `all-j=0` by a sequence of plaquette raisings (since SU(2) flux loops tile the lattice dually to plaquettes on simply-connected open BC).

---

## 3. Module layout

```
src/vmc/peps/su2_gi/
  __init__.py
  group.py           # GaugeGroup protocol + SU2 impl + CG/6j tables
  block_table.py     # precompute allowed-block tables per lattice position
  intertwiner.py     # intertwiner basis I^{(ι)} in s-channel tree decomposition
  model.py           # SU2GIPEPS nnx.Module, sample rep, init / random_physical_configuration
  block_ops.py       # sector-aware block gather, brick assembly, per-sector vmapped QR
  contraction.py     # block-aware boundary-MPS: env build, _apply_mpo_variational_su2
  kernels.py         # build_mc_kernels dispatch (init_cache, transition, estimate)
  local_terms.py     # PlaquetteSU2Term, LinkCasimirTerm
  compat.py          # flatten/unflatten sample, apply(), ...
```

**No shared code with `src/vmc/peps/gi/`**; no refactor of existing code.

### Reuse precisely from `src/vmc/peps/common/`

The following **lift verbatim** because they operate at the "one brick per site" level and are oblivious to how the brick was built:

- `_apply_mpo_from_below` (`contraction.py:60`) — when passed block-structured tensors as pytree leaves per sector.
- `_compute_right_envs` (`contraction.py:69`) — same.
- `_contract_bottom`, `_contract_2row_2col`, `_contract_2row_1col` — pure einsum, work on block leaves.

The following **need SU(2)-specific replacements** because they assume a physical-index slice into a dense site tensor:

- `_build_row_mpo` (`contraction.py:23`) → `_build_row_mpo_su2` in `block_ops.py`. Inputs the sample irrep indices, outputs per-site brick of block structure.
- `_apply_mpo_variational` (`strategy.py:174`) → `_apply_mpo_variational_su2` in `contraction.py`. Sector-aware Variational+QR.
- `_estimate_sweep` (from `common/energy.py`) → SU(2) variant; indexing by sample differs because there is no matter physical index.

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
        """((j_out, multiplicity), ...)."""
    def cg(self, j1: float, j2: float, j: float) -> jax.Array:
        """Clebsch–Gordan block, shape (dim(j1), dim(j2), dim(j), mult(j1,j2→j))."""
    def casimir(self, j: float) -> float: ...  # j(j+1) for SU(2)
```

Concrete `SU2(j_max)` uses **Condon–Shortley convention** throughout. CG coefficients computed once via Racah's formula at construction, cached as a dict `{(j1,j2,j): jnp.asarray}`. All returned arrays are `jnp.complex128` for uniform dtype handling (imaginary parts vanish in Condon–Shortley but this avoids dtype juggling).

**Protocol adequacy across groups.** The signature handles:
- **U(1) truncated**: `dim=1`, `fuse` returns singleton, `cg` is scalar `1.0`. Trivially Abelian.
- **$\mathbb Z_N$**: identical to U(1) but with mod-$N$ arithmetic in `fuse`/`dual`.
- **SU(3)**: `fuse` returns multiplicity-weighted Littlewood–Richardson decomposition; `cg` carries the `mult` axis. The protocol is adequate but **the contraction layer currently assumes SU(2)-style multiplicity-at-vertex only**; extending to SU(3) requires handling multiplicity at every 2-way fusion (deferred).

### 4.2 Intertwiner basis (`intertwiner.py`)

For each 4-leg irrep tuple `(j_l, j_u, j_r, j_d)` with target singlet, fix the intertwiner basis in the **s-channel tree decomposition**:

```
I^{(j_m)}_{m_l m_u m_r m_d} = Σ_{m_m} CG(j_l, j_u → j_m)[m_l, m_u, m_m]
                              · CG(j_r^*, j_d^* → j_m)[m_r, m_d, m_m]·(−1)^{normalization}
```

where `j_m` runs over the intermediate irreps of `(j_l ⊗ j_u) ∩ (j_r* ⊗ j_d*)*`. For `(½)^⊗⁴`, `j_m ∈ {0, 1}` gives the two intertwiners; other tuples have ≤ 1 intertwiner. This basis is orthonormal under $\sum_m I^{(\iota)} I^{(\iota')*} = δ_{\iota \iota'}$ and is used consistently by `block_table`, `block_ops`, and `local_terms`.

### 4.3 Block table (`block_table.py`)

For each lattice position `(r, c)` enumerate the allowed singlet-fusion tuples:

```python
allowed_blocks[(r, c)] = tuple of (
    (j_left, j_up, j_right, j_down),          # each ∈ group.irreps(); boundary legs fixed to j=0
    iota                                       # 0 ≤ iota < intertwiner_multiplicity(tuple)
)
```

Ordering is canonical (lexicographic on `(j_left, j_up, j_right, j_down, iota)`) so the block-id is a small integer. A static lookup table `block_id[r, c][j_l, j_u, j_r, j_d, iota] → int` is materialized as a `jnp.asarray` for `jnp.take` access during sampling. Invalid (non-singlet) combinations receive sentinel `-1` — sampling never hits them by construction.

### 4.4 `SU2GIPEPS` module (`model.py`)

```python
@dataclass(frozen=True)
class SU2GIPEPSConfig:
    shape: tuple[int, int]
    group: GaugeGroup                # SU2(j_max)
    D: int                           # uniform reduced bond dim per sector on PEPS virtual legs
    chi: int                         # uniform reduced bond dim per sector on boundary-MPS bonds
    Qx: tuple[tuple[int, ...], ...]  # target irrep per vertex; 0 (singlet) everywhere for MVP
    dtype: Any = jnp.complex128

class SU2GIPEPS(nnx.Module):
    # Variational parameters
    tensors: list[list[nnx.Param]]   # tensors[r][c] shape (N_blocks[r,c], D, D, D, D)
    # Static metadata (NOT nnx.Param, baked into graphdef)
    block_ids:   jax.Array           # static block-id lookup tables per (r, c)
    intertwiners: dict               # I^{(ι)} per 4-tuple (see §4.2)
    cg_cache:    dict                # CG tensors per (j1, j2, j)
    plaquette_table: Any             # precomputed plaquette matrix elements (see §5.4)
```

Each `A[r][c]` block has shape `(D, D, D, D)`; all blocks share the same shape because of uniform `D_j = D` (MVP simplification). `N_blocks` is position-dependent and baked into model-build.

Sample representation: `jnp.int32` tensors `h_links[(n_rows, n_cols-1)]` and `v_links[(n_rows-1, n_cols)]` storing irrep indices into `group.irreps()`. Flattened via `SU2GIPEPS.flatten_sample(h_links, v_links)` → `jax.Array` for `Cache`/`Context` transport.

**Dtype policy**: `complex128` throughout. SU(2) has real irreps for integer `j` and pseudoreal for half-integer; while ground-state pure-gauge can in principle be stored real, real-time evolution requires complex, and the extra memory cost is tolerable. One dtype throughout avoids juggling.

**Parameter initialization**: random Gaussian, scaled `1/√(D⁴)`; singlet block `(0,0,0,0)` initialized to a small constant plus noise so the initial wavefunction overlaps the `all-j=0` vacuum.

### 4.5 Boundary-MPS tensor layout

Each boundary-MPS site tensor at row `r` lives on bonds of block structure. The natural layout:

```python
# Boundary-MPS at row r has one tensor per column.
# Each tensor is a dict of dense blocks keyed by leg-irrep tuples,
# but *materialized as a stacked static array* for vmap:
bmps[r, c] : shape (N_bond_blocks[r,c], chi, d_phys_block, chi)
```

where the leading axis indexes allowed `(j_l, j_p, j_r)` tuples for which the 3-leg fusion-to-singlet condition holds (for MPS tensors generated from PEPS contractions, the physical-index irrep `j_p` is determined by the row-MPO's column structure). `d_phys_block` is the fused dim on the physical axis from contracting one PEPS row against the boundary. All three leg-widths are uniform `χ` or `D` respectively under the MVP uniform-per-sector convention.

### 4.6 Parameter count and efficiency accounting

Bulk site, pure SU(2) `j_max=1/2`, uniform `D_j=D`, uniform `D_j^{MPS}=χ`:

- Vertex tensor storage per site: `9 D⁴` complex entries.
- Boundary-MPS tensor storage per position: `N_bond_blocks · χ² · d_phys_block`.
- Hypothetical full-dense unfolded (no symmetry awareness): `(3D)⁴ = 81 D⁴` per vertex, `(3χ)² · (3D)²` per MPS site.

**Savings under the block-aware pipeline (everywhere):**

| Axis | Savings |
|------|---------|
| Vertex tensor storage | 9× |
| SR / QGT matrix dimension | 9× |
| Gradient accumulation compute | 9× |
| Boundary-MPS tensor storage | $\sim (∑(2j+1))^2 / \sum(2j+1) = 3×$ |
| Boundary-MPS compression compute (per-sector QR) | $(3D)^4/(D^4 \cdot N_{blocks}) \cdot $(sector overhead)$ \sim 9× $ |
| MPO-MPS contraction compute | similar 9× |

The savings come from (i) never instantiating disallowed fusion blocks, (ii) per-sector QR on the static sector axis (small matrices, well-batched under `vmap`), (iii) sector-diagonal bond structure throughout compression.

---

## 5. Execution pipeline

### 5.1 Sample → per-site MPO brick (block-structured)

Given a sample `s = {j_ℓ}`, each vertex tensor `A[r][c]` selects a single block via static lookup:

```python
tup = (s.j_left(r, c), s.j_up(r, c), s.j_right(r, c), s.j_down(r, c))
b_id = block_ids[r, c][tup]                      # jnp.take, static shape
reduced_block = A[r, c][b_id]                    # shape (D, D, D, D), for ι=0
# If N_iota(tup) > 1, A[r,c][b_id] packs all ι's in an extra leading axis (see §4.3 ordering).
```

**The MPO brick carries full magnetic structure.** Contracting the intertwiner in magnetic indices:

$$\text{brick}_{(a_l m_l)\,(a_u m_u)\,(a_r m_r)\,(a_d m_d)} \;=\; \sum_{\iota} A^{[\text{bid},\iota]}_{a_l a_u a_r a_d} \cdot I^{(\iota)}_{m_l m_u m_r m_d}$$

where `I^{(ι)}` is the intertwiner basis tensor from §4.2 (precomputed at model build). The brick is **sector-structured**: its legs carry both a reduced index `a` (dim `D`) and a magnetic index `m` (dim `2j+1`). Under the block-aware representation, we keep these as a *single static axis per sector* rather than fusing dense → the brick at position `(r,c)` for sampled tuple `tup` is a single dense tensor of shape `(D·(2j_l+1), D·(2j_u+1), D·(2j_r+1), D·(2j_d+1))` carrying the labels `(j_l, j_u, j_r, j_d)` as static metadata for downstream sector matching.

### 5.2 Block-aware boundary-MPS

Each boundary-MPS tensor is a collection of dense blocks (pytree leaves, or stacked along a static block axis) indexed by its three legs' irrep labels. Contractions, env building, and compression all walk the static block axis.

**Row-MPO application `_apply_mpo_from_below`:** for each output MPS site, the left/physical/right irrep triples that produce nonzero results come from the fusion rules `j_l^{in} ⊗ j_p^{MPO} → j_l^{out}`. All such triples are enumerated once at model build; at runtime, per-sector einsums produce per-sector output blocks. No disallowed fusion is ever computed.

**Variational+QR compression (`_apply_mpo_variational_su2`):** mirrors the algorithm in `common/strategy.py:_apply_mpo_variational` (left-to-right QR initialization + iterative sweeps) but with every tensor replaced by its block-structured version:

1. **Initialization sweep.** For each site left → right: contract `θ = L · M · W · R` per sector; stack per-sector `θ` blocks into `(N_out_sectors, Dl·dp, Dr)` and `vmap(_qr_compactwy)` over the sector axis → get `Q, R` per sector. Truncate each sector to its allocated `χ_j^{MPS}` from the static schedule.
2. **Iterative sweeps.** Environments `L̃, R̃` are themselves block-structured. Each sweep step's optimal tensor computation and QR is `vmap`ed over sectors.

Because per-sector block shapes are *static* (uniform `χ` and `D`), `vmap(_qr_compactwy)` batches all sector QRs into a single GPU call on a `(N_sectors, ..., ...)` array.

**Static sector schedule.** Every boundary-MPS bond has a predetermined `{j: χ_j}` at model-build. `χ_j` defaults to the user-specified uniform `χ` (MVP). The schedule never changes during sweeps — no reallocation, no dynamic shapes, no `jit` recompile.

### 5.3 Per-sample `vmap`

All operations above are `vmap`-friendly: block-id gather per sample uses `jnp.take` on a static axis; intertwiner contraction is a static einsum per sector; boundary-MPS compression `vmap`s over the sector axis inside each sweep and can be outer-`vmap`ed over samples.

### 5.4 Plaquette operator (`local_terms.py`)

```python
@register
class PlaquetteSU2Term(TransitionOperator):
    row: int; col: int          # top-left corner
```

Evaluation dispatches on this term (following the `_eval_term` pattern in `src/vmc/peps/gi/model.py:866`). At runtime:

1. **Precomputed outcome table** (at model build). For each allowed plaquette input-irrep tuple `(j_1^{in}, j_2^{in}, j_3^{in}, j_4^{in})` — the 4 links around the plaquette — enumerate the finite set of non-zero output tuples `(j_1^{out}, j_2^{out}, j_3^{out}, j_4^{out})` with matrix element

   $$\langle s'| \hat U_\square + \hat U_\square^\dagger |s\rangle \;=\; \prod_{\text{corners}} C_{\text{corner}} \cdot \{6j\}_{\text{recoupling}}$$

   where each $C_{\text{corner}}$ is a CG-like factor at the vertex contracted with the SU(2) generators of the two border links meeting at that corner, and the $\{6j\}$ factor arises from re-coupling the intertwiner basis when the link irreps shift. For `j_max=1/2`, each plaquette has ≤ 16 outcome configurations, many with zero CG weight. All amplitudes tabulated once and stored as `plaquette_table[input_tuple] → list[(output_tuple, amplitude)]`.

2. **Runtime evaluation.** Given sample `s` at the plaquette, look up the outcome list. For each outcome:
   - Rebuild the four updated MPO bricks via §5.1 with the outcome irreps.
   - Evaluate the 2×2 window via `_contract_2row_2col` (reused from `common/contraction.py:90`) using the *updated* bricks against the *current* 2-row envs.
   - Weight by the precomputed amplitude and sum.

Cost per plaquette term: `O(N_outcomes · D^4 · (2j+1)^4 · contraction_cost)` in the worst case; for `j_max=1/2` bounded and amortized.

### 5.5 Casimir / electric term (`LinkCasimirTerm`)

Purely diagonal in the sample: for each link `ℓ` with sampled irrep `j_ℓ`, add `g_E · j_ℓ (j_ℓ+1)`. Implemented as a `DiagonalOperator` subclass with `energy(h_links, v_links)` returning the sum. Zero contraction cost.

### 5.6 Kernels (`kernels.py`)

Mirrors `src/vmc/peps/gi/kernels.py` structure with three differences: (i) no matter index, (ii) block-aware MPO bricks, (iii) gradient indexed by block-id.

- **`init_cache`**: for each chain, build bottom envs by sweeping row-wise bottom→top. Uses `_build_row_mpo_su2` (block_ops.py) delegating to §5.1. Compression via `_apply_mpo_variational_su2` (contraction.py).
- **`transition`**: plaquette-flip sequential sweep over row pairs, mirroring `_plaquette_sweep_row_pair` in `src/vmc/peps/gi/model.py:1134`. For each plaquette, propose an outcome drawn uniformly from the non-zero-amplitude outcome list of `Û□+Û†□` (symmetric proposal → no Hastings correction), Metropolis-accept on `|Ψ(s')/Ψ(s)|²`, update the 4 bricks in place, maintain left envs.
- **`estimate`**: sweep rows top→bottom, compute diagonal energy (Casimir via `LinkCasimirTerm.energy`), transition-term energies (plaquette via §5.4), accumulate env-gradients `G = (1/Ψ) · ∂Ψ/∂A[r,c][b_id]`. Gradient collection follows `src/vmc/peps/gi/kernels.py:148-170` but indexes into the `N_blocks` axis rather than the `Nc` axis.

### 5.7 Driver & integrator plumbing

**Zero changes to `TDVPDriver`**. The only plumbing addition is the `noqa: F401` import of `vmc.peps.su2_gi.kernels` next to the existing GI registration in `src/vmc/drivers/tdvp.py:32`.

**Ground state**: `TimeUnit=ImaginaryTimeUnit()`, `Integrator=Euler()`.
**Real-time**: `TimeUnit=RealTimeUnit()`, `Integrator=RK4()`.

Both share the same sampling / `build_mc_kernels` / `SRPreconditioner` stack.

---

## 6. Efficiency contract (end-to-end)

**Hard invariants — must hold throughout:**

1. **No padding, no mask.** Every entry in every stored block is a meaningful DoF (or a mandatory magnetic-multiplet slot).
2. **Static shapes.** Block counts, sector schedules, outcome lists — all compile-time constants derived from `(group, shape, Qx, D, χ)`.
3. **`vmap`-friendly end to end.** Outer `vmap` over samples, inner `vmap` over sector axis during QR/env ops. No dict-of-varying-shape pytrees in hot paths.
4. **QR-only.** No SVD. All compression via `_qr_compactwy` vmapped over sectors.

**MVP simplifications (explicit):**

- **Uniform `D_j ≡ D`** (and `χ_j ≡ χ`) across irrep sectors. Heterogeneous `D_j` is a separate optimization (requires pytree-of-varying-shape blocks — defers nicely, won't break existing code when added).
- **Target irrep `Q_x = 0` everywhere** (singlet). Nonzero static background charges are a per-vertex config knob, trivially additive.

---

## 7. Extensibility notes

Explicitly designed into the module boundaries:

- **Adding matter fields** (bosonic in fundamental irrep): extend `SU2GIPEPSConfig` with a `matter_irreps: tuple` field; each vertex tensor gains a 5th physical leg with magnetic structure fixed by the matter irrep(s). Block tables extended to require $\bigotimes_\ell j_\ell \otimes p \to Q_x$. No structural change to contraction / sampling / GCF.
- **Adding fermionic matter** (e.g. staggered Kogut–Susskind): orthogonal axis. Requires swap-gate/parity-charge machinery on top of the block-sparse layer (Wu–Dai 2025 style). The block-sparse tensor layer doesn't change; the vertex tensor gains a $\mathbb Z_2$ parity label on each virtual leg. Can be slotted in as a mixin or a sibling module.
- **Adding background charges `Q_x ≠ 0`**: pure config knob in `SU2GIPEPSConfig.Qx`. Block tables recompute per-vertex; everything else unchanged.
- **Adding U(1), $\mathbb Z_N$**: implement the `GaugeGroup` protocol. All irreps 1-dim, CGs trivially $\delta$, intertwiner multiplicity always 1 — the SU(2) machinery degenerates cleanly.
- **Adding SU(3)**: requires extending the contraction layer's multiplicity handling: SU(3) has multiplicity > 1 at 2-way fusion (e.g. `8⊗8`), so every per-sector contraction picks up a fusion-multiplicity axis. The protocol signature already carries this axis; the runtime does not yet consume it beyond the vertex level.
- **Larger `j_max` with heterogeneous `D_j`**: requires moving from stacked-array block storage to pytree-of-dense-blocks-per-sector. Outer `vmap` over samples still works (gather the right pytree leaves), but per-sector `vmap` becomes per-sector `scan` over the block list (still static, just different compile shape).

---

## 8. Tests

Organize as `tests/test_su2_gi_*.py`.

### 8.1 Group primitives
- `test_su2_cg_unitarity`: `∑_{m1 m2} CG(j1 m1; j2 m2 | j m) CG*(j1 m1; j2 m2 | j' m') = δ_{jj'} δ_{mm'}`.
- `test_su2_cg_condon_shortley`: spot-check against `sympy.physics.quantum.cg` for `j1, j2 ∈ {½, 1}`.
- `test_su2_fusion_counts`: `fuse` matches hand table for `j_max ∈ {½, 1}`.
- `test_su2_casimir`: `casimir(j) == j*(j+1)`.

### 8.2 Intertwiners and block tables
- `test_su2_intertwiner_orthonormal`: `∑_m I^{(ι)} I^{(ι')*} = δ_{ιι'}` per 4-tuple.
- `test_su2_bulk_block_count`: `j_max=½` bulk vertex gives exactly 9 blocks including intertwiner multiplicity.
- `test_su2_boundary_block_count`: corner/edge block counts match theory.

### 8.3 End-to-end amplitude
- `test_su2_amplitude_gauge_invariance`: vertex-local SU(2) rotation leaves `|Ψ(s)|²` unchanged on a `2×2` lattice.
- `test_su2_amplitude_matches_exact_dense`: random `SU2GIPEPS` at `D=2` on `2×2` lattice agrees with its fully-unfolded-dense contraction within `1e-10`.
- `test_su2_block_aware_matches_unfolded`: the block-aware `_apply_mpo_variational_su2` produces the same MPS (up to per-sector gauge) as a reference unfolded implementation on `3×3`.

### 8.4 Sampling
- `test_su2_plaquette_flip_preserves_gauss_law`: sequential plaquette sweeps never produce a Gauss-law-violating `s`.
- `test_su2_plaquette_proposal_symmetric`: empirically verify symmetric proposal (`q(s→s') = q(s'→s)`).
- `test_su2_plaquette_detailed_balance`: on a tiny Hilbert space (e.g. `1×2`) Monte Carlo distribution matches `|Ψ|²` under exact enumeration.
- `test_su2_plaquette_ergodicity`: from `all-j=0`, sequential plaquette sweeps visit every Gauss-law-compatible config on a `2×2` lattice (brute force).

### 8.5 Ground state (imaginary time)
- `test_su2_pure_gauge_4x4_jmax_half_gs_vs_ed`: `4×4` at `j_max=½`, SR + imaginary-time Euler, 2000 steps, compare to ED (Hilbert space tractable at this size). Match within sampling error < `1e-3`.
- `test_su2_pure_gauge_convergence_in_D`: same lattice, sweep `D ∈ {2, 4}`, energy decreases monotonically.

### 8.6 Real-time (tVMC)
- `test_su2_pure_gauge_energy_conservation`: RK4 real-time evolution of a glueball-like initial state on `4×4`, `j_max=½`, `T=5`, `dt=0.005`; total energy conserved to `<0.5%` (analog of the Abelian vison test in Wu–Liu SM Fig. S3).

All tests run under `JAX_PLATFORM_NAME=cpu`, `pytest -m "not slow"`.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| **CG/{6j} convention errors** (sign / normalization vary between references) | Pin Condon–Shortley in `group.py`; cross-check against `sympy.physics.quantum.cg`; unit tests §8.1. |
| **Intertwiner basis choice** affects what `ι` means in stored blocks | Fix s-channel tree decomposition (§4.2). Document prominently in `intertwiner.py`. |
| **Plaquette outcome set grows** with `j_max` | Bounded at `j_max=½`; re-evaluate when lifting to `j_max ≥ 1`. |
| **Sample-space ergodicity** of plaquette-only moves | Verified for open BC + singlet `Q_x=0`; test §8.4. For PBC (not MVP), Polyakov-loop-charged sectors would need additional moves. |
| **Uniform `D_j` suboptimal at large `j_max`** | Known; deferred. Architecturally: heterogeneous `D_j` lives in `block_ops.py` as a pytree-of-blocks with per-sector `scan`. |
| **GCF derivation** for non-Abelian not in Wu–Liu paper proper | Self-contained derivation in §2 GCF paragraph + doc in `group.py`: reduces to Schur's lemma on the reduced index. |
| **`_apply_mpo_variational_su2` correctness** vs reference unfolded | Tested via §8.3 `test_su2_block_aware_matches_unfolded`. |
| **Per-sector QR batching overhead** if block sizes are small | For `j_max=½`, `D≥2` gives blocks `≥ 16` — small but well within GPU batched QR efficiency. Profile during benchmark. |

---

## 10. Benchmarks

`4×4` pure SU(2) YM at `j_max=½`, varying `g_E/g_B`, `D ∈ {2, 4}`, `χ = 2D`, 1024 samples, 2000 SR steps:

| Target | Value | Reference |
|--------|-------|-----------|
| Ground-state energy at `g_E=g_B=1` | match ED within sampling error | ED of hardcore-gluon `4×4` |
| `D=2 → D=4` energy decrease | monotone | — |
| Plaquette expectation `⟨U□⟩` | match ED | — |
| Real-time energy drift, `T=5, dt=0.005` | < `0.5%` | Wu–Liu SM Fig. S3 analog |
| Per-sweep wall time (one sample, `D=4, χ=8`) | < ~5× the equivalent Abelian GI-PEPS `Z₂` with same `D` | Existing GI `3×3` pure-gauge benchmark |

---

## 11. Implementation ordering

Follow-on implementation plan will cover this in detail. High-level:

1. **`group.py`** (SU2 + CG) — testable in isolation (§8.1).
2. **`intertwiner.py`** (s-channel basis) — testable (§8.2).
3. **`block_table.py`** — testable against hand counts (§8.2).
4. **`model.py` skeleton** (`SU2GIPEPS`, sample flatten/unflatten, `random_physical_configuration`).
5. **`block_ops.py`** (§5.1 brick assembly, per-sector vmapped QR primitives).
6. **`contraction.py`** (`_build_row_mpo_su2`, `_apply_mpo_variational_su2`, env ops).
7. **`local_terms.py`** (`LinkCasimirTerm`, `PlaquetteSU2Term`, outcome table).
8. **`kernels.py`** (`init_cache`, `transition`, `estimate`).
9. **Driver plumbing** (one-line `noqa: F401` import).
10. **Tests §8.1–8.6** in order.
11. **Benchmark §10**.
