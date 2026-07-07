# SU(2) Gauge-Invariant PEPS — Design

Date: 2026-04-18
Status: Proposed (revised: generic bosonic matter via `phys_dim ≥ 1`; block-aware boundary-MPS; (A)-style Metropolis–Hastings proposal; explicit intertwiner-consumption recipe)
Scope: New module `src/vmc/peps/su2_gi/` for non-Abelian SU(2) lattice gauge theory with VMC. Handles pure gauge (`phys_dim = 1`) and generic bosonic matter (`phys_dim ≥ 2`) with arbitrary per-state SU(2) charges, matching the `phys_dim`/`charge_of_site` pattern of the existing Abelian `src/vmc/peps/gi/`. Ground-state + real-time. No touching of existing Abelian `src/vmc/peps/gi/`.

---

## 1. Goals and non-goals

### In scope

- **SU(2) Yang–Mills** on an open-boundary square lattice with Kogut–Susskind Hamiltonian at truncation `j_max` (first validation target: `j_max = 1/2`, hardcore-gluon).
- **Generic bosonic matter** via a per-site physical axis `p ∈ {0, …, phys_dim−1}` with configurable SU(2) charge `q_p = charge_of_site[p]` per state. `phys_dim = 1` reproduces pure gauge. `phys_dim ≥ 2` adds matter with no additional structural changes — this mirrors the existing `GIPEPSConfig`/`GIPEPS` pattern.
- **Static background charges** `Q_x` per vertex (scalar singlet or nontrivial target irrep), settable as a config knob.
- **Block-sparse storage** for both vertex tensors *and* boundary-MPS bonds: every object stored as a collection of dense blocks keyed by irrep-sector tuples. No padding, no mask.
- **Gauge canonical form (GCF)** for SU(2): link reduced tensors are parameter-free identities on reduced indices (bookkeeping only, not stored).
- **MC sampling** over matter + gauge-field configurations in the electric-basis irrep labeling, with plaquette-flip + (when `phys_dim ≥ 2`) matter-hopping + link-flip Metropolis–Hastings moves preserving Gauss's law.
- **Block-aware Variational+QR boundary-MPS compression** — per-sector QR vmapped over a static sector axis. No SVD.
- **Ground state (SR / imaginary-time) and real-time dynamics (tVMC)** via the existing `TDVPDriver` + `Euler/RK4` + `SRPreconditioner`, unchanged.
- **Typed `GaugeGroup` protocol** with SU(2) as its first and only concrete instance.

### Out of scope (deferred, additive later)

- **Fermionic matter** (e.g. staggered Kogut–Susskind). Requires swap-gate / Grassmann-PEPS parity machinery (Wu–Dai 2025 style). The block-sparse layer doesn't change; adds a sibling module on top.
- **Heterogeneous `D_j` per sector** (MVP uses uniform `D_j = D`, uniform `D_j^{MPS} = χ`). Needed for efficient `j_max ≥ 1`; defers by swapping stacked-array block storage for pytree-of-blocks.
- U(1), SU(3), abelian unification.
- Refactoring of existing `src/vmc/peps/gi/`.

### Non-goals

- A generic "non-Abelian framework first, specialize to SU(2) later" abstraction layer.
- A full symmetric-tensor library (TensorKit-style). Structured block-sparse with static sector axes gives the algorithmic gain without a library dependency cliff.

---

## 2. Physics contract

### Hamiltonian (SU(2) Kogut–Susskind)

Open-boundary square lattice `Λ = (n_rows, n_cols)`. Gauge links carry `|j, m_L, m_R⟩` with `j ≤ j_max`. Pure-gauge Hamiltonian:

```
Ĥ_gauge = g_E · Σ_links Ê²_link   −   g_B · Σ_plaq (Û□ + Û†□)
```

When `phys_dim ≥ 2`, matter terms (on-site mass `m · M̂_x`, gauge-covariant hopping `w · ψ̂†_x U_⟨x,y⟩ ψ̂_y + h.c.`) are appended. The matter terms are bosonic; fermionic parity is out of scope. Concrete local-term classes are added in `local_terms.py` only when `phys_dim ≥ 2`.

- Electric energy: `Ê² |j,m_L,m_R⟩ = j(j+1) |j,m_L,m_R⟩` (diagonal Casimir).
- Plaquette: `Û□ = tr(U_1 U_2 U_3† U_4†)` around a 2×2 plaquette; fundamental-rep parallel transporters raise/lower link irreps with SU(2) CG coefficients as matrix elements. Hermitian combination `Û□+Û†□` is used in the Hamiltonian.
- **Truncation at `j_max`** is the standard projection `Ĥ_trunc = P_{j_max}·Ĥ·P_{j_max}` (a Hilbert-space truncation, no bond-dim approximation). Any outcome of `Û□`, `Û□†`, or matter hopping that would push a link past `j_max` is mapped to zero by `P_{j_max}`; Hermiticity survives because `P` is Hermitian. For `j_max=½`, each link admits exactly one allowed transition (`0 ↔ ½`), so per-plaquette and per-hop outcome sets are finite and enumerable at model build.

### Gauss's law

For each vertex `x`, the 4 adjacent link irreps, the matter charge `q_p` carried by the sampled physical state (only when `phys_dim ≥ 2`), and the static background charge `Q_x` must fuse consistently:

```
j_left ⊗ j_up ⊗ j_right* ⊗ j_down* ⊗ j_{q_p}  ⊇  Q_x.
```

For pure gauge (`phys_dim = 1`, `q_p ≡ 0`, `Q_x ≡ 0`), this reduces to the standard "fuse to singlet" constraint.

Intertwiner multiplicity `ι` counts the number of distinct fusions for a given leg tuple. **For `j_max = 1/2` bulk vertex, pure gauge / `Q_x = 0`:**

| leg tuple (position-ordered)                      | count | intertwiner mult. |
|---------------------------------------------------|-------|-------------------|
| `(0,0,0,0)`                                       | 1     | 1                 |
| `(½,½,0,0)` and 5 other 2-out-of-4 leg positions  | 6     | 1 each            |
| `(½,½,½,½)`                                       | 1     | 2                 |
| **total**                                         |       | **9 blocks**      |

Boundary vertices: legs pointing outside the lattice are fixed to `j=0`; block count reduces accordingly. For `j_max=1/2`: corner has 2 active-leg pairs `{(0,0), (½,½)}` → **2 blocks**; edge has 4 active-leg triples `{(0,0,0), (½,½,0), (½,0,½), (0,½,½)}` → **4 blocks**. Matter (`phys_dim ≥ 2`) or nonzero `Q_x` add additional blocks per vertex; the block table (§4.3) enumerates them uniformly.

### GCF (why tensors are parameter-economical)

Reduced link tensor factorizes as `B^j_{a_l a_r} = δ_{a_l a_r}` after gauge fixing. Derivation: Schur's lemma forces the bond-gauge freedom $X$ to be `⊕_j X^j` with `X^j ∈ GL(D_j)`; choosing `X^j = (𝓑^j)^{-1/2}` absorbs the full variational content of `B` into the adjacent `A`'s. **Post-GCF, link "tensors" are pure bookkeeping** — they are never stored or indexed; the fact that neighboring `A`'s share the same virtual irrep `j` on a common bond suffices.

### Sampling basis and variational state

A sample `s` consists of:
- `{j_ℓ}` — integer-labeled irreps per edge (indices into `group.irreps()`),
- `{p_x}` — matter physical states per vertex (only when `phys_dim ≥ 2`).

Magnetic indices `m` are *not* sampled — they are summed analytically via CG/intertwiner structure folded into the MPO bricks (see §5.1). Intertwiner labels `ι` are *not* sampled — they are consumed at brick assembly via a weighted sum over blocks (see §5.1–§5.2).

**Variational state.** Sampling `{j_ℓ}` without `ι` defines a variational state in the **intertwiner-symmetric subspace** of the gauge-invariant Hilbert space. Define the orthonormal basis `|{j}⟩_sym = (1/√N_ι({j})) Σ_ι |{j}, {ι}⟩`, where `N_ι = ∏_x n_x(tup_x)` is the total intertwiner multiplicity. The correctly normalized amplitude is `Ψ_sym({j}) = ⟨{j}_sym|Ψ⟩ = (1/√N_ι) Σ_ι Ψ({j}, {ι})`. This normalization is achieved by absorbing a `1/√n_x` factor into each vertex's consumption weight (see §5.2), so the brick contraction directly produces `Ψ_sym`. The VMC samples `{j}` with `|Ψ_sym|²` and computes `E_loc` using intertwiner-symmetrized Hamiltonian matrix elements `⟨j_sym|H|j'_sym⟩` (see §5.4). This yields a correct variational upper bound `⟨Φ|H|Φ⟩/⟨Φ|Φ⟩ ≥ E_0` for the state `|Φ⟩ = Σ_j Ψ_sym(j) |j⟩_sym`. All intertwiner blocks contribute through the consumption weights and remain independently optimizable via SR/TDVP.

### Proposal moves (detailed balance)

**Plaquette-flip move (scheme A, amplitude-weighted):** at plaquette `p` with input irreps `tup_in` on the 4 border links, look up the finite non-zero outcome list `outcomes(tup_in) = {(tup_out, A_sym(tup_out))}` of `Û□ + Û†□` (§5.4). The amplitude `A_sym` is the **intertwiner-symmetrized** matrix element — summed over all corner-intertwiner pairs (see §5.4 for the formula). Propose `s'` by sampling outcome `tup_out` with probability `q(s→s') = |A_sym(tup_out)|² / Z(s)` where `Z(s) = Σ_{tup'} |A_sym(tup')|²`. By Hermiticity, `|⟨s'|Û|s⟩|² = |⟨s|Û|s'⟩|²`, but the normalizations differ: `q(s→s') / q(s'→s) = Z(s') / Z(s)`. **A Hastings correction is required.** Accept with

```
min(1, |Ψ(s') / Ψ(s)|² · Z(s) / Z(s')).
```

Wire via the existing `_metropolis_hastings_accept(key, p_cur, p_prop, proposal_ratio=Z(s)/Z(s'))` in `src/vmc/utils/utils.py` — same plumbing used in `src/vmc/peps/gi/model.py:1247-1250` for matter link sweeps. `Z(s)` is a scalar lookup on a per-input-tuple static table built at model build.

**Matter-hopping and link sweeps (only when `phys_dim ≥ 2`):** sequential sweeps over horizontal hopping, vertical hopping, and (for non-number-conserving matter) link-flip moves, mirroring the structure of `src/vmc/peps/gi/model.py:1593-1776` (`transition`). Each move uses `_metropolis_hastings_accept` with the appropriate `proposal_ratio` derived from charge-degeneracy counts and/or amplitude normalizations — the Abelian GI code already implements this pattern.

Sequential (not random) sweeping over plaquettes, mirroring `_plaquette_sweep_row_pair` in `src/vmc/peps/gi/model.py:1134`.

**Ergodicity.** For open BC with pure gauge and target singlet `Q_x=0`, plaquette operators generate the gauge-invariant algebra on the singlet subspace (Kogut–Susskind 1975 §IV), so sequential plaquette sweeps + Hermiticity give ergodicity on simply-connected OBC lattices. With `phys_dim ≥ 2`, matter-hopping + link-flip moves close the remaining matter-charge sectors (same logic as `src/vmc/peps/gi/`).

---

## 3. Module layout

Extends the existing `src/vmc/peps/gi/` footprint. The Abelian gi/ has 5 files (`__init__.py`, `compat.py`, `kernels.py`, `local_terms.py`, `model.py`) and reuses `common/contraction.py` directly. SU(2) needs **sector-aware** contraction, so it adds a dedicated `contraction.py` plus a `group.py` for CG / {6j} / intertwiner machinery:

```
src/vmc/peps/su2_gi/
  __init__.py
  group.py           # SU2 + CG + {6j} + intertwiner basis + per-position block tables
                     # + static intertwiner-consumption weights (§5.2)
  model.py           # SU2GIPEPS nnx.Module: tensors, sample rep, flatten/unflatten, apply
  contraction.py     # SU(2)-specific MPO brick assembly, boundary-MPS Variational+QR, env ops
  local_terms.py     # PlaquetteSU2Term, LinkCasimirTerm, (matter terms when phys_dim ≥ 2),
                     # precomputed amplitude tables
  kernels.py         # build_mc_kernels dispatch (init_cache, transition, estimate)
```

**No shared code with `src/vmc/peps/gi/`**; no refactor of existing code.

### Reuse precisely from `src/vmc/peps/common/`

Sector-structured bricks require sector-aware contractions, so the `common/` primitives cannot be lifted verbatim. The SU(2) `contraction.py` re-implements the three operations its callers need, following the existing shape of the common primitives but walking a static sector axis:

- `_build_row_mpo_su2` — SU(2) analog of `common/contraction.py:_build_row_mpo`; assembles the reduced-only per-sector brick from the sample, including the matter leg when `phys_dim ≥ 2`.
- `_apply_mpo_variational_su2` — SU(2) analog of `common/strategy.py:_apply_mpo_variational`; per-sector QR `vmap`ed over a static sector axis. No SVD.
- `_estimate_sweep_su2` — SU(2) analog of `common/energy.py:_estimate_sweep`; indexing differs because bricks carry sector blocks rather than a single dense tensor.
- `_contract_bottom_su2` — block-sparse version: left-to-right contraction over bmps sites, summing over compatible bond-sector sequences. Each step is a per-sector einsum; the final result is a scalar (amplitude).
- `_contract_2row_2col_su2`, `_contract_2row_1col_su2` — block-sparse versions: 2-row environments carry sector structure on bmps bonds. For each compatible sector combination, the einsum has the same structure as the common/ version but operates on per-sector blocks. Sum over compatible sectors yields the scalar window amplitude.
- `_compute_right_envs_su2`, `_update_left_env_2row_su2` — block-sparse env update: at each column, the env tensor has sector indices on bmps-bond legs (fixed sector on MPO legs from the sample). Each column step is a per-sector einsum vmapped over compatible sector pairs.

**No dense fallback.** All contraction primitives in `contraction.py` maintain block-sparsity. The common/ primitives are NOT reused; the SU(2) versions re-implement each operation with explicit sector-axis walking.

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

### 4.2 Intertwiner basis (in `group.py`)

For each leg tuple at vertex `x` — `(j_l, j_u, j_r, j_d)` for pure gauge, extended by `j_{q_p}` when `phys_dim ≥ 2` — with target `Q_x`, fix the intertwiner basis by **recursive binary fusion along a canonical s-channel tree**. Each binary fusion contributes one CG and one internal irrep label; the intertwiner label is the tuple of internal labels `ι = (j_e)_{e ∈ internal edges}`. Normalize by `∏_e 1/√(2 j_e + 1)` so the magnetic-index inner product is a pure Kronecker delta.

Canonical tree:
- `(j_l, j_u) → j_m` via `CG(j_l, j_u; j_m)`.
- `(j_r^*, j_d^*) → j_n` via `CG(j_r^*, j_d^*; j_n)`.
- With matter: `(j_m, j_n) → j_k` then `(j_k, j_{q_p}) → Q_x` via two binary CGs, giving `ι = (j_m, j_n, j_k)`.
- Pure gauge: `(j_m, j_n) → Q_x` directly, giving `ι = (j_m, j_n)` (with `j_m = j_n` enforced when `Q_x = 0`, so `ι` effectively reduces to `j_m`).

Orthonormality `Σ_{all m} I^{(ι)} I^{(ι')*} = δ_{ιι'}` follows from CG unitarity + the `1/√(2 j_e + 1)` per-internal-edge normalization. `group.py` implements the recursion generically for any `(j_max, phys_dim, charge_of_site, Q_x)` — no per-target branch. Pin Condon–Shortley phase in `group.py` docstrings; verify orthonormality against sympy-derived CGs in §8.2 `test_su2_intertwiner_orthonormal`.

### 4.3 Block table (in `group.py`)

For each lattice position `(r, c)` and (when `phys_dim ≥ 2`) matter state `p`, `block_table(r, c, p)` enumerates the allowed `((j_l, j_u, j_r, j_d), ι)` pairs — each 4-tuple whose fusion with `j_{q_p}` contains the target `Q_x[r, c]`, and each intertwiner `ι` within that tuple. Ordering is canonical (lexicographic on `(j_l, j_u, j_r, j_d, ι)`), giving every pair a small integer `block_id`. Boundary legs are fixed to `j = 0`. For pure gauge (`phys_dim = 1`), the `p` axis is trivial; for `phys_dim ≥ 2` the table is position × physical-state indexed.

At sample time, the brick assembly (§5.1) needs the `≤ ι_max` `block_id`s that share a given `(sector_tuple, p)`. This is one `jnp.take` on a static lookup built from the enumeration; see §5.1 for the exact usage. No sampling path ever indexes an invalid sector.

### 4.4 `SU2GIPEPS` module (`model.py`)

```python
@dataclass(frozen=True)
class SU2GIPEPSConfig:
    shape: tuple[int, int]
    group: GaugeGroup                        # SU2(j_max)
    D: int                                   # uniform reduced bond dim per sector on PEPS virtual legs
    chi: int                                 # uniform reduced bond dim per sector on boundary-MPS bonds
    # Matter (phys_dim == 1 ⇒ pure gauge; phys_dim ≥ 2 ⇒ bosonic matter)
    phys_dim: int = 1
    charge_of_site: tuple[float, ...] = (0.0,)  # SU(2) irrep j_p for each physical state p
    # Per-vertex target irrep; scalar 0 or shape-matching tuple of tuples of floats
    Qx: Any = 0.0
    dtype: Any = jnp.complex128

class SU2GIPEPS(nnx.Module):
    # Variational parameters — one `nnx.Param` per site, flat shape.
    tensors: list[list[nnx.Param]]
        # tensors[r][c] shape (total_blocks[r, c], D, D, D, D), where
        # total_blocks[r, c] = Σ_p N_blocks[r, c, p] enumerates all
        # Gauss-compatible (p, sector_tuple, ι) triples at vertex (r, c)
        # in a fixed canonical order. No pad, no mask, no ragged pytree.
        # Pure gauge (phys_dim == 1): total_blocks[r, c] == N_blocks[r, c].
    # Static metadata (NOT nnx.Param, baked into graphdef via dataclass fields)
    tables: SU2GITables
        # group.py-built bundle: block_ids[r, c][p, tup, :ι_max] -> flat
        # indices into tensors[r][c] (with out-of-range sentinels for
        # missing intertwiners, consumed by jnp.take mode='fill');
        # intertwiner-consumption weights w[r, c](tup, p, ι) (§5.2);
        # plaquette outcomes + amplitudes (§5.4); matter-hopping outcomes
        # when phys_dim ≥ 2; shared-bond scalars.
```

Every block has shape `(D, D, D, D)` on virtual legs under uniform `D_j = D`. `total_blocks[r, c]` is position-dependent and baked into model-build.

**Build-time cost.** `SU2GITables` is constructed once in `SU2GIPEPS.__init__` by running the group-theoretic recursion (§4.2) + CG/{6j}/{9j} enumeration; all downstream tables (block ids, weights `w`, plaquette amplitudes, shared-bond scalars) follow deterministically. The bundle is frozen thereafter — no recompute per sample, per sweep, or per `jit` trace. Cost is `O(ms)` at `j_max ≤ 1`; negligible versus the VMC loop.

**Sample representation:** `jnp.int32` tensors storing irrep / state indices:
- `h_links[(n_rows, n_cols-1)]` — horizontal-link irreps (indices into `group.irreps()`).
- `v_links[(n_rows-1, n_cols)]` — vertical-link irreps.
- `sites[(n_rows, n_cols)]` — matter physical states (present only when `phys_dim ≥ 2`; absent/trivial for pure gauge, exactly like `src/vmc/peps/gi/` handles `phys_dim = 1`).

Flatten via `SU2GIPEPS.flatten_sample(sites, h_links, v_links)` (drop `sites` when `phys_dim = 1`) → `jax.Array` for `Cache`/`Context` transport.

**Why link-per-edge (not pack-per-site as in Abelian GI-PEPS).** The existing `src/vmc/peps/gi/` packs link irreps into a per-site `nc = N^{num_links-1}` index because Gauss's law lets you drop one redundant link per vertex. That is cheap for Abelian because one arithmetic constraint suffices. For SU(2), Gauss's law at a vertex is a *non-Abelian fusion rule with intertwiner multiplicity* — the "independent DoF" count per vertex is not a single scalar and would fight the block-sparse brick convention. Storing one irrep per edge gives a clean global sample; plaquette and hopping moves update only the affected edges, and intertwiner multiplicity lives entirely inside the vertex tensor block structure.

**Dtype policy**: `complex128` throughout. Real-time evolution needs complex; pure-gauge ground-state could use real but the simplification of a single dtype wins. Revisit if profiling shows dense CPU/GPU QR is the bottleneck.

**Parameter initialization**: random Gaussian, scaled `1/√(D⁴)`; the trivial block (all legs at `j=0`, `p=0` when present) initialized to a small constant plus noise so the initial wavefunction overlaps the electric vacuum. Callers that want excited initial states (e.g. for real-time string-dynamics protocols) override this with a domain-specific initializer.

### 4.5 Boundary-MPS tensor layout

Each boundary-MPS tensor stores **reduced-space blocks only**; magnetic structure enters through the intertwiner-derived amplitudes of §5.2. Stacked for `vmap`:

```python
# One tensor per column of the boundary-MPS.
# Leading axis enumerates allowed 3-leg irrep tuples (j_l, j_p, j_r)
# whose fusion contains a singlet.
bmps[r, c] : shape (N_bond_blocks[r, c], chi, D, chi)
```

Every block has the same reduced shape `(χ, D, χ)` — dim `D` on the physical leg (reduced-index slice of the PEPS vertical bond, not `D · (2j_p + 1)`). The outer sector label `(j_l, j_p, j_r)` is static metadata; magnetic multiplicity `(2j_l+1)(2j_p+1)(2j_r+1)` enters only as scalar factors at contraction time.

### 4.6 Parameter count and efficiency accounting

Bulk site, SU(2) `j_max=1/2`, pure gauge (`phys_dim=1`, `Q_x=0`), uniform `D_j=D`, uniform `D_j^{MPS}=χ`; link leg carries $\dim(\tfrac12) + \dim(0) = 2 + 1 = 3$ irrep labels → naive leg dim `3D`.

- **Vertex tensor storage per site** (block-aware, reduced-only, pure gauge): `9 · D⁴`.
- **Boundary-MPS tensor storage per bulk position**: 4 allowed `(j_l, j_p, j_r)` singlet-fusion triples → `4 · χ² · D`.
- **Dense-unfolded reference** (no symmetry awareness): `(3D)⁴ = 81 D⁴` per vertex; `(3χ)² · 3D = 27 χ² D` per MPS site.

Matter (`phys_dim ≥ 2`) multiplies vertex storage by roughly `phys_dim` (more precisely, by `Σ_p N_blocks[r, c, p] / N_blocks[r, c, 0]`); boundary-MPS storage is unchanged because intertwiners are summed at brick assembly (§5.2) and the bmps never carries a matter axis.

**Savings under the reduced-only Option B pipeline:**

| Axis | Block-aware | Dense-unfolded | Ratio |
|------|-------------|----------------|-------|
| Vertex tensor storage (per site) | 9 D⁴ | 81 D⁴ | **9×** |
| SR / QGT matrix dimension | 9 n_sites D⁴ | 81 n_sites D⁴ | **9×** (81× on `S` storage) |
| Boundary-MPS storage (per site, `j_max=½`) | 4 χ² D | 27 χ² D | **≈6.75×** |
| Boundary-MPS per-sweep compute | `(N_sec) · D^4 · …` | `(3D)^4 · …` | **≈9×** |
| MPO-MPS contraction compute | `(N_sec) · D^4` per site | `(3D)^4` per site | **≈9×** |

The savings come from (i) never instantiating disallowed fusion blocks, (ii) per-sector QR on the static sector axis (small matrices, well-batched under `vmap`), (iii) sector-diagonal bond structure throughout compression, (iv) magnetic-index contractions pre-evaluated once into scalar coupling coefficients (§5.2) and reused.

---

## 5. Execution pipeline

### 5.1 Sample → per-site MPO brick

Sampling produces the sector tuple `tup = (j_l, j_u, j_r, j_d)` per vertex plus (when `phys_dim ≥ 2`) the matter state `p = sites[r, c]`. The intertwiner label `ι` is **not sampled**; it is **consumed at brick assembly** by a static sample-dependent weighted sum (see §5.2).

Storage: `A[r, c]` has shape `(total_blocks[r, c], D, D, D, D)` — one `(D, D, D, D)` block per allowed `(p, sector, ι)` triple, no padding.

**Runtime brick assembly** (per vertex, per sample):

1. Gather the ≤ `ι_max` blocks sharing `(sector_tuple, p)` via a single static lookup:
   ```
   flat_ids = block_ids[r, c][p, tup]                        # (ι_max,)
   raw      = jnp.take(A[r, c], flat_ids, mode='fill')       # (ι_max, D, D, D, D)
   ```
   `block_ids` holds out-of-range sentinel indices for `(p, tup)` entries with fewer than `ι_max` intertwiners; `mode='fill'` returns zeros there. No branch, no dynamic shape.

2. Multiply by the **intertwiner-consumption weight** `w[r, c](tup, p, ι)` from the static table (§5.2) and reduce:
   ```
   brick[r, c] = Σ_ι  w[r, c](tup, p, ι) · raw[ι]            # (D, D, D, D)
   ```

The sector label `tup` is carried alongside the brick as static metadata. Gradient scatter in `estimate` reuses the same `flat_ids` — structurally parallel to the Abelian `site * nc + cfg_idx` packing in `src/vmc/peps/gi/kernels.py:148-170`, but the flat index is precomputed in the static table rather than composed arithmetically.

### 5.2 Boundary-MPS contraction

Every runtime object in the pipeline — PEPS brick, boundary-MPS site, left/right environments — stores **reduced-space blocks only** (dims `D` or `χ` per virtual leg). SU(2) covariance enters through two classes of **precomputed static scalar tables**, built once in `group.py` from CG + `{6j}` and indexed by sector tuples.

**(i) Intertwiner-consumption weight `w[r, c](tup, p, ι)` (§5.1).** This scalar is the result of integrating the vertex's 4-leg (or 5-leg with matter) magnetic-index sum against the shared-bond magnetic-index contributions from the neighboring boundary-MPS, *excluding* the reduced-index contraction. Concretely,

```
w[r, c](tup, p, ι) = [1/√n_x(tup, p)] · [∏_ℓ attached to (r,c)  d_{j_ℓ}^{½}] · f_x(tup, p, ι),
```

where `n_x(tup, p)` is the intertwiner multiplicity at vertex `(r, c)` for leg tuple `tup` and matter state `p`, `d_j = 2j + 1`, and `f_x(tup, p, ι)` is the vertex-local spin-network value in the orthonormal recursive-binary-fusion basis of §4.2. The `1/√n_x` factor ensures the brick contraction `Σ_ι w · A(ι)` directly produces the correctly normalized amplitude `Ψ_sym({j}) = ⟨{j}_sym|Ψ⟩` — no extra normalization at contraction time. `f_x` is a product of `{6j}` (pure gauge) and `{9j}` (matter / nontrivial `Q_x`) symbols assembled by contracting CGs along the canonical s-channel tree. `group.py` computes all three factors generically for any `(j_max, phys_dim, charge_of_site, Q_x)` at model-build time; no per-target hard-coded table.

**(ii) Shared-bond dimension factor.** Each PEPS edge contributes exactly one `d_j` factor to the integrated spin-network value, partitioned as `d_j^{½}` absorbed into each of its two adjacent vertex weights (the `∏_ℓ d_{j_ℓ}^{½}` in §5.2(i)). PEPS edges internal to a row pair are fully accounted for by the two `w`-factors at their endpoints; **no additional edge-scalar** is inserted at contraction time for these. The one exception is edges crossing the boundary between the row-MPO and the boundary-MPS (the shared physical leg `j_p` of the bmps): here only one endpoint is a vertex (in the PEPS bulk); the other endpoint is the bmps site. The bmps side contributes the missing `d_{j_p}^{½}` factor, once per bmps site per row-MPO application. All other bmps virtual bonds are internal to the compressed environment and carry no extra scalar (the environment is pure block-reduced linear algebra).

**Row-MPO application (`_apply_mpo_from_below`).** At column `c`, the row-MPO brick has legs `(j_MPO,L, j_MPO,R, j_MPO,U, j_MPO,D)`; the bmps site below has legs `(j_bmps,L, j_p, j_bmps,R)`. The MPO-`D` leg contracts with the bmps-`j_p` leg (shared). The new bmps left bond is the fusion `j_bmps,L ⊗ j_MPO,L → j_out,L` (per-sector einsum yielding one output block per allowed fusion), similarly on the right; the new bmps physical leg is `j_MPO,U`. Enumerate allowed fusion pairs at model build; at runtime, per-sector einsums weighted by the bmps-side `d_{j_p}^{½}` scalar produce the output blocks. Disallowed fusions never instantiate.

**Variational+QR compression (`_apply_mpo_variational_su2`):** mirrors `common/strategy.py:_apply_mpo_variational` — left-to-right QR init + iterative sweeps — with every tensor replaced by its reduced-only block version. Each sweep step stacks per-sector `θ` blocks into `(N_out_sectors, D_l · D, D_r)` and `vmap(_qr_compactwy)` over the sector axis, batching all sector QRs into one GPU call. Sector shapes are identical under the uniform-`χ` convention, so `vmap` is immediate; lifting this to heterogeneous `χ_j` swaps the outer `vmap` for a per-sector `scan` without touching downstream code.

**Static schedule.** Block counts, sector schedules, outcome lists, intertwiner-consumption weights `w`, and shared-bond scalars are compile-time constants derived from `(group, shape, Q_x, phys_dim, charge_of_site, D, χ)`. Nothing reshapes during sweeps; no `jit` recompile.

### 5.3 Per-sample `vmap`

All operations above are `vmap`-friendly: block-id gather per sample uses `jnp.take` on a static axis; intertwiner contraction is a static einsum per sector; boundary-MPS compression `vmap`s over the sector axis inside each sweep and can be outer-`vmap`ed over samples.

### 5.4 Plaquette operator (`local_terms.py`)

```python
@register
class PlaquetteSU2Term(TransitionOperator):
    row: int; col: int          # top-left corner
```

Evaluation dispatches on this term (following the `_eval_term` pattern in `src/vmc/peps/gi/model.py:866`). At runtime:

1. **Precomputed outcome enumeration** (at model build). For each allowed plaquette input-irrep tuple `(j_1^{in}, j_2^{in}, j_3^{in}, j_4^{in})` — the 4 links around the plaquette — enumerate the finite set of non-zero output tuples `(j_1^{out}, j_2^{out}, j_3^{out}, j_4^{out})`. The enumeration is generic in `j_max`: loop over input tuples allowed by `P_{j_max}`, apply each of the 2⁴ fundamental raise/lower patterns from `Û□` and `Û□†`, drop patterns violating `P_{j_max}`.

   **Intertwiner-symmetrized amplitude.** Since ι is not sampled, the sampling basis is `|{j}⟩_sym = (1/√N_ι) Σ_ι |{j}, {ι}⟩` (equal superposition over all intertwiner configurations). The correct matrix element in this basis is:

   ```
   A_sym(tup_in → tup_out) = [1/√(∏_c n_c(tup_in) · n_c(tup_out))]
                             × Σ_{ι_corners, ι'_corners} ⟨tup_out, ι'|Û□ + Û†□|tup_in, ι⟩
   ```

   where `n_c(tup)` is the intertwiner multiplicity at corner `c` for the leg tuple induced by `tup` and the external (non-border) legs at that corner, and the sum runs over all corner-intertwiner pairs. Non-corner intertwiners cancel (they don't change and sum to `δ`). Each bare matrix element `⟨tup_out, ι'|Û|tup_in, ι⟩ = ∏_corners C_corner(ι, ι') · {6j}_recoupling` is a product of CG factors and {6j} symbols at the 4 corners. The sum is bounded by `∏_c n_c(tup_in) × n_c(tup_out)` terms — at most `(ι_max)^{2×4}` — and **precomputed once at model build** from the CG/{6j} tables.

   For `j_max=½`: `ι_max=2`, so at most 256 terms per (input, output) pair. Each term is a product of {6j} symbols — no tensor contraction, pure scalar arithmetic.

2. **Static table + `lax.switch` dispatch.** Input tuples get integer ids `input_id ∈ [0, N_inputs)`. Each `input_id` has its own compile-time outcome list `outcomes[input_id] = [(tup_out_k, amp_sym_k), …]` of length `n_k = len(outcomes[input_id])` — no padding, no zero-amp slots. At runtime, `jax.lax.switch(input_id, branches)` dispatches to the branch for that input tuple; each branch unrolls exactly its `n_k` outcomes. `Z_table[input_id] = Σ_k |amp_sym_k|²` is the matching proposal-normalization lookup.

3. **Runtime evaluation per branch.** For each outcome `k` in the dispatched branch:
   - Rebuild the four updated MPO bricks via §5.1 with the outcome irreps (using the current matter states `{p_x}` at the 4 corners when `phys_dim ≥ 2`).
   - Evaluate the 2×2 window via `_contract_2row_2col_su2` using the *updated* bricks against the *current* block-sparse 2-row envs.
   - Weight by `amp_sym_k` and sum.

Cost per plaquette term: `O(n_k · contraction_cost)` where `n_k` is the actual outcome count for this input tuple — `(2j+1)` magnetic-multiplicity factors and intertwiner-symmetrization already collapsed into `amp_sym_k` at model build. Matter-hopping outcome tables (when `phys_dim ≥ 2`) follow the same `lax.switch` pattern with analogous intertwiner-symmetrized amplitudes, keyed by `(tup_in, p_x, p_y)`, mirroring `src/vmc/peps/gi/model.py:677-755`.

### 5.5 Casimir / electric term (`LinkCasimirTerm`)

Purely diagonal in the sample: for each link `ℓ` with sampled irrep `j_ℓ`, add `g_E · j_ℓ (j_ℓ+1)`. Implemented as a `DiagonalOperator` subclass with `energy(h_links, v_links)` returning the sum. Zero contraction cost.

### 5.6 Kernels (`kernels.py`)

Mirrors `src/vmc/peps/gi/kernels.py` structure with three differences: (i) block-aware MPO bricks (sector-indexed), (ii) intertwiner-consumption weighted sum at brick assembly (§5.1–§5.2), (iii) gradient indexed by the single flat `block_ids[r, c][p, tup]` axis — no arithmetic packing needed (contrast Abelian `site * nc + cfg_idx`).

- **`init_cache`**: for each chain, build bottom envs by sweeping row-wise bottom→top. Uses `_build_row_mpo_su2` delegating to §5.1. Compression via `_apply_mpo_variational_su2` (§5.2). **Initial (below-the-lattice) bmps** is the all-singlet trivial environment: one sector `(j_l, j_p, j_r) = (0, 0, 0)` with block shape `(1, 1, 1)` — the SU(2) analog of `jnp.ones((1, 1, 1))` in `gi/kernels.py:84`.
- **`transition`**: sequence of sweeps over the lattice, mirroring `src/vmc/peps/gi/model.py:1593-1776`:
  1. **Plaquette sweep** over row pairs via an SU(2) analog of `_plaquette_sweep_row_pair` (`src/vmc/peps/gi/model.py:1134`). Maintain 2-row right envs (precomputed once per row pair) + a sliding 2-row left env updated after each column. For each plaquette, propose an outcome via scheme (A) of §2 (amplitude-weighted on the static outcome list), Metropolis–Hastings accept on `|Ψ(s')/Ψ(s)|² · Z(s)/Z(s')` via `_metropolis_hastings_accept(..., proposal_ratio=Z(s)/Z(s'))`, update the 4 bricks in place, advance the left env.
  2. **Matter hopping + link sweeps** (only when `phys_dim ≥ 2`): horizontal hopping per row, vertical hopping per row pair, and link-charge sweeps as in `_horizontal_link_sweep_row` / `_vertical_link_sweep_row_pair` (`src/vmc/peps/gi/model.py:1196`, `1287`). Each uses its own `_metropolis_hastings_accept` with the appropriate `proposal_ratio`.

  Per-sweep cost `O(N · D⁴ · χ²)` following Liu 2021.
- **`estimate`**: sweep rows top→bottom; at each row, reuse the cached block-sparse top/bottom envs to evaluate plaquette matrix elements using the intertwiner-symmetrized amplitudes `amp_sym_k` from §5.4, weighted by `Ψ_CW(s')/Ψ_CW(s)` via `_contract_2row_2col_su2`, accumulate `E_loc(S)`. Casimir is diagonal and adds zero-cost per link (§5.5). Gradients `G = (1/Ψ_CW) · ∂Ψ_CW/∂A[r,c][flat_id]` accumulate via the same block-sparse envs in one defect-network pass (Liu 2021 Eq. 6). Note `∂Ψ_CW/∂A[r,c](ι)` carries the consumption weight `w(ι)` (chain rule through the brick sum). Gradient collection mirrors `src/vmc/peps/gi/kernels.py:148-170` but scatters into the single flat `total_blocks[r, c]` axis via the same `block_ids[r, c][p, tup]` indices used at brick assembly; no arithmetic index packing.

### 5.7 Driver & integrator plumbing

**Zero changes to `TDVPDriver`**. The only plumbing addition is the `noqa: F401` import of `vmc.peps.su2_gi.kernels` next to the existing GI registration in `src/vmc/drivers/tdvp.py:32`.

**Ground state**: `TimeUnit=ImaginaryTimeUnit()`, `Integrator=Euler()`.
**Real-time**: `TimeUnit=RealTimeUnit()`, `Integrator=RK4()`.

Both share the same sampling / `build_mc_kernels` / `SRPreconditioner` stack.

---

## 6. Efficiency contract (end-to-end)

**Hard invariants — must hold throughout:**

1. **No padding, no mask on stored parameters.** Every `(p, sector, ι)` triple gets exactly `D⁴` reduced entries in the flat `A[r, c]`. The `ι_max` axis on runtime gathers is a compute pattern (sentinel indices + `jnp.take mode='fill'`), not storage padding.
2. **Reduced-only runtime tensors.** Every PEPS brick, boundary-MPS block, and env carries only reduced dims (`D` / `χ`). `(2j+1)` magnetic multiplicities enter only through precomputed scalar amplitudes at contraction time (§5.2) — never as a tensor axis.
3. **Static shapes.** Block counts, sector schedules, outcome lists, intertwiner-consumption weights `w`, shared-bond scalars, amplitude tables — all baked into `SU2GITables` at `SU2GIPEPS.__init__` from `(group, shape, Q_x, phys_dim, charge_of_site, D, χ)`. No `jit` recompile during the VMC loop.
4. **`vmap`-friendly end to end.** Outer `vmap` over samples, inner `vmap` over sector axis during QR / env ops. No dict-of-varying-shape pytrees in hot paths.
5. **QR-only.** No SVD. All compression via `_qr_compactwy` vmapped over sectors.

**Scope simplifications (explicit):**

- **Uniform `D_j ≡ D`** (and `χ_j ≡ χ`) across irrep sectors. Heterogeneous `D_j` is an additive extension (stacked-array → pytree-of-blocks + per-sector `scan`), does not touch the enumeration / weight / amplitude machinery.
- **Bosonic matter only.** Fermionic parity / swap-gate machinery is out of scope.

---

## 7. Extensibility notes

Built into the design from day one:

- **Matter fields (bosonic)** via `phys_dim ≥ 2` + `charge_of_site` in `SU2GIPEPSConfig`. Block tables already carry the `p` axis (§4.3); brick assembly consumes it (§5.1); local-term factories add matter terms conditionally (§2).
- **Background charges `Q_x ≠ 0`**: per-vertex target irrep in `SU2GIPEPSConfig.Qx`. Block tables recompute per-vertex; everything else unchanged.

Future extensions:

- **Fermionic matter** (e.g. staggered Kogut–Susskind): requires swap-gate/parity-charge machinery on top of the block-sparse layer (Wu–Dai 2025 style). The block-sparse tensor layer doesn't change; the vertex tensor gains a $\mathbb Z_2$ parity label on each virtual leg. Slotted in as a sibling module.
- **U(1), $\mathbb Z_N$ via the same `GaugeGroup` protocol**: all irreps 1-dim, CGs trivially $\delta$, intertwiner multiplicity always 1 — the SU(2) machinery degenerates cleanly.
- **SU(3)**: requires extending the contraction layer's multiplicity handling: SU(3) has multiplicity > 1 at 2-way fusion (e.g. `8⊗8`), so every per-sector contraction picks up a fusion-multiplicity axis. The protocol signature already carries this axis; the runtime does not yet consume it beyond the vertex level.
- **Larger `j_max` with heterogeneous `D_j`**: requires moving from stacked-array block storage to pytree-of-dense-blocks-per-sector. Outer `vmap` over samples still works (gather the right pytree leaves), but per-sector `vmap` becomes per-sector `scan` over the block list (still static, just different compile shape).

---

## 8. Tests

Organize as `tests/test_su2_gi_*.py`.

### 8.1 Group primitives
- `test_su2_cg_unitarity`: `∑_{m1 m2} CG(j1 m1; j2 m2 | j m) CG*(j1 m1; j2 m2 | j' m') = δ_{jj'} δ_{mm'}`.
- `test_su2_cg_condon_shortley`: spot-check against `sympy.physics.quantum.cg` for `j1, j2 ∈ {½, 1}`.
- `test_su2_fusion_counts`: `fuse` matches hand table for `j_max ∈ {½, 1}`.
- `test_su2_casimir`: `casimir(j) == j*(j+1)`.

### 8.2 Intertwiners, block tables, and consumption weights
- `test_su2_intertwiner_orthonormal`: `∑_m I^{(ι)} I^{(ι')*} = δ_{ιι'}` per leg tuple, at `j_max ∈ {½, 1}`, `phys_dim ∈ {1, 2}`, `Q_x ∈ {0, ½}`.
- `test_su2_block_count_vs_counted_enumeration`: `N_blocks[r, c, p]` matches an independent brute-force enumeration of Gauss-compatible `(tup, ι)` pairs for bulk / edge / corner vertices at `j_max ∈ {½, 1}`.
- `test_su2_weight_w_vs_sympy`: `w[r, c](tup, p, ι)` matches a sympy-based reference (`sympy.physics.quantum.cg`, `sympy.physics.wigner.wigner_6j/9j`) for randomly sampled leg tuples across `j_max ∈ {½, 1}`, `phys_dim ∈ {1, 2}`, `Q_x ∈ {0, ½}`. No hand-tabulated numbers.
- `test_su2_plaquette_amplitude_sym_vs_sympy`: intertwiner-symmetrized plaquette amplitudes `A_sym(tup_in → tup_out)` match an independent sympy CG + `wigner_6j` reference that explicitly sums over corner-intertwiner pairs, same parameter grid.

### 8.3 End-to-end amplitude
- `test_su2_amplitude_gauge_invariance`: vertex-local SU(2) rotation leaves `|Ψ(s)|²` unchanged on a `2×2` lattice.
- `test_su2_amplitude_matches_exact_dense`: random `SU2GIPEPS` at `D=2` on `2×2` lattice agrees with a fully-unfolded-dense contraction within `1e-10`.
- `test_su2_block_aware_matches_unfolded`: the block-aware `_apply_mpo_variational_su2` produces the same MPS (up to per-sector gauge) as a reference unfolded implementation on `3×3`.

The unfolded-dense reference contractor lives in `tests/helpers/su2_unfold.py` — it inflates per-sector blocks with their magnetic axes + CG vertex factors and does a plain einsum, serving as an independent oracle. It is imported **only** by tests; the production `src/vmc/peps/su2_gi/` path never densifies.

### 8.4 Sampling
- `test_su2_plaquette_flip_preserves_gauss_law`: sequential plaquette sweeps never produce a Gauss-law-violating `s`.
- `test_su2_plaquette_hastings_ratio`: scheme (A) proposal — verify `q(s→s') · Z(s) = |A(tup_out)|²` on a hand-tabulated outcome list, and that the Metropolis–Hastings accept using `Z(s)/Z(s')` satisfies detailed balance (emp. distribution of a long chain matches `|Ψ|²` under exact enumeration on `1×2` / `2×2`).
- `test_su2_plaquette_ergodicity`: from `all-j=0`, sequential plaquette sweeps visit every Gauss-law-compatible config on a `2×2` lattice (brute force).
- `test_su2_matter_hopping_detailed_balance` (only `phys_dim ≥ 2`): same detailed-balance check for the matter-hopping sweep on a `1×2` lattice.

### 8.5 Ground state (imaginary time)
- `test_su2_pure_gauge_3x3_jmax_half_gs_vs_ed`: `3×3` at `j_max=½`, `phys_dim=1`, SR + imaginary-time Euler, 2000 steps, compare to ED. Match within sampling error < `1e-3`. CI gate.
- `test_su2_pure_gauge_4x4_jmax_half_gs_vs_ed` (marked `slow`): same on `4×4`; singlet sector still tractable by `ed-lgt` but heavier. Benchmark gate only.
- `test_su2_pure_gauge_convergence_in_D`: `3×3`, sweep `D ∈ {2, 4}`, energy decreases monotonically.
- `test_su2_hardcore_boson_matter_gs_vs_ed` (marked `slow`): `3×3` at `j_max=½`, `phys_dim=2` with `charge_of_site=(0, ½)` (SU(2) hardcore-boson matter — analog of Wu–Liu's Z₂ hardcore-boson case). Match ED within sampling error.

### 8.6 Real-time (tVMC)
- `test_su2_pure_gauge_energy_conservation`: RK4 real-time evolution of a glueball-like initial state on `3×3`, `j_max=½`, `phys_dim=1`, `T=5`, `dt=0.005`; total energy conserved to `<0.5%` (analog of the Abelian vison test in Wu–Liu SM Fig. S3).

All non-`slow` tests run under `JAX_PLATFORM_NAME=cpu`, `pytest -m "not slow"`.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| **CG/{6j} convention errors** (sign / normalization vary between references) | Pin Condon–Shortley in `group.py`; cross-check against `sympy.physics.quantum.cg`; unit tests §8.1. |
| **Intertwiner basis choice** affects what `ι` means in stored blocks | Fix canonical s-channel tree + recursive binary fusion (§4.2). Document in `group.py`. |
| **Plaquette outcome set grows with `j_max`** | Enumeration is generic (§5.4 loop over `P_{j_max}`-allowed output tuples); compute + storage scale with no code change. |
| **Sample-space ergodicity** of plaquette-only moves | Verified for open BC + singlet `Q_x=0`; test §8.4. For PBC or Polyakov-loop-charged sectors, additional moves will be needed (out of scope). |
| **Uniform `D_j`** suboptimal at large `j_max` | Additive extension: stacked-array → pytree-of-blocks + per-sector `scan`; does not touch enumeration / weight / amplitude machinery. |
| **GCF derivation** for non-Abelian not in Wu–Liu paper proper | Self-contained Schur's-lemma argument in §2 GCF paragraph + doc in `group.py`. Verify analytically that per-sector `GL(D_j)` gauge freedom on each bond does not absorb any intertwiner freedom (intertwiner lives on the vertex, bond gauge acts on a single leg). |
| **Intertwiner-consumption weights** `w[r, c](tup, p, ι)` derivation | Closed-form recursive construction (§5.2) from the canonical s-channel tree; cross-checked against an independent sympy CG + `wigner_6j/9j` reference in §8.2 `test_su2_weight_w_vs_sympy` and end-to-end via `test_su2_amplitude_matches_exact_dense` (§8.3). No hand-coded tables. |
| **Not sampling ι** restricts variational state | The state lives in the intertwiner-symmetric subspace `|{j}⟩_sym` (§2). Plaquette matrix elements use intertwiner-symmetrized amplitudes (§5.4). Variational freedom loss is bounded: `ι_max=2` for `j_max=½` → at most 1 out of 9 bulk blocks constrained. Full expressiveness recoverable by upgrading to ι-sampling later (additive change to sample rep + sweep moves; no structural change to tensors, block tables, or contraction). |
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

1. **`group.py`** — SU2 + CG + {6j} + intertwiner basis + per-position-per-`p` block table + intertwiner-consumption weights `w` + shared-bond scalars + plaquette amplitudes. Testable in isolation (§8.1–8.2).
2. **`model.py` skeleton** — `SU2GIPEPS`, `SU2GIPEPSConfig` with `phys_dim`, `charge_of_site`, `Qx`; sample flatten/unflatten including optional `sites` axis; `random_physical_configuration` handling pure-gauge and matter cases.
3. **`contraction.py`** — `_build_row_mpo_su2` (§5.1 brick assembly, matter-aware) + `_apply_mpo_variational_su2` (§5.2 per-sector `vmap`ed QR) + env ops.
4. **`local_terms.py`** — `LinkCasimirTerm`, `PlaquetteSU2Term`, plaquette outcome table (§5.4). When `phys_dim ≥ 2`: `OnSiteMassSU2Term`, `HorizontalMatterHoppingSU2Term`, `VerticalMatterHoppingSU2Term` + hopping outcome tables.
5. **`kernels.py`** — `init_cache`, `transition`, `estimate` (§5.6); transition dispatches on whether `phys_dim == 1` to skip matter/link sweeps.
6. **Driver plumbing** — one-line `noqa: F401` import.
7. **Tests §8.1–8.6** in order. Pure-gauge path (steps 1–6 with `phys_dim = 1`) is the first CI gate; matter path validated next.
8. **Benchmark §10**.
