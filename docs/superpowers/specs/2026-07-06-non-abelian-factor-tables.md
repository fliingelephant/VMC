# Non-Abelian GI-PEPS — Factored Operator Tables

Date: 2026-07-06
Status: Implemented
Supersedes: sections 5.5, 5.6, and the dense matrix-table parts of section 10 in
`docs/superpowers/specs/2026-04-24-su2-gi-peps-full-intertwiner-design.md`.
Everything else in that spec (sampled spin-network formulation, sweeps,
contraction, gradients) is unchanged.

## 1. Problem

Off-diagonal operator data is currently stored as row-sparse outcome tables
whose row index (`starts`, `counts`, `proposal_norms`) is dense over tuples of
vertex block ids: `n_blocks²` for hopping, `n_blocks⁴` per plaquette. The
outcomes are sparse but the row index is a materialized join: it costs
`∏ n_blocks` in memory and Python build time to hold information whose true
content is `Σ n_blocks` per vertex. At `j_max = 1/2` this is invisible; at
`j_max = 1` it is gigabytes per plaquette and ~10⁸ Python iterations to build.
A first-principles representation must mirror the operator's algebraic
structure and hold at every truncation.

## 2. Factorization theorem

Every Hamiltonian and observable term in this family is a *string or loop of
link operators in one irrep R* (KS plaquette: loop in the fundamental f;
matter hopping: length-1 string in f with matter operators at the endpoints;
Wilson-loop observables: loops in any R). For such operators the spin-network
matrix element factorizes over the visited vertices into plain scalars:

```
⟨out | O_string | in⟩ = Π_{x ∈ visited vertices} λ_x(b_in_x, b_out_x)
```

The key is the canonical form of daggered links. Writing `U†` segments
(bottom/left plaquette links, h.c. hops) with *sector-swapped conjugated
couplings of the same operator irrep R* — never with couplings in the dual
irrep R̄ —

```
⟨out|U_αβ|in⟩      = √(dim_in/dim_out) · C[o_s, α, i_s] · conj(C[o_t, β, i_t]),
                     C = coupling(out, R, in)
⟨out|(U†)_αβ|in⟩   = √(dim_out/dim_in) · conj(C'[i_s, α, o_s]) · C'[i_t, β, o_t],
                     C' = coupling(in, R, out)
```

makes every operator-index tie of the loop a *corner-local bra–ket
contraction*, valid in any basis of any compact group. (The su3-style form
with R̄-couplings and plain ties silently requires `D^R̄ = conj(D^R)` — true
in SU(3)'s basis, impossible for the pseudo-real SU(2) fundamental, which is
why the old SU(2) code needed ε-metric connectors.) Each vertex then absorbs
the coupling half of every adjacent link end plus the intertwiner overlap
over spectator legs, and contracts to one plain scalar per (block_in,
block_out) pair. κ ≡ 1; there are no invariant patterns and no open ring
indices. Fusion multiplicity changes the *number* of valid pairs, never the
structure.

Corollary (self-crossing loops, future observables): a vertex visited twice
keeps open R-indices and the factor gains one small ring axis with a Schur
pattern; the table contract below extends by appending that axis. Implement
scalars now.

Hermitian terms `O + O†`: two orientation products from the *same* corner
functions, `λ_bwd(in→out) = conj(λ_fwd(out→in))`, with forward/reverse fusion
tables swapped. The sum of the two products does not factor further and is
formed at runtime (two products, then add).

## 3. Table contract

One `VertexFactorTable` per (site, role, orientation). Roles: plaquette
TL/TR/BL/BR, hopping h-src/h-tgt/v-src/v-tgt. Key = (input block, new sectors
of the touched legs); role determines the key arity (plaquette: two legs;
hopping: one leg — matter-state changes live in `out_blocks`, not the key).
Under the forward orientation, undaggered legs (top/right, hop link) take new
sectors from forward fusion `fuse(j, R)` and daggered legs (bottom/left) from
reverse fusion `{k : j ∈ fuse(k, R)}`; the backward orientation swaps the two
maps. The dual irrep R̄ never appears explicitly.

```
VertexFactorTable:
  group_starts: (n_blocks, n_irreps[, n_irreps]) int32   # slice start per key
  group_counts: (n_blocks, n_irreps[, n_irreps]) int32   # candidates per key
  max_candidates: int                                    # static loop bound
  out_blocks:   (total,) int32                           # candidate out block
  factors:      (total,) complex128                      # λ (oriented)
  w2_sums:      (n_blocks, n_irreps[, n_irreps]) float64 # Σ_cand |λ|² per key
```

Plus tiny shared fusion metadata (`FusionOutputs` for forward and reverse
fusion with R) — replaces `PlaquetteLinkTransitions`.

Storage and build per site-role: `O(n_blocks · c)`, `c ≲ 16`. Nothing dense in
more than one block index exists anywhere. `w2_sums` is the single declared
redundancy (derivable from `factors`; cached because proposals read it every
site visit).

## 4. Runtime evaluation — the multilinear fold

The window amplitude is **multilinear in the corner tensors**, and the ι-sums
at different corners are independent. Therefore they fold *inside* the window
contraction:

```
Σ_outcomes ME·Ψ(out)
  = Σ_links' Window( M_tl(links'), M_tr(links'), M_bl(links'), M_br(links') )
  where M_x = Σ_cand λ_x[cand] · tensors[x][out_blocks[cand]]
```

Per plaquette and orientation: one λ-weighted gather-sum per corner (cost
`O(k_ι D⁴)`) and **one window contraction per valid links' combination**
(static bound `Π_legs max|fuse(j, R)|`, e.g. 1 at SU(2) j_max=1/2, ≤16 at
j_max=1), instead of one window per outcome. This is strictly fewer window
contractions than the dense-table loop at every truncation (e.g. j_max=1/2
bulk: 2 windows vs ~16). Hopping: identical with two endpoint tensors and one
folded link.

Diagonal terms, gradients, environments, sweeps, `initial.py`: untouched.

## 5. Transition proposal — ancestral, exact

Proposal distribution `w(out) = |ME_O(out)|² + |ME_O†(out)|²` (both factor per
orientation). Sample hierarchically: orientation ∝ Z_orient, links' combo ∝
`Π_x w2_sums[...]`, then per-corner candidate ∝ |λ|². The realized density is
exactly `w(out)/Z(in)` with

```
Z(in) = Σ_orient Σ_links' Π_x w2_sums[x]      # O(n_irreps^legs) flops
```

`w` is symmetric (`w(out|in) = w(in|out)` — conjugation preserves |λ|² and
the orientation products swap), so the Hastings ratio needs only `Z(out)`,
computed at runtime from the out-blocks' tables the same way.
`proposal_norms` and `reverse_proposal_weights` cease to exist. Validity is
by construction: every candidate comes from its own vertex table with shared
new links, so link-consistency and block-validity of the joint outcome are
automatic (the candidate-masking helper dies).

## 6. Generic builder + self-calibration

New module `non_abelian_gi/factors.py`, group-generic. Group backends supply
via typed dispatch only three primitives:

- `fundamental_irrep(group)` — the operator irrep label R.
- `coupling_tensor(group, out, op, in)` — the unit-norm coupling
  `⟨out m_o|op m_a, in m_i⟩`, shape `(dim out, dim op, dim in)`, any fixed
  phase convention (phases cancel per operator: each coupling enters once
  plain and once conjugated).
- `vertex_tensor(group, block)` — orthonormal intertwiner tensor, axes
  `(l, u, r, d[, matter])`.

The generic module then provides:

1. **Oracles**: single-einsum full matrix elements for the plaquette ring and
   the hop string (build-time/test-only).
2. **Factor extraction**: per role, contract overlap × coupling halves → the
   scalar λ.
3. **Split assert**: at every table build, join a few instances across the
   per-site tables and **assert oracle == Πλ**. No convention is ever derived
   on faith; the numbers pin it.
4. Table packing into `VertexFactorTable`.

The SU(2)/SU(3) backends shrink to group mathematics (irreps, couplings,
intertwiner enumeration); all operator-table logic lives once in the generic
module. The old ring contractions survive only as parity references until
deletion.

## 7. Migration and validation

Order (suite green at every step):

1. `VertexFactorTable` added alongside the old classes.
2. `factors.py` with split asserts (`ring == Πλ` at build time).
3. **Ground-truth gate** (`tests/test_non_abelian_factors.py`). The old
   dense tables were disproven against Haar-measure integration, so the
   gate anchors to external truths instead of old-value parity:
   - plaquette vacuum↔loop saturates `|<loop|tr U|vac>| = 1` exactly
     (SU(2) −1 at both truncations; SU(3) +1);
   - SU(2) matter vertex tensors are exact Gauss intertwiners — V on
     incoming/matter slots, conj(V) on outgoing, the laws forced by
     `u → V_src u V_tgt†` and invariance of `ψ†Uψ`; the matter slot is
     dualized like an incoming leg (fixed in `su2.py`);
   - the meson hop reduced element is exactly 1 (exact Gauss-sector
     construction on the full matter-Fock ⊗ link space), with occupancy
     filtering (fwd creates at src / annihilates at tgt) and hermitian
     fwd/bwd pairing.
   Old-value record: old su2 plaquette = −½ × truth uniformly; old su3 off
   by ×3/×243 non-uniformly; old su2 hopping ±0.25 was a degenerate-tie
   artifact on matter states that lay outside the Gauss sector entirely.
4. Kernels switch (fold evaluation + ancestral proposal); ED and
   detailed-balance tests rebuilt on the factored tables.
5. Delete `PlaquetteMatrixTable`, `HoppingMatrixTable`, `_pack_sparse_rows`,
   `PlaquetteLinkTransitions`, the ring loops in su2.py/su3.py, and
   `_plaquette_candidate_samples`; migrate tests that touched those APIs.
   Shared exact-enumeration test helpers live in `tests/nonabelian_exact.py`;
   ED baselines were re-anchored to the canonical matrix elements
   (su2 3x3: -4.8234564111911995; su3 2x2: -0.8509840232358125). `group.fuse`
   now takes `(link irrep, any operator irrep)` — truncation filters outputs
   only — so over-tight truncations yield empty tables instead of crashing.

Note: this supersedes the `reverse_proposal_weights` portion of the current
working-tree changes; the rest of that WIP (initial.py DP sampler,
`from_blocks` consolidation, int16 slice indices) is orthogonal and stays.

## 8. Invariants

1. No array indexed by more than one vertex-block axis, anywhere.
2. Every factor table row is derivable from one vertex's data alone.
3. The split assert (oracle == Πλ on joined instances) runs at every table
   build.
4. Window-contraction count per term ≤ 2 × (valid links' combos).
5. Proposal normalizations are computed, not stored (except `w2_sums`).
6. Group backends contain no operator-table logic, only group primitives.
7. The dual irrep appears nowhere in operator tensors; daggered legs use
   reverse fusion of R.
