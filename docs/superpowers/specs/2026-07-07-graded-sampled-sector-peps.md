# Graded Sampled-Sector PEPS — Unifying Statistics and Gauge Structure

Date: 2026-07-07
Status: Design accepted; implementation not started
Relates to: `2026-07-06-non-abelian-factor-tables.md` (factor tables are the
gauge half of this design and are unchanged by it).
Primary reference: Wu & Dai, arXiv:2506.20106 (fPEPS VMC in the swap-gate
formulation; detailed balance of sequential sampling).

## 1. Problem

Fermionic statistics is needed twice: ordinary fermionic PEPS (Hubbard-type)
and LGT PEPS with fermionic matter. Defining families per feature
(standard, fermionic-standard, GI, GI-with-fermionic-matter, ...) multiplies
modules along independent axes. A first-principles decomposition must make
features compose as data instead of forking code.

Additionally, the current LGT hopping carries no Jordan-Wigner string signs:
vertical hops in 2D are wrong whenever the intervening occupation parity is
odd (today's matter is effectively hardcore-bosonic). This is a correctness
bug independent of any ansatz question.

## 2. The object

Every model in this repo is one thing: **a PEPS over a graded sector
structure, evaluated in a basis that resolves all sectors.**

- Legs carry sector labels from a fusion structure (trivial, Abelian charges,
  non-Abelian irreps × multiplicity).
- Site tensors are degeneracy data constrained to well-formed blocks
  (Gauss/intertwiner constraint) and to even fermion parity.
- Fermionic statistics is **one extra bit of structure**: a parity function
  on sector labels (plus optional parity vectors on degeneracy legs), with
  the braiding rule (−1)^{P·P'}. It is never a family.

The named models are coordinates in a product space:

| model                    | sector structure | parity map            |
|--------------------------|------------------|-----------------------|
| standard PEPS            | trivial          | 0                     |
| fermionic PEPS           | trivial          | occupation parity     |
| non-Abelian GI PEPS      | spin network     | 0                     |
| GI + fermionic matter    | spin network     | matter parity(block)  |

Physics may tie the axes together (staggered matter: parity is a function of
the Gauss sector); the code dependence stays one-directional (parity =
f(sector label)) and composes.

## 3. Why VMC collapses the grading (the load-bearing fact)

Gauge structure and fermionic statistics are properties of the **basis**, and
VMC evaluates amplitudes in a fixed basis. After sampling:

- gauge structure → block gather (already implemented: sampled blocks),
- fermionic braiding → diagonal ±1 gates on virtual legs determined by
  cumulative sample parities (Wu-Dai Eq. 8), absorbable into site tensors as
  elementwise masks, plus **scalar** JW string signs on operator matrix
  elements (Wu-Dai Eq. 13),
- what remains → an ordinary dense tensor network.

Design principle (governs everything below): *sampled quantum numbers live in
the sample; tensors carry only well-formedness constraints (Gauss blocks,
parity evenness); all statistics evaluate to classical bookkeeping on the
sample.* Number conservation is likewise imposed by sampling in the fixed-N
sector, never by U(1)-blocking tensors.

Corollary — **no graded tensor algebra**: no graded einsum, no graded SVD, no
swap-gate objects at runtime, no fpeps/ directory. Boundary compression acts
on the sampled, sign-absorbed, ungraded network; ordinary SVD is exact in the
same sense as today. A graded algebra could only be needed for state-level
algorithms (full update, canonical forms) this repo does not run; if that day
comes it enters as a substrate swap *below* `peps/common`, invisible to
models.

## 4. Layering

1. **Dense substrate** — `peps/common` (contraction, environments,
   compression, energy/gradients, cache turnover). Sector- and
   statistics-blind. Byte-identical to today.
2. **Structure layer** — static metadata: sector tables
   (`PureGaugeTables`), factor tables (`factors.py`), and the one new
   module `peps/grading.py` (§5).
3. **Assembly** — model classes: parameters + sample↔sector codec + one rule
   `(tensors, sample) → dense site tensors` (slice or block-gather, then
   parity masks). Standard PEPS is the trivial-structure point of this layer.
4. **Kernels** — shared sweep skeleton, per-term move/ME objects (dense local
   matrices or factor tables), and a universal statistics decorator (§6).

## 5. The grading module (`peps/grading.py`) — the only new structure

Contents (all static metadata + pure functions; no tensor algebra):

- **Parity data.** Per-bond parity vectors for degeneracy legs
  (contiguous layout: first `D_even` indices even, rest odd) and the parity
  map on physical/sector labels (standard: occupation parity; LGT:
  `matter_number(block) mod 2`, already in the tables).
- **Even-parity masks.** For each site, the 0/1 mask zeroing entries with
  `Σ P_legs ≠ P_phys mod 2`. Applied to stored dense tensors at assembly.
  Gradients of masked entries vanish identically (environments have definite
  parity), so masked-dense storage is consistent; the QGT parameter space
  excludes structurally-zero entries via the existing sliced-Jacobian
  machinery. Storage stays dense — XLA-fused masks beat block einsums at
  these bond dimensions.
- **Planar convention.** Mode order and physical-leg routing are fixed
  together: legs routed parallel to columns so swap gates land only on
  within-row (horizontal) bonds; the induced JW mode order is column-major.
  The gate on bond h(r, c) is the diagonal (−1)^{p̃·P_k} with p̃ the prefix
  parity of column c above row r — data that rides along with the top-env
  cache turnover in O(1) per site, exactly like environments. (Wu-Dai's
  construction transposed to our row sweeps; their two properties —
  routing-gauge equivalence and locality under fermionic gates — carry over.)
- **String signs.** `(−1)^{#ij}` with `#ij` = occupied modes strictly between
  i and j in the mode order, evaluated from maintained prefix parities
  (never an O(N) scan in hot loops). In column-major order vertical hops are
  string-free; horizontal hops read two column-prefix parities. Multi-mode
  sites (spin/color) fix a within-site mode order; site parity is total
  occupation mod 2.

No convention above is trusted on derivation alone; each is pinned by the
gates in §7 (same discipline as the factor tables).

## 6. Operators and kernels

- Term types are typed objects; fermionic transition terms (e.g.
  `FermionicHorizontalHopping`) declare their statistics, and kernels
  multiply the matrix element by the braiding scalar. Diagonal terms are
  untouched.
- LGT: the string sign multiplies the hop λ-product as a per-window scalar
  (the string covers sites outside the window whose occupations the move does
  not change; the matter-parity shift is uniform within a fusion combo, so
  the multilinear fold is untouched). Factor tables never learn about
  statistics.
- **Sampling is statistics-blind**: |sign| = 1, so proposal weights, Z
  ratios, and Hastings ratios are unchanged. Statistics enter amplitudes
  (masks) and ME phases only. Wu-Dai Theorem 1 (detailed balance of
  sequential sampling) covers our sweep scheme as-is.
- Bosonic models compile to today's kernels exactly: trivial grading is a
  static `None` branch at kernel build; no masks, no signs, zero overhead.

## 7. Validation gates (external anchors only)

Phase 1 (fermionic standard PEPS):
- Brute-force reference: an independent small-lattice contraction in the
  ordered Fock space with explicit swap matrices; random even tensors;
  amplitudes and hop local energies must match the production path exactly.
- Free-fermion ED: quadratic Hamiltonians (insulator/Dirac/Chern per Wu-Dai
  §IV) — SR convergence to exact energies; boson-PEPS control run expected
  to fail (their Fig. 3 asymmetry is the point of the grading).
- Detailed-balance test on the fermionic sampler (existing pattern).

Phase 2 (LGT string signs):
- Exact Gauss-sector framework extended with the JW mode order (including
  color modes): vertical-bond hop on 2×2 with **odd intervening parity** as
  the anchor; hermitian pairing must survive the signs.

## 8. Migration order

1. `peps/grading.py` + fermionic standard PEPS as metadata on the existing
   family (`PEPSConfig` gains optional grading; no new model class), with
   the Phase-1 gate.
2. Parity column + string signs in `non_abelian_gi` hopping (correctness
   fix), with the Phase-2 gate.
3. Parity-graded degeneracy legs for LGT matter (§8.3, accepted 2026-07-07;
   the sampled-basis analogue of fermionic rishons). Pinned construction:

   - **Config.** `NonAbelianGIPEPSConfig.n_even: int | None = None`; `None`
     is today's ungraded ansatz (statistics via ME strings only, phase 2).
     Set, it grades every degeneracy leg with the contiguous parity layout
     of §5 (dimension-1 boundary legs are all-even). The physical parity of
     a block is `matter_numbers[matter_state_by_block] mod 2`; gauge links
     stay bosonic (no parity on irrep labels).
   - **Assembly.** Per-site static mask over `(block, up, down, left,
     right)` keeping entries with `P_phys(block) + Σ P_legs` even, folded
     into the tensors once per kernel call; sample-keyed right-leg gate
     `(−1)^{prefix[r,c]·P_right}` with `prefix` the matter-parity column
     prefixes (phase-2 data). Gates are block-independent, so one decorated
     site array serves row MPOs, `_block_mpo` gathers, and `_folded_mpo`
     candidate folds alike.
   - **Ansatz/operator split.** Because a hop moves exactly one fermion,
     every matter-hop window corrects the *ansatz* side unconditionally:
     horizontal windows flip the right site's down leg and carry the
     re-gauge scalar `(−1)^{suffix[r,c+1]}` (the phase-1 pair-cancelled
     re-gauge, matter parities in place of occupation parities); vertical
     windows flip the lower site's right leg, scalar-free. The JW string
     stays the *operator*-side decorator of phase 2, unchanged. Invariant 3
     is preserved verbatim.
   - **Transition.** Plaquette and iota moves conserve matter parity, so
     they run on the decorated tensors with zero changes and the cached
     bottom envs stay valid. The horizontal hopping phase transposes the
     phase-1 Phase-A scheme: fresh row gates from maintained `pi_row`,
     stale rebuilt-at-phase-start bottom envs re-gauged through per-column
     interface exponents `delta` (accepted hop at bond (c,c+1) toggles
     `delta[c+1]` only), working MPOs stripped at row end; the maintained
     amp is magnitude-true under the scalar drift and acceptance reads
     magnitudes only. The vertical hopping phase is Phase-B: gate flips are
     in-window (below-pair prefixes never change), bottom envs stay exact.
     Proposal combos are still drawn from `|λ|²` tables — statistics never
     enters weights or Hastings ratios.
   - **Gradients.** Environment gradients of the decorated network are
     un-decorated at the sampled block by `mask[block] × gate`, exactly the
     phase-1 rule; masked entries have identically zero gradients and ride
     the existing sliced-Jacobian machinery.
   - **Gate.** Independent reference contraction of the masked/gated block
     network per sample; kernel local energies against the dense fermionic
     Hamiltonian over the full valid basis with the graded ansatz; sampler
     stationarity against `|ψ_graded|²`; `n_even=None` byte-identical to
     phase-2 kernels.

4. Deferred: folding standard PEPS into the sector-PEPS model as the
   trivial-group point — only if net slimmer with zero efficiency loss on
   the dense path.

## 9. Invariants

1. Statistics is metadata: a parity function on sector labels plus optional
   leg parity vectors. No fermionic family, no graded tensor type, anywhere.
2. `peps/common` remains sector- and statistics-blind.
3. Fermionic signs enter at exactly two points: assembly masks and the ME
   phase decorator. Nothing else may consult parity.
4. Proposal weights and Hastings ratios never see statistics.
5. Conventions are pinned by external anchors (brute-force graded
   contraction, free-fermion ED, exact Gauss sector), never by parity with
   prior code.
6. Trivial grading compiles to today's kernels exactly.
7. Factor tables remain group-only; string signs are kernel-level scalars.
8. String-sign and gate data are maintained incrementally with the sweep
   (O(1) per site), never recomputed by scanning the sample.
