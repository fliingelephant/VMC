# SU(2) Gauge-Invariant PEPS - Full Intertwiner Design

Date: 2026-04-24
Status: Implementation in progress
Supersedes: `docs/superpowers/specs/2026-04-18-su2-gi-peps-design.md`

This design extends the Abelian GI-PEPS idea to SU(2) by sampling the full
spin-network basis: matter labels, link irreps, and vertex intertwiners. The
core principle is the same as the Abelian implementation: a physical sample
selects exactly one allowed local tensor block at each vertex. No invalid
blocks are stored, no masks are applied to parameters, and no hidden
intertwiner sum creates redundant variational directions.

The implementation split is:

- `src/vmc/peps/su2_gi/group.py`: SU(2) symmetry semantics and static metadata.
- `src/vmc/peps/common/block_sparse.py`: generic scheduled block-sparse linear
  algebra.
- `src/vmc/peps/su2_gi/`: the SU(2) model, local terms, kernels, and tests.

The first implementation gate is pure gauge SU(2) Yang-Mills with open
boundaries. Matter is deferred until the pure-gauge amplitude, sampling,
gradient, and ED checks pass.

---

## 1. Design Decision

### 1.1 Intertwiners are sampled

For an Abelian lattice gauge theory, Gauss law makes the local invariant space
dimension either zero or one. Once matter and link charges are sampled, there is
no additional local physical label.

For SU(2), a fixed local tuple of adjacent link irreps can fuse to the target
charge in more than one independent way. The extra basis label is a vertex
intertwiner. Therefore a complete gauge-invariant spin-network basis state is:

```text
sample = (matter states, link irreps, vertex intertwiners)
```

For the pure-gauge MVP, matter states are trivial:

```text
sample = (h_links, v_links, intertwiners)
```

This is the cleanest VMC representation because each sampled vertex state maps
to one tensor block:

```text
(j_l, j_u, j_r, j_d, iota) -> block_id
```

No contraction-time sum over unsampled `iota` is needed. This avoids null
parameter directions and keeps the current `SlicedJacobian` small-o trick valid:
one active parameter slice per site per sample.

### 1.2 Magnetic indices are not sampled

The magnetic indices `m` on link basis states are still not sampled. They are
contracted analytically through the chosen orthonormal intertwiner basis and
through precomputed operator matrix elements. Runtime tensors carry reduced
degeneracy dimensions only.

### 1.3 The ansatz targets the full gauge-invariant sector

Sampling intertwiners gives the full truncated SU(2) spin-network Hilbert space
for the chosen `j_max` and boundary charges. A symmetric-projected ansatz that
does not sample intertwiners is a possible later reduced ansatz, but it is not
the default because it restricts variational expressivity.

---

## 2. Scope

### 2.1 In scope for MVP

- Pure SU(2) Yang-Mills on an open-boundary square lattice.
- Kogut-Susskind Hamiltonian with link truncation `j <= j_max`.
- First validation target: `j_max = 1/2`.
- Gauge canonical form: link tensors are parameter-free identities on reduced
  degeneracy indices.
- Full spin-network sampling: link irreps plus vertex intertwiners.
- Block-sparse storage for vertex tensors and boundary-MPS environments.
- Static metadata closed over in `kernels.py`.
- Typed dispatch for model/operator/kernel behavior.
- Compatibility with the current sampler, TDVP driver, SR preconditioner, and
  sliced QGT path.

### 2.2 Deferred

- Bosonic matter.
- Fermionic matter.
- SU(3) or generic non-Abelian groups.
- Heterogeneous reduced dimensions `D_j` and `chi_j`.
- Refactoring existing `src/vmc/peps/gi/`.
- Exact hidden-intertwiner marginalization.
- Symmetric-projected ansatz without intertwiner sampling.

---

## 3. Physics Contract

### 3.1 Hilbert-space basis

Each oriented link carries the truncated SU(2) electric basis:

```text
|j, m_L, m_R>,    j in {0, 1/2, 1, ..., j_max}
```

The gauge-invariant basis on the full lattice is the spin-network basis:

```text
|{j_link}, {iota_vertex}>
```

where each `iota_vertex` labels an orthonormal basis vector in:

```text
Inv(j_l x j_u x j_r* x j_d* x Q_x*)
```

For SU(2), all irreps are self-dual, but the incoming/outgoing orientation is
still part of the convention and must be fixed consistently.

Boundary legs outside the lattice are fixed to `j = 0`. For the MVP,
`Q_x = 0` at every vertex.

### 3.2 Hamiltonian

Pure gauge Hamiltonian:

```text
H = g_E * sum_links E_link^2 - g_B * sum_plaquettes (U_square + U_square^dagger)
```

with:

```text
E_link^2 |j, m_L, m_R> = j(j+1) |j, m_L, m_R>
```

The plaquette term is evaluated in the spin-network basis. It changes the four
plaquette-border link irreps and the four adjacent vertex intertwiners. Matrix
elements are precomputed from SU(2) recoupling data for each local plaquette
input state.

### 3.3 Gauge canonical form

The link tensor has no variational content after gauge fixing. For each irrep
sector `j`, Schur's lemma leaves only a reduced degeneracy-space gauge freedom.
That freedom can be absorbed into adjacent vertex tensors, leaving the link as
an identity on matching reduced indices.

Therefore the only variational parameters are vertex tensor blocks:

```text
A_x[block_id, a_l, a_u, a_r, a_d]
```

where `block_id` is an allowed local spin-network state
`(j_l, j_u, j_r, j_d, iota)`.

---

## 4. Module Layout

```text
src/vmc/peps/common/
  block_sparse.py      # generic scheduled dense-block execution

src/vmc/peps/su2_gi/
  __init__.py
  compat.py           # apply/value compatibility surface, row-brick assembly
  group.py             # SU(2), CG, intertwiners, static tables, schedules
  model.py             # SU2GIPEPS config/module/sample helpers
  local_terms.py       # Horizontal/VerticalLinkCasimirTerm, PlaquetteSU2Term,
                       # outcome tables
  kernels.py           # build_mc_kernels dispatch
```

No existing Abelian GI code is changed.

---

## 5. Symmetry Layer: `su2_gi/group.py`

`group.py` owns the mathematical meaning of sectors and intertwiners. It should
not perform boundary-MPS compression or environment contraction. It builds the
static metadata consumed by generic block-sparse execution.

### 5.1 `SU2`

Minimal concrete group object:

```python
@dataclass(frozen=True)
class SU2:
    j_max_twice: int

    def irreps(self) -> tuple[int, ...]: ...
    def dim(self, j_twice: int) -> int: ...
    def fuse(self, a_twice: int, b_twice: int) -> tuple[int, ...]: ...
    def casimir(self, j_twice: int) -> float: ...
    def cg(self, a_twice: int, b_twice: int, c_twice: int) -> jax.Array: ...
```

Use integer labels `j_twice = 2j` everywhere in runtime tables. This avoids
float keys and keeps all static lookups integer-indexable.

### 5.2 Intertwiner basis

For each vertex position and local link tuple:

```text
(j_l, j_u, j_r, j_d)
```

enumerate all fusion paths that end in target `Q_x`. Use one canonical fusion
tree throughout the implementation and tests. For pure gauge:

```text
(j_l, j_u) -> j_m
(j_r, j_d) -> j_n
(j_m, j_n) -> Q_x
```

The intertwiner label is the internal fusion path. For `Q_x = 0`, this often
reduces to a single internal channel, but not always one local basis state:
four spin-1/2 legs have two singlet intertwiners.

The basis must be orthonormal under explicit magnetic-index contraction. Tests
compare against independent SymPy CG and Wigner-symbol references.

Implementation note: `group.py` owns the Condon-Shortley Clebsch-Gordan
coefficients and the canonical `VertexBlock -> magnetic-index tensor`
construction. Plaquette recoupling tables must use those same functions, not a
separate phase convention.

### 5.3 Vertex block table

For each vertex `(r, c)`, build:

```text
blocks[r][c][block_id] =
    VertexBlock(j_l, j_u, j_r, j_d, iota_id)
```

Only valid blocks are stored. Boundary legs outside the lattice are fixed to
`j = 0`, so corners and edges have fewer blocks.

Also build an inverse lookup:

```text
block_id[r, c, j_l, j_u, j_r, j_d, iota_slot] -> int
```

This lookup is static metadata. Runtime row-brick assembly performs one
integer lookup and one tensor gather per site.

Static numeric metadata that is indexed inside kernels is stored as dense JAX
arrays, not NumPy arrays or Python tuples. `VertexBlock` records and lookup
dicts remain Python build-time metadata; `block_id_lookup`, plaquette outcome
indices, matrix elements, proposal weights, and proposal norms are JAX arrays
closed over by kernels.

### 5.4 Intertwiner slots

The sampled `iota` is local to the current link tuple. Runtime samples store an
integer slot:

```text
iotas[r, c] in [0, n_iota(r, c, j_l, j_u, j_r, j_d))
```

Changing a link can change the valid intertwiner count at neighboring vertices.
Therefore every off-diagonal proposal table must update both affected link
irreps and affected `iota` slots.

### 5.5 Plaquette transition table

For each plaquette and each valid local input state, precompute finite
outcomes:

```text
input:
  border link irreps
  external corner link irreps
  four corner iotas

outcome:
  new border link irreps
  new four corner iotas
  matrix_element = <out|U_square + U_square^dagger|in>
```

The external corner link irreps are needed because they determine the local
intertwiner spaces at the four vertices.

Store:

```text
outcome_ids[input_id]       # static branch data
matrix_elements[input_id]
proposal_weights[input_id] # default abs(matrix_element)^2
proposal_norm[input_id]    # sum proposal_weights
```

Use an amplitude-weighted proposal by default:

```text
q(in -> out) = abs(H_out_in)^2 / Z(in)
```

The Metropolis-Hastings correction is:

```text
accept = min(1, |Psi(out)/Psi(in)|^2 * Z(in)/Z(out))
```

This is static-table-driven and works even when different inputs have different
numbers of outcomes.

### 5.6 Boundary-sector schedules

`group.py` builds schedules for block-sparse contractions but does not execute
them. Schedules describe compatible sector combinations and output sector ids:

```text
MPS sector + MPO sector -> output MPS sector
left env sector + site sector + right env sector -> scalar contribution
```

The schedule is group-specific metadata. The executor in `common/block_sparse.py`
only sees integer block ids, dense arrays, and einsum patterns.

---

## 6. Generic Block-Sparse Layer: `common/block_sparse.py`

`block_sparse.py` is not SU(2)-aware. It operates on dense blocks according to
static schedules. The same executor should be usable later for U(1), Z_N, or
other symmetric PEPS layouts.

### 6.1 Required concepts

Keep the API small. Use typed dataclasses only where they remove ambiguity:

```python
@dataclass(frozen=True)
class BlockLayout:
    block_shape: tuple[int, ...]
    n_blocks: int

@dataclass(frozen=True)
class BlockMPSLayout:
    n_sites: int
    site_layouts: tuple[BlockLayout, ...]

@dataclass(frozen=True)
class BlockApplySchedule:
    # Static integer arrays describing compatible input/output blocks.
    ...

@dataclass(frozen=True)
class BlockWindowSchedule:
    # Static integer arrays for bottom/window/env contractions.
    ...
```

Do not introduce a symmetric-tensor library abstraction. These are execution
plans for dense blocks.

### 6.2 Runtime objects

Runtime physical block-sparse arrays are stacked dense blocks:

```text
blocks: jax.Array  # shape (n_blocks, *block_shape)
```

For uniform MVP dimensions:

```text
vertex block:       (D_u, D_d, D_l, D_r), inactive boundary legs have dim 1
boundary-MPS block: (chi_l, D_vertical, chi_r)
env block:          schedule-dependent reduced shape
```

The leading block axis is static. Invalid group-theory sectors are never
materialized.

Current production interpretation:

- Vertex tensors are block-sparse over sampled spin-network physical states.
- A Monte Carlo sample selects exactly one vertex block per site.
- The selected row-MPO is then a dense degeneracy tensor network and uses the
  existing `ContractionStrategy` interface. The default is QR-based
  `Variational`.
- Sector-block boundary MPS objects are deferred until virtual bonds carry
  explicit sector labels. Adding them before that would introduce unused
  structure.

### 6.3 Operations

The implemented generic layer currently provides the pieces used by the SU(2)
MVP:

```python
def gather_block(blocks, block_id): ...
def scatter_block_grad(block_grad, block_id, n_blocks): ...
def build_eval_schedule(bucketed_terms, eval_span): ...
```

Future sector-block boundary operations, if virtual sectors are introduced,
should be generic functions with static schedules:

```python
def apply_block_mpo(mps, mpo, schedule, strategy): ...
def contract_bottom(mps, schedule): ...
def contract_2row_2col(envs, bricks, schedule): ...
def compute_right_envs(...): ...
def update_left_env(...): ...
def evaluate_bucketed_terms(...): ...
```

Those functions should use static schedules and `jax.vmap` over compatible
block lists where shapes match. They must not inspect SU(2) charges,
intertwiners, CG coefficients, or plaquette rules.

### 6.4 Operator evaluation scheduler

Operator evaluation follows the existing codebase convention:

- Physical operator support is defined by typed `support_span(term)` dispatch.
- Model-specific contraction windows are defined by `type(model).eval_span(term)`.
- `kernels.py` calls `merge_operators(..., eval_span=type(model).eval_span)`.
- Runtime evaluation calls typed `_eval_term.dispatch` on the operator type and
  an environment context.

`eval_span` is allowed to differ from physical support. For example,
`BlockadePEPS` evaluates one-site operators in a `2x2` window because the local
constraint makes the larger environment cheaper to reuse than rebuilding many
small contractions.

The block-sparse executor must treat local-energy evaluation as a sweep over
shared environments, not as independent contractions per operator. The static
bucketed schedule should be equivalent to:

```text
for r:
  top_env = top_envs[r]
  bottom_envs for all needed dr are already cached or built once
  row bricks for rows r ... r + max_dr - 1 are built once

  for dr:
    bottom_env = bottom_env below the dr-row window
    right_envs = suffix environments for this (r, dr), built once
    left_env = identity

    for c:
      env_window = (top_env, bottom_env, left_env, right_envs, row bricks)

      for dc:
        evaluate all operators anchored at (r, c) with eval_span=(dr, dc)
        reuse the same env_window and the matching right suffix

      left_env = advance by one column through the dr-row window
```

This reuse pattern is a hard invariant. Top, bottom, right, and left
environments are built once for their sweep scope and reused across all
operators sharing `(r, dr, c)`; operators with different `dc` select different
window closures but do not rebuild the row-pair environments.

The existing `BucketedOperators` structure groups terms by row, effective `dr`,
and anchor column. For SU(2) block-sparse evaluation, build an additional static
`BlockEvalSchedule` from that bucketed data that groups the column entries by
`dc`. This keeps the public operator convention unchanged while making window
reuse explicit in the executor.

### 6.5 Compression

For the current sampled-block MVP, use the existing dense `Variational`
boundary-MPS compression after active physical blocks are selected. This is the
canonical path until virtual bonds become sector-labeled.

If sector-block boundary MPS objects are introduced later, use a block-sparse
version of the existing `Variational` compression:

- Left-to-right QR initialization.
- Alternating least-squares sweeps.
- QR-only canonicalization.
- Per-sector QR batched by `vmap` when block shapes match.
- No SVD in the SU(2) production path.

Typed dispatch should route:

```text
Variational + dense tuple       -> existing common strategy
Variational + BlockSparseMPS    -> block_sparse implementation
```

Do not add `BlockSparseMPS` or a new strategy type until there is a concrete
sector-labeled virtual layout that uses it.

---

## 7. Model: `su2_gi/model.py`

### 7.1 Config

Pure-gauge MVP config:

```python
@dataclass(frozen=True)
class SU2GIPEPSConfig:
    shape: tuple[int, int]
    j_max_twice: int
    D: int
    chi: int
    Qx_twice: Any = 0
    dtype: Any = jnp.complex128
```

Matter fields are added later by extending the config with `phys_dim` and
`charge_of_site_twice`. Do not design the pure-gauge implementation around
matter branches.

### 7.2 Parameters

At each site:

```text
tensors[r][c].value.shape == (n_blocks[r, c], D_u, D_d, D_l, D_r)
```

where `n_blocks[r, c]` is the number of valid
`(j_l, j_u, j_r, j_d, iota)` blocks at that position. The tensor axis order
after `block_id` follows the existing PEPS convention: `(up, down, left,
right)`. Open-boundary legs have reduced dimension `1`; active internal legs
have reduced dimension `D`.

The all-zero irrep block is initialized with a small positive vacuum overlap
plus noise. Other blocks are initialized with scaled complex Gaussian noise.

### 7.3 Sample representation

For pure gauge:

```text
h_links: shape (n_rows, n_cols - 1), int32 j_twice ids
v_links: shape (n_rows - 1, n_cols), int32 j_twice ids
iotas:   shape (n_rows, n_cols), int32 local intertwiner slot
```

Flatten in a deterministic order:

```text
flatten_sample(h_links, v_links, iotas) -> int32 vector
```

Unflatten uses only `shape`, so kernels can close over the model metadata.

### 7.4 Sliced-QGT metadata

For each site:

```text
params_per_site[r, c] = D_u * D_d * D_l * D_r
sliced_dims[r, c] = n_blocks[r, c]
```

For a sample, the active slice index at a site is exactly:

```text
block_id(r, c, j_l, j_u, j_r, j_d, iota)
```

This matches the existing `SlicedJacobian` assumption: one active slice per site.

---

## 8. Kernels: `su2_gi/kernels.py`

`kernels.py` follows the current Abelian pattern: the dispatch function captures
all static metadata and returns `init_cache`, `transition`, and `estimate`.

### 8.1 Dispatch

Register:

```python
@build_mc_kernels.dispatch
def build_mc_kernels(
    model: SU2GIPEPS,
    operator: object,
    *,
    full_gradient: bool = False,
    observables: tuple = (),
) -> tuple[Any, Any, Any]:
    ...
```

No string routing. No naming-based `_su2` dispatch at call sites.

### 8.2 Static closure

At kernel-build time, close over:

- `shape`
- `config`
- `strategy`
- `tables`
- `bucketed_terms` from `merge_operators`
- `coeff_structure`
- vertex block layouts
- boundary-MPS layouts
- block apply schedules
- bottom/window/env contraction schedules
- block evaluation schedules grouped by `(row, dr, col, dc)`
- plaquette outcome tables
- `params_per_site`
- `sliced_dims`

Operator bucketing follows the existing PEPS kernels:

```python
all_operators = (operator,) + observables
bucketed_terms, coeff_structure = merge_operators(
    all_operators,
    shape,
    eval_span=type(model).eval_span,
)
```

`support_span(term)` remains the typed physical-support dispatch. `eval_span`
is the typed model-specific contraction-window dispatch.

The dynamic kernel arguments are only:

- tensors
- sample
- PRNG key
- cache/context
- optional time-dependent coefficients

### 8.3 `init_cache`

For each chain:

1. Unflatten the sample.
2. Build row bricks by one active block gather per site.
3. Sweep bottom to top.
4. Apply row MPOs using `block_sparse.apply_block_mpo`.
5. Store bottom environments before each row.

The below-lattice environment is the trivial singlet boundary sector with one
dense block equal to one.

### 8.4 `transition`

Pure-gauge transition is a sequential plaquette sweep:

1. Maintain cached top/bottom block-sparse environments for each row pair.
2. At each plaquette, compute the local `input_id` from current links and
   corner intertwiners.
3. Sample an outcome from the static proposal distribution.
4. Update the four plaquette-border links and four corner intertwiners in a
   proposed sample.
5. Rebuild only changed row bricks.
6. Contract the changed 2x2 window to get `Psi(proposed)`.
7. Accept/reject with the Hastings correction.
8. Update row bricks and sliding left environment after the decision.

The output context includes the final amplitude and top environments for the
accepted sample, following the existing cache-turnover pattern.

### 8.5 `estimate`

For a sampled state `s`:

```text
E_loc(s) = sum_sprime H(sprime, s) * Psi(sprime) / Psi(s)
```

Implementation:

- Add diagonal link Casimir contribution directly from sampled `h_links` and
  `v_links`.
- Build bottom environments once during the backward sweep, exactly following
  the cache-turnover pattern.
- For each row `r`, reuse the cached top environment and the running bottom
  environment.
- For each effective row span `dr`, precompute the right environments once and
  sweep a running left environment across columns.
- For each anchor column `c`, evaluate all bucketed operators sharing
  `(r, dr, c)`, grouped by `dc`, using the same environment window.
- For plaquette terms, use the same static outcome table as `transition`; for
  every nonzero outcome, rebuild only changed local bricks and contract the
  affected window against the reused environments.
- Multiply each off-diagonal contribution by the precomputed matrix element and
  accumulate local estimates for the Hamiltonian and observables.

Gradients:

- The defect-network sweep produces a gradient for the active block at each
  site.
- For `full_gradient=True`, scatter into the full
  `(n_blocks, D_u,D_d,D_l,D_r)` tensor.
- For sliced mode, concatenate active block gradients and emit
  `active_slice_indices = block_id` repeated `params_per_site[r,c]` times.

---

## 9. Local Terms: `su2_gi/local_terms.py`

Local terms follow the repository's existing operator convention:

- Define operator dataclasses as pytrees when they carry dynamic arrays.
- Register physical supports with `support_span.dispatch`.
- Register model-specific evaluation windows with `SU2GIPEPS.eval_span`, falling
  back to `support_span` unless a larger reuse window is beneficial.
- Register typed `_eval_term.dispatch` overloads for SU(2) term types.

### 9.1 `HorizontalLinkCasimirTerm` / `VerticalLinkCasimirTerm`

Diagonal electric term on one oriented lattice link:

```text
g_E * j(j+1)
```

No PEPS contraction is needed. Use typed horizontal and vertical classes rather
than a string direction field, matching the repository preference for typed
dispatch over manual naming branches. The terms are self-contained diagonal
link operators: construct them from the model's `SU2` metadata so each term
stores the static `j(j+1)` diagonal array and can be bucketed by
`merge_operators` as a diagonal term.

### 9.2 `PlaquetteSU2Term`

Off-diagonal transition term:

```python
@dataclass(frozen=True)
class PlaquetteSU2Term:
    row: int
    col: int
```

The term does not compute CG coefficients at runtime. It indexes the static
plaquette table built by `group.py`.

Evaluation returns the contribution:

```text
sum_out matrix_element(out, in) * Psi(out) / Psi(in)
```

Physical support and evaluation span are both `2x2` for the pure-gauge
plaquette term:

```python
@support_span.dispatch
def support_span(_: PlaquetteSU2Term) -> tuple[int, int]:
    return 2, 2
```

If future SU(2) terms have physical support smaller than their cheapest
evaluation window, follow the `BlockadePEPS` pattern: keep `support_span` equal
to physical support and override `SU2GIPEPS.eval_span` for the contraction
window.

Transition uses the same table for proposal outcomes.

The implementation is split deliberately:

- `PlaquetteLinkTransitions` stores dense static candidate link outputs in
  plaquette order `(top, right, bottom, left)`, obtained by fusing each link
  irrep with the fundamental.
- Candidate sample generation combines `PlaquetteLinkTransitions` with
  `PureGaugeTables.block_id_lookup` to mask invalid intertwiner choices and
  preserve Gauss law.
- Magnetic matrix elements are a separate static table. Do not substitute
  placeholder unit weights; add the recoupling coefficients before enabling
  plaquette local-energy evaluation.

---

## 10. Efficiency Invariants

Hard requirements:

1. Store only valid vertex blocks. No parameter masks.
2. Sample intertwiners. No hidden `sum_iota w_iota A_iota` in row bricks.
3. Runtime PEPS and boundary tensors carry reduced dimensions only.
4. Magnetic-index and recoupling data appear only in static operator tables.
5. One sampled site selects one tensor block.
6. Static metadata is built once in `SU2GIPEPS.__init__` and closed over by
   `kernels.py`.
7. Block-sparse contractions execute static schedules. No dictionaries in hot
   JAX paths.
8. Operator evaluation reuses environments by sweep scope: one top/bottom pair,
   one right-env suffix table per `(r, dr)`, and one running left env per
   column sweep.
9. Compression is QR-only for the MVP.
10. The Abelian GI implementation remains untouched.

Expected `j_max = 1/2` pure-gauge storage:

- Bulk vertex allowed blocks: all valid four-leg singlet fusion states,
  including distinct intertwiners.
- Tensor storage per bulk site: `n_blocks * D^4`.
- Tensor storage per boundary site: `n_blocks * D^k`, where `k` is the number
  of active lattice legs.
- Sliced-QGT active parameters per sample per site:
  `params_per_site[r,c] = D^k`.

This is the same computational pattern as Abelian GI-PEPS: total stored slices
increase with the number of allowed local physical states, but each sample uses
only one slice per site.

---

## 11. Implementation Order

### Current status checkpoint

Finished in the current implementation:

- Full pure-gauge spin-network samples: horizontal links, vertical links, and
  vertex intertwiners.
- Unbiased random SU(2) tensor initialization with no forced electric-vacuum
  block offset.
- Valid batched initial configurations via static plaquette-table warmup from
  the electric-vacuum sample.
- Boundary-aware vertex block tables with JAX-array lookup metadata.
- SU(2) Clebsch-Gordan/intertwiner primitives and static plaquette matrix
  tables.
- Diagonal electric-energy terms evaluated directly from sampled links.
- Plaquette magnetic local-energy evaluation from static matrix tables.
- Plaquette Metropolis transition sweep using static proposal weights and
  Hastings correction.
- Kernel dispatch, cache turnover, sliced active-block gradients, static and
  time-dependent coefficients, and basic JIT smoke coverage.
- Generic `BlockEvalSchedule` wiring for transition-term estimate sweeps.
- Generic `make_mc_sampler` integration for SU(2) kernels.
- Public `vmc.peps` exports and dispatch registration for SU(2).
- Exact 2x2 detailed-balance and plaquette-transition graph connectivity tests.
- Sliced-QGT parity against full dense Jacobians on the 2x2 pure-gauge system.
- Exact 2x2 Hamiltonian-matrix local-energy comparison for electric plus
  plaquette terms.
- Exact 3x3 Hamiltonian-matrix local-energy comparison over the full
  `j_max = 1/2` spin-network basis.
- Exact 2x2 ED ground-state local-energy comparison.
- TDVP-driver run-level energy check on the exact 2x2 ED ground state.
- Imaginary-time SR optimization benchmark on 2x2 with `D = 2`, `chi = D^2`,
  non-collapsed chain diagnostics, and comparison against the ED ground energy.
- Exact 3x3 ED ground-energy baseline for the full `j_max = 1/2` basis.
- Slow imaginary-time SR optimization benchmark on 3x3 with `D = 2`,
  `chi = D^2`, non-collapsed chain diagnostics, and comparison against the ED
  ground energy.

Still unfinished:

- Matter extensions.

Deferred until virtual bonds carry explicit sector labels:

- Generic sector-block boundary-MPS contraction/compression beyond selected
  dense row-MPO blocks.

### Phase 1: SU(2) tables

- Add `su2_gi/group.py`.
- Implement integer-irrep `SU2`.
- Implement CG and fusion tests.
- Implement canonical intertwiner enumeration.
- Implement vertex block tables and inverse block lookup.
- Add block-count tests for corner, edge, and bulk vertices.

### Phase 2: Model skeleton

- Add `su2_gi/model.py`.
- Implement `SU2GIPEPSConfig`.
- Implement `SU2GIPEPS` tensor initialization.
- Implement sample flatten/unflatten.
- Implement all-zero electric-vacuum sample.
- Implement random valid sample generation through table-driven plaquette
  warmup or simple valid enumeration for small lattices.

### Phase 3: Generic block-sparse executor

- Add `common/block_sparse.py`.
- Implement stacked-block gather/scatter helpers.
- Implement block-sparse bottom contraction.
- Implement block-sparse row-MPO application.
- Implement block-sparse right/left environment updates.
- Implement `BlockEvalSchedule` construction from `BucketedOperators`, grouped
  by `(row, dr, col, dc)`.
- Implement the generic operator-evaluation sweep that reuses top, bottom,
  right, and left environments across all terms in the same window scope.
- Implement variational QR compression.
- Test against a dense unfolded contractor on small layouts.

### Phase 4: Compatibility amplitude path

- Implement row-brick assembly as active block gathers.
- Keep `compat.py` and `SU2GIPEPS.apply` as compatibility/debug surfaces only.
- Compare compatibility amplitudes against an explicitly unfolded dense
  spin-network contraction on `2x2` and `3x3`.
- Keep production VMC contractions in `kernels.py`, backed by generic
  block-sparse row-MPO application and Variational QR compression.

### Phase 5: Local terms and transition tables

- Add typed `HorizontalLinkCasimirTerm` and `VerticalLinkCasimirTerm`.
- Add `PlaquetteSU2Term`.
- Build plaquette input/output tables.
- Verify Hermiticity of plaquette matrix elements.
- Verify every proposed output preserves Gauss law.

### Phase 6: Kernels

- Add `su2_gi/kernels.py`.
- Register typed `build_mc_kernels` dispatch.
- Implement `init_cache`.
- Implement pure-gauge plaquette `transition`.
- Implement `estimate` through the generic block-sparse evaluation scheduler.
- Add the one-line driver import for dispatch registration.

### Phase 7: VMC and QGT validation

- Verify sliced gradients against full gradients.
- Verify `SlicedJacobian` QGT against dense Jacobian on tiny lattices.
- Verify detailed balance by exact enumeration on `2x2`.
- Verify plaquette-sweep ergodicity inside the open-boundary singlet sector.
- Compare `2x2` and `3x3` ground-state estimates against ED.

### Phase 8: Matter extension

Only after pure gauge passes:

- Extend config with `phys_dim` and `charge_of_site_twice`.
- Extend vertex block tables with matter charge.
- Add matter hopping term tables.
- Add matter transition sweeps.
- Add ED tests for the smallest bosonic matter systems.

---

## 12. Test Plan

### 12.1 Group tests

- `test_su2_fusion_rules_jmax_half`
- `test_su2_fusion_rules_jmax_one`
- `test_su2_cg_unitarity`
- `test_su2_cg_condon_shortley_sympy`
- `test_su2_casimir`

### 12.2 Intertwiner and block-table tests

- `test_su2_intertwiner_orthonormal`
- `test_su2_bulk_block_count_jmax_half`
- `test_su2_edge_block_count_jmax_half`
- `test_su2_corner_block_count_jmax_half`
- `test_su2_block_lookup_roundtrip`
- `test_su2_invalid_iota_rejected`

### 12.3 Block-sparse executor tests

- `test_block_sparse_contract_bottom_matches_dense`
- `test_block_sparse_apply_mpo_matches_dense`
- `test_block_sparse_window_contract_matches_dense`
- `test_block_sparse_eval_schedule_reuses_envs_by_span`
- `test_block_sparse_variational_qr_matches_dense_reference`

### 12.4 Amplitude tests

- `test_su2_amplitude_all_zero_vacuum_nonzero`
- `test_su2_amplitude_matches_dense_unfolded_2x2`
- `test_su2_amplitude_matches_dense_unfolded_3x3`
- `test_su2_gauge_rotation_leaves_amplitude_invariant`

### 12.5 Operator tests

- `test_su2_link_casimir_diagonal`
- `test_su2_plaquette_table_preserves_gauss_law`
- `test_su2_plaquette_table_hermitian`
- `test_su2_plaquette_matrix_elements_match_sympy_reference`

### 12.6 Sampling tests

- `test_su2_transition_preserves_gauss_law`
- `test_su2_plaquette_hastings_ratio`
- `test_su2_detailed_balance_exact_2x2`
- `test_su2_plaquette_ergodicity_2x2`

### 12.7 Gradient and VMC tests

- `test_su2_sliced_gradient_matches_full_gradient`
- `test_su2_sliced_qgt_matches_dense_qgt`
- `test_su2_pure_gauge_2x2_energy_vs_ed`
- `test_su2_pure_gauge_3x3_energy_vs_ed`

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| CG or phase convention mismatch | Pin Condon-Shortley convention and compare against SymPy. |
| Intertwiner basis mismatch between tensors and operators | Build both block tables and plaquette tables from the same canonical fusion tree. |
| Plaquette table key becomes too large | Start with `j_max = 1/2`; use static branch tables; profile before `j_max = 1`. |
| Block-sparse executor becomes a hidden tensor library | Keep it schedule-driven and minimal: stacked dense blocks plus static integer schedules. |
| Runtime recompilation from dynamic shapes | All block counts, schedules, and outcome structures are built before kernel dispatch. |
| Sliced-QGT mismatch | Preserve exactly one active block per site per sample; test sliced vs full gradients. |
| Boundary-MPS sector bugs | Compare every block-sparse contraction against dense unfolded references on small systems. |

---

## 14. Naming Rules

- Use `symmetric` only for group-theory semantics: irreps, fusion, CG,
  intertwiners, recoupling, and Gauss-law block construction.
- Use `block_sparse` only for storage and execution: stacked dense blocks,
  schedules, contractions, QR, compression, and environments.
- Avoid function names ending in `_su2` inside generic code. Put SU(2)-specific
  logic behind typed metadata and dispatch.
- Do not add adapter layers whose only purpose is renaming dense APIs. Add a
  small typed layout or schedule only when it removes ambiguity or prevents
  repeated manual indexing logic.

---

## 15. Success Criteria

The MVP is successful when:

1. A pure-gauge `SU2GIPEPS` can sample valid spin-network states on open
   lattices.
2. Amplitudes match an independent dense unfolded spin-network contraction.
3. Plaquette transitions preserve Gauss law and satisfy detailed balance.
4. Sliced gradients match full gradients.
5. `2x2` and `3x3`, `j_max = 1/2`, pure-gauge energies match ED within sampling
   error.
6. No existing Abelian GI behavior changes.
