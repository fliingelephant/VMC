# SU(2) Gauge-Invariant PEPS - Full Intertwiner Design

Date: 2026-04-24
Status: Pure-gauge SU(2) MVP implemented; non-Abelian extensions in progress
Supersedes: `docs/superpowers/specs/2026-04-18-su2-gi-peps-design.md`

This design extends the Abelian GI-PEPS idea to SU(2) by sampling the full
spin-network basis: matter labels, link irreps, and vertex intertwiners. The
core principle is the same as the Abelian implementation: a physical sample
selects exactly one allowed local tensor block at each vertex. No invalid
blocks are stored, no masks are applied to parameters, and no hidden
intertwiner sum creates redundant variational directions.

The implementation split is:

- `src/vmc/gauge_groups/su2.py`: SU(2) symmetry semantics and static metadata.
- `src/vmc/gauge_groups/su3.py`: fundamental-truncated SU(3) backend using
  exact low-dimensional invariant tensors.
- `src/vmc/peps/common/block_sparse.py`: generic scheduled block-sparse linear
  algebra.
- `src/vmc/peps/non_abelian_gi/`: the generic non-Abelian GI-PEPS model, local
  terms, kernels, and table contract.

The first implementation gate is pure gauge SU(2) Yang-Mills with open
boundaries. That gate is now implemented and benchmarked against exact
diagonalization on `2x2` and `3x3` `j_max = 1/2` systems. The remaining work is
extension work: matter, larger/non-fundamental group truncations, and optional
sector-block boundary-MPS machinery if heterogeneous virtual sectors become
necessary.

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

For the first matter extension, matter is represented by reduced local matter
states, not by sampled magnetic/color components:

```text
sample = (matter_states, h_links, v_links, intertwiners)
matter_state -> matter irrep
```

This is the cleanest VMC representation because each sampled vertex state maps
to one tensor block:

```text
pure gauge: (j_l, j_u, j_r, j_d, iota) -> block_id
matter:     (matter_state, j_l, j_u, j_r, j_d, iota) -> block_id
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
- Block-sparse storage for vertex tensors over sampled spin-network blocks.
- Dense boundary-MPS contraction of the selected row-MPO blocks with QR-based
  `Variational` compression.
- Static metadata closed over in `kernels.py`.
- Typed dispatch for model/operator/kernel behavior.
- Compatibility with the current sampler, TDVP driver, SR preconditioner, and
  sliced QGT path.

### 2.2 Next extension target

- Reduced matter-basis support through config fields:
  `phys_dim`, `matter_irreps`, `matter_numbers`, and `particle_number`.
- First matter case: two reduced states with
  `matter_irreps=(0, 1)` and `matter_numbers=(0, 1)`, i.e. empty and one
  fundamental matter excitation in `j_twice` labels.
- Fixed total `particle_number` and number-conserving hopping only.
- Matter-aware vertex block tables, plaquette transition tables, and horizontal
  and vertical hopping transition tables.
- Row-sparse connected-outcome operator tables before adding matter, so the
  matter extension does not inflate dense four-site transition arrays.

### 2.3 Deferred

- Fermionic matter.
- Higgs-like matter pair creation/annihilation terms.
- Full SU(3) beyond the fundamental truncation.
- Other generic non-Abelian or finite-group backends.
- Heterogeneous reduced dimensions `D_j` and `chi_j`.
- True sector-block boundary-MPS contraction/compression.
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

For a matter-aware basis, each vertex also carries one reduced matter-state
label:

```text
|{q_vertex}, {j_link}, {iota_vertex}>
```

The matter-state label is a local reduced basis index. It is not a sampled color
or magnetic index. Static metadata maps it to a matter irrep and particle number:

```text
matter_irreps[q]   # irrep label, e.g. SU(2) j_twice
matter_numbers[q]  # integer conserved number label
```

where each `iota_vertex` labels an orthonormal basis vector in:

```text
pure gauge: Inv(j_l x j_u x j_r* x j_d* x Q_x*)
matter:     Inv(j_l x j_u x j_r* x j_d* x matter_irrep[q]* x Q_x*)
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

The first matter extension adds a diagonal mass/number term and
number-conserving nearest-neighbor hopping. A hopping term changes the two
endpoint matter states, the connecting link irrep, and the two endpoint
intertwiners. It is evaluated from static connected-outcome tables, not by
sampling magnetic indices.

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
`(j_l, j_u, j_r, j_d, iota)` for pure gauge and
`(matter_state, j_l, j_u, j_r, j_d, iota)` when matter is enabled.

---

## 4. Module Layout

```text
src/vmc/peps/common/
  block_sparse.py      # generic scheduled dense-block execution

src/vmc/peps/non_abelian_gi/
  builders.py          # typed static metadata builder dispatch
  tables.py            # group-independent sampled block table containers
  model.py             # NonAbelianGIPEPS config/module/sample helpers
  local_terms.py       # generic link Casimir and plaquette term types
  contraction.py       # active block row-MPO assembly and apply path
  kernels.py           # generic sampled-block build_mc_kernels dispatch

src/vmc/gauge_groups/
  __init__.py
  su2.py               # SU(2), CG, intertwiners, and typed table builders
  su3.py               # SU(3) p+q<=1 backend and typed table builders
```

No existing Abelian GI code is changed.

---

## 5. Symmetry Layer: `gauge_groups/su2.py`

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

and, when matter is enabled, local matter state `q`, enumerate all fusion paths
that end in target `Q_x`. Use one canonical fusion tree throughout the
implementation and tests. For pure gauge:

```text
(j_l, j_u) -> j_m
(j_r, j_d) -> j_n
(j_m, j_n) -> Q_x
```

For matter, append the matter irrep to the same canonical tree rather than
sampling its magnetic index:

```text
(j_l, j_u) -> j_m
(j_r, j_d) -> j_n
(j_m, j_n) -> j_p
(j_p, matter_irrep[q]) -> Q_x
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
    VertexBlock(matter_state, j_l, j_u, j_r, j_d, iota_id)
```

Only valid blocks are stored. Boundary legs outside the lattice are fixed to
`j = 0`, so corners and edges have fewer blocks. In pure gauge,
`matter_state = 0` with matter irrep `0`, so the table naturally reduces to the
current pure-gauge block table.

Also build an inverse lookup:

```text
block_id[r, c, matter_state, j_l, j_u, j_r, j_d, iota_slot] -> int
```

This lookup is static metadata. Runtime row-brick assembly performs one
integer lookup and one tensor gather per site.

Static numeric metadata that is indexed inside kernels is stored as dense JAX
arrays, not NumPy arrays or Python tuples. `VertexBlock` records and lookup
dicts remain Python build-time metadata; `block_id_lookup`, plaquette outcome
indices, matrix elements, proposal weights, and proposal norms are JAX arrays
closed over by kernels.

### 5.4 Intertwiner slots

The sampled `iota` is local to the current matter/link tuple. Runtime samples
store an integer slot:

```text
iotas[r, c] in [0, n_iota(r, c, matter_state, j_l, j_u, j_r, j_d))
```

Changing a link can change the valid intertwiner count at neighboring vertices.
Changing a matter state can do the same. Therefore every off-diagonal proposal
table must update all affected matter labels, link irreps, and `iota` slots.

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

The clean target representation is row-sparse over connected outcomes:

```text
starts[input_id]
counts[input_id]
max_count                         # static loop bound for this table/signature
output_block_ids[start:start+count]
matrix_elements[start:start+count]
proposal_weights[start:start+count] # default abs(matrix_element)^2
proposal_norm[input_id]             # sum proposal_weights for that row
```

Sample updates are derived from `output_block_ids` and closed-over block
property arrays, not stored redundantly in the transition table:

```text
matter_state_by_block
j_l_by_block, j_u_by_block, j_r_by_block, j_d_by_block
iota_by_block
```

Runtime kernels still loop over a static bound:

```text
for k in range(max_count):
    valid = k < counts[input_id]
    out = output_block_ids[starts[input_id] + k]
```

The current pure-gauge implementation may retain a dense padded
`max_outputs` table as a compatibility representation for small validation
systems, but new matter-aware tables should use the row-sparse connected-outcome
contract. This avoids materializing impossible four-corner block products.

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

### 5.6 Hopping transition tables

Number-conserving matter hopping is also table-driven. For a horizontal or
vertical link, the table input is the pair of active endpoint block ids and any
static signature data needed to verify the shared connecting link. A connected
outcome stores:

```text
new endpoint matter states
new connecting link irrep
new endpoint iota slots
new endpoint block ids
matrix_element = <out|matter_hop|in>
```

External links around the two endpoint vertices are fixed by the current
sample and are not proposed. They still determine which endpoint intertwiners
are valid, so hopping outcomes must be built from the same matter-aware vertex
block table as plaquette outcomes. Hopping tables use the same row-sparse
`starts/counts` connected-outcome representation as plaquette tables.

### 5.7 Boundary-sector schedules

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

## 7. Model: `non_abelian_gi/model.py`

### 7.1 Config

Pure-gauge MVP config:

```python
@dataclass(frozen=True)
class NonAbelianGIPEPSConfig:
    shape: tuple[int, int]
    gauge_group: Any
    D: int
    chi: int
    phys_dim: int = 1
    matter_irreps: tuple[int, ...] = (0,)
    matter_numbers: tuple[int, ...] = (0,)
    particle_number: int = 0
    target_charge: Any = 0
    dtype: Any = jnp.complex128
```

For SU(2), set `gauge_group=SU2(j_max_twice=j_max_twice)`. SU(2)-specific
fusion and plaquette metadata are registered through typed table-builder
dispatch, not through an `SU2GIPEPS` model wrapper.

Matter is described by a reduced local basis. `phys_dim` is the number of
matter states, `matter_irreps[q]` is the irrep carried by state `q`, and
`matter_numbers[q]` is the conserved integer number label. Pure gauge is the
special case `phys_dim=1`, `matter_irreps=(0,)`, `matter_numbers=(0,)`, and
`particle_number=0`; it should not require a separate model class.

The first matter target is:

```python
phys_dim = 2
matter_irreps = (0, 1)    # SU(2) j_twice labels: singlet, fundamental
matter_numbers = (0, 1)
```

with fixed total `particle_number`. For zero background charge and open singlet
boundaries, odd `particle_number` is invalid for fundamental SU(2) matter and
must be rejected during configuration/table construction.

### 7.2 Parameters

At each site:

```text
tensors[r][c].value.shape == (n_blocks[r, c], D_u, D_d, D_l, D_r)
```

where `n_blocks[r, c]` is the number of valid
`(matter_state, j_l, j_u, j_r, j_d, iota)` blocks at that position. The tensor
axis order after `block_id` follows the existing PEPS convention:
`(up, down, left, right)`. Open-boundary legs have reduced dimension `1`;
active internal legs have reduced dimension `D`.

There is no dense matter axis:

```text
do not use: A[matter_state, link_config, iota, ...]
use:        A[allowed_block_id, ...]
```

All blocks are initialized with scaled complex Gaussian noise. No physical
configuration is given a hard-coded variational bias.

### 7.3 Sample representation

For the uniform representation:

```text
matter:  shape (n_rows, n_cols), int32 reduced matter-state ids
h_links: shape (n_rows, n_cols - 1), int32 j_twice ids
v_links: shape (n_rows - 1, n_cols), int32 j_twice ids
iotas:   shape (n_rows, n_cols), int32 local intertwiner slot
```

For pure gauge, `matter` is the all-zero array and may be omitted from public
helpers for backwards compatibility, but kernels should operate on the uniform
matter-aware representation once matter support is added.

Flatten in a deterministic order:

```text
flatten_sample(matter, h_links, v_links, iotas) -> int32 vector
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
block_id(r, c, matter_state, j_l, j_u, j_r, j_d, iota)
```

This matches the existing `SlicedJacobian` assumption: one active slice per site.

---

## 8. Kernels: `non_abelian_gi/kernels.py`

`kernels.py` follows the current Abelian pattern: the dispatch function captures
all static metadata and returns `init_cache`, `transition`, and `estimate`.

### 8.1 Dispatch

Register:

```python
@build_mc_kernels.dispatch
def build_mc_kernels(
    model: NonAbelianGIPEPS,
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
- `strategy`
- `bucketed_terms` from `merge_operators`
- `coeff_structure`
- scalar config values needed in kernels, decomposed from `config`
- dense JAX arrays from `tables`
- block property arrays:
  `matter_state_by_block`, `j_l/u/r/d_by_block`, and `iota_by_block`
- vertex block layouts
- boundary-MPS layouts
- block apply schedules
- bottom/window/env contraction schedules
- block evaluation schedules grouped by `(row, dr, col, dc)`
- plaquette outcome tables
- horizontal and vertical hopping outcome tables when matter is enabled
- static matter irrep and number arrays
- `params_per_site`
- `sliced_dims`

The builder phase may use Python dataclasses, records, dictionaries, and group
objects. The returned kernels must not inspect those objects in hot paths.
Before `build_mc_kernels` returns, every metadata item used inside JAX kernels
must be converted to closed-over scalars, tuples with static length, or JAX
arrays. In particular, kernels should not read nested `config` fields, call
group methods, index Python dicts, or walk `VertexBlock` records.

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

The pure-gauge transition is a sequential plaquette sweep:

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

With matter enabled, the transition adds sequential horizontal and vertical
number-conserving hopping sweeps after the plaquette sweep. A hopping proposal
uses its static connected-outcome table to update exactly the two endpoint
matter states, the connecting link irrep, and the two endpoint `iota` slots.
The fixed `particle_number` sector is therefore preserved by construction.

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
- For hopping terms, use the horizontal or vertical connected-outcome table;
  every nonzero outcome rebuilds only the two endpoint bricks and contracts the
  corresponding one-row or two-row window against the reused environments.
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

## 9. Local Terms: `non_abelian_gi/local_terms.py`

Local terms follow the repository's existing operator convention:

- Define operator dataclasses as pytrees when they carry dynamic arrays.
- Register physical supports with `support_span.dispatch`.
- Register model-specific evaluation windows with `NonAbelianGIPEPS.eval_span`, falling
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

### 9.2 `PlaquetteTerm`

Off-diagonal transition term:

```python
@dataclass(frozen=True)
class PlaquetteTerm:
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
def support_span(_: PlaquetteTerm) -> tuple[int, int]:
    return 2, 2
```

If future SU(2) terms have physical support smaller than their cheapest
evaluation window, follow the `BlockadePEPS` pattern: keep `support_span` equal
to physical support and override `NonAbelianGIPEPS.eval_span` for the contraction
window.

Transition uses the same table for proposal outcomes.

The implementation is split deliberately:

- `PlaquetteLinkTransitions` stores dense static candidate link outputs in
  plaquette order `(top, right, bottom, left)`, obtained by fusing each link
  irrep with the fundamental.
- Candidate sample generation combines `PlaquetteLinkTransitions` with
  the spin-network table `block_id_lookup` to mask invalid intertwiner choices and
  preserve Gauss law.
- Magnetic matrix elements are a separate static table. Do not substitute
  placeholder unit weights; add the recoupling coefficients before enabling
  plaquette local-energy evaluation.

For the matter extension, plaquette outcomes should be read through the generic
row-sparse connected-outcome table. Matter labels are spectators for the
plaquette operator, but they still affect the valid corner intertwiners and
therefore the output block ids.

### 9.3 Matter number/mass term

The first matter diagonal term is a one-site number term:

```text
m * matter_numbers[matter_state]
```

No PEPS contraction is needed. The term reads the active matter state from the
sample or, equivalently, a precomputed `matter_number_by_block` static array
indexed by active block id. It remains a `DiagonalOperator` so
`merge_operators` can bucket it with other diagonal terms.

### 9.4 Horizontal/Vertical matter hopping terms

Number-conserving hopping is represented by typed horizontal and vertical term
classes, not by a string direction field:

```python
@dataclass(frozen=True)
class HorizontalMatterHoppingTerm:
    row: int
    col: int

@dataclass(frozen=True)
class VerticalMatterHoppingTerm:
    row: int
    col: int
```

Physical supports are:

```python
@support_span.dispatch
def support_span(_: HorizontalMatterHoppingTerm) -> tuple[int, int]:
    return 1, 2

@support_span.dispatch
def support_span(_: VerticalMatterHoppingTerm) -> tuple[int, int]:
    return 2, 1
```

The default evaluation spans are the same as the physical supports. If a later
non-Abelian matter term is cheaper in a larger shared window, override
`NonAbelianGIPEPS.eval_span` by typed dispatch without changing
`support_span`.

Hopping evaluation indexes the static connected-outcome table and computes:

```text
sum_out matrix_element(out, in) * Psi(out) / Psi(in)
```

The hopping matrix elements are built from the same group convention as
plaquette matrix elements. For the initial two-state matter basis, the only
matter-state transitions are `1,0 -> 0,1` and `0,1 -> 1,0`; there are no
fermionic signs and no pair-creation terms.

---

## 10. Efficiency Invariants

Hard requirements:

1. Store only valid vertex blocks. No parameter masks.
2. Sample intertwiners. No hidden `sum_iota w_iota A_iota` in row bricks.
3. Matter labels are reduced basis states. Do not sample magnetic/color
   components.
4. Runtime PEPS and boundary tensors carry reduced dimensions only.
5. Do not add a dense matter or physical axis to non-Abelian tensors; matter
   states are part of the allowed `block_id`.
6. Magnetic-index and recoupling data appear only in static operator tables.
7. One sampled site selects one tensor block.
8. Static metadata is built once in `NonAbelianGIPEPS.__init__` and closed over by
   `kernels.py` as scalars, static-length tuples, or JAX arrays. Python group
   objects, config objects, dicts, and `VertexBlock` records stay out of hot
   kernels.
9. Block-sparse contractions execute static schedules. No dictionaries in hot
   JAX paths.
10. Operator evaluation reuses environments by sweep scope: one top/bottom pair,
   one right-env suffix table per `(r, dr)`, and one running left env per
   column sweep.
11. Off-diagonal operator tables store only connected outcomes when matter is
    enabled; dense padded tables are temporary pure-gauge compatibility data.
12. Row-sparse transition tables carry a static `max_count` loop bound per
    table/signature, so storage is sparse without dynamic JAX control flow.
13. Compression is QR-only for the MVP.
14. The Abelian GI implementation remains untouched.

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

With matter, storage becomes:

```text
n_blocks(r,c) * D^k
```

where `n_blocks(r,c)` already includes all allowed
`(matter_state, links, iota)` tuples. It must not become:

```text
phys_dim * n_link_configs * max_iotas * D^k
```

---

## 11. Implementation Order

### Current status checkpoint

Implemented for pure-gauge SU(2):

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
- Selected-block dense row-MPO contraction with QR-based `Variational`
  compression.
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

Implemented for fundamental-truncated SU(3):

- Generic `NonAbelianGIPEPS` construction with `gauge_group=SU3(max_weight_sum=1)`.
- Fundamental and antifundamental invariant-tensor backend for `p + q <= 1`.
- Pure-gauge sampling, electric and plaquette terms, local-energy evaluation,
  and plaquette transitions through the same generic table contract.
- Exact 2x2 ED ground-energy baseline and imaginary-time SR optimization
  benchmark against ED.

Current representation and efficiency boundary:

- Vertex tensors are block-sparse over sampled spin-network physical states.
- A Monte Carlo sample selects one dense degeneracy block per site.
- Boundary contraction currently acts on those selected dense row-MPO blocks.
- Plaquette magnetic terms currently use dense per-plaquette static matrix
  tables; the next implementation step is to migrate transition evaluation to
  row-sparse connected-outcome tables before adding matter.

Still unfinished:

- Commit a SU(3) 3x3 ED/optimization benchmark if that becomes a validation
  target.
- Replace dense per-plaquette matrix tables with row-sparse connected-outcome
  transition tables.
- Reuse plaquette tables by plaquette-space signature instead of rebuilding
  equivalent boundary/interior cases independently.
- Add fixed-`particle_number` reduced matter-basis support.
- Extend SU(3) beyond the fundamental truncation `p + q <= 1`.
- Add heterogeneous `D_j`, heterogeneous `chi_j`, and true sector-block
  boundary-MPS contraction/compression if non-uniform virtual sectors are
  needed.

### Phase 1: SU(2) tables

- Add `gauge_groups/su2.py`.
- Implement integer-irrep `SU2`.
- Implement CG and fusion tests.
- Implement canonical intertwiner enumeration.
- Implement vertex block tables and inverse block lookup.
- Add block-count tests for corner, edge, and bulk vertices.

### Phase 2: Model skeleton

- Add `non_abelian_gi/model.py`.
- Implement `NonAbelianGIPEPSConfig`.
- Implement `NonAbelianGIPEPS` tensor initialization.
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

### Phase 4: Active-block amplitude path

- Implement row-brick assembly as active block gathers.
- Compare compatibility amplitudes against an explicitly unfolded dense
  spin-network contraction on `2x2` and `3x3`.
- Keep production VMC contractions in `non_abelian_gi/kernels.py`, backed by generic
  block-sparse row-MPO application and Variational QR compression.

### Phase 5: Local terms and transition tables

- Add typed `HorizontalLinkCasimirTerm` and `VerticalLinkCasimirTerm`.
- Add `PlaquetteTerm`.
- Build plaquette input/output tables.
- Convert off-diagonal transition tables to the generic row-sparse
  connected-outcome contract.
- Verify Hermiticity of plaquette matrix elements.
- Verify every proposed output preserves Gauss law.

### Phase 6: Kernels

- Add `non_abelian_gi/kernels.py`.
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

After pure-gauge validation:

- Extend config with `phys_dim`, `matter_irreps`, `matter_numbers`, and fixed
  `particle_number`.
- Use pure gauge as the uniform `phys_dim=1`, `matter_irreps=(0,)` case rather
  than a separate model branch.
- Extend vertex block tables with `matter_state` and matter irrep.
- Extend plaquette transition tables so matter is a spectator but valid
  intertwiners are matter-aware.
- Add row-sparse horizontal and vertical matter hopping tables.
- Add diagonal matter number/mass terms.
- Add matter transition sweeps that preserve fixed `particle_number`.
- Add ED tests for the smallest two-state matter systems.

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
- `test_su2_matter_block_lookup_roundtrip`
- `test_su2_matter_odd_particle_number_rejected`

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
- `test_su2_sparse_transition_table_matches_dense_plaquette_reference`
- `test_su2_matter_hopping_table_preserves_gauss_law`
- `test_su2_matter_hopping_matrix_elements_match_dense_reference`

### 12.6 Sampling tests

- `test_su2_transition_preserves_gauss_law`
- `test_su2_plaquette_hastings_ratio`
- `test_su2_detailed_balance_exact_2x2`
- `test_su2_plaquette_ergodicity_2x2`
- `test_su2_matter_transition_preserves_particle_number`
- `test_su2_matter_transition_preserves_gauss_law`

### 12.7 Gradient and VMC tests

- `test_su2_sliced_gradient_matches_full_gradient`
- `test_su2_sliced_qgt_matches_dense_qgt`
- `test_su2_pure_gauge_2x2_energy_vs_ed`
- `test_su2_pure_gauge_3x3_energy_vs_ed`
- `test_su2_matter_sliced_gradient_matches_full_gradient`
- `test_su2_matter_2x2_energy_vs_ed`

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| CG or phase convention mismatch | Pin Condon-Shortley convention and compare against SymPy. |
| Intertwiner basis mismatch between tensors and operators | Build both block tables and plaquette tables from the same canonical fusion tree. |
| Plaquette table key becomes too large | Start with `j_max = 1/2`; use static branch tables; profile before `j_max = 1`. |
| Dense transition tables become too large with matter | Use row-sparse connected-outcome tables before enabling matter kernels. |
| Invalid global matter sector | Validate fixed `particle_number` against boundary/background charges before sampling. |
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

The pure-gauge SU(2) MVP is successful when:

1. A pure-gauge `NonAbelianGIPEPS` with `gauge_group=SU2(...)` can sample valid
   spin-network states on open lattices.
2. Amplitudes match an independent dense unfolded spin-network contraction.
3. Plaquette transitions preserve Gauss law and satisfy detailed balance.
4. Sliced gradients match full gradients.
5. `2x2` and `3x3`, `j_max = 1/2`, pure-gauge energies match ED within sampling
   error.
6. No existing Abelian GI behavior changes.

The generic non-Abelian extension is additionally validated when:

1. Fundamental `gauge_group=SU3(max_weight_sum=1)` can sample valid pure-gauge
   spin-network states.
2. SU(3) `2x2` pure-gauge ED and optimization benchmarks pass through the same
   `NonAbelianGIPEPS` model, local-term, kernel, sampler, and TDVP paths.
3. Any future SU(3) 3x3 benchmark or larger truncation is added as an explicit
   extension target, not treated as part of the completed SU(2) MVP.

The first matter extension is successful when:

1. `NonAbelianGIPEPS` accepts a reduced two-state matter basis through
   `phys_dim`, `matter_irreps`, `matter_numbers`, and fixed `particle_number`
   without introducing a dense tensor matter axis or a separate matter model
   wrapper.
2. Matter-aware block tables store only valid
   `(matter_state, links, iota)` local blocks.
3. Plaquette and hopping local-energy evaluation use row-sparse connected
   outcomes while reusing the same row/span environments as pure gauge.
4. Horizontal and vertical hopping transitions preserve fixed `particle_number`
   and local Gauss law including updated endpoint intertwiners.
5. Small-lattice matter energies and sliced gradients match exact dense
   references.
