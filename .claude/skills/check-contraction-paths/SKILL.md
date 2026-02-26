---
name: check-contraction-paths
description: Verify tensor contraction paths and identify inefficiencies in PEPS contraction code. Checks both standalone einsums and implicit multi-step sequences.
---

# Check Contraction Paths

Verify that hand-tuned tensor contraction paths are optimal, and identify implicit multi-step sequences where shared intermediates or reordering could reduce cost.

## Step 0: Gather context

Before any analysis, identify what you need to know and **ask the user**. Never assume silently — if you are uncertain about something, ask.

Examples of things to ask (not exhaustive — ask whatever you need):

1. **Reference papers.** Invoke `/read-notes` to list papers in `/notes/` and ask the user which to read. Tell it to extract contraction patterns, asymptotic costs, and environment reuse recommendations.
2. **Code scope.** Which parts of the codebase should be analyzed? (e.g., standard PEPS only, include TDVP driver, include GI/Blockade PEPS, specific files)
3. **Tensor dimensions.** Bond dimensions of PEPS, bond dimensions of boundary MPS, physical dimensions, lattice shape, number of vmapped batches, etc. These are required for cost analysis — do not use trivial defaults like D=1.
4. **Anything else** you are uncertain about: contraction strategy in use, which Hamiltonian terms are active, whether edge effects matter, etc.

Present questions as options where possible. Wait for answers before proceeding.

## Step 2: Identify all contraction operations

Read the in-scope code and catalog **every** tensor contraction, including:

- **Standalone `jnp.einsum`** calls with hand-tuned `optimize=` paths
- **`jnp.tensordot`** + transpose/reshape sequences (equivalent to an einsum)
- **Implicit multi-step sequences**: function calls that compute intermediate tensors (e.g., environments), followed by contractions consuming those intermediates. These must be analyzed as a whole — not just the individual einsums in isolation.

For each contraction, record:
- The einsum subscript (or equivalent)
- The hand-tuned contraction path (if any)
- Which function it lives in and where it sits in the larger computation flow
- Which tensors are shared with other contractions at the same call site

## Step 3: Map index dimensions

Using the dimensions from Step 0, assign a concrete size to every index in every contraction. Be precise about:
- Bulk vs edge dimensions (boundary sites may have bond dim 1)
- Which indices are shared between tensors
- The physical meaning of each index (boundary bond, PEPS virtual bond, physical index, etc.)

## Step 4: Verify standalone paths

For each standalone einsum with a hand-tuned path, compare against `opt_einsum.contract_path` with `optimize='optimal'`:

```python
import opt_einsum
_, info_hand = opt_einsum.contract_path(subscript, *operands, optimize=hand_path)
_, info_opt  = opt_einsum.contract_path(subscript, *operands, optimize='optimal')
```

Report: cost, largest intermediate, and whether the hand path matches optimal.

## Step 5: Analyze implicit sequences

For multi-step sequences identified in Step 2, check:

1. **Shared sub-expressions.** Do sequential operations at the same call site (e.g., gradient computation + left-env update at the same column) share input tensors? If so, is the shared intermediate computed redundantly? Note whether XLA CSE would catch this.
2. **Cross-function opportunities.** Could a function that computes an environment and a function that consumes it be restructured to avoid redundant work?
3. **Combined cost.** Model the total cost of the sequence (e.g., full backward sweep per row) and identify where the dominant cost lies.

## Step 6: Report

Present findings as:

1. **Cost table** — per-einsum cost and largest intermediate, with optimal comparison
2. **Suboptimal paths** — any hand-tuned path that is worse than optimal, with the optimal alternative
3. **Implicit sequence findings** — shared-intermediate opportunities, redundant sub-contractions, and whether XLA CSE is expected to handle them
4. **Per-row / per-sweep cost model** — total cost breakdown showing where computation time is spent
