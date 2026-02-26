---
name: check-efficiency
description: Comprehensive efficiency audit for PEPS VMC code. Invokes read-notes, check-contraction-paths, and check-jax-hygiene, then checks domain-specific issues like missed environment reuse and redundant computation.
---

# Check Efficiency

Comprehensive audit for computational inefficiencies — both generic JAX issues and domain-specific tensor network patterns.

Refer to `CLAUDE.md` conventions on least-redundancy priority and theory-first verification — this skill enforces those.

## Step 0: Gather context

Ask the user what you need to know. Examples:

1. **Code scope.** Which modules to audit.
2. **Tensor dimensions.** Bond dimensions, boundary MPS dimensions, physical dimensions, lattice shape, batch sizes, etc. Required for cost estimates — do not use trivial defaults.
3. **Anything else** you are uncertain about: which contraction strategy is in use, target hardware, known bottlenecks, etc.

Wait for answers before proceeding.

## Step 1: Read reference papers

Invoke `/read-notes`. Ask it to extract:
- Expected computational costs and scaling
- Recommended environment reuse and caching patterns
- Any algorithmic optimizations described in the papers

## Step 2: Check contraction paths

Invoke `/check-contraction-paths` with the code scope and dimensions from Step 0.

## Step 3: Check JAX hygiene

Invoke `/check-jax-hygiene` with the code scope from Step 0.

## Step 4: Domain-specific efficiency checks

These checks require understanding the PEPS VMC algorithm and cannot be automated by generic tools.

### Environment and intermediate reuse

- **Cache-turnover pattern.** Verify that the transition → estimate cache handoff is correct: bottom_envs from estimate are reused by the next transition, top_envs from transition are consumed by estimate. No environments should be rebuilt unnecessarily.
- **Cross-step reuse.** In multi-step integrators (e.g., RK4 calling the derivative function 4 times), check whether environments or samples from one evaluation are avoidably discarded and rebuilt in the next.
- **Shared intermediates within a sweep.** At each column in the backward pass, multiple operations (gradient, left-env update, term evaluation) may share input tensors. Check whether shared sub-expressions are computed redundantly. Note whether XLA CSE is expected to handle this automatically.

### Redundant computation

- **Rebuilding what's already cached.** Are row MPOs, right environments, or boundary states recomputed when they could be carried from a previous step?
- **Unnecessary recomputation across MC sweeps.** Does the sampler recompute quantities that are unchanged between sweeps (e.g., static operator structure, bucketed terms)?

### Memory and data flow

- **Large intermediate tensors.** Flag any contraction that produces intermediates much larger than the inputs (e.g., O(Dc^3) when O(Dc^2) is achievable).
- **Unnecessary copies.** Transpose, reshape, or concatenation that could be avoided by adjusting index conventions upstream.

## Step 5: Consolidated report

Merge findings from Steps 2–4 into a single report:

1. **Summary table** — all findings ranked by estimated impact
2. **Contraction path results** — from `/check-contraction-paths`
3. **JAX hygiene results** — from `/check-jax-hygiene`
4. **Domain-specific findings** — environment reuse, redundant computation, memory issues
5. **Cost model** — per-sweep total cost breakdown showing where time is spent
