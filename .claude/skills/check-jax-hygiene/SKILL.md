---
name: check-jax-hygiene
description: Check JAX code for GPU-blocking behavior, missed buffer donations, JIT/vmap boundary issues, and other non-idiomatic patterns that hurt performance.
---

# Check JAX Hygiene

Scan JAX code for performance pitfalls that are independent of algorithm or domain logic.

Refer to `CLAUDE.md` conventions on `block_until_ready`, `vmap`, `lax.scan`, and logging guards — this skill enforces those.

## Step 0: Gather context

Ask the user what you need to know. Examples:

1. **Code scope.** Which files or modules to check.
2. **Execution target.** CPU-only, single GPU, multi-GPU — some issues (e.g., donation, host callbacks) matter more on GPU.
3. **Anything else** you are uncertain about.

Wait for answers before proceeding.

## Step 1: GPU-blocking behavior

Scan for operations that force GPU-to-CPU synchronization inside JIT-compiled or `lax.scan`/`lax.fori_loop` regions:

- `jax.debug.print` — inserts a host callback, stalls the pipeline every call
- `jax.debug.callback` — same issue
- `jnp.array(...).item()`, `float(...)`, `int(...)` on traced values — forces materialization
- `print()` on JAX arrays inside traced code
- `jax.block_until_ready` in hot paths (appropriate at measurement boundaries, not inside loops)

For each finding, note whether it is inside a JIT/scan/fori_loop scope and estimate the frequency (once per step vs once per sweep vs once per sample).

## Step 2: Buffer donation

For each `@jax.jit` or `@functools.partial(jax.jit, ...)` call:

1. Identify `donate_argnums` (or `donate_argnames`).
2. At the call site, check whether the donated buffer is used after the call. If the caller immediately overwrites the variable, donation is safe.
3. Check for **missed donations**: arguments whose buffers are overwritten at the call site but not listed in `donate_argnums`. These waste GPU memory by keeping the old buffer alive.

## Step 3: JIT boundaries

Check that JIT compilation is applied at the right level:

- **Too low**: JIT on small helper functions that are called inside an already-JIT'd function — redundant, may cause retracing overhead.
- **Too high**: Large functions with Python-level control flow (if/for on non-static values) that prevent tracing.
- **Static argnums**: Verify that `static_argnums` / `static_argnames` are correct — wrong static args cause unnecessary recompilation; missing static args cause tracing errors on Python values.

## Step 4: vmap usage

Check that `jax.vmap` is applied efficiently:

- **Outermost vmap**: vmap should wrap the largest possible computation for XLA fusion. Vmapping small inner functions misses fusion opportunities.
- **Unnecessary vmap**: Manual batching (e.g., leading batch dimension in einsums) where vmap would be cleaner, or vice versa.
- **vmap + JIT interaction**: vmap inside JIT is fine; JIT inside vmap causes per-batch recompilation.

## Step 5: Other patterns

- **Python loops over JAX arrays** that should be `lax.scan` or `lax.fori_loop` for XLA compilation.
- **Unnecessary `.reshape(-1)` / `.flatten()`** creating copies where views would suffice.
- **`jnp.concatenate` in loops** — repeated concatenation is O(n²); accumulate in a list and concatenate once, or use `lax.scan`.

## Step 6: Report

Present findings grouped by severity:

1. **High** — GPU-blocking in hot paths, missed donations of large buffers
2. **Medium** — JIT boundary issues, suboptimal vmap placement
3. **Low** — style issues, minor inefficiencies

For each finding: file, line, what the issue is, and suggested fix.
