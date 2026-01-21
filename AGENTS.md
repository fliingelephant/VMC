# Agent Guidelines

Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), with JAX-native patterns in this document taking precedence.

**Virtual environment**: Use `.venv` (activate with `source .venv/bin/activate`).

## Dispatching

Use a **hybrid approach**: ABCs for type hierarchies + plum `@dispatch` for multi-type functions.

- **Single-type dispatch** → ABC with `@abstractmethod`.
- **Multi-type dispatch** → `@dispatch` functions (when behavior depends on multiple argument types).
- **Extending library functions** → `@library.func.dispatch`.
- **NO strings for dispatching.** Use typed objects.
- **Minimize helper functions.** Inline short logic; only extract when reused 3+ times or significantly improves readability.

## Design Decisions

- **Prefer composition over inheritance.** Combine small strategy objects rather than deep class trees.
- **No factory functions.** Use direct instantiation with explicit arguments.
- **DRY (Don't Repeat Yourself).** Consolidate duplicated implementations into a single source of truth.
- **Unified eval API (core).** `_value`, `_grad`, and `_value_and_grad` are the only evaluation entrypoints; every other evaluation is a variant of these (plum-dispatched for MPS/PEPS). Avoid manual-dispatch name variants, `log_*` helpers, or `*_fn` wrappers—inline `log`/ratio math and use `jax.vmap` for batching.
- **Let it crash.** Avoid defensive parameter checks (e.g., `if _value_and_grad is None`); assume correct wiring and let errors surface.
- **Sampling gradients.** When a sampler records gradients, compute value+Jacobian for each proposal together and keep gradients only for accepted proposals.
- **Uncertain correctness.** Implementation might be totally incorrect; for uncertain behavior, refer to notes or ask the user.
- **Think twice.** For complicated or important algorithms, think twice before implementing.
- **Julia-style defaults.** Put defaults in function signature, not body. Use `def foo(x, y=10):` not `def foo(x, y=None): y = y or 10`.
- **No unnecessary intermediate variables.** Return directly: `return expr` not `result = expr; return result`.
- **No unused imports/variables.** Remove any defined but unreferenced code.

## JAX Patterns

- **Use `jax.lax.scan`** over Python loops for operations on sequences. Use `jax.lax.scan` only when the loop body is shape-uniform; for open-boundary MPS/PEPS where edge contractions differ from bulk, keep explicit loops or split into separate scans.
- **Use `jax.vmap`** over nested Python loops for batch operations.
- **No `jax.block_until_ready` in hot paths.** It breaks XLA fusion.
- **Configurable dtype** with `jnp.complex128` as default.

## Logging

- Use Python `logging` module, not `print()`.
- **Guard expensive debug computations** with `logger.isEnabledFor(logging.DEBUG)`.
- Control via environment variable (e.g., `VMC_LOG_LEVEL=DEBUG`).
