# Agent Guidelines

Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), with JAX-native patterns in this document taking precedence.

## Dispatching

Use a **hybrid approach**: ABCs for type hierarchies + plum `@dispatch` for multi-type functions.

- **Single-type dispatch** → ABC with `@abstractmethod`.
- **Multi-type dispatch** → `@dispatch` functions (when behavior depends on multiple argument types).
- **Extending library functions** → `@library.func.dispatch`.
- **NO strings for dispatching.** Use typed objects.

## Design Decisions

- **Prefer composition over inheritance.** Combine small strategy objects rather than deep class trees.
- **No factory functions.** Use direct instantiation with explicit arguments.
- **DRY (Don't Repeat Yourself).** Consolidate duplicated implementations into a single source of truth.

## JAX Patterns

- **Use `jax.lax.scan`** over Python loops for operations on sequences. Use `jax.lax.scan` only when the loop body is shape-uniform; for open-boundary MPS/PEPS where edge contractions differ from bulk, keep explicit loops or split into separate scans.
- **Use `jax.vmap`** over nested Python loops for batch operations.
- **No `jax.block_until_ready` in hot paths.** It breaks XLA fusion.
- **Configurable dtype** with `jnp.complex128` as default.

## Logging

- Use Python `logging` module, not `print()`.
- **Guard expensive debug computations** with `logger.isEnabledFor(logging.DEBUG)`.
- Control via environment variable (e.g., `VMC_LOG_LEVEL=DEBUG`).
