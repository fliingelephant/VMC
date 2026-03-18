"""JAX Lanczos for pure Z2 gauge theory in the constrained plaquette-bit basis.

This is a throwaway helper script. It computes only the lowest eigenvalue.
The Krylov tridiagonal is diagonalized with ``jax.scipy.linalg.eigh``.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

jax.config.update("jax_enable_x64", True)


def plaquette_index(L: int, row: int, col: int) -> int:
    return row * (L - 1) + col


def build_link_terms(L: int) -> tuple[tuple[int, ...], ...]:
    """Return plaquette supports for all physical links."""
    n = L - 1
    terms = []

    for col in range(n):
        terms.append((plaquette_index(L, 0, col),))
    for row in range(1, n):
        for col in range(n):
            terms.append((plaquette_index(L, row - 1, col), plaquette_index(L, row, col)))
    for col in range(n):
        terms.append((plaquette_index(L, n - 1, col),))

    for row in range(n):
        terms.append((plaquette_index(L, row, 0),))
    for col in range(1, n):
        for row in range(n):
            terms.append((plaquette_index(L, row, col - 1), plaquette_index(L, row, col)))
    for row in range(n):
        terms.append((plaquette_index(L, row, n - 1),))

    return tuple(terms)


def build_link_count_diagonal(L: int) -> np.ndarray:
    """Build the electric-term diagonal in the plaquette-bit basis."""
    n_plaquettes = (L - 1) ** 2
    dim = 1 << n_plaquettes
    basis = np.arange(dim, dtype=np.uint32)
    counts = np.zeros(dim, dtype=np.uint8)
    for term in build_link_terms(L):
        values = ((basis >> np.uint32(term[0])) & np.uint32(1)).astype(np.uint8)
        if len(term) == 2:
            values ^= ((basis >> np.uint32(term[1])) & np.uint32(1)).astype(np.uint8)
        counts += values
    return counts.astype(np.float64)


def flip_bit_blocks(v: jax.Array, bit: int) -> jax.Array:
    """Apply the single-bit flip as a contiguous block swap."""
    block = 1 << bit
    group = 2 * block
    grouped = v.reshape((-1, group))
    return jnp.concatenate((grouped[:, block:], grouped[:, :block]), axis=1).reshape((-1,))


class PureZ2GaugeReducedHamiltonianJAX:
    """Matrix-free pure-Z2 gauge Hamiltonian on the constrained subspace."""

    def __init__(self, *, L: int, h: float, g: float) -> None:
        self.L = int(L)
        self.h = float(h)
        self.g = float(g)
        self.n_plaquettes = (L - 1) ** 2
        self.dim = 1 << self.n_plaquettes
        self.diagonal = jax.device_put(4.0 * self.g * build_link_count_diagonal(L))
        self.magnetic_coeff = jnp.asarray(-2.0 * self.h, dtype=self.diagonal.dtype)

        diagonal = self.diagonal
        magnetic_coeff = self.magnetic_coeff
        n_plaquettes = self.n_plaquettes

        @jax.jit
        def matvec(v: jax.Array) -> jax.Array:
            out = diagonal * v
            for bit in range(n_plaquettes):
                out = out + magnetic_coeff * flip_bit_blocks(v, bit)
            return out

        self.matvec = matvec


def lowest_ritz_value(alphas: list[float], betas: list[float]) -> float:
    """Return the smallest Ritz value of the Lanczos tridiagonal."""
    diag = jnp.asarray(alphas, dtype=jnp.float64)
    T = jnp.diag(diag)
    if len(alphas) > 1:
        off = jnp.asarray(betas[: len(alphas) - 1], dtype=jnp.float64)
        T = T + jnp.diag(off, 1) + jnp.diag(off, -1)
    return float(jsp.linalg.eigh(T, eigvals_only=True)[0])


def lanczos_lowest_eigenvalue(
    hamiltonian: PureZ2GaugeReducedHamiltonianJAX,
    *,
    n_iter: int,
    tol: float,
    check_every: int,
    seed: int,
) -> tuple[float, int]:
    """Run a memory-light Lanczos iteration for the lowest eigenvalue."""
    key = jax.random.key(seed)
    q = jax.random.normal(key, (hamiltonian.dim,), dtype=jnp.float64)
    q = q / jnp.linalg.norm(q)
    q_prev = jnp.zeros_like(q)
    beta_prev = jnp.asarray(0.0, dtype=jnp.float64)

    alphas: list[float] = []
    betas: list[float] = []
    previous_ritz: float | None = None

    for step in range(n_iter):
        z = hamiltonian.matvec(q) - beta_prev * q_prev
        alpha = jnp.vdot(q, z).real
        z = z - alpha * q
        beta = jnp.linalg.norm(z)

        alpha_value = float(alpha)
        beta_value = float(beta)
        alphas.append(alpha_value)
        if step < n_iter - 1:
            betas.append(beta_value)

        if step + 1 >= 2 and ((step + 1) % check_every == 0 or step + 1 == n_iter):
            ritz = lowest_ritz_value(alphas, betas)
            print(
                f"iter={step + 1:4d} ritz={ritz:.12f} beta={beta_value:.6e}",
                flush=True,
            )
            if previous_ritz is not None and abs(ritz - previous_ritz) < tol:
                return ritz, step + 1
            previous_ritz = ritz

        if beta_value < 1e-14:
            return lowest_ritz_value(alphas, betas), step + 1

        q_prev, q = q, z / beta
        beta_prev = beta

    return lowest_ritz_value(alphas, betas), n_iter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JAX Lanczos ground-state energy for pure Z2 gauge theory in the constrained plaquette-bit basis.",
    )
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=0.1)
    parser.add_argument("--n-iter", type=int, default=80)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--check-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    hamiltonian = PureZ2GaugeReducedHamiltonianJAX(L=args.L, h=args.h, g=args.g)
    t1 = time.perf_counter()
    energy, n_iter_used = lanczos_lowest_eigenvalue(
        hamiltonian,
        n_iter=args.n_iter,
        tol=args.tol,
        check_every=args.check_every,
        seed=args.seed,
    )
    t2 = time.perf_counter()
    print(f"backend={jax.default_backend()}")
    print(f"L={args.L}")
    print(f"dimension={hamiltonian.dim}")
    print(f"ground_state_energy={energy:.12f}")
    print(f"iterations={n_iter_used}")
    print(f"build_seconds={t1 - t0:.3f}")
    print(f"lanczos_seconds={t2 - t1:.3f}")


if __name__ == "__main__":
    main() # -38.53687034986