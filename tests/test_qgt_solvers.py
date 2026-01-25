"""Solver correctness checks for QGT linear solvers."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp

from vmc.qgt.solvers import solve_cg, solve_cholesky, solve_svd


class QGTSolverTest(unittest.TestCase):
    def test_solve_cg_matches_dense(self) -> None:
        key = jax.random.key(0)
        mat = jax.random.normal(key, (24, 24))
        mat = mat.T @ mat + 0.1 * jnp.eye(24)
        rhs = jax.random.normal(jax.random.key(1), (24,))

        x_cg = solve_cg(mat, rhs, tol=1e-10, maxiter=2000)
        x_ref = jnp.linalg.solve(mat, rhs)
        rel_err = float(jnp.linalg.norm(x_cg - x_ref) / jnp.linalg.norm(x_ref))
        self.assertLess(rel_err, 1e-6)

    def test_solve_svd_matches_dense(self) -> None:
        key = jax.random.key(2)
        mat = jax.random.normal(key, (16, 16), dtype=jnp.complex128)
        mat = mat @ mat.conj().T + 0.2 * jnp.eye(16, dtype=jnp.complex128)
        rhs = jax.random.normal(jax.random.key(3), (16,), dtype=jnp.complex128)

        x_svd = solve_svd(mat, rhs, rcond=1e-12)
        x_ref = jnp.linalg.solve(mat, rhs)
        rel_err = float(jnp.linalg.norm(x_svd - x_ref) / jnp.linalg.norm(x_ref))
        self.assertLess(rel_err, 1e-8)

    def test_solve_cholesky_matches_dense(self) -> None:
        key = jax.random.key(4)
        mat = jax.random.normal(key, (12, 12), dtype=jnp.float64)
        mat = mat.T @ mat + 0.5 * jnp.eye(12, dtype=jnp.float64)
        rhs = jax.random.normal(jax.random.key(5), (12,), dtype=jnp.float64)

        x_chol = solve_cholesky(mat, rhs)
        x_ref = jnp.linalg.solve(mat, rhs)
        rel_err = float(jnp.linalg.norm(x_chol - x_ref) / jnp.linalg.norm(x_ref))
        self.assertLess(rel_err, 1e-10)


if __name__ == "__main__":
    unittest.main()
