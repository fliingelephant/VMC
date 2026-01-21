"""Integrator correctness checks against exact linear evolution."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp

from vmc.drivers import DynamicsDriver, Euler, RK4


class _LinearDriver:
    def __init__(self, mat: jax.Array):
        self._mat = mat

    def _time_derivative(self, params: jax.Array, t: float, *, stage: int):
        _ = (t, stage)
        return self._mat @ params

    _tree_add_scaled = staticmethod(DynamicsDriver._tree_add_scaled)
    _tree_weighted_sum = staticmethod(DynamicsDriver._tree_weighted_sum)


class IntegratorTest(unittest.TestCase):
    def test_rk4_matches_expm(self) -> None:
        key = jax.random.key(0)
        mat = jax.random.normal(key, (4, 4), dtype=jnp.complex128)
        mat = 0.1 * (mat + mat.conj().T)
        driver = _LinearDriver(mat)
        y0 = jax.random.normal(jax.random.key(1), (4,), dtype=jnp.complex128)
        dt = 0.1

        rk4 = RK4()
        y1 = rk4.step(driver, y0, 0.0, dt)
        y_exact = jax.scipy.linalg.expm(mat * dt) @ y0
        rel_err = float(jnp.linalg.norm(y1 - y_exact) / jnp.linalg.norm(y_exact))
        self.assertLess(rel_err, 1e-7)

    def test_euler_is_first_order(self) -> None:
        key = jax.random.key(2)
        mat = jax.random.normal(key, (3, 3))
        mat = 0.05 * (mat + mat.T)
        driver = _LinearDriver(mat)
        y0 = jax.random.normal(jax.random.key(3), (3,))
        dt = 0.1

        euler = Euler()
        y1 = euler.step(driver, y0, 0.0, dt)
        y_exact = jax.scipy.linalg.expm(mat * dt) @ y0
        rel_err = float(jnp.linalg.norm(y1 - y_exact) / jnp.linalg.norm(y_exact))
        self.assertLess(rel_err, 1e-3)


if __name__ == "__main__":
    unittest.main()
