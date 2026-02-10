"""Integrator correctness checks against exact linear evolution."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp

from vmc.drivers import Euler, RK4


def _linear_derivative(
    mat: jax.Array,
    params: jax.Array,
    t: float,
    carry: tuple[jax.Array, jax.Array],
):
    _ = t
    return mat @ params, carry, (jnp.zeros((1,)), {})


class IntegratorTest(unittest.TestCase):
    def test_rk4_matches_expm(self) -> None:
        key = jax.random.key(0)
        mat = jax.random.normal(key, (4, 4), dtype=jnp.complex128)
        mat = 0.1 * (mat + mat.conj().T)
        y0 = jax.random.normal(jax.random.key(1), (4,), dtype=jnp.complex128)
        dt = 0.1
        key_state = jax.random.key(11)
        config_states = jnp.zeros((1, 1), dtype=jnp.int32)

        rk4 = RK4()
        y1, _, _, _ = rk4.step(
            lambda params, t, carry: _linear_derivative(
                mat, params, t, carry
            ),
            y0,
            0.0,
            dt,
            (key_state, config_states),
        )
        y_exact = jax.scipy.linalg.expm(mat * dt) @ y0
        rel_err = float(jnp.linalg.norm(y1 - y_exact) / jnp.linalg.norm(y_exact))
        self.assertLess(rel_err, 1e-7)

    def test_euler_is_first_order(self) -> None:
        key = jax.random.key(2)
        mat = jax.random.normal(key, (3, 3))
        mat = 0.05 * (mat + mat.T)
        y0 = jax.random.normal(jax.random.key(3), (3,))
        dt = 0.1
        key_state = jax.random.key(13)
        config_states = jnp.zeros((1, 1), dtype=jnp.int32)

        euler = Euler()
        y1, _, _, _ = euler.step(
            lambda params, t, carry: _linear_derivative(
                mat, params, t, carry
            ),
            y0,
            0.0,
            dt,
            (key_state, config_states),
        )
        y_exact = jax.scipy.linalg.expm(mat * dt) @ y0
        rel_err = float(jnp.linalg.norm(y1 - y_exact) / jnp.linalg.norm(y_exact))
        self.assertLess(rel_err, 1e-3)


if __name__ == "__main__":
    unittest.main()
