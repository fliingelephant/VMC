"""SR preconditioner update post-processing tests."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax.numpy as jnp

from vmc.preconditioners import SRPreconditioner


class _FakeModel:
    def __init__(self, dtype) -> None:
        self.dtype = dtype
        self.params_per_site = (1,)
        self.sliced_dims = (1,)


class SRPreconditionerPostSolveTest(unittest.TestCase):
    def _apply(
        self,
        *,
        model_dtype,
        params_dtype,
        grad_factor,
    ):
        preconditioner = SRPreconditioner()
        model = _FakeModel(model_dtype)
        params = jnp.zeros((1,), dtype=params_dtype)
        samples = jnp.zeros((2, 1), dtype=jnp.int32)
        o = jnp.zeros((2, 1), dtype=jnp.complex128)
        local_energies = jnp.array([1.0, -1.0], dtype=jnp.complex128)

        with patch(
            "vmc.preconditioners.preconditioners._solve_sr",
            return_value=(jnp.array([1.0 + 2.0j], dtype=jnp.complex128), {}),
        ):
            updates, _ = preconditioner.apply(
                model,
                params,
                samples,
                o,
                None,
                local_energies,
                grad_factor=grad_factor,
            )
        return updates

    def test_imag_time_complex_model_doubles_complex_updates(self) -> None:
        updates = self._apply(
            model_dtype=jnp.complex128,
            params_dtype=jnp.complex128,
            grad_factor=-1.0,
        )
        self.assertTrue(
            jnp.allclose(updates, jnp.array([2.0 + 4.0j], dtype=jnp.complex128))
        )

    def test_imag_time_real_model_doubles_real_part_only(self) -> None:
        updates = self._apply(
            model_dtype=jnp.float64,
            params_dtype=jnp.float64,
            grad_factor=-1.0,
        )
        self.assertTrue(jnp.allclose(updates, jnp.array([2.0], dtype=jnp.float64)))

    def test_real_time_complex_model_keeps_complex_updates(self) -> None:
        updates = self._apply(
            model_dtype=jnp.complex128,
            params_dtype=jnp.complex128,
            grad_factor=-1.0j,
        )
        self.assertTrue(
            jnp.allclose(updates, jnp.array([1.0 + 2.0j], dtype=jnp.complex128))
        )


if __name__ == "__main__":
    unittest.main()
