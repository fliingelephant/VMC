"""Tests for DynamicsDriver."""
from __future__ import annotations

import functools
import unittest

from VMC import config  # noqa: F401

import jax
import jax.numpy as jnp
import netket as nk
from flax import nnx

from VMC.drivers import DynamicsDriver, ImaginaryTimeUnit, RealTimeUnit
from VMC.models.mps import MPS
from VMC.preconditioners import SRPreconditioner
from VMC.samplers.sequential import sequential_sample_with_gradients
from VMC.utils.smallo import params_per_site


class DynamicsDriverTest(unittest.TestCase):
    def _build_driver(self, *, time_unit):
        n_sites = 8
        hi = nk.hilbert.Spin(s=1 / 2, N=n_sites)
        H = nk.operator.Heisenberg(
            hi, nk.graph.Chain(length=n_sites), dtype=jnp.complex128
        )
        model = MPS(rngs=nnx.Rngs(0), n_sites=hi.size, bond_dim=2)
        sampler = functools.partial(
            sequential_sample_with_gradients,
            n_samples=16,
            burn_in=0,
            full_gradient=False,
        )
        preconditioner = SRPreconditioner(diag_shift=1e-2)
        driver = DynamicsDriver(
            model,
            H,
            sampler=sampler,
            preconditioner=preconditioner,
            dt=0.01,
            time_unit=time_unit,
            sampler_key=jax.random.key(0),
        )
        return driver

    def _assert_driver_step(self, driver):
        self.assertIsNotNone(driver.energy)
        self.assertIsNotNone(driver.last_samples)
        self.assertIsNotNone(driver.last_o)
        self.assertIsNotNone(driver.last_p)
        self.assertEqual(driver.last_samples.shape[1], driver.model.n_sites)
        self.assertEqual(driver.last_samples.shape[0], driver.last_o.shape[0])
        self.assertEqual(driver.last_o.shape, driver.last_p.shape)
        expected_params = sum(params_per_site(driver.model))
        self.assertEqual(driver.last_o.shape[1], expected_params)
        self.assertTrue(bool(jnp.isfinite(driver.energy.mean)))
        self.assertEqual(driver.step_count, 1)
        self.assertAlmostEqual(driver.t, driver.dt)

    def test_real_time_step_updates_energy(self):
        driver = self._build_driver(time_unit=RealTimeUnit())
        driver.step()
        self._assert_driver_step(driver)

    def test_imaginary_time_step_updates_energy(self):
        driver = self._build_driver(time_unit=ImaginaryTimeUnit())
        driver.step()
        self._assert_driver_step(driver)


if __name__ == "__main__":
    unittest.main()
