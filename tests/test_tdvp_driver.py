"""TDVP driver kernel caching checks."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.drivers import TDVPDriver
import vmc.drivers.tdvp as tdvp_module
from vmc.operators import LocalHamiltonian
from vmc.peps import NoTruncation, PEPS


class _ZeroPreconditioner:
    """Minimal preconditioner stub for driver plumbing tests."""

    def apply(
        self,
        model,
        params,
        samples,
        o,
        p,
        local_energies,
        *,
        grad_factor,
    ):
        _ = (model, samples, o, p, local_energies, grad_factor)
        return jax.tree_util.tree_map(jnp.zeros_like, params), {}


class TDVPKernelCacheTest(unittest.TestCase):
    def test_static_operator_reuses_kernels(self) -> None:
        model = PEPS(
            rngs=nnx.Rngs(0),
            shape=(1, 1),
            bond_dim=1,
            contraction_strategy=NoTruncation(),
        )
        with patch(
            "vmc.drivers.tdvp.build_mc_kernels",
            wraps=tdvp_module.build_mc_kernels,
        ) as mocked_build:
            driver = TDVPDriver(
                model,
                LocalHamiltonian(shape=(1, 1), terms=()),
                preconditioner=_ZeroPreconditioner(),
                dt=0.1,
                n_samples=1,
                n_chains=1,
            )
            params = driver._tensors
            key = driver._sampler_key
            config_states = driver._sampler_configuration.reshape(driver.n_chains, -1)
            _, (key, config_states), _ = driver._time_derivative(
                params,
                0.0,
                (key, config_states),
            )
            driver._time_derivative(
                params,
                0.0,
                (key, config_states),
            )
            self.assertEqual(mocked_build.call_count, 1)

    def test_run_chunked_k5(self) -> None:
        model = PEPS(
            rngs=nnx.Rngs(0),
            shape=(1, 1),
            bond_dim=1,
            contraction_strategy=NoTruncation(),
        )
        driver = TDVPDriver(
            model,
            LocalHamiltonian(shape=(1, 1), terms=()),
            preconditioner=_ZeroPreconditioner(),
            dt=0.1,
            n_samples=1,
            n_chains=1,
        )
        k = 5
        for _ in range(2):
            driver.run(k * driver.dt)
        self.assertEqual(driver.step_count, 10)
        self.assertAlmostEqual(driver.t, 1.0, places=12)

if __name__ == "__main__":
    unittest.main()
