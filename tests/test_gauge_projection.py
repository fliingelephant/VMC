"""Unit tests for gauge projection utilities."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax.numpy as jnp
from flax import nnx

from vmc.gauge import GaugeConfig, compute_gauge_projection
from vmc.peps import PEPS


class GaugeProjectionTest(unittest.TestCase):
    def test_peps_projection_not_implemented(self) -> None:
        model = PEPS(rngs=nnx.Rngs(2), shape=(2, 2), bond_dim=2)
        params = {"tensors": [[jnp.asarray(t) for t in row] for row in model.tensors]}
        cfg = GaugeConfig()
        with self.assertRaises(NotImplementedError):
            compute_gauge_projection(cfg, model, params)


if __name__ == "__main__":
    unittest.main()
