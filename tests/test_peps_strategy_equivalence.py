"""Contraction strategy consistency checks for PEPS."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.models.peps import DensityMatrix, NoTruncation, PEPS, ZipUp


class PEPSStrategyEquivalenceTest(unittest.TestCase):
    def test_strategies_match_no_truncation(self) -> None:
        shape = (2, 3)
        bond_dim = 2
        model = PEPS(
            rngs=nnx.Rngs(0),
            shape=shape,
            bond_dim=bond_dim,
            contraction_strategy=NoTruncation(),
        )
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]

        key = jax.random.key(0)
        n_sites = shape[0] * shape[1]
        samples = jax.random.bernoulli(key, 0.5, (64, n_sites))
        samples = jnp.where(samples, 1, -1).astype(jnp.int32)

        def amps_for(strategy):
            return jax.vmap(
                lambda s: PEPS.apply(tensors, s, shape, strategy)
            )(samples)

        amps_ref = amps_for(NoTruncation())
        for strategy in (ZipUp(64), DensityMatrix(64)):
            amps = amps_for(strategy)
            max_diff = float(jnp.max(jnp.abs(amps - amps_ref)))
            self.assertLess(max_diff, 1e-7)


if __name__ == "__main__":
    unittest.main()
