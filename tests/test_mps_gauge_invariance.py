"""Gauge invariance checks for MPS amplitudes."""
from __future__ import annotations

import unittest

import numpy as np

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.models.mps import MPS


class MPSGaugeInvarianceTest(unittest.TestCase):
    def test_random_gauge_preserves_amplitudes(self) -> None:
        n_sites = 10
        bond_dim = 4
        model = MPS(rngs=nnx.Rngs(0), n_sites=n_sites, bond_dim=bond_dim)
        tensors = [np.asarray(t) for t in model.tensors]

        bond = 2
        dim = tensors[bond].shape[2]
        rng = np.random.default_rng(0)
        G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        G = G + 2.0 * np.eye(dim)
        G_inv = np.linalg.inv(G)

        tensors_gauge = [t.copy() for t in tensors]
        tensors_gauge[bond] = np.einsum("alr,rs->als", tensors_gauge[bond], G)
        tensors_gauge[bond + 1] = np.einsum(
            "ml,alr->amr", G_inv, tensors_gauge[bond + 1]
        )

        key = jax.random.key(0)
        samples = jax.random.bernoulli(key, 0.5, (128, n_sites))
        samples = jnp.where(samples, 1, -1).astype(jnp.int32)

        amps_ref = MPS._batch_amplitudes(
            [jnp.asarray(t) for t in tensors], samples
        )
        amps_gauge = MPS._batch_amplitudes(
            [jnp.asarray(t) for t in tensors_gauge], samples
        )
        max_diff = float(jnp.max(jnp.abs(amps_ref - amps_gauge)))
        self.assertLess(max_diff, 1e-10)


if __name__ == "__main__":
    unittest.main()
