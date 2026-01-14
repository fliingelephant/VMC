"""Unit tests for sequential sampling."""
from __future__ import annotations

import unittest

import numpy as np

from VMC import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from flax import nnx

from VMC.samplers.sequential import sequential_sample
from VMC.models.mps import MPS
from VMC.models.peps import PEPS


def _spins_to_index(spins: np.ndarray) -> int:
    bits = (spins + 1) // 2
    idx = 0
    for bit in bits:
        idx = (idx << 1) | int(bit)
    return idx


class SequentialSamplingTest(unittest.TestCase):
    def test_mps_distribution_small(self) -> None:
        n_sites = 4
        n_samples = 256
        n_sweeps = 2
        model = MPS(rngs=nnx.Rngs(0), n_sites=n_sites, bond_dim=2)
        key = jax.random.key(0)
        samples = sequential_sample(
            model, n_samples=n_samples, n_sweeps=n_sweeps, key=key
        )

        samples_np = np.asarray(samples)
        counts = np.zeros(2**n_sites, dtype=np.int32)
        for sample in samples_np:
            counts[_spins_to_index(sample)] += 1
        empirical = counts / float(n_samples)

        basis_bits = np.array(
            [[(i >> (n_sites - 1 - k)) & 1 for k in range(n_sites)] for i in range(2**n_sites)],
            dtype=np.int32,
        )
        spins_basis = 2 * basis_bits - 1
        amps = model._batch_amplitudes(
            [jnp.asarray(t) for t in model.tensors],
            jnp.asarray(spins_basis),
        )
        probs = np.asarray(jnp.abs(amps) ** 2)
        probs = probs / np.sum(probs)

        max_diff = np.max(np.abs(empirical - probs))
        self.assertLess(max_diff, 0.1)

    def test_peps_distribution_small(self) -> None:
        shape = (2, 2)
        n_sites = shape[0] * shape[1]
        n_samples = 256
        n_sweeps = 2
        model = PEPS(rngs=nnx.Rngs(1), shape=shape, bond_dim=2)
        key = jax.random.key(1)
        samples = sequential_sample(
            model, n_samples=n_samples, n_sweeps=n_sweeps, key=key
        )

        samples_np = np.asarray(samples)
        counts = np.zeros(2**n_sites, dtype=np.int32)
        for sample in samples_np:
            counts[_spins_to_index(sample)] += 1
        empirical = counts / float(n_samples)

        basis_bits = np.array(
            [[(i >> (n_sites - 1 - k)) & 1 for k in range(n_sites)] for i in range(2**n_sites)],
            dtype=np.int32,
        )
        spins_basis = 2 * basis_bits - 1
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        amps = []
        for sample in spins_basis:
            amp = PEPS._single_amplitude(
                tensors,
                jnp.asarray(sample),
                shape,
                model.strategy,
            )
            amps.append(amp)
        amps = jnp.asarray(amps)
        probs = np.asarray(jnp.abs(amps) ** 2)
        probs = probs / np.sum(probs)

        max_diff = np.max(np.abs(empirical - probs))
        self.assertLess(max_diff, 0.2)


if __name__ == "__main__":
    unittest.main()
