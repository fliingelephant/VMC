"""Consistency checks for sequential samplers with gradients."""
from __future__ import annotations

import logging
import unittest

from VMC import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from flax import nnx

from VMC.core import _value_and_grad
from VMC.models.mps import MPS
from VMC.models.peps import NoTruncation, PEPS
from VMC.samplers.sequential import sequential_sample_with_gradients

logger = logging.getLogger(__name__)


class SequentialGradientTest(unittest.TestCase):
    SEED = 0
    N_SAMPLES_MPS = 512
    N_SAMPLES_PEPS = 256
    N_CHAINS = 8
    BURN_IN = 10

    def _check_sliced_gradients(self, model, *, n_samples: int) -> None:
        key = jax.random.key(self.SEED)
        samples, grads, p, _, _, info = sequential_sample_with_gradients(
            model,
            n_samples=n_samples,
            n_chains=self.N_CHAINS,
            burn_in=self.BURN_IN,
            key=key,
            full_gradient=False,
            return_prob=True,
        )
        amps, grads_ref, p_ref = _value_and_grad(
            model, samples, full_gradient=False
        )
        grads_ref = grads_ref / amps[:, None]
        max_diff = float(jnp.max(jnp.abs(grads - grads_ref)))
        max_prob_diff = float(
            jnp.max(jnp.abs(info["prob"] - jnp.abs(amps) ** 2))
        )
        logger.info(
            "sliced_grad_max_diff=%s prob_max_diff=%s", max_diff, max_prob_diff
        )
        self.assertLess(max_diff, 1e-8)
        self.assertLess(max_prob_diff, 1e-8)
        self.assertTrue(jnp.array_equal(p, p_ref))

    def _check_full_gradients(self, model, *, n_samples: int) -> None:
        key = jax.random.key(self.SEED + 1)
        samples, grads, p, _, _, info = sequential_sample_with_gradients(
            model,
            n_samples=n_samples,
            n_chains=self.N_CHAINS,
            burn_in=self.BURN_IN,
            key=key,
            full_gradient=True,
            return_prob=True,
        )
        self.assertIsNone(p)
        amps, grads_ref, _ = _value_and_grad(
            model, samples, full_gradient=True
        )
        grads_ref = grads_ref / amps[:, None]
        max_diff = float(jnp.max(jnp.abs(grads - grads_ref)))
        max_prob_diff = float(
            jnp.max(jnp.abs(info["prob"] - jnp.abs(amps) ** 2))
        )
        logger.info(
            "full_grad_max_diff=%s prob_max_diff=%s", max_diff, max_prob_diff
        )
        self.assertLess(max_diff, 1e-8)
        self.assertLess(max_prob_diff, 1e-8)

    def test_mps_sequential_gradients(self) -> None:
        model = MPS(
            rngs=nnx.Rngs(self.SEED),
            n_sites=10,
            bond_dim=6,
        )
        self._check_sliced_gradients(model, n_samples=self.N_SAMPLES_MPS)
        self._check_full_gradients(model, n_samples=self.N_SAMPLES_MPS // 2)

    def test_peps_sequential_gradients(self) -> None:
        model = PEPS(
            rngs=nnx.Rngs(self.SEED),
            shape=(3, 3),
            bond_dim=3,
            contraction_strategy=NoTruncation(),
        )
        self._check_sliced_gradients(model, n_samples=self.N_SAMPLES_PEPS)


if __name__ == "__main__":
    unittest.main()
