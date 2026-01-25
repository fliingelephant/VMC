"""Unit tests for sequential sampling."""
from __future__ import annotations

import itertools
import logging
import time
import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
import netket as nk
from flax import nnx
from plum import dispatch

from vmc.core import _value_and_grad
from vmc.samplers.sequential import sequential_sample, sequential_sample_with_gradients
from vmc.models.mps import MPS
from vmc.models.peps import NoTruncation, PEPS
from vmc.operators import LocalHamiltonian
from vmc.utils.utils import occupancy_to_spin, spin_to_occupancy

logger = logging.getLogger(__name__)


@dispatch
def _max_full_vs_sliced_diff(
    model: MPS,
    samples: jax.Array,
    grads_full: jax.Array,
    grads_sliced: jax.Array,
) -> float:
    n_sites, bond_dim, phys_dim = model.n_sites, model.bond_dim, model.phys_dim
    indices = spin_to_occupancy(samples)
    max_diff = offset_full = offset_sliced = 0
    for site in range(n_sites):
        left_dim, right_dim = MPS.site_dims(site, n_sites, bond_dim)
        params_per_phys = left_dim * right_dim
        full_site = grads_full[
            :, offset_full : offset_full + phys_dim * params_per_phys
        ].reshape(samples.shape[0], phys_dim, params_per_phys)
        selected = jnp.take_along_axis(
            full_site, indices[:, site][:, None, None], axis=1
        ).squeeze(axis=1)
        sliced_site = grads_sliced[:, offset_sliced : offset_sliced + params_per_phys]
        max_diff = jnp.maximum(max_diff, jnp.max(jnp.abs(selected - sliced_site)))
        offset_full += phys_dim * params_per_phys
        offset_sliced += params_per_phys
    return float(max_diff)


@dispatch
def _max_full_vs_sliced_diff(
    model: PEPS,
    samples: jax.Array,
    grads_full: jax.Array,
    grads_sliced: jax.Array,
) -> float:
    n_rows, n_cols = model.shape
    bond_dim, phys_dim = model.bond_dim, model.phys_dim
    spins = spin_to_occupancy(samples).reshape(samples.shape[0], n_rows, n_cols)
    max_diff = offset_full = offset_sliced = 0
    for row in range(n_rows):
        for col in range(n_cols):
            up, down, left, right = PEPS.site_dims(
                row, col, n_rows, n_cols, bond_dim
            )
            params_per_phys = up * down * left * right
            full_site = grads_full[
                :, offset_full : offset_full + phys_dim * params_per_phys
            ].reshape(samples.shape[0], phys_dim, params_per_phys)
            selected = jnp.take_along_axis(
                full_site, spins[:, row, col][:, None, None], axis=1
            ).squeeze(axis=1)
            sliced_site = grads_sliced[:, offset_sliced : offset_sliced + params_per_phys]
            max_diff = jnp.maximum(max_diff, jnp.max(jnp.abs(selected - sliced_site)))
            offset_full += phys_dim * params_per_phys
            offset_sliced += params_per_phys
    return float(max_diff)


class SequentialSamplingTest(unittest.TestCase):
    SHAPE = (3, 3)
    N_SITES = SHAPE[0] * SHAPE[1]
    SHIFTS = jnp.arange(N_SITES - 1, -1, -1, dtype=jnp.int32)
    WEIGHTS = (1 << SHIFTS).astype(jnp.int32)
    BASIS = jnp.arange(2**N_SITES, dtype=jnp.int32)
    BASIS_BITS = (BASIS[:, None] >> SHIFTS[None, :]) & 1
    SPINS_BASIS = occupancy_to_spin(BASIS_BITS)
    
    SAMPLES = 40960
    BURN_IN = 3
    CHAINS = [1, 10]
    SEEDS = [42, 91, 10001]
    MAX_DIFF = 2e-3
    MAX_GRAD_DIFF = 1e-14
    MPS_BOND_DIM = 4
    PEPS_BOND_DIM = 3
    PEPS_STRATEGY = NoTruncation()

    @classmethod
    def _make_mps(cls, seed: int) -> MPS:
        return MPS(
            rngs=nnx.Rngs(seed),
            n_sites=cls.N_SITES,
            bond_dim=cls.MPS_BOND_DIM,
        )

    @classmethod
    def _make_peps(cls, seed: int) -> PEPS:
        return PEPS(
            rngs=nnx.Rngs(seed),
            shape=cls.SHAPE,
            bond_dim=cls.PEPS_BOND_DIM,
            contraction_strategy=cls.PEPS_STRATEGY,
        )

    def test_sequential_sample(self) -> None:
        """Validate sequential_sample distribution against exact probabilities."""
        spins_basis = self.SPINS_BASIS
        weights = self.WEIGHTS
        for make_model, n_chains, seed in itertools.product(
            [self._make_mps, self._make_peps], self.CHAINS, self.SEEDS
        ):
            with self.subTest(
                model=make_model.__name__,
                n_chains=n_chains,
                seed=seed,
            ):
                model = make_model(seed)
                amps_basis, _, _ = _value_and_grad(
                    model, jnp.asarray(spins_basis), full_gradient=False
                )
                probs = jnp.abs(amps_basis) ** 2
                probs /= probs.sum()
                samples = sequential_sample(
                    model,
                    n_samples=self.SAMPLES,
                    n_chains=n_chains,
                    burn_in=self.BURN_IN,
                    key=jax.random.key(seed),
                )
                indices = jnp.sum(
                    spin_to_occupancy(samples) * weights, axis=-1
                ).astype(jnp.int32)
                empirical = jnp.bincount(indices, length=2**self.N_SITES) / samples.shape[0]
                self.assertLess(float(jnp.max(jnp.abs(empirical - probs))), self.MAX_DIFF)

    def test_sequential_sample_with_gradients(self) -> None:
        """Validate distribution, p-indexing, and full/sliced gradient consistency."""
        spins_basis = self.SPINS_BASIS
        weights = self.WEIGHTS

        for make_model, n_chains, seed in itertools.product(
            [self._make_mps, self._make_peps], self.CHAINS, self.SEEDS
        ):
            model = make_model(seed)
            if isinstance(model, MPS):
                hi = nk.hilbert.Spin(s=1 / 2, N=self.N_SITES)
                operator = nk.operator.Ising(
                    hi,
                    nk.graph.Chain(length=self.N_SITES),
                    h=0.0,
                    J=0.0,
                    dtype=jnp.complex128,
                )
            else:
                operator = LocalHamiltonian(shape=self.SHAPE, terms=())
            amps_basis, _, p_ref = _value_and_grad(
                model, jnp.asarray(spins_basis), full_gradient=False
            )
            probs = jnp.abs(amps_basis) ** 2
            probs /= probs.sum()
            key = jax.random.key(seed)
            start = time.perf_counter()
            samples_sliced, grads_sliced, p_sliced, _, _, _, _ = sequential_sample_with_gradients(
                model,
                operator,
                n_samples=self.SAMPLES,
                n_chains=n_chains,
                burn_in=self.BURN_IN,
                key=key,
                full_gradient=False,
            )
            elapsed = time.perf_counter() - start
            logger.info(
                "sequential_sample_with_gradients full_gradient=%s model=%s n_chains=%d seed=%d took %.3fs",
                False,
                make_model.__name__,
                n_chains,
                seed,
                elapsed,
            )
            start = time.perf_counter()
            samples_full, grads_full, p_full, _, _, _, _ = sequential_sample_with_gradients(
                model,
                operator,
                n_samples=self.SAMPLES,
                n_chains=n_chains,
                burn_in=self.BURN_IN,
                key=key,
                full_gradient=True,
            )
            elapsed = time.perf_counter() - start
            logger.info(
                "sequential_sample_with_gradients full_gradient=%s model=%s n_chains=%d seed=%d took %.3fs",
                True,
                make_model.__name__,
                n_chains,
                seed,
                elapsed,
            )
            self.assertEqual(grads_sliced.shape, p_sliced.shape)
            self.assertEqual(grads_sliced.shape[0], samples_sliced.shape[0])
            self.assertEqual(grads_full.shape[0], samples_full.shape[0])
            self.assertEqual(grads_full.shape[1], grads_sliced.shape[1] * model.phys_dim)
            indices_sliced = jnp.sum(
                spin_to_occupancy(samples_sliced) * weights, axis=-1
            ).astype(jnp.int32)
            empirical_sliced = jnp.bincount(indices_sliced, length=2**self.N_SITES) / samples_sliced.shape[0]
            indices_full = jnp.sum(
                spin_to_occupancy(samples_full) * weights, axis=-1
            ).astype(jnp.int32)
            empirical_full = jnp.bincount(indices_full, length=2**self.N_SITES) / samples_full.shape[0]
            with self.subTest(
                model=make_model.__name__,
                n_chains=n_chains,
                seed=seed,
                full_gradient="align",
            ):
                self.assertTrue(jnp.array_equal(samples_sliced, samples_full))
                self.assertLess(
                    _max_full_vs_sliced_diff(model, samples_sliced, grads_full, grads_sliced),
                    self.MAX_GRAD_DIFF,
                )

            with self.subTest(
                model=make_model.__name__,
                n_chains=n_chains,
                seed=seed,
                full_gradient=False,
            ):
                self.assertLess(float(jnp.max(jnp.abs(empirical_sliced - probs))), self.MAX_DIFF)
                self.assertTrue(jnp.array_equal(p_sliced, p_ref[indices_sliced]))

            with self.subTest(
                model=make_model.__name__,
                n_chains=n_chains,
                seed=seed,
                full_gradient=True,
            ):
                self.assertLess(float(jnp.max(jnp.abs(empirical_full - probs))), self.MAX_DIFF)
                amps, grads_ref_full, _ = _value_and_grad(
                    model, jnp.asarray(samples_full), full_gradient=True
                )
                grads_ref_full = grads_ref_full / amps[:, None]
                self.assertLess(
                    float(jnp.max(jnp.abs(
                        jnp.linalg.norm(grads_full, axis=1) - jnp.linalg.norm(grads_ref_full, axis=1)
                    ))),
                    self.MAX_GRAD_DIFF,
                )
                self.assertLess(float(jnp.max(jnp.abs(grads_full - grads_ref_full))), self.MAX_GRAD_DIFF)
                self.assertIsNone(p_full)


if __name__ == "__main__":
    unittest.main()
