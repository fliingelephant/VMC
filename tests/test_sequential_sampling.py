"""Unit tests for sequential sampling."""
from __future__ import annotations

import itertools
import unittest

from VMC import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from flax import nnx
from plum import dispatch

from VMC.core import _value_and_grad
from VMC.samplers.sequential import sequential_sample, sequential_sample_with_gradients
from VMC.models.mps import MPS
from VMC.models.peps import NoTruncation, PEPS
from VMC.utils.smallo import mps_site_dims, peps_site_dims
from VMC.utils.utils import occupancy_to_spin, spin_to_occupancy


@dispatch
def _max_full_vs_sliced_diff(
    model: MPS,
    samples: jax.Array,
    grads_full: jax.Array,
    grads_sliced: jax.Array,
) -> float:
    samples = jnp.asarray(samples)
    max_diff = jnp.asarray(0.0)
    n_sites = model.n_sites
    bond_dim = model.bond_dim
    phys_dim = model.phys_dim
    indices = spin_to_occupancy(samples)
    offset_full = 0
    offset_sliced = 0
    for site in range(n_sites):
        left_dim, right_dim = mps_site_dims(site, n_sites, bond_dim)
        params_per_phys = left_dim * right_dim
        full_site = grads_full[
            :, offset_full : offset_full + phys_dim * params_per_phys
        ]
        full_site = full_site.reshape(
            samples.shape[0], phys_dim, params_per_phys
        )
        selected = jnp.take_along_axis(
            full_site,
            indices[:, site][:, None, None],
            axis=1,
        ).squeeze(axis=1)
        sliced_site = grads_sliced[
            :, offset_sliced : offset_sliced + params_per_phys
        ]
        max_diff = jnp.maximum(
            max_diff, jnp.max(jnp.abs(selected - sliced_site))
        )
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
    samples = jnp.asarray(samples)
    max_diff = jnp.asarray(0.0)
    n_rows, n_cols = model.shape
    bond_dim = model.bond_dim
    phys_dim = model.phys_dim
    spins = spin_to_occupancy(samples).reshape(
        samples.shape[0], n_rows, n_cols
    )
    offset_full = 0
    offset_sliced = 0
    for row in range(n_rows):
        for col in range(n_cols):
            up, down, left, right = peps_site_dims(
                row, col, n_rows, n_cols, bond_dim
            )
            params_per_phys = up * down * left * right
            full_site = grads_full[
                :, offset_full : offset_full + phys_dim * params_per_phys
            ]
            full_site = full_site.reshape(
                samples.shape[0], phys_dim, params_per_phys
            )
            selected = jnp.take_along_axis(
                full_site,
                spins[:, row, col][:, None, None],
                axis=1,
            ).squeeze(axis=1)
            sliced_site = grads_sliced[
                :, offset_sliced : offset_sliced + params_per_phys
            ]
            max_diff = jnp.maximum(
                max_diff, jnp.max(jnp.abs(selected - sliced_site))
            )
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
                    model,
                    jnp.asarray(spins_basis),
                    full_gradient=False,
                )
                probs = jnp.abs(amps_basis) ** 2
                probs = probs / jnp.sum(probs)
                key = jax.random.key(seed)
                samples = sequential_sample(
                    model,
                    n_samples=self.SAMPLES,
                    n_chains=n_chains,
                    burn_in=self.BURN_IN,
                    key=key,
                )
                indices = jnp.sum(
                    spin_to_occupancy(samples) * weights,
                    axis=-1,
                ).astype(jnp.int32)
                counts = jnp.bincount(indices, length=2**self.N_SITES)
                empirical = counts / samples.shape[0]
                max_diff = float(jnp.max(jnp.abs(empirical - probs)))
                self.assertLess(max_diff, self.MAX_DIFF)

    def test_sequential_sample_with_gradients(self) -> None:
        """Validate distribution, p-indexing, and full/sliced gradient consistency."""
        spins_basis = self.SPINS_BASIS
        weights = self.WEIGHTS

        for make_model, n_chains, seed in itertools.product(
            [self._make_mps, self._make_peps], self.CHAINS, self.SEEDS
        ):
            model = make_model(seed)
            amps_basis, _, p_ref = _value_and_grad(
                model,
                jnp.asarray(spins_basis),
                full_gradient=False,
            )
            probs = jnp.abs(amps_basis) ** 2
            probs = probs / jnp.sum(probs)
            key = jax.random.key(seed)
            samples_sliced, grads_sliced, p_sliced, _, _ = (
                sequential_sample_with_gradients(
                    model,
                    n_samples=self.SAMPLES,
                    n_chains=n_chains,
                    burn_in=self.BURN_IN,
                    key=key,
                    full_gradient=False,
                    return_prob=False,
                )
            )
            samples_full, grads_full, p_full, _, _ = (
                sequential_sample_with_gradients(
                    model,
                    n_samples=self.SAMPLES,
                    n_chains=n_chains,
                    burn_in=self.BURN_IN,
                    key=key,
                    full_gradient=True,
                    return_prob=False,
                )
            )
            phys_dim = model.phys_dim
            self.assertEqual(grads_sliced.shape, p_sliced.shape)
            self.assertEqual(grads_sliced.shape[0], samples_sliced.shape[0])
            self.assertEqual(grads_full.shape[0], samples_full.shape[0])
            self.assertEqual(
                grads_full.shape[1],
                grads_sliced.shape[1] * phys_dim,
            )
            indices_sliced = jnp.sum(
                spin_to_occupancy(samples_sliced) * weights,
                axis=-1,
            ).astype(jnp.int32)
            counts_sliced = jnp.bincount(indices_sliced, length=2**self.N_SITES)
            empirical_sliced = counts_sliced / samples_sliced.shape[0]
            indices_full = jnp.sum(
                spin_to_occupancy(samples_full) * weights,
                axis=-1,
            ).astype(jnp.int32)
            counts_full = jnp.bincount(indices_full, length=2**self.N_SITES)
            empirical_full = counts_full / samples_full.shape[0]
            with self.subTest(
                model=make_model.__name__,
                n_chains=n_chains,
                seed=seed,
                full_gradient="align",
            ):
                self.assertTrue(
                    jnp.array_equal(samples_sliced, samples_full)
                )
                max_align_diff = _max_full_vs_sliced_diff(
                    model,
                    samples_sliced,
                    grads_full,
                    grads_sliced,
                )
                self.assertLess(max_align_diff, self.MAX_GRAD_DIFF)

            with self.subTest(
                model=make_model.__name__,
                n_chains=n_chains,
                seed=seed,
                full_gradient=False,
            ):
                max_diff = float(jnp.max(jnp.abs(empirical_sliced - probs)))
                self.assertLess(max_diff, self.MAX_DIFF)
                p_ref_sliced = p_ref[indices_sliced]
                self.assertTrue(jnp.array_equal(p_sliced, p_ref_sliced))

            with self.subTest(
                model=make_model.__name__,
                n_chains=n_chains,
                seed=seed,
                full_gradient=True,
            ):
                max_diff = float(jnp.max(jnp.abs(empirical_full - probs)))
                self.assertLess(max_diff, self.MAX_DIFF)
                amps, grads_ref_full, _ = _value_and_grad(
                    model,
                    jnp.asarray(samples_full),
                    full_gradient=True,
                )
                grads_ref_full = grads_ref_full / amps[:, None]
                max_grad_diff = float(
                    jnp.max(jnp.abs(grads_full - grads_ref_full))
                )
                ok_norm = jnp.linalg.norm(grads_full, axis=1)
                ok_norm_ref = jnp.linalg.norm(
                    grads_ref_full, axis=1
                )
                max_ok_diff = float(
                    jnp.max(jnp.abs(ok_norm - ok_norm_ref))
                )
                self.assertLess(max_ok_diff, self.MAX_GRAD_DIFF)
                self.assertLess(max_grad_diff, self.MAX_GRAD_DIFF)
                self.assertIsNone(p_full)


if __name__ == "__main__":
    unittest.main()
