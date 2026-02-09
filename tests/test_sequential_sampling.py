"""Unit tests for PEPS sequential Monte Carlo sampling kernels."""
from __future__ import annotations

import itertools
import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.core import _sample_counts, _trim_samples, make_mc_sampler
from vmc.operators import LocalHamiltonian
from vmc.peps import NoTruncation, PEPS, build_mc_kernels
from vmc.peps.standard.compat import _value_and_grad


def _max_full_vs_sliced_diff(
    model: PEPS,
    samples: jax.Array,
    grads_full: jax.Array,
    grads_sliced: jax.Array,
) -> float:
    n_rows, n_cols = model.shape
    bond_dim, phys_dim = model.bond_dim, model.phys_dim
    samples = samples.reshape(samples.shape[0], n_rows, n_cols)
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
                full_site, samples[:, row, col][:, None, None], axis=1
            ).squeeze(axis=1)
            sliced_site = grads_sliced[
                :, offset_sliced : offset_sliced + params_per_phys
            ]
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
    SAMPLES = 40960
    CHAINS = [1, 10]
    SEEDS = [42, 91, 10001]
    MAX_DIFF = 2e-3
    MAX_GRAD_DIFF = 1e-14
    PEPS_BOND_DIM = 3
    PEPS_STRATEGY = NoTruncation()

    @classmethod
    def _make_peps(cls, seed: int) -> PEPS:
        return PEPS(
            rngs=nnx.Rngs(seed),
            shape=cls.SHAPE,
            bond_dim=cls.PEPS_BOND_DIM,
            contraction_strategy=cls.PEPS_STRATEGY,
        )

    def _sample_with_kernels(
        self,
        model: PEPS,
        *,
        n_samples: int,
        n_chains: int,
        key: jax.Array,
        full_gradient: bool,
        initial_configuration: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array | None]:
        _, num_chains, chain_length, total_samples = _sample_counts(
            n_samples, n_chains
        )
        if initial_configuration is None:
            key, init_key = jax.random.split(key)
            initial_configuration = model.random_physical_configuration(
                init_key, n_samples=num_chains
            )
        config_states = initial_configuration.reshape(num_chains, -1)
        chain_keys = jax.random.split(key, num_chains)
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        operator = LocalHamiltonian(shape=self.SHAPE, terms=())
        init_cache, transition, estimate = build_mc_kernels(
            model,
            operator,
            full_gradient=full_gradient,
        )
        cache = init_cache(tensors, config_states)
        mc_sampler = make_mc_sampler(transition, estimate)
        (_, _, _), (samples_hist, estimates) = mc_sampler(
            tensors,
            config_states,
            chain_keys,
            cache,
            n_steps=chain_length,
        )
        samples = _trim_samples(samples_hist, total_samples, n_samples)
        grads = _trim_samples(
            estimates.local_log_derivatives,
            total_samples,
            n_samples,
        )
        if full_gradient:
            p = None
        else:
            p = _trim_samples(
                estimates.active_slice_indices,
                total_samples,
                n_samples,
            )
        return samples, grads, p

    def test_sequential_sample(self) -> None:
        """Validate PEPS sampled distribution against exact probabilities."""
        for n_chains, seed in itertools.product(self.CHAINS, self.SEEDS):
            with self.subTest(n_chains=n_chains, seed=seed):
                model = self._make_peps(seed)
                amps_basis, _, _ = _value_and_grad(
                    model,
                    jnp.asarray(self.BASIS_BITS),
                    full_gradient=False,
                )
                probs = jnp.abs(amps_basis) ** 2
                probs /= probs.sum()
                samples, _, _ = self._sample_with_kernels(
                    model,
                    n_samples=self.SAMPLES,
                    n_chains=n_chains,
                    key=jax.random.key(seed),
                    full_gradient=False,
                )
                indices = jnp.sum(samples * self.WEIGHTS, axis=-1).astype(jnp.int32)
                empirical = jnp.bincount(indices, length=2**self.N_SITES) / samples.shape[0]
                self.assertLess(
                    float(jnp.max(jnp.abs(empirical - probs))),
                    self.MAX_DIFF,
                )

    def test_sequential_sample_with_gradients(self) -> None:
        """Validate PEPS full/sliced gradient consistency and p-indexing."""
        for n_chains, seed in itertools.product(self.CHAINS, self.SEEDS):
            with self.subTest(n_chains=n_chains, seed=seed):
                model = self._make_peps(seed)
                init_key = jax.random.key(seed + 17)
                initial_configuration = model.random_physical_configuration(
                    init_key, n_samples=n_chains
                )
                amps_basis, _, p_ref = _value_and_grad(
                    model,
                    jnp.asarray(self.BASIS_BITS),
                    full_gradient=False,
                )
                probs = jnp.abs(amps_basis) ** 2
                probs /= probs.sum()

                samples_sliced, grads_sliced, p_sliced = self._sample_with_kernels(
                    model,
                    n_samples=self.SAMPLES,
                    n_chains=n_chains,
                    key=jax.random.key(seed),
                    full_gradient=False,
                    initial_configuration=initial_configuration,
                )
                samples_full, grads_full, p_full = self._sample_with_kernels(
                    model,
                    n_samples=self.SAMPLES,
                    n_chains=n_chains,
                    key=jax.random.key(seed),
                    full_gradient=True,
                    initial_configuration=initial_configuration,
                )

                self.assertIsNone(p_full)
                self.assertEqual(grads_sliced.shape, p_sliced.shape)
                self.assertEqual(grads_sliced.shape[0], samples_sliced.shape[0])
                self.assertEqual(grads_full.shape[0], samples_full.shape[0])
                self.assertEqual(
                    grads_full.shape[1],
                    grads_sliced.shape[1] * model.phys_dim,
                )
                self.assertTrue(jnp.array_equal(samples_sliced, samples_full))
                self.assertLess(
                    _max_full_vs_sliced_diff(
                        model, samples_sliced, grads_full, grads_sliced
                    ),
                    self.MAX_GRAD_DIFF,
                )

                indices = jnp.sum(samples_sliced * self.WEIGHTS, axis=-1).astype(jnp.int32)
                self.assertTrue(jnp.array_equal(p_sliced, p_ref[indices]))
                empirical = jnp.bincount(indices, length=2**self.N_SITES) / samples_sliced.shape[0]
                self.assertLess(
                    float(jnp.max(jnp.abs(empirical - probs))),
                    self.MAX_DIFF,
                )


if __name__ == "__main__":
    unittest.main()
