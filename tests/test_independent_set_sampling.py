"""Unit tests for independent-set samplers and enumeration."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp

from vmc.utils.independent_set_sampling import (
    DiscardBlockedSampler,
    IndependentSetSampler,
    build_neighbor_arrays,
    config_codes,
    enumerate_all_configs,
    enumerate_independent_sets_grid,
    grid_edges,
    independent_set_violations,
)


class IndependentSetSamplingTest(unittest.TestCase):
    SEED = 0

    def test_enumeration_matches_filter(self) -> None:
        n_rows, n_cols = 2, 3
        n_sites = n_rows * n_cols
        edges = grid_edges(n_rows, n_cols)
        neighbors, mask = build_neighbor_arrays(n_sites, edges)

        enum = enumerate_independent_sets_grid(n_rows, n_cols)
        violations = independent_set_violations(enum, neighbors, mask)
        self.assertFalse(bool(jnp.any(violations)))

        all_configs = enumerate_all_configs(n_sites)
        all_ok = ~independent_set_violations(all_configs, neighbors, mask)
        enum_codes = set(map(int, config_codes(enum)))
        filt_codes = set(map(int, config_codes(all_configs[all_ok])))
        self.assertEqual(enum_codes, filt_codes)

    def test_independent_set_sampler(self) -> None:
        n_rows, n_cols = 3, 3
        n_sites = n_rows * n_cols
        edges = grid_edges(n_rows, n_cols)
        neighbors, mask = build_neighbor_arrays(n_sites, edges)
        sampler = IndependentSetSampler(n_sites, edges)

        key = jax.random.key(self.SEED)
        samples, log_prob, _, acceptance = sampler.sample(
            log_prob_fn=None,
            n_samples=128,
            n_steps=64,
            key=key,
        )
        violations = independent_set_violations(samples, neighbors, mask)
        self.assertFalse(bool(jnp.any(violations)))
        self.assertGreater(float(acceptance), 0.0)
        self.assertEqual(samples.shape[0], log_prob.shape[0])

    def test_discard_blocked_sampler(self) -> None:
        n_rows, n_cols = 3, 2
        n_sites = n_rows * n_cols
        edges = grid_edges(n_rows, n_cols)
        neighbors, mask = build_neighbor_arrays(n_sites, edges)
        sampler = DiscardBlockedSampler(n_sites, edges)

        def log_prob_fn(batch_samples: jax.Array) -> jax.Array:
            return jnp.zeros((batch_samples.shape[0],), dtype=jnp.float64)

        key = jax.random.key(self.SEED + 1)
        samples, log_prob, _, acceptance = sampler.sample(
            log_prob_fn=log_prob_fn,
            n_samples=128,
            n_steps=64,
            key=key,
        )
        violations = independent_set_violations(samples, neighbors, mask)
        self.assertFalse(bool(jnp.any(violations)))
        self.assertGreater(float(acceptance), 0.0)
        self.assertEqual(samples.shape[0], log_prob.shape[0])


if __name__ == "__main__":
    unittest.main()
