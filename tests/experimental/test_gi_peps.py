"""Tests for GI-PEPS gauge constraint preservation."""
import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.core import make_mc_sampler
from vmc.peps import NoTruncation, build_mc_kernels
from vmc.peps.gi import GILocalHamiltonian, GIPEPS, GIPEPSConfig


class GIPepsGaussLawTest(unittest.TestCase):
    """Check Gauss-law constraint after sampling moves."""

    def _gauss_law_satisfied(self, sample: jax.Array, config: GIPEPSConfig) -> bool:
        sites, h_links, v_links = GIPEPS.unflatten_sample(sample, config.shape)
        n = jnp.asarray(config.N, dtype=h_links.dtype)
        nl = jnp.pad(h_links, ((0, 0), (1, 0)), constant_values=0)
        nr = jnp.pad(h_links, ((0, 0), (0, 1)), constant_values=0)
        nu = jnp.pad(v_links, ((1, 0), (0, 0)), constant_values=0)
        nd = jnp.pad(v_links, ((0, 1), (0, 0)), constant_values=0)
        div = (nl + nd - nu - nr) % n
        charge_of_site = jnp.asarray(config.charge_of_site, dtype=sites.dtype)
        charge = charge_of_site[sites]
        valid = (div + charge) % n == jnp.asarray(config.Qx, dtype=div.dtype)
        return bool(jnp.all(valid))

    def test_gauss_law_preserved_3x3(self):
        config = GIPEPSConfig(
            shape=(3, 3),
            N=2,
            phys_dim=1,
            Qx=0,
            degeneracy_per_charge=(1, 1),
            charge_of_site=(0,),
        )
        model = GIPEPS(
            rngs=nnx.Rngs(0),
            config=config,
            contraction_strategy=NoTruncation(),
        )

        key = jax.random.key(0)
        sample = model.random_physical_configuration(key, n_samples=1)[0]
        self.assertTrue(self._gauss_law_satisfied(sample, config))

        operator = GILocalHamiltonian(shape=config.shape, terms=())
        init_cache, transition, estimate = build_mc_kernels(
            model,
            operator,
            full_gradient=False,
        )
        mc_sampler = make_mc_sampler(transition, estimate)
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        config_states = sample.reshape(1, -1)
        chain_keys = jax.random.split(key, 1)
        cache = init_cache(tensors, config_states)
        (_, _, _), (samples_hist, _) = mc_sampler(
            tensors,
            config_states,
            chain_keys,
            cache,
            n_steps=3,
        )
        for i in range(samples_hist.shape[0]):
            self.assertTrue(self._gauss_law_satisfied(samples_hist[i, 0], config))


if __name__ == "__main__":
    unittest.main()
