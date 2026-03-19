"""Tests for GI-PEPS Z2 matter dynamics and local terms."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.core import make_mc_sampler
from vmc.operators.local_terms import merge_operators
from vmc.peps import NoTruncation, build_mc_kernels
from vmc.peps.common.contraction import _forward_with_cache
from vmc.peps.gi import GILocalHamiltonian, GIPEPS, GIPEPSConfig
from vmc.peps.gi.local_terms import (
    HorizontalMatterHoppingTerm,
    VerticalMatterHoppingTerm,
)
from vmc.peps.gi.model import assemble_tensors, estimate


class GIMatterTest(unittest.TestCase):
    def _make_model(
        self,
        *,
        Qx: int | jax.Array = 0,
        particle_number: int | None = 2,
    ) -> GIPEPS:
        cfg = GIPEPSConfig(
            shape=(2, 2),
            N=2,
            phys_dim=2,
            Qx=Qx,
            degeneracy_per_charge=(1, 1),
            charge_of_site=(0, 1),
            particle_number=particle_number,
        )
        return GIPEPS(
            rngs=nnx.Rngs(0),
            config=cfg,
            contraction_strategy=NoTruncation(),
        )

    def _assert_gauss(self, sample: jax.Array, cfg: GIPEPSConfig) -> None:
        sites, h_links, v_links = GIPEPS.unflatten_sample(sample, cfg.shape)
        n = jnp.asarray(cfg.N, dtype=h_links.dtype)
        nl = jnp.pad(h_links, ((0, 0), (1, 0)), constant_values=0)
        nr = jnp.pad(h_links, ((0, 0), (0, 1)), constant_values=0)
        nu = jnp.pad(v_links, ((1, 0), (0, 0)), constant_values=0)
        nd = jnp.pad(v_links, ((0, 1), (0, 0)), constant_values=0)
        charge = jnp.asarray(cfg.charge_of_site, dtype=sites.dtype)[sites]
        valid = (nl + nd - nu - nr + charge) % n == jnp.asarray(cfg.Qx, dtype=n.dtype)
        self.assertTrue(bool(jax.device_get(jnp.all(valid))))

    def _local_energy_for_sample(
        self,
        model: GIPEPS,
        operator: GILocalHamiltonian,
        sites: jax.Array,
        h_links: jax.Array,
        v_links: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        cfg = model.config
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        sample = GIPEPS.flatten_sample(sites, h_links, v_links)
        eff_tensors = assemble_tensors(tensors, h_links, v_links, cfg)
        amp, top_envs = _forward_with_cache(eff_tensors, sites, cfg.shape, model.strategy)
        terms, coeff_structure = merge_operators(
            (operator,), cfg.shape, eval_span=type(model).eval_span
        )
        _, energies, _ = estimate(
            tensors,
            sample,
            amp,
            cfg,
            model.strategy,
            top_envs,
            terms=terms,
            coeffs=coeff_structure.build_coeffs(0.0),
        )
        return amp, energies

    def test_fixed_particle_number_initialization_and_transition(self) -> None:
        model = self._make_model(particle_number=2)
        cfg = model.config
        key = jax.random.key(0)
        init_samples = model.random_physical_configuration(key, n_samples=2)

        operator = GILocalHamiltonian(shape=cfg.shape, terms=())
        init_cache, transition, estimate_kernel = build_mc_kernels(
            model,
            operator,
            full_gradient=False,
        )
        mc_sampler = make_mc_sampler(transition, estimate_kernel)
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        config_states = init_samples.reshape(2, -1)
        chain_keys = jax.random.split(key, 2)
        cache = init_cache(tensors, config_states)
        (_, _, _), (samples_hist, _) = mc_sampler(
            tensors,
            config_states,
            chain_keys,
            cache,
            n_steps=3,
        )

        all_samples = jnp.concatenate([init_samples, samples_hist[:, 0], samples_hist[:, 1]], axis=0)
        for sample in all_samples:
            sites, _, _ = GIPEPS.unflatten_sample(sample, cfg.shape)
            self.assertEqual(int(jnp.sum(sites)), 2)
            self._assert_gauss(sample, cfg)

    def test_site_dependent_qx_fixed_particle_number(self) -> None:
        model = self._make_model(
            Qx=jnp.asarray([[1, 0], [0, 0]], dtype=jnp.int32),
            particle_number=1,
        )
        cfg = model.config
        key = jax.random.key(1)
        samples = model.random_physical_configuration(key, n_samples=4)
        for sample in samples:
            sites, _, _ = GIPEPS.unflatten_sample(sample, cfg.shape)
            self.assertEqual(int(jnp.sum(sites)), 1)
            self._assert_gauss(sample, cfg)

    def test_horizontal_hopping_local_energy_matches_manual_ratio(self) -> None:
        model = self._make_model()
        cfg = model.config
        term = HorizontalMatterHoppingTerm(row=0, col=0)
        operator = GILocalHamiltonian(
            shape=cfg.shape,
            terms=(term,),
            coeffs=(jnp.asarray(0.7),),
        )

        sites = jnp.asarray([[1, 0], [0, 1]], dtype=jnp.int32)
        h_links = jnp.asarray([[0], [1]], dtype=jnp.int32)
        v_links = jnp.asarray([[1, 0]], dtype=jnp.int32)
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        amp, energies = self._local_energy_for_sample(
            model, operator, sites, h_links, v_links
        )

        sites_prop = sites.at[0, 0].set(0).at[0, 1].set(1)
        h_prop = h_links.at[0, 0].set(1)
        amp_prop = GIPEPS.apply(
            tensors,
            GIPEPS.flatten_sample(sites_prop, h_prop, v_links),
            cfg.shape,
            cfg,
            model.strategy,
        )
        expected = jnp.asarray(0.7) * amp_prop / amp
        self.assertAlmostEqual(float(jnp.abs(energies[0] - expected)), 0.0, places=9)

    def test_vertical_hopping_local_energy_matches_manual_ratio(self) -> None:
        model = self._make_model()
        cfg = model.config
        term = VerticalMatterHoppingTerm(row=0, col=0)
        operator = GILocalHamiltonian(
            shape=cfg.shape,
            terms=(term,),
            coeffs=(jnp.asarray(-0.3),),
        )

        sites = jnp.asarray([[1, 0], [0, 1]], dtype=jnp.int32)
        h_links = jnp.asarray([[0], [1]], dtype=jnp.int32)
        v_links = jnp.asarray([[1, 0]], dtype=jnp.int32)
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        amp, energies = self._local_energy_for_sample(
            model, operator, sites, h_links, v_links
        )

        sites_prop = sites.at[0, 0].set(0).at[1, 0].set(1)
        v_prop = v_links.at[0, 0].set(0)
        amp_prop = GIPEPS.apply(
            tensors,
            GIPEPS.flatten_sample(sites_prop, h_links, v_prop),
            cfg.shape,
            cfg,
            model.strategy,
        )
        expected = jnp.asarray(-0.3) * amp_prop / amp
        self.assertAlmostEqual(float(jnp.abs(energies[0] - expected)), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
