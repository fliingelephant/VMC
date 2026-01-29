import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.experimental.lgt.gi_local_terms import GILocalHamiltonian, build_electric_terms
from vmc.experimental.lgt.gi_peps import GIPEPS, GIPEPSConfig
from vmc.experimental.lgt.gi_sampler import sequential_sample_with_gradients
from vmc.models.peps import DensityMatrix, NoTruncation, ZipUp
from vmc.operators import PlaquetteTerm


class GIPEPSTest(unittest.TestCase):
    def _plaquette_terms(self, shape, coeff):
        n_rows, n_cols = shape
        return tuple(
            PlaquetteTerm(row=r, col=c, coeff=coeff)
            for r in range(n_rows - 1)
            for c in range(n_cols - 1)
        )

    def _assert_gauss(self, sites, h_links, v_links, config):
        charge = jnp.take(jnp.asarray(config.charge_of_site), sites)
        nl = jnp.pad(h_links, ((0, 0), (1, 0)), constant_values=0)
        nr = jnp.pad(h_links, ((0, 0), (0, 1)), constant_values=0)
        nu = jnp.pad(v_links, ((1, 0), (0, 0)), constant_values=0)
        nd = jnp.pad(v_links, ((0, 1), (0, 0)), constant_values=0)
        gauss = (nl + nu - nr - nd + charge - config.Qx) % config.N
        ok = jax.device_get(jnp.all(gauss == 0))
        self.assertTrue(bool(ok))

    def test_flatten_roundtrip(self):
        sites = jnp.asarray(
            [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
            dtype=jnp.int32,
        )
        h_links = jnp.asarray(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1]],
            dtype=jnp.int32,
        )
        v_links = jnp.asarray(
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
            dtype=jnp.int32,
        )
        sample = GIPEPS.flatten_sample(sites, h_links, v_links)
        sites_out, h_out, v_out = GIPEPS.unflatten_sample(sample, (4, 4))
        self.assertTrue(jnp.array_equal(sites, sites_out))
        self.assertTrue(jnp.array_equal(h_links, h_out))
        self.assertTrue(jnp.array_equal(v_links, v_out))

    def test_sampler_shapes(self):
        strategies = [
            NoTruncation(),
            ZipUp(truncate_bond_dimension=4),
            DensityMatrix(truncate_bond_dimension=4),
        ]
        for phys_dim in (1, 2):
            charge_of_site = tuple(range(phys_dim))
            config = GIPEPSConfig(
                shape=(4, 4),
                N=2,
                phys_dim=phys_dim,
                Qx=0,
                degeneracy_per_charge=(2, 2),
                charge_of_site=charge_of_site,
            )
            electric_terms = build_electric_terms(config.shape, coeff=0.1, N=config.N)
            plaquette_terms = self._plaquette_terms(config.shape, coeff=0.2)
            operator = GILocalHamiltonian(
                shape=config.shape,
                terms=electric_terms + plaquette_terms,
            )
            n_chains = 2
            for idx, strategy in enumerate(strategies):
                with self.subTest(phys_dim=phys_dim, strategy=strategy.__class__.__name__):
                    model = GIPEPS(
                        rngs=nnx.Rngs(idx),
                        config=config,
                        contraction_strategy=strategy,
                    )
                    bulk = jnp.asarray(model.tensors[1][1])
                    edge = jnp.asarray(model.tensors[0][1])
                    corner = jnp.asarray(model.tensors[0][0])
                    self.assertEqual(
                        bulk.shape,
                        (config.phys_dim, config.N**3, config.dmax, config.dmax, config.dmax, config.dmax),
                    )
                    self.assertEqual(
                        edge.shape,
                        (config.phys_dim, config.N**2, 1, config.dmax, config.dmax, config.dmax),
                    )
                    self.assertEqual(
                        corner.shape,
                        (config.phys_dim, config.N, 1, config.dmax, 1, config.dmax),
                    )
                    key = jax.random.key(idx)
                    key, init_key = jax.random.split(key)
                    samples, grads, _, _, _, amps, energies = sequential_sample_with_gradients(
                        model,
                        operator,
                        n_samples=3,
                        n_chains=n_chains,
                        key=key,
                        initial_configuration=model.random_physical_configuration(
                            init_key, n_samples=n_chains
                        ),
                        burn_in=1,
                        full_gradient=True,
                    )
                    self.assertEqual(samples.shape[0], 3)
                    self.assertEqual(grads.shape[0], 3)
                    self.assertEqual(amps.shape[0], 3)
                    self.assertEqual(energies.shape[0], 3)
                    amps_ok = jax.device_get(jnp.all(jnp.isfinite(jnp.asarray(amps))))
                    energies_ok = jax.device_get(jnp.all(jnp.isfinite(jnp.asarray(energies))))
                    self.assertTrue(bool(amps_ok))
                    self.assertTrue(bool(energies_ok))
                    samples_unflat = jax.vmap(
                        lambda s: GIPEPS.unflatten_sample(s, config.shape)
                    )(samples)
                    sites_s, h_s, v_s = samples_unflat
                    for i in range(samples.shape[0]):
                        self._assert_gauss(sites_s[i], h_s[i], v_s[i], config)

    def test_sampler_matches_value_and_grad(self):
        config = GIPEPSConfig(
            shape=(4, 4),
            N=2,
            phys_dim=2,
            Qx=0,
            degeneracy_per_charge=(2, 2),
            charge_of_site=(0, 1),
        )
        strategy = NoTruncation()
        model = GIPEPS(rngs=nnx.Rngs(0), config=config, contraction_strategy=strategy)
        electric_terms = build_electric_terms(config.shape, coeff=0.1, N=config.N)
        plaquette_terms = self._plaquette_terms(config.shape, coeff=0.2)
        operator = GILocalHamiltonian(
            shape=config.shape,
            terms=electric_terms + plaquette_terms,
        )
        key = jax.random.key(0)
        key, init_key = jax.random.split(key)
        samples, grads, _, _, _, amps, _ = sequential_sample_with_gradients(
            model,
            operator,
            n_samples=20,
            n_chains=1,
            key=key,
            initial_configuration=model.random_physical_configuration(
                init_key, n_samples=1
            ),
            burn_in=1,
            full_gradient=True,
        )
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        for i in range(samples.shape[0]):
            sample = samples[i]

            def amplitude_from_tensors(tensors_pytree):
                return GIPEPS.apply(tensors_pytree, sample, config.shape, config, strategy)

            amp_ref, grads_ref = jax.value_and_grad(amplitude_from_tensors, holomorphic=True)(
                tensors
            )
            grad_parts = []
            for r in range(config.shape[0]):
                for c in range(config.shape[1]):
                    grad_parts.append(grads_ref[r][c].reshape(-1))
            grad_ref = jnp.concatenate(grad_parts) / amp_ref

            amp_ok = jax.device_get(jnp.allclose(amps[i], amp_ref, rtol=1e-5, atol=1e-6))
            grad_ok = jax.device_get(jnp.allclose(grads[i], grad_ref, rtol=1e-5, atol=1e-6))
            self.assertTrue(bool(amp_ok))
            self.assertTrue(bool(grad_ok))

    def test_sampler_matches_value_and_grad_phys_dim_not_equal_N(self):
        config = GIPEPSConfig(
            shape=(3, 3),
            N=3,
            phys_dim=4,
            Qx=0,
            degeneracy_per_charge=(2, 2, 2),
            charge_of_site=(0, 1, 2, 0),
        )
        strategy = NoTruncation()
        model = GIPEPS(rngs=nnx.Rngs(1), config=config, contraction_strategy=strategy)
        electric_terms = build_electric_terms(config.shape, coeff=0.1, N=config.N)
        plaquette_terms = self._plaquette_terms(config.shape, coeff=0.2)
        operator = GILocalHamiltonian(
            shape=config.shape,
            terms=electric_terms + plaquette_terms,
        )
        key = jax.random.key(1)
        key, init_key = jax.random.split(key)
        samples, grads, _, _, _, amps, _ = sequential_sample_with_gradients(
            model,
            operator,
            n_samples=2,
            n_chains=1,
            key=key,
            initial_configuration=model.random_physical_configuration(
                init_key, n_samples=1
            ),
            burn_in=1,
            full_gradient=True,
        )
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        for i in range(samples.shape[0]):
            sample = samples[i]

            def amplitude_from_tensors(tensors_pytree):
                return GIPEPS.apply(tensors_pytree, sample, config.shape, config, strategy)

            amp_ref, grads_ref = jax.value_and_grad(amplitude_from_tensors, holomorphic=True)(
                tensors
            )
            grad_parts = []
            for r in range(config.shape[0]):
                for c in range(config.shape[1]):
                    grad_parts.append(grads_ref[r][c].reshape(-1))
            grad_ref = jnp.concatenate(grad_parts) / amp_ref

            amp_ok = jax.device_get(jnp.allclose(amps[i], amp_ref, rtol=1e-5, atol=1e-6))
            grad_ok = jax.device_get(jnp.allclose(grads[i], grad_ref, rtol=1e-5, atol=1e-6))
            self.assertTrue(bool(amp_ok))
            self.assertTrue(bool(grad_ok))



if __name__ == "__main__":
    unittest.main()
