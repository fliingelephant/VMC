import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.experimental.lgt.gi_local_terms import (
    GILocalHamiltonian,
    PlaquetteTerm,
    build_electric_terms,
)
from vmc.experimental.lgt.gi_peps import GIPEPS, GIPEPSConfig, _apply_gauss_mask
from vmc.experimental.lgt.gi_sampler import sequential_sample_with_gradients
from vmc.models.peps import DensityMatrix, NoTruncation, ZipUp


class GIPEPSTest(unittest.TestCase):
    def test_gauss_mask(self):
        N = 2
        phys_dim = 2
        Qx = 0
        degeneracy = (2, 2)
        dmax = max(degeneracy)
        tensor = jnp.ones((phys_dim, N, dmax, N, dmax, N, dmax, N, dmax), dtype=jnp.complex128)
        masked = _apply_gauss_mask(tensor, N, Qx, degeneracy)
        for phys in range(phys_dim):
            for ku in range(N):
                for kd in range(N):
                    for kl in range(N):
                        for kr in range(N):
                            valid = (kl + ku - kr - kd - Qx - phys) % N == 0
                            block = masked[phys, ku, :, kd, :, kl, :, kr, :]
                            if valid:
                                self.assertTrue(bool(jax.device_get(jnp.any(block != 0))))
                            else:
                                self.assertTrue(bool(jax.device_get(jnp.all(block == 0))))

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
            config = GIPEPSConfig(
                shape=(4, 4),
                N=2,
                phys_dim=phys_dim,
                Qx=0,
                degeneracy_per_charge=(2, 2),
            )
            electric_terms = build_electric_terms(config.shape, coeff=0.1, N=config.N)
            operator = GILocalHamiltonian(
                shape=config.shape,
                electric_terms=electric_terms,
                plaquette=PlaquetteTerm(coeff=0.2),
            )
            n_chains = 2
            for idx, strategy in enumerate(strategies):
                with self.subTest(phys_dim=phys_dim, strategy=strategy.__class__.__name__):
                    model = GIPEPS(
                        rngs=nnx.Rngs(idx),
                        config=config,
                        contraction_strategy=strategy,
                    )
                    key = jax.random.key(idx)
                    key, init_key = jax.random.split(key)
                    init_keys = jax.random.split(init_key, n_chains)
                    initial_configuration = jax.vmap(
                        model.random_physical_configuration
                    )(init_keys)
                    samples, grads, _, _, _, amps, energies = sequential_sample_with_gradients(
                        model,
                        operator,
                        n_samples=3,
                        n_chains=n_chains,
                        key=key,
                        initial_configuration=initial_configuration,
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

    def test_sampler_matches_value_and_grad(self):
        config = GIPEPSConfig(shape=(4, 4), N=2, phys_dim=2, Qx=0, degeneracy_per_charge=(2, 2))
        strategy = NoTruncation()
        model = GIPEPS(rngs=nnx.Rngs(0), config=config, contraction_strategy=strategy)
        electric_terms = build_electric_terms(config.shape, coeff=0.1, N=config.N)
        operator = GILocalHamiltonian(
            shape=config.shape,
            electric_terms=electric_terms,
            plaquette=PlaquetteTerm(coeff=0.2),
        )
        key = jax.random.key(0)
        key, init_key = jax.random.split(key)
        initial_configuration = jax.vmap(
            model.random_physical_configuration
        )(jax.random.split(init_key, 1))
        samples, grads, _, _, _, amps, _ = sequential_sample_with_gradients(
            model,
            operator,
            n_samples=20,
            n_chains=1,
            key=key,
            initial_configuration=initial_configuration,
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
