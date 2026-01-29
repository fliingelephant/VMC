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

    def test_sliced_gradients_buggy_encoding_fails(self):
        """Prove that the old buggy encoding (phys_idx only) fails reconstruction.
        
        This test demonstrates that encoding only phys_idx in p (without cfg_idx)
        causes reconstruction to fail, proving the bug fix was necessary.
        """
        from vmc.experimental.lgt.gi_peps import _link_value_or_zero, _site_cfg_index

        config = GIPEPSConfig(
            shape=(3, 3),
            N=2,
            phys_dim=2,
            Qx=0,
            degeneracy_per_charge=(2, 2),
            charge_of_site=(0, 1),
        )
        strategy = NoTruncation()
        model = GIPEPS(rngs=nnx.Rngs(42), config=config, contraction_strategy=strategy)
        electric_terms = build_electric_terms(config.shape, coeff=0.1, N=config.N)
        plaquette_terms = self._plaquette_terms(config.shape, coeff=0.2)
        operator = GILocalHamiltonian(
            shape=config.shape,
            terms=electric_terms + plaquette_terms,
        )
        key = jax.random.key(42)
        key, init_key = jax.random.split(key)
        init_cfg = model.random_physical_configuration(init_key, n_samples=1)

        # Get full gradients as ground truth
        samples_full, grads_full, _, _, _, _, _ = sequential_sample_with_gradients(
            model,
            operator,
            n_samples=5,
            n_chains=1,
            key=key,
            initial_configuration=init_cfg,
            burn_in=1,
            full_gradient=True,
        )

        # Get sliced gradients (with fixed encoding)
        _, grads_sliced, p_fixed, _, _, _, _ = sequential_sample_with_gradients(
            model,
            operator,
            n_samples=5,
            n_chains=1,
            key=key,
            initial_configuration=init_cfg,
            burn_in=1,
            full_gradient=False,
        )

        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        n_rows, n_cols = config.shape

        def reconstruct_with_fixed_encoding(grads_sliced_i, p_i):
            """Reconstruct using fixed encoding: p = phys_idx * nc + cfg_idx."""
            grad_reconstructed = []
            offset = 0
            for r in range(n_rows):
                for c in range(n_cols):
                    t = tensors[r][c]
                    phys_dim, nc = t.shape[0], t.shape[1]
                    bond_size = t[0, 0].size

                    sliced_grad = grads_sliced_i[offset : offset + bond_size]
                    combined_idx = int(p_i[offset])
                    phys_idx = combined_idx // nc
                    cfg_idx = combined_idx % nc

                    full_site = jnp.zeros(t.size, dtype=grads_sliced_i.dtype)
                    start = (phys_idx * nc + cfg_idx) * bond_size
                    full_site = full_site.at[start : start + bond_size].set(sliced_grad)
                    grad_reconstructed.append(full_site)
                    offset += bond_size
            return jnp.concatenate(grad_reconstructed)

        def reconstruct_with_buggy_encoding(grads_sliced_i, sample):
            """Reconstruct using buggy encoding: p = phys_idx only (cfg_idx unknown).
            
            This simulates what would happen with the old buggy code:
            We only know phys_idx, so we must guess cfg_idx (e.g., assume 0).
            """
            sites, _, _ = GIPEPS.unflatten_sample(sample, config.shape)
            grad_reconstructed = []
            offset = 0
            for r in range(n_rows):
                for c in range(n_cols):
                    t = tensors[r][c]
                    phys_dim, nc = t.shape[0], t.shape[1]
                    bond_size = t[0, 0].size

                    sliced_grad = grads_sliced_i[offset : offset + bond_size]
                    phys_idx = int(sites[r, c])
                    cfg_idx = 0  # BUGGY: we don't know cfg_idx, assume 0

                    full_site = jnp.zeros(t.size, dtype=grads_sliced_i.dtype)
                    start = (phys_idx * nc + cfg_idx) * bond_size
                    full_site = full_site.at[start : start + bond_size].set(sliced_grad)
                    grad_reconstructed.append(full_site)
                    offset += bond_size
            return jnp.concatenate(grad_reconstructed)

        # Test reconstruction
        fixed_matches = 0
        buggy_matches = 0
        for i in range(samples_full.shape[0]):
            recon_fixed = reconstruct_with_fixed_encoding(grads_sliced[i], p_fixed[i])
            recon_buggy = reconstruct_with_buggy_encoding(grads_sliced[i], samples_full[i])

            fixed_ok = jax.device_get(
                jnp.allclose(recon_fixed, grads_full[i], rtol=1e-5, atol=1e-6)
            )
            buggy_ok = jax.device_get(
                jnp.allclose(recon_buggy, grads_full[i], rtol=1e-5, atol=1e-6)
            )

            if fixed_ok:
                fixed_matches += 1
            if buggy_ok:
                buggy_matches += 1

        # Fixed encoding should always work
        self.assertEqual(
            fixed_matches,
            5,
            "Fixed encoding (phys_idx * Nc + cfg_idx) should reconstruct all samples",
        )
        # Buggy encoding should fail for most samples (cfg_idx != 0)
        self.assertLess(
            buggy_matches,
            5,
            f"Buggy encoding (phys_idx only) should fail for samples with cfg_idx != 0, "
            f"but {buggy_matches}/5 matched",
        )

    def test_sliced_gradients_reconstruct_to_full(self):
        """Verify full_gradient=False produces gradients that reconstruct to full."""
        config = GIPEPSConfig(
            shape=(3, 3),
            N=2,
            phys_dim=2,
            Qx=0,
            degeneracy_per_charge=(2, 2),
            charge_of_site=(0, 1),
        )
        strategy = NoTruncation()
        model = GIPEPS(rngs=nnx.Rngs(42), config=config, contraction_strategy=strategy)
        electric_terms = build_electric_terms(config.shape, coeff=0.1, N=config.N)
        plaquette_terms = self._plaquette_terms(config.shape, coeff=0.2)
        operator = GILocalHamiltonian(
            shape=config.shape,
            terms=electric_terms + plaquette_terms,
        )
        key = jax.random.key(42)
        key, init_key = jax.random.split(key)
        init_cfg = model.random_physical_configuration(init_key, n_samples=1)

        # Get full gradients
        key1, key2 = jax.random.split(key)
        samples_full, grads_full, _, _, _, amps_full, _ = sequential_sample_with_gradients(
            model,
            operator,
            n_samples=5,
            n_chains=1,
            key=key1,
            initial_configuration=init_cfg,
            burn_in=1,
            full_gradient=True,
        )

        # Get sliced gradients with same key for same samples
        samples_sliced, grads_sliced, p, _, _, amps_sliced, _ = sequential_sample_with_gradients(
            model,
            operator,
            n_samples=5,
            n_chains=1,
            key=key1,
            initial_configuration=init_cfg,
            burn_in=1,
            full_gradient=False,
        )

        # Verify samples match (same RNG key)
        samples_match = jax.device_get(jnp.array_equal(samples_full, samples_sliced))
        self.assertTrue(samples_match, "Samples should match with same RNG key")

        # Reconstruct full gradients from sliced
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        n_rows, n_cols = config.shape

        for i in range(samples_full.shape[0]):
            # Build param_sizes and nc_per_site for reconstruction
            param_sizes = []
            nc_per_site = []
            for r in range(n_rows):
                for c in range(n_cols):
                    t = tensors[r][c]
                    param_sizes.append(t.size)
                    nc_per_site.append(t.shape[1])

            # Reconstruct full gradient from sliced
            grad_reconstructed = jnp.zeros_like(grads_full[i])
            offset = 0
            param_offset = 0
            for site_idx in range(n_rows * n_cols):
                r, c = divmod(site_idx, n_cols)
                t = tensors[r][c]
                phys_dim = t.shape[0]
                nc = t.shape[1]
                bond_size = t[0, 0].size  # size of (mu_u, mu_d, mu_l, mu_r)

                # Extract this site's sliced gradient and p value
                sliced_grad = grads_sliced[i, offset : offset + bond_size]
                combined_idx = p[i, offset]  # all p values for this site are the same
                phys_idx = combined_idx // nc
                cfg_idx = combined_idx % nc

                # Place in reconstructed array at correct position
                # Full tensor layout: (phys_dim, nc, mu_u, mu_d, mu_l, mu_r).reshape(-1)
                # Position = phys_idx * (nc * bond_size) + cfg_idx * bond_size
                start_pos = param_offset + phys_idx * nc * bond_size + cfg_idx * bond_size
                grad_reconstructed = grad_reconstructed.at[start_pos : start_pos + bond_size].set(
                    sliced_grad
                )

                offset += bond_size
                param_offset += phys_dim * nc * bond_size

            # Compare reconstructed to full
            match = jax.device_get(
                jnp.allclose(grad_reconstructed, grads_full[i], rtol=1e-5, atol=1e-6)
            )
            self.assertTrue(
                match,
                f"Sample {i}: Reconstructed gradient should match full gradient",
            )



if __name__ == "__main__":
    unittest.main()
