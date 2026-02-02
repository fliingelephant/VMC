"""Tests for QGT."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.models.mps import MPS
from vmc.core import _value_and_grad
from vmc.qgt import QGT, Jacobian, SlicedJacobian, SliceOrdering, SiteOrdering, ParameterSpace, SampleSpace, solve_cholesky
from vmc.utils.smallo import params_per_site, sliced_dims
from vmc.utils.vmc_utils import flatten_samples
from vmc.samplers.sequential import sequential_sample


class QGTTest(unittest.TestCase):

    def test_full_vs_sliced_sample_space(self):
        """Full and sliced Jacobian should produce same OO†."""
        model = MPS(rngs=nnx.Rngs(0), n_sites=6, bond_dim=3)
        samples = sequential_sample(
            model,
            n_samples=128,
            key=jax.random.key(0),
            initial_configuration=model.random_physical_configuration(
                jax.random.key(1), n_samples=1
            ),
        )
        samples_flat = flatten_samples(samples)

        amps, grads_full, _ = _value_and_grad(model, samples_flat, full_gradient=True)
        qgt_full = QGT(Jacobian(grads_full / amps[:, None]), space=SampleSpace())

        amps, grads, p = _value_and_grad(model, samples_flat, full_gradient=False)
        qgt_sliced = QGT(SlicedJacobian(grads / amps[:, None], p, sliced_dims(model)), space=SampleSpace())

        err = float(jnp.linalg.norm(qgt_full.to_dense() - qgt_sliced.to_dense()) / jnp.linalg.norm(qgt_full.to_dense()))
        self.assertLess(err, 1e-10)

    def test_full_vs_sliced_parameter_space(self):
        """Full and sliced Jacobian should produce same O†O via to_dense()."""
        model = MPS(rngs=nnx.Rngs(0), n_sites=6, bond_dim=3)
        samples = sequential_sample(
            model,
            n_samples=128,
            key=jax.random.key(0),
            initial_configuration=model.random_physical_configuration(
                jax.random.key(1), n_samples=1
            ),
        )
        samples_flat = flatten_samples(samples)

        amps, grads_full, _ = _value_and_grad(model, samples_flat, full_gradient=True)
        qgt_full = QGT(Jacobian(grads_full / amps[:, None]), space=ParameterSpace())

        amps, grads, p = _value_and_grad(model, samples_flat, full_gradient=False)
        pps = tuple(params_per_site(model))
        qgt_sliced = QGT(
            SlicedJacobian(grads / amps[:, None], p, sliced_dims(model), SiteOrdering(pps)),
            space=ParameterSpace(),
        )

        err = float(jnp.linalg.norm(qgt_full.to_dense() - qgt_sliced.to_dense()) / jnp.linalg.norm(qgt_full.to_dense()))
        self.assertLess(err, 1e-10)

    def test_ordering_equivalence(self):
        """SliceOrdering and SiteOrdering should produce same QGT."""
        model = MPS(rngs=nnx.Rngs(0), n_sites=6, bond_dim=3)
        samples = sequential_sample(
            model,
            n_samples=128,
            key=jax.random.key(0),
            initial_configuration=model.random_physical_configuration(
                jax.random.key(1), n_samples=1
            ),
        )
        samples_flat = flatten_samples(samples)

        amps, grads, p = _value_and_grad(model, samples_flat, full_gradient=False)
        o = grads / amps[:, None]

        sd = sliced_dims(model)
        qgt_phys = QGT(SlicedJacobian(o, p, sd, SliceOrdering()), space=SampleSpace())
        pps = tuple(params_per_site(model))
        qgt_site = QGT(SlicedJacobian(o, p, sd, SiteOrdering(pps)), space=SampleSpace())

        err = float(jnp.linalg.norm(qgt_phys.to_dense() - qgt_site.to_dense()) / jnp.linalg.norm(qgt_phys.to_dense()))
        self.assertLess(err, 1e-10)

    def test_solve_residual_parameter_space(self):
        """Solve residual should be small for parameter space."""
        model = MPS(rngs=nnx.Rngs(0), n_sites=8, bond_dim=4)
        samples = sequential_sample(
            model,
            n_samples=512,
            key=jax.random.key(0),
            initial_configuration=model.random_physical_configuration(
                jax.random.key(1), n_samples=1
            ),
        )
        samples_flat = flatten_samples(samples)
        diag_shift = 1e-4

        amps, grads, p = _value_and_grad(model, samples_flat, full_gradient=False)
        o = grads / amps[:, None]
        jac = SlicedJacobian(o, p, sliced_dims(model))
        qgt = QGT(jac, space=ParameterSpace())

        S = qgt.to_dense()
        mat = S + diag_shift * jnp.eye(S.shape[0], dtype=S.dtype)
        rhs = jax.random.normal(jax.random.key(1), (S.shape[0],), dtype=jnp.complex128)
        x = solve_cholesky(mat, rhs)

        residual = mat @ x - rhs
        rel_err = float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs))
        self.assertLess(rel_err, 1e-4)

    def test_solve_residual_sample_space(self):
        """Solve residual should be small for sample space."""
        model = MPS(rngs=nnx.Rngs(0), n_sites=8, bond_dim=4)
        samples = sequential_sample(
            model,
            n_samples=512,
            key=jax.random.key(0),
            initial_configuration=model.random_physical_configuration(
                jax.random.key(1), n_samples=1
            ),
        )
        samples_flat = flatten_samples(samples)
        diag_shift = 1e-4

        amps, grads, p = _value_and_grad(model, samples_flat, full_gradient=False)
        o = grads / amps[:, None]
        jac = SlicedJacobian(o, p, sliced_dims(model))
        qgt = QGT(jac, space=SampleSpace())

        S = qgt.to_dense()
        mat = S + diag_shift * jnp.eye(S.shape[0], dtype=S.dtype)
        rhs = jax.random.normal(jax.random.key(1), (samples_flat.shape[0],), dtype=jnp.complex128)
        x = solve_cholesky(mat, rhs)

        residual = mat @ x - rhs
        rel_err = float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs))
        self.assertLess(rel_err, 1e-4)

    def test_matvec_vs_to_dense(self):
        """Matvec should match explicit matrix multiplication."""
        model = MPS(rngs=nnx.Rngs(0), n_sites=6, bond_dim=3)
        samples = sequential_sample(
            model,
            n_samples=128,
            key=jax.random.key(0),
            initial_configuration=model.random_physical_configuration(
                jax.random.key(1), n_samples=1
            ),
        )
        samples_flat = flatten_samples(samples)

        amps, grads, p = _value_and_grad(model, samples_flat, full_gradient=False)
        o = grads / amps[:, None]
        jac = SlicedJacobian(o, p, sliced_dims(model))

        # Test physical space
        qgt_phys = QGT(jac, space=ParameterSpace())
        v_phys = jax.random.normal(jax.random.key(1), (qgt_phys.to_dense().shape[0],), dtype=jnp.complex128)
        matvec_result = qgt_phys @ v_phys
        dense_result = qgt_phys.to_dense() @ v_phys
        err = float(jnp.linalg.norm(matvec_result - dense_result) / jnp.linalg.norm(dense_result))
        self.assertLess(err, 1e-10)

        # Test sample space
        qgt_sample = QGT(jac, space=SampleSpace())
        v_sample = jax.random.normal(jax.random.key(2), (samples_flat.shape[0],), dtype=jnp.complex128)
        matvec_result = qgt_sample @ v_sample
        dense_result = qgt_sample.to_dense() @ v_sample
        err = float(jnp.linalg.norm(matvec_result - dense_result) / jnp.linalg.norm(dense_result))
        self.assertLess(err, 1e-10)

    def test_peps_slice_ordering_reorder(self):
        """SliceOrdering Jacobian after reordering should match full Jacobian for PEPS."""
        from vmc.models.peps import PEPS, NoTruncation
        from vmc.qgt.qgt import _sliced_dense_blocks

        model = PEPS(rngs=nnx.Rngs(0), shape=(2, 3), bond_dim=2, contraction_strategy=NoTruncation())
        samples = sequential_sample(
            model,
            n_samples=32,
            key=jax.random.key(0),
            initial_configuration=model.random_physical_configuration(
                jax.random.key(1), n_samples=1
            ),
        )
        samples_flat = flatten_samples(samples)

        amps, grads_full, _ = _value_and_grad(model, samples_flat, full_gradient=True)
        amps, grads, p = _value_and_grad(model, samples_flat, full_gradient=False)

        o_full = grads_full / amps[:, None]
        o = grads / amps[:, None]
        sd = sliced_dims(model)
        pps = tuple(params_per_site(model))

        jac_slice = SlicedJacobian(o, p, sd, SliceOrdering())
        O_slice = _sliced_dense_blocks(jac_slice)

        # Build permutation and reorder
        perm = []
        total = sum(pps)
        site_offset = 0
        for site_idx, n in enumerate(pps):
            for k in range(sd[site_idx]):
                base = k * total + site_offset
                perm.extend(range(base, base + n))
            site_offset += n
        O_slice_reordered = O_slice[:, jnp.asarray(perm)]

        self.assertTrue(jnp.allclose(O_slice_reordered, o_full, rtol=1e-10))

    def test_peps_slice_ordering_solve(self):
        """SliceOrdering SR solve with _reorder_updates should match SiteOrdering solve."""
        from vmc.models.peps import PEPS, NoTruncation
        from vmc.preconditioners.preconditioners import _reorder_updates

        model = PEPS(rngs=nnx.Rngs(0), shape=(2, 3), bond_dim=2, contraction_strategy=NoTruncation())
        samples = sequential_sample(
            model,
            n_samples=64,
            key=jax.random.key(0),
            initial_configuration=model.random_physical_configuration(
                jax.random.key(1), n_samples=1
            ),
        )
        samples_flat = flatten_samples(samples)

        amps, grads, p = _value_and_grad(model, samples_flat, full_gradient=False)
        o = grads / amps[:, None]
        sd = sliced_dims(model)
        pps = tuple(params_per_site(model))
        diag_shift = 1e-4

        # Solve with SliceOrdering
        jac_slice = SlicedJacobian(o, p, sd, SliceOrdering())
        qgt_slice = QGT(jac_slice, space=ParameterSpace())
        S_slice = qgt_slice.to_dense()
        mat_slice = S_slice + diag_shift * jnp.eye(S_slice.shape[0], dtype=S_slice.dtype)
        rhs_slice = jax.random.normal(jax.random.key(1), (S_slice.shape[0],), dtype=jnp.complex128)
        x_slice = solve_cholesky(mat_slice, rhs_slice)
        x_slice_reordered = _reorder_updates(SliceOrdering(), x_slice, pps, sd)

        # Solve with SiteOrdering
        jac_site = SlicedJacobian(o, p, sd, SiteOrdering(pps))
        qgt_site = QGT(jac_site, space=ParameterSpace())
        S_site = qgt_site.to_dense()
        mat_site = S_site + diag_shift * jnp.eye(S_site.shape[0], dtype=S_site.dtype)
        # RHS must also be reordered for fair comparison
        rhs_site = _reorder_updates(SliceOrdering(), rhs_slice, pps, sd)
        x_site = solve_cholesky(mat_site, rhs_site)

        err = float(jnp.linalg.norm(x_slice_reordered - x_site) / jnp.linalg.norm(x_site))
        self.assertLess(err, 1e-6)

    def test_gipeps_slice_ordering_reorder(self):
        """SliceOrdering Jacobian after reordering should match full Jacobian for GIPEPS."""
        from vmc.experimental.lgt.gi_peps import GIPEPS, GIPEPSConfig
        from vmc.experimental.lgt.gi_local_terms import GILocalHamiltonian, build_electric_terms
        from vmc.experimental.lgt.gi_sampler import sequential_sample_with_gradients
        from vmc.models.peps import NoTruncation
        from vmc.operators import PlaquetteTerm
        from vmc.qgt.qgt import _sliced_dense_blocks

        config = GIPEPSConfig(
            shape=(2, 2), N=2, phys_dim=2, Qx=0,
            degeneracy_per_charge=(2, 2), charge_of_site=(0, 1),
        )
        model = GIPEPS(rngs=nnx.Rngs(123), config=config, contraction_strategy=NoTruncation())

        sd = sliced_dims(model)
        pps = tuple(params_per_site(model))

        key = jax.random.key(123)
        init_cfg = model.random_physical_configuration(key, n_samples=1)

        electric_terms = build_electric_terms(config.shape, coeff=0.1, N=config.N)
        plaquette_terms = tuple(
            PlaquetteTerm(row=r, col=c, coeff=0.2)
            for r in range(config.shape[0] - 1)
            for c in range(config.shape[1] - 1)
        )
        operator = GILocalHamiltonian(shape=config.shape, terms=electric_terms + plaquette_terms)

        _, grads_full, _, _, _, amps, _ = sequential_sample_with_gradients(
            model, operator, n_samples=8, n_chains=1, key=key,
            initial_configuration=init_cfg, burn_in=1, full_gradient=True,
        )
        _, grads, p, _, _, _, _ = sequential_sample_with_gradients(
            model, operator, n_samples=8, n_chains=1, key=key,
            initial_configuration=init_cfg, burn_in=1, full_gradient=False,
        )

        o_full = grads_full / amps[:, None]
        o = grads / amps[:, None]

        jac_slice = SlicedJacobian(o, p, sd, SliceOrdering())
        O_slice = _sliced_dense_blocks(jac_slice)

        # Build permutation and reorder
        perm = []
        total = sum(pps)
        site_offset = 0
        for site_idx, n in enumerate(pps):
            for k in range(sd[site_idx]):
                base = k * total + site_offset
                perm.extend(range(base, base + n))
            site_offset += n
        O_slice_reordered = O_slice[:, jnp.asarray(perm)]

        self.assertTrue(jnp.allclose(O_slice_reordered, o_full, rtol=1e-5, atol=1e-6))

    def test_gipeps_slice_ordering_solve(self):
        """SliceOrdering SR solve with _reorder_updates should match SiteOrdering solve for GIPEPS."""
        from vmc.experimental.lgt.gi_peps import GIPEPS, GIPEPSConfig
        from vmc.experimental.lgt.gi_local_terms import GILocalHamiltonian, build_electric_terms
        from vmc.experimental.lgt.gi_sampler import sequential_sample_with_gradients
        from vmc.models.peps import NoTruncation
        from vmc.operators import PlaquetteTerm
        from vmc.preconditioners.preconditioners import _reorder_updates

        config = GIPEPSConfig(
            shape=(2, 2), N=2, phys_dim=2, Qx=0,
            degeneracy_per_charge=(2, 2), charge_of_site=(0, 1),
        )
        model = GIPEPS(rngs=nnx.Rngs(123), config=config, contraction_strategy=NoTruncation())

        sd = sliced_dims(model)
        pps = tuple(params_per_site(model))

        key = jax.random.key(123)
        init_cfg = model.random_physical_configuration(key, n_samples=1)

        electric_terms = build_electric_terms(config.shape, coeff=0.1, N=config.N)
        plaquette_terms = tuple(
            PlaquetteTerm(row=r, col=c, coeff=0.2)
            for r in range(config.shape[0] - 1)
            for c in range(config.shape[1] - 1)
        )
        operator = GILocalHamiltonian(shape=config.shape, terms=electric_terms + plaquette_terms)

        _, grads, p, _, _, amps, _ = sequential_sample_with_gradients(
            model, operator, n_samples=16, n_chains=1, key=key,
            initial_configuration=init_cfg, burn_in=1, full_gradient=False,
        )

        o = grads / amps[:, None]
        diag_shift = 1e-4

        # Solve with SliceOrdering
        jac_slice = SlicedJacobian(o, p, sd, SliceOrdering())
        qgt_slice = QGT(jac_slice, space=ParameterSpace())
        S_slice = qgt_slice.to_dense()
        mat_slice = S_slice + diag_shift * jnp.eye(S_slice.shape[0], dtype=S_slice.dtype)
        rhs_slice = jax.random.normal(jax.random.key(1), (S_slice.shape[0],), dtype=jnp.complex128)
        x_slice = solve_cholesky(mat_slice, rhs_slice)
        x_slice_reordered = _reorder_updates(SliceOrdering(), x_slice, pps, sd)

        # Solve with SiteOrdering
        jac_site = SlicedJacobian(o, p, sd, SiteOrdering(pps))
        qgt_site = QGT(jac_site, space=ParameterSpace())
        S_site = qgt_site.to_dense()
        mat_site = S_site + diag_shift * jnp.eye(S_site.shape[0], dtype=S_site.dtype)
        # RHS must also be reordered for fair comparison
        rhs_site = _reorder_updates(SliceOrdering(), rhs_slice, pps, sd)
        x_site = solve_cholesky(mat_site, rhs_site)

        err = float(jnp.linalg.norm(x_slice_reordered - x_site) / jnp.linalg.norm(x_site))
        self.assertLess(err, 1e-5)


if __name__ == "__main__":
    unittest.main()
