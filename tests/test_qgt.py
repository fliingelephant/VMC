"""Tests for QGT."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.models.mps import MPS
from vmc.core import _value_and_grad
from vmc.qgt import QGT, Jacobian, SlicedJacobian, PhysicalOrdering, SiteOrdering, ParameterSpace, SampleSpace, solve_cholesky
from vmc.utils.smallo import params_per_site
from vmc.utils.vmc_utils import flatten_samples
from vmc.samplers.sequential import sequential_sample


class QGTTest(unittest.TestCase):

    def test_full_vs_sliced_sample_space(self):
        """Full and sliced Jacobian should produce same OO†."""
        model = MPS(rngs=nnx.Rngs(0), n_sites=6, bond_dim=3)
        samples = sequential_sample(model, n_samples=128, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)

        amps, grads_full, _ = _value_and_grad(model, samples_flat, full_gradient=True)
        qgt_full = QGT(Jacobian(grads_full / amps[:, None]), space=SampleSpace())

        amps, grads, p = _value_and_grad(model, samples_flat, full_gradient=False)
        qgt_sliced = QGT(SlicedJacobian(grads / amps[:, None], p, model.phys_dim), space=SampleSpace())

        err = float(jnp.linalg.norm(qgt_full.to_dense() - qgt_sliced.to_dense()) / jnp.linalg.norm(qgt_full.to_dense()))
        self.assertLess(err, 1e-10)

    def test_full_vs_sliced_parameter_space(self):
        """Full and sliced Jacobian should produce same O†O via to_dense()."""
        model = MPS(rngs=nnx.Rngs(0), n_sites=6, bond_dim=3)
        samples = sequential_sample(model, n_samples=128, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)

        amps, grads_full, _ = _value_and_grad(model, samples_flat, full_gradient=True)
        qgt_full = QGT(Jacobian(grads_full / amps[:, None]), space=ParameterSpace())

        amps, grads, p = _value_and_grad(model, samples_flat, full_gradient=False)
        pps = tuple(params_per_site(model))
        qgt_sliced = QGT(
            SlicedJacobian(grads / amps[:, None], p, model.phys_dim, SiteOrdering(pps)),
            space=ParameterSpace(),
        )

        err = float(jnp.linalg.norm(qgt_full.to_dense() - qgt_sliced.to_dense()) / jnp.linalg.norm(qgt_full.to_dense()))
        self.assertLess(err, 1e-10)

    def test_ordering_equivalence(self):
        """PhysicalOrdering and SiteOrdering should produce same QGT."""
        model = MPS(rngs=nnx.Rngs(0), n_sites=6, bond_dim=3)
        samples = sequential_sample(model, n_samples=128, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)

        amps, grads, p = _value_and_grad(model, samples_flat, full_gradient=False)
        o = grads / amps[:, None]

        qgt_phys = QGT(SlicedJacobian(o, p, model.phys_dim, PhysicalOrdering()), space=SampleSpace())
        pps = tuple(params_per_site(model))
        qgt_site = QGT(SlicedJacobian(o, p, model.phys_dim, SiteOrdering(pps)), space=SampleSpace())

        err = float(jnp.linalg.norm(qgt_phys.to_dense() - qgt_site.to_dense()) / jnp.linalg.norm(qgt_phys.to_dense()))
        self.assertLess(err, 1e-10)

    def test_solve_residual_parameter_space(self):
        """Solve residual should be small for parameter space."""
        model = MPS(rngs=nnx.Rngs(0), n_sites=8, bond_dim=4)
        samples = sequential_sample(model, n_samples=512, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)
        diag_shift = 1e-4

        amps, grads, p = _value_and_grad(model, samples_flat, full_gradient=False)
        o = grads / amps[:, None]
        jac = SlicedJacobian(o, p, model.phys_dim)
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
        samples = sequential_sample(model, n_samples=512, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)
        diag_shift = 1e-4

        amps, grads, p = _value_and_grad(model, samples_flat, full_gradient=False)
        o = grads / amps[:, None]
        jac = SlicedJacobian(o, p, model.phys_dim)
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
        samples = sequential_sample(model, n_samples=128, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)

        amps, grads, p = _value_and_grad(model, samples_flat, full_gradient=False)
        o = grads / amps[:, None]
        jac = SlicedJacobian(o, p, model.phys_dim)

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

if __name__ == "__main__":
    unittest.main()
