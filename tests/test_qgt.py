"""Tests for QGT."""
from __future__ import annotations

import unittest

from VMC import config  # noqa: F401

import jax
import jax.numpy as jnp
from flax import nnx

from VMC.models.mps import SimpleMPS
from VMC.models.peps import SimplePEPS, ZipUp
from VMC.core import _value_and_grad_batch
from VMC.qgt import QGT, Jacobian, SlicedJacobian, PhysicalOrdering, SiteOrdering
from VMC.utils.smallo import params_per_site
from VMC.utils.vmc_utils import flatten_samples
from VMC.samplers.sequential import sequential_sample


class QGTTest(unittest.TestCase):

    def test_full_vs_sliced_sample_space(self):
        """Full and sliced Jacobian should produce same OO†."""
        model = SimpleMPS(rngs=nnx.Rngs(0), n_sites=4, bond_dim=2)
        samples = sequential_sample(model, n_samples=32, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)

        amps, grads_full, _ = _value_and_grad_batch(model, samples_flat, full_gradient=True)
        qgt_full = QGT(Jacobian(grads_full / amps[:, None]), space="sample")

        amps, grads, p = _value_and_grad_batch(model, samples_flat, full_gradient=False)
        qgt_sliced = QGT(SlicedJacobian(grads / amps[:, None], p, model.phys_dim), space="sample")

        err = float(jnp.linalg.norm(qgt_full.matrix - qgt_sliced.matrix) / jnp.linalg.norm(qgt_full.matrix))
        self.assertLess(err, 1e-10)

    def test_full_vs_sliced_physical_space(self):
        """Full and sliced Jacobian should produce same O†O via to_dense()."""
        model = SimpleMPS(rngs=nnx.Rngs(0), n_sites=4, bond_dim=2)
        samples = sequential_sample(model, n_samples=32, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)

        amps, grads_full, _ = _value_and_grad_batch(model, samples_flat, full_gradient=True)
        qgt_full = QGT(Jacobian(grads_full / amps[:, None]), space="physical")

        amps, grads, p = _value_and_grad_batch(model, samples_flat, full_gradient=False)
        qgt_sliced = QGT(SlicedJacobian(grads / amps[:, None], p, model.phys_dim), space="physical")

        err = float(jnp.linalg.norm(qgt_full.matrix - qgt_sliced.to_dense()) / jnp.linalg.norm(qgt_full.matrix))
        self.assertLess(err, 1e-10)

    def test_ordering_equivalence(self):
        """PhysicalOrdering and SiteOrdering should produce same QGT."""
        model = SimpleMPS(rngs=nnx.Rngs(0), n_sites=4, bond_dim=2)
        samples = sequential_sample(model, n_samples=32, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)

        amps, grads, p = _value_and_grad_batch(model, samples_flat, full_gradient=False)
        o = grads / amps[:, None]

        qgt_phys = QGT(SlicedJacobian(o, p, model.phys_dim, PhysicalOrdering()), space="sample")
        pps = tuple(params_per_site(model))
        qgt_site = QGT(SlicedJacobian(o, p, model.phys_dim, SiteOrdering(pps)), space="sample")

        err = float(jnp.linalg.norm(qgt_phys.matrix - qgt_site.matrix) / jnp.linalg.norm(qgt_phys.matrix))
        self.assertLess(err, 1e-10)

    def test_solve_residual_mps(self):
        """Solve residual should be small for MPS in physical space."""
        model = SimpleMPS(rngs=nnx.Rngs(0), n_sites=8, bond_dim=4)
        samples = sequential_sample(model, n_samples=256, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)
        diag_shift = 1e-8

        amps, grads, p = _value_and_grad_batch(model, samples_flat, full_gradient=False)
        o = grads / amps[:, None]
        jac = SlicedJacobian(o, p, model.phys_dim)
        qgt = QGT(jac, space="physical")

        # Random RHS in physical space - solve against block-diagonal matrix
        E = jax.random.normal(jax.random.key(1), (qgt.matrix.shape[0],), dtype=jnp.complex128)
        u, _ = qgt.solve(E, diag_shift, project_null=False)

        residual = (qgt.matrix + diag_shift * jnp.eye(qgt.matrix.shape[0])) @ u - E
        rel_err = float(jnp.linalg.norm(residual) / jnp.linalg.norm(E))
        self.assertLess(rel_err, 1e-8)

    def test_solve_residual_peps(self):
        """Solve residual should be small for PEPS with SiteOrdering."""
        phys_dim = 2  # Standard spin-1/2
        model = SimplePEPS(
            rngs=nnx.Rngs(0), shape=(3, 3), bond_dim=2,
            contraction_strategy=ZipUp(4),
        )
        samples = sequential_sample(model, n_samples=64, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)
        diag_shift = 1e-8

        amps, grads, p = _value_and_grad_batch(model, samples_flat, full_gradient=False)
        o = grads / amps[:, None]
        pps = tuple(params_per_site(model))
        jac = SlicedJacobian(o, p, phys_dim, SiteOrdering(pps))
        qgt = QGT(jac, space="physical")

        E = jax.random.normal(jax.random.key(1), (qgt.matrix.shape[0],), dtype=jnp.complex128)
        u, _ = qgt.solve(E, diag_shift, project_null=False)

        residual = (qgt.matrix + diag_shift * jnp.eye(qgt.matrix.shape[0])) @ u - E
        rel_err = float(jnp.linalg.norm(residual) / jnp.linalg.norm(E))
        self.assertLess(rel_err, 1e-8)

    def test_solve_shape(self):
        """Solve should return correct shape."""
        model = SimpleMPS(rngs=nnx.Rngs(0), n_sites=4, bond_dim=2)
        samples = sequential_sample(model, n_samples=32, key=jax.random.key(0))
        jac = SlicedJacobian.from_samples(model, samples)
        qgt = QGT(jac, space="sample")

        dv = jax.random.normal(jax.random.key(1), (32,), dtype=jnp.complex128)
        updates, _ = qgt.solve(dv, diag_shift=1e-8, samples=samples)

        n_full = sum(model.phys_dim * p for p in params_per_site(model))
        self.assertEqual(updates.shape, (n_full,))

    def test_sample_space_solve_residual(self):
        """Sample space solve residual after recovery O†y should be small."""
        model = SimpleMPS(rngs=nnx.Rngs(0), n_sites=8, bond_dim=4)
        samples = sequential_sample(model, n_samples=256, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)
        diag_shift = 1e-4

        # Build full Jacobian for verification
        amps, grads_full, _ = _value_and_grad_batch(model, samples_flat, full_gradient=True)
        O = grads_full / amps[:, None]

        # Sliced solve
        jac = SlicedJacobian.from_samples(model, samples)
        qgt = QGT(jac, space="sample")
        dv = jax.random.normal(jax.random.key(1), (256,), dtype=jnp.complex128)
        updates, _ = qgt.solve(dv, diag_shift, samples=samples)

        # Verify: O @ updates should give the sample-space solution y
        # Then (OO† + λI) @ y ≈ dv (projected)
        y = O @ updates
        dv_proj = dv - jnp.mean(dv)
        residual = (qgt.matrix + diag_shift * jnp.eye(qgt.matrix.shape[0])) @ y - dv_proj
        rel_err = float(jnp.linalg.norm(residual) / jnp.linalg.norm(dv_proj))
        self.assertLess(rel_err, 1e-4)

    def test_weight_null_projection(self):
        """Solution y should be orthogonal to all-ones after null projection."""
        model = SimpleMPS(rngs=nnx.Rngs(0), n_sites=4, bond_dim=2)
        samples = sequential_sample(model, n_samples=32, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)
        diag_shift = 1e-4

        amps, grads_full, _ = _value_and_grad_batch(model, samples_flat, full_gradient=True)
        O = grads_full / amps[:, None]

        jac = SlicedJacobian.from_samples(model, samples)
        qgt = QGT(jac, space="sample")
        dv = jax.random.normal(jax.random.key(1), (32,), dtype=jnp.complex128)
        updates, _ = qgt.solve(dv, diag_shift, samples=samples, project_null=True)

        # y = O @ updates should be orthogonal to all-ones
        y = O @ updates
        ones = jnp.ones(32, dtype=jnp.complex128)
        overlap = jnp.abs(jnp.vdot(ones, y)) / (jnp.linalg.norm(ones) * jnp.linalg.norm(y))
        self.assertLess(float(overlap), 1e-8)

    def test_peps_sample_space(self):
        """PEPS sample space QGT and solve."""
        phys_dim = 2  # Standard spin-1/2
        model = SimplePEPS(
            rngs=nnx.Rngs(0), shape=(3, 3), bond_dim=2,
            contraction_strategy=ZipUp(4),
        )
        samples = sequential_sample(model, n_samples=64, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)
        diag_shift = 1e-4

        amps, grads, p = _value_and_grad_batch(model, samples_flat, full_gradient=False)
        o = grads / amps[:, None]
        pps = tuple(params_per_site(model))
        jac = SlicedJacobian(o, p, phys_dim, SiteOrdering(pps))
        qgt = QGT(jac, space="sample")

        dv = jax.random.normal(jax.random.key(1), (64,), dtype=jnp.complex128)
        updates, _ = qgt.solve(dv, diag_shift, samples=samples)

        n_full = sum(phys_dim * pp for pp in pps)
        self.assertEqual(updates.shape, (n_full,))


if __name__ == "__main__":
    unittest.main()
