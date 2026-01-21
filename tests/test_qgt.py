"""Tests for QGT."""
from __future__ import annotations

import unittest

from VMC import config  # noqa: F401

import jax
import jax.numpy as jnp
from flax import nnx

from VMC.models.mps import MPS
from VMC.models.peps import PEPS, ZipUp
from VMC.core import _value_and_grad
from VMC.qgt import QGT, DiagonalQGT, Jacobian, SlicedJacobian, PhysicalOrdering, SiteOrdering, ParameterSpace, SampleSpace, solve_cholesky
from VMC.preconditioners.preconditioners import DiagonalSolve, _solve_sr
from VMC.utils.smallo import params_per_site
from VMC.utils.vmc_utils import flatten_samples
from VMC.samplers.sequential import sequential_sample


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

    def test_diagonal_qgt_blocks(self):
        """DiagonalQGT should keep only per-site blocks."""
        model = MPS(rngs=nnx.Rngs(0), n_sites=6, bond_dim=3)
        samples = sequential_sample(model, n_samples=128, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)

        amps, grads_full, _ = _value_and_grad(model, samples_flat, full_gradient=True)
        O = grads_full / amps[:, None]
        pps = tuple(params_per_site(model))

        qgt_full = QGT(Jacobian(O), space=ParameterSpace())
        diag_qgt = DiagonalQGT(Jacobian(O), space=ParameterSpace(), params_per_site=pps)

        full = qgt_full.to_dense()
        diag = diag_qgt.to_dense()
        expected = jnp.zeros_like(full)
        i = 0
        for n in pps:
            expected = expected.at[i : i + n, i : i + n].set(full[i : i + n, i : i + n])
            i += n
        err = float(jnp.linalg.norm(diag - expected) / jnp.linalg.norm(expected))
        self.assertLess(err, 1e-10)

    def test_diagonal_solve_parameter_space(self):
        """DiagonalSolve should solve per-site blocks."""
        model = MPS(rngs=nnx.Rngs(0), n_sites=6, bond_dim=3)
        samples = sequential_sample(model, n_samples=128, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)
        diag_shift = 1e-4

        amps, grads_full, _ = _value_and_grad(model, samples_flat, full_gradient=True)
        O = grads_full / amps[:, None]
        pps = tuple(params_per_site(model))
        dv = jax.random.normal(jax.random.key(3), (samples_flat.shape[0],), dtype=jnp.complex128)

        x, _ = _solve_sr(
            DiagonalSolve(solver=solve_cholesky, params_per_site=pps),
            ParameterSpace(),
            Jacobian(O),
            dv,
            diag_shift,
        )
        mean = jnp.mean(O, axis=0)
        rhs = O.conj().T @ dv - mean.conj() * jnp.sum(dv)
        full = QGT(Jacobian(O), space=ParameterSpace()).to_dense()
        expected = jnp.zeros_like(rhs)
        i = 0
        for n in pps:
            block = full[i : i + n, i : i + n] + diag_shift * jnp.eye(n, dtype=full.dtype)
            expected = expected.at[i : i + n].set(solve_cholesky(block, rhs[i : i + n]))
            i += n
        err = float(jnp.linalg.norm(x - expected) / jnp.linalg.norm(expected))
        self.assertLess(err, 1e-10)

    def test_diagonal_sample_space_not_supported(self):
        """DiagonalQGT should reject SampleSpace."""
        model = MPS(rngs=nnx.Rngs(0), n_sites=6, bond_dim=3)
        samples = sequential_sample(model, n_samples=128, key=jax.random.key(0))
        samples_flat = flatten_samples(samples)

        amps, grads_full, _ = _value_and_grad(model, samples_flat, full_gradient=True)
        O = grads_full / amps[:, None]
        pps = tuple(params_per_site(model))
        dv = jax.random.normal(jax.random.key(4), (samples_flat.shape[0],), dtype=jnp.complex128)

        diag_qgt = DiagonalQGT(Jacobian(O), space=SampleSpace(), params_per_site=pps)
        with self.assertRaises(NotImplementedError):
            diag_qgt.to_dense()
        with self.assertRaises(NotImplementedError):
            _solve_sr(
                DiagonalSolve(solver=solve_cholesky, params_per_site=pps),
                SampleSpace(),
                Jacobian(O),
                dv,
                1e-4,
            )


if __name__ == "__main__":
    unittest.main()
