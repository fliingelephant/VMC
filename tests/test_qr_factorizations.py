"""Tests for custom Householder QR implementations."""
from __future__ import annotations

import unittest
from itertools import product

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp

from vmc.utils.factorizations import _qr_compactwy, _qr_householder

SHAPES = [(32, 16), (16, 32), (16, 16), (2, 16), (64, 8), (4, 4), (1, 5), (7, 1)]
DTYPES = [jnp.float64, jnp.complex128]
BATCH_SIZES = [(), (4,), (1024,), (2, 8)]
TOL = 1e-12


def _random_matrix(key, shape, dtype):
    a = jax.random.normal(key, shape, dtype=jnp.float64)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        a = a + 1j * jax.random.normal(jax.random.fold_in(key, 1), shape, dtype=jnp.float64)
    return a.astype(dtype)


class _QRTestBase:
    """Mixin providing shared QR correctness checks."""

    qr_fn = None

    def _check_qr(self, a, tag=""):
        q, r = self.qr_fn(a)
        m, n = a.shape[-2], a.shape[-1]
        k = min(m, n)

        # Q shape
        self.assertEqual(q.shape, a.shape[:-1] + (k,), f"Q shape {tag}")
        # R shape
        self.assertEqual(r.shape, a.shape[:-2] + (k, n), f"R shape {tag}")

        # Reconstruction: QR ≈ A
        recon = jnp.linalg.norm(q @ r - a, axis=(-2, -1))
        self.assertTrue(
            jnp.all(recon < TOL),
            f"QR reconstruction {tag}: max err {jnp.max(recon):.2e}",
        )

        # Orthogonality: Q^H Q ≈ I
        eye_k = jnp.eye(k, dtype=a.dtype)
        qtq = jnp.einsum("...ji,...jk->...ik", q.conj(), q)
        ortho = jnp.linalg.norm(qtq - eye_k, axis=(-2, -1))
        self.assertTrue(
            jnp.all(ortho < TOL),
            f"Q orthogonality {tag}: max err {jnp.max(ortho):.2e}",
        )

        # R upper triangular
        tri_err = jnp.linalg.norm(r - jnp.triu(r), axis=(-2, -1))
        self.assertTrue(
            jnp.all(tri_err < TOL),
            f"R triangularity {tag}: max err {jnp.max(tri_err):.2e}",
        )

    def test_shapes_and_dtypes(self):
        key = jax.random.key(42)
        for shape, dtype in product(SHAPES, DTYPES):
            with self.subTest(shape=shape, dtype=dtype):
                a = _random_matrix(key, shape, dtype)
                self._check_qr(a, tag=f"{shape} {dtype}")

    def test_batched(self):
        key = jax.random.key(7)
        shape = (32, 16)
        dtype = jnp.complex128
        for batch in BATCH_SIZES:
            if batch == ():
                continue
            with self.subTest(batch=batch):
                full_shape = batch + shape
                a = _random_matrix(key, full_shape, dtype)
                self._check_qr(a, tag=f"batch={batch}")

    def test_vmap(self):
        key = jax.random.key(99)
        for batch_size in [4, 256]:
            with self.subTest(batch_size=batch_size):
                keys = jax.random.split(key, batch_size)
                a = jax.vmap(
                    lambda k: _random_matrix(k, (32, 16), jnp.complex128)
                )(keys)
                q, r = jax.vmap(self.qr_fn)(a)
                recon = jnp.max(
                    jnp.linalg.norm(
                        jnp.einsum("...ij,...jk->...ik", q, r) - a,
                        axis=(-2, -1),
                    )
                )
                self.assertLess(float(recon), TOL, f"vmap batch={batch_size}")

    def test_zero_matrix(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                a = jnp.zeros((8, 4), dtype=dtype)
                q, r = self.qr_fn(a)
                self.assertEqual(q.shape, (8, 4))
                self.assertEqual(r.shape, (4, 4))

    def test_identity(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                a = jnp.eye(6, dtype=dtype)
                self._check_qr(a, tag=f"identity {dtype}")

    def test_single_column(self):
        key = jax.random.key(11)
        a = _random_matrix(key, (10, 1), jnp.complex128)
        self._check_qr(a, tag="single column")

    def test_single_row(self):
        key = jax.random.key(12)
        a = _random_matrix(key, (1, 10), jnp.complex128)
        self._check_qr(a, tag="single row")


class CompactWYTest(_QRTestBase, unittest.TestCase):
    qr_fn = staticmethod(_qr_compactwy)


class SequentialHouseholderTest(_QRTestBase, unittest.TestCase):
    qr_fn = staticmethod(_qr_householder)


class CrossConsistencyTest(unittest.TestCase):
    """Verify both implementations agree."""

    def test_q_spans_same_subspace(self):
        key = jax.random.key(55)
        for shape, dtype in product([(32, 16), (8, 16), (16, 16)], DTYPES):
            with self.subTest(shape=shape, dtype=dtype):
                a = _random_matrix(key, shape, dtype)
                q_wy, _ = _qr_compactwy(a)
                q_seq, _ = _qr_householder(a)
                # Q columns span the same subspace: Q_wy^H Q_seq should be unitary
                cross = q_wy.conj().T @ q_seq
                k = min(shape)
                err = jnp.linalg.norm(cross @ cross.conj().T - jnp.eye(k, dtype=dtype))
                self.assertLess(
                    float(err), TOL, f"subspace mismatch {shape} {dtype}",
                )


if __name__ == "__main__":
    unittest.main()
