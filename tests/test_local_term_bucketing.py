"""Tests for local operator bucketing validation."""
from __future__ import annotations

import unittest

import jax.numpy as jnp

from vmc.operators.local_terms import (
    HorizontalTwoSiteOperator,
    LocalHamiltonian,
    OneSiteOperator,
    PlaquetteOperator,
    VerticalTwoSiteOperator,
    bucket_operators,
)


class LocalTermBucketingTest(unittest.TestCase):
    """Tests for geometry checks in bucket_operators."""

    def test_horizontal_out_of_bounds_raises(self) -> None:
        ham = LocalHamiltonian(
            shape=(2, 2),
            terms=(HorizontalTwoSiteOperator(row=0, col=1, op=jnp.eye(4)),),
        )
        with self.assertRaises(ValueError):
            bucket_operators(ham.terms, ham.shape)

    def test_vertical_out_of_bounds_raises(self) -> None:
        ham = LocalHamiltonian(
            shape=(2, 2),
            terms=(VerticalTwoSiteOperator(row=1, col=0, op=jnp.eye(4)),),
        )
        with self.assertRaises(ValueError):
            bucket_operators(ham.terms, ham.shape)

    def test_plaquette_out_of_bounds_raises(self) -> None:
        ham = LocalHamiltonian(
            shape=(2, 2),
            terms=(PlaquetteOperator(row=1, col=1, coeff=jnp.asarray(1.0)),),
        )
        with self.assertRaises(ValueError):
            bucket_operators(ham.terms, ham.shape)

    def test_one_site_can_be_routed_to_span_22_for_blockade(self) -> None:
        ham = LocalHamiltonian(
            shape=(2, 2),
            terms=(OneSiteOperator(row=1, col=1, op=jnp.eye(2)),),
        )
        terms = bucket_operators(ham.terms, ham.shape, eval_span=lambda _: (2, 2))
        self.assertEqual(len(terms.rows[1]), 1)
        dr, cols = terms.rows[1][0]
        self.assertEqual(dr, 1)
        self.assertEqual(len(cols[1]), 1)
        self.assertEqual(cols[1][0][2], (1, 1))
        self.assertEqual(terms.rows[0], ())


if __name__ == "__main__":
    unittest.main()
