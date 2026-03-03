"""Tests for local operator bucketing validation."""
from __future__ import annotations

import unittest

import jax.numpy as jnp

from vmc.operators.local_terms import (
    DiagonalOperator,
    HorizontalTwoSiteOperator,
    LocalHamiltonian,
    OneSiteOperator,
    PlaquetteOperator,
    VerticalTwoSiteOperator,
    bucket_operators,
    merge_operators,
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
        # New format: (term, span, contributions)
        _term, span, contributions = cols[1][0]
        self.assertEqual(span, (1, 1))
        self.assertEqual(contributions, ((0, 0),))
        self.assertEqual(terms.rows[0], ())


class MergeOperatorsTest(unittest.TestCase):
    """Tests for merge_operators."""

    def test_single_operator_matches_bucket_operators(self) -> None:
        """merge_operators with single op should produce same structure as bucket_operators."""
        shape = (2, 3)
        ham = LocalHamiltonian(
            shape=shape,
            terms=(
                OneSiteOperator(row=0, col=0, op=jnp.eye(2)),
                HorizontalTwoSiteOperator(row=0, col=0, op=jnp.eye(4)),
                VerticalTwoSiteOperator(row=0, col=1, op=jnp.eye(4)),
                DiagonalOperator(sites=((1, 2),), diag=jnp.array([1.0, -1.0])),
            ),
        )
        bucketed = bucket_operators(ham.terms, shape)
        merged, coeff_struct = merge_operators((ham,), shape)

        self.assertEqual(merged.n_ops, 1)
        self.assertEqual(len(merged.diagonal), len(bucketed.diagonal))
        self.assertEqual(len(merged.rows), len(bucketed.rows))

        # Check that coefficient structure is consistent
        self.assertEqual(coeff_struct.n_terms_per_op, (4,))
        self.assertEqual(len(coeff_struct.sources), 4)
        # All schedules should be None for static operator
        self.assertTrue(all(s is None for s in coeff_struct.schedules))
        # build_coeffs should return ones
        coeffs = coeff_struct.build_coeffs()
        self.assertTrue(jnp.allclose(coeffs, jnp.ones(4)))

    def test_multi_operator_contributions(self) -> None:
        """merge_operators with multiple ops should set correct contributions."""
        shape = (2, 2)
        op_a = LocalHamiltonian(
            shape=shape,
            terms=(
                OneSiteOperator(row=0, col=0, op=jnp.eye(2)),
                HorizontalTwoSiteOperator(row=0, col=0, op=jnp.eye(4)),
            ),
        )
        op_b = LocalHamiltonian(
            shape=shape,
            terms=(
                OneSiteOperator(row=1, col=0, op=jnp.eye(2)),
            ),
        )
        merged, coeff_struct = merge_operators((op_a, op_b), shape)

        self.assertEqual(merged.n_ops, 2)
        self.assertEqual(coeff_struct.n_terms_per_op, (2, 1))
        self.assertEqual(len(coeff_struct.sources), 3)
        # sources: [(0,0), (0,1), (1,0)]
        self.assertEqual(coeff_struct.sources[0], (0, 0))
        self.assertEqual(coeff_struct.sources[1], (0, 1))
        self.assertEqual(coeff_struct.sources[2], (1, 0))

        # Check that terms from op_a have op_idx=0 and from op_b have op_idx=1
        for row_passes in merged.rows:
            for dr, cols in row_passes:
                for col_terms in cols:
                    for term, span, contributions in col_terms:
                        for op_idx, coeff_idx in contributions:
                            self.assertIn(op_idx, (0, 1))

    def test_multi_operator_n_ops_field(self) -> None:
        """Verify n_ops is set correctly."""
        shape = (2, 2)
        ops = tuple(
            LocalHamiltonian(
                shape=shape,
                terms=(OneSiteOperator(row=0, col=0, op=jnp.eye(2)),),
            )
            for _ in range(3)
        )
        merged, coeff_struct = merge_operators(ops, shape)
        self.assertEqual(merged.n_ops, 3)
        self.assertEqual(len(coeff_struct.schedules), 3)

    def test_diagonal_contributions_multi_op(self) -> None:
        """Diagonal terms from different operators get separate contributions."""
        shape = (2, 2)
        diag = jnp.array([1.0, -1.0])
        op_a = LocalHamiltonian(
            shape=shape,
            terms=(DiagonalOperator(sites=((0, 0),), diag=diag),),
        )
        op_b = LocalHamiltonian(
            shape=shape,
            terms=(DiagonalOperator(sites=((0, 0),), diag=diag),),
        )
        merged, coeff_struct = merge_operators((op_a, op_b), shape)
        self.assertEqual(len(merged.diagonal), 2)
        # First diagonal term is from op_a (op_idx=0)
        _, contribs_a = merged.diagonal[0]
        self.assertEqual(contribs_a[0][0], 0)
        # Second diagonal term is from op_b (op_idx=1)
        _, contribs_b = merged.diagonal[1]
        self.assertEqual(contribs_b[0][0], 1)


if __name__ == "__main__":
    unittest.main()
