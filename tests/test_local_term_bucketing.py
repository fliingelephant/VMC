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
    merge_operators,
)
from vmc.peps.gi.local_terms import (
    HorizontalMatterHoppingTerm,
    build_electric_terms,
)


class LocalTermBucketingTest(unittest.TestCase):
    """Tests for geometry checks in merge_operators."""

    def test_horizontal_out_of_bounds_raises(self) -> None:
        ham = LocalHamiltonian(
            shape=(2, 2),
            terms=(HorizontalTwoSiteOperator(row=0, col=1, op=jnp.eye(4)),),
        )
        with self.assertRaises(ValueError):
            merge_operators((ham,), ham.shape)

    def test_vertical_out_of_bounds_raises(self) -> None:
        ham = LocalHamiltonian(
            shape=(2, 2),
            terms=(VerticalTwoSiteOperator(row=1, col=0, op=jnp.eye(4)),),
        )
        with self.assertRaises(ValueError):
            merge_operators((ham,), ham.shape)

    def test_plaquette_out_of_bounds_raises(self) -> None:
        ham = LocalHamiltonian(
            shape=(2, 2),
            terms=(PlaquetteOperator(row=1, col=1),),
        )
        with self.assertRaises(ValueError):
            merge_operators((ham,), ham.shape)

    def test_one_site_can_be_routed_to_span_22_for_blockade(self) -> None:
        ham = LocalHamiltonian(
            shape=(2, 2),
            terms=(OneSiteOperator(row=1, col=1, op=jnp.eye(2)),),
        )
        terms, _ = merge_operators((ham,), ham.shape, eval_span=lambda _: (2, 2))
        self.assertEqual(len(terms.rows[1]), 1)
        dr, cols = terms.rows[1][0]
        self.assertEqual(dr, 1)
        self.assertEqual(len(cols[1]), 1)
        _term, contributions = cols[1][0]
        self.assertEqual(contributions, ((0, 0),))
        self.assertEqual(terms.rows[0], ())


class MergeOperatorsTest(unittest.TestCase):
    """Tests for merge_operators."""

    def test_single_operator(self) -> None:
        """merge_operators with single op produces correct structure."""
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
        merged, coeff_struct = merge_operators((ham,), shape)

        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged.diagonal), 1)
        self.assertEqual(tuple(len(coeffs) for coeffs in coeff_struct.base_coeffs), (4,))
        self.assertTrue(all(s is None for s in coeff_struct.schedules))
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

        self.assertEqual(len(merged), 2)
        self.assertEqual(tuple(len(coeffs) for coeffs in coeff_struct.base_coeffs), (2, 1))

        for row_passes in merged.rows:
            for dr, cols in row_passes:
                for col_terms in cols:
                    for term, contributions in col_terms:
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
        self.assertEqual(len(merged), 3)
        self.assertEqual(len(coeff_struct.schedules), 3)

    def test_dedup_value_equal_terms(self) -> None:
        """Value-equal terms (distinct objects) are deduplicated."""
        shape = (2, 2)
        term_a = HorizontalTwoSiteOperator(row=0, col=0, op=jnp.eye(4))
        term_b = HorizontalTwoSiteOperator(row=0, col=0, op=jnp.eye(4))
        self.assertIsNot(term_a, term_b)
        op_a = LocalHamiltonian(shape=shape, terms=(term_a,))
        op_b = LocalHamiltonian(shape=shape, terms=(term_b,))
        merged, _ = merge_operators((op_a, op_b), shape)
        cell = merged.rows[0][0][1][0]  # row=0, first dr pass, col=0
        self.assertEqual(len(cell), 1)
        _, contributions = cell[0]
        self.assertEqual(len(contributions), 2)
        self.assertEqual(contributions[0][0], 0)  # op_a
        self.assertEqual(contributions[1][0], 1)  # op_b

    def test_diagonal_dedup_value_equal(self) -> None:
        """Value-equal diagonal terms (distinct objects) are deduplicated."""
        shape = (2, 2)
        term_a = DiagonalOperator(sites=((0, 0),), diag=jnp.array([1.0, -1.0]))
        term_b = DiagonalOperator(sites=((0, 0),), diag=jnp.array([1.0, -1.0]))
        self.assertIsNot(term_a, term_b)
        op_a = LocalHamiltonian(shape=shape, terms=(term_a,))
        op_b = LocalHamiltonian(shape=shape, terms=(term_b,))
        merged, _ = merge_operators((op_a, op_b), shape)
        self.assertEqual(len(merged.diagonal), 1)
        _, contribs = merged.diagonal[0]
        self.assertEqual(len(contribs), 2)
        self.assertEqual(contribs[0][0], 0)
        self.assertEqual(contribs[1][0], 1)

    def test_same_plaquette_with_different_static_coeffs_is_deduplicated(self) -> None:
        """Equal plaquette geometry should share one bucket entry across operators."""
        shape = (3, 3)
        op_a = LocalHamiltonian(
            shape=shape,
            terms=(PlaquetteOperator(row=0, col=1),),
            coeffs=(jnp.asarray(-1.0),),
        )
        op_b = LocalHamiltonian(
            shape=shape,
            terms=(PlaquetteOperator(row=0, col=1),),
            coeffs=(jnp.asarray(0.5),),
        )

        merged, coeff_struct = merge_operators((op_a, op_b), shape)

        cell = merged.rows[0][0][1][1]  # row=0, dr=2, col=1
        self.assertEqual(len(cell), 1)
        _, contributions = cell[0]
        self.assertEqual(len(contributions), 2)
        self.assertEqual(contributions[0][0], 0)
        self.assertEqual(contributions[1][0], 1)
        self.assertTrue(
            jnp.allclose(
                coeff_struct.build_coeffs(),
                jnp.asarray([-1.0, 0.5]),
            )
        )

    def test_same_electric_link_with_different_static_coeffs_is_deduplicated(self) -> None:
        """Equal electric-link geometry should share one diagonal bucket entry."""
        shape = (2, 2)
        term = build_electric_terms(shape, N=2)[0]
        op_a = LocalHamiltonian(shape=shape, terms=(term,), coeffs=(jnp.asarray(0.3),))
        op_b = LocalHamiltonian(shape=shape, terms=(term,), coeffs=(jnp.asarray(0.5),))

        merged, coeff_struct = merge_operators((op_a, op_b), shape)

        self.assertEqual(len(merged.diagonal), 1)
        _, contributions = merged.diagonal[0]
        self.assertEqual(len(contributions), 2)
        self.assertTrue(
            jnp.allclose(
                coeff_struct.build_coeffs(),
                jnp.asarray([0.3, 0.5]),
            )
        )

    def test_same_gi_hopping_with_different_static_coeffs_is_deduplicated(self) -> None:
        """Equal GI hopping geometry should share one transition bucket entry."""
        shape = (2, 2)
        term = HorizontalMatterHoppingTerm(row=0, col=0)
        op_a = LocalHamiltonian(shape=shape, terms=(term,), coeffs=(jnp.asarray(0.7),))
        op_b = LocalHamiltonian(shape=shape, terms=(term,), coeffs=(jnp.asarray(-0.2),))

        merged, coeff_struct = merge_operators((op_a, op_b), shape)

        cell = merged.rows[0][0][1][0]  # row=0, dr=1, col=0
        self.assertEqual(len(cell), 1)
        _, contributions = cell[0]
        self.assertEqual(len(contributions), 2)
        self.assertTrue(
            jnp.allclose(
                coeff_struct.build_coeffs(),
                jnp.asarray([0.7, -0.2]),
            )
        )

    def test_scalar_coeff_for_single_term_is_accepted(self) -> None:
        shape = (2, 2)
        op = LocalHamiltonian(
            shape=shape,
            terms=(PlaquetteOperator(row=0, col=0),),
            coeffs=jnp.asarray(2.5),
        )
        _, coeff_struct = merge_operators((op,), shape)
        self.assertTrue(jnp.allclose(coeff_struct.build_coeffs(), jnp.asarray([2.5])))

    def test_wrapper_with_terms_only_defaults_to_unit_coeffs(self) -> None:
        shape = (2, 2)

        class Wrapper:
            def __init__(self, terms):
                self.terms = terms

        wrapped = Wrapper((OneSiteOperator(row=0, col=0, op=jnp.eye(2)),))
        merged, coeff_struct = merge_operators((wrapped,), shape)
        self.assertEqual(len(merged), 1)
        self.assertTrue(jnp.allclose(coeff_struct.build_coeffs(), jnp.asarray([1.0])))


if __name__ == "__main__":
    unittest.main()
