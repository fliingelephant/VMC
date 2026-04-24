import jax.numpy as jnp

from vmc.operators.local_terms import LocalHamiltonian, merge_operators, support_span
from vmc.peps.non_abelian_gi import (
    HorizontalLinkCasimirTerm,
    PlaquetteTerm,
    VerticalLinkCasimirTerm,
    build_link_casimir_terms,
    casimir_diagonal,
    link_casimir_energy,
)
from vmc.gauge_groups import SU2


def test_link_casimir_energy_for_horizontal_and_vertical_terms():
    group = SU2(j_max_twice=2)
    h_links = jnp.array([[0, 1], [2, 1]], dtype=jnp.int32)
    v_links = jnp.array([[1, 0, 2]], dtype=jnp.int32)
    diag = casimir_diagonal(group)

    assert link_casimir_energy(
        HorizontalLinkCasimirTerm(row=1, col=0, diag=diag),
        h_links,
        v_links,
    ) == 2.0
    assert link_casimir_energy(
        VerticalLinkCasimirTerm(row=0, col=2, diag=diag),
        h_links,
        v_links,
    ) == 2.0


def test_build_link_casimir_terms_covers_open_lattice_links_in_order():
    terms = build_link_casimir_terms((2, 3), SU2(j_max_twice=1))

    assert [(type(term), term.row, term.col) for term in terms] == [
        (HorizontalLinkCasimirTerm, 0, 0),
        (HorizontalLinkCasimirTerm, 0, 1),
        (HorizontalLinkCasimirTerm, 1, 0),
        (HorizontalLinkCasimirTerm, 1, 1),
        (VerticalLinkCasimirTerm, 0, 0),
        (VerticalLinkCasimirTerm, 0, 1),
        (VerticalLinkCasimirTerm, 0, 2),
    ]
    assert all(jnp.array_equal(term.diag, jnp.asarray([0.0, 0.75])) for term in terms)


def test_link_casimir_terms_bucket_as_diagonal_terms():
    group = SU2(j_max_twice=1)
    term = build_link_casimir_terms((1, 2), group)[0]
    op_a = LocalHamiltonian(
        shape=(1, 2),
        terms=(term,),
        coeffs=(jnp.asarray(0.3),),
    )
    op_b = LocalHamiltonian(
        shape=(1, 2),
        terms=(term,),
        coeffs=(jnp.asarray(0.5),),
    )

    merged, coeff_structure = merge_operators((op_a, op_b), (1, 2))

    assert len(merged.diagonal) == 1
    _term, contributions = merged.diagonal[0]
    assert contributions == ((0, 0), (1, 1))
    assert jnp.array_equal(coeff_structure.build_coeffs(), jnp.asarray([0.3, 0.5]))


def test_plaquette_su2_term_has_2x2_support_and_buckets_as_transition():
    term = PlaquetteTerm(row=0, col=1)
    hamiltonian = LocalHamiltonian(shape=(3, 3), terms=(term,))

    bucketed, _coeff_structure = merge_operators((hamiltonian,), (3, 3))

    assert support_span(term) == (2, 2)
    dr, cols = bucketed.rows[0][0]
    assert dr == 2
    assert cols[1] == ((term, ((0, 0),)),)
