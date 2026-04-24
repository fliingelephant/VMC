"""Local terms for pure-gauge SU(2) GI-PEPS."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from plum import dispatch

from vmc.operators.local_terms import DiagonalOperator, TransitionOperator, support_span
from vmc.peps.su2_gi.group import SU2

__all__ = [
    "HorizontalLinkCasimirTerm",
    "PlaquetteSU2Term",
    "VerticalLinkCasimirTerm",
    "build_link_casimir_terms",
    "casimir_diagonal",
    "link_casimir_energy",
]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class HorizontalLinkCasimirTerm(DiagonalOperator):
    """Diagonal electric-energy term on one horizontal link."""

    row: int
    col: int

    def __init__(self, *, row: int, col: int, diag: jax.Array) -> None:
        super().__init__(sites=((row, col),), diag=diag)
        object.__setattr__(self, "row", row)
        object.__setattr__(self, "col", col)

    def tree_flatten(self):
        return (self.diag,), (self.row, self.col)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (diag,) = children
        row, col = aux_data
        return cls(row=row, col=col, diag=diag)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class VerticalLinkCasimirTerm(DiagonalOperator):
    """Diagonal electric-energy term on one vertical link."""

    row: int
    col: int

    def __init__(self, *, row: int, col: int, diag: jax.Array) -> None:
        super().__init__(sites=((row, col),), diag=diag)
        object.__setattr__(self, "row", row)
        object.__setattr__(self, "col", col)

    def tree_flatten(self):
        return (self.diag,), (self.row, self.col)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (diag,) = children
        row, col = aux_data
        return cls(row=row, col=col, diag=diag)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PlaquetteSU2Term(TransitionOperator):
    """Magnetic plaquette term on the square with top-left corner ``(row, col)``."""

    row: int
    col: int

    def tree_flatten(self):
        return (), (self.row, self.col)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        row, col = aux_data
        return cls(row=row, col=col)


@support_span.dispatch
def support_span(_: PlaquetteSU2Term) -> tuple[int, int]:
    return 2, 2


@dispatch
def link_casimir_energy(
    term: HorizontalLinkCasimirTerm,
    h_links: jax.Array,
    v_links: jax.Array,
) -> jax.Array:
    del v_links
    return term.diag[h_links[term.row, term.col]]


@link_casimir_energy.dispatch
def link_casimir_energy(
    term: VerticalLinkCasimirTerm,
    h_links: jax.Array,
    v_links: jax.Array,
) -> jax.Array:
    del h_links
    return term.diag[v_links[term.row, term.col]]


def casimir_diagonal(group: SU2) -> jax.Array:
    return jnp.asarray(
        tuple(group.casimir(j_twice) for j_twice in group.irreps()),
        dtype=jnp.float64,
    )


def build_link_casimir_terms(
    shape: tuple[int, int],
    group: SU2,
) -> tuple[HorizontalLinkCasimirTerm | VerticalLinkCasimirTerm, ...]:
    """Build one Casimir term for every open-boundary lattice link."""
    n_rows, n_cols = shape
    diag = casimir_diagonal(group)
    return (
        *(
            HorizontalLinkCasimirTerm(row=r, col=c, diag=diag)
            for r in range(n_rows)
            for c in range(n_cols - 1)
        ),
        *(
            VerticalLinkCasimirTerm(row=r, col=c, diag=diag)
            for r in range(n_rows - 1)
            for c in range(n_cols)
        ),
    )
