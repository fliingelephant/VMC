"""Generic local terms for sampled pure-gauge non-Abelian GI-PEPS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from plum import dispatch

from vmc.operators.local_terms import (
    DiagonalOperator,
    PlaquetteOperator as PlaquetteTerm,
    TransitionOperator,
    support_span,
)

__all__ = [
    "HorizontalLinkCasimirTerm",
    "HorizontalMatterHoppingTerm",
    "MatterNumberTerm",
    "PlaquetteTerm",
    "VerticalLinkCasimirTerm",
    "VerticalMatterHoppingTerm",
    "build_link_casimir_terms",
    "build_matter_number_terms",
    "casimir_diagonal",
    "link_casimir_energy",
    "matter_number_energy",
]


@dataclass(frozen=True, init=False)
class _SiteDiagonalTerm(DiagonalOperator):
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
class HorizontalLinkCasimirTerm(_SiteDiagonalTerm):
    """Diagonal electric-energy term on one horizontal link."""


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class VerticalLinkCasimirTerm(_SiteDiagonalTerm):
    """Diagonal electric-energy term on one vertical link."""


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class MatterNumberTerm(_SiteDiagonalTerm):
    """Diagonal matter-number term on one site."""


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HorizontalMatterHoppingTerm(TransitionOperator):
    """Number-conserving gauge-covariant matter hopping on a horizontal link."""

    row: int
    col: int

    @property
    def sites(self) -> tuple[tuple[int, int], ...]:
        return ((self.row, self.col), (self.row, self.col + 1))

    def tree_flatten(self):
        return (), (self.row, self.col)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        row, col = aux_data
        return cls(row=row, col=col)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VerticalMatterHoppingTerm(TransitionOperator):
    """Number-conserving gauge-covariant matter hopping on a vertical link."""

    row: int
    col: int

    @property
    def sites(self) -> tuple[tuple[int, int], ...]:
        return ((self.row, self.col), (self.row + 1, self.col))

    def tree_flatten(self):
        return (), (self.row, self.col)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        row, col = aux_data
        return cls(row=row, col=col)


@support_span.dispatch
def support_span(_: HorizontalMatterHoppingTerm) -> tuple[int, int]:
    return 1, 2


@support_span.dispatch
def support_span(_: VerticalMatterHoppingTerm) -> tuple[int, int]:
    return 2, 1


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


def matter_number_energy(
    term: MatterNumberTerm,
    matter: jax.Array,
) -> jax.Array:
    return term.diag[matter[term.row, term.col]]


def casimir_diagonal(group: Any) -> jax.Array:
    return jnp.asarray(
        tuple(group.casimir(irrep) for irrep in group.irreps()),
        dtype=jnp.float64,
    )


def build_link_casimir_terms(
    shape: tuple[int, int],
    group: Any,
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


def build_matter_number_terms(
    shape: tuple[int, int],
    matter_numbers: tuple[int, ...],
) -> tuple[MatterNumberTerm, ...]:
    """Build one diagonal matter-number term per site."""
    n_rows, n_cols = shape
    diag = jnp.asarray(matter_numbers, dtype=jnp.float64)
    return tuple(
        MatterNumberTerm(row=r, col=c, diag=diag)
        for r in range(n_rows)
        for c in range(n_cols)
    )
