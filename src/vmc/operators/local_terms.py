"""Local operator terms for PEPS energy evaluation."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TypeAlias

import jax
import jax.numpy as jnp

IndexedDiagonalTerm: TypeAlias = tuple[int, "DiagonalTerm"]
IndexedOneSiteTerm: TypeAlias = tuple[int, "OneSiteTerm"]
IndexedHorizontalTwoSiteTerm: TypeAlias = tuple[int, "HorizontalTwoSiteTerm"]
IndexedVerticalTwoSiteTerm: TypeAlias = tuple[int, "VerticalTwoSiteTerm"]
IndexedPlaquetteTerm: TypeAlias = tuple[int, "PlaquetteTerm"]

__all__ = [
    "LocalTerm",
    "OneSiteTerm",
    "DiagonalTerm",
    "HorizontalTwoSiteTerm",
    "VerticalTwoSiteTerm",
    "PlaquetteTerm",
    "BucketedTerms",
    "LocalHamiltonian",
    "bucket_terms",
]


class LocalTerm(abc.ABC):
    """Abstract base class for local operator terms."""


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OneSiteTerm(LocalTerm):
    """Single-site operator term acting at (row, col)."""

    row: int
    col: int
    op: jax.Array

    @property
    def sites(self) -> tuple[tuple[int, int], ...]:
        return ((self.row, self.col),)

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", jnp.asarray(self.op))

    def tree_flatten(self):
        return (self.op,), (self.row, self.col)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (op,) = children
        row, col = aux_data
        return cls(row=row, col=col, op=op)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DiagonalTerm(LocalTerm):
    """Diagonal operator term on one or two sites."""

    sites: tuple[tuple[int, int], ...]
    diag: jax.Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "diag", jnp.asarray(self.diag))

    def tree_flatten(self):
        return (self.diag,), (self.sites,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (diag,) = children
        (sites,) = aux_data
        return cls(sites=sites, diag=diag)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HorizontalTwoSiteTerm(LocalTerm):
    """Two-site operator on horizontal neighbor (row, col) -> (row, col+1)."""

    row: int
    col: int
    op: jax.Array

    @property
    def sites(self) -> tuple[tuple[int, int], ...]:
        return ((self.row, self.col), (self.row, self.col + 1))

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", jnp.asarray(self.op))

    def tree_flatten(self):
        return (self.op,), (self.row, self.col)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (op,) = children
        row, col = aux_data
        return cls(row=row, col=col, op=op)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VerticalTwoSiteTerm(LocalTerm):
    """Two-site operator on vertical neighbor (row, col) -> (row+1, col)."""

    row: int
    col: int
    op: jax.Array

    @property
    def sites(self) -> tuple[tuple[int, int], ...]:
        return ((self.row, self.col), (self.row + 1, self.col))

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", jnp.asarray(self.op))

    def tree_flatten(self):
        return (self.op,), (self.row, self.col)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (op,) = children
        row, col = aux_data
        return cls(row=row, col=col, op=op)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PlaquetteTerm(LocalTerm):
    """Plaquette term on the square with top-left corner at (row, col)."""

    row: int
    col: int
    coeff: jax.Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "coeff", jnp.asarray(self.coeff))

    def tree_flatten(self):
        return (self.coeff,), (self.row, self.col)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (coeff,) = children
        row, col = aux_data
        return cls(row=row, col=col, coeff=coeff)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LocalHamiltonian:
    """Container for local PEPS operator terms."""

    shape: tuple[int, int]
    terms: tuple[LocalTerm, ...] = ()

    def tree_flatten(self):
        return (self.terms,), (self.shape,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (terms,) = children
        (shape,) = aux_data
        return cls(shape=shape, terms=terms)


@dataclass(frozen=True)
class BucketedTerms:
    """Indexed, shape-grouped local terms.

    Each entry stores ``(term_idx, term)`` where ``term_idx`` is the index in
    the original ``LocalHamiltonian.terms`` tuple. This provides a stable global
    indexing that remains valid after bucketing.
    """

    diagonal: tuple[IndexedDiagonalTerm, ...]
    one_site: tuple[tuple[tuple[IndexedOneSiteTerm, ...], ...], ...]
    horizontal: tuple[tuple[tuple[IndexedHorizontalTwoSiteTerm, ...], ...], ...]
    vertical: tuple[tuple[tuple[IndexedVerticalTwoSiteTerm, ...], ...], ...]
    plaquette: tuple[tuple[tuple[IndexedPlaquetteTerm, ...], ...], ...]
    n_terms: int


def bucket_terms(
    terms: tuple[LocalTerm, ...],
    shape: tuple[int, int],
) -> BucketedTerms:
    """Group terms by type and lattice location."""
    n_rows, n_cols = shape
    one_site_terms = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
    horizontal_terms = [[[] for _ in range(max(n_cols - 1, 0))] for _ in range(n_rows)]
    vertical_terms = [[[] for _ in range(n_cols)] for _ in range(max(n_rows - 1, 0))]
    plaquette_terms = [
        [[] for _ in range(max(n_cols - 1, 0))]
        for _ in range(max(n_rows - 1, 0))
    ]
    diagonal_terms: list[IndexedDiagonalTerm] = []

    for term_idx, term in enumerate(terms):
        if isinstance(term, OneSiteTerm):
            one_site_terms[term.row][term.col].append((term_idx, term))
        elif isinstance(term, HorizontalTwoSiteTerm):
            horizontal_terms[term.row][term.col].append((term_idx, term))
        elif isinstance(term, VerticalTwoSiteTerm):
            vertical_terms[term.row][term.col].append((term_idx, term))
        elif isinstance(term, DiagonalTerm):
            diagonal_terms.append((term_idx, term))
        elif isinstance(term, PlaquetteTerm):
            plaquette_terms[term.row][term.col].append((term_idx, term))
        else:
            raise TypeError(f"Unsupported term type: {type(term)!r}")

    return BucketedTerms(
        diagonal=tuple(diagonal_terms),
        one_site=tuple(tuple(tuple(cell) for cell in row) for row in one_site_terms),
        horizontal=tuple(
            tuple(tuple(cell) for cell in row) for row in horizontal_terms
        ),
        vertical=tuple(tuple(tuple(cell) for cell in row) for row in vertical_terms),
        plaquette=tuple(tuple(tuple(cell) for cell in row) for row in plaquette_terms),
        n_terms=len(terms),
    )
