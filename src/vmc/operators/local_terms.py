"""Local operator terms for PEPS energy evaluation."""
from __future__ import annotations

import abc
from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = [
    "LocalTerm",
    "OneSiteTerm",
    "DiagonalTerm",
    "HorizontalTwoSiteTerm",
    "VerticalTwoSiteTerm",
    "LocalHamiltonian",
    "bucket_terms",
]


class LocalTerm(abc.ABC):
    """Abstract base class for local operator terms."""


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


@dataclass(frozen=True)
class DiagonalTerm(LocalTerm):
    """Diagonal operator term on one or two sites."""

    sites: tuple[tuple[int, int], ...]
    diag: jax.Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "diag", jnp.asarray(self.diag))


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


@dataclass(frozen=True)
class LocalHamiltonian:
    """Container for local PEPS operator terms."""

    shape: tuple[int, int]
    terms: tuple[LocalTerm, ...] = ()


def bucket_terms(
    terms: tuple[LocalTerm, ...],
    shape: tuple[int, int],
) -> tuple[
    list[DiagonalTerm],
    list[list[list[OneSiteTerm]]],
    list[list[list[HorizontalTwoSiteTerm]]],
    list[list[list[VerticalTwoSiteTerm]]],
]:
    """Group terms by type and lattice location."""
    n_rows, n_cols = shape
    one_site_terms = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
    horizontal_terms = [[[] for _ in range(max(n_cols - 1, 0))] for _ in range(n_rows)]
    vertical_terms = [[[] for _ in range(n_cols)] for _ in range(max(n_rows - 1, 0))]
    diagonal_terms: list[DiagonalTerm] = []

    for term in terms:
        if isinstance(term, OneSiteTerm):
            one_site_terms[term.row][term.col].append(term)
        elif isinstance(term, HorizontalTwoSiteTerm):
            horizontal_terms[term.row][term.col].append(term)
        elif isinstance(term, VerticalTwoSiteTerm):
            vertical_terms[term.row][term.col].append(term)
        elif isinstance(term, DiagonalTerm):
            diagonal_terms.append(term)
        else:
            raise TypeError(f"Unsupported term type: {type(term)!r}")

    return diagonal_terms, one_site_terms, horizontal_terms, vertical_terms
