"""Local operators for PEPS energy evaluation."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, TypeAlias

import jax
import jax.numpy as jnp
from plum import dispatch

IndexedOperator: TypeAlias = tuple[int, "Operator"]

__all__ = [
    "Operator",
    "TransitionOperator",
    "OneSiteOperator",
    "DiagonalOperator",
    "HorizontalTwoSiteOperator",
    "VerticalTwoSiteOperator",
    "PlaquetteOperator",
    "BucketedOperators",
    "LocalHamiltonian",
    "support_span",
    "eval_span",
    "bucket_operators",
]


class Operator(abc.ABC):
    """Abstract base class for local operator terms."""


class TransitionOperator(Operator):
    """Operator anchored at lattice coordinate (row, col)."""

    row: int
    col: int


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OneSiteOperator(TransitionOperator):
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
class DiagonalOperator(Operator):
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
class HorizontalTwoSiteOperator(TransitionOperator):
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
class VerticalTwoSiteOperator(TransitionOperator):
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
class PlaquetteOperator(TransitionOperator):
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
    terms: tuple[Operator, ...] = ()

    def tree_flatten(self):
        return (self.terms,), (self.shape,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (terms,) = children
        (shape,) = aux_data
        return cls(shape=shape, terms=terms)


@dataclass(frozen=True)
class BucketedOperators:
    """Indexed, shape-grouped local terms.

    Each entry stores ``(term_idx, term)`` where ``term_idx`` is the index in
    the original ``LocalHamiltonian.terms`` tuple. This provides a stable global
    indexing that remains valid after bucketing.
    """

    diagonal: tuple[IndexedOperator, ...]
    span_11: tuple[tuple[tuple[IndexedOperator, ...], ...], ...]
    span_12: tuple[tuple[tuple[IndexedOperator, ...], ...], ...]
    span_21: tuple[tuple[tuple[IndexedOperator, ...], ...], ...]
    span_22: tuple[tuple[tuple[IndexedOperator, ...], ...], ...]
    n_terms: int


@dispatch
def support_span(term: TransitionOperator) -> tuple[int, int]:
    raise TypeError(f"Unsupported term type: {type(term)!r}")


@support_span.dispatch
def support_span(term: OneSiteOperator) -> tuple[int, int]:
    del term
    return 1, 1


@support_span.dispatch
def support_span(term: HorizontalTwoSiteOperator) -> tuple[int, int]:
    del term
    return 1, 2


@support_span.dispatch
def support_span(term: VerticalTwoSiteOperator) -> tuple[int, int]:
    del term
    return 2, 1


@support_span.dispatch
def support_span(term: PlaquetteOperator) -> tuple[int, int]:
    del term
    return 2, 2


@dispatch
def eval_span(model: object, term: TransitionOperator) -> tuple[int, int]:
    del model
    return support_span(term)


def bucket_operators(
    terms: tuple[Operator, ...],
    shape: tuple[int, int],
    *,
    eval_span: Callable[[TransitionOperator], tuple[int, int]] | None = None,
) -> BucketedOperators:
    """Group terms by type and lattice location."""
    n_rows, n_cols = shape
    span_of = support_span if eval_span is None else eval_span
    span_11 = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
    span_12 = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
    span_21 = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
    span_22 = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
    span_grids = {
        (1, 1): span_11,
        (1, 2): span_12,
        (2, 1): span_21,
        (2, 2): span_22,
    }
    diagonal_operators: list[IndexedOperator] = []

    for term_idx, term in enumerate(terms):
        if isinstance(term, DiagonalOperator):
            diagonal_operators.append((term_idx, term))
            continue
        if not isinstance(term, TransitionOperator):
            raise TypeError(f"Unsupported term type: {type(term)!r}")
        dr, dc = support_span(term)
        if not (
            0 <= term.row < n_rows
            and 0 <= term.col < n_cols
            and term.row + dr <= n_rows
            and term.col + dc <= n_cols
        ):
            raise ValueError(f"Operator {term!r} is outside shape {shape}.")
        dr_eval, dc_eval = span_of(term)
        grid = span_grids.get((dr_eval, dc_eval))
        if grid is None:
            raise ValueError(
                f"Unsupported eval span {(dr_eval, dc_eval)} for {term!r}."
            )
        grid[term.row][term.col].append((term_idx, term))

    frozen_span_grids = {
        span: tuple(tuple(tuple(cell) for cell in row) for row in grid)
        for span, grid in span_grids.items()
    }

    return BucketedOperators(
        diagonal=tuple(diagonal_operators),
        span_11=frozen_span_grids[(1, 1)],
        span_12=frozen_span_grids[(1, 2)],
        span_21=frozen_span_grids[(2, 1)],
        span_22=frozen_span_grids[(2, 2)],
        n_terms=len(terms),
    )
