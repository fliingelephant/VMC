"""Local operators for PEPS energy evaluation."""
from __future__ import annotations

import abc
from dataclasses import InitVar, dataclass
from typing import Any, Callable, TypeAlias

import jax
import jax.numpy as jnp
from plum import dispatch

from vmc.operators.time_dependent import (
    TimeDependentHamiltonian,
    coeffs_at,
)

Contribution: TypeAlias = tuple[int, int]  # (op_idx, coeff_idx)
Contributions: TypeAlias = tuple["Contribution", ...]
TaggedDiagonal: TypeAlias = tuple["Operator", "Contributions"]
TaggedTransition: TypeAlias = tuple["TransitionOperator", "Contributions"]

__all__ = [
    "Operator",
    "TransitionOperator",
    "OneSiteOperator",
    "DiagonalOperator",
    "HorizontalTwoSiteOperator",
    "VerticalTwoSiteOperator",
    "PlaquetteOperator",
    "BucketedOperators",
    "CoefficientStructure",
    "LocalHamiltonian",
    "support_span",
    "merge_operators",
]


def _normalize_coeffs(
    coeffs: Any,
    n_terms: int,
) -> tuple[jax.Array, ...]:
    if coeffs is None:
        return (jnp.asarray(1.0),) * n_terms
    if isinstance(coeffs, tuple):
        if len(coeffs) != n_terms:
            raise ValueError(f"Expected {n_terms} coefficients, got {len(coeffs)}.")
        return tuple(jnp.asarray(coeff) for coeff in coeffs)
    coeff_array = jnp.asarray(coeffs)
    if coeff_array.ndim == 0:
        if n_terms != 1:
            raise ValueError(f"Expected {n_terms} coefficients, got scalar.")
        return (coeff_array,)
    if coeff_array.shape != (n_terms,):
        raise ValueError(f"Expected {n_terms} coefficients, got shape {coeff_array.shape}.")
    return tuple(coeff_array[idx] for idx in range(n_terms))


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

    def tree_flatten(self):
        return (), (self.row, self.col)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        row, col = aux_data
        return cls(row=row, col=col)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LocalHamiltonian:
    """Container for local PEPS operator terms."""

    shape: tuple[int, int]
    terms: tuple[Operator, ...] = ()
    coeffs: tuple[jax.Array, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coeffs",
            _normalize_coeffs(self.coeffs, len(self.terms)),
        )

    def tree_flatten(self):
        return (self.terms, self.coeffs), (self.shape,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        terms, coeffs = children
        (shape,) = aux_data
        return cls(shape=shape, terms=terms, coeffs=coeffs)


@dataclass(frozen=True)
class BucketedOperators:
    """Local terms grouped by row and effective row span.

    Each transition term is ``(term, contributions)`` where
    ``contributions = tuple[(op_idx, coeff_idx), ...]`` maps the term
    to one or more output operator slots with associated coefficient indices.

    Each diagonal term is ``(term, contributions)``.
    """

    diagonal: tuple[TaggedDiagonal, ...]
    rows: tuple[
        tuple[
            tuple[int, tuple[tuple[TaggedTransition, ...], ...]],
            ...,
        ],
        ...,
    ]
    n_ops: InitVar[int] = 1

    def __post_init__(self, n_ops: int) -> None:
        object.__setattr__(self, 'n_ops', n_ops)

    def __len__(self) -> int:
        """Number of source operators before bucketing."""
        return self.n_ops


@dispatch
def support_span(term: TransitionOperator) -> tuple[int, int]:
    raise TypeError(f"Unsupported term type: {type(term)!r}")


@support_span.dispatch
def support_span(_: OneSiteOperator) -> tuple[int, int]:
    return 1, 1


@support_span.dispatch
def support_span(_: HorizontalTwoSiteOperator) -> tuple[int, int]:
    return 1, 2


@support_span.dispatch
def support_span(_: VerticalTwoSiteOperator) -> tuple[int, int]:
    return 2, 1


@support_span.dispatch
def support_span(_: PlaquetteOperator) -> tuple[int, int]:
    return 2, 2


@dataclass(frozen=True)
class CoefficientStructure:
    """Maps flat coefficient indices back to source operators.

    ``schedules`` holds the (optional) time-dependent schedule for each
    source operator.
    """

    base_coeffs: tuple[jax.Array, ...]
    schedules: tuple[Any, ...]  # TermCoefficientSchedule | None per op

    def build_coeffs(self, t: float | jax.Array | None = None) -> jax.Array:
        """Build flat coefficient array at time *t*."""
        parts: list[jax.Array] = []
        for op_idx, base in enumerate(self.base_coeffs):
            sched = self.schedules[op_idx]
            if sched is None:
                parts.append(base)
                continue
            if t is None:
                raise ValueError(
                    "Time-dependent operators require a non-None time `t`."
                )
            sched_coeffs = coeffs_at(sched, t)
            if sched_coeffs.shape != base.shape:
                raise ValueError(
                    f"Expected schedule coefficients of shape {base.shape}, got {sched_coeffs.shape}."
                )
            parts.append(base * sched_coeffs)
        return jnp.concatenate(parts) if len(parts) > 1 else parts[0]


def merge_operators(
    operators: tuple[LocalHamiltonian | TimeDependentHamiltonian, ...],
    shape: tuple[int, int],
    eval_span: Callable[[TransitionOperator], tuple[int, int]] | None = None,
) -> tuple[BucketedOperators, CoefficientStructure]:
    """Bucket and merge operators into a single :class:`BucketedOperators`.

    Accepts one or more ``LocalHamiltonian`` / ``TimeDependentHamiltonian``
    operators and groups their terms by row and effective span. Identical
    transition terms (same type, anchor, and array object) across operators
    are deduplicated, sharing a single ``_eval_term`` call at runtime.
    Returns both the bucketed terms and a :class:`CoefficientStructure`
    for building the flat coefficient array.
    """
    n_rows, n_cols = shape
    span_of = support_span if eval_span is None else eval_span

    # Flatten all terms with source tracking
    flat_terms: list[tuple[Operator, int, int]] = []  # (term, op_idx, term_within_op_idx)
    base_coeffs: list[jax.Array] = []
    schedules: list = []
    for op_idx, op in enumerate(operators):
        if isinstance(op, TimeDependentHamiltonian):
            base = op.base
            schedule = op.schedule
        else:
            base = op
            schedule = None
        if not isinstance(base, LocalHamiltonian):
            raise TypeError(f"Unsupported operator type: {type(base)!r}")
        terms = base.terms
        base_coeffs.append(jnp.asarray(base.coeffs))
        schedules.append(schedule)
        for local_idx, term in enumerate(terms):
            flat_terms.append((term, op_idx, local_idx))

    # Build coefficient offset per operator
    coeff_offset: list[int] = []
    offset = 0
    for coeffs in base_coeffs:
        coeff_offset.append(offset)
        offset += len(coeffs)

    coeff_struct = CoefficientStructure(
        base_coeffs=tuple(base_coeffs),
        schedules=tuple(schedules),
    )

    # Bucket terms, deduplicating identical operators across sources.
    # Key: (type, pytree aux_data, array values) → (entry_list, index)
    def _dedup_key(term: Operator) -> tuple:
        arrays, aux = term.tree_flatten()
        return (type(term), aux, tuple(a.tobytes() for a in arrays))

    rows: list[dict[int, list[list]]] = [{} for _ in range(n_rows)]
    diagonal_operators: list[TaggedDiagonal] = []
    dedup: dict[tuple, tuple[list, int]] = {}

    for term, op_idx, local_idx in flat_terms:
        global_coeff_idx = coeff_offset[op_idx] + local_idx
        contribution = (op_idx, global_coeff_idx)
        key = _dedup_key(term)

        if key in dedup:
            entries, idx = dedup[key]
            existing_term, existing_contribs = entries[idx]
            entries[idx] = (existing_term, existing_contribs + (contribution,))
            continue

        if isinstance(term, DiagonalOperator):
            dedup[key] = (diagonal_operators, len(diagonal_operators))
            diagonal_operators.append((term, (contribution,)))
            continue
        if not isinstance(term, TransitionOperator):
            raise TypeError(f"Unsupported term type: {type(term)!r}")
        support_dr, support_dc = support_span(term)
        if not (
            0 <= term.row < n_rows
            and 0 <= term.col < n_cols
            and term.row + support_dr <= n_rows
            and term.col + support_dc <= n_cols
        ):
            raise ValueError(f"Operator {term!r} is outside shape {shape}.")
        dr_eval, dc_eval = span_of(term)
        if dr_eval <= 0 or dc_eval <= 0:
            raise ValueError(
                f"Unsupported eval span {(dr_eval, dc_eval)} for {term!r}."
            )
        dr_eff = min(dr_eval, n_rows - term.row)
        row_passes = rows[term.row]
        if dr_eff not in row_passes:
            row_passes[dr_eff] = [[] for _ in range(n_cols)]
        cell = row_passes[dr_eff][term.col]
        dedup[key] = (cell, len(cell))
        cell.append((term, (contribution,)))

    return BucketedOperators(
        diagonal=tuple(diagonal_operators),
        rows=tuple(
            tuple(
                (dr, tuple(tuple(cell) for cell in cols))
                for dr, cols in sorted(row_passes.items())
            )
            for row_passes in rows
        ),
        n_ops=len(operators),
    ), coeff_struct
