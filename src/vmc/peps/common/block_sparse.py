"""Generic scheduled block-sparse execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from vmc.operators.local_terms import BucketedOperators, TransitionOperator

__all__ = [
    "BlockEvalColumn",
    "BlockEvalPass",
    "BlockEvalSchedule",
    "build_eval_schedule",
]


@dataclass(frozen=True)
class BlockEvalColumn:
    """Operators at one anchor column sharing the same evaluation width."""

    dc: int
    terms: tuple[tuple[TransitionOperator, tuple[tuple[int, int], ...]], ...]


@dataclass(frozen=True)
class BlockEvalPass:
    """One row-span pass in the local-estimate sweep."""

    dr: int
    columns: tuple[tuple[BlockEvalColumn, ...], ...]


@dataclass(frozen=True)
class BlockEvalSchedule:
    """Bucketed local terms grouped by row, row span, column, and column span."""

    rows: tuple[tuple[BlockEvalPass, ...], ...]


def build_eval_schedule(
    terms: BucketedOperators,
    eval_span: Callable[[TransitionOperator], tuple[int, int]],
) -> BlockEvalSchedule:
    """Group bucketed terms by effective column span for env reuse."""
    return BlockEvalSchedule(
        rows=tuple(
            tuple(
                BlockEvalPass(
                    dr=dr,
                    columns=tuple(
                        _group_column_by_dc(col_terms, eval_span)
                        for col_terms in columns
                    ),
                )
                for dr, columns in row_passes
            )
            for row_passes in terms.rows
        )
    )


def _group_column_by_dc(
    col_terms: tuple[tuple[TransitionOperator, tuple[tuple[int, int], ...]], ...],
    eval_span: Callable[[TransitionOperator], tuple[int, int]],
) -> tuple[BlockEvalColumn, ...]:
    grouped: dict[int, list] = {}
    for term, contributions in col_terms:
        _dr, dc = eval_span(term)
        grouped.setdefault(dc, []).append((term, contributions))
    return tuple(
        BlockEvalColumn(dc=dc, terms=tuple(grouped[dc])) for dc in sorted(grouped)
    )
