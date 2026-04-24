"""Generic scheduled block-sparse execution helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from vmc.operators.local_terms import BucketedOperators, TransitionOperator

__all__ = [
    "BlockEvalColumn",
    "BlockEvalPass",
    "BlockEvalSchedule",
    "build_eval_schedule",
    "gather_block",
    "scatter_block_grad",
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


def gather_block(blocks: jax.Array, block_id: jax.Array | int) -> jax.Array:
    """Gather one active dense block from a stacked block array."""
    return blocks[block_id]


def scatter_block_grad(
    block_grad: jax.Array,
    *,
    block_id: jax.Array | int,
    n_blocks: int,
) -> jax.Array:
    """Scatter one active dense-block gradient into a stacked block array."""
    return jnp.zeros(
        (n_blocks, *block_grad.shape),
        dtype=block_grad.dtype,
    ).at[block_id].set(block_grad)


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
    col_terms: tuple[tuple[Any, tuple[tuple[int, int], ...]], ...],
    eval_span: Callable[[TransitionOperator], tuple[int, int]],
) -> tuple[BlockEvalColumn, ...]:
    grouped: dict[int, list] = {}
    for term, contributions in col_terms:
        _dr, dc = eval_span(term)
        grouped.setdefault(dc, []).append((term, contributions))
    return tuple(
        BlockEvalColumn(dc=dc, terms=tuple(grouped[dc]))
        for dc in sorted(grouped)
    )
