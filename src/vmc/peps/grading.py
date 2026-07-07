"""Fermionic grading of sampled PEPS families.

Statistics is metadata, never a family (spec
``docs/superpowers/specs/2026-07-07-graded-sampled-sector-peps.md``): a
parity per physical label plus a parity vector per virtual leg.  In the
sampled basis every fermionic effect collapses to classical bookkeeping on
the sample (Wu & Dai, arXiv:2506.20106, transposed to row sweeps):

- even-parity masks zero tensor entries whose leg parities sum odd,
- swap gates become diagonal signs on virtual legs whose exponents are
  column prefix parities of the sample: physical wires route down their
  column, crossing the horizontal bonds to their right, so the gate on the
  right leg of site ``(r, c)`` carries ``prefix[r, c] = sum_{r'<r}
  parity(n[r', c])`` and the induced Jordan-Wigner mode order is
  column-major,
- moving one fermion across a bond flips exactly one downstream gate
  stack, which re-gauges (through the even-parity constraint) to a single
  in-window leg sign plus a scalar; the composed scalar signs live next to
  the leg flips they are paired with, in the fermionic term evaluations.

Everything here is static metadata or O(N) integer bookkeeping; no tensor
algebra.  The conventions are pinned by ``tests/fermionic_exact.py``, not
trusted on derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "FermionSigns",
    "Grading",
    "column_prefix_parities",
    "even_mask",
    "leg_parities",
]


@dataclass(frozen=True)
class Grading:
    """Fermionic grading and sampled sector of one PEPS family.

    ``phys_parity[p]`` is the fermion parity of physical label ``p``,
    ``filling[p]`` the number of sites carrying label ``p`` in the sampled
    fixed-sector ensemble (adjacent-exchange dynamics conserves all label
    counts), and ``n_even`` the number of leading even-parity states on
    bulk virtual legs (contiguous layout; dimension-1 boundary legs are
    always even).
    """

    phys_parity: tuple[int, ...]
    filling: tuple[int, ...]
    n_even: int


class FermionSigns(NamedTuple):
    """Per-sample sign data threaded to graded term evaluations."""

    prefix: jax.Array  # (n_rows, n_cols) column prefix parities
    suffix: jax.Array  # (n_rows, n_cols) column suffix parities
    down_flip: list  # [r][c] static down-leg sign vectors (-1)^{P_d}
    right_flip: list  # [r][c] static right-leg sign vectors (-1)^{P_r}


def leg_parities(dim: int, n_even: int) -> np.ndarray:
    """Parity vector of a virtual leg: leading ``n_even`` states even."""
    return (np.arange(dim) >= n_even).astype(np.int64)


def even_mask(
    parities: np.ndarray,
    dims: tuple[int, int, int, int],
    n_even: int,
) -> np.ndarray:
    """0/1 mask over ``(leading, up, down, left, right)`` keeping even entries.

    ``parities`` grades the leading axis (physical labels for standard PEPS,
    vertex blocks for LGT); the four virtual legs carry the contiguous
    ``n_even`` layout.
    """
    total = np.asarray(parities).reshape(-1, 1, 1, 1, 1)
    for axis, dim in enumerate(dims):
        shape = [1] * 5
        shape[axis + 1] = dim
        total = total + leg_parities(dim, n_even).reshape(shape)
    return (total % 2 == 0).astype(np.float64)


def _grading_statics(grading: Grading, tensors: list[list]) -> tuple[list, list, list]:
    """Per-site even masks and right/down leg parity vectors (jnp constants)."""
    site_dims = [[jnp.asarray(t).shape[1:] for t in row] for row in tensors]
    masks = [
        [
            jnp.asarray(even_mask(grading.phys_parity, dims, grading.n_even))
            for dims in row
        ]
        for row in site_dims
    ]
    right_par, down_par = (
        [
            [
                jnp.asarray(leg_parities(dims[axis], grading.n_even), dtype=jnp.float64)
                for dims in row
            ]
            for row in site_dims
        ]
        for axis in (3, 1)
    )
    return masks, right_par, down_par


def column_prefix_parities(parities: jax.Array) -> jax.Array:
    """``prefix[r, c] = sum_{r'<r} parities[r', c] mod 2``."""
    return jnp.cumsum(jnp.pad(parities[:-1], ((1, 0), (0, 0))), axis=0) % 2
