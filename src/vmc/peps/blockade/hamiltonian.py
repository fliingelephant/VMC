"""Rydberg Hamiltonian builder for Blockade PEPS.

H = (Omega/2) sum_i sigma^x_i - Delta sum_i n_i + V_nnn sum_<<ij>> n_i n_j

Note: Nearest-neighbor n_i*n_j terms are omitted because they're always 0
due to the blockade constraint (n_i * n_j = 0 for adjacent sites).
"""
from __future__ import annotations

import jax.numpy as jnp

from vmc.operators.local_terms import (
    DiagonalTerm,
    LocalHamiltonian,
    OneSiteTerm,
)

__all__ = ["rydberg_hamiltonian"]


def rydberg_hamiltonian(
    shape: tuple[int, int],
    Omega: float,
    Delta: float,
    V_nnn: float = 0.0,
) -> LocalHamiltonian:
    """Build Rydberg Hamiltonian for blockade PEPS.

    H = (Omega/2) sum_i sigma^x_i - Delta sum_i n_i + V_nnn sum_<<ij>> n_i n_j

    Args:
        shape: (n_rows, n_cols) lattice shape
        Omega: Rabi frequency (X term coefficient)
        Delta: Detuning (n term coefficient, enters as -Delta * n_i)
        V_nnn: Next-nearest-neighbor interaction (optional)

    Returns:
        LocalHamiltonian with appropriate terms

    Note:
        NN n_i*n_j terms are omitted - always 0 by blockade constraint.
    """
    terms = []
    n_rows, n_cols = shape

    # sigma^x = [[0, 1], [1, 0]] in |0>, |1> basis
    sigma_x = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)

    # n diagonal: diag[n=0] = 0, diag[n=1] = -Delta
    n_diag = jnp.array([0.0, -Delta], dtype=jnp.complex128)

    # n_i n_j diagonal for 2-site term:
    # diag indexed by n1 * 2 + n2
    # (0,0)->0, (0,1)->0, (1,0)->0, (1,1)->V
    nn_diag = jnp.array([0.0, 0.0, 0.0, V_nnn], dtype=jnp.complex128)

    for r in range(n_rows):
        for c in range(n_cols):
            # X term (off-diagonal)
            terms.append(OneSiteTerm(row=r, col=c, op=0.5 * Omega * sigma_x))
            # n_i term (diagonal)
            terms.append(DiagonalTerm(sites=((r, c),), diag=n_diag))

    # NNN interactions (diagonal, NOT always 0)
    if V_nnn != 0.0:
        for r in range(n_rows):
            for c in range(n_cols):
                # Diagonal NNN: (r, c) and (r+1, c+1)
                if r + 1 < n_rows and c + 1 < n_cols:
                    terms.append(
                        DiagonalTerm(sites=((r, c), (r + 1, c + 1)), diag=nn_diag)
                    )
                # Anti-diagonal NNN: (r, c) and (r+1, c-1)
                if r + 1 < n_rows and c - 1 >= 0:
                    terms.append(
                        DiagonalTerm(sites=((r, c), (r + 1, c - 1)), diag=nn_diag)
                    )

    return LocalHamiltonian(shape=shape, terms=tuple(terms))
