"""Jump operators for T1 and T2 noise in open quantum systems.

T1 (amplitude damping) and T2 (dephasing) noise are characterized by:
    - T1 decay: L = sqrt(gamma) * sigma_minus, gamma = 1/T1
    - Pure dephasing: L = sqrt(gamma_phi) * sigma_z, gamma_phi = 1/T_phi
    - Total T2: 1/T2 = 1/(2*T1) + 1/T_phi

For quantum trajectories, we need:
    - The jump operator L (as OneSiteTerm)
    - The diagonal operator L^dag L (as DiagonalTerm) for computing jump probabilities
    - The rate gamma
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from vmc.operators.local_terms import DiagonalTerm, OneSiteTerm

__all__ = [
    "JumpOperator",
    "SIGMA_MINUS",
    "SIGMA_Z",
]

# Pauli matrices for spin-1/2
SIGMA_MINUS = jnp.array([[0, 1], [0, 0]], dtype=jnp.complex128)  # |0><1|
SIGMA_Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)


@dataclass(frozen=True)
class JumpOperator:
    """Jump operator with rate and L^dag L for probability computation.

    Attributes:
        rate: Jump rate gamma (1/T1 for amplitude damping, 1/T_phi for dephasing)
        L: The jump operator as OneSiteTerm
        LdagL: L^dag L as DiagonalTerm for computing <L^dag L>
    """

    rate: float
    L: OneSiteTerm
    LdagL: DiagonalTerm

    @property
    def row(self) -> int:
        return self.L.row

    @property
    def col(self) -> int:
        return self.L.col

    @property
    def site(self) -> tuple[int, int]:
        return (self.L.row, self.L.col)

    @classmethod
    def t1(cls, row: int, col: int, T1: float) -> JumpOperator:
        """T1 (amplitude damping) jump operator at (row, col).

        T1 decay: L = sigma_minus = |0><1|, rate = 1/T1
        L^dag L = |1><1| has diagonal [0, 1] in occupancy basis.

        Args:
            row: Row index of the site.
            col: Column index of the site.
            T1: T1 relaxation time (energy relaxation).
        """
        return cls(
            rate=1.0 / T1,
            L=OneSiteTerm(row=row, col=col, op=SIGMA_MINUS),
            LdagL=DiagonalTerm(sites=((row, col),), diag=jnp.array([0.0, 1.0])),
        )

    @classmethod
    def dephasing(cls, row: int, col: int, T_phi: float) -> JumpOperator:
        """Pure dephasing jump operator at (row, col).

        Pure dephasing: L = sigma_z, rate = 1/T_phi
        L^dag L = sigma_z^2 = I has diagonal [1, 1] in occupancy basis.

        Note: Total T2 dephasing time satisfies 1/T2 = 1/(2*T1) + 1/T_phi.

        Args:
            row: Row index of the site.
            col: Column index of the site.
            T_phi: Pure dephasing time.
        """
        return cls(
            rate=1.0 / T_phi,
            L=OneSiteTerm(row=row, col=col, op=SIGMA_Z),
            LdagL=DiagonalTerm(sites=((row, col),), diag=jnp.array([1.0, 1.0])),
        )
