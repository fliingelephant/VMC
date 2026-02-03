"""Blockade PEPS for Rydberg atom simulation.

This module implements PEPS with nearest-neighbor blockade constraint
(n_i * n_j = 0 for adjacent sites), enforced by only parameterizing valid
sector configurations - similar to gauge-invariant PEPS.
"""
from vmc.experimental.rydberg.blockade_peps import (
    BlockadePEPS,
    BlockadePEPSConfig,
    random_independent_set,
)
from vmc.experimental.rydberg.hamiltonian import rydberg_hamiltonian

# Import to register dispatches (not directly exported)
from vmc.experimental.rydberg import blockade_sampler as _blockade_sampler  # noqa: F401

__all__ = [
    "BlockadePEPS",
    "BlockadePEPSConfig",
    "random_independent_set",
    "rydberg_hamiltonian",
]
