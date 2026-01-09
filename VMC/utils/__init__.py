"""Utility helpers for VMC workflows."""
from __future__ import annotations

from VMC.utils.independent_set_sampling import (
    IndependentSetSampler,
    build_neighbor_arrays,
    independent_set_violations,
    occupancy_to_spin,
    spin_to_occupancy,
)
from VMC.utils.vmc_utils import (
    build_dense_jac,
    build_dense_jac_from_state,
    flatten_samples,
    get_apply_fun,
)

__all__ = [
    "IndependentSetSampler",
    "build_neighbor_arrays",
    "independent_set_violations",
    "occupancy_to_spin",
    "spin_to_occupancy",
    "build_dense_jac",
    "build_dense_jac_from_state",
    "flatten_samples",
    "get_apply_fun",
]
