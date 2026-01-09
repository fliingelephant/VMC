"""Utility helpers for VMC workflows."""
from __future__ import annotations

from VMC.utils.independent_set_sampling import (
    DiscardBlockedSampler,
    IndependentSetSampler,
    all_config_batches,
    build_neighbor_arrays,
    config_codes,
    enumerate_all_configs,
    enumerate_independent_sets_grid,
    grid_edges,
    independent_set_violations,
)
from VMC.utils.utils import (
    occupancy_to_spin,
    spin_to_occupancy,
)
from VMC.utils.vmc_utils import (
    batched_eval,
    build_dense_jac,
    build_dense_jac_from_state,
    flatten_samples,
    get_apply_fun,
)

__all__ = [
    "DiscardBlockedSampler",
    "IndependentSetSampler",
    "all_config_batches",
    "batched_eval",
    "build_neighbor_arrays",
    "config_codes",
    "enumerate_all_configs",
    "enumerate_independent_sets_grid",
    "grid_edges",
    "independent_set_violations",
    "occupancy_to_spin",
    "spin_to_occupancy",
    "build_dense_jac",
    "build_dense_jac_from_state",
    "flatten_samples",
    "get_apply_fun",
]
