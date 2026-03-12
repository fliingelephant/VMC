"""Shared PEPS contraction backend."""
from __future__ import annotations

from vmc.peps.common.contraction import (
    _apply_mpo_from_below,
    _build_row_mpo,
    _compute_right_envs,
    _contract_bottom,
    _forward_with_cache,
)
from vmc.peps.common.energy import (
    RowEnvs,
    TwoRowEnvs,
    _eval_term,
    _compute_all_gradients,
    _compute_all_row_gradients,
    _estimate_sweep,
    _compute_right_envs_2row,
    _compute_single_gradient,
)
from vmc.peps.common.strategy import (
    ContractionStrategy,
    DensityMatrix,
    NoTruncation,
    Variational,
    ZipUp,
)

__all__ = [
    "ContractionStrategy",
    "DensityMatrix",
    "NoTruncation",
    "Variational",
    "ZipUp",
    "_apply_mpo_from_below",
    "_build_row_mpo",
    "_compute_right_envs",
    "_contract_bottom",
    "_forward_with_cache",
    "RowEnvs",
    "TwoRowEnvs",
    "_eval_term",
    "_compute_all_gradients",
    "_compute_all_row_gradients",
    "_estimate_sweep",
    "_compute_right_envs_2row",
    "_compute_single_gradient",
]
