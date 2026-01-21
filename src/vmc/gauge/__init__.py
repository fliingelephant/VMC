"""Gauge removal utilities for tensor network models."""
from __future__ import annotations

from vmc.gauge.gauge import GaugeConfig, compute_gauge_projection
from vmc.gauge.weight import WeightConfig, compute_weight_projection

__all__ = [
    "GaugeConfig",
    "compute_gauge_projection",
    "WeightConfig",
    "compute_weight_projection",
]
