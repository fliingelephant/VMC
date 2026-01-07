"""Gauge removal utilities for tensor network models."""
from __future__ import annotations

from VMC.gauge.gauge import GaugeConfig, compute_gauge_projection

__all__ = [
    "GaugeConfig",
    "compute_gauge_projection",
]
