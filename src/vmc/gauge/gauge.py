"""Gauge removal configuration and API."""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

from dataclasses import dataclass

import jax
from plum import dispatch

__all__ = [
    "GaugeConfig",
    "compute_gauge_projection",
]


@dataclass(frozen=True)
class GaugeConfig:
    """Configuration for gauge removal."""

    rcond: float | None = None
    include_global_scale: bool = True


@dispatch
def compute_gauge_projection(
    cfg: GaugeConfig,
    model: object,
    params: dict,
    *,
    return_info: bool = False,
) -> jax.Array | tuple[jax.Array, dict]:
    """Gauge projection is not implemented in the PEPS-only runtime."""
    raise NotImplementedError(
        "Gauge removal is not implemented in the PEPS-only runtime. "
        "Use gauge_config=None."
    )
