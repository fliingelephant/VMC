"""Small-o metadata for tensor network states."""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

__all__ = ["params_per_site", "sliced_dims"]


def params_per_site(model) -> tuple[int, ...]:
    """Number of parameters in the active slice at each site."""
    return model.params_per_site


def sliced_dims(model) -> tuple[int, ...]:
    """Number of distinct active slices at each site."""
    return model.sliced_dims
