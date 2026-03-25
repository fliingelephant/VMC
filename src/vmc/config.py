"""JAX and logging configuration module.

This module must be imported before any other JAX imports to ensure
64-bit precision is enabled throughout the codebase.
"""
from __future__ import annotations

import logging
import os

from jax import config

# Enable 64-bit precision for JAX
config.update("jax_enable_x64", True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.expanduser("jax_cache")
config.update("jax_persistent_cache_min_compile_time_secs", 0)
config.update("jax_persistent_cache_min_entry_size_bytes", -1)
config.update("jax_persistent_cache_enable_xla_caches", "all")

_xla_flags = os.environ.get("XLA_FLAGS", "")
if "xla_gpu_force_compilation_parallelism" not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} --xla_gpu_force_compilation_parallelism=16".strip()


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for an application entry point.

    Call this from your script's main(), not at import time.
    The library should not configure logging — that's the application's job.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

