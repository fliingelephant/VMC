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


def setup_logging() -> None:
    """Configure logging based on environment variables.

    Control log level via VMC_LOG_LEVEL environment variable.

    Examples:
        # Default (WARNING level) - skip expensive debug computations
        python main.py

        # Debug mode - compute and log everything
        VMC_LOG_LEVEL=DEBUG python main.py

        # Info mode - basic progress information
        VMC_LOG_LEVEL=INFO python main.py
    """
    level_name = os.environ.get("VMC_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)

    logging.basicConfig(
        level=level,
        format="%(name)s - %(levelname)s - %(message)s",
    )


# Configure logging on import
setup_logging()

