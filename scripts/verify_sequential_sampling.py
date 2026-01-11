#!/usr/bin/env python3
"""Compatibility wrapper for sequential MH verification."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> None:
    _ensure_repo_on_path()
    from VMC import config  # noqa: F401 - configure logging
    logger = logging.getLogger(__name__)
    logger.info("Redirecting to VMC.examples.verify_sequential_mh")
    from VMC.examples.verify_sequential_mh import main as run

    run()


if __name__ == "__main__":
    main()
