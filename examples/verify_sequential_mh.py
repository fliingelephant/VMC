#!/usr/bin/env python3
"""Entry point for sequential MH verification under the refactored package."""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> None:
    _ensure_repo_on_path()
    from VMC.examples.verify_sequential_mh import main as run

    run()


if __name__ == "__main__":
    main()
