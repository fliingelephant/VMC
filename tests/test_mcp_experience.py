"""Tests for MCP experience query tools."""
from __future__ import annotations

import sys
from pathlib import Path

_tools_dir = str(Path(__file__).resolve().parents[1] / "tools" / "vmc-mcp")
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

from experience import query_experience


def test_query_gpu():
    results = query_experience("GPU")
    assert len(results) > 0
    assert any("Variational" in r for r in results)


def test_query_bond_dimension():
    results = query_experience("bond dimension Z2")
    assert len(results) > 0
    assert any("D_k=2" in r for r in results)


def test_query_solver():
    results = query_experience("solver")
    assert len(results) > 0
    assert any("Cholesky" in r for r in results)


def test_query_minsr():
    results = query_experience("minSR")
    assert len(results) > 0


def test_query_convergence():
    results = query_experience("convergence")
    assert len(results) > 0


def test_query_no_match():
    results = query_experience("quantum error correction")
    assert len(results) == 0
