"""Tests for MCP runner tools."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_tools_dir = str(Path(__file__).resolve().parents[1] / "tools" / "vmc-mcp")
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

from runner_tools import _load_jsonl, _jsonl_to_columnar


def test_load_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "metrics.jsonl"
        path.write_text('{"step":1,"energy":-0.5}\n{"step":2,"energy":-0.6}\n')
        rows = _load_jsonl(path)
        assert len(rows) == 2
        assert rows[0]["step"] == 1
        assert rows[1]["energy"] == -0.6


def test_jsonl_to_columnar():
    rows = [
        {"step": 1, "energy_mean": -0.5},
        {"step": 2, "energy_mean": -0.6},
    ]
    series = _jsonl_to_columnar(rows)
    assert series["step"] == [1, 2]
    assert series["energy_mean"] == [-0.5, -0.6]
