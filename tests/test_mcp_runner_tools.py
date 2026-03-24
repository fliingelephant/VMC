"""Tests for MCP runner tools."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_tools_dir = str(Path(__file__).resolve().parents[1] / "tools" / "vmc-mcp")
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

from runner_tools import read_checkpoint_metadata


def test_read_checkpoint_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_data = {
            "step": 100,
            "time": 1.0,
            "config": {"model": "GIPEPS", "shape": [6, 6]},
        }
        (Path(tmpdir) / "latest.json").write_text(json.dumps(json_data))
        meta = read_checkpoint_metadata(tmpdir)
        assert meta["step"] == 100
        assert meta["config"]["model"] == "GIPEPS"


def test_read_checkpoint_metadata_with_series():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_data = {
            "step": 50,
            "time": 0.5,
            "config": {},
            "series": {"step": [1, 2], "energy_mean": [-0.5, -0.6]},
        }
        (Path(tmpdir) / "latest.json").write_text(json.dumps(json_data))
        meta = read_checkpoint_metadata(tmpdir)
        assert meta["series"]["energy_mean"] == [-0.5, -0.6]
