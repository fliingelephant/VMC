"""Tests for MCP visualization tools."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_tools_dir = str(Path(__file__).resolve().parents[1] / "tools" / "vmc-mcp")
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

from visualization import plot_convergence, plot_heatmap, animate


def _make_mock_run_dir(tmpdir: str, extra_series: dict | None = None) -> Path:
    """Create minimal runner output for testing."""
    run_dir = Path(tmpdir)
    series = {
        "step": [1, 2, 3],
        "time": [0.01, 0.02, 0.03],
        "energy_mean": [-0.5, -0.6, -0.65],
        "energy_error": [0.1, 0.05, 0.02],
        "energy_variance": [0.5, 0.3, 0.1],
    }
    if extra_series:
        series.update(extra_series)
    (run_dir / "latest.json").write_text(
        json.dumps({"step": 3, "time": 0.03, "series": series})
    )
    return run_dir


def test_plot_convergence():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = _make_mock_run_dir(tmpdir)
        result = plot_convergence(str(run_dir))
        assert Path(result["path"]).exists()
        assert result["path"].endswith(".png")


def test_plot_convergence_with_observables():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = _make_mock_run_dir(tmpdir, {
            "mx_mean": [0.5, 0.4, 0.3],
            "mx_error": [0.05, 0.03, 0.01],
        })
        result = plot_convergence(str(run_dir))
        assert Path(result["path"]).exists()
        assert "mx_mean" in result["description"]


def test_plot_heatmap_underscore_naming():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = _make_mock_run_dir(tmpdir, {
            "P_0_0_mean": [0.8, 0.7, 0.6],
            "P_0_1_mean": [0.7, 0.6, 0.5],
            "P_1_0_mean": [0.6, 0.5, 0.4],
            "P_1_1_mean": [0.5, 0.4, 0.3],
        })
        result = plot_heatmap(str(run_dir), step=0, observable_prefix="P_")
        assert Path(result["path"]).exists()


def test_plot_heatmap_concat_naming():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = _make_mock_run_dir(tmpdir, {
            "P_00_mean": [0.8, 0.7, 0.6],
            "P_01_mean": [0.7, 0.6, 0.5],
            "P_10_mean": [0.6, 0.5, 0.4],
            "P_11_mean": [0.5, 0.4, 0.3],
        })
        result = plot_heatmap(str(run_dir), step=0, observable_prefix="P_")
        assert Path(result["path"]).exists()


def test_animate():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = _make_mock_run_dir(tmpdir, {
            "P_0_0_mean": [0.8, 0.7, 0.6],
            "P_0_1_mean": [0.7, 0.6, 0.5],
            "P_1_0_mean": [0.6, 0.5, 0.4],
            "P_1_1_mean": [0.5, 0.4, 0.3],
        })
        result = animate(str(run_dir), "P_")
        assert Path(result["path"]).exists()
        assert result["path"].endswith(".gif")
