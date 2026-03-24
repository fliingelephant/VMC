"""Tests for MCP compatibility tools."""
from __future__ import annotations

import sys
from pathlib import Path

_tools_dir = str(Path(__file__).resolve().parents[1] / "tools" / "vmc-mcp")
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

from compatibility import check_compatibility, check_feasibility


def test_peps_with_standard_operators():
    result = check_compatibility("PEPS", ["OneSiteOperator", "DiagonalOperator"])
    assert result["compatible"] is True


def test_gipeps_with_plaquette():
    result = check_compatibility("GIPEPS", ["PlaquetteOperator", "LinkDiagonalTerm"])
    assert result["compatible"] is True


def test_peps_with_gi_terms():
    result = check_compatibility("PEPS", ["MatterMassTerm"])
    assert result["compatible"] is False


def test_unknown_model():
    result = check_compatibility("UnknownModel", ["OneSiteOperator"])
    assert result["compatible"] is False


def test_feasibility_z2_ground_state():
    config = {
        "gauge_group": "Z2",
        "lattice": (8, 8),
        "terms": ["electric", "plaquette"],
        "dynamics": "imaginary_time",
    }
    result = check_feasibility(config)
    assert result["feasible"] is True
    assert "GIPEPS" in result["suggested_model"]


def test_feasibility_unsupported_gauge():
    config = {"gauge_group": "SU2", "lattice": (8, 8), "terms": ["plaquette"]}
    result = check_feasibility(config)
    assert result["feasible"] is False


def test_feasibility_standard_peps():
    config = {"lattice": (4, 4), "terms": ["onsite", "diagonal"], "dynamics": "real_time"}
    result = check_feasibility(config)
    assert result["feasible"] is True
    assert result["suggested_model"] == "PEPS"


def test_feasibility_higgs():
    config = {
        "gauge_group": "Z2",
        "lattice": (8, 8),
        "terms": ["electric", "plaquette", "matter_mass", "higgs"],
    }
    result = check_feasibility(config)
    assert result["feasible"] is True
    assert any("conserve_particle_number" in n for n in result.get("notes", []))
