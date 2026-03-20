"""Tests for the VMC MCP discovery module."""
from __future__ import annotations

import sys
from pathlib import Path

# Enable JAX float64 before any vmc imports.
from vmc import config  # noqa: F401

# Add the tools directory so discovery is importable.
_tools_dir = str(Path(__file__).resolve().parents[1] / "tools" / "vmc-mcp")
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

import discovery  # noqa: E402


# ------------------------------------------------------------------ #
# list_models
# ------------------------------------------------------------------ #


def test_list_models_returns_three():
    models = discovery.list_models()
    assert len(models) == 3


def test_list_models_expected_names():
    names = {m["name"] for m in discovery.list_models()}
    assert names == {"PEPS", "GIPEPS", "BlockadePEPS"}


def test_list_models_have_required_keys():
    for m in discovery.list_models():
        assert "name" in m
        assert "module" in m
        assert "description" in m
        assert "key_parameters" in m
        assert len(m["description"]) > 0


# ------------------------------------------------------------------ #
# list_operators
# ------------------------------------------------------------------ #


def test_list_operators_standard():
    ops = discovery.list_operators()
    names = {o["name"] for o in ops}
    for expected in (
        "OneSiteOperator",
        "DiagonalOperator",
        "HorizontalTwoSiteOperator",
        "VerticalTwoSiteOperator",
        "PlaquetteOperator",
    ):
        assert expected in names, f"Missing standard operator: {expected}"


def test_list_operators_gi():
    ops = discovery.list_operators()
    names = {o["name"] for o in ops}
    for expected in (
        "LinkDiagonalTerm",
        "MatterMassTerm",
        "HorizontalMatterHoppingTerm",
        "VerticalMatterHoppingTerm",
        "HorizontalHiggsLinkTerm",
        "VerticalHiggsLinkTerm",
    ):
        assert expected in names, f"Missing GI operator: {expected}"


def test_list_operators_have_required_keys():
    for o in discovery.list_operators():
        assert "name" in o
        assert "module" in o
        assert "description" in o
        assert len(o["description"]) > 0


# ------------------------------------------------------------------ #
# list_strategies
# ------------------------------------------------------------------ #


def test_list_strategies_expected():
    strats = discovery.list_strategies()
    names = {s["name"] for s in strats}
    assert names == {"NoTruncation", "ZipUp", "DensityMatrix", "Variational"}


def test_list_strategies_have_required_keys():
    for s in discovery.list_strategies():
        assert "name" in s
        assert "module" in s
        assert "description" in s


# ------------------------------------------------------------------ #
# list_solvers
# ------------------------------------------------------------------ #


def test_list_solvers_expected_solvers():
    solvers = discovery.list_solvers()
    solver_names = {s["name"] for s in solvers if s["kind"] == "solver"}
    assert solver_names == {"solve_cholesky", "solve_svd", "solve_cg"}


def test_list_solvers_expected_spaces():
    solvers = discovery.list_solvers()
    space_names = {s["name"] for s in solvers if s["kind"] == "space"}
    assert space_names == {"ParameterSpace", "SampleSpace"}


def test_list_solvers_have_required_keys():
    for s in discovery.list_solvers():
        assert "name" in s
        assert "module" in s
        assert "kind" in s
        assert "description" in s


# ------------------------------------------------------------------ #
# list_examples
# ------------------------------------------------------------------ #


def test_list_examples_nonempty():
    examples = discovery.list_examples()
    assert len(examples) > 0


def test_list_examples_skips_excluded_files():
    examples = discovery.list_examples()
    basenames = {Path(e["path"]).name for e in examples}
    for skip in ("runner.py", "__init__.py", "physics.py", "common.py", "plot.py"):
        assert skip not in basenames, f"Should skip {skip}"


def test_list_examples_have_required_keys():
    for e in discovery.list_examples():
        assert "path" in e
        assert "description" in e
        assert "model_family" in e
        assert e["model_family"] in ("standard", "gi", "blockade")


def test_list_examples_contains_known_scripts():
    examples = discovery.list_examples()
    paths = {e["path"] for e in examples}
    assert any("heisenberg" in p for p in paths)
    assert any("z2_pure_gauge" in p for p in paths)


# ------------------------------------------------------------------ #
# find_closest_example
# ------------------------------------------------------------------ #


def test_find_closest_example_ising():
    result = discovery.find_closest_example("TFIM Ising ground state benchmark")
    assert result is not None
    assert "ising" in result["path"].lower() or "ising" in result["description"].lower()


def test_find_closest_example_z2_gauge():
    result = discovery.find_closest_example("Z2 pure gauge theory")
    assert result is not None
    assert "z2" in result["path"].lower() or "z2" in result["description"].lower()


def test_find_closest_example_heisenberg():
    result = discovery.find_closest_example("Heisenberg antiferromagnet")
    assert result is not None
    assert "heisenberg" in result["path"].lower()


def test_find_closest_example_vison():
    result = discovery.find_closest_example("vison dynamics propagation")
    assert result is not None
    assert "vison" in result["path"].lower()
