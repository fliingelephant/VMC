"""VMC MCP server — structured tools for the PEPS-tVMC codebase.

Run with:  uv run python tools/vmc-mcp/server.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the tools directory and project src are on the path.
_tools_dir = str(Path(__file__).resolve().parent)
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)
_src_dir = str(Path(__file__).resolve().parents[2] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from mcp.server.fastmcp import FastMCP

import discovery
import compatibility as compat_mod
import experience as exp_mod
import visualization as viz_mod
import runner_tools

mcp = FastMCP("vmc-mcp")


# ---------------------------------------------------------------------------
# Discovery tools
# ---------------------------------------------------------------------------

@mcp.tool(description="List all PEPS model families (standard, gauge-invariant, blockade).")
def tool_list_models() -> str:
    return json.dumps(discovery.list_models(), indent=2)


@mcp.tool(description="List all operator term types (standard and GI-specific).")
def tool_list_operators() -> str:
    return json.dumps(discovery.list_operators(), indent=2)


@mcp.tool(description="List all contraction strategies for boundary-MPO contraction.")
def tool_list_strategies() -> str:
    return json.dumps(discovery.list_strategies(), indent=2)


@mcp.tool(description="List QGT solvers and QGT space formulations.")
def tool_list_solvers() -> str:
    return json.dumps(discovery.list_solvers(), indent=2)


@mcp.tool(description="List example scripts with descriptions and model families.")
def tool_list_examples() -> str:
    return json.dumps(discovery.list_examples(), indent=2)


@mcp.tool(description="Find the example script that best matches a description.")
def tool_find_closest_example(description: str) -> str:
    result = discovery.find_closest_example(description)
    if result is None:
        return json.dumps({"error": "No examples found."})
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Compatibility tools
# ---------------------------------------------------------------------------

@mcp.tool(description="Check if a model supports the given operator term types.")
def tool_check_compatibility(model: str, term_types: list[str]) -> str:
    return json.dumps(compat_mod.check_compatibility(model, term_types), indent=2)


@mcp.tool(description="Check if a simulation config is feasible with this codebase.")
def tool_check_feasibility(config: dict) -> str:
    return json.dumps(compat_mod.check_feasibility(config), indent=2)


# ---------------------------------------------------------------------------
# Experience tools
# ---------------------------------------------------------------------------

@mcp.tool(description="Query EXPERIENCE.md for practitioner advice on a topic.")
def tool_query_experience(topic: str) -> str:
    return json.dumps(exp_mod.query_experience(topic), indent=2)


# ---------------------------------------------------------------------------
# Visualization tools
# ---------------------------------------------------------------------------

@mcp.tool(description="Plot observable convergence from runner output.")
def tool_plot_convergence(run_dir: str, keys: list[str] | None = None) -> str:
    return json.dumps(viz_mod.plot_convergence(run_dir, keys), indent=2)


@mcp.tool(description="Plot a 2D heatmap of spatial observables at a given step.")
def tool_plot_heatmap(run_dir: str, step: int, observable_prefix: str) -> str:
    return json.dumps(viz_mod.plot_heatmap(run_dir, step, observable_prefix), indent=2)


@mcp.tool(description="Create a GIF animation of spatial observables over time.")
def tool_animate(run_dir: str, observable_prefix: str, fps: int = 5) -> str:
    return json.dumps(viz_mod.animate(run_dir, observable_prefix, fps), indent=2)


# ---------------------------------------------------------------------------
# Runner tools
# ---------------------------------------------------------------------------

@mcp.tool(description="Run a script with tiny parameters to verify it doesn't crash.")
def tool_smoke_test(
    script_path: str,
    overrides: dict | None = None,
    chain_state: str | None = None,
) -> str:
    return json.dumps(
        runner_tools.smoke_test(script_path, overrides, chain_state), indent=2,
    )


@mcp.tool(description="Read parsed metadata from a runner checkpoint.")
def tool_read_checkpoint_metadata(run_dir: str) -> str:
    return json.dumps(runner_tools.read_checkpoint_metadata(run_dir), indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
