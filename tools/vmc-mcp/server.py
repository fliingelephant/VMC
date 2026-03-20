"""VMC MCP server entry point.

Registers discovery tools that let agents query what the codebase can do.
Run with:  uv run python tools/vmc-mcp/server.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the tools directory is on the path so discovery can be imported.
_tools_dir = str(Path(__file__).resolve().parent)
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

# Ensure the project src directory is on the path for vmc imports.
_src_dir = str(Path(__file__).resolve().parents[2] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from mcp.server.fastmcp import FastMCP

import discovery

mcp = FastMCP("vmc-mcp")


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


if __name__ == "__main__":
    mcp.run(transport="stdio")
