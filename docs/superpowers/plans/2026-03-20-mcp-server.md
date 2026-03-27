# VMC MCP Server & EXPERIENCE.md Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an MCP server (`vmc-mcp`) providing structured tools for codebase discovery, compatibility checking, experience queries, visualization, and smoke testing — plus an EXPERIENCE.md knowledge base.

**Architecture:** Single Python MCP server using the `mcp` SDK, importing from `vmc` directly for live introspection. Tools organized into modules by domain (discovery, compatibility, experience, visualization, runner). EXPERIENCE.md is a flat markdown file at the project root, parsed by the experience module. The server is configured in `.claude/settings.json`.

**Tech Stack:** Python MCP SDK (`mcp`), JAX, matplotlib, imageio (for GIFs), `vmc` package.

**Spec:** `docs/superpowers/specs/2026-03-20-simulation-skills-design.md`

---

## File Structure

```
EXPERIENCE.md                              # project root — practitioner wisdom
tools/vmc-mcp/
  server.py                                # MCP entry point, registers all tools
  discovery.py                             # list_models, list_operators, list_strategies, list_solvers, list_examples, find_closest_example
  compatibility.py                         # COMPATIBILITY_MATRIX, check_compatibility, check_feasibility
  experience.py                            # parse EXPERIENCE.md, query_experience
  visualization.py                         # plot_convergence, plot_heatmap, animate
  runner_tools.py                          # smoke_test, read_checkpoint_metadata
tests/
  test_mcp_discovery.py                    # tests for discovery tools
  test_mcp_compatibility.py                # tests for compatibility tools
  test_mcp_experience.py                   # tests for experience query
  test_mcp_visualization.py               # tests for visualization tools
.claude/settings.json                      # MCP server registration (modify)
```

---

### Task 1: EXPERIENCE.md

**Files:**
- Create: `EXPERIENCE.md`

Create the practitioner knowledge base with initial entries from our brainstorming discussion and the papers.

- [ ] **Step 1: Write EXPERIENCE.md**

```markdown
# EXPERIENCE.md — Practitioner Knowledge for PEPS-tVMC

Operational wisdom accumulated from running simulations. Not systematic
rules (see CLAUDE.md for those), but the kind of knowledge that belongs
to a skilled practitioner (熟练工).

## Contraction Strategy

- **Always use Variational on GPU.** ZipUp involves SVD which is not well
  batched on GPU. Variational uses iterative sweeps that parallelize better.
- **ZipUp is fine on CPU** for small systems (L <= 6) where SVD cost is
  manageable.
- **Boundary dimension D' ~ 2D to 3D** is typical for Variational. Too small
  gives inaccurate contraction; too large wastes compute without improving
  accuracy.

## Bond Dimension

- **Z2 LGT converges well with D_k=2** for lattice sizes up to 32x32
  (Wu & Liu 2025, Table I).
- **For Z2 Higgs, D_k=2 is sufficient** for both deconfined and Higgs phases
  (Wu & Nys 2026).
- **Start with D_k=2, increase if energy variance is large.**
- **Standard PEPS (Heisenberg, TFIM):** D=3-4 for small lattices (L <= 8),
  D=6-8 for production (L >= 16). See Liu et al. 2021 for finite-size scaling.

## Solver Choice

- **Cholesky is default.** Faster and more parallelizable on GPU than SVD.
- **SVD is more robust** for ill-conditioned QGT. Use when Cholesky gives NaN.
- **CG (conjugate gradient)** for very large parameter counts where direct
  solve is too expensive.

## Solver Space (SR vs minSR)

- **For GIPEPS with large per-site parameters, use minSR** (`SampleSpace`).
  Avoids materializing the full Jacobian.
- **For standard PEPS, SR** (`ParameterSpace`) is fine — N_p is typically
  smaller than N_s.
- **Crossover:** when N_s > N_p - N_gv - 2, use minSR
  (Wu & Nys 2026, Sec. III.C).

## Sampling

- **n_samples=10240, n_chains=1024** is a good starting point for production
  on GPU.
- **For testing/debugging, use n_samples=64, n_chains=8.**
- **Sequential sampling** visits bonds in order along the lattice, reducing
  cost from O(N_site^2) to O(N_site) per sweep (Liu et al. 2021).

## Time Steps

- **Imaginary time:** dt=0.005 to 0.01 is typical for ground-state SR.
- **Real time:** dt=0.005 to 0.01 for RK4. Smaller dt gives better energy
  conservation but costs more steps.
- **For quenches with smooth ramps** (Schmitt protocol), dt=0.01 works well
  (Wu & Nys 2026, Fig. 5c).

## Convergence

- **FS_norm_squared should decrease** during imaginary-time optimization. If
  it plateaus, the state is near convergence or a local minimum.
- **Energy drift < 0.5%** over the full trajectory indicates stable real-time
  evolution.
- **TDVP residual 1e-9 to 1e-25** is normal and indicates the TDVP equation
  is being solved accurately (Wu & Nys 2026, Fig. 6).

## Gauge Removal

- **Always use gauge removal for real-time dynamics.** Without it, the QGT
  is ill-conditioned and the TDVP equation is unstable.
- **For imaginary-time (ground state), gauge removal is optional** but
  improves convergence speed.
- **minSR achieves gauge removal automatically** — the parameter index is
  contracted away, so gauge directions vanish.

## Diag Shift (Tikhonov Regularization)

- **1e-4 for ground state** is a safe starting point.
- **1e-6 to 1e-8 for real-time dynamics** — needs to be small to avoid
  biasing the time evolution, but large enough to regularize.
- **If solver gives NaN, increase diag_shift** or switch to SVD solver.
```

- [ ] **Step 2: Commit**

```bash
git add EXPERIENCE.md
git commit -m "Add EXPERIENCE.md — practitioner knowledge base"
```

---

### Task 2: MCP server scaffold + discovery tools

**Files:**
- Create: `tools/vmc-mcp/server.py`
- Create: `tools/vmc-mcp/discovery.py`
- Create: `tests/test_mcp_discovery.py`

The discovery module introspects the codebase to list available models, operators, strategies, solvers, and examples.

- [ ] **Step 1: Write test for discovery tools**

```python
"""Tests for MCP discovery tools."""
from __future__ import annotations

from vmc import config  # noqa: F401

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "vmc-mcp"))

from discovery import (
    list_models,
    list_operators,
    list_strategies,
    list_solvers,
    list_examples,
)


def test_list_models():
    models = list_models()
    names = [m["name"] for m in models]
    assert "PEPS" in names
    assert "GIPEPS" in names


def test_list_operators():
    ops = list_operators()
    names = [o["name"] for o in ops]
    assert "OneSiteOperator" in names
    assert "PlaquetteOperator" in names
    # GI-specific terms must also be listed
    assert "MatterMassTerm" in names


def test_list_strategies():
    strategies = list_strategies()
    names = [s["name"] for s in strategies]
    assert "Variational" in names
    assert "ZipUp" in names


def test_list_solvers():
    solvers = list_solvers()
    names = [s["name"] for s in solvers]
    assert "solve_cholesky" in names
    assert "solve_svd" in names


def test_list_examples():
    examples = list_examples()
    assert len(examples) > 0
    # Should find z2_vison ground_state
    paths = [e["path"] for e in examples]
    assert any("z2_vison" in p for p in paths)
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_mcp_discovery.py -v`

- [ ] **Step 3: Implement discovery.py**

The discovery module uses introspection to find available types. Key implementation details:
- `list_models()`: scan `vmc.peps` for model classes (PEPS, GIPEPS, BlockadePEPS)
- `list_operators()`: scan both `vmc.operators.local_terms` AND `vmc.peps.gi.local_terms` for operator classes
- `list_strategies()`: scan `vmc.peps.common.strategy`
- `list_solvers()`: return the `SOLVERS` dict from `runner.py` plus solver spaces
- `list_examples()`: walk `examples/` directory, read docstrings from .py files
- `find_closest_example(description)`: keyword match against example descriptions

Each function returns a list of dicts with `name`, `module`, `description`, and relevant parameters.

- [ ] **Step 4: Implement server.py scaffold**

The MCP server entry point using the `mcp` SDK. Registers tools from each module. Initially only discovery tools — other modules added in later tasks.

Note: `mcp` SDK needs to be added to project dependencies: `uv add --dev mcp`

```python
"""VMC MCP server — structured tools for the PEPS-tVMC codebase."""
from mcp.server import Server
from mcp.server.stdio import stdio_server
import json

app = Server("vmc-mcp")

# Import and register tool modules
from discovery import (
    list_models, list_operators, list_strategies,
    list_solvers, list_examples, find_closest_example,
)

@app.tool()
async def tool_list_models() -> str:
    """List available PEPS model families with their parameters."""
    return json.dumps(list_models(), indent=2)

# ... register each tool similarly ...

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

- [ ] **Step 5: Run tests — expect PASS**
- [ ] **Step 6: Commit**

---

### Task 3: Compatibility tools

**Files:**
- Create: `tools/vmc-mcp/compatibility.py`
- Create: `tests/test_mcp_compatibility.py`

The compatibility matrix and feasibility checker.

- [ ] **Step 1: Write tests**

```python
"""Tests for MCP compatibility tools."""
from __future__ import annotations

from vmc import config  # noqa: F401

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "vmc-mcp"))

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


def test_feasibility_unsupported():
    config = {
        "gauge_group": "SU2",
        "lattice": (8, 8),
        "terms": ["plaquette"],
    }
    result = check_feasibility(config)
    assert result["feasible"] is False
```

- [ ] **Step 2: Run tests — expect FAIL**
- [ ] **Step 3: Implement compatibility.py**

The compatibility matrix is a curated dict:

```python
# Which term types are valid with which model family
TERM_MODEL_COMPAT = {
    "PEPS": {
        "OneSiteOperator", "DiagonalOperator",
        "HorizontalTwoSiteOperator", "VerticalTwoSiteOperator",
        "PlaquetteOperator",
    },
    "GIPEPS": {
        "PlaquetteOperator", "LinkDiagonalTerm",
        "MatterMassTerm",
        "HorizontalMatterHoppingTerm", "VerticalMatterHoppingTerm",
        "HorizontalHiggsLinkTerm", "VerticalHiggsLinkTerm",
    },
    "BlockadePEPS": {
        "OneSiteOperator", "DiagonalOperator",
        "HorizontalTwoSiteOperator", "VerticalTwoSiteOperator",
        "PlaquetteOperator",
    },
}

# Supported gauge groups
GAUGE_GROUPS = {"Z2": 2, "Z3": 3, "Z4": 4}  # name -> N
```

`check_compatibility(model, term_types)` checks against the matrix.
`check_feasibility(config)` maps physics description to model family + terms, then checks compatibility. Returns `{feasible, suggested_model, reason, missing_features}`.

- [ ] **Step 4: Run tests — expect PASS**
- [ ] **Step 5: Register tools in server.py**
- [ ] **Step 6: Commit**

---

### Task 4: Experience query tools

**Files:**
- Create: `tools/vmc-mcp/experience.py`
- Create: `tests/test_mcp_experience.py`

Parses EXPERIENCE.md and answers topic-based queries.

- [ ] **Step 1: Write tests**

```python
"""Tests for MCP experience tools."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "vmc-mcp"))

from experience import query_experience


def test_query_gpu():
    results = query_experience("GPU")
    assert len(results) > 0
    assert any("Variational" in r for r in results)


def test_query_bond_dimension():
    results = query_experience("bond dimension Z2")
    assert len(results) > 0
    assert any("D_k=2" in r for r in results)


def test_query_no_match():
    results = query_experience("quantum error correction")
    assert len(results) == 0
```

- [ ] **Step 2: Run tests — expect FAIL**
- [ ] **Step 3: Implement experience.py**

Parse EXPERIENCE.md into sections and entries. `query_experience(topic)` does keyword matching against section headers and entry text. Returns list of matching entry strings.

```python
def _parse_experience(path: Path) -> list[dict]:
    """Parse EXPERIENCE.md into {section, entries} list."""
    # Split by ## headers, then by - bullet points
    ...

def query_experience(topic: str, experience_path: Path = None) -> list[str]:
    """Return entries matching the topic keywords."""
    ...
```

- [ ] **Step 4: Run tests — expect PASS**
- [ ] **Step 5: Register in server.py**
- [ ] **Step 6: Commit**

---

### Task 5: Visualization tools

**Files:**
- Create: `tools/vmc-mcp/visualization.py`
- Create: `tests/test_mcp_visualization.py`

Plotting from runner output: convergence, heatmaps, GIF animations.

- [ ] **Step 1: Write tests**

Tests use a mock run_dir with a synthetic latest.json containing minimal series data.

```python
"""Tests for MCP visualization tools."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "vmc-mcp"))

from visualization import plot_convergence, plot_heatmap


def _make_mock_run_dir(tmpdir: str) -> Path:
    """Create a minimal runner output for testing."""
    run_dir = Path(tmpdir)
    series = {
        "step": [1, 2, 3],
        "time": [0.01, 0.02, 0.03],
        "energy_mean": [-0.5, -0.6, -0.65],
        "energy_error": [0.1, 0.05, 0.02],
        "energy_variance": [0.5, 0.3, 0.1],
    }
    json_data = {"step": 3, "time": 0.03, "series": series, "config": {}}
    (run_dir / "latest.json").write_text(json.dumps(json_data))
    return run_dir


def test_plot_convergence():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = _make_mock_run_dir(tmpdir)
        result = plot_convergence(str(run_dir))
        assert Path(result["path"]).exists()
        assert result["path"].endswith(".png")


def test_plot_heatmap_with_plaquettes():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        # 3x3 lattice → 2x2 plaquette grid
        series = {
            "step": [1], "time": [0.01],
            "energy_mean": [-1.0], "energy_error": [0.01],
            "energy_variance": [0.1],
            "P_0_0_mean": [0.8], "P_0_1_mean": [0.7],
            "P_1_0_mean": [0.6], "P_1_1_mean": [0.5],
        }
        json_data = {"step": 1, "time": 0.01, "series": series, "config": {}}
        (run_dir / "latest.json").write_text(json.dumps(json_data))
        result = plot_heatmap(str(run_dir), step=0, observable_prefix="P_")
        assert Path(result["path"]).exists()
```

- [ ] **Step 2: Run tests — expect FAIL**
- [ ] **Step 3: Implement visualization.py**

Three functions:
- `plot_convergence(run_dir, keys=None)`: reads latest.json series, plots selected keys vs time using matplotlib. Saves to `run_dir/convergence.png`.
- `plot_heatmap(run_dir, step, observable_prefix)`: extracts 2D grid from flat `{prefix}{r}_{c}_mean` keys at given step index. Saves to `run_dir/heatmap_step{N}.png`.
- `animate(run_dir, observable_prefix, fps=5)`: generates heatmap per step, assembles into GIF using `imageio`. Saves to `run_dir/animation.gif`.

All functions return `{"path": str, "description": str}`.

- [ ] **Step 4: Run tests — expect PASS**
- [ ] **Step 5: Register in server.py**
- [ ] **Step 6: Commit**

---

### Task 6: Runner tools (smoke test + checkpoint metadata)

**Files:**
- Create: `tools/vmc-mcp/runner_tools.py`
- Create: `tests/test_mcp_runner_tools.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for MCP runner tools."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "vmc-mcp"))

from runner_tools import read_checkpoint_metadata


def test_read_checkpoint_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_data = {
            "step": 100, "time": 1.0,
            "config": {"model": "GIPEPS", "shape": [6, 6]},
            "series": {"step": [1, 2], "energy_mean": [-0.5, -0.6]},
        }
        (Path(tmpdir) / "latest.json").write_text(json.dumps(json_data))
        meta = read_checkpoint_metadata(tmpdir)
        assert meta["step"] == 100
        assert meta["config"]["model"] == "GIPEPS"
```

Note: `smoke_test` is harder to unit test (requires full driver construction). Test it manually with an existing example script. The implementation should:
1. Run the script as a subprocess with `cwd` set to the script's parent directory
2. Override params: `--n-steps 2 --n-samples 32 --n-chains 4 --bond-dim 2 --log-every 1 --save-every 2`
3. Check exit code
4. Clean up the generated `data/` directory

- [ ] **Step 2: Run tests — expect FAIL**
- [ ] **Step 3: Implement runner_tools.py**

```python
import json
import shutil
import subprocess
from pathlib import Path


def read_checkpoint_metadata(run_dir: str) -> dict:
    """Read parsed metadata from a runner checkpoint."""
    with open(Path(run_dir) / "latest.json") as f:
        return json.load(f)


def smoke_test(
    script_path: str,
    overrides: dict | None = None,
    chain_state: str | None = None,
) -> dict:
    """Run a script with tiny parameters and check it doesn't crash.

    For two-stage workflows, pass chain_state as the ground-state run_dir
    to use as --state for dynamics scripts.
    """
    script = Path(script_path)
    defaults = {
        "--n-steps": "2", "--n-samples": "32", "--n-chains": "4",
        "--bond-dim": "2", "--boundary-dim": "4",
        "--log-every": "1", "--save-every": "2",
    }
    if overrides:
        defaults.update(overrides)
    args = [item for pair in defaults.items() for item in pair]
    if chain_state:
        args.extend(["--state", chain_state])

    result = subprocess.run(
        ["uv", "run", "python", str(script)] + args,
        cwd=script.parent,
        capture_output=True, text=True, timeout=300,
    )
    # Find and clean up generated data directory
    # (scripts write to data/ relative paths)
    data_dir = script.parent / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:] if result.stdout else "",
        "stderr": result.stderr[-2000:] if result.stderr else "",
    }
```

- [ ] **Step 4: Run tests — expect PASS**
- [ ] **Step 5: Register in server.py, finalize all tool registrations**
- [ ] **Step 6: Commit**

---

### Task 7: MCP server config + dependency + integration test

**Files:**
- Modify: `.claude/settings.json` — add mcpServers entry
- Modify: `pyproject.toml` — add `mcp` to dev dependencies

- [ ] **Step 1: Add mcp dependency**

Run: `uv add --dev mcp`

- [ ] **Step 2: Add MCP server config to settings**

Add to `.claude/settings.json` (create if needed, merge with existing):

```json
{
  "mcpServers": {
    "vmc": {
      "command": "uv",
      "args": ["run", "--directory", "${projectRoot}", "python", "tools/vmc-mcp/server.py"]
    }
  }
}
```

- [ ] **Step 3: Test server starts**

Run: `uv run python tools/vmc-mcp/server.py &` and verify it doesn't crash immediately.

- [ ] **Step 4: Run all MCP tests**

Run: `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_mcp_*.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `JAX_PLATFORM_NAME=cpu uv run pytest -m "not slow" -v`

- [ ] **Step 6: Commit**

---

## Execution Notes

- Tasks 1-6 are mostly sequential (each builds on server.py scaffold from Task 2)
- Task 1 (EXPERIENCE.md) is independent and can be done first or in parallel
- The `mcp` SDK needs `uv add --dev mcp` before Task 2
- For visualization tests, `matplotlib` is already available (used by existing example scripts). `imageio` may need `uv add --dev imageio` for GIF generation.
- The `smoke_test` tool uses subprocess to run scripts — it does not import them. This avoids module-level side effects.
- Generated scripts must be placed in the `examples/` tree for `sys.path` imports to work correctly.
