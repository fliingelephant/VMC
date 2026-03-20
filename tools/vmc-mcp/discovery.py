"""Discovery module for VMC codebase introspection.

Provides functions that inspect the vmc package to enumerate models,
operators, contraction strategies, solvers, and example scripts.

All vmc imports are lazy (deferred to first call) so that this module
can be imported without triggering JAX initialization.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Lazy import cache
# ---------------------------------------------------------------------------

_cache: dict[str, object] = {}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _import_module(dotpath: str):
    """Lazily import a module and cache the result."""
    if dotpath not in _cache:
        import importlib
        _cache[dotpath] = importlib.import_module(dotpath)
    return _cache[dotpath]


# ---------------------------------------------------------------------------
# 1. list_models
# ---------------------------------------------------------------------------

def list_models() -> list[dict]:
    """Return metadata for every PEPS model family."""
    std = _import_module("vmc.peps.standard.model")
    gi = _import_module("vmc.peps.gi.model")
    blk = _import_module("vmc.peps.blockade.model")

    return [
        {
            "name": "PEPS",
            "module": "vmc.peps.standard.model",
            "description": std.PEPS.__doc__.strip(),
            "key_parameters": ["shape", "bond_dim", "phys_dim", "contraction_strategy"],
        },
        {
            "name": "GIPEPS",
            "module": "vmc.peps.gi.model",
            "description": gi.GIPEPS.__doc__.strip(),
            "key_parameters": ["config (GIPEPSConfig)", "contraction_strategy"],
        },
        {
            "name": "BlockadePEPS",
            "module": "vmc.peps.blockade.model",
            "description": blk.BlockadePEPS.__doc__.split("\n")[0].strip(),
            "key_parameters": ["config (BlockadePEPSConfig)", "contraction_strategy"],
        },
    ]


# ---------------------------------------------------------------------------
# 2. list_operators
# ---------------------------------------------------------------------------

def list_operators() -> list[dict]:
    """Return metadata for every operator term type."""
    lt = _import_module("vmc.operators.local_terms")
    gi_lt = _import_module("vmc.peps.gi.local_terms")

    standard_ops = [
        ("OneSiteOperator", lt, "Single-site operator term acting at (row, col)."),
        ("DiagonalOperator", lt, "Diagonal operator term on one or two sites."),
        (
            "HorizontalTwoSiteOperator",
            lt,
            "Two-site operator on horizontal neighbor (row, col) -> (row, col+1).",
        ),
        (
            "VerticalTwoSiteOperator",
            lt,
            "Two-site operator on vertical neighbor (row, col) -> (row+1, col).",
        ),
        (
            "PlaquetteOperator",
            lt,
            "Plaquette term on the square with top-left corner at (row, col).",
        ),
    ]

    gi_ops = [
        ("LinkDiagonalTerm", gi_lt, "Diagonal term on link degrees of freedom."),
        (
            "MatterMassTerm",
            gi_lt,
            "Matter mass term m_x n_x (diagonal on the matter site).",
        ),
        (
            "HorizontalMatterHoppingTerm",
            gi_lt,
            "Gauge-covariant hard-core hopping on a horizontal link.",
        ),
        (
            "VerticalMatterHoppingTerm",
            gi_lt,
            "Gauge-covariant hard-core hopping on a vertical link.",
        ),
        (
            "HorizontalHiggsLinkTerm",
            gi_lt,
            "Z2 Higgs link term sigma_x X sigma_x on a horizontal link.",
        ),
        (
            "VerticalHiggsLinkTerm",
            gi_lt,
            "Z2 Higgs link term sigma_x X sigma_x on a vertical link.",
        ),
    ]

    results = []
    for name, mod, desc in standard_ops:
        cls = getattr(mod, name)
        results.append({
            "name": name,
            "module": mod.__name__,
            "description": cls.__doc__.strip() if cls.__doc__ else desc,
        })
    for name, mod, desc in gi_ops:
        cls = getattr(mod, name)
        results.append({
            "name": name,
            "module": mod.__name__,
            "description": cls.__doc__.strip() if cls.__doc__ else desc,
        })
    return results


# ---------------------------------------------------------------------------
# 3. list_strategies
# ---------------------------------------------------------------------------

def list_strategies() -> list[dict]:
    """Return metadata for every contraction strategy."""
    strat = _import_module("vmc.peps.common.strategy")

    entries = []
    for name in ("NoTruncation", "ZipUp", "DensityMatrix", "Variational"):
        cls = getattr(strat, name)
        doc = cls.__doc__ or ""
        entries.append({
            "name": name,
            "module": "vmc.peps.common.strategy",
            "description": doc.split("\n")[0].strip(),
        })
    return entries


# ---------------------------------------------------------------------------
# 4. list_solvers
# ---------------------------------------------------------------------------

def list_solvers() -> list[dict]:
    """Return metadata for QGT solvers and QGT spaces."""
    solvers_mod = _import_module("vmc.qgt.solvers")
    qgt_mod = _import_module("vmc.qgt.qgt")

    results = []
    for name in ("solve_cholesky", "solve_svd", "solve_cg"):
        fn = getattr(solvers_mod, name)
        doc = fn.__doc__ or ""
        results.append({
            "name": name,
            "module": "vmc.qgt.solvers",
            "kind": "solver",
            "description": doc.strip(),
        })

    for name, desc in [
        ("ParameterSpace", "O^dag O formulation (n_params x n_params)."),
        ("SampleSpace", "OO^dag formulation (n_samples x n_samples)."),
    ]:
        cls = getattr(qgt_mod, name)
        results.append({
            "name": name,
            "module": "vmc.qgt.qgt",
            "kind": "space",
            "description": cls.__doc__.strip() if cls.__doc__ else desc,
        })
    return results


# ---------------------------------------------------------------------------
# 5. list_examples
# ---------------------------------------------------------------------------

_SKIP_BASENAMES = {"runner.py", "__init__.py", "physics.py", "common.py", "plot.py"}


def _extract_docstring(filepath: str) -> str:
    """Extract the first docstring line from a Python file using AST."""
    try:
        with open(filepath, "r") as f:
            tree = ast.parse(f.read(), filename=filepath)
        docstring = ast.get_docstring(tree)
        if docstring:
            return docstring.strip().split("\n")[0]
    except Exception:
        pass
    return ""


def _infer_model_family(filepath: str, description: str) -> str:
    """Infer which model family an example uses from its path and docstring."""
    path_lower = filepath.lower()
    desc_lower = description.lower()
    combined = path_lower + " " + desc_lower

    if any(kw in combined for kw in ("gi-peps", "gipeps", "gauge-invariant", "lgt", "z2_", "z3_", "vison")):
        return "gi"
    if any(kw in combined for kw in ("blockade", "rydberg")):
        return "blockade"
    return "standard"


def list_examples() -> list[dict]:
    """Return metadata for every example script under examples/."""
    examples_dir = _PROJECT_ROOT / "examples"
    results = []

    for root, _dirs, files in os.walk(examples_dir):
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            if fname in _SKIP_BASENAMES:
                continue
            full = os.path.join(root, fname)
            relpath = os.path.relpath(full, _PROJECT_ROOT)
            desc = _extract_docstring(full)
            family = _infer_model_family(relpath, desc)
            results.append({
                "path": relpath,
                "description": desc,
                "model_family": family,
            })

    return results


# ---------------------------------------------------------------------------
# 6. find_closest_example
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Split text into lowercase tokens on non-alphanumeric boundaries."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def find_closest_example(description: str) -> dict | None:
    """Find the example whose description and filename best match the query.

    Uses simple keyword overlap scoring with alphanumeric tokenization.
    """
    examples = list_examples()
    if not examples:
        return None

    query_words = _tokenize(description)

    best_score = -1
    best_example = None
    for ex in examples:
        target = ex["description"] + " " + ex["path"]
        target_words = _tokenize(target)
        score = len(query_words & target_words)
        if score > best_score:
            best_score = score
            best_example = ex

    return best_example
