"""Runner tools for smoke testing and checkpoint metadata reading."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import orbax.checkpoint as ocp


def _load_jsonl(path: Path) -> list[dict]:
    """Load all lines from a JSONL file."""
    return [
        json.loads(line)
        for line in path.read_text().strip().split("\n")
        if line.strip()
    ]


def _jsonl_to_columnar(rows: list[dict]) -> dict[str, list]:
    """Convert JSONL rows to columnar series dict."""
    series: dict[str, list] = {}
    for row in rows:
        for key, value in row.items():
            series.setdefault(key, []).append(value)
    return series


def read_checkpoint_metadata(run_dir: str) -> dict:
    """Read checkpoint metadata and series from a run directory."""
    run_dir_path = Path(run_dir)
    mgr = ocp.CheckpointManager(
        run_dir_path,
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    meta = mgr.metadata()
    config = dict(meta.custom_metadata) if hasattr(meta, "custom_metadata") else {}
    metrics_path = run_dir_path / "metrics.jsonl"
    series = (
        _jsonl_to_columnar(_load_jsonl(metrics_path)) if metrics_path.exists() else {}
    )
    return {
        "step": mgr.latest_step(),
        "config": config,
        "series": series,
    }


def smoke_test(
    script_path: str,
    overrides: dict | None = None,
    chain_state: str | None = None,
) -> dict:
    """Run a script with tiny parameters and check it doesn't crash.

    ``--output`` is always set to a temporary directory to prevent
    smoke tests from overwriting real experiment data. Scripts that
    do not accept ``--output`` will fail with an argparse error.

    Returns {"passed": bool, "returncode": int, "stdout": str, "stderr": str}.
    """
    script = Path(script_path).resolve()
    args_dict = dict(overrides) if overrides else {}

    with tempfile.TemporaryDirectory() as tmpdir:
        args_dict["--output"] = tmpdir
        args = []
        for key, value in args_dict.items():
            if isinstance(value, bool):
                if value:
                    args.append(str(key))
            else:
                args.extend([str(key), str(value)])
        if chain_state:
            args.extend(["--state", str(chain_state)])

        try:
            result = subprocess.run(
                ["uv", "run", "python", str(script)] + args,
                cwd=script.parent,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Timeout after 300s",
            }

        return {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }
