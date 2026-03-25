"""Runner tools for smoke testing and checkpoint metadata reading."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

_SMOKE_DEFAULTS = {
    "--n-steps": "2",
    "--n-samples": "32",
    "--n-chains": "4",
    "--bond-dim": "2",
    "--boundary-dim": "4",
    "--log-every": "1",
    "--save-every": "2",
}


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

    Output is directed to a temporary directory via --output, so no user
    data is affected. For two-stage workflows, pass chain_state as the
    ground-state output directory to use as --state for dynamics scripts.

    Returns {"passed": bool, "returncode": int, "stdout": str, "stderr": str}.
    """
    script = Path(script_path).resolve()
    args_dict = dict(_SMOKE_DEFAULTS)
    if overrides:
        args_dict.update(overrides)

    with tempfile.TemporaryDirectory() as tmpdir:
        args_dict["--output"] = tmpdir
        args = [str(item) for pair in args_dict.items() for item in pair]
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
            passed = result.returncode == 0
            returncode = result.returncode
            stdout = result.stdout[-2000:] if result.stdout else ""
            stderr = result.stderr[-2000:] if result.stderr else ""
        except subprocess.TimeoutExpired:
            passed = False
            returncode = -1
            stdout = ""
            stderr = "Smoke test timed out after 300 seconds."

    return {
        "passed": passed,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }
