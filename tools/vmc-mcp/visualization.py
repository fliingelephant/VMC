"""Visualization tools for PEPS-tVMC runner output."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_series(run_dir: str) -> dict[str, list]:
    """Load series data from metrics.jsonl into columnar format."""
    rows = [
        json.loads(line)
        for line in (Path(run_dir) / "metrics.jsonl").read_text().strip().split("\n")
        if line.strip()
    ]
    series: dict[str, list] = {}
    for row in rows:
        for key, value in row.items():
            series.setdefault(key, []).append(value)
    return series


def plot_convergence(run_dir: str, keys: list[str] | None = None) -> dict:
    """Plot observable convergence from runner output.

    Args:
        run_dir: Path to the runner output directory.
        keys: Series keys to plot. Default: energy_mean + all *_mean keys.

    Returns:
        {"path": str, "description": str}
    """
    series = _load_series(run_dir)
    x = series.get("time", series.get("step", []))
    x_label = "time" if "time" in series else "step"

    if keys is None:
        keys = ["energy_mean"] + [
            k for k in sorted(series)
            if k.endswith("_mean") and k != "energy_mean"
        ]

    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 3 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        y = series.get(key, [])
        err_key = key.replace("_mean", "_error")
        err = series.get(err_key)
        if err:
            ax.errorbar(x, y, yerr=err, linewidth=1.2, capsize=2, label=key)
        else:
            ax.plot(x, y, linewidth=1.2, label=key)
        ax.set_ylabel(key)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel(x_label)
    fig.tight_layout()

    out_path = str(Path(run_dir) / "convergence.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"path": out_path, "description": f"Convergence plot of {', '.join(keys)}"}


def _extract_grid(series: dict, step: int, prefix: str) -> tuple[list[list[float]], int, int]:
    """Extract a 2D grid from flat observable keys at a given step index.

    Handles both underscore-separated (P_0_0_mean) and concatenated (P_00_mean) naming.
    """
    # Find matching keys and extract row/col
    pattern_underscore = re.compile(rf"^{re.escape(prefix)}(\d+)_(\d+)_mean$")
    pattern_concat = re.compile(rf"^{re.escape(prefix)}(\d)(\d)_mean$")

    coords: dict[tuple[int, int], float] = {}
    for key, values in series.items():
        m = pattern_underscore.match(key) or pattern_concat.match(key)
        if m:
            r, c = int(m.group(1)), int(m.group(2))
            coords[(r, c)] = values[step]

    if not coords:
        return [], 0, 0

    max_r = max(r for r, _ in coords) + 1
    max_c = max(c for _, c in coords) + 1
    grid = [
        [coords.get((r, c), 0.0) for c in range(max_c)]
        for r in range(max_r)
    ]
    return grid, max_r, max_c


def plot_heatmap(run_dir: str, step: int, observable_prefix: str) -> dict:
    """Plot a 2D heatmap of spatial observables at a given step.

    Args:
        run_dir: Path to the runner output directory.
        step: Step index in the series (0-based).
        observable_prefix: Key prefix like "P_" to match P_0_0_mean etc.

    Returns:
        {"path": str, "description": str}
    """
    series = _load_series(run_dir)
    grid, n_rows, n_cols = _extract_grid(series, step, observable_prefix)

    if not grid:
        return {"path": "", "description": "No matching observables found."}

    fig, ax = plt.subplots(figsize=(max(4, n_cols * 0.8), max(4, n_rows * 0.8)))
    im = ax.imshow(grid, cmap="RdBu_r", origin="lower")
    fig.colorbar(im, ax=ax)

    time_val = series.get("time", series.get("step", [None]))[step]
    ax.set_title(f"{observable_prefix}* at step {step} (t={time_val})")
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.tight_layout()

    out_path = str(Path(run_dir) / f"heatmap_step{step}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"path": out_path, "description": f"Heatmap of {observable_prefix}* at step {step}"}


def animate(run_dir: str, observable_prefix: str, fps: int = 5) -> dict:
    """Create a GIF animation of spatial observables over time.

    Args:
        run_dir: Path to the runner output directory.
        observable_prefix: Key prefix like "P_" to match P_0_0_mean etc.
        fps: Frames per second for the GIF.

    Returns:
        {"path": str, "description": str}
    """
    import imageio.v3 as iio
    import io

    series = _load_series(run_dir)
    n_steps = len(series.get("step", []))
    if n_steps == 0:
        return {"path": "", "description": "No steps found in series."}

    frames = []
    for step_idx in range(n_steps):
        grid, n_rows, n_cols = _extract_grid(series, step_idx, observable_prefix)
        if not grid:
            continue

        fig, ax = plt.subplots(figsize=(max(4, n_cols * 0.8), max(4, n_rows * 0.8)))
        im = ax.imshow(grid, cmap="RdBu_r", origin="lower")
        fig.colorbar(im, ax=ax)
        time_val = series.get("time", series.get("step", [None]))[step_idx]
        ax.set_title(f"{observable_prefix}* t={time_val}")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        frames.append(iio.imread(buf))

    if not frames:
        return {"path": "", "description": "No frames generated."}

    out_path = str(Path(run_dir) / "animation.gif")
    iio.imwrite(out_path, frames, duration=1000 // fps, loop=0)
    return {"path": out_path, "description": f"Animation of {observable_prefix}* ({len(frames)} frames)"}
