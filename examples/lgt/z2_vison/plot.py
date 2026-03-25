"""Plot Fig. 5(a) — overlay GI-PEPS and exact selected plaquette traces.

Reads the runner JSON output from a real-time dynamics run and compares
against the upstream exact open-data trace from Wu & Liu (2025).
"""
from __future__ import annotations

import argparse
import io
import json
import ssl
import urllib.request
from pathlib import Path

import numpy as np

FIG5A_PLAQUETTES = ((0, 0), (0, 1), (2, 2))
FIG5A_OPEN_DATA_COLUMNS = (3, 4, 15)
EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_EXACT_OPEN_DATA_URL = (
    "https://raw.githubusercontent.com/yantaow/open_data/main/"
    "wu2025accurate/Fig5/6x6_vison/vison_6x6_g0.1_h1.0_dt0.1_exact.out"
)
DEFAULT_EXACT_OPEN_DATA_CACHE = (
    EXAMPLE_DIR / "vison_6x6_g0.1_h1.0_dt0.1_exact.out"
)


def _download_text(url: str) -> str:
    verify_paths = ssl.get_default_verify_paths()
    for cafile in (
        verify_paths.cafile,
        verify_paths.openssl_cafile,
        "/etc/ssl/cert.pem",
        "/opt/homebrew/etc/openssl@3/cert.pem",
    ):
        if cafile and Path(cafile).exists():
            context = ssl.create_default_context(cafile=cafile)
            with urllib.request.urlopen(url, context=context) as response:
                return response.read().decode("utf-8")
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def _load_or_download_text(path: Path, url: str) -> str:
    if path.exists():
        return path.read_text()
    text = _download_text(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"Cached {path}", flush=True)
    return text


def load_exact_open_data(
    *,
    cache_path: Path = DEFAULT_EXACT_OPEN_DATA_CACHE,
    url: str = DEFAULT_EXACT_OPEN_DATA_URL,
) -> dict:
    """Load the upstream exact open-data trace used for Fig. 5(a)."""
    data = np.loadtxt(io.StringIO(_load_or_download_text(cache_path, url)))
    selected = [
        [float(row[column]) for column in FIG5A_OPEN_DATA_COLUMNS]
        for row in data
    ]
    return {
        "time": data[:, 1].astype(float).tolist(),
        "selected_plaquette_mean": selected,
    }


def plot_fig5a(
    peps_json_path: Path,
    exact_result: dict,
    output_path: Path,
) -> None:
    """Overlay GI-PEPS and upstream exact selected plaquette traces."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    # Read JSONL metrics from run directory
    rows = [
        json.loads(line)
        for line in (peps_json_path / "metrics.jsonl").read_text().strip().split("\n")
        if line.strip()
    ]
    plaq_keys = [f"P_{r}{c}_mean" for r, c in FIG5A_PLAQUETTES]
    peps_time = [row["time"] for row in rows]
    peps_values = [[row[key] for key in plaq_keys] for row in rows]

    exact_time = exact_result["time"]
    exact_values = exact_result["selected_plaquette_mean"]

    from vmc.workflow import read_config
    config = read_config(peps_json_path).get("extra", {})
    L = config.get("L", "?")
    g = config.get("g", "?")

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.6), sharex=True)
    for axis, index, plaquette in zip(axes, range(3), FIG5A_PLAQUETTES, strict=True):
        axis.plot(
            peps_time,
            [row[index] for row in peps_values],
            color="#1f77b4",
            linewidth=1.6,
            label="GI-PEPS",
        )
        axis.plot(
            exact_time,
            [row[index] for row in exact_values],
            color="#ff7f0e",
            linewidth=1.4,
            label="exact open data",
        )
        axis.set_ylabel(rf"$\langle P_{{{plaquette[0]},{plaquette[1]}}}\rangle / 2$")
        axis.grid(alpha=0.25)
        if index == 0:
            axis.legend(frameon=False, loc="best")
    axes[0].set_title(
        rf"(a) ${L}\times {L}$ $Z_2$ gauge theory, $g={g}$"
    )
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Fig. 5(a) — GI-PEPS vs exact vison propagation.",
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to dynamics run directory")
    parser.add_argument("--exact-cache", type=Path,
                        default=DEFAULT_EXACT_OPEN_DATA_CACHE)
    parser.add_argument("--exact-url", default=DEFAULT_EXACT_OPEN_DATA_URL)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    exact_result = load_exact_open_data(
        cache_path=args.exact_cache, url=args.exact_url,
    )
    output_path = args.output or args.input.with_name(
        f"{args.input.stem}_fig5a.pdf"
    )
    plot_fig5a(args.input, exact_result, output_path)


if __name__ == "__main__":
    main()
