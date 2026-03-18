#!/usr/bin/env python3
"""Plot log-scale QGT spectra from gauge-probe JSON files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANK_CUTOFF = 1e-8


@dataclass(frozen=True)
class PlotCase:
    """Metadata and one QGT-spectrum record for plotting."""

    record: dict[str, object]
    L: int
    D: int
    scheme: str
    n_params: int
    n_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render log-scale QGT spectrum plots. Without an argument, scan all "
            "case folders below this script and plot each qgt_spectra.json "
            "separately."
        )
    )
    parser.add_argument(
        "json_path",
        type=Path,
        nargs="?",
        help="Optional qgt_spectra.json path to plot by itself.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path for plotting one JSON file.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution in dots per inch.",
    )
    return parser.parse_args()


def _ordered_plot_values(eigenvalues: list[float]) -> np.ndarray:
    """Sort all modes by |lambda| descending for a log-axis spectrum plot."""
    ordered = np.sort(np.abs(np.asarray(eigenvalues, dtype=float)))[::-1]
    return np.maximum(ordered, np.finfo(float).tiny)


def _default_output_path(json_path: Path) -> Path:
    return json_path.with_name("qgt_spectra.png")


def _scheme_mathtext(label: str) -> str:
    if label == "Exact":
        return r"\mathrm{Exact}"
    if label == "Variational(D_c=16)":
        return r"\mathrm{Variational}(D_c=16)"
    return rf"\mathrm{{{label}}}"


def _case_title(case: PlotCase) -> str:
    return (
        rf"${case.L}\times {case.L}\; D={case.D}\; {_scheme_mathtext(case.scheme)}\; "
        rf"\mathrm{{PEPS}}\; \mathrm{{QGT}}\; \mathrm{{spectrum}}\; "
        rf"(n_{{\mathrm{{params}}}}={case.n_params},\; "
        rf"n_{{\mathrm{{samples}}}}={case.n_samples})$"
    )


def _n_gv(case: PlotCase) -> int:
    return 2 * case.L * (case.L - 1) * case.D**2 - (case.L - 1) ** 2


def _projected_dimension(case: PlotCase) -> int:
    return case.n_params - _n_gv(case) - 1


def _separator_rank(eigenvalues: np.ndarray) -> int:
    return int(np.count_nonzero(eigenvalues > RANK_CUTOFF))


def _draw_case(ax: plt.Axes, case: PlotCase, *, show_legend: bool) -> None:
    qgt = _ordered_plot_values(case.record["qgt_eigenvalues"])
    projected = _ordered_plot_values(case.record["projected_qgt_eigenvalues"])
    separator_rank = _separator_rank(qgt)

    x_qgt = np.arange(1, qgt.size + 1)
    x_projected = np.arange(1, projected.size + 1)
    separator = separator_rank + 0.5
    markevery = max(1, projected.size // 90)

    ax.semilogy(
        x_qgt,
        qgt,
        color="#7b2cbf",
        linewidth=1.8,
        alpha=0.95,
        label="QGT",
        zorder=2,
    )
    ax.semilogy(
        x_projected,
        projected,
        color="#2a9d8f",
        linewidth=1.1,
        linestyle="--",
        alpha=0.95,
        marker="o",
        markersize=2.8,
        markevery=markevery,
        markerfacecolor="white",
        markeredgewidth=0.8,
        label="QGT (gauge removed)",
        zorder=3,
    )
    ax.axvline(separator, color="#e76f51", linewidth=1.0, alpha=0.9)
    ax.set_xlim(1, qgt.size)
    ax.set_xlabel(r"eigenmodes (sorted by $|\lambda|$, descending)")
    ax.set_ylabel(r"$|\lambda|$")
    ax.set_title(_case_title(case))
    ax.grid(True, which="both", alpha=0.22)
    if show_legend:
        ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.02,
        0.055,
        (
            rf"$n_{{\mathrm{{gv}}}} = 2L(L-1)D^2 - (L-1)^2 = {_n_gv(case)}$"
            "\n"
            rf"$n_{{\mathrm{{params}}}} - n_{{\mathrm{{gv}}}} - 1 = {_projected_dimension(case)}$"
        ),
        transform=ax.transAxes,
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "0.75",
        },
    )
    ax.annotate(
        f"{separator_rank}",
        xy=(separator, 0),
        xycoords=ax.get_xaxis_transform(),
        xytext=(0, -26),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=10,
        color="#b22222",
        annotation_clip=False,
    )


def save_plot(case: PlotCase, output_path: Path, *, dpi: int = 300) -> Path:
    """Render one figure for one case."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    _draw_case(ax, case, show_legend=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _load_case(json_path: Path) -> PlotCase:
    records = json.loads(json_path.read_text())
    if not records:
        raise ValueError(f"No records found in {json_path}")
    record = records[0]
    return PlotCase(
        record=record,
        L=int(record["shape"][0]),
        D=int(record["bond_dim"]),
        scheme=str(record["scheme"]),
        n_params=int(record["n_params"]),
        n_samples=int(record["n_samples"]),
    )


def save_json_plot(
    json_path: Path,
    *,
    output_path: Path | None = None,
    dpi: int = 300,
) -> Path:
    """Render one spectrum plot for one qgt_spectra.json file."""
    return save_plot(
        _load_case(json_path),
        output_path or _default_output_path(json_path),
        dpi=dpi,
    )


def save_all_plots(root: Path, *, dpi: int = 300) -> tuple[Path, ...]:
    """Render one spectrum plot per case folder below root."""
    return tuple(
        save_json_plot(json_path, dpi=dpi)
        for json_path in sorted(root.glob("*/qgt_spectra.json"))
    )


def main() -> None:
    args = parse_args()
    if args.json_path is not None:
        print(save_json_plot(args.json_path, output_path=args.output, dpi=args.dpi))
        return
    if args.output is not None:
        raise ValueError("--output requires a json_path")
    for output_path in save_all_plots(Path(__file__).resolve().parent, dpi=args.dpi):
        print(output_path)


if __name__ == "__main__":
    main()
