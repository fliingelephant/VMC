"""Exact Fig. 5(a) benchmark for Z2 vison propagation.

This script benchmarks the 2025 Wu-Liu 6x6 pure-Z2 vison propagation example
in the gauge-invariant reduced basis of dimension ``2^((L - 1)^2)``.

The basis is parameterized by plaquette bits on the ``(L - 1) x (L - 1)``
OBC plaquette grid. In this basis:

- the plaquette term flips one plaquette bit;
- the electric term is diagonal, obtained from link parities induced by the
  plaquette bits;
- the bottom-left boundary ``sigma_z`` vison insertion is a sign on the
  bottom-left plaquette bit.

The script can also overlay the exact traces with the PEPS output from
``z2_vison_propagation.py`` to reproduce Fig. 5(a).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, expm_multiply


FIG5A_PLAQUETTES = ((0, 0), (0, 1), (2, 2))

DEFAULT_L = 6
DEFAULT_H = 1.0
DEFAULT_G = 0.1
DEFAULT_T = 18.0
DEFAULT_DT = 0.005
DEFAULT_TOL = 1e-8
DEFAULT_MAXITER = None
DEFAULT_NCV = None


@dataclass(frozen=True)
class RunConfig:
    """Exact benchmark configuration."""

    L: int = DEFAULT_L
    h: float = DEFAULT_H
    g: float = DEFAULT_G
    T: float = DEFAULT_T
    dt: float = DEFAULT_DT
    tol: float = DEFAULT_TOL
    maxiter: int | None = DEFAULT_MAXITER
    ncv: int | None = DEFAULT_NCV
    output: Path | None = None
    peps_json: Path | None = None
    plot_output: Path | None = None


def _plaquette_index(L: int, row: int, col: int) -> int:
    return row * (L - 1) + col


def _selected_masks(L: int) -> tuple[int, ...]:
    return tuple(
        1 << _plaquette_index(L, L - 2 - y, x)
        for x, y in FIG5A_PLAQUETTES
    )


def _bottom_left_vison_mask(L: int) -> int:
    return 1 << _plaquette_index(L, L - 2, 0)


def _build_link_terms(L: int) -> tuple[tuple[int, ...], ...]:
    """Build the plaquette-bit support for each physical link parity."""
    n = L - 1
    terms = []
    for col in range(n):
        terms.append((_plaquette_index(L, 0, col),))
    for row in range(1, n):
        for col in range(n):
            terms.append(
                (
                    _plaquette_index(L, row - 1, col),
                    _plaquette_index(L, row, col),
                )
            )
    for col in range(n):
        terms.append((_plaquette_index(L, n - 1, col),))
    for row in range(n):
        terms.append((_plaquette_index(L, row, 0),))
    for col in range(1, n):
        for row in range(n):
            terms.append(
                (
                    _plaquette_index(L, row, col - 1),
                    _plaquette_index(L, row, col),
                )
            )
    for row in range(n):
        terms.append((_plaquette_index(L, row, n - 1),))
    return tuple(terms)


def _bit_values(basis: np.ndarray, bit: int) -> np.ndarray:
    return ((basis >> np.uint64(bit)) & np.uint64(1)).astype(np.uint8)


class PureZ2GaugeReducedHamiltonian:
    """Matrix-free pure-Z2 gauge Hamiltonian in the plaquette-bit basis."""

    def __init__(self, *, L: int, h: float, g: float) -> None:
        self.L = L
        self.h = float(h)
        self.g = float(g)
        self.n_plaquettes = (L - 1) ** 2
        self.dim = 1 << self.n_plaquettes
        self.basis = np.arange(self.dim, dtype=np.uint64)
        self.flip_masks = np.asarray(
            [np.uint64(1) << np.uint64(bit) for bit in range(self.n_plaquettes)],
            dtype=np.uint64,
        )
        self.link_terms = _build_link_terms(L)
        self.diagonal = 4.0 * self.g * self._build_link_count_diagonal()
        self.magnetic_coeff = -2.0 * self.h

    def _build_link_count_diagonal(self) -> np.ndarray:
        counts = np.zeros(self.dim, dtype=np.uint8)
        for term in self.link_terms:
            values = _bit_values(self.basis, term[0])
            if len(term) == 2:
                values ^= _bit_values(self.basis, term[1])
            counts += values
        return counts.astype(np.float64)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        vector = np.asarray(x).reshape(-1)
        out = self.diagonal * vector
        for mask in self.flip_masks:
            out += self.magnetic_coeff * vector[self.basis ^ mask]
        return out

    def linear_operator(self, *, dtype: np.dtype) -> LinearOperator:
        return LinearOperator(
            shape=(self.dim, self.dim),
            dtype=dtype,
            matvec=self.matvec,
            rmatvec=self.matvec,
        )

    def apply_bottom_left_vison(self, psi: np.ndarray) -> np.ndarray:
        phase = 1.0 - 2.0 * _bit_values(self.basis, _plaquette_index(self.L, self.L - 2, 0))
        return psi * phase.astype(psi.dtype, copy=False)

    def selected_plaquette_values(self, psi: np.ndarray) -> list[float]:
        values = []
        for mask in _selected_masks(self.L):
            values.append(float(np.vdot(psi, psi[self.basis ^ np.uint64(mask)]).real))
        return values

    def energy(self, psi: np.ndarray) -> float:
        return float(np.vdot(psi, self.matvec(psi)).real)


def _default_output_path(cfg: RunConfig) -> Path:
    t_token = format(cfg.T, ".3f").replace(".", "p")
    return (
        Path(__file__).resolve().parent
        / f"z2_vison_propagation_exact_L{cfg.L}_T{t_token}.json"
    )


def _ground_state(
    hamiltonian: PureZ2GaugeReducedHamiltonian,
    cfg: RunConfig,
) -> tuple[float, np.ndarray]:
    operator = hamiltonian.linear_operator(dtype=np.float64)
    eigenvalues, eigenvectors = eigsh(
        operator,
        k=1,
        which="SA",
        tol=cfg.tol,
        maxiter=cfg.maxiter,
        ncv=cfg.ncv,
    )
    psi = np.asarray(eigenvectors[:, 0], dtype=np.float64)
    psi /= np.linalg.norm(psi)
    return float(eigenvalues[0]), psi


def _real_time_trajectory(
    hamiltonian: PureZ2GaugeReducedHamiltonian,
    psi0: np.ndarray,
    cfg: RunConfig,
) -> dict[str, list]:
    n_steps = int(round(cfg.T / cfg.dt))
    if abs(cfg.T - n_steps * cfg.dt) > 1e-12 * max(1.0, abs(cfg.T), abs(cfg.dt)):
        raise ValueError(f"T={cfg.T} must be an integer multiple of dt={cfg.dt}.")
    evolution = LinearOperator(
        shape=(hamiltonian.dim, hamiltonian.dim),
        dtype=np.complex128,
        matvec=lambda x: (-1.0j * cfg.dt) * hamiltonian.matvec(x),
        rmatvec=lambda x: (1.0j * cfg.dt) * hamiltonian.matvec(x),
    )

    psi = np.asarray(psi0, dtype=np.complex128)
    reference_energy = hamiltonian.energy(psi)
    series = {
        "step": [0],
        "time": [0.0],
        "energy_mean": [reference_energy],
        "energy_drift_percent": [0.0],
        "selected_plaquette_mean": [hamiltonian.selected_plaquette_values(psi)],
    }
    print(
        "[exact] step t dt energy drift_percent p00 p01 p22",
        flush=True,
    )
    print(
        (
            f"[exact] {0:4d} {0.0:.6f} {cfg.dt:.6f} {reference_energy:.10f} "
            f"{0.0:.6f} {series['selected_plaquette_mean'][0][0]:.10f} "
            f"{series['selected_plaquette_mean'][0][1]:.10f} "
            f"{series['selected_plaquette_mean'][0][2]:.10f}"
        ),
        flush=True,
    )
    for step in range(1, n_steps + 1):
        psi = expm_multiply(
            evolution,
            psi,
            traceA=(-1.0j * cfg.dt) * np.sum(hamiltonian.diagonal),
        )
        psi /= np.linalg.norm(psi)
        energy = hamiltonian.energy(psi)
        drift = abs(energy - reference_energy) / abs(reference_energy) * 100.0
        selected = hamiltonian.selected_plaquette_values(psi)
        series["step"].append(step)
        series["time"].append(step * cfg.dt)
        series["energy_mean"].append(energy)
        series["energy_drift_percent"].append(drift)
        series["selected_plaquette_mean"].append(selected)
        print(
            (
                f"[exact] {step:4d} {step * cfg.dt:.6f} {cfg.dt:.6f} {energy:.10f} "
                f"{drift:.6f} {selected[0]:.10f} {selected[1]:.10f} {selected[2]:.10f}"
            ),
            flush=True,
        )
    return series


def run_exact_benchmark(cfg: RunConfig) -> dict:
    """Run the exact reduced-basis benchmark."""
    if cfg.L < 4:
        raise ValueError("Fig. 5(a) selected plaquettes require L >= 4.")
    if cfg.L != 6:
        print(
            f"Running L={cfg.L}. The strict 2025 Fig. 5(a) target is L=6.",
            flush=True,
        )
    hamiltonian = PureZ2GaugeReducedHamiltonian(L=cfg.L, h=cfg.h, g=cfg.g)
    ground_energy, psi_ground = _ground_state(hamiltonian, cfg)
    psi_vison = hamiltonian.apply_bottom_left_vison(psi_ground)
    psi_vison /= np.linalg.norm(psi_vison)
    series = _real_time_trajectory(hamiltonian, psi_vison, cfg)
    return {
        "problem": {
            "gauge_group": "Z2",
            "L": cfg.L,
            "shape": [cfg.L, cfg.L],
            "h": cfg.h,
            "g": cfg.g,
            "basis": "plaquette_bit_reduced",
            "dimension": hamiltonian.dim,
        },
        "selected_plaquettes": [list(site) for site in FIG5A_PLAQUETTES],
        "ground_state": {
            "energy": ground_energy,
            "solver": "eigsh",
            "tol": cfg.tol,
            "maxiter": cfg.maxiter,
            "ncv": cfg.ncv,
        },
        "vison": {
            "operator": "sigma_z",
            "orientation": "v",
            "link_row": cfg.L - 2,
            "link_col": 0,
            "plaquettes": [[cfg.L - 2, 0]],
        },
        "real_time": {
            "T": cfg.T,
            "dt": cfg.dt,
            **series,
        },
        "summary": {
            "final_energy_mean": series["energy_mean"][-1],
            "final_energy_drift_percent": series["energy_drift_percent"][-1],
        },
    }


def plot_fig5a(
    peps_result: dict,
    exact_result: dict,
    output_path: Path,
) -> None:
    """Overlay PEPS and exact selected plaquette traces as in Fig. 5(a)."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    peps_time = peps_result["real_time"]["time"]
    peps_values = peps_result["real_time"]["selected_plaquette_mean"]
    exact_time = exact_result["real_time"]["time"]
    exact_values = exact_result["real_time"]["selected_plaquette_mean"]
    selected = exact_result["selected_plaquettes"]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.6), sharex=True)
    for axis, index, plaquette in zip(axes, range(3), selected, strict=True):
        axis.plot(
            peps_time,
            [row[index] for row in peps_values],
            color="#1f77b4",
            linewidth=1.6,
            label="GI-PEPS D=6",
        )
        axis.plot(
            exact_time,
            [row[index] for row in exact_values],
            color="#ff7f0e",
            linewidth=1.4,
            label="exact",
        )
        axis.set_ylabel(rf"$\langle P_{{{plaquette[0]},{plaquette[1]}}}\rangle / 2$")
        axis.grid(alpha=0.25)
        if index == 0:
            axis.legend(frameon=False, loc="best")
    axes[0].set_title(r"(a) $6\times 6$ $Z_2$ gauge theory, $g=0.1$")
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Exact reduced-basis benchmark for Fig. 5(a) Z2 vison propagation.",
    )
    parser.add_argument("--L", type=int, default=DEFAULT_L)
    parser.add_argument("--h", type=float, default=DEFAULT_H)
    parser.add_argument("--g", type=float, default=DEFAULT_G)
    parser.add_argument("--T", type=float, default=DEFAULT_T)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--ncv", type=int, default=DEFAULT_NCV)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--peps-json", type=Path, default=None)
    parser.add_argument("--plot-output", type=Path, default=None)
    args = parser.parse_args()
    return RunConfig(
        L=args.L,
        h=args.h,
        g=args.g,
        T=args.T,
        dt=args.dt,
        tol=args.tol,
        maxiter=args.maxiter,
        ncv=args.ncv,
        output=args.output,
        peps_json=args.peps_json,
        plot_output=args.plot_output,
    )


def main() -> None:
    cfg = _parse_args()
    result = run_exact_benchmark(cfg)
    output_path = cfg.output or _default_output_path(cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {output_path}", flush=True)

    if cfg.peps_json is not None and cfg.plot_output is not None:
        peps_result = json.loads(cfg.peps_json.read_text())
        plot_fig5a(peps_result, result, cfg.plot_output)
        print(f"Saved {cfg.plot_output}", flush=True)


if __name__ == "__main__":
    main()
