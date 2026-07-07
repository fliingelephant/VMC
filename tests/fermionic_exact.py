"""Brute-force graded references for the fermionic PEPS Phase-1 gate.

Independent of every kernel: amplitudes come from an explicit sum over all
virtual bond configurations with per-crossing signs of the declared planar
routing (physical wires run down their column, crossing the right-leg bond
of every site below them), and Hamiltonians are dense matrices built from
second-quantized operator algebra in the column-major Jordan-Wigner mode
order.  Everything is NumPy; clarity over speed.
"""

from __future__ import annotations

import itertools

import numpy as np

from vmc.peps.grading import Grading, leg_parities


def graded_amplitude(
    tensors: list[list[np.ndarray]],
    sample: np.ndarray,
    grading: Grading,
) -> complex:
    """Amplitude of one occupancy configuration by explicit crossing count.

    Sums over every virtual bond assignment; each term is the plain product
    of tensor entries times ``(-1)`` per crossing of an odd physical wire
    over an odd horizontal bond index.
    """
    n_rows, n_cols = len(tensors), len(tensors[0])
    spins = np.asarray(sample).reshape(n_rows, n_cols)
    phys_par = np.asarray(grading.phys_parity)[spins]
    crossings = (
        np.cumsum(np.vstack([np.zeros((1, n_cols), np.int64), phys_par[:-1]]), axis=0)
        % 2
    )
    v_bonds = [(r, c) for r in range(n_rows - 1) for c in range(n_cols)]
    h_bonds = [(r, c) for r in range(n_rows) for c in range(n_cols - 1)]
    h_par = {
        (r, c): leg_parities(tensors[r][c].shape[4], grading.n_even)
        for (r, c) in h_bonds
    }
    amp = 0.0j
    for v in itertools.product(*(range(tensors[r][c].shape[2]) for r, c in v_bonds)):
        vidx = dict(zip(v_bonds, v))
        for h in itertools.product(
            *(range(tensors[r][c].shape[4]) for r, c in h_bonds)
        ):
            hidx = dict(zip(h_bonds, h))
            term = 1.0 + 0.0j
            for r in range(n_rows):
                for c in range(n_cols):
                    term *= tensors[r][c][
                        spins[r, c],
                        vidx.get((r - 1, c), 0),
                        vidx.get((r, c), 0),
                        hidx.get((r, c - 1), 0),
                        hidx.get((r, c), 0),
                    ]
            exponent = sum(
                crossings[r, c] * h_par[r, c][hidx[r, c]] for (r, c) in h_bonds
            )
            amp += term * (-1.0) ** exponent
    return amp


def dense_hop_hamiltonian(
    shape: tuple[int, int],
    t: float = 1.0,
    *,
    fermionic: bool = True,
) -> np.ndarray:
    """Dense ``-t * sum_<ij> (c_i^dag c_j + h.c.)`` on the full occupancy basis.

    Basis states are occupancy bitstrings over sites in row-major sample
    order (basis index = big-endian bits).  For ``fermionic=True`` matrix
    elements carry the Jordan-Wigner string of the column-major mode order
    ``mode(r, c) = c * n_rows + r``; otherwise hardcore bosons.
    """
    n_rows, n_cols = shape
    n = n_rows * n_cols
    site_of_mode = [(m % n_rows) * n_cols + m // n_rows for m in range(n)]
    edges = [((r, c), (r, c + 1)) for r in range(n_rows) for c in range(n_cols - 1)] + [
        ((r, c), (r + 1, c)) for r in range(n_rows - 1) for c in range(n_cols)
    ]
    h = np.zeros((2**n, 2**n))
    for occ in itertools.product((0, 1), repeat=n):
        idx = int("".join(map(str, occ)), 2)
        for (r1, c1), (r2, c2) in edges:
            s1, s2 = r1 * n_cols + c1, r2 * n_cols + c2
            if occ[s1] == occ[s2]:
                continue
            a, b = sorted((c1 * n_rows + r1, c2 * n_rows + r2))
            string = sum(occ[site_of_mode[m]] for m in range(a + 1, b))
            hopped = list(occ)
            hopped[s1], hopped[s2] = occ[s2], occ[s1]
            sign = (-1.0) ** string if fermionic else 1.0
            h[int("".join(map(str, hopped)), 2), idx] += -t * sign
    return h


def sector_configs(shape: tuple[int, int], n_fermions: int) -> np.ndarray:
    """All occupancy configurations (flat, row-major) with fixed particle number."""
    n = shape[0] * shape[1]
    return np.array(
        [occ for occ in itertools.product((0, 1), repeat=n) if sum(occ) == n_fermions],
        dtype=np.int32,
    )


def basis_index(sample: np.ndarray) -> int:
    """Occupancy configuration -> dense basis index (big-endian bits)."""
    return int("".join(map(str, np.asarray(sample).reshape(-1))), 2)
