"""3x3 Schmitt-style TFIM quench with exact reference and PEPS TDVP.

This example follows the linear ramp from Schmitt et al.:

    H(t) = -J(t) sum_<ij> sigma^z_i sigma^z_j - g(t) sum_i sigma^x_i
    J(t) = 1 + t / tau_q
    g(t) = g_c * (1 - t / tau_q)

for t in [-tau_q, tau_q] on a 3x3 open lattice.

The exact reference starts from the fully +x-polarized product state.
The PEPS starts from the same exact state, embedded at bond dimension D and
gauge-randomized by seed-fixed unitary bond transformations.
"""

from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import argparse
import csv
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from flax import nnx

from vmc.drivers import RK4, RealTimeUnit, TDVPDriver
from vmc.operators import (
    AffineSchedule,
    DiagonalOperator,
    LocalHamiltonian,
    OneSiteOperator,
    TimeDependentHamiltonian,
)
from vmc.peps import NoTruncation, PEPS
from vmc.preconditioners import DirectSolve, SRPreconditioner, solve_svd


SHAPE = (3, 3)
GC = 3.04438


def _site_index(row: int, col: int, n_cols: int) -> int:
    return row * n_cols + col


def _nearest_neighbor_edges(shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    n_rows, n_cols = shape
    edges = []
    for row in range(n_rows):
        for col in range(n_cols):
            if col + 1 < n_cols:
                edges.append(
                    (_site_index(row, col, n_cols), _site_index(row, col + 1, n_cols))
                )
            if row + 1 < n_rows:
                edges.append(
                    (_site_index(row, col, n_cols), _site_index(row + 1, col, n_cols))
                )
    return tuple(edges)


def _axis_pairs(
    shape: tuple[int, int], distance: int
) -> tuple[tuple[int, int], ...]:
    n_rows, n_cols = shape
    pairs = []
    for row in range(n_rows):
        for col in range(n_cols - distance):
            pairs.append(
                (_site_index(row, col, n_cols), _site_index(row, col + distance, n_cols))
            )
    for row in range(n_rows - distance):
        for col in range(n_cols):
            pairs.append(
                (_site_index(row, col, n_cols), _site_index(row + distance, col, n_cols))
            )
    return tuple(pairs)


def _all_basis_states(n_sites: int) -> jax.Array:
    basis = jnp.arange(1 << n_sites, dtype=jnp.uint32)
    bit_positions = jnp.arange(n_sites, dtype=jnp.uint32)
    return ((basis[:, None] >> bit_positions[None, :]) & 1).astype(jnp.int32)


def _haar_unitary(key: jax.Array, dim: int) -> jax.Array:
    real_key, imag_key = jax.random.split(key)
    z = (
        jax.random.normal(real_key, (dim, dim), dtype=jnp.float64)
        + 1j * jax.random.normal(imag_key, (dim, dim), dtype=jnp.float64)
    )
    q, r = jnp.linalg.qr(z)
    phases = jnp.diag(r)
    phases = phases / jnp.where(jnp.abs(phases) > 0, jnp.abs(phases), 1.0)
    return q * phases


def _apply_leg_matrix(
    tensor: jax.Array,
    axis: int,
    matrix: jax.Array,
    *,
    left_multiply: bool,
) -> jax.Array:
    if left_multiply:
        return jnp.moveaxis(
            jnp.tensordot(matrix, tensor, axes=(1, axis)),
            0,
            axis,
        )
    return jnp.moveaxis(
        jnp.tensordot(tensor, matrix, axes=(axis, 0)),
        -1,
        axis,
    )


def _build_plus_x_peps(shape: tuple[int, int], bond_dim: int, seed: int) -> PEPS:
    model = PEPS(
        rngs=nnx.Rngs(seed),
        shape=shape,
        bond_dim=bond_dim,
        phys_dim=2,
        contraction_strategy=NoTruncation(),
    )
    plus_x = jnp.asarray([1.0, 1.0], dtype=jnp.complex128) / jnp.sqrt(2.0)
    n_rows, n_cols = shape
    tensors = []
    for row in range(n_rows):
        tensor_row = []
        for col in range(n_cols):
            up, down, left, right = model.site_dims(
                row, col, n_rows, n_cols, bond_dim
            )
            tensor = jnp.zeros((2, up, down, left, right), dtype=jnp.complex128)
            tensor_row.append(tensor.at[:, 0, 0, 0, 0].set(plus_x))
        tensors.append(tensor_row)

    if bond_dim > 1:
        n_bonds = n_rows * (n_cols - 1) + (n_rows - 1) * n_cols
        keys = jax.random.split(jax.random.key(seed + 1), n_bonds)
        k = 0
        for row in range(n_rows):
            for col in range(n_cols - 1):
                unitary = _haar_unitary(keys[k], bond_dim)
                k += 1
                tensors[row][col] = _apply_leg_matrix(
                    tensors[row][col],
                    4,
                    unitary,
                    left_multiply=False,
                )
                tensors[row][col + 1] = _apply_leg_matrix(
                    tensors[row][col + 1],
                    3,
                    unitary.conj().T,
                    left_multiply=True,
                )
        for row in range(n_rows - 1):
            for col in range(n_cols):
                unitary = _haar_unitary(keys[k], bond_dim)
                k += 1
                tensors[row][col] = _apply_leg_matrix(
                    tensors[row][col],
                    2,
                    unitary,
                    left_multiply=False,
                )
                tensors[row + 1][col] = _apply_leg_matrix(
                    tensors[row + 1][col],
                    1,
                    unitary.conj().T,
                    left_multiply=True,
                )

    for row in range(n_rows):
        for col in range(n_cols):
            model.tensors[row][col][...] = tensors[row][col]
    return model


def _build_exact_hamiltonian_parts(
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    n_sites = shape[0] * shape[1]
    dim = 1 << n_sites
    basis = np.arange(dim, dtype=np.uint32)
    h_zz_diag = np.zeros((dim,), dtype=np.float64)
    for left, right in _nearest_neighbor_edges(shape):
        z_left = 1.0 - 2.0 * ((basis >> left) & 1).astype(np.float64)
        z_right = 1.0 - 2.0 * ((basis >> right) & 1).astype(np.float64)
        h_zz_diag -= z_left * z_right

    h_zz = np.diag(h_zz_diag.astype(np.complex128))
    h_x = np.zeros((dim, dim), dtype=np.complex128)
    indices = np.arange(dim, dtype=np.int64)
    for site in range(n_sites):
        h_x[indices, indices ^ (1 << site)] += -1.0
    return h_zz, h_x


def _build_time_dependent_hamiltonian(
    shape: tuple[int, int],
    tau_q: float,
) -> TimeDependentHamiltonian:
    sigma_x = -jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    sigma_zz = -jnp.asarray([1.0, -1.0, -1.0, 1.0], dtype=jnp.complex128)
    terms = []
    offsets = []
    slopes = []
    n_rows, n_cols = shape
    for row in range(n_rows):
        for col in range(n_cols):
            terms.append(OneSiteOperator(row=row, col=col, op=sigma_x))
            offsets.append(GC)
            slopes.append(-GC / tau_q)
            if col + 1 < n_cols:
                terms.append(
                    DiagonalOperator(
                        sites=((row, col), (row, col + 1)),
                        diag=sigma_zz,
                    )
                )
                offsets.append(1.0)
                slopes.append(1.0 / tau_q)
            if row + 1 < n_rows:
                terms.append(
                    DiagonalOperator(
                        sites=((row, col), (row + 1, col)),
                        diag=sigma_zz,
                    )
                )
                offsets.append(1.0)
                slopes.append(1.0 / tau_q)
    return TimeDependentHamiltonian(
        base=LocalHamiltonian(shape=shape, terms=tuple(terms)),
        schedule=AffineSchedule(
            offset=jnp.asarray(offsets, dtype=jnp.float64),
            slope=jnp.asarray(slopes, dtype=jnp.float64),
        ),
    )


def _build_mx_observable(shape: tuple[int, int]) -> LocalHamiltonian:
    sigma_x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    coeff = 1.0 / (shape[0] * shape[1])
    return LocalHamiltonian(
        shape=shape,
        terms=tuple(
            OneSiteOperator(row=row, col=col, op=coeff * sigma_x)
            for row in range(shape[0])
            for col in range(shape[1])
        ),
    )


def _build_czz_observable(
    shape: tuple[int, int],
    distance: int,
) -> LocalHamiltonian:
    pairs = _axis_pairs(shape, distance)
    diag = jnp.asarray([1.0, -1.0, -1.0, 1.0], dtype=jnp.complex128) / len(pairs)
    return LocalHamiltonian(
        shape=shape,
        terms=tuple(
            DiagonalOperator(
                sites=(
                    divmod(left, shape[1]),
                    divmod(right, shape[1]),
                ),
                diag=diag,
            )
            for left, right in pairs
        ),
    )


def _j_of_t(t: float, tau_q: float) -> float:
    return 1.0 + t / tau_q


def _g_of_t(t: float, tau_q: float) -> float:
    return GC * (1.0 - t / tau_q)


def _hamiltonian_at(
    h_zz: np.ndarray,
    h_x: np.ndarray,
    t: float,
    tau_q: float,
) -> np.ndarray:
    return _j_of_t(t, tau_q) * h_zz + _g_of_t(t, tau_q) * h_x


def _ground_energy_per_site(
    h_zz: np.ndarray,
    h_x: np.ndarray,
    t: float,
    tau_q: float,
    n_sites: int,
    cache: dict[float, float],
) -> float:
    key = round(float(t), 12)
    if key not in cache:
        cache[key] = float(
            np.min(np.linalg.eigvalsh(_hamiltonian_at(h_zz, h_x, t, tau_q)).real)
            / n_sites
        )
    return cache[key]


def _czz_values(states: np.ndarray, pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    if not pairs:
        return np.zeros((states.shape[0],), dtype=np.float64)
    z = 1.0 - 2.0 * states.astype(np.float64)
    return sum(z[:, left] * z[:, right] for left, right in pairs) / len(pairs)


def _measure_state(
    psi: np.ndarray,
    t: float,
    tau_q: float,
    h_zz: np.ndarray,
    h_x: np.ndarray,
    n_sites: int,
    czz_r1: np.ndarray,
    czz_r2: np.ndarray,
    ground_cache: dict[float, float],
) -> dict[str, float]:
    h_t = _hamiltonian_at(h_zz, h_x, t, tau_q)
    probs = np.abs(psi) ** 2
    energy_per_site = float(np.vdot(psi, h_t @ psi).real / n_sites)
    ground_per_site = _ground_energy_per_site(
        h_zz,
        h_x,
        t,
        tau_q,
        n_sites,
        ground_cache,
    )
    return {
        "energy": energy_per_site,
        "ground_energy": ground_per_site,
        "q": energy_per_site - ground_per_site,
        "mx": float(-np.vdot(psi, h_x @ psi).real / n_sites),
        "czz_r1": float(np.dot(probs, czz_r1)),
        "czz_r2": float(np.dot(probs, czz_r2)),
    }


def _run_exact_trajectory(
    t0: float,
    steps: int,
    dt: float,
    exact_substeps: int,
    tau_q: float,
    psi0: np.ndarray,
    h_zz: np.ndarray,
    h_x: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    dt_sub = dt / exact_substeps
    psi = jnp.asarray(psi0)
    t = float(t0)
    times = [t]
    states = [np.asarray(psi)]
    for _ in range(steps):
        for substep in range(exact_substeps):
            t_mid = t + (substep + 0.5) * dt_sub
            h_mid = _hamiltonian_at(h_zz, h_x, t_mid, tau_q)
            propagator = jsp_linalg.expm(
                -1j * jnp.asarray(dt_sub, dtype=jnp.float64) * jnp.asarray(h_mid)
            )
            psi = propagator @ psi
        psi = psi / jnp.linalg.norm(psi)
        t += dt
        times.append(t)
        states.append(np.asarray(psi))
    return np.asarray(times), states


def _run_peps_trajectory(
    model: PEPS,
    t0: float,
    steps: int,
    dt: float,
    tau_q: float,
    n_samples: int,
    n_chains: int,
    diag_shift: float,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    n_sites = model.shape[0] * model.shape[1]
    driver = TDVPDriver(
        model,
        _build_time_dependent_hamiltonian(model.shape, tau_q),
        observables=(
            _build_mx_observable(model.shape),
            _build_czz_observable(model.shape, 1),
            _build_czz_observable(model.shape, 2),
        ),
        preconditioner=SRPreconditioner(
            diag_shift=diag_shift,
            strategy=DirectSolve(solver=solve_svd),
        ),
        dt=dt,
        t0=t0,
        time_unit=RealTimeUnit(),
        integrator=RK4(),
        sampler_key=jax.random.key(seed + 17),
        n_samples=n_samples,
        n_chains=n_chains,
        full_gradient=True,
    )
    times = [t0]
    measurements: list[dict[str, float]] = []
    for _ in range(steps):
        driver.run(dt)
        times.append(float(driver.t))
        measurements.append(
            {
                "energy": float(driver.energy.mean.real) / n_sites,
                "mx": float(driver.observable_stats[0].mean.real),
                "czz_r1": float(driver.observable_stats[1].mean.real),
                "czz_r2": float(driver.observable_stats[2].mean.real),
            }
        )
    return np.asarray(times), measurements


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 3x3 Schmitt-style TFIM quench with exact reference."
    )
    parser.add_argument("--tau-q", type=float, default=0.4)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--bond-dim", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--n-chains", type=int, default=64)
    parser.add_argument("--diag-shift", type=float, default=1e-8)
    parser.add_argument("--exact-substeps", type=int, default=8)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    if args.tau_q <= 0.0:
        raise ValueError("--tau-q must be positive.")
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive.")
    if args.bond_dim < 1:
        raise ValueError("--bond-dim must be at least 1.")
    if args.exact_substeps < 1:
        raise ValueError("--exact-substeps must be at least 1.")
    return args


def _plot_rows(rows: list[dict[str, float]], plot_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vmc")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    time = np.asarray([row["time"] for row in rows], dtype=np.float64)

    def save_plot(
        name: str,
        y_exact_key: str,
        y_peps_key: str,
        ylabel: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.plot(time, [row[y_exact_key] for row in rows], label="Exact", linewidth=2.0)
        ax.plot(time, [row[y_peps_key] for row in rows], label="PEPS", linewidth=2.0)
        ax.set_xlabel("t")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / name, dpi=200)
        plt.close(fig)

    save_plot("energy_vs_time.png", "energy_exact", "energy_peps", "Energy / site")
    save_plot("q_vs_time.png", "q_exact", "q_peps", "Q")
    save_plot("mx_vs_time.png", "mx_exact", "mx_peps", "m_x")
    save_plot("czz_r1_vs_time.png", "czz_r1_exact", "czz_r1_peps", "Czz(R=1)")
    save_plot("czz_r2_vs_time.png", "czz_r2_exact", "czz_r2_peps", "Czz(R=2)")


def main() -> None:
    args = _parse_args()
    total_time = 2.0 * args.tau_q
    steps = int(round(total_time / args.dt))
    if not np.isclose(total_time, steps * args.dt, atol=1e-12, rtol=1e-12):
        raise ValueError("2 * tau_q must be an integer multiple of dt.")

    n_sites = SHAPE[0] * SHAPE[1]
    states = _all_basis_states(n_sites)
    state_table = np.asarray(states)
    czz_r1 = _czz_values(state_table, _axis_pairs(SHAPE, 1))
    czz_r2 = _czz_values(state_table, _axis_pairs(SHAPE, 2))
    h_zz, h_x = _build_exact_hamiltonian_parts(SHAPE)

    psi0_exact = np.ones((1 << n_sites,), dtype=np.complex128) / np.sqrt(1 << n_sites)
    model = _build_plus_x_peps(SHAPE, args.bond_dim, args.seed)

    t0 = -args.tau_q
    exact_t, exact_states = _run_exact_trajectory(
        t0,
        steps,
        args.dt,
        args.exact_substeps,
        args.tau_q,
        psi0_exact,
        h_zz,
        h_x,
    )
    peps_t, peps_states = _run_peps_trajectory(
        model,
        t0,
        steps,
        args.dt,
        args.tau_q,
        args.n_samples,
        args.n_chains,
        args.diag_shift,
        args.seed,
    )
    if not np.allclose(exact_t, peps_t):
        raise ValueError("Time grids differ between exact and PEPS trajectories.")

    ground_cache: dict[float, float] = {}
    exact_measurements = [
        _measure_state(
            psi,
            float(t),
            args.tau_q,
            h_zz,
            h_x,
            n_sites,
            czz_r1,
            czz_r2,
            ground_cache,
        )
        for t, psi in zip(exact_t, exact_states, strict=True)
    ]
    peps_measurements = [exact_measurements[0]]
    peps_measurements.extend(peps_states)
    rows = []
    for step, (t, exact_obs, peps_obs) in enumerate(
        zip(exact_t, exact_measurements, peps_measurements, strict=True)
    ):
        rows.append(
            {
                "step": step,
                "time": float(t),
                "epsilon": float(t / args.tau_q),
                "J": _j_of_t(float(t), args.tau_q),
                "g": _g_of_t(float(t), args.tau_q),
                "energy_exact": exact_obs["energy"],
                "energy_peps": peps_obs["energy"],
                "ground_energy": exact_obs["ground_energy"],
                "q_exact": exact_obs["q"],
                "q_peps": peps_obs["energy"] - exact_obs["ground_energy"],
                "mx_exact": exact_obs["mx"],
                "mx_peps": peps_obs["mx"],
                "czz_r1_exact": exact_obs["czz_r1"],
                "czz_r1_peps": peps_obs["czz_r1"],
                "czz_r2_exact": exact_obs["czz_r2"],
                "czz_r2_peps": peps_obs["czz_r2"],
            }
        )

    headers = tuple(rows[0].keys())
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                str(row[key]) if key == "step" else f"{float(row[key]):.10f}"
                for key in headers
            )
        )
    print(
        "\nfinal_q_exact="
        f"{rows[-1]['q_exact']:.10f}, "
        f"final_q_peps={rows[-1]['q_peps']:.10f}"
    )

    if args.csv:
        output = Path(args.csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {output}")

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        _plot_rows(rows, plot_dir)
        print(f"saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
