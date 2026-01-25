"""Imaginary-time convergence checks for solvable models."""
from __future__ import annotations

import functools
import logging
import unittest

import numpy as np

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
import netket as nk
from flax import nnx

from vmc.core import _value, _value_and_grad
from vmc.drivers import DynamicsDriver, ImaginaryTimeUnit
from vmc.models.mps import MPS
from vmc.models.peps import NoTruncation, PEPS
from vmc.utils.vmc_utils import local_estimate

logger = logging.getLogger(__name__)


def _kron_all(mats: list[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out


def _exact_sampler_with_gradients(
    states: jax.Array,
    model,
    *,
    n_samples: int,
    key: jax.Array,
    initial_configuration: jax.Array | None = None,
):
    del (initial_configuration, n_samples)
    amps_full = _value(model, states)
    mask = jnp.abs(amps_full) > 1e-12
    samples = states[mask]
    amps, grads, _ = _value_and_grad(
        model, samples, full_gradient=True
    )
    o = grads / amps[:, None]
    return samples, o, None, key, None


class ExactSRPreconditioner:
    """Exact SR update using full basis weights (no MC noise)."""

    def __init__(self, diag_shift: float):
        self.diag_shift = diag_shift

    def apply(
        self,
        model,
        samples: jax.Array,
        o: jax.Array,
        p: jax.Array | None,
        local_energies: jax.Array,
        *,
        step: int | None = None,
        grad_factor: complex = 1.0,
        stage: int = 0,
    ):
        del (p, step, stage)
        amps = _value(model, samples)
        weights = jnp.abs(amps) ** 2
        weights = weights / jnp.sum(weights)

        o_mean = jnp.sum(weights[:, None] * o, axis=0)
        weighted_o = o * jnp.sqrt(weights[:, None])
        s_mat = weighted_o.conj().T @ weighted_o - jnp.outer(
            o_mean.conj(), o_mean
        )
        e_mean = jnp.sum(weights * local_energies)
        forces = o.conj().T @ (weights * (local_energies - e_mean))

        mat = s_mat + self.diag_shift * jnp.eye(
            s_mat.shape[0], dtype=s_mat.dtype
        )
        update_flat = jnp.linalg.solve(mat, grad_factor * forces)

        params = jax.tree_util.tree_map(jnp.asarray, model.tensors)
        _, unravel = jax.flatten_util.ravel_pytree(params)
        updates = unravel(update_flat)
        return jax.tree_util.tree_map(
            lambda u, t: u.astype(t.dtype), updates, params
        )


def _exact_energy_from_samples(
    model,
    states: jax.Array,
    operator: nk.operator.AbstractOperator,
) -> jax.Array:
    amps = _value(model, states)
    probs = jnp.abs(amps) ** 2
    mask = probs > 1e-12
    probs = probs[mask]
    probs = probs / jnp.sum(probs)
    local = local_estimate(model, states[mask], operator, amps[mask])
    return jnp.sum(probs * local)


def _tfim_energy_formula(n_sites: int, *, J: float, h: float) -> float:
    ks = (2.0 * np.pi / n_sites) * (np.arange(n_sites) + 0.5)
    eps = 2.0 * np.sqrt((J - h * np.cos(ks)) ** 2 + (h * np.sin(ks)) ** 2)
    return float(-0.5 * np.sum(eps))


def _xy_energy_formula(
    n_sites: int, *, J: float, gamma: float, h: float
) -> float:
    ks = (2.0 * np.pi / n_sites) * (np.arange(n_sites) + 0.5)
    eps = 2.0 * np.sqrt(
        (h - J * np.cos(ks)) ** 2 + (J * gamma * np.sin(ks)) ** 2
    )
    return float(-0.5 * np.sum(eps))


def _j1j2_graph(n_sites: int) -> nk.graph.Graph:
    edges = set()
    for i in range(n_sites):
        for offset, color in ((1, 0), (2, 1)):
            j = (i + offset) % n_sites
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b, color))
    return nk.graph.Graph(list(edges))


def _square_j1j2_graph(shape: tuple[int, int]) -> nk.graph.Graph:
    n_rows, n_cols = shape

    def idx(r: int, c: int) -> int:
        return r * n_cols + c

    edges = set()
    for r in range(n_rows):
        for c in range(n_cols):
            if r + 1 < n_rows:
                edges.add((idx(r, c), idx(r + 1, c), 0))
            if c + 1 < n_cols:
                edges.add((idx(r, c), idx(r, c + 1), 0))
            if r + 1 < n_rows and c + 1 < n_cols:
                edges.add((idx(r, c), idx(r + 1, c + 1), 1))
            if r + 1 < n_rows and c - 1 >= 0:
                edges.add((idx(r, c), idx(r + 1, c - 1), 1))
    return nk.graph.Graph(list(edges))


def _toric_code_hamiltonian(
    n_rows: int,
    n_cols: int,
) -> tuple[nk.operator.LocalOperator, nk.hilbert.Spin, int]:
    n_sites = 2 * n_rows * n_cols
    hi = nk.hilbert.Spin(s=0.5, N=n_sites)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    hamiltonian = nk.operator.LocalOperator(hi, dtype=jnp.complex128)

    def h_edge(x: int, y: int) -> int:
        return y * n_cols + x

    def v_edge(x: int, y: int) -> int:
        return n_rows * n_cols + y * n_cols + x

    for x in range(n_cols):
        for y in range(n_rows):
            star_edges = [
                h_edge(x, y),
                h_edge((x - 1) % n_cols, y),
                v_edge(x, y),
                v_edge(x, (y - 1) % n_rows),
            ]
            hamiltonian += -nk.operator.LocalOperator(
                hi, _kron_all([X, X, X, X]), star_edges
            )
            plaquette_edges = [
                h_edge(x, y),
                v_edge((x + 1) % n_cols, y),
                h_edge(x, (y + 1) % n_rows),
                v_edge(x, y),
            ]
            hamiltonian += -nk.operator.LocalOperator(
                hi, _kron_all([Z, Z, Z, Z]), plaquette_edges
            )
    n_terms = 2 * n_rows * n_cols
    return hamiltonian, hi, n_terms


def _cluster_2d_hamiltonian(
    shape: tuple[int, int],
) -> tuple[nk.operator.LocalOperator, nk.hilbert.Spin, int]:
    n_rows, n_cols = shape
    n_sites = n_rows * n_cols
    hi = nk.hilbert.Spin(s=0.5, N=n_sites)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    hamiltonian = nk.operator.LocalOperator(hi, dtype=jnp.complex128)

    def idx(r: int, c: int) -> int:
        return r * n_cols + c

    for r in range(n_rows):
        for c in range(n_cols):
            indices = [idx(r, c)]
            mats = [X]
            if r > 0:
                indices.append(idx(r - 1, c))
                mats.append(Z)
            if r < n_rows - 1:
                indices.append(idx(r + 1, c))
                mats.append(Z)
            if c > 0:
                indices.append(idx(r, c - 1))
                mats.append(Z)
            if c < n_cols - 1:
                indices.append(idx(r, c + 1))
                mats.append(Z)
            hamiltonian += -nk.operator.LocalOperator(
                hi, _kron_all(mats), indices
            )
    return hamiltonian, hi, n_sites


def _run_imag_time(
    model,
    hamiltonian,
    states: jax.Array,
    *,
    n_samples: int,
    n_steps: int,
    dt: float,
    diag_shift: float,
    key: jax.Array,
    gauge_config=None,
):
    sampler = functools.partial(
        _exact_sampler_with_gradients,
        states,
        n_samples=n_samples,
    )
    _ = gauge_config
    preconditioner = ExactSRPreconditioner(diag_shift=diag_shift)
    driver = DynamicsDriver(
        model,
        hamiltonian,
        sampler=sampler,
        preconditioner=preconditioner,
        dt=dt,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=key,
    )
    for _ in range(n_steps):
        driver.step()
    return driver.model


class PhysicsModelTest(unittest.TestCase):
    SEED = 0
    DIAG_SHIFT = 1e-3

    CLUSTER_SHAPE = (3, 4)
    CLUSTER_SITES = CLUSTER_SHAPE[0] * CLUSTER_SHAPE[1]
    CLUSTER_BOND_DIM = 3
    CLUSTER_SAMPLES = 2 ** CLUSTER_SITES
    CLUSTER_STEPS = 80
    CLUSTER_DT = 0.1
    CLUSTER_TOL = 1.5e-2

    MG_SITES = 12
    MG_BOND_DIM = 6
    MG_SAMPLES = 2 ** MG_SITES
    MG_STEPS = 120
    MG_DT = 0.1
    MG_TOL = 2.5e-2
    MG_PEPS_SHAPE = (2, MG_SITES // 2)
    MG_PEPS_BOND_DIM = MG_BOND_DIM

    TFIM_SITES = 12
    TFIM_BOND_DIM = 8
    TFIM_SAMPLES = 2 ** TFIM_SITES
    TFIM_STEPS = 100
    TFIM_DT = 0.1
    TFIM_FIELD = 0.7
    TFIM_TOL = 2e-2
    TFIM_PEPS_SHAPE = (2, TFIM_SITES // 2)
    TFIM_PEPS_BOND_DIM = TFIM_BOND_DIM

    XY_SITES = 12
    XY_BOND_DIM = 8
    XY_SAMPLES = 2 ** XY_SITES
    XY_STEPS = 100
    XY_DT = 0.1
    XY_FIELD = 0.5
    XY_ANISO = 0.4
    XY_TOL = 2.5e-2
    XY_PEPS_SHAPE = (2, XY_SITES // 2)
    XY_PEPS_BOND_DIM = XY_BOND_DIM

    HEISENBERG_SITES = 12
    HEISENBERG_BOND_DIM = 8
    HEISENBERG_SAMPLES = 2 ** HEISENBERG_SITES
    HEISENBERG_STEPS = 100
    HEISENBERG_DT = 0.1
    HEISENBERG_TOL = 2.5e-1
    HEISENBERG_PEPS_SHAPE = (2, HEISENBERG_SITES // 2)
    HEISENBERG_PEPS_BOND_DIM = HEISENBERG_BOND_DIM

    TORIC_ROWS = 2
    TORIC_COLS = 3
    TORIC_SHAPE = (TORIC_ROWS, 2 * TORIC_COLS)
    TORIC_BOND_DIM = 3
    TORIC_SAMPLES = 2 ** (2 * TORIC_ROWS * TORIC_COLS)
    TORIC_STEPS = 80
    TORIC_DT = 0.1
    TORIC_TOL = 2.5e-2

    SQUARE_HEISENBERG_SHAPE = (3, 4)
    SQUARE_HEISENBERG_SITES = (
        SQUARE_HEISENBERG_SHAPE[0] * SQUARE_HEISENBERG_SHAPE[1]
    )
    SQUARE_HEISENBERG_BOND_DIM = 3
    SQUARE_HEISENBERG_SAMPLES = 2 ** SQUARE_HEISENBERG_SITES
    SQUARE_HEISENBERG_STEPS = 120
    SQUARE_HEISENBERG_DT = 0.1
    SQUARE_HEISENBERG_TOL = HEISENBERG_TOL

    TRI_HEISENBERG_SHAPE = (3, 3)
    TRI_HEISENBERG_SITES = (
        TRI_HEISENBERG_SHAPE[0] * TRI_HEISENBERG_SHAPE[1]
    )
    TRI_HEISENBERG_BOND_DIM = 3
    TRI_HEISENBERG_SAMPLES = 2 ** TRI_HEISENBERG_SITES
    TRI_HEISENBERG_STEPS = 120
    TRI_HEISENBERG_DT = 0.1
    TRI_HEISENBERG_TOL = HEISENBERG_TOL

    SQUARE_J1J2_SHAPE = (3, 3)
    SQUARE_J1J2_SITES = SQUARE_J1J2_SHAPE[0] * SQUARE_J1J2_SHAPE[1]
    SQUARE_J1J2_BOND_DIM = 3
    SQUARE_J1J2_SAMPLES = 2 ** SQUARE_J1J2_SITES
    SQUARE_J1J2_STEPS = 120
    SQUARE_J1J2_DT = 0.1
    SQUARE_J1J2_TOL = HEISENBERG_TOL

    J1 = 1.0
    J2 = 0.5

    def test_cluster_state_energy(self) -> None:
        hamiltonian, hi, n_sites = _cluster_2d_hamiltonian(
            self.CLUSTER_SHAPE
        )
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        model = PEPS(
            rngs=nnx.Rngs(self.SEED),
            shape=self.CLUSTER_SHAPE,
            bond_dim=self.CLUSTER_BOND_DIM,
            contraction_strategy=NoTruncation(),
        )

        model = _run_imag_time(
            model,
            hamiltonian,
            states,
            n_samples=self.CLUSTER_SAMPLES,
            n_steps=self.CLUSTER_STEPS,
            dt=self.CLUSTER_DT,
            diag_shift=self.DIAG_SHIFT,
            key=jax.random.key(self.SEED),
        )
        energy = _exact_energy_from_samples(model, states, hamiltonian)
        expected = -float(n_sites)
        err = float(jnp.abs(energy - expected))
        logger.info("cluster_energy=%s expected=%s err=%s", energy, expected, err)
        self.assertLess(err, self.CLUSTER_TOL)

    def test_majumdar_ghosh_energy(self) -> None:
        n_sites = self.MG_SITES
        if n_sites % 2 != 0:
            self.skipTest("Majumdar-Ghosh requires even site count")
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        model = MPS(
            rngs=nnx.Rngs(self.SEED + 1),
            n_sites=n_sites,
            bond_dim=self.MG_BOND_DIM,
        )

        graph = _j1j2_graph(n_sites)
        hamiltonian = nk.operator.Heisenberg(
            hi,
            graph,
            J=(0.25 * self.J1, 0.25 * self.J2),
            dtype=jnp.complex128,
        )

        model = _run_imag_time(
            model,
            hamiltonian,
            states,
            n_samples=self.MG_SAMPLES,
            n_steps=self.MG_STEPS,
            dt=self.MG_DT,
            diag_shift=self.DIAG_SHIFT,
            key=jax.random.key(self.SEED + 1),
        )
        energy = _exact_energy_from_samples(model, states, hamiltonian)
        expected = -(3.0 / 8.0) * self.J1 * n_sites
        err = float(jnp.abs(energy - expected))
        logger.info("mg_energy=%s expected=%s err=%s", energy, expected, err)
        self.assertLess(err, self.MG_TOL)

    def test_majumdar_ghosh_energy_peps(self) -> None:
        n_sites = self.MG_SITES
        if n_sites % 2 != 0:
            self.skipTest("Majumdar-Ghosh requires even site count")
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        model = PEPS(
            rngs=nnx.Rngs(self.SEED + 10),
            shape=self.MG_PEPS_SHAPE,
            bond_dim=self.MG_PEPS_BOND_DIM,
            contraction_strategy=NoTruncation(),
        )

        graph = _j1j2_graph(n_sites)
        hamiltonian = nk.operator.Heisenberg(
            hi,
            graph,
            J=(0.25 * self.J1, 0.25 * self.J2),
            dtype=jnp.complex128,
        )

        model = _run_imag_time(
            model,
            hamiltonian,
            states,
            n_samples=self.MG_SAMPLES,
            n_steps=self.MG_STEPS,
            dt=self.MG_DT,
            diag_shift=self.DIAG_SHIFT,
            key=jax.random.key(self.SEED + 10),
        )
        energy = _exact_energy_from_samples(model, states, hamiltonian)
        expected = -(3.0 / 8.0) * self.J1 * n_sites
        err = float(jnp.abs(energy - expected))
        logger.info(
            "mg_energy_peps=%s expected=%s err=%s", energy, expected, err
        )
        self.assertLess(err, self.MG_TOL)

    def test_toric_code_energy(self) -> None:
        hamiltonian, hi, n_terms = _toric_code_hamiltonian(
            self.TORIC_ROWS, self.TORIC_COLS
        )
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        model = PEPS(
            rngs=nnx.Rngs(self.SEED + 2),
            shape=self.TORIC_SHAPE,
            bond_dim=self.TORIC_BOND_DIM,
            contraction_strategy=NoTruncation(),
        )

        model = _run_imag_time(
            model,
            hamiltonian,
            states,
            n_samples=self.TORIC_SAMPLES,
            n_steps=self.TORIC_STEPS,
            dt=self.TORIC_DT,
            diag_shift=self.DIAG_SHIFT,
            key=jax.random.key(self.SEED + 2),
        )
        energy = _exact_energy_from_samples(model, states, hamiltonian)
        expected = -float(n_terms)
        err = float(jnp.abs(energy - expected))
        logger.info("toric_energy=%s expected=%s err=%s", energy, expected, err)
        self.assertLess(err, self.TORIC_TOL)

    def test_tfim_free_fermion_energy(self) -> None:
        n_sites = self.TFIM_SITES
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        graph = nk.graph.Hypercube(length=n_sites, n_dim=1, pbc=True)
        hamiltonian = nk.operator.Ising(
            hi,
            graph=graph,
            h=self.TFIM_FIELD,
            J=-self.J1,
            dtype=jnp.complex128,
        )
        model = MPS(
            rngs=nnx.Rngs(self.SEED + 3),
            n_sites=n_sites,
            bond_dim=self.TFIM_BOND_DIM,
        )

        model = _run_imag_time(
            model,
            hamiltonian,
            states,
            n_samples=self.TFIM_SAMPLES,
            n_steps=self.TFIM_STEPS,
            dt=self.TFIM_DT,
            diag_shift=self.DIAG_SHIFT,
            key=jax.random.key(self.SEED + 3),
        )
        energy = _exact_energy_from_samples(model, states, hamiltonian)
        expected = _tfim_energy_formula(
            n_sites, J=self.J1, h=self.TFIM_FIELD
        )
        err = float(jnp.abs(energy - expected))
        logger.info("tfim_energy=%s expected=%s err=%s", energy, expected, err)
        self.assertLess(err, self.TFIM_TOL)

    def test_tfim_free_fermion_energy_peps(self) -> None:
        n_sites = self.TFIM_SITES
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        graph = nk.graph.Hypercube(length=n_sites, n_dim=1, pbc=True)
        hamiltonian = nk.operator.Ising(
            hi,
            graph=graph,
            h=self.TFIM_FIELD,
            J=-self.J1,
            dtype=jnp.complex128,
        )
        model = PEPS(
            rngs=nnx.Rngs(self.SEED + 11),
            shape=self.TFIM_PEPS_SHAPE,
            bond_dim=self.TFIM_PEPS_BOND_DIM,
            contraction_strategy=NoTruncation(),
        )

        model = _run_imag_time(
            model,
            hamiltonian,
            states,
            n_samples=self.TFIM_SAMPLES,
            n_steps=self.TFIM_STEPS,
            dt=self.TFIM_DT,
            diag_shift=self.DIAG_SHIFT,
            key=jax.random.key(self.SEED + 11),
        )
        energy = _exact_energy_from_samples(model, states, hamiltonian)
        expected = _tfim_energy_formula(
            n_sites, J=self.J1, h=self.TFIM_FIELD
        )
        err = float(jnp.abs(energy - expected))
        logger.info(
            "tfim_energy_peps=%s expected=%s err=%s", energy, expected, err
        )
        self.assertLess(err, self.TFIM_TOL)

    def test_xy_free_fermion_energy(self) -> None:
        n_sites = self.XY_SITES
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        hamiltonian = nk.operator.LocalOperator(hi, dtype=jnp.complex128)
        coupling_x = -0.5 * self.J1 * (1 + self.XY_ANISO)
        coupling_y = -0.5 * self.J1 * (1 - self.XY_ANISO)
        field = -self.XY_FIELD
        for i in range(n_sites):
            j = (i + 1) % n_sites
            hamiltonian += coupling_x * nk.operator.LocalOperator(
                hi, _kron_all([X, X]), [i, j]
            )
            hamiltonian += coupling_y * nk.operator.LocalOperator(
                hi, _kron_all([Y, Y]), [i, j]
            )
            hamiltonian += field * nk.operator.LocalOperator(hi, Z, [i])

        model = MPS(
            rngs=nnx.Rngs(self.SEED + 4),
            n_sites=n_sites,
            bond_dim=self.XY_BOND_DIM,
        )
        model = _run_imag_time(
            model,
            hamiltonian,
            states,
            n_samples=self.XY_SAMPLES,
            n_steps=self.XY_STEPS,
            dt=self.XY_DT,
            diag_shift=self.DIAG_SHIFT,
            key=jax.random.key(self.SEED + 4),
        )
        energy = _exact_energy_from_samples(model, states, hamiltonian)
        expected = _xy_energy_formula(
            n_sites, J=self.J1, gamma=self.XY_ANISO, h=self.XY_FIELD
        )
        err = float(jnp.abs(energy - expected))
        logger.info("xy_energy=%s expected=%s err=%s", energy, expected, err)
        self.assertLess(err, self.XY_TOL)

    def test_xy_free_fermion_energy_peps(self) -> None:
        n_sites = self.XY_SITES
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        hamiltonian = nk.operator.LocalOperator(hi, dtype=jnp.complex128)
        coupling_x = -0.5 * self.J1 * (1 + self.XY_ANISO)
        coupling_y = -0.5 * self.J1 * (1 - self.XY_ANISO)
        field = -self.XY_FIELD
        for i in range(n_sites):
            j = (i + 1) % n_sites
            hamiltonian += coupling_x * nk.operator.LocalOperator(
                hi, _kron_all([X, X]), [i, j]
            )
            hamiltonian += coupling_y * nk.operator.LocalOperator(
                hi, _kron_all([Y, Y]), [i, j]
            )
            hamiltonian += field * nk.operator.LocalOperator(hi, Z, [i])

        model = PEPS(
            rngs=nnx.Rngs(self.SEED + 12),
            shape=self.XY_PEPS_SHAPE,
            bond_dim=self.XY_PEPS_BOND_DIM,
            contraction_strategy=NoTruncation(),
        )
        model = _run_imag_time(
            model,
            hamiltonian,
            states,
            n_samples=self.XY_SAMPLES,
            n_steps=self.XY_STEPS,
            dt=self.XY_DT,
            diag_shift=self.DIAG_SHIFT,
            key=jax.random.key(self.SEED + 12),
        )
        energy = _exact_energy_from_samples(model, states, hamiltonian)
        expected = _xy_energy_formula(
            n_sites, J=self.J1, gamma=self.XY_ANISO, h=self.XY_FIELD
        )
        err = float(jnp.abs(energy - expected))
        logger.info(
            "xy_energy_peps=%s expected=%s err=%s", energy, expected, err
        )
        self.assertLess(err, self.XY_TOL)

    def test_heisenberg_energy_constant(self) -> None:
        n_sites = self.HEISENBERG_SITES
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        graph = nk.graph.Chain(length=n_sites, pbc=True)
        hamiltonian = nk.operator.Heisenberg(
            hi, graph, J=self.J1, dtype=jnp.complex128
        )
        model = MPS(
            rngs=nnx.Rngs(self.SEED + 5),
            n_sites=n_sites,
            bond_dim=self.HEISENBERG_BOND_DIM,
        )

        model = _run_imag_time(
            model,
            hamiltonian,
            states,
            n_samples=self.HEISENBERG_SAMPLES,
            n_steps=self.HEISENBERG_STEPS,
            dt=self.HEISENBERG_DT,
            diag_shift=self.DIAG_SHIFT,
            key=jax.random.key(self.SEED + 5),
        )
        energy = _exact_energy_from_samples(model, states, hamiltonian)
        w = nk.exact.lanczos_ed(hamiltonian, k=1)
        err = float(jnp.abs(energy - w[0]))
        logger.info("heisenberg_energy=%s expected=%s err=%s", energy, w[0], err)
        self.assertLess(err, self.HEISENBERG_TOL)

    def test_heisenberg_energy_constant_peps(self) -> None:
        n_sites = self.HEISENBERG_SITES
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        graph = nk.graph.Chain(length=n_sites, pbc=True)
        hamiltonian = nk.operator.Heisenberg(
            hi, graph, J=self.J1, dtype=jnp.complex128
        )
        model = PEPS(
            rngs=nnx.Rngs(self.SEED + 13),
            shape=self.HEISENBERG_PEPS_SHAPE,
            bond_dim=self.HEISENBERG_PEPS_BOND_DIM,
            contraction_strategy=NoTruncation(),
        )

        model = _run_imag_time(
            model,
            hamiltonian,
            states,
            n_samples=self.HEISENBERG_SAMPLES,
            n_steps=self.HEISENBERG_STEPS,
            dt=self.HEISENBERG_DT,
            diag_shift=self.DIAG_SHIFT,
            key=jax.random.key(self.SEED + 13),
        )
        energy = _exact_energy_from_samples(model, states, hamiltonian)
        w = nk.exact.lanczos_ed(hamiltonian, k=1)
        err = float(jnp.abs(energy - w[0]))
        logger.info(
            "heisenberg_energy_peps=%s expected=%s err=%s",
            energy,
            w[0],
            err,
        )
        self.assertLess(err, self.HEISENBERG_TOL)

    def test_square_heisenberg_energy_peps(self) -> None:
        shape = self.SQUARE_HEISENBERG_SHAPE
        n_sites = self.SQUARE_HEISENBERG_SITES
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        graph = nk.graph.Grid(extent=shape, pbc=False)
        hamiltonian = nk.operator.Heisenberg(
            hi, graph, J=self.J1, dtype=jnp.complex128
        )
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        model = PEPS(
            rngs=nnx.Rngs(self.SEED + 20),
            shape=shape,
            bond_dim=self.SQUARE_HEISENBERG_BOND_DIM,
            contraction_strategy=NoTruncation(),
        )

        model = _run_imag_time(
            model,
            hamiltonian,
            states,
            n_samples=self.SQUARE_HEISENBERG_SAMPLES,
            n_steps=self.SQUARE_HEISENBERG_STEPS,
            dt=self.SQUARE_HEISENBERG_DT,
            diag_shift=self.DIAG_SHIFT,
            key=jax.random.key(self.SEED + 20),
        )
        energy = _exact_energy_from_samples(model, states, hamiltonian)
        w = nk.exact.lanczos_ed(hamiltonian, k=1)
        err = float(jnp.abs(energy - w[0]))
        logger.info(
            "square_heisenberg_peps_energy=%s expected=%s err=%s",
            energy,
            w[0],
            err,
        )
        self.assertLess(err, self.SQUARE_HEISENBERG_TOL)

    def test_triangular_heisenberg_energy_peps(self) -> None:
        shape = self.TRI_HEISENBERG_SHAPE
        n_sites = self.TRI_HEISENBERG_SITES
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        graph = nk.graph.Triangular(extent=shape, pbc=False)
        hamiltonian = nk.operator.Heisenberg(
            hi, graph, J=self.J1, dtype=jnp.complex128
        )
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        model = PEPS(
            rngs=nnx.Rngs(self.SEED + 21),
            shape=shape,
            bond_dim=self.TRI_HEISENBERG_BOND_DIM,
            contraction_strategy=NoTruncation(),
        )

        model = _run_imag_time(
            model,
            hamiltonian,
            states,
            n_samples=self.TRI_HEISENBERG_SAMPLES,
            n_steps=self.TRI_HEISENBERG_STEPS,
            dt=self.TRI_HEISENBERG_DT,
            diag_shift=self.DIAG_SHIFT,
            key=jax.random.key(self.SEED + 21),
        )
        energy = _exact_energy_from_samples(model, states, hamiltonian)
        w = nk.exact.lanczos_ed(hamiltonian, k=1)
        err = float(jnp.abs(energy - w[0]))
        logger.info(
            "tri_heisenberg_peps_energy=%s expected=%s err=%s",
            energy,
            w[0],
            err,
        )
        self.assertLess(err, self.TRI_HEISENBERG_TOL)

    def test_square_j1j2_heisenberg_energy_peps(self) -> None:
        shape = self.SQUARE_J1J2_SHAPE
        n_sites = self.SQUARE_J1J2_SITES
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        graph = _square_j1j2_graph(shape)
        hamiltonian = nk.operator.Heisenberg(
            hi,
            graph,
            J=(0.25 * self.J1, 0.25 * self.J2),
            dtype=jnp.complex128,
        )
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        model = PEPS(
            rngs=nnx.Rngs(self.SEED + 22),
            shape=shape,
            bond_dim=self.SQUARE_J1J2_BOND_DIM,
            contraction_strategy=NoTruncation(),
        )

        model = _run_imag_time(
            model,
            hamiltonian,
            states,
            n_samples=self.SQUARE_J1J2_SAMPLES,
            n_steps=self.SQUARE_J1J2_STEPS,
            dt=self.SQUARE_J1J2_DT,
            diag_shift=self.DIAG_SHIFT,
            key=jax.random.key(self.SEED + 22),
        )
        energy = _exact_energy_from_samples(model, states, hamiltonian)
        w = nk.exact.lanczos_ed(hamiltonian, k=1)
        err = float(jnp.abs(energy - w[0]))
        logger.info(
            "square_j1j2_peps_energy=%s expected=%s err=%s",
            energy,
            w[0],
            err,
        )
        self.assertLess(err, self.SQUARE_J1J2_TOL)

if __name__ == "__main__":
    unittest.main()
