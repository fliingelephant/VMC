"""Tests for examples/runner.py."""
from __future__ import annotations

from vmc import config  # noqa: F401

import json
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from runner import (  # noqa: E402
    DEFAULT_METRICS_CONFIG,
    add_common_args,
    load_checkpoint,
    resolve_solver,
    run,
    save_checkpoint,
)

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver  # noqa: E402
from vmc.operators import DiagonalOperator, LocalHamiltonian, OneSiteOperator  # noqa: E402
from vmc.peps import PEPS  # noqa: E402
from vmc.peps.common.strategy import Variational  # noqa: E402
from vmc.preconditioners import DirectSolve, SRPreconditioner, solve_cholesky  # noqa: E402


def _make_tiny_driver():
    shape = (2, 2)
    model = PEPS(
        rngs=nnx.Rngs(0),
        shape=shape,
        bond_dim=2,
        contraction_strategy=Variational(4),
        dtype=jnp.complex128,
    )
    sigma_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    zz_diag = jnp.array([1, -1, -1, 1], dtype=jnp.complex128)
    hamiltonian = LocalHamiltonian(
        shape=shape,
        terms=(
            OneSiteOperator(0, 0, sigma_x),
            DiagonalOperator(((0, 0), (0, 1)), zz_diag),
        ),
    )
    return TDVPDriver(
        model,
        hamiltonian,
        preconditioner=SRPreconditioner(
            diag_shift=1e-2,
            strategy=DirectSolve(solver=solve_cholesky),
            metrics_config=DEFAULT_METRICS_CONFIG,
        ),
        dt=0.01,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(42),
        n_samples=64,
        n_chains=8,
    )


def test_resolve_solver():
    assert resolve_solver("cholesky") is solve_cholesky


def test_add_common_args():
    import argparse

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(
        ["--n-samples", "2048", "--solver", "svd", "--resume"]
    )
    assert args.n_samples == 2048
    assert args.solver == "svd"
    assert args.resume is True
    assert args.dt == 0.01
    assert args.diag_shift == 1e-4
    assert args.seed == 42


def test_checkpoint_round_trip():
    driver = _make_tiny_driver()
    driver.run(driver.dt)
    driver.run(driver.dt)
    step_before = driver.step_count
    time_before = driver.t
    tensors_before = jax.tree.map(lambda x: x.copy(), driver._tensors)

    with tempfile.TemporaryDirectory() as tmpdir:
        series = {"step": [1, 2], "energy_mean": [-0.5, -0.6]}
        save_checkpoint(tmpdir, driver, step_before, series=series, extra="test")
        assert (Path(tmpdir) / "latest").exists()
        assert (Path(tmpdir) / "latest.json").exists()

        driver2 = _make_tiny_driver()
        metadata = load_checkpoint(tmpdir, driver2)
        assert driver2.step_count == step_before
        assert driver2.t == time_before
        assert metadata["series"]["energy_mean"] == [-0.5, -0.6]
        assert metadata["extra"] == "test"
        for row in driver2._tensors:
            for col in driver2._tensors[row]:
                np.testing.assert_array_equal(
                    np.asarray(driver2._tensors[row][col]),
                    np.asarray(tensors_before[row][col]),
                )


def test_run_fresh():
    driver = _make_tiny_driver()
    with tempfile.TemporaryDirectory() as tmpdir:
        run(driver, n_steps=5, run_dir=tmpdir, log_every=1, save_every=5)
        assert (Path(tmpdir) / "latest").exists()
        with open(Path(tmpdir) / "latest.json") as f:
            data = json.load(f)
        assert data["step"] == 5
        assert len(data["series"]["step"]) == 5
        assert len(data["series"]["energy_mean"]) == 5


def test_run_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        driver = _make_tiny_driver()
        run(driver, n_steps=3, run_dir=tmpdir, log_every=1, save_every=3)
        with open(Path(tmpdir) / "latest.json") as f:
            data = json.load(f)
        assert data["step"] == 3

        driver2 = _make_tiny_driver()
        run(
            driver2,
            n_steps=2,
            run_dir=tmpdir,
            log_every=1,
            save_every=5,
            resume=True,
        )
        with open(Path(tmpdir) / "latest.json") as f:
            data = json.load(f)
        assert data["step"] == 5
        assert len(data["series"]["step"]) == 5
        assert data["series"]["step"] == [1, 2, 3, 4, 5]
