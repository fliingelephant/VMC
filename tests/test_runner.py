"""Tests for vmc.workflow."""
from __future__ import annotations

from vmc import config  # noqa: F401

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from vmc.workflow import (
    DEFAULT_METRICS_CONFIG,
    add_common_args,
    resolve_solver,
    run,
)

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver
from vmc.operators import DiagonalOperator, LocalHamiltonian, OneSiteOperator
from vmc.peps import PEPS
from vmc.peps.common.strategy import Variational
from vmc.preconditioners import DirectSolve, SRPreconditioner, solve_cholesky


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
    args = parser.parse_args(["--n-samples", "2048", "--solver", "svd", "--resume"])
    assert args.n_samples == 2048
    assert args.solver == "svd"
    assert args.resume is True


def test_run_fresh():
    driver = _make_tiny_driver()
    with tempfile.TemporaryDirectory() as tmpdir:
        run(driver, n_steps=5, run_dir=tmpdir, log_every=1, save_every=5)
        # CheckpointManager creates step-numbered directories
        mgr = ocp.CheckpointManager(tmpdir, options=ocp.CheckpointManagerOptions(read_only=True))
        assert mgr.latest_step() == 5


def test_run_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        driver = _make_tiny_driver()
        run(driver, n_steps=3, run_dir=tmpdir, log_every=1, save_every=3)

        driver2 = _make_tiny_driver()
        run(driver2, n_steps=2, run_dir=tmpdir, log_every=1, save_every=5, resume=True)
        assert driver2.step_count == 5

        mgr = ocp.CheckpointManager(tmpdir, options=ocp.CheckpointManagerOptions(read_only=True))
        assert mgr.latest_step() == 5


def test_run_with_jsonl_logger():
    driver = _make_tiny_driver()
    with tempfile.TemporaryDirectory() as tmpdir:
        run(driver, n_steps=3, run_dir=tmpdir, log_every=1, save_every=3)
        import json
        jsonl_path = Path(tmpdir) / "metrics.jsonl"
        lines = [json.loads(line) for line in jsonl_path.read_text().strip().split("\n")]
        assert len(lines) == 3
        assert "energy_mean" in lines[0]
        assert lines[0]["step"] == 1
        assert lines[2]["step"] == 3


def test_n_steps_and_T_final_mutual_exclusion():
    driver = _make_tiny_driver()
    import pytest
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(TypeError, match="n_steps or T_final"):
            run(driver, n_steps=5, T_final=1.0, run_dir=tmpdir)
        with pytest.raises(TypeError, match="n_steps or T_final"):
            run(driver, run_dir=tmpdir)
