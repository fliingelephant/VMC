"""Checkpoint helpers for the Z2 hard-core-boson example scripts."""
from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp


class Z2HardcoreBosonExamplesTest(unittest.TestCase):
    def test_latest_checkpoint_round_trip_restores_driver_state(self) -> None:
        common = importlib.import_module("examples.lgt.z2_hardcore_boson.common")

        driver = common.build_ground_state_driver(
            shape=(2, 2),
            h=1.0,
            g=0.2,
            J=0.1,
            m=0.0,
            particle_number=2,
            bond_dim_per_charge=1,
            boundary_dim=3,
            boundary_sweeps=2,
            seed=0,
            n_samples=8,
            n_chains=2,
            dt=0.01,
            diag_shift=1e-4,
        )
        driver.run(driver.dt)
        saved_tensors = {
            row: {col: jnp.array(tensor) for col, tensor in tensors.items()}
            for row, tensors in driver._tensors.items()
        }
        saved_config = jnp.array(driver._sampler_configuration)
        saved_key = jax.random.key_data(driver._sampler_key)
        saved_step = driver.step_count
        saved_t = driver.t

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            common.save_latest(
                run_dir,
                driver=driver,
                problem={"shape": [2, 2]},
                latest_metrics={"energy_mean": 0.0},
            )

            restored = common.build_ground_state_driver(
                shape=(2, 2),
                h=1.0,
                g=0.2,
                J=0.1,
                m=0.0,
                particle_number=2,
                bond_dim_per_charge=1,
                boundary_dim=3,
                boundary_sweeps=2,
                seed=123,
                n_samples=8,
                n_chains=2,
                dt=0.01,
                diag_shift=1e-4,
            )
            common.restore_latest(run_dir, restored)

            self.assertEqual(restored.step_count, saved_step)
            self.assertAlmostEqual(restored.t, saved_t, places=12)
            self.assertTrue(jnp.array_equal(restored._sampler_configuration, saved_config))
            self.assertTrue(
                jnp.array_equal(jax.random.key_data(restored._sampler_key), saved_key)
            )
            for row, tensors in saved_tensors.items():
                for col, tensor in tensors.items():
                    self.assertTrue(jnp.allclose(restored._tensors[row][col], tensor))

            latest = common.load_latest_json(run_dir)
            self.assertEqual(latest["progress"]["completed_steps"], saved_step)
            self.assertIn("energy_mean", latest["latest_metrics"])


if __name__ == "__main__":
    unittest.main()
