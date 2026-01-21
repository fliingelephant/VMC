"""Stochastic DynamicsDriver checks against NetKet TDVP."""
from __future__ import annotations

import functools
import logging
import os
import unittest

from VMC import config  # noqa: F401

os.environ.setdefault("NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION", "1")

import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental.driver as nkx_driver
import netket.experimental.dynamics as nkx_dynamics
from netket import stats as nkstats
from flax import nnx

from VMC.drivers import DynamicsDriver, ImaginaryTimeUnit, RealTimeUnit
from VMC.models.mps import MPS
from VMC.models.peps import PEPS, ZipUp
from VMC.preconditioners import SRPreconditioner
from VMC.samplers.sequential import sequential_sample, sequential_sample_with_gradients
from VMC.core.eval import _value_and_grad
from VMC.utils.utils import spin_to_occupancy
from VMC.utils.vmc_utils import local_estimate, model_params

logger = logging.getLogger(__name__)


def _make_apply_fun(model):
    """Create a NetKet-compatible apply function from a model."""
    from VMC.core import _value
    def apply_fun(variables, x, **kwargs):
        del kwargs
        # Temporarily swap in NetKet's params
        original = [jnp.asarray(t) for t in model.tensors]
        for target, val in zip(model.tensors, variables["params"]["tensors"]):
            if hasattr(target, "copy_from"):
                target.copy_from(val)
            else:
                target[...] = val
        samples = x if x.ndim == 2 else x[None, :]
        amps = _value(model, samples)
        log_amps = jnp.log(amps)
        # Restore original
        for target, val in zip(model.tensors, original):
            if hasattr(target, "copy_from"):
                target.copy_from(val)
            else:
                target[...] = val
        return log_amps if x.ndim == 2 else log_amps[0]
    return apply_fun


def _mps_apply_fun(variables, x, **kwargs):
    del kwargs
    tensors = variables["params"]["tensors"]
    samples = x if x.ndim == 2 else x[None, :]
    amps = MPS._batch_amplitudes(tensors, samples)
    log_amps = jnp.log(amps)
    return log_amps if x.ndim == 2 else log_amps[0]


class DynamicsDriverTest(unittest.TestCase):
    N_SITES = 12
    BOND_DIM = 2
    N_SAMPLES_REAL = 65536
    N_SAMPLES_IMAG = 16384
    N_STEPS_REAL = 100
    N_STEPS_IMAG = 100
    DT_REAL = 0.01
    DT_IMAG = 0.001
    DIAG_SHIFT = 1e-2
    SEED = 0
    BURN_IN = 20
    N_CHAINS = 8
    ENERGY_SIGMA_MULT = 5.0

    def _build_system(self):
        hi = nk.hilbert.Spin(s=1 / 2, N=self.N_SITES)
        hamiltonian = nk.operator.Heisenberg(
            hi, nk.graph.Chain(length=self.N_SITES), dtype=jnp.complex128
        )
        model = MPS(
            rngs=nnx.Rngs(self.SEED),
            n_sites=hi.size,
            bond_dim=self.BOND_DIM,
        )
        return hi, hamiltonian, model

    def _build_driver(
        self,
        model,
        hamiltonian,
        *,
        time_unit,
        dt: float,
        n_samples: int,
    ):
        sampler = functools.partial(
            sequential_sample_with_gradients,
            n_samples=n_samples,
            n_chains=self.N_CHAINS,
            burn_in=self.BURN_IN,
            full_gradient=True,
        )
        preconditioner = SRPreconditioner(diag_shift=self.DIAG_SHIFT)
        return DynamicsDriver(
            model,
            hamiltonian,
            sampler=sampler,
            preconditioner=preconditioner,
            dt=dt,
            time_unit=time_unit,
            sampler_key=jax.random.key(self.SEED),
        )

    def _build_netket_driver(
        self,
        hi,
        hamiltonian,
        model,
        *,
        propagation_type: str,
        ode_solver,
        n_samples: int,
    ):
        sampler = nk.sampler.MetropolisLocal(
            hi,
            n_chains=self.N_CHAINS,
            sweep_size=int(hi.size),
            reset_chains=False,
        )
        vstate = nk.vqs.MCState(
            sampler,
            model=None,
            n_samples=n_samples,
            n_discard_per_chain=self.BURN_IN,
            variables={"params": model_params(model)},
            apply_fun=_mps_apply_fun,
            sampler_seed=self.SEED,
        )
        qgt = functools.partial(
            nk.optimizer.qgt.QGTJacobianDense,
            holomorphic=True,
            diag_shift=self.DIAG_SHIFT,
        )
        driver = nkx_driver.TDVP(
            hamiltonian,
            vstate,
            ode_solver,
            propagation_type=propagation_type,
            qgt=qgt,
            linear_solver=nk.optimizer.solver.cholesky,
        )
        return driver, vstate

    def _energy_stats(self, model, hamiltonian, *, key, n_samples: int):
        samples, key = sequential_sample(
            model,
            n_samples=n_samples,
            n_chains=self.N_CHAINS,
            burn_in=self.BURN_IN,
            key=key,
            return_key=True,
        )
        local_energies = local_estimate(model, samples, hamiltonian)
        chain_length = n_samples // self.N_CHAINS
        local_energies = local_energies.reshape(chain_length, self.N_CHAINS).T
        return nkstats.statistics(local_energies), key

    def _log_stats(self, stats, *, source: str, label: str) -> None:
        logger.warning(
            "stats source=%s label=%s mean=%s error=%s variance=%s tau_corr=%s R_hat=%s tau_corr_max=%s",
            source,
            label,
            complex(stats.mean),
            float(stats.error_of_mean),
            float(stats.variance),
            float(stats.tau_corr),
            float(stats.R_hat),
            float(stats.tau_corr_max),
        )

    def _assert_energy_close(self, ours, netket, *, label: str):
        err = float(ours.error_of_mean) + float(netket.error_of_mean)
        diff = abs(complex(ours.mean) - complex(netket.mean))
        ratio = diff / (err + 1e-12)
        logger.warning(
            "energy_diff=%s combined_err=%s ratio=%s ours_mean=%s netket_mean=%s label=%s",
            diff,
            err,
            ratio,
            complex(ours.mean),
            complex(netket.mean),
            label,
        )
        self._log_stats(ours, source="ours", label=label)
        self._log_stats(netket, source="netket", label=label)
        self.assertLess(diff, self.ENERGY_SIGMA_MULT * (err + 1e-12))

    def _run_comparison(
        self,
        *,
        time_unit,
        propagation_type: str,
        ode_solver,
        dt: float,
        n_steps: int,
        n_samples: int,
    ):
        hi, hamiltonian, model = self._build_system()
        driver = self._build_driver(
            model, hamiltonian, time_unit=time_unit, dt=dt, n_samples=n_samples
        )
        nk_driver, vstate = self._build_netket_driver(
            hi,
            hamiltonian,
            model,
            propagation_type=propagation_type,
            ode_solver=ode_solver,
            n_samples=n_samples,
        )

        key = jax.random.key(self.SEED)
        stats_ours, key = self._energy_stats(
            driver.model, hamiltonian, key=key, n_samples=n_samples
        )
        stats_netket = vstate.expect(hamiltonian)
        self._assert_energy_close(stats_ours, stats_netket, label="t=0")

        for step in range(1, n_steps + 1):
            driver.step()
            nk_driver.advance(dt)
            stats_ours, key = self._energy_stats(
                driver.model, hamiltonian, key=key, n_samples=n_samples
            )
            stats_netket = vstate.expect(hamiltonian)
            self._assert_energy_close(stats_ours, stats_netket, label=f"step={step}")

    def test_log_derivative_matches_netket(self):
        hi, _, model = self._build_system()
        params = {"tensors": [jnp.asarray(t) for t in model.tensors]}
        sampler = nk.sampler.MetropolisLocal(
            hi,
            n_chains=self.N_CHAINS,
            sweep_size=int(hi.size),
            reset_chains=False,
        )
        vstate = nk.vqs.MCState(
            sampler,
            model=None,
            n_samples=self.N_SAMPLES_REAL,
            n_discard_per_chain=self.BURN_IN,
            variables={"params": params},
            apply_fun=_mps_apply_fun,
            sampler_seed=self.SEED,
        )
        samples = jnp.asarray(vstate.samples).reshape(-1, hi.size)
        amps, grads_sliced, _ = _value_and_grad(
            model,
            samples,
            full_gradient=False,
        )
        n_sites = model.n_sites
        bond_dim = model.bond_dim
        phys_dim = model.phys_dim

        def expand_sliced(sample, grad_sliced):
            indices = spin_to_occupancy(sample)
            parts = []
            offset = 0
            for site in range(n_sites):
                left_dim = 1 if site == 0 else bond_dim
                right_dim = 1 if site == n_sites - 1 else bond_dim
                size_per_phys = left_dim * right_dim
                grad_site = grad_sliced[offset : offset + size_per_phys]
                offset += size_per_phys
                full_site = jnp.zeros(
                    (phys_dim, size_per_phys), dtype=grad_sliced.dtype
                )
                full_site = full_site.at[indices[site]].set(grad_site)
                parts.append(full_site.reshape(-1))
            return jnp.concatenate(parts)

        grads_full = jax.vmap(expand_sliced)(samples, grads_sliced)
        grads_log = grads_full / amps[:, None]

        def log_psi(p, s):
            return _mps_apply_fun({"params": p}, s)

        jac_fun = jax.jacrev(log_psi, holomorphic=True)
        jac_tree = jax.vmap(jac_fun, in_axes=(None, 0))(params, samples)
        jac_leaves = [
            leaf.reshape(samples.shape[0], -1)
            for leaf in jax.tree_util.tree_leaves(jac_tree)
        ]
        jac = jnp.concatenate(jac_leaves, axis=1)
        diff = jnp.max(jnp.abs(grads_log - jac))
        logger.warning("max_log_deriv_diff=%s", float(diff))
        self.assertLess(diff, 1e-12)

    def test_params_update_after_step(self):
        """Regression test: verify model params change after driver.step()."""
        _, hamiltonian, model = self._build_system()
        driver = self._build_driver(
            model,
            hamiltonian,
            time_unit=ImaginaryTimeUnit(),
            dt=self.DT_IMAG,
            n_samples=self.N_SAMPLES_IMAG,
        )
        params_before = jax.tree.map(jnp.copy, driver._get_model_params())
        driver.step()
        params_after = driver._get_model_params()
        changed = jax.tree.map(
            lambda a, b: jnp.any(a != b), params_before, params_after
        )
        any_changed = jax.tree.reduce(lambda x, y: x or y, changed)
        self.assertTrue(any_changed, "Parameters should change after step()")

    def test_real_time_matches_netket(self):
        self._run_comparison(
            time_unit=RealTimeUnit(),
            propagation_type="real",
            ode_solver=nkx_dynamics.RK4(self.DT_REAL),
            dt=self.DT_REAL,
            n_steps=self.N_STEPS_REAL,
            n_samples=self.N_SAMPLES_REAL,
        )

    def test_imaginary_time_matches_netket(self):
        self._run_comparison(
            time_unit=ImaginaryTimeUnit(),
            propagation_type="imag",
            ode_solver=nkx_dynamics.Euler(self.DT_IMAG),
            dt=self.DT_IMAG,
            n_steps=self.N_STEPS_IMAG,
            n_samples=self.N_SAMPLES_IMAG,
        )


if __name__ == "__main__":
    unittest.main()
