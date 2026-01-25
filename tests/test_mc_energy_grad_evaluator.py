"""Monte Carlo energy/gradient checks against exact summation."""
from __future__ import annotations

import logging
import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
import netket as nk
from flax import nnx

from vmc.core import _value_and_grad
from vmc.models.mps import MPS
from vmc.models.peps import NoTruncation, PEPS
from vmc.samplers.sequential import sequential_sample
from vmc.utils.vmc_utils import local_estimate

logger = logging.getLogger(__name__)


def _exact_energy_and_grad(
    model,
    states: jax.Array,
    operator: nk.operator.AbstractOperator,
) -> tuple[jax.Array, jax.Array]:
    amps, grads, _ = _value_and_grad(model, states, full_gradient=True)
    weights = jnp.abs(amps) ** 2
    mask = weights > 1e-12
    weights = weights[mask]
    weights = weights / jnp.sum(weights)
    local = local_estimate(model, states[mask], operator, amps[mask])
    energy = jnp.sum(weights * local)
    o = grads[mask] / amps[mask, None]
    grad = jnp.sum(
        weights[:, None] * (local - energy)[:, None] * o.conj(), axis=0
    )
    return energy, grad


def _mc_energy_and_grad(
    model,
    operator: nk.operator.AbstractOperator,
    *,
    n_samples: int,
    n_chains: int,
    burn_in: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    samples = sequential_sample(
        model,
        n_samples=n_samples,
        n_chains=n_chains,
        burn_in=burn_in,
        key=key,
    )
    amps, grads, _ = _value_and_grad(model, samples, full_gradient=True)
    local = local_estimate(model, samples, operator, amps)
    energy = jnp.mean(local)
    o = grads / amps[:, None]
    contrib = (local - energy)[:, None] * o.conj()
    grad = jnp.mean(contrib, axis=0)

    energy_var = jnp.mean(jnp.abs(local - energy) ** 2)
    energy_err = jnp.sqrt(energy_var / samples.shape[0])
    grad_var = jnp.mean(jnp.abs(contrib - grad) ** 2, axis=0)
    grad_err = jnp.sqrt(grad_var / samples.shape[0])
    return energy, grad, energy_err, grad_err


class MCEnergyGradEvaluatorTest(unittest.TestCase):
    SEED = 0
    N_CHAINS = 8
    BURN_IN = 40
    ENERGY_SIGMA_MULT = 6.0
    GRAD_SIGMA_MULT = 6.0

    MPS_SITES = 10
    MPS_BOND_DIM = 4
    MPS_SAMPLES = 16384

    PEPS_SHAPE = (3, 3)
    PEPS_BOND_DIM = 3
    PEPS_SAMPLES = 8192

    def _assert_energy_grad_close(
        self,
        *,
        exact_energy: jax.Array,
        exact_grad: jax.Array,
        mc_energy: jax.Array,
        mc_grad: jax.Array,
        energy_err: jax.Array,
        grad_err: jax.Array,
        label: str,
    ) -> None:
        energy_diff = float(jnp.abs(mc_energy - exact_energy))
        energy_tol = float(self.ENERGY_SIGMA_MULT * (energy_err + 1e-12))
        grad_diff = float(jnp.linalg.norm(mc_grad - exact_grad))
        grad_tol = float(
            self.GRAD_SIGMA_MULT * (jnp.linalg.norm(grad_err) + 1e-12)
        )
        logger.info(
            "mc_exact_compare label=%s energy_diff=%s energy_tol=%s grad_diff=%s grad_tol=%s",
            label,
            energy_diff,
            energy_tol,
            grad_diff,
            grad_tol,
        )
        self.assertLess(energy_diff, energy_tol)
        self.assertLess(grad_diff, grad_tol)

    def test_mps_mc_energy_grad_matches_exact(self) -> None:
        n_sites = self.MPS_SITES
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        graph = nk.graph.Chain(length=n_sites, pbc=True)
        hamiltonian = nk.operator.Heisenberg(
            hi, graph, dtype=jnp.complex128
        )
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        model = MPS(
            rngs=nnx.Rngs(self.SEED),
            n_sites=n_sites,
            bond_dim=self.MPS_BOND_DIM,
        )

        exact_energy, exact_grad = _exact_energy_and_grad(
            model, states, hamiltonian
        )
        mc_energy, mc_grad, energy_err, grad_err = _mc_energy_and_grad(
            model,
            hamiltonian,
            n_samples=self.MPS_SAMPLES,
            n_chains=self.N_CHAINS,
            burn_in=self.BURN_IN,
            key=jax.random.key(self.SEED),
        )
        self._assert_energy_grad_close(
            exact_energy=exact_energy,
            exact_grad=exact_grad,
            mc_energy=mc_energy,
            mc_grad=mc_grad,
            energy_err=energy_err,
            grad_err=grad_err,
            label="mps_heisenberg",
        )

    def test_peps_mc_energy_grad_matches_exact(self) -> None:
        shape = self.PEPS_SHAPE
        n_sites = shape[0] * shape[1]
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        graph = nk.graph.Grid(extent=shape, pbc=False)
        hamiltonian = nk.operator.Heisenberg(
            hi, graph, dtype=jnp.complex128
        )
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        model = PEPS(
            rngs=nnx.Rngs(self.SEED + 1),
            shape=shape,
            bond_dim=self.PEPS_BOND_DIM,
            contraction_strategy=NoTruncation(),
        )

        exact_energy, exact_grad = _exact_energy_and_grad(
            model, states, hamiltonian
        )
        mc_energy, mc_grad, energy_err, grad_err = _mc_energy_and_grad(
            model,
            hamiltonian,
            n_samples=self.PEPS_SAMPLES,
            n_chains=self.N_CHAINS,
            burn_in=self.BURN_IN,
            key=jax.random.key(self.SEED + 1),
        )
        self._assert_energy_grad_close(
            exact_energy=exact_energy,
            exact_grad=exact_grad,
            mc_energy=mc_energy,
            mc_grad=mc_grad,
            energy_err=energy_err,
            grad_err=grad_err,
            label="peps_heisenberg",
        )


if __name__ == "__main__":
    unittest.main()
