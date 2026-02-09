"""Monte Carlo PEPS energy/gradient checks against exact summation."""
from __future__ import annotations

import logging
import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
import netket as nk
from flax import nnx

from vmc.core import _sample_counts, _trim_samples, make_mc_sampler
from vmc.operators.local_terms import (
    HorizontalTwoSiteTerm,
    LocalHamiltonian,
    VerticalTwoSiteTerm,
)
from vmc.peps import NoTruncation, PEPS, build_mc_kernels
from vmc.peps.standard.compat import _value_and_grad, local_estimate
from vmc.utils.utils import spin_to_occupancy

logger = logging.getLogger(__name__)


def _heisenberg_local_hamiltonian(shape: tuple[int, int]) -> LocalHamiltonian:
    sz_sz = jnp.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
        dtype=jnp.complex128,
    )
    exchange = jnp.array(
        [
            [0, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=jnp.complex128,
    )
    bond_op = sz_sz - exchange
    horizontal_terms = []
    vertical_terms = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            if c + 1 < shape[1]:
                horizontal_terms.append(HorizontalTwoSiteTerm(r, c, bond_op))
            if r + 1 < shape[0]:
                vertical_terms.append(VerticalTwoSiteTerm(r, c, bond_op))
    return LocalHamiltonian(shape=shape, terms=tuple(horizontal_terms + vertical_terms))


def _exact_energy_and_grad(
    model: PEPS,
    states_spin: jax.Array,
    operator: LocalHamiltonian,
) -> tuple[jax.Array, jax.Array]:
    states = spin_to_occupancy(states_spin)
    amps, grads, _ = _value_and_grad(model, states, full_gradient=True)
    weights = jnp.abs(amps) ** 2
    mask = weights > 1e-12
    weights = weights[mask]
    weights = weights / jnp.sum(weights)
    local = local_estimate(model, states[mask], operator, amps[mask])
    energy = jnp.sum(weights * local)
    o = grads[mask] / amps[mask, None]
    grad = jnp.sum(weights[:, None] * (local - energy)[:, None] * o.conj(), axis=0)
    return energy, grad


def _mc_energy_and_grad(
    model: PEPS,
    operator: LocalHamiltonian,
    *,
    n_samples: int,
    n_chains: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    n_samples, num_chains, chain_length, total_samples = _sample_counts(
        n_samples,
        n_chains,
    )
    key, init_key = jax.random.split(key)
    config_states = model.random_physical_configuration(
        init_key, n_samples=num_chains
    ).reshape(num_chains, -1)
    chain_keys = jax.random.split(key, num_chains)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    init_cache, transition, estimate = build_mc_kernels(
        model,
        operator,
        full_gradient=True,
    )
    cache = init_cache(tensors, config_states)
    mc_sampler = make_mc_sampler(transition, estimate)
    (_, _, _), (_, estimates) = mc_sampler(
        tensors,
        config_states,
        chain_keys,
        cache,
        n_steps=chain_length,
    )

    local = _trim_samples(estimates.local_estimate, total_samples, n_samples)
    o = _trim_samples(estimates.local_log_derivatives, total_samples, n_samples)
    energy = jnp.mean(local)
    contrib = (local - energy)[:, None] * o.conj()
    grad = jnp.mean(contrib, axis=0)

    energy_var = jnp.mean(jnp.abs(local - energy) ** 2)
    energy_err = jnp.sqrt(energy_var / local.shape[0])
    grad_var = jnp.mean(jnp.abs(contrib - grad) ** 2, axis=0)
    grad_err = jnp.sqrt(grad_var / local.shape[0])
    return energy, grad, energy_err, grad_err


class MCEnergyGradEvaluatorTest(unittest.TestCase):
    SEED = 0
    N_CHAINS = 8
    ENERGY_SIGMA_MULT = 6.0
    GRAD_SIGMA_MULT = 6.0

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

    def test_peps_mc_energy_grad_matches_exact(self) -> None:
        shape = self.PEPS_SHAPE
        n_sites = shape[0] * shape[1]
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        operator = _heisenberg_local_hamiltonian(shape)
        model = PEPS(
            rngs=nnx.Rngs(self.SEED + 1),
            shape=shape,
            bond_dim=self.PEPS_BOND_DIM,
            contraction_strategy=NoTruncation(),
        )

        exact_energy, exact_grad = _exact_energy_and_grad(model, states, operator)
        mc_energy, mc_grad, energy_err, grad_err = _mc_energy_and_grad(
            model,
            operator,
            n_samples=self.PEPS_SAMPLES,
            n_chains=self.N_CHAINS,
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
