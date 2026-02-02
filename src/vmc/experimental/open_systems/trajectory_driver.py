"""Quantum trajectory driver for open system evolution.

This module implements the quantum trajectory method for simulating open quantum
systems with Lindblad-type dissipation (e.g., T1 and T2 noise).

The algorithm per time step:
    1. Compute jump probabilities dp_k = gamma_k * dt * <L_k^dag L_k>
    2. Stochastic decision: if random < sum(dp_k), apply jump; else no-jump
    3. No-jump: propagate with H_eff = H - (i/2) sum_k gamma_k L_k^dag L_k
    4. Jump: select site k with probability dp_k/sum(dp), apply |psi> -> L_k|psi>
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.drivers.custom_driver import DynamicsDriver, RealTimeUnit
from vmc.models.peps import PEPS
from vmc.operators.local_terms import DiagonalTerm, LocalHamiltonian

if TYPE_CHECKING:
    from collections.abc import Callable

    from vmc.experimental.open_systems.jump_operators import JumpOperator
    from vmc.preconditioners import SRPreconditioner

logger = logging.getLogger(__name__)

__all__ = ["QuantumTrajectoryDriver"]


class QuantumTrajectoryDriver(DynamicsDriver):
    """Driver for open system evolution via quantum trajectories.

    Extends DynamicsDriver with stochastic quantum jumps for Lindblad dissipation.
    Each trajectory evolves with:
        - No-jump evolution using H_eff = H - (i/2) sum_k gamma_k L_k^dag L_k
        - Stochastic jumps |psi> -> L_k|psi> with probability dp_k = gamma_k dt <L_k^dag L_k>

    Attributes:
        jump_operators: List of JumpOperator instances for T1/T2 noise.
        hamiltonian: Original Hamiltonian (before adding dissipative terms).
        jump_count: Total number of jumps that have occurred.
        last_jumped: Whether the last step involved a jump.
        last_dp_total: Total jump probability in the last step.
    """

    def __init__(
        self,
        model: PEPS,
        hamiltonian: LocalHamiltonian,
        jump_operators: list[JumpOperator],
        *,
        sampler: Callable,
        preconditioner: SRPreconditioner,
        dt: float,
        t0: float = 0.0,
        sampler_key: jax.Array = jax.random.key(0),
        n_chains: int = 1,
    ):
        """Initialize quantum trajectory driver.

        Args:
            model: PEPS model to evolve.
            hamiltonian: System Hamiltonian (LocalHamiltonian).
            jump_operators: List of JumpOperator instances defining dissipation.
            sampler: Sampler function for VMC.
            preconditioner: SR preconditioner for tVMC.
            dt: Time step.
            t0: Initial time.
            sampler_key: JAX random key.
            n_chains: Number of Markov chains for sampling.
        """
        self.jump_operators = jump_operators
        self.hamiltonian = hamiltonian

        # Precompute arrays for vectorized jump probability computation
        self._rates = jnp.array([j.rate for j in jump_operators])
        self._jump_rows = jnp.array([j.row for j in jump_operators])
        self._jump_cols = jnp.array([j.col for j in jump_operators])
        self._diag_lookup = jnp.stack([j.LdagL.diag for j in jump_operators])  # (n_ops, phys_dim)
        self._flat_indices = self._jump_rows * model.shape[1] + self._jump_cols

        # Build H_eff = H - (i/2) sum_k gamma_k L_k^dag L_k
        h_eff = self._build_effective_hamiltonian(hamiltonian, jump_operators)

        super().__init__(
            model=model,
            operator=h_eff,
            sampler=sampler,
            preconditioner=preconditioner,
            dt=dt,
            t0=t0,
            time_unit=RealTimeUnit(),
            sampler_key=sampler_key,
            n_chains=n_chains,
        )

        # Trajectory diagnostics
        self.jump_count = 0
        self.last_jumped = False
        self.last_dp_total = 0.0
        self.last_jump_site: tuple[int, int] | None = None

    @staticmethod
    def _build_effective_hamiltonian(
        hamiltonian: LocalHamiltonian,
        jump_operators: list[JumpOperator],
    ) -> LocalHamiltonian:
        """Build H_eff = H - (i/2) sum_k gamma_k L_k^dag L_k.

        The L_k^dag L_k terms are diagonal, so we add them as DiagonalTerm.
        """
        dissipative_terms = tuple(
            DiagonalTerm(sites=j.LdagL.sites, diag=-0.5j * j.rate * j.LdagL.diag)
            for j in jump_operators
        )
        return LocalHamiltonian(
            shape=hamiltonian.shape,
            terms=hamiltonian.terms + dissipative_terms,
        )

    def _compute_jump_probabilities(
        self, sample: jax.Array, dt: float
    ) -> tuple[jax.Array, float]:
        """Compute jump probabilities for diagonal L†L operators.

        For diagonal operators: ⟨L†L⟩ = diag[spin_value], no PEPS contraction needed.
        Uses precomputed index arrays for vectorized lookup (no Python loop).

        Args:
            sample: Current sample configuration (flat array).
            dt: Time step.

        Returns:
            Tuple of (dp array for each jump operator, total jump probability).
        """
        # Vectorized lookup: get spin at each jump site (uses precomputed indices)
        site_spins = sample[self._flat_indices]

        # Vectorized diagonal lookup: expectations[i] = diag_lookup[i, spin[i]]
        expectations = self._diag_lookup[jnp.arange(len(self.jump_operators)), site_spins].real

        # dp_k = gamma_k * dt * ⟨L_k†L_k⟩
        dp = self._rates * dt * expectations
        return dp, float(jnp.sum(dp))

    def _apply_jump(self, jump_op: JumpOperator) -> None:
        """Apply jump operator L to the PEPS state.

        Modifies the tensor at the jump site: A_new = L @ A_old
        where L acts on the physical index.

        Args:
            jump_op: The jump operator to apply.
        """
        row, col = jump_op.row, jump_op.col
        tensor = jnp.asarray(self.model.tensors[row][col][...])
        # tensor shape: (phys, up, down, left, right)
        # L.op shape: (phys_out, phys_in)
        new_tensor = jnp.einsum("pq,qudlr->pudlr", jump_op.L.op, tensor)
        self.model.tensors[row][col][...] = new_tensor

        # Update internal state to reflect model change
        self._graphdef, params, model_state = nnx.split(self.model, nnx.Param, ...)
        self._params = nnx.to_pure_dict(params)
        self._model_state = nnx.to_pure_dict(model_state)

        # Update sample configuration to reflect the jump
        # After L|s⟩, the new sample value is argmax(|L[:, old_sample]|)
        # For T1 (σ⁻): |1⟩ → |0⟩; for dephasing (σ_z): sample unchanged
        flat_idx = row * self.model.shape[1] + col
        old_sample_val = int(self._sampler_configuration.reshape(-1, self._sampler_configuration.shape[-1])[0, flat_idx])
        new_sample_val = int(jnp.argmax(jnp.abs(jump_op.L.op[:, old_sample_val])))
        self._sampler_configuration = self._sampler_configuration.at[..., flat_idx].set(new_sample_val)

    def step(self, dt: float | None = None) -> dict[str, Any]:
        """Perform one quantum trajectory step.

        Algorithm:
            1. Compute jump probabilities dp_k = gamma_k * dt * <L_k^dag L_k>
            2. Draw random number r ~ U[0,1]
            3. If r < sum(dp_k): select jump k, apply L_k to state
            4. Else: evolve with H_eff using standard tVMC

        Args:
            dt: Time step (uses self.dt if None).

        Returns:
            Dict with step diagnostics: jumped, dp_total, jump_site (if jumped).
        """
        dt_step = self.dt if dt is None else float(dt)

        # Get current sample configuration (use first chain for jump decision)
        sample = self._sampler_configuration[0] if self._sampler_configuration.ndim > 1 else self._sampler_configuration

        # Compute jump probabilities
        dp, dp_total = self._compute_jump_probabilities(sample, dt_step)
        self.last_dp_total = dp_total

        # Stochastic decision
        self._sampler_key, jump_key, select_key = jax.random.split(self._sampler_key, 3)
        r = float(jax.random.uniform(jump_key))

        if r < dp_total:
            # Jump occurred - select which jump via categorical sampling
            k = int(jax.random.categorical(select_key, jnp.log(dp / dp_total + 1e-30)))
            jump_op = self.jump_operators[k]
            self._apply_jump(jump_op)

            self.last_jumped = True
            self.last_jump_site = jump_op.site
            self.jump_count += 1
            self.t += dt_step
            self.step_count += 1

            logger.debug(
                "Step %d: Jump at site %s (dp_total=%.4f)",
                self.step_count, jump_op.site, dp_total
            )
        else:
            # No jump: evolve with H_eff
            super().step(dt_step)
            self.last_jumped = False
            self.last_jump_site = None

            logger.debug(
                "Step %d: No jump (dp_total=%.4f)",
                self.step_count, dp_total
            )

        return {
            "jumped": self.last_jumped,
            "dp_total": dp_total,
            "jump_site": self.last_jump_site,
        }

    def run(
        self,
        T: float,
        *,
        show_progress: bool = True,
        callback: Callable[[QuantumTrajectoryDriver], None] | None = None,
    ) -> dict[str, Any]:
        """Run trajectory for total time T.

        Args:
            T: Total evolution time.
            show_progress: Whether to show progress bar.
            callback: Optional callback called after each step.

        Returns:
            Dict with run statistics.
        """
        from tqdm.auto import tqdm

        t_end = self.t + float(T)
        total_jumps = 0
        total_steps = 0

        pbar = tqdm(total=float(T), disable=not show_progress, unit="t")
        while self.t < t_end:
            dt_step = min(self.dt, t_end - self.t)
            result = self.step(dt_step)
            if result["jumped"]:
                total_jumps += 1
            total_steps += 1
            pbar.update(dt_step)
            if callback is not None:
                callback(self)
        pbar.close()

        return {"total_jumps": total_jumps, "total_steps": total_steps}
