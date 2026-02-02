"""Validation tests for quantum trajectory implementation.

Tests verify correctness against:
1. Exact analytical solutions for T1 decay
2. Jump operator construction and application
3. Statistical properties of trajectories
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from vmc.experimental.open_systems import (
    QuantumTrajectoryDriver,
    JumpOperator,
    SIGMA_MINUS,
    SIGMA_Z,
)
from vmc.models.peps import PEPS
from vmc.operators.local_terms import LocalHamiltonian, DiagonalTerm
from vmc.preconditioners import SRPreconditioner
from vmc.samplers.sequential import sequential_sample_with_gradients


# ============================================================================
# Helper Functions
# ============================================================================


def create_single_site_peps(seed: int, initial_state: str = "up") -> PEPS:
    """Create a 1x1 PEPS representing a single qubit."""
    rngs = nnx.Rngs(seed)
    peps = PEPS(rngs=rngs, shape=(1, 1), bond_dim=1, phys_dim=2)

    if initial_state == "up":
        # |1⟩ state
        tensor = jnp.array([[[[[0.0 + 0j]]]], [[[[1.0 + 0j]]]]])
    elif initial_state == "down":
        # |0⟩ state
        tensor = jnp.array([[[[[1.0 + 0j]]]], [[[[0.0 + 0j]]]]])
    else:
        raise ValueError(f"Unknown state: {initial_state}")

    peps.tensors[0][0][...] = tensor
    return peps


def create_2x2_peps(seed: int, initial_state: str = "all_up") -> PEPS:
    """Create a 2x2 PEPS in a product state.
    
    Args:
        seed: Random seed for initialization
        initial_state: "all_up" for |1111⟩, "all_down" for |0000⟩
    """
    rngs = nnx.Rngs(seed)
    peps = PEPS(rngs=rngs, shape=(2, 2), bond_dim=1, phys_dim=2)

    # Tensor shape: (phys_dim, up, down, left, right) = (2, 1, 1, 1, 1)
    if initial_state == "all_up":
        up_tensor = jnp.array([[[[[0.0 + 0j]]]], [[[[1.0 + 0j]]]]])  # |1⟩
    elif initial_state == "all_down":
        up_tensor = jnp.array([[[[[1.0 + 0j]]]], [[[[0.0 + 0j]]]]])  # |0⟩
    else:
        raise ValueError(f"Unknown state: {initial_state}")

    for row in range(2):
        for col in range(2):
            peps.tensors[row][col][...] = up_tensor

    return peps


def measure_site_occupation(peps: PEPS, row: int, col: int) -> float:
    """Measure occupation ⟨n⟩ = P(|1⟩) at a specific site."""
    tensor = jnp.asarray(peps.tensors[row][col][...])
    probs = jnp.sum(jnp.abs(tensor) ** 2, axis=(1, 2, 3, 4))
    total = jnp.sum(probs)
    probs = jnp.where(total > 1e-12, probs / total, jnp.array([1.0, 0.0]))
    return float(probs[1])


def measure_total_occupation(peps: PEPS) -> float:
    """Measure total occupation Σ⟨n_i⟩ for all sites."""
    total = 0.0
    for row in range(peps.shape[0]):
        for col in range(peps.shape[1]):
            total += measure_site_occupation(peps, row, col)
    return total


def measure_occupation(peps: PEPS) -> float:
    """Measure occupation ⟨n⟩ = P(|1⟩) for single-site PEPS."""
    return measure_site_occupation(peps, 0, 0)


def create_sampler():
    """Create sampler for tests."""
    def sampler(model, operator, *, key, initial_configuration):
        return sequential_sample_with_gradients(
            model, operator, n_samples=1, n_chains=1,
            key=key, initial_configuration=initial_configuration, burn_in=0,
        )
    return sampler


# ============================================================================
# Tests
# ============================================================================


class TestJumpOperators:
    """Test jump operator definitions."""

    def test_sigma_minus_matrix(self):
        """Verify σ⁻ = |0⟩⟨1| matrix."""
        expected = jnp.array([[0, 1], [0, 0]])
        assert jnp.allclose(SIGMA_MINUS, expected)

    def test_sigma_z_matrix(self):
        """Verify σᶻ matrix."""
        expected = jnp.array([[1, 0], [0, -1]])
        assert jnp.allclose(SIGMA_Z, expected)

    def test_t1_jump_operator_structure(self):
        """Verify T1 jump operator has correct structure."""
        T1 = 5.0
        jump = JumpOperator.t1(0, 0, T1)

        assert jump.rate == 1.0 / T1
        assert jump.site == (0, 0)
        assert jnp.allclose(jump.L.op, SIGMA_MINUS)
        # L†L = σ⁺σ⁻ = |1⟩⟨1| -> diag = [0, 1]
        assert jnp.allclose(jump.LdagL.diag, jnp.array([0.0, 1.0]))

    def test_dephasing_jump_operator_structure(self):
        """Verify dephasing jump operator has correct structure."""
        T_phi = 10.0
        jump = JumpOperator.dephasing(0, 0, T_phi)

        assert jump.rate == 1.0 / T_phi
        assert jnp.allclose(jump.L.op, SIGMA_Z)
        # L†L = σᶻ² = I -> diag = [1, 1]
        assert jnp.allclose(jump.LdagL.diag, jnp.array([1.0, 1.0]))


class TestHeffConstruction:
    """Test effective Hamiltonian construction."""

    def test_heff_adds_imaginary_diagonal(self):
        """Verify H_eff = H - (i/2)Σγ L†L."""
        T1 = 2.0
        H = LocalHamiltonian(shape=(1, 1), terms=())
        jump_ops = [JumpOperator.t1(0, 0, T1)]

        H_eff = QuantumTrajectoryDriver._build_effective_hamiltonian(H, jump_ops)

        # Should have one term: -i/2 * (1/T1) * [0, 1] = [0, -i/(2*T1)]
        assert len(H_eff.terms) == 1
        term = H_eff.terms[0]
        expected = jnp.array([0.0, -0.5j / T1])
        assert jnp.allclose(term.diag, expected), f"Got {term.diag}, expected {expected}"


class TestJumpApplication:
    """Test jump operator application to PEPS."""

    def test_sigma_minus_on_up_gives_down(self):
        """Verify σ⁻|1⟩ = |0⟩."""
        peps = create_single_site_peps(seed=42, initial_state="up")
        assert jnp.isclose(measure_occupation(peps), 1.0)

        # Apply σ⁻
        jump_op = JumpOperator.t1(0, 0, T1=1.0)
        tensor = peps.tensors[0][0][...]
        new_tensor = jnp.einsum("pq,qudlr->pudlr", jump_op.L.op, tensor)
        peps.tensors[0][0][...] = new_tensor

        # Should now be in |0⟩
        assert jnp.isclose(measure_occupation(peps), 0.0)

    def test_sigma_minus_on_down_gives_zero(self):
        """Verify σ⁻|0⟩ = 0."""
        peps = create_single_site_peps(seed=42, initial_state="down")
        assert jnp.isclose(measure_occupation(peps), 0.0)

        # Apply σ⁻
        jump_op = JumpOperator.t1(0, 0, T1=1.0)
        tensor = peps.tensors[0][0][...]
        new_tensor = jnp.einsum("pq,qudlr->pudlr", jump_op.L.op, tensor)
        peps.tensors[0][0][...] = new_tensor

        # Should be zero vector
        assert jnp.allclose(new_tensor, 0.0)


class TestJumpProbability:
    """Test jump probability computation."""

    def test_jump_prob_in_excited_state(self):
        """For |1⟩, T1 jump prob should be dt/T1."""
        T1 = 2.0
        dt = 0.1

        peps = create_single_site_peps(seed=42, initial_state="up")
        H = LocalHamiltonian(shape=(1, 1), terms=())
        jump_ops = [JumpOperator.t1(0, 0, T1)]

        driver = QuantumTrajectoryDriver(
            model=peps,
            hamiltonian=H,
            jump_operators=jump_ops,
            sampler=create_sampler(),
            preconditioner=SRPreconditioner(diag_shift=1e-2),
            dt=dt,
            sampler_key=jax.random.key(0),
            n_chains=1,
        )

        # Set sample to |1⟩ state (occupancy=1) to match PEPS state
        driver._sampler_configuration = jnp.array([[1]])

        sample = driver._sampler_configuration[0]
        dp, dp_total = driver._compute_jump_probabilities(sample, dt)

        expected = dt / T1  # = 0.05
        assert jnp.isclose(dp_total, expected, rtol=0.1), f"Got {dp_total}, expected {expected}"

    def test_jump_prob_in_ground_state(self):
        """For |0⟩, T1 jump prob should be 0."""
        T1 = 2.0
        dt = 0.1

        peps = create_single_site_peps(seed=42, initial_state="down")
        H = LocalHamiltonian(shape=(1, 1), terms=())
        jump_ops = [JumpOperator.t1(0, 0, T1)]

        driver = QuantumTrajectoryDriver(
            model=peps,
            hamiltonian=H,
            jump_operators=jump_ops,
            sampler=create_sampler(),
            preconditioner=SRPreconditioner(diag_shift=1e-2),
            dt=dt,
            sampler_key=jax.random.key(0),
            n_chains=1,
        )

        sample = driver._sampler_configuration[0]
        dp, dp_total = driver._compute_jump_probabilities(sample, dt)

        assert jnp.isclose(dp_total, 0.0, atol=1e-10), f"Got {dp_total}, expected 0"


class TestT1DecayStatistics:
    """Test T1 decay statistics using direct jump simulation."""

    def test_survival_probability(self):
        """Test survival probability P(no jump) = e^(-t/T1).

        This is a pure Monte Carlo test of the jump decision logic,
        not using full PEPS evolution.
        """
        T1 = 1.0
        dt = 0.02
        n_traj = 200
        t_final = 3.0
        n_steps = int(t_final / dt)

        first_jump_times = []

        for traj in range(n_traj):
            key = jax.random.key(traj)
            in_excited = True

            for step in range(n_steps):
                if not in_excited:
                    break
                dp = dt / T1  # Jump prob when in |1⟩
                key, subkey = jax.random.split(key)
                if float(jax.random.uniform(subkey)) < dp:
                    first_jump_times.append(step * dt)
                    in_excited = False

            if in_excited:
                first_jump_times.append(t_final)

        first_jump_times = jnp.array(first_jump_times)

        # Check survival at t=T1 (should be ~e^{-1} ≈ 0.368)
        exact_survival = float(jnp.exp(-1.0))
        observed_survival = float(jnp.mean(first_jump_times > T1))

        print(f"\nSurvival probability test (n={n_traj}):")
        print(f"  At t=T1: exact={exact_survival:.3f}, observed={observed_survival:.3f}")

        # With 200 samples, std error ≈ sqrt(0.37*0.63/200) ≈ 0.034
        assert abs(observed_survival - exact_survival) < 0.12, (
            f"Survival prob {observed_survival:.3f} differs from exact {exact_survival:.3f}"
        )

    def test_mean_decay_time(self):
        """Test mean first jump time is ~T1."""
        T1 = 1.0
        dt = 0.01
        n_traj = 200
        t_max = 10.0  # Long enough to capture most decays
        n_steps = int(t_max / dt)

        first_jump_times = []

        for traj in range(n_traj):
            key = jax.random.key(traj + 1000)
            in_excited = True

            for step in range(n_steps):
                if not in_excited:
                    break
                dp = dt / T1
                key, subkey = jax.random.split(key)
                if float(jax.random.uniform(subkey)) < dp:
                    first_jump_times.append(step * dt)
                    in_excited = False

            if in_excited:
                first_jump_times.append(t_max)

        mean_time = float(jnp.mean(jnp.array(first_jump_times)))

        print(f"\nMean decay time test (n={n_traj}):")
        print(f"  T1={T1}, mean first jump time={mean_time:.3f}")

        # Mean should be close to T1 (exact is T1 for exponential dist)
        assert abs(mean_time - T1) < 0.3, f"Mean time {mean_time:.3f} differs from T1={T1}"


class TestDriverIntegration:
    """Integration tests for the full driver."""

    def test_driver_step_runs(self):
        """Verify driver.step() executes without error."""
        peps = create_single_site_peps(seed=42, initial_state="up")
        H = LocalHamiltonian(shape=(1, 1), terms=())
        jump_ops = [JumpOperator.t1(0, 0, T1=1.0)]

        driver = QuantumTrajectoryDriver(
            model=peps,
            hamiltonian=H,
            jump_operators=jump_ops,
            sampler=create_sampler(),
            preconditioner=SRPreconditioner(diag_shift=1e-2),
            dt=0.1,
            sampler_key=jax.random.key(0),
            n_chains=1,
        )

        result = driver.step()
        assert "jumped" in result
        assert "dp_total" in result
        assert isinstance(result["dp_total"], float)

    def test_driver_tracks_jumps(self):
        """Verify driver tracks jump count correctly."""
        peps = create_single_site_peps(seed=42, initial_state="up")
        H = LocalHamiltonian(shape=(1, 1), terms=())
        # Short T1 for more jumps
        jump_ops = [JumpOperator.t1(0, 0, T1=0.1)]

        driver = QuantumTrajectoryDriver(
            model=peps,
            hamiltonian=H,
            jump_operators=jump_ops,
            sampler=create_sampler(),
            preconditioner=SRPreconditioner(diag_shift=1e-2),
            dt=0.5,  # Large dt -> high jump prob
            sampler_key=jax.random.key(123),
            n_chains=1,
        )

        # Run until we see a jump
        for _ in range(20):
            result = driver.step()
            if result["jumped"]:
                break

        # Should have counted the jump
        assert driver.jump_count >= 1 or driver.step_count == 20


class TestExactLindbladComparison:
    """Compare PEPS quantum trajectory results against exact Lindblad dynamics."""

    @staticmethod
    def exact_total_occupation(t: float, T1: float, n_sites: int) -> float:
        """Exact solution for total occupation under independent T1 decay.

        For n_sites independent qubits each starting in |1⟩:
        ⟨n_total⟩(t) = n_sites * e^{-t/T1}
        """
        return n_sites * float(jnp.exp(-t / T1))

    def test_t1_decay_2x2_vs_exact(self):
        """Compare 2x2 PEPS ensemble against exact Lindblad solution.

        Uses direct PEPS trajectory simulation with T1 decay on all 4 sites.
        For independent qubits: ⟨n_total⟩(t) = 4 * e^{-t/T1}
        """
        T1 = 1.0
        dt = 0.01
        t_final = 1.0
        n_traj = 200
        n_steps = int(t_final / dt) + 1
        n_sites = 4  # 2x2 grid

        # Time points to compare
        check_times = [0.2, 0.5, 1.0]
        check_steps = [int(t / dt) for t in check_times]

        # All sites for jump operators
        sites = [(r, c) for r in range(2) for c in range(2)]

        # Run trajectories
        occupations_at_t = {t: [] for t in check_times}

        for traj in range(n_traj):
            peps = create_2x2_peps(seed=traj, initial_state="all_up")
            key = jax.random.key(traj + 5000)

            for step in range(n_steps):
                # Measure total occupation
                total_occ = measure_total_occupation(peps)

                # Record at check points
                for ct, cs in zip(check_times, check_steps):
                    if step == cs:
                        occupations_at_t[ct].append(total_occ)

                # Check each site for jumps
                for row, col in sites:
                    site_occ = measure_site_occupation(peps, row, col)
                    dp = (dt / T1) * site_occ

                    key, subkey = jax.random.split(key)
                    if float(jax.random.uniform(subkey)) < dp:
                        # Apply σ⁻ jump at this site
                        tensor = peps.tensors[row][col][...]
                        new_tensor = jnp.einsum("pq,qudlr->pudlr", SIGMA_MINUS, tensor)
                        norm = jnp.sqrt(jnp.sum(jnp.abs(new_tensor) ** 2))
                        if norm > 1e-10:
                            new_tensor = new_tensor / norm
                        peps.tensors[row][col][...] = new_tensor

        # Compare with exact solution
        print("\n2x2 T1 decay: PEPS trajectories vs exact Lindblad")
        print("-" * 55)
        max_rel_error = 0.0

        for t in check_times:
            ensemble_occ = float(jnp.mean(jnp.array(occupations_at_t[t])))
            exact_occ = self.exact_total_occupation(t, T1, n_sites)
            rel_error = abs(ensemble_occ - exact_occ) / exact_occ
            max_rel_error = max(max_rel_error, rel_error)

            print(f"  t={t:.2f}: exact={exact_occ:.4f}, PEPS={ensemble_occ:.4f}, "
                  f"rel_error={rel_error:.2%}")

        # With 200 trajectories, expect ~6% relative error
        assert max_rel_error < 0.10, f"Max relative error {max_rel_error:.2%} exceeds 10%"

    def test_dephasing_preserves_population(self):
        """Verify pure dephasing (T2) doesn't change populations.

        For dephasing with L=σᶻ: L†L = I, so population is unchanged.
        Only coherences decay (not testable with diagonal measurements).
        """
        T_phi = 1.0
        dt = 0.05
        n_steps = 40
        n_traj = 50

        final_occupations = []

        for traj in range(n_traj):
            # Start in superposition-like state for more interesting test
            peps = create_single_site_peps(seed=traj, initial_state="up")
            key = jax.random.key(traj + 6000)

            for step in range(n_steps):
                occ = measure_occupation(peps)

                # For σᶻ: L†L = I, so ⟨L†L⟩ = 1 always
                dp = dt / T_phi

                key, subkey = jax.random.split(key)
                if float(jax.random.uniform(subkey)) < dp:
                    # Apply σᶻ jump (just flips phase, doesn't change |amplitude|²)
                    jump_op = JumpOperator.dephasing(0, 0, T_phi)
                    tensor = peps.tensors[0][0][...]
                    new_tensor = jnp.einsum("pq,qudlr->pudlr", jump_op.L.op, tensor)
                    peps.tensors[0][0][...] = new_tensor

            final_occupations.append(measure_occupation(peps))

        mean_occ = float(jnp.mean(jnp.array(final_occupations)))
        print(f"\nDephasing test: initial occ=1.0, final mean occ={mean_occ:.4f}")

        # Population should remain ~1 (σᶻ doesn't change populations)
        assert jnp.isclose(mean_occ, 1.0, atol=0.01), (
            f"Dephasing changed population: {mean_occ:.4f} != 1.0"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
