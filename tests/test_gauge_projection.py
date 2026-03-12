"""Tests for PEPS gauge projection (Wu & Nys 2026, Sec. III.A)."""
from __future__ import annotations

from vmc import config  # noqa: F401

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax.flatten_util import ravel_pytree

from vmc.drivers import RK4, RealTimeUnit, TDVPDriver
from vmc.gauge import GaugeConfig, compute_gauge_projection
from vmc.gauge.peps_gauge import _gauge_vectors_horizontal, _gauge_vectors_vertical
from vmc.operators import (
    AffineSchedule,
    DiagonalOperator,
    LocalHamiltonian,
    OneSiteOperator,
    TimeDependentHamiltonian,
)
from vmc.peps import PEPS, Variational
from vmc.peps.standard.compat import _value
from vmc.preconditioners import DirectSolve, SRPreconditioner, solve_svd
from vmc.qgt.jacobian import SiteOrdering, SliceOrdering
from vmc.qgt.qgt import _iter_sliced_blocks
from vmc.utils.smallo import params_per_site, sliced_dims


def _make_peps(n_rows, n_cols, bond_dim, phys_dim=2, seed=0):
    model = PEPS(
        rngs=nnx.Rngs(seed),
        shape=(n_rows, n_cols),
        bond_dim=bond_dim,
        phys_dim=phys_dim,
    )
    params = [
        [jnp.asarray(model.tensors[r][c].value) for c in range(n_cols)]
        for r in range(n_rows)
    ]
    return model, params


def _site_layout(n_rows, n_cols, bond_dim, phys_dim, params_flat):
    """Compute site offsets, sizes, and tensor accessor."""
    offsets, sizes = [], []
    off = 0
    for r in range(n_rows):
        for c in range(n_cols):
            u, dn, l, ri = PEPS.site_dims(r, c, n_rows, n_cols, bond_dim)
            sz = phys_dim * u * dn * l * ri
            offsets.append(off)
            sizes.append(sz)
            off += sz

    def get_tensor(r, c):
        i = r * n_cols + c
        u, dn, l, ri = PEPS.site_dims(r, c, n_rows, n_cols, bond_dim)
        return params_flat[offsets[i]:offsets[i] + sizes[i]].reshape(
            phys_dim, u, dn, l, ri
        )

    return offsets, sizes, get_tensor


def _make_full_gauge_vector(bond_type, r, c, col, n_rows, n_cols, D, N_p,
                            offsets, sizes, get_tensor, dtype):
    """Build a full N_p gauge vector for a single E^{ij}."""
    nc = n_cols
    if bond_type == 'h':
        V1, V2 = _gauge_vectors_horizontal(
            get_tensor(r, c), get_tensor(r, c + 1), D
        )
        i1, i2 = r * nc + c, r * nc + c + 1
    else:
        V1, V2 = _gauge_vectors_vertical(
            get_tensor(r, c), get_tensor(r + 1, c), D
        )
        i1, i2 = r * nc + c, (r + 1) * nc + c
    v = jnp.zeros(N_p, dtype=dtype)
    v = v.at[offsets[i1]:offsets[i1] + sizes[i1]].set(V1[:, col])
    v = v.at[offsets[i2]:offsets[i2] + sizes[i2]].set(V2[:, col])
    return v


def _expected_n_gv(n_rows, n_cols, D):
    n_h = n_rows * (n_cols - 1)
    n_v = (n_rows - 1) * n_cols
    M = (n_h + n_v) * D * D
    n_plaq = (n_rows - 1) * (n_cols - 1)
    return M - n_plaq


def _all_occupancy_states(n_sites: int) -> jax.Array:
    basis = jnp.arange(1 << n_sites, dtype=jnp.uint32)
    bit_positions = jnp.arange(n_sites, dtype=jnp.uint32)
    return ((basis[:, None] >> bit_positions[None, :]) & 1).astype(jnp.int32)


def _build_time_dependent_hamiltonian(
    shape: tuple[int, int], *, jzz: float = -1.0, hx0: float = 0.4, hx_slope: float = -0.15
) -> TimeDependentHamiltonian:
    sigmax = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    sigmaz_sigmaz_diag = jnp.array([1.0, -1.0, -1.0, 1.0], dtype=jnp.complex128)
    terms = []
    offsets = []
    slopes = []
    n_rows, n_cols = shape
    for row in range(n_rows):
        for col in range(n_cols):
            terms.append(OneSiteOperator(row=row, col=col, op=-sigmax))
            offsets.append(hx0)
            slopes.append(hx_slope)
            if col + 1 < n_cols:
                terms.append(
                    DiagonalOperator(
                        sites=((row, col), (row, col + 1)),
                        diag=jzz * sigmaz_sigmaz_diag,
                    )
                )
                offsets.append(1.0)
                slopes.append(0.0)
            if row + 1 < n_rows:
                terms.append(
                    DiagonalOperator(
                        sites=((row, col), (row + 1, col)),
                        diag=jzz * sigmaz_sigmaz_diag,
                    )
                )
                offsets.append(1.0)
                slopes.append(0.0)
    return TimeDependentHamiltonian(
        base=LocalHamiltonian(shape=shape, terms=tuple(terms)),
        schedule=AffineSchedule(
            offset=jnp.asarray(offsets, dtype=jnp.float64),
            slope=jnp.asarray(slopes, dtype=jnp.float64),
        ),
    )


def _clone_model(
    shape: tuple[int, int],
    bond_dim: int,
    tensors: list[list[jax.Array]],
    truncate_bond_dimension: int,
) -> PEPS:
    model = PEPS(
        rngs=nnx.Rngs(123),
        shape=shape,
        bond_dim=bond_dim,
        contraction_strategy=Variational(truncate_bond_dimension, n_sweeps=2),
    )
    for row in range(shape[0]):
        for col in range(shape[1]):
            model.tensors[row][col][...] = jnp.array(tensors[row][col])
    return model


def _normalized_state(model: PEPS, states: jax.Array) -> jax.Array:
    amplitudes = _value(model, states)
    return amplitudes / jnp.linalg.norm(amplitudes)


def _phase_aligned_state_error(state_ref: jax.Array, state_test: jax.Array) -> float:
    overlap = jnp.vdot(state_ref, state_test)
    phase = overlap / jnp.where(jnp.abs(overlap) > 0, jnp.abs(overlap), 1.0)
    return float(jnp.linalg.norm(state_ref - phase * state_test))


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestDimensions:
    @pytest.mark.parametrize("shape,D", [
        ((2, 2), 2),
        ((2, 3), 2),
        ((3, 3), 2),
        ((3, 3), 3),
        ((2, 2), 3),
    ])
    def test_shape(self, shape, D):
        model, params = _make_peps(*shape, D)
        cfg = GaugeConfig(include_global_scale=True)
        Q, info = compute_gauge_projection(cfg, model, params, return_info=True)
        N_p = ravel_pytree(params)[0].shape[0]
        N_gv = _expected_n_gv(*shape, D)
        assert Q.shape == (N_p, N_p - N_gv - 1)
        assert info["n_reduced"] == N_p - N_gv - 1

    def test_no_global_scale(self):
        model, params = _make_peps(2, 2, 2)
        cfg = GaugeConfig(include_global_scale=False)
        Q, info = compute_gauge_projection(cfg, model, params, return_info=True)
        N_p = ravel_pytree(params)[0].shape[0]
        N_gv = _expected_n_gv(2, 2, 2)
        assert Q.shape == (N_p, N_p - N_gv)


class TestOrthonormality:
    @pytest.mark.parametrize("shape,D", [
        ((2, 2), 2),
        ((3, 3), 2),
    ])
    def test_orthonormal_columns(self, shape, D):
        model, params = _make_peps(*shape, D)
        Q = compute_gauge_projection(GaugeConfig(), model, params)
        QtQ = Q.conj().T @ Q
        assert jnp.allclose(QtQ, jnp.eye(Q.shape[1]), atol=1e-10)


class TestPlaquetteConstraint:
    """Oriented sum Σ_k v_b(E^{kk}) around a plaquette must vanish."""

    @pytest.mark.parametrize("shape,D", [
        ((2, 2), 2),
        ((3, 3), 2),
        ((2, 3), 3),
    ])
    def test_plaquette_sum_zero(self, shape, D):
        n_rows, n_cols = shape
        model, params = _make_peps(n_rows, n_cols, D)
        params_flat, _ = ravel_pytree(params)
        N_p = params_flat.shape[0]
        offsets, sizes, get_tensor = _site_layout(
            n_rows, n_cols, D, 2, params_flat
        )

        for r in range(n_rows - 1):
            for c in range(n_cols - 1):
                total = jnp.zeros(N_p, dtype=params_flat.dtype)
                for k in range(D):
                    kk = k * D + k
                    total += _make_full_gauge_vector(
                        'h', r, c, kk, n_rows, n_cols, D, N_p,
                        offsets, sizes, get_tensor, params_flat.dtype
                    )
                    total += _make_full_gauge_vector(
                        'v', r, c + 1, kk, n_rows, n_cols, D, N_p,
                        offsets, sizes, get_tensor, params_flat.dtype
                    )
                    total -= _make_full_gauge_vector(
                        'h', r + 1, c, kk, n_rows, n_cols, D, N_p,
                        offsets, sizes, get_tensor, params_flat.dtype
                    )
                    total -= _make_full_gauge_vector(
                        'v', r, c, kk, n_rows, n_cols, D, N_p,
                        offsets, sizes, get_tensor, params_flat.dtype
                    )
                assert jnp.allclose(total, 0.0, atol=1e-12)


class TestGaugeVectorsInNullSpace:
    """Every gauge vector must satisfy Q† v = 0."""

    @pytest.mark.parametrize("shape,D", [
        ((2, 2), 2),
        ((2, 3), 2),
    ])
    def test_gauge_orthogonal_to_Q(self, shape, D):
        n_rows, n_cols = shape
        model, params = _make_peps(n_rows, n_cols, D)
        Q = compute_gauge_projection(GaugeConfig(), model, params)
        params_flat, _ = ravel_pytree(params)
        N_p = params_flat.shape[0]
        offsets, sizes, get_tensor = _site_layout(
            n_rows, n_cols, D, 2, params_flat
        )

        # Horizontal gauge vectors
        for r in range(n_rows):
            for c in range(n_cols - 1):
                for ij in range(D * D):
                    v = _make_full_gauge_vector(
                        'h', r, c, ij, n_rows, n_cols, D, N_p,
                        offsets, sizes, get_tensor, params_flat.dtype
                    )
                    assert jnp.allclose(Q.conj().T @ v, 0.0, atol=1e-10)

        # Vertical gauge vectors
        for r in range(n_rows - 1):
            for c in range(n_cols):
                for ij in range(D * D):
                    v = _make_full_gauge_vector(
                        'v', r, c, ij, n_rows, n_cols, D, N_p,
                        offsets, sizes, get_tensor, params_flat.dtype
                    )
                    assert jnp.allclose(Q.conj().T @ v, 0.0, atol=1e-10)


class TestGlobalScaleInNullSpace:
    """u₁ = params_flat must satisfy Q† u₁ = 0 when include_global_scale=True."""

    def test_u1_orthogonal(self):
        model, params = _make_peps(2, 3, 2)
        Q = compute_gauge_projection(
            GaugeConfig(include_global_scale=True), model, params
        )
        u1, _ = ravel_pytree(params)
        assert jnp.allclose(Q.conj().T @ u1, 0.0, atol=1e-10)

    def test_u1_not_orthogonal_when_disabled(self):
        model, params = _make_peps(2, 3, 2)
        Q = compute_gauge_projection(
            GaugeConfig(include_global_scale=False), model, params
        )
        u1, _ = ravel_pytree(params)
        # u₁ should NOT be orthogonal to Q when not included in null space
        assert jnp.linalg.norm(Q.conj().T @ u1) > 1e-6


class TestSlicedJacobianGaugeProjection:
    """Expanding sliced Jacobian with SiteOrdering matches full O @ Q."""

    @pytest.mark.parametrize("shape,D", [
        ((2, 2), 2),
        ((2, 3), 2),
    ])
    def test_site_ordering_matches_full(self, shape, D):
        """O_full @ Q == expand_site_order(o, p) @ Q."""
        n_rows, n_cols = shape
        model, params = _make_peps(n_rows, n_cols, D)
        Q = compute_gauge_projection(GaugeConfig(), model, params)
        params_flat, _ = ravel_pytree(params)
        N_p = params_flat.shape[0]
        d = model.phys_dim
        pps = tuple(params_per_site(model))
        sd = sliced_dims(model)

        # Build full O from params_flat (ground truth, site-major order)
        N_s = 5
        key = jax.random.PRNGKey(42)
        O_full = jax.random.normal(key, (N_s, N_p))

        # Build sliced o and p from O_full
        # o has columns = bond params per site, p has physical index per column
        total_pps = sum(pps)
        o = jnp.zeros((N_s, total_pps), dtype=O_full.dtype)
        p = jnp.zeros((N_s, total_pps), dtype=jnp.int32)
        samples = jax.random.randint(
            jax.random.PRNGKey(7), (N_s, n_rows * n_cols), 0, d
        )

        # For each sample, extract the active slice
        site_offset_full = 0
        site_offset_o = 0
        for site_idx in range(n_rows * n_cols):
            n = pps[site_idx]
            for s in range(N_s):
                k = samples[s, site_idx]
                o = o.at[s, site_offset_o:site_offset_o + n].set(
                    O_full[s, site_offset_full + k * n:site_offset_full + (k + 1) * n]
                )
                p = p.at[s, site_offset_o:site_offset_o + n].set(k)
            site_offset_full += d * n
            site_offset_o += n

        # Expand with SiteOrdering and project
        site_order = SiteOrdering(pps)
        blocks = [ok for ok, _ in _iter_sliced_blocks(o, p, sd, site_order)]
        O_expanded = jnp.concatenate(blocks, axis=1)

        # Zero out O_full entries not matching samples (to match sliced behavior)
        O_masked = jnp.zeros_like(O_full)
        site_offset = 0
        for site_idx in range(n_rows * n_cols):
            n = pps[site_idx]
            for k in range(d):
                mask = (samples[:, site_idx] == k)[:, None]
                O_masked = O_masked.at[:, site_offset + k * n:site_offset + (k + 1) * n].set(
                    jnp.where(mask, O_full[:, site_offset + k * n:site_offset + (k + 1) * n], 0)
                )
            site_offset += d * n

        # Both should give the same projection
        assert jnp.allclose(O_expanded @ Q, O_masked @ Q, atol=1e-10)

    @pytest.mark.parametrize("shape,D", [
        ((2, 2), 2),
        ((2, 3), 2),
    ])
    def test_slice_ordering_mismatch(self, shape, D):
        """SliceOrdering expansion has different column order than params_flat."""
        n_rows, n_cols = shape
        model, params = _make_peps(n_rows, n_cols, D)
        pps = tuple(params_per_site(model))
        sd = sliced_dims(model)
        d = model.phys_dim

        N_s = 5
        total_pps = sum(pps)
        key = jax.random.PRNGKey(42)
        o = jax.random.normal(key, (N_s, total_pps))
        samples = jax.random.randint(
            jax.random.PRNGKey(7), (N_s, n_rows * n_cols), 0, d
        )
        p = jnp.zeros((N_s, total_pps), dtype=jnp.int32)
        site_offset = 0
        for site_idx in range(n_rows * n_cols):
            n = pps[site_idx]
            for s in range(N_s):
                p = p.at[s, site_offset:site_offset + n].set(samples[s, site_idx])
            site_offset += n

        # Expand with both orderings
        site_blocks = [ok for ok, _ in _iter_sliced_blocks(o, p, sd, SiteOrdering(pps))]
        slice_blocks = [ok for ok, _ in _iter_sliced_blocks(o, p, sd, SliceOrdering())]
        O_site = jnp.concatenate(site_blocks, axis=1)
        O_slice = jnp.concatenate(slice_blocks, axis=1)

        # Same content but different column ordering
        assert O_site.shape == O_slice.shape
        assert not jnp.allclose(O_site, O_slice, atol=1e-10)


class TestGaugeRemovalTDVPAlignment:
    SHAPE = (3, 3)
    BOND_DIM = 4
    TRUNC_BOND_DIM = 16
    N_SAMPLES = 8192
    N_CHAINS = 64
    DT = 0.01
    N_STEPS = 10
    DIAG_SHIFT = 1e-8
    STATE_TOL = 1e-7
    ENERGY_TOL = 1e-8

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "full_gradient",
        [False, True],
        ids=["sliced_jacobian", "full_jacobian"],
    )
    def test_real_time_gauge_removal_matches_baseline(self, full_gradient: bool):
        """Gauge-removed TDVP should match baseline real-time dynamics."""
        n_sites = self.SHAPE[0] * self.SHAPE[1]
        states = _all_occupancy_states(n_sites)
        operator = _build_time_dependent_hamiltonian(self.SHAPE)

        base_model = PEPS(
            rngs=nnx.Rngs(0),
            shape=self.SHAPE,
            bond_dim=self.BOND_DIM,
            contraction_strategy=Variational(self.TRUNC_BOND_DIM, n_sweeps=2),
        )
        tensors = [
            [jnp.array(base_model.tensors[row][col]) for col in range(self.SHAPE[1])]
            for row in range(self.SHAPE[0])
        ]

        common = dict(
            dt=self.DT,
            t0=0.0,
            time_unit=RealTimeUnit(),
            integrator=RK4(),
            sampler_key=jax.random.key(11),
            n_samples=self.N_SAMPLES,
            n_chains=self.N_CHAINS,
            full_gradient=full_gradient,
        )
        baseline = TDVPDriver(
            _clone_model(
                self.SHAPE,
                self.BOND_DIM,
                tensors,
                self.TRUNC_BOND_DIM,
            ),
            operator,
            preconditioner=SRPreconditioner(
                strategy=DirectSolve(solver=solve_svd),
                diag_shift=self.DIAG_SHIFT,
            ),
            **common,
        )
        projected = TDVPDriver(
            _clone_model(
                self.SHAPE,
                self.BOND_DIM,
                tensors,
                self.TRUNC_BOND_DIM,
            ),
            operator,
            preconditioner=SRPreconditioner(
                strategy=DirectSolve(solver=solve_svd),
                diag_shift=self.DIAG_SHIFT,
                gauge_config=GaugeConfig(),
            ),
            **common,
        )

        state_base = _normalized_state(baseline.model, states)
        state_proj = _normalized_state(projected.model, states)
        assert _phase_aligned_state_error(state_base, state_proj) < 1e-14

        max_state_error = 0.0
        for _ in range(self.N_STEPS):
            baseline.run(self.DT)
            projected.run(self.DT)
            state_base = _normalized_state(baseline.model, states)
            state_proj = _normalized_state(projected.model, states)
            max_state_error = max(
                max_state_error,
                _phase_aligned_state_error(state_base, state_proj),
            )
            assert (
                abs(float(baseline.energy.mean.real - projected.energy.mean.real))
                < self.ENERGY_TOL
            )

        assert max_state_error < self.STATE_TOL
