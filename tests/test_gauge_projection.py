"""Tests for PEPS gauge projection (Wu & Nys 2026, Sec. III.A)."""
from __future__ import annotations

from vmc import config  # noqa: F401

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax.flatten_util import ravel_pytree

from vmc.gauge import GaugeConfig, compute_gauge_projection
from vmc.gauge.peps_gauge import _gauge_vectors_horizontal, _gauge_vectors_vertical
from vmc.peps import PEPS
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
