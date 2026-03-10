"""Gauge removal for standard PEPS.

Implements Wu & Nys (2026) Sec. III.A: analytically removes gauge redundancy
from the PEPS parameter space via a single QR decomposition.
"""
from __future__ import annotations

from vmc import config  # noqa: F401

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from vmc.gauge.gauge import GaugeConfig, compute_gauge_projection
from vmc.peps.standard.model import PEPS


def _gauge_vectors_horizontal(A1, A2, D):
    """All D² gauge vectors for a horizontal bond.

    A1 (left site, shape (d,u,dn,l,D)), A2 (right site, shape (d,u,dn,D,r)).
    Returns V1 (site1_size, D²), V2 (site2_size, D²).
    """
    eye = jnp.eye(D, dtype=A1.dtype)
    # V1[p,u,d,l,r,i,j] = A1[p,u,d,l,i] * δ(r,j)
    V1 = jnp.einsum('pudli,rj->pudlrij', A1, eye)
    # V2[p,u,d,l,r,i,j] = -A2[p,u,d,j,r] * δ(l,i)
    V2 = -jnp.einsum('pudjr,li->pudlrij', A2, eye)
    return V1.reshape(-1, D * D), V2.reshape(-1, D * D)


def _gauge_vectors_vertical(A1, A2, D):
    """All D² gauge vectors for a vertical bond.

    A1 (top site, shape (d,u,D,l,r)), A2 (bottom site, shape (d,D,dn,l,r)).
    Returns V1 (site1_size, D²), V2 (site2_size, D²).
    """
    eye = jnp.eye(D, dtype=A1.dtype)
    # V1[p,u,d,l,r,i,j] = A1[p,u,i,l,r] * δ(d,j)
    V1 = jnp.einsum('puilr,dj->pudlrij', A1, eye)
    # V2[p,u,d,l,r,i,j] = -A2[p,j,d,l,r] * δ(u,i)
    V2 = -jnp.einsum('pjdlr,ui->pudlrij', A2, eye)
    return V1.reshape(-1, D * D), V2.reshape(-1, D * D)


def _plaquette_constraints(n_rows, n_cols, D, n_h, h_bond_idx, v_bond_idx):
    """Build constraint vectors W in gauge-index space.

    For each elementary plaquette, the oriented sum Σ_k v_b(E^{kk}) around the
    loop vanishes. Convention: horizontal bonds point right, vertical bonds
    point down; clockwise loop gives +1 for forward (top, right) and -1 for
    backward (bottom, left).

    Returns W (M, n_plaq) where M = n_bonds * D².
    """
    n_bonds = len(h_bond_idx) + len(v_bond_idx)
    M = n_bonds * D * D
    n_plaq = (n_rows - 1) * (n_cols - 1)
    if n_plaq == 0:
        return jnp.zeros((M, 0), dtype=jnp.float64)

    rows, cols, vals = [], [], []
    plaq = 0
    for r in range(n_rows - 1):
        for c in range(n_cols - 1):
            bonds_signs = [
                (h_bond_idx[(r, c)], +1.0),        # top: forward
                (v_bond_idx[(r, c + 1)], +1.0),     # right: forward
                (h_bond_idx[(r + 1, c)], -1.0),     # bottom: backward
                (v_bond_idx[(r, c)], -1.0),          # left: backward
            ]
            for b_idx, sign in bonds_signs:
                for k in range(D):
                    rows.append(b_idx * D * D + k * D + k)
                    cols.append(plaq)
                    vals.append(sign)
            plaq += 1

    W = jnp.zeros((M, n_plaq), dtype=jnp.float64)
    for row, col, val in zip(rows, cols, vals):
        W = W.at[row, col].set(val)
    return W


@compute_gauge_projection.dispatch
def compute_gauge_projection(
    cfg: GaugeConfig,
    model: PEPS,
    params,
    *,
    return_info: bool = False,
):
    """Gauge projection for standard PEPS (Wu & Nys 2026, Sec. III.A).

    Returns Q (N_p, N_reduced): orthonormal basis for the physical (gauge-free)
    parameter subspace.
    """
    n_rows, n_cols = model.shape
    D = model.bond_dim
    d = model.phys_dim

    params_flat, _ = ravel_pytree(params)
    N_p = params_flat.shape[0]
    dtype = params_flat.dtype

    # Site offsets and shapes in the flat parameter vector
    offsets = []
    site_sizes = []
    offset = 0
    for r in range(n_rows):
        for c in range(n_cols):
            u, dn, l, ri = PEPS.site_dims(r, c, n_rows, n_cols, D)
            sz = d * u * dn * l * ri
            offsets.append(offset)
            site_sizes.append(sz)
            offset += sz

    def site_idx(r, c):
        return r * n_cols + c

    def get_tensor(r, c):
        i = site_idx(r, c)
        u, dn, l, ri = PEPS.site_dims(r, c, n_rows, n_cols, D)
        return params_flat[offsets[i]:offsets[i] + site_sizes[i]].reshape(d, u, dn, l, ri)

    # Enumerate bonds
    h_bond_idx = {}
    idx = 0
    for r in range(n_rows):
        for c in range(n_cols - 1):
            h_bond_idx[(r, c)] = idx
            idx += 1
    n_h = idx
    v_bond_idx = {}
    for r in range(n_rows - 1):
        for c in range(n_cols):
            v_bond_idx[(r, c)] = idx
            idx += 1
    n_bonds = idx
    M = n_bonds * D * D

    # Plaquette constraints in gauge-index space
    W = _plaquette_constraints(n_rows, n_cols, D, n_h, h_bond_idx, v_bond_idx)
    n_plaq = W.shape[1]
    N_gv = M - n_plaq

    # Orthogonal complement of constraints
    if n_plaq > 0:
        Q_w_full, _ = jnp.linalg.qr(W, mode='complete')
        Q_w_perp = Q_w_full[:, n_plaq:].astype(dtype)  # (M, N_gv)
    else:
        Q_w_perp = jnp.eye(M, dtype=dtype)

    # Build G_indep = G @ Q_w_perp without materializing G.
    # Accumulate bond-by-bond: each bond contributes to 2 sites.
    G_indep = jnp.zeros((N_p, N_gv), dtype=dtype)

    for (r, c), b_idx in h_bond_idx.items():
        V1, V2 = _gauge_vectors_horizontal(get_tensor(r, c), get_tensor(r, c + 1), D)
        qwp = Q_w_perp[b_idx * D * D:(b_idx + 1) * D * D, :]  # (D², N_gv)
        i1, s1 = site_idx(r, c), site_sizes[site_idx(r, c)]
        i2, s2 = site_idx(r, c + 1), site_sizes[site_idx(r, c + 1)]
        G_indep = G_indep.at[offsets[i1]:offsets[i1] + s1, :].add(V1 @ qwp)
        G_indep = G_indep.at[offsets[i2]:offsets[i2] + s2, :].add(V2 @ qwp)

    for (r, c), b_idx in v_bond_idx.items():
        V1, V2 = _gauge_vectors_vertical(get_tensor(r, c), get_tensor(r + 1, c), D)
        qwp = Q_w_perp[b_idx * D * D:(b_idx + 1) * D * D, :]
        i1, s1 = site_idx(r, c), site_sizes[site_idx(r, c)]
        i2, s2 = site_idx(r + 1, c), site_sizes[site_idx(r + 1, c)]
        G_indep = G_indep.at[offsets[i1]:offsets[i1] + s1, :].add(V1 @ qwp)
        G_indep = G_indep.at[offsets[i2]:offsets[i2] + s2, :].add(V2 @ qwp)

    # T = [G_indep | u₁]  (u₁ = global rescaling direction)
    if cfg.include_global_scale:
        T = jnp.concatenate([G_indep, params_flat[:, None]], axis=1)
        n_null = N_gv + 1
    else:
        T = G_indep
        n_null = N_gv

    # Complete QR: T = Q̃ R.  T has full column rank (paper Sec. III.A).
    # Q = last N_p - n_null columns of Q̃.
    Q_tilde, _ = jnp.linalg.qr(T, mode='complete')
    Q = Q_tilde[:, n_null:]

    info = {"n_gauge_vectors": N_gv, "n_null": n_null, "n_reduced": N_p - n_null}
    return (Q, info) if return_info else Q
