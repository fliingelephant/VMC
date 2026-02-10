"""Tests for PEPS/GI QGT sliced ordering behavior."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.core import _sample_counts, _trim_samples, make_mc_sampler
from vmc.operators import LocalHamiltonian, PlaquetteTerm
from vmc.peps import NoTruncation, PEPS, build_mc_kernels
from vmc.peps.gi.local_terms import GILocalHamiltonian, build_electric_terms
from vmc.peps.gi.model import GIPEPS, GIPEPSConfig
from vmc.peps.standard.compat import _value_and_grad
from vmc.preconditioners.preconditioners import _reorder_updates
from vmc.qgt import (
    ParameterSpace,
    QGT,
    SlicedJacobian,
    SiteOrdering,
    SliceOrdering,
    solve_cholesky,
)
from vmc.qgt.qgt import _sliced_dense_blocks
from vmc.utils.smallo import params_per_site, sliced_dims


def _sample_with_kernels(
    model,
    operator,
    *,
    n_samples: int,
    key: jax.Array,
    full_gradient: bool,
    n_chains: int = 1,
    initial_configuration: jax.Array | None = None,
):
    n_samples, num_chains, chain_length, total_samples = _sample_counts(
        n_samples, n_chains
    )
    if initial_configuration is None:
        key, init_key = jax.random.split(key)
        initial_configuration = model.random_physical_configuration(
            init_key, n_samples=num_chains
        )
    config_states = initial_configuration.reshape(num_chains, -1)
    chain_keys = jax.random.split(key, num_chains)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    init_cache, transition, estimate = build_mc_kernels(
        model,
        operator,
        full_gradient=full_gradient,
    )
    cache = init_cache(tensors, config_states)
    mc_sampler = make_mc_sampler(transition, estimate)
    (_, _, _), (samples_hist, estimates) = mc_sampler(
        tensors,
        config_states,
        chain_keys,
        cache,
        n_steps=chain_length,
    )
    samples = _trim_samples(samples_hist, total_samples, n_samples)
    local_log_derivatives = _trim_samples(
        estimates.local_log_derivatives,
        total_samples,
        n_samples,
    )
    if full_gradient:
        active_slice_indices = None
    else:
        active_slice_indices = _trim_samples(
            estimates.active_slice_indices,
            total_samples,
            n_samples,
        )
    return samples, local_log_derivatives, active_slice_indices


class QGTTest(unittest.TestCase):
    def test_peps_slice_ordering_reorder(self):
        """SliceOrdering Jacobian reordered should match full Jacobian for PEPS."""
        model = PEPS(
            rngs=nnx.Rngs(0),
            shape=(2, 3),
            bond_dim=2,
            contraction_strategy=NoTruncation(),
        )
        samples, _, _ = _sample_with_kernels(
            model,
            LocalHamiltonian(shape=model.shape, terms=()),
            n_samples=32,
            key=jax.random.key(0),
            full_gradient=False,
            n_chains=1,
        )

        amps, grads_full, _ = _value_and_grad(model, samples, full_gradient=True)
        amps, grads, p = _value_and_grad(model, samples, full_gradient=False)
        o_full = grads_full / amps[:, None]
        o = grads / amps[:, None]
        sd = sliced_dims(model)
        pps = tuple(params_per_site(model))

        jac_slice = SlicedJacobian(o, p, sd, SliceOrdering())
        o_slice = _sliced_dense_blocks(jac_slice)
        perm = []
        total = sum(pps)
        site_offset = 0
        for site_idx, n in enumerate(pps):
            for k in range(sd[site_idx]):
                base = k * total + site_offset
                perm.extend(range(base, base + n))
            site_offset += n
        o_slice_reordered = o_slice[:, jnp.asarray(perm)]
        self.assertTrue(jnp.allclose(o_slice_reordered, o_full, rtol=1e-10))

    def test_peps_slice_ordering_solve(self):
        """SliceOrdering solve should match SiteOrdering solve for PEPS."""
        model = PEPS(
            rngs=nnx.Rngs(0),
            shape=(2, 3),
            bond_dim=2,
            contraction_strategy=NoTruncation(),
        )
        samples, _, _ = _sample_with_kernels(
            model,
            LocalHamiltonian(shape=model.shape, terms=()),
            n_samples=64,
            key=jax.random.key(0),
            full_gradient=False,
            n_chains=1,
        )
        amps, grads, p = _value_and_grad(model, samples, full_gradient=False)
        o = grads / amps[:, None]
        sd = sliced_dims(model)
        pps = tuple(params_per_site(model))
        diag_shift = 1e-4

        jac_slice = SlicedJacobian(o, p, sd, SliceOrdering())
        qgt_slice = QGT(jac_slice, space=ParameterSpace())
        s_slice = qgt_slice.to_dense()
        mat_slice = s_slice + diag_shift * jnp.eye(s_slice.shape[0], dtype=s_slice.dtype)
        rhs_slice = jax.random.normal(
            jax.random.key(1), (s_slice.shape[0],), dtype=jnp.complex128
        )
        x_slice = solve_cholesky(mat_slice, rhs_slice)
        x_slice_reordered = _reorder_updates(SliceOrdering(), x_slice, pps, sd)

        jac_site = SlicedJacobian(o, p, sd, SiteOrdering(pps))
        qgt_site = QGT(jac_site, space=ParameterSpace())
        s_site = qgt_site.to_dense()
        mat_site = s_site + diag_shift * jnp.eye(s_site.shape[0], dtype=s_site.dtype)
        rhs_site = _reorder_updates(SliceOrdering(), rhs_slice, pps, sd)
        x_site = solve_cholesky(mat_site, rhs_site)
        err = float(jnp.linalg.norm(x_slice_reordered - x_site) / jnp.linalg.norm(x_site))
        self.assertLess(err, 1e-6)

    def test_gipeps_slice_ordering_reorder(self):
        """SliceOrdering Jacobian reordered should match full Jacobian for GIPEPS."""
        config = GIPEPSConfig(
            shape=(2, 2),
            N=2,
            phys_dim=2,
            Qx=0,
            degeneracy_per_charge=(2, 2),
            charge_of_site=(0, 1),
        )
        model = GIPEPS(
            rngs=nnx.Rngs(123),
            config=config,
            contraction_strategy=NoTruncation(),
        )
        sd = sliced_dims(model)
        pps = tuple(params_per_site(model))
        key = jax.random.key(123)
        init_cfg = model.random_physical_configuration(key, n_samples=1)

        electric_terms = build_electric_terms(config.shape, coeff=0.1, N=config.N)
        plaquette_terms = tuple(
            PlaquetteTerm(row=r, col=c, coeff=0.2)
            for r in range(config.shape[0] - 1)
            for c in range(config.shape[1] - 1)
        )
        operator = GILocalHamiltonian(
            shape=config.shape, terms=electric_terms + plaquette_terms
        )

        samples_full, grads_full, _ = _sample_with_kernels(
            model,
            operator,
            n_samples=8,
            n_chains=1,
            key=key,
            initial_configuration=init_cfg,
            full_gradient=True,
        )
        samples_sliced, grads_sliced, p = _sample_with_kernels(
            model,
            operator,
            n_samples=8,
            n_chains=1,
            key=key,
            initial_configuration=init_cfg,
            full_gradient=False,
        )
        self.assertTrue(jnp.array_equal(samples_full, samples_sliced))
        jac_slice = SlicedJacobian(grads_sliced, p, sd, SliceOrdering())
        o_slice = _sliced_dense_blocks(jac_slice)
        perm = []
        total = sum(pps)
        site_offset = 0
        for site_idx, n in enumerate(pps):
            for k in range(sd[site_idx]):
                base = k * total + site_offset
                perm.extend(range(base, base + n))
            site_offset += n
        o_slice_reordered = o_slice[:, jnp.asarray(perm)]
        self.assertTrue(jnp.allclose(o_slice_reordered, grads_full, rtol=1e-5, atol=1e-6))

    def test_gipeps_slice_ordering_solve(self):
        """SliceOrdering solve should match SiteOrdering solve for GIPEPS."""
        config = GIPEPSConfig(
            shape=(2, 2),
            N=2,
            phys_dim=2,
            Qx=0,
            degeneracy_per_charge=(2, 2),
            charge_of_site=(0, 1),
        )
        model = GIPEPS(
            rngs=nnx.Rngs(123),
            config=config,
            contraction_strategy=NoTruncation(),
        )
        sd = sliced_dims(model)
        pps = tuple(params_per_site(model))
        key = jax.random.key(123)
        init_cfg = model.random_physical_configuration(key, n_samples=1)

        electric_terms = build_electric_terms(config.shape, coeff=0.1, N=config.N)
        plaquette_terms = tuple(
            PlaquetteTerm(row=r, col=c, coeff=0.2)
            for r in range(config.shape[0] - 1)
            for c in range(config.shape[1] - 1)
        )
        operator = GILocalHamiltonian(
            shape=config.shape, terms=electric_terms + plaquette_terms
        )
        _, grads, p = _sample_with_kernels(
            model,
            operator,
            n_samples=16,
            n_chains=1,
            key=key,
            initial_configuration=init_cfg,
            full_gradient=False,
        )
        diag_shift = 1e-4

        jac_slice = SlicedJacobian(grads, p, sd, SliceOrdering())
        qgt_slice = QGT(jac_slice, space=ParameterSpace())
        s_slice = qgt_slice.to_dense()
        mat_slice = s_slice + diag_shift * jnp.eye(s_slice.shape[0], dtype=s_slice.dtype)
        rhs_slice = jax.random.normal(
            jax.random.key(1), (s_slice.shape[0],), dtype=jnp.complex128
        )
        x_slice = solve_cholesky(mat_slice, rhs_slice)
        x_slice_reordered = _reorder_updates(SliceOrdering(), x_slice, pps, sd)

        jac_site = SlicedJacobian(grads, p, sd, SiteOrdering(pps))
        qgt_site = QGT(jac_site, space=ParameterSpace())
        s_site = qgt_site.to_dense()
        mat_site = s_site + diag_shift * jnp.eye(s_site.shape[0], dtype=s_site.dtype)
        rhs_site = _reorder_updates(SliceOrdering(), rhs_slice, pps, sd)
        x_site = solve_cholesky(mat_site, rhs_site)

        err = float(jnp.linalg.norm(x_slice_reordered - x_site) / jnp.linalg.norm(x_site))
        self.assertLess(err, 1e-5)


if __name__ == "__main__":
    unittest.main()
