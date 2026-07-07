"""Phase-1 gate for the graded (fermionic) standard PEPS.

Four gates, in dependency order: amplitudes against the brute-force
crossing-count reference, local energies and log-derivatives against the
dense Jordan-Wigner Hamiltonian, sector preservation plus stationarity of
the exchange sampler, and exact-SR convergence to the free-fermion ground
state with a hardcore-boson control.
"""

from __future__ import annotations

import itertools

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from fermionic_exact import (
    basis_index,
    dense_hop_hamiltonian,
    graded_amplitude,
    sector_configs,
)
from vmc.operators.local_terms import (
    FermionicHorizontalTwoSiteOperator,
    FermionicVerticalTwoSiteOperator,
    HorizontalTwoSiteOperator,
    LocalHamiltonian,
    VerticalTwoSiteOperator,
)
from vmc.peps import NoTruncation, PEPS, build_mc_kernels
from vmc.peps.common.contraction import _forward_with_cache
from vmc.peps.common.kernels import Context
from vmc.peps.grading import Grading, even_mask
from vmc.peps.standard.compat import (
    _graded_forward,
    _value_and_grad,
    graded_peps_apply,
    local_estimate,
    peps_apply,
)

HOP = np.zeros((4, 4))
HOP[0b01, 0b10] = HOP[0b10, 0b01] = -1.0


def _grading(shape: tuple[int, int], bond_dim: int) -> Grading:
    n_odd = 2 * ((shape[0] * shape[1]) // 4) or 2
    return Grading(
        phys_parity=(0, 1),
        filling=(shape[0] * shape[1] - n_odd, n_odd),
        n_even=bond_dim // 2 or 1,
    )


def _random_graded_tensors(seed: int, shape: tuple[int, int], grading: Grading):
    rng = np.random.default_rng(seed)
    n_rows, n_cols = shape
    tensors = [
        [
            (rng.standard_normal((2, *dims)) + 1j * rng.standard_normal((2, *dims)))
            * even_mask(grading, dims)
            for c in range(n_cols)
            for dims in [PEPS.site_dims(r, c, n_rows, n_cols, 2)]
        ]
        for r in range(n_rows)
    ]
    return tensors, [[jnp.asarray(t) for t in row] for row in tensors]


def _hop_hamiltonian(shape: tuple[int, int], *, fermionic: bool) -> LocalHamiltonian:
    n_rows, n_cols = shape
    horizontal = (
        FermionicHorizontalTwoSiteOperator if fermionic else HorizontalTwoSiteOperator
    )
    vertical = (
        FermionicVerticalTwoSiteOperator if fermionic else VerticalTwoSiteOperator
    )
    return LocalHamiltonian(
        shape=shape,
        terms=tuple(
            horizontal(r, c, HOP) for r in range(n_rows) for c in range(n_cols - 1)
        )
        + tuple(vertical(r, c, HOP) for r in range(n_rows - 1) for c in range(n_cols)),
    )


@pytest.mark.parametrize("shape", [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)])
def test_graded_amplitude_matches_bruteforce(shape) -> None:
    grading = _grading(shape, 2)
    np_tensors, tensors = _random_graded_tensors(7, shape, grading)
    n = shape[0] * shape[1]
    for occ in itertools.product((0, 1), repeat=n):
        sample = jnp.asarray(occ, dtype=jnp.int32)
        amp = graded_peps_apply(tensors, sample, shape, NoTruncation(), grading=grading)
        ref = graded_amplitude(np_tensors, np.asarray(occ), grading)
        assert abs(complex(amp) - ref) < 1e-10 * max(1.0, abs(ref))


def test_graded_amplitude_matches_bruteforce_3x3_spot() -> None:
    shape = (3, 3)
    grading = Grading(phys_parity=(0, 1), filling=(5, 4), n_even=1)
    np_tensors, tensors = _random_graded_tensors(11, shape, grading)
    rng = np.random.default_rng(3)
    for occ in rng.integers(0, 2, size=(8, 9)):
        amp = graded_peps_apply(
            tensors,
            jnp.asarray(occ, dtype=jnp.int32),
            shape,
            NoTruncation(),
            grading=grading,
        )
        ref = graded_amplitude(np_tensors, occ, grading)
        assert abs(complex(amp) - ref) < 1e-10 * max(1.0, abs(ref))


def test_trivial_grading_matches_bosonic_apply() -> None:
    shape = (2, 3)
    grading = Grading(phys_parity=(0, 0), filling=(4, 2), n_even=2)
    _, tensors = _random_graded_tensors(13, shape, grading)
    for occ in itertools.product((0, 1), repeat=6):
        sample = jnp.asarray(occ, dtype=jnp.int32)
        graded = graded_peps_apply(
            tensors, sample, shape, NoTruncation(), grading=grading
        )
        plain = peps_apply(tensors, sample, shape, NoTruncation())
        assert abs(complex(graded - plain)) < 1e-12


@pytest.mark.parametrize("shape", [(2, 2), (2, 3), (3, 2)])
def test_local_estimate_matches_jw_dense(shape) -> None:
    grading = _grading(shape, 2)
    model = PEPS(
        rngs=nnx.Rngs(5),
        shape=shape,
        bond_dim=2,
        contraction_strategy=NoTruncation(),
        grading=grading,
    )
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    ham = _hop_hamiltonian(shape, fermionic=True)
    init_cache, transition, estimate = build_mc_kernels(model, ham)
    _, _, estimate_full = build_mc_kernels(model, ham, full_gradient=True)

    n = shape[0] * shape[1]
    all_configs = jnp.asarray(
        list(itertools.product((0, 1), repeat=n)), dtype=jnp.int32
    )
    psi = jax.vmap(
        lambda s: graded_peps_apply(tensors, s, shape, NoTruncation(), grading=grading)
    )(all_configs)
    h_dense = jnp.asarray(dense_hop_hamiltonian(shape), dtype=psi.dtype)
    expected_eloc = (h_dense @ psi) / psi

    key = jax.random.key(0)
    for start in sector_configs(shape, grading.filling[1]):
        key, chain_key = jax.random.split(key)
        cache = jax.tree.map(
            lambda x: x[0], init_cache(tensors, jnp.asarray(start)[None, :])
        )
        cfg, chain_key, context = transition(
            tensors, jnp.asarray(start), chain_key, cache
        )
        assert int(jnp.sum(cfg)) == int(np.sum(start))
        idx = basis_index(np.asarray(cfg))
        assert abs(complex(context.amp - psi[idx])) < 1e-10 * abs(complex(psi[idx]))

        _, estimates = estimate(tensors, cfg, context)
        assert abs(
            complex(estimates.local_estimate[0] - expected_eloc[idx])
        ) < 1e-9 * max(1.0, abs(complex(expected_eloc[idx])))

        _, estimates_full = estimate_full(tensors, cfg, context)
        tangent = jax.tree.map(
            lambda t: jnp.asarray(
                np.random.default_rng(1).standard_normal(t.shape)
                + 1j * np.random.default_rng(2).standard_normal(t.shape)
            ),
            tensors,
        )
        _, jvp_val = jax.jvp(
            lambda ts: jnp.log(
                graded_peps_apply(ts, cfg, shape, NoTruncation(), grading=grading)
            ),
            (tensors,),
            (tangent,),
        )
        tangent_flat = jnp.concatenate([t.reshape(-1) for row in tangent for t in row])
        assert abs(
            complex(
                jnp.dot(estimates_full.local_log_derivatives, tangent_flat) - jvp_val
            )
        ) < 1e-9 * max(1.0, abs(complex(jvp_val)))

        amp_c, grad_c, _ = _value_and_grad(model, cfg)
        cfg_2d = cfg.reshape(shape)
        tangent_active = jnp.concatenate(
            [
                tangent[r][c][cfg_2d[r, c]].reshape(-1)
                for r in range(shape[0])
                for c in range(shape[1])
            ]
        )
        assert abs(complex(amp_c - psi[idx])) < 1e-10 * abs(complex(psi[idx]))
        expected_dpsi = jvp_val * psi[idx]
        assert abs(
            complex(jnp.dot(grad_c, tangent_active) - expected_dpsi)
        ) < 1e-9 * max(1.0, abs(complex(expected_dpsi)))

    sector = jnp.asarray(sector_configs(shape, grading.filling[1]))
    sector_idx = jnp.asarray([basis_index(s) for s in np.asarray(sector)])
    local = local_estimate(model, sector, ham, psi[sector_idx])
    np.testing.assert_allclose(
        np.asarray(local), np.asarray(expected_eloc[sector_idx]), rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize("shape", [(2, 2), (1, 4), (4, 1)])
def test_exchange_sampler_preserves_sector_and_stationarity(shape) -> None:
    grading = _grading(shape, 2)
    model = PEPS(
        rngs=nnx.Rngs(9),
        shape=shape,
        bond_dim=2,
        contraction_strategy=NoTruncation(),
        grading=grading,
    )
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    init_cache, transition, _ = build_mc_kernels(
        model, _hop_hamiltonian(shape, fermionic=True)
    )

    n_fermions = grading.filling[1]
    sector = sector_configs(shape, n_fermions)
    pi = (
        np.abs(
            np.asarray(
                jax.vmap(
                    lambda s: graded_peps_apply(
                        tensors, s, shape, NoTruncation(), grading=grading
                    )
                )(jnp.asarray(sector))
            )
        )
        ** 2
    )
    pi = pi / pi.sum()

    def step(carry, _):
        sample, key = carry
        cache = jax.tree.map(lambda x: x[0], init_cache(tensors, sample[None, :]))
        sample, key, _ = transition(tensors, sample, key, cache)
        return (sample, key), sample

    start = model.random_physical_configuration(jax.random.key(1))[0].reshape(-1)
    (_, _), samples = jax.jit(
        lambda s, k: jax.lax.scan(step, (s, k), None, length=6000)
    )(start, jax.random.key(2))

    samples = np.asarray(samples)[1000:]
    assert np.all(samples.sum(axis=1) == n_fermions)
    weights = 2 ** np.arange(samples.shape[1] - 1, -1, -1)
    counts = np.bincount(samples @ weights, minlength=2 ** samples.shape[1])
    freq = counts[sector @ weights] / samples.shape[0]
    assert np.max(np.abs(freq - pi)) < 0.04


def test_free_fermion_exact_sr_convergence_with_boson_control() -> None:
    shape, bond_dim, n_fermions = (2, 3), 4, 2
    n_sites = 6
    site = lambda r, c: r * shape[1] + c  # noqa: E731
    h1 = np.zeros((n_sites, n_sites))
    for r in range(shape[0]):
        for c in range(shape[1]):
            if c + 1 < shape[1]:
                h1[site(r, c), site(r, c + 1)] = h1[site(r, c + 1), site(r, c)] = -1.0
            if r + 1 < shape[0]:
                h1[site(r, c), site(r + 1, c)] = h1[site(r + 1, c), site(r, c)] = -1.0
    e_ff = float(np.sum(np.sort(np.linalg.eigvalsh(h1))[:n_fermions]))

    sector = sector_configs(shape, n_fermions)
    sector_idx = [basis_index(s) for s in sector]
    h_f = dense_hop_hamiltonian(shape)[np.ix_(sector_idx, sector_idx)]
    assert abs(float(np.min(np.linalg.eigvalsh(h_f))) - e_ff) < 1e-10
    h_b = dense_hop_hamiltonian(shape, fermionic=False)[np.ix_(sector_idx, sector_idx)]
    e_bos = float(np.min(np.linalg.eigvalsh(h_b)))
    assert e_bos < e_ff - 0.05

    configs = jnp.asarray(sector)
    dt, diag_shift, n_steps, tol = 0.05, 1e-3, 400, 0.02

    def run(model, estimate, forward):
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        sizes = [int(np.prod(t.shape)) for row in tensors for t in row]
        splits = np.cumsum(sizes)[:-1]

        @jax.jit
        def sr_step(tensors):
            def per_config(cfg):
                amp, top_envs = forward(tensors, cfg.reshape(shape))
                _, est = estimate(
                    tensors, cfg, Context(amp=amp, top_envs=tuple(top_envs))
                )
                return amp, est.local_estimate[0], est.local_log_derivatives

            amps, eloc, o = jax.vmap(per_config)(configs)
            w = jnp.abs(amps) ** 2
            w = w / jnp.sum(w)
            energy = jnp.sum(w * eloc)
            o_mean = jnp.sum(w[:, None] * o, axis=0)
            weighted = o * jnp.sqrt(w[:, None])
            s_mat = weighted.conj().T @ weighted - jnp.outer(o_mean.conj(), o_mean)
            forces = o.conj().T @ (w * (eloc - energy))
            update = jnp.linalg.solve(
                s_mat + diag_shift * jnp.eye(s_mat.shape[0], dtype=s_mat.dtype),
                -forces,
            )
            parts = jnp.split(update, splits)
            return [
                [
                    t + dt * parts[r * shape[1] + c].reshape(t.shape)
                    for c, t in enumerate(row)
                ]
                for r, row in enumerate(tensors)
            ], energy

        energy = None
        for _ in range(n_steps):
            tensors, energy = sr_step(tensors)
        return float(jnp.real(energy))

    grading = Grading(phys_parity=(0, 1), filling=(4, 2), n_even=2)
    model_f = PEPS(
        rngs=nnx.Rngs(21),
        shape=shape,
        bond_dim=bond_dim,
        contraction_strategy=NoTruncation(),
        grading=grading,
    )
    _, _, estimate_f = build_mc_kernels(
        model_f, _hop_hamiltonian(shape, fermionic=True), full_gradient=True
    )
    e_vmc_f = run(
        model_f,
        estimate_f,
        lambda tensors, spins: _graded_forward(
            tensors, spins, shape, NoTruncation(), grading
        ),
    )
    assert abs(e_vmc_f - e_ff) < tol
    assert abs(e_vmc_f - e_bos) > 2 * tol

    model_b = PEPS(
        rngs=nnx.Rngs(22),
        shape=shape,
        bond_dim=bond_dim,
        contraction_strategy=NoTruncation(),
    )
    _, _, estimate_b = build_mc_kernels(
        model_b, _hop_hamiltonian(shape, fermionic=False), full_gradient=True
    )
    e_vmc_b = run(
        model_b,
        estimate_b,
        lambda tensors, spins: _forward_with_cache(
            tensors, spins, shape, NoTruncation()
        ),
    )
    assert abs(e_vmc_b - e_bos) < tol
