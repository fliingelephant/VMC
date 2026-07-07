"""Spec §8.3 gate: parity-graded degeneracy legs for fermionic LGT matter.

The graded assembly is pinned by an independent reference: even-parity
masks, right-leg gates, and column prefix parities are re-derived here by
explicit entry enumeration (never via ``vmc.peps.grading``) and the 2x2
block network is contracted with one dense einsum.  Kernel local energies
are gated against the dense fermionic Hamiltonian whose Jordan-Wigner signs
come from explicit mode enumeration, transition contexts against the graded
amplitude of the returned sample, gradients against forward-mode autodiff,
and the sampler against ``|psi_graded|**2`` stationarity.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nonabelian_exact import context_for_sample, exact_matter_hamiltonian
from vmc.gauge_groups import SU2
from vmc.operators.local_terms import LocalHamiltonian
from vmc.peps.common.strategy import NoTruncation
from vmc.peps.non_abelian_gi import (
    FermionicHorizontalMatterHoppingTerm,
    FermionicVerticalMatterHoppingTerm,
    NonAbelianGIPEPS,
    NonAbelianGIPEPSConfig,
    PlaquetteTerm,
    build_link_casimir_terms,
    build_matter_number_terms,
    build_mc_kernels,
)
from vmc.peps.non_abelian_gi.contraction import non_abelian_gi_apply

COEFFS = {
    "electric_coeff": 0.7,
    "hopping_coeff": 1.3,
    "mass_coeff": 0.5,
    "plaquette_coeff": -1.2,
}


@pytest.fixture(scope="module")
def model():
    return NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 2),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
            phys_dim=2,
            matter_irreps=(0, 1),
            matter_numbers=(0, 1),
            particle_number=2,
            n_even=1,
        ),
        contraction_strategy=NoTruncation(),
    )


@pytest.fixture(scope="module")
def tensors(model):
    return [[jnp.asarray(tensor) for tensor in row] for row in model.tensors]


@pytest.fixture(scope="module")
def dense(model, tensors):
    samples, h_ferm = exact_matter_hamiltonian(model, fermionic=True, **COEFFS)
    psi = jnp.asarray(
        [
            model.apply(tensors, sample, model.shape, model.tables, model.strategy)
            for sample in samples
        ]
    )
    return samples, h_ferm, psi


def reference_graded_amplitude(model, tensors, sample):
    """Masked/gated 2x2 block network, re-derived entrywise and einsummed."""
    matter = np.asarray(
        NonAbelianGIPEPS.unflatten_spin_network_sample(sample, model.shape)[0]
    )
    active = np.asarray(model.active_block_ids(sample))
    parity = np.asarray([n % 2 for n in model.matter_numbers])[matter]
    sites = []
    for r in range(2):
        for c in range(2):
            block = np.asarray(tensors[r][c])[active[r, c]]
            prefix = int(np.sum(parity[:r, c])) % 2
            gated = np.zeros_like(block)
            for legs in itertools.product(*(range(dim) for dim in block.shape)):
                leg_par = [int(i >= model.n_even) for i in legs]  # (u, d, l, r)
                if (int(parity[r, c]) + sum(leg_par)) % 2:
                    continue
                gated[legs] = block[legs] * (-1) ** (prefix * leg_par[3])
            sites.append(gated.squeeze())
    # Squeezed axes: s00 (down, right), s01 (down, left), s10 (up, right),
    # s11 (up, left); bonds a = v(0,0), b = h(0,0), c = v(0,1), e = h(1,0).
    return jnp.asarray(np.einsum("ab,cb,ae,ce->", *sites))


def test_graded_amplitude_matches_reference_contraction(model, tensors, dense):
    samples, _, psi = dense
    gated = 0
    for idx, sample in enumerate(samples):
        assert jnp.allclose(
            psi[idx], reference_graded_amplitude(model, tensors, sample)
        )
        matter = NonAbelianGIPEPS.unflatten_spin_network_sample(sample, model.shape)[0]
        gated += bool(jnp.any(matter[0] == 1))
        assert not jnp.isclose(
            psi[idx],
            non_abelian_gi_apply(
                tensors, sample, model.shape, model.tables, model.strategy
            ),
        )
    assert gated  # odd column prefixes exercised the right-leg gates


def test_local_energy_matches_dense_fermionic_hamiltonian(model, tensors, dense):
    samples, h_ferm, psi = dense
    n_rows, n_cols = model.shape
    electric_terms = build_link_casimir_terms(model.shape, model.gauge_group)
    number_terms = build_matter_number_terms(model.shape, model.matter_numbers)
    hop_terms = (
        *(
            FermionicHorizontalMatterHoppingTerm(row=row, col=col)
            for row in range(n_rows)
            for col in range(n_cols - 1)
        ),
        *(
            FermionicVerticalMatterHoppingTerm(row=row, col=col)
            for row in range(n_rows - 1)
            for col in range(n_cols)
        ),
    )
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=(*electric_terms, *number_terms, *hop_terms, PlaquetteTerm(row=0, col=0)),
        coeffs=(jnp.asarray(0.7),) * len(electric_terms)
        + (jnp.asarray(0.5),) * len(number_terms)
        + (jnp.asarray(1.3),) * len(hop_terms)
        + (jnp.asarray(-1.2),),
    )
    _init_cache, _transition, estimate = build_mc_kernels(model, operator)
    h_psi = h_ferm @ psi
    for idx, sample in enumerate(samples):
        context = context_for_sample(model, tensors, sample)
        _cache, estimates = estimate(tensors, sample, context)
        assert jnp.allclose(estimates.local_estimate[0], h_psi[idx] / psi[idx])


def _particle_count(model, sample):
    matter = NonAbelianGIPEPS.unflatten_spin_network_sample(sample, model.shape)[0]
    return int(jnp.sum(jnp.asarray(model.matter_numbers)[matter]))


def test_transition_context_amp_is_exact_after_moves(model, tensors, dense):
    samples, _, _ = dense
    init_cache, transition, _estimate = build_mc_kernels(
        model, LocalHamiltonian(shape=model.shape, terms=())
    )
    moved = 0
    for seed, sample in enumerate(samples):
        cache = jax.tree_util.tree_map(
            lambda x: x[0], init_cache(tensors, sample[None, :])
        )
        sample_next, _key, context = transition(
            tensors, sample, jax.random.PRNGKey(seed), cache
        )
        assert _particle_count(model, sample_next) == _particle_count(model, sample)
        assert jnp.allclose(
            context.amp,
            model.apply(
                tensors, sample_next, model.shape, model.tables, model.strategy
            ),
        )
        moved += not jnp.array_equal(sample_next, sample)
    assert moved  # accepted hops exercised the interface re-gauge bookkeeping


def test_full_gradient_matches_forward_derivative(model, tensors, dense):
    samples, _, _ = dense
    _init_cache, _transition, estimate = build_mc_kernels(
        model, LocalHamiltonian(shape=model.shape, terms=()), full_gradient=True
    )
    for seed, idx in ((0, 0), (1, 7), (2, 13)):
        sample = samples[idx]
        key = jax.random.PRNGKey(seed)
        tangents = [
            [
                jax.random.normal(
                    jax.random.fold_in(key, 2 * r + c), tensor.shape, tensor.dtype
                )
                for c, tensor in enumerate(row)
            ]
            for r, row in enumerate(tensors)
        ]
        _amp, expected = jax.jvp(
            lambda ts: model.apply(
                ts, sample, model.shape, model.tables, model.strategy
            ),
            (tensors,),
            (tangents,),
        )
        context = context_for_sample(model, tensors, sample)
        _cache, estimates = estimate(tensors, sample, context)
        assert estimates.active_slice_indices is None
        tangent_flat = jnp.concatenate(
            [tangent.reshape(-1) for row in tangents for tangent in row]
        )
        assert jnp.allclose(
            jnp.dot(estimates.local_log_derivatives, tangent_flat) * context.amp,
            expected,
        )


def test_transition_is_stationary_for_graded_weight(model, tensors, dense):
    """Frequencies match ``|psi_graded|**2`` on the fixed-particle sector.

    The transition conserves the particle number, so the chain explores the
    sector of its start; ``valid_samples`` spans all sectors and is
    conditioned here.
    """
    samples, _, psi = dense
    sector = [
        idx for idx, sample in enumerate(samples) if _particle_count(model, sample) == 2
    ]
    pi = np.abs(np.asarray(psi))[sector] ** 2
    pi = pi / pi.sum()
    init_cache, transition, _estimate = build_mc_kernels(
        model, LocalHamiltonian(shape=model.shape, terms=())
    )

    def step(carry, _):
        sample, key = carry
        cache = jax.tree_util.tree_map(
            lambda x: x[0], init_cache(tensors, sample[None, :])
        )
        sample, key, _ = transition(tensors, sample, key, cache)
        return (sample, key), sample

    (_, _), visited = jax.jit(
        lambda s, k: jax.lax.scan(step, (s, k), None, length=6000)
    )(samples[sector[0]], jax.random.PRNGKey(3))
    visited = np.asarray(visited)[1000:]
    keys = {tuple(np.asarray(samples[idx]).tolist()): i for i, idx in enumerate(sector)}
    counts = np.zeros(len(sector))
    for sample in visited:
        counts[keys[tuple(sample.tolist())]] += 1.0
    assert np.max(np.abs(counts / visited.shape[0] - pi)) < 0.04
