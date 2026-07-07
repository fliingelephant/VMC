"""Phase-2 gate: fermionic matter statistics in non-Abelian GI-PEPS kernels.

Pins the column-major Jordan-Wigner convention against an independent dense
reference: ``exact_matter_hamiltonian`` derives string signs by explicit mode
enumeration (``jw_string_sign``), never by the kernel's prefix/suffix
bookkeeping.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from nonabelian_exact import (
    context_for_sample,
    exact_matter_hamiltonian,
    hopping_outcomes,
    jw_string_sign,
    valid_samples,
    weighted_block_tensors,
)
from vmc.gauge_groups import SU2
from vmc.operators.local_terms import LocalHamiltonian
from vmc.peps.common.strategy import NoTruncation
from vmc.peps.non_abelian_gi import (
    FermionicHorizontalMatterHoppingTerm,
    FermionicVerticalMatterHoppingTerm,
    HorizontalMatterHoppingTerm,
    NonAbelianGIPEPS,
    NonAbelianGIPEPSConfig,
    PlaquetteTerm,
    VerticalMatterHoppingTerm,
    build_link_casimir_terms,
    build_matter_number_terms,
    build_mc_kernels,
)

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
        ),
        contraction_strategy=NoTruncation(),
    )


@pytest.fixture(scope="module")
def dense(model):
    samples, h_ferm = exact_matter_hamiltonian(model, fermionic=True, **COEFFS)
    _, h_bos = exact_matter_hamiltonian(model, fermionic=False, **COEFFS)
    return samples, h_ferm, h_bos


def test_fermionic_hamiltonian_is_hermitian(dense):
    _, h_ferm, h_bos = dense
    assert jnp.allclose(h_bos, h_bos.conj().T)
    assert jnp.allclose(h_ferm, h_ferm.conj().T)


def test_fermionic_and_bosonic_spectra_differ(dense):
    """Hop-loops closed by a plaquette flip exchange the pair: -1 JW flux.

    Without the plaquette term every closed configuration loop toggles each
    link evenly and realizes an even permutation, so the strings would be a
    removable gauge and the spectra would coincide.
    """
    _, h_ferm, h_bos = dense
    assert not jnp.allclose(jnp.linalg.eigvalsh(h_ferm), jnp.linalg.eigvalsh(h_bos))


def test_horizontal_hop_local_energy_carries_string_sign(model):
    """Anchor: hop (0,0)-(0,1) crosses mode (1,0); odd parity there flips it."""
    matter_parity = jnp.asarray([n % 2 for n in model.matter_numbers])
    kernels = [
        build_mc_kernels(model, LocalHamiltonian(shape=model.shape, terms=(term,)))[2]
        for term in (
            HorizontalMatterHoppingTerm(row=0, col=0),
            FermionicHorizontalMatterHoppingTerm(row=0, col=0),
        )
    ]
    tensors = weighted_block_tensors(model)
    flipped = 0
    for sample in valid_samples(model):
        if not hopping_outcomes(model, sample, row=0, col=0, horizontal=True):
            continue
        matter = NonAbelianGIPEPS.unflatten_spin_network_sample(sample, model.shape)[0]
        sign = jw_string_sign(matter_parity[matter], (0, 0), (0, 1))
        context = context_for_sample(model, tensors, sample)
        bos, ferm = (
            estimate(tensors, sample, context)[1].local_estimate[0]
            for estimate in kernels
        )
        assert jnp.allclose(ferm, sign * bos)
        flipped += sign == -1 and not jnp.allclose(bos, 0.0)
    assert flipped


def test_vertical_hop_is_string_free(model):
    """Vertical neighbors are adjacent modes: fermionic == bosonic evaluation."""
    kernels = [
        build_mc_kernels(model, LocalHamiltonian(shape=model.shape, terms=(term,)))[2]
        for term in (
            VerticalMatterHoppingTerm(row=0, col=0),
            FermionicVerticalMatterHoppingTerm(row=0, col=0),
        )
    ]
    tensors = weighted_block_tensors(model)
    hopped = 0
    for sample in valid_samples(model):
        if not hopping_outcomes(model, sample, row=0, col=0, horizontal=False):
            continue
        context = context_for_sample(model, tensors, sample)
        bos, ferm = (
            estimate(tensors, sample, context)[1].local_estimate[0]
            for estimate in kernels
        )
        assert jnp.allclose(ferm, bos)
        hopped += not jnp.allclose(bos, 0.0)
    assert hopped


def test_local_energy_matches_dense_fermionic_hamiltonian(model, dense):
    samples, h_ferm, _ = dense
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
    tensors = weighted_block_tensors(model)
    psi = jnp.asarray(
        [
            model.apply(tensors, sample, model.shape, model.tables, model.strategy)
            for sample in samples
        ]
    )
    h_psi = h_ferm @ psi
    for idx, sample in enumerate(samples):
        context = context_for_sample(model, tensors, sample)
        _cache, estimates = estimate(tensors, sample, context)
        assert jnp.allclose(estimates.local_estimate[0], h_psi[idx] / psi[idx])
