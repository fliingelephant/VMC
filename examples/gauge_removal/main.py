"""Minimal gauge-removal QGT spectrum probe for standard PEPS.

This example keeps all inspection-only QGT spectrum work local to the example:
it samples configurations with an empty Hamiltonian, computes the unprojected
and gauge-projected QGT spectra, and writes one ``qgt_spectra.json`` per case.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from flax import nnx
from jax.flatten_util import ravel_pytree

from vmc.core import _sample_counts, _trim_samples, make_mc_sampler
from vmc.gauge import GaugeConfig, compute_gauge_projection
from vmc.operators import LocalHamiltonian
from vmc.peps import NoTruncation, PEPS, Variational, build_mc_kernels
from vmc.qgt import Jacobian, ParameterSpace, QGT, SlicedJacobian
from vmc.qgt.jacobian import SliceOrdering


BOND_DIM = 4
SEED = 42
N_CHAINS = 64


@dataclass(frozen=True)
class CaseConfig:
    name: str
    shape: tuple[int, int]
    contraction_strategy: object
    scheme_label: str
    n_samples: int
    expected_n_params: int


CASES = (
    CaseConfig(
        name="5x5_D4_exact_ns20480",
        shape=(5, 5),
        contraction_strategy=NoTruncation(),
        scheme_label="Exact",
        n_samples=20480,
        expected_n_params=6272,
    ),
    CaseConfig(
        name="5x5_D4_variational_Dc16_ns20480",
        shape=(5, 5),
        contraction_strategy=Variational(16),
        scheme_label="Variational(D_c=16)",
        n_samples=20480,
        expected_n_params=6272,
    ),
    CaseConfig(
        name="8x8_D4_variational_Dc16_ns24000",
        shape=(8, 8),
        contraction_strategy=Variational(16),
        scheme_label="Variational(D_c=16)",
        n_samples=24000,
        expected_n_params=21632,
    ),
)


def build_model(case: CaseConfig) -> PEPS:
    """Build one PEPS instance for the requested gauge-removal probe."""
    return PEPS(
        rngs=nnx.Rngs(SEED),
        shape=case.shape,
        bond_dim=BOND_DIM,
        contraction_strategy=case.contraction_strategy,
        dtype=jnp.float64,
    )


def output_path(case: CaseConfig) -> Path:
    """Return the JSON path for one gauge-removal case."""
    return Path(__file__).resolve().parent / case.name / "qgt_spectra.json"


def _is_cuda_backend() -> bool:
    return any(
        device.platform == "gpu" and "NVIDIA" in device.device_kind.upper()
        for device in jax.devices()
    )


def selected_cases() -> tuple[CaseConfig, ...]:
    """Choose the benchmark set matching the active backend."""
    if _is_cuda_backend():
        return CASES
    return tuple(case for case in CASES if case.shape == (5, 5))


def _site_major_dense_jacobian(
    o: jax.Array,
    p: jax.Array,
    sliced_dims: tuple[int, ...],
    params_per_site: tuple[int, ...],
) -> Jacobian:
    """Expand a sliced Jacobian into dense site-major order."""
    blocks = []
    offset = 0
    for site_idx, n_params in enumerate(params_per_site):
        for k in range(sliced_dims[site_idx]):
            blocks.append(jnp.where(p[:, offset : offset + n_params] == k, o[:, offset : offset + n_params], 0))
        offset += n_params
    return Jacobian(jnp.concatenate(blocks, axis=1))


def _qgt_rank(eigvals: jax.Array) -> jax.Array:
    """Numerical rank derived from QGT eigenvalues."""
    if eigvals.size == 0:
        return jnp.asarray(0, dtype=jnp.int32)
    scale = jnp.maximum(
        jnp.max(jnp.abs(eigvals)),
        jnp.asarray(1.0, dtype=eigvals.dtype),
    )
    tol = jnp.finfo(eigvals.dtype).eps * eigvals.shape[0] * scale
    return jnp.sum(eigvals > tol).astype(jnp.int32)


def _make_sampler(case: CaseConfig, model: PEPS):
    """Build a jitted sequential sampler for one case."""
    n_chains = min(N_CHAINS, case.n_samples)
    _, n_chains, chain_length, total_samples = _sample_counts(
        case.n_samples,
        n_chains,
    )
    operator = LocalHamiltonian(shape=case.shape, terms=())
    init_cache, transition, estimate = build_mc_kernels(
        model,
        operator,
        full_gradient=False,
    )
    mc_sampler = make_mc_sampler(transition, estimate)

    @jax.jit
    def sample(
        tensors: list[list[jax.Array]],
        config_states: jax.Array,
        chain_keys: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        cache = init_cache(tensors, config_states)
        (_, _, _), (samples_hist, estimates) = mc_sampler(
            tensors,
            config_states,
            chain_keys,
            cache,
            n_steps=chain_length,
        )
        samples = _trim_samples(samples_hist, total_samples, case.n_samples)
        o = _trim_samples(
            estimates.local_log_derivatives,
            total_samples,
            case.n_samples,
        )
        p = _trim_samples(
            estimates.active_slice_indices,
            total_samples,
            case.n_samples,
        )
        return samples, o, p

    return sample, n_chains


def _make_inspector(
    model: PEPS,
    gauge_projection: jax.Array,
):
    """Build a jitted QGT-spectrum inspector for one model."""
    sliced_dims = model.sliced_dims
    params_per_site = model.params_per_site

    @jax.jit
    def inspect(o: jax.Array, p: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        jac_unprojected = SlicedJacobian(o, p, sliced_dims, SliceOrdering())
        qgt_eigenvalues = jnp.linalg.eigvalsh(
            QGT(jac_unprojected, space=ParameterSpace()).to_dense()
        )
        jac_site = _site_major_dense_jacobian(
            o,
            p,
            sliced_dims,
            params_per_site,
        )
        jac_projected = Jacobian(jac_site.O @ gauge_projection)
        projected_qgt_eigenvalues = jnp.linalg.eigvalsh(
            QGT(jac_projected, space=ParameterSpace()).to_dense()
        )
        return (
            qgt_eigenvalues,
            _qgt_rank(qgt_eigenvalues),
            projected_qgt_eigenvalues,
            _qgt_rank(projected_qgt_eigenvalues),
        )

    return inspect


def build_record(
    case: CaseConfig,
    tensors: list[list[jax.Array]],
    gauge_info: dict,
    qgt_eigenvalues: jax.Array,
    qgt_rank: jax.Array,
    projected_qgt_eigenvalues: jax.Array,
    projected_qgt_rank: jax.Array,
) -> dict:
    """Convert one completed probe run into a compact JSON record."""
    n_params = ravel_pytree(tensors)[0].shape[0]
    assert n_params == case.expected_n_params, (
        f"{case.name}: expected {case.expected_n_params} params, got {n_params}"
    )
    return {
        "name": case.name,
        "shape": list(case.shape),
        "bond_dim": BOND_DIM,
        "scheme": case.scheme_label,
        "n_samples": case.n_samples,
        "n_params": n_params,
        "step": 0,
        "imaginary_time": 0.0,
        "qgt_rank": int(qgt_rank),
        "projected_qgt_rank": int(projected_qgt_rank),
        "gauge_n_null": int(gauge_info["n_null"]),
        "gauge_n_reduced": int(gauge_info["n_reduced"]),
        "qgt_eigenvalues": jnp.asarray(qgt_eigenvalues).tolist(),
        "projected_qgt_eigenvalues": jnp.asarray(
            projected_qgt_eigenvalues
        ).tolist(),
    }


def run_case(case: CaseConfig) -> dict:
    """Run one single-step gauge-removal spectrum probe."""
    model = build_model(case)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    sampler, n_chains = _make_sampler(case, model)
    key = jax.random.key(SEED)
    key, init_key = jax.random.split(key)
    config_states = model.random_physical_configuration(
        init_key, n_samples=n_chains
    ).reshape(n_chains, -1)
    chain_keys = jax.random.split(key, n_chains)
    _, o, p = sampler(tensors, config_states, chain_keys)
    gauge_projection, gauge_info = compute_gauge_projection(
        GaugeConfig(),
        model,
        tensors,
        return_info=True,
    )
    inspect = _make_inspector(model, gauge_projection)
    (
        qgt_eigenvalues,
        qgt_rank,
        projected_qgt_eigenvalues,
        projected_qgt_rank,
    ) = inspect(o, p)
    jax.block_until_ready(projected_qgt_eigenvalues)

    record = build_record(
        case,
        tensors,
        gauge_info,
        qgt_eigenvalues,
        qgt_rank,
        projected_qgt_eigenvalues,
        projected_qgt_rank,
    )
    path = output_path(case)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([record], indent=2))
    print(f"[{case.name}] saved {path}", flush=True)
    return record


def main() -> None:
    """Run the backend-selected gauge-removal probes."""
    for case in selected_cases():
        run_case(case)


if __name__ == "__main__":
    main()
