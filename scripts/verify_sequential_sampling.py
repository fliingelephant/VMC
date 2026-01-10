#!/usr/bin/env python3
"""Verify sequential Metropolis sampling against FullSum and NetKet MH."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
import time
from typing import Iterable


def _bootstrap_vmc() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if "VMC" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "VMC",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to bootstrap VMC package.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["VMC"] = module
    spec.loader.exec_module(module)


_bootstrap_vmc()

import jax
import jax.numpy as jnp
import netket as nk

from examples.real_time import build_heisenberg_square
from examples.real_time_minimal import random_flip_sample
from VMC.samplers.sequential import peps_sequential_sample, sequential_sample_mps
from models.mps import SimpleMPS
from models.peps import SimplePEPS
from utils.vmc_utils import get_apply_fun
from flax import nnx


def _stats_from_samples(
    vstate: nk.vqs.MCState,
    hamiltonian,
    samples: jax.Array,
) -> nk.stats.Stats:
    n_chains = int(vstate.sampler.n_chains)
    n_sites = int(vstate.hilbert.size)
    chain_length = int(samples.shape[0] // n_chains)
    vstate._samples = samples.reshape(n_chains, chain_length, n_sites)
    apply_fun, params, model_state, kwargs = get_apply_fun(vstate)
    flat = vstate._samples.reshape(-1, n_sites)
    logpsi = jax.vmap(
        lambda s: apply_fun({"params": params, **model_state}, s, **kwargs)
    )(flat)
    vstate._logpsi = logpsi.reshape(n_chains, chain_length)
    return vstate.expect(hamiltonian)


def _sequential_mps_samples(
    model: SimpleMPS,
    *,
    n_samples: int,
    n_sweeps: int,
    burn_in: int,
    key: jax.Array,
) -> jax.Array:
    return sequential_sample_mps(
        model,
        n_samples=n_samples,
        n_sweeps=n_sweeps,
        burn_in=burn_in,
        key=key,
    )


def _sequential_peps_samples(
    model: SimplePEPS,
    *,
    n_samples: int,
    n_sweeps: int,
    burn_in: int,
    key: jax.Array,
) -> jax.Array:
    samples, _ = peps_sequential_sample(
        model,
        n_samples=n_samples,
        n_sweeps=n_sweeps,
        burn_in=burn_in,
        key=key,
    )
    return samples


def _build_vstate(
    hi,
    model,
    *,
    n_samples: int,
    sweep_size: int,
    seed: int,
) -> nk.vqs.MCState:
    sampler = nk.sampler.MetropolisLocal(
        hi, n_chains=1, sweep_size=sweep_size, reset_chains=False
    )
    return nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=0,
        sampler_seed=seed,
    )


def _print_stats(prefix: str, stats: nk.stats.Stats) -> None:
    std = jnp.sqrt(stats.variance)
    print(f"  {prefix} mean {stats.mean}")
    print(f"  {prefix} std  {std}")


def _fullsum_energy(hi, hamiltonian, model, params, model_state) -> jax.Array:
    fullsum_state = nk.vqs.FullSumState(hi, model)
    fullsum_state.parameters = params
    if model_state:
        fullsum_state.model_state = model_state
    return fullsum_state.expect(hamiltonian).mean


def _run_fullsum_check(
    *,
    model_kind: str,
    length: int,
    n_samples: int,
    n_sweeps: int,
    burn_in: int,
    bond_dim: int,
    seed: int,
) -> None:
    hi, hamiltonian, _ = build_heisenberg_square(length, pbc=False)
    sweep_size = int(hi.size * n_sweeps)

    if model_kind == "mps":
        model_seq = SimpleMPS(
            rngs=nnx.Rngs(seed),
            n_sites=hi.size,
            bond_dim=bond_dim,
        )
        model_mh = SimpleMPS(
            rngs=nnx.Rngs(seed + 1),
            n_sites=hi.size,
            bond_dim=bond_dim,
        )
        seq_samples = _sequential_mps_samples(
            model_seq,
            n_samples=n_samples,
            n_sweeps=n_sweeps,
            burn_in=burn_in,
            key=jax.random.key(seed),
        )
    elif model_kind == "peps":
        model_seq = SimplePEPS(
            rngs=nnx.Rngs(seed),
            shape=(length, length),
            bond_dim=bond_dim,
        )
        model_mh = SimplePEPS(
            rngs=nnx.Rngs(seed + 1),
            shape=(length, length),
            bond_dim=bond_dim,
        )
        seq_samples = _sequential_peps_samples(
            model_seq,
            n_samples=n_samples,
            n_sweeps=n_sweeps,
            burn_in=burn_in,
            key=jax.random.key(seed),
        )
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")

    vstate_seq = _build_vstate(
        hi,
        model_seq,
        n_samples=n_samples,
        sweep_size=sweep_size,
        seed=seed,
    )
    vstate_mh = _build_vstate(
        hi,
        model_mh,
        n_samples=n_samples,
        sweep_size=sweep_size,
        seed=seed + 11,
    )
    vstate_mh.parameters = vstate_seq.parameters
    vstate_mh.model_state = vstate_seq.model_state

    fullsum_energy = _fullsum_energy(
        hi,
        hamiltonian,
        model_mh,
        vstate_mh.parameters,
        vstate_mh.model_state,
    )
    seq_stats = _stats_from_samples(vstate_seq, hamiltonian, seq_samples)
    mh_stats = vstate_mh.expect(hamiltonian)

    print(f"FullSum check ({model_kind}) L={length}")
    print(f"  samples={n_samples} sweeps={n_sweeps} burn_in={burn_in}")
    print(f"  fullsum mean {fullsum_energy}")
    _print_stats("sequential", seq_stats)
    _print_stats("netket", mh_stats)
    print(f"  seq-full diff {seq_stats.mean - fullsum_energy}")
    print(f"  netket-full diff {mh_stats.mean - fullsum_energy}")
    if hasattr(vstate_mh.sampler_state, "acceptance"):
        print(f"  netket acceptance {float(vstate_mh.sampler_state.acceptance)}")


def _run_mh_check(
    *,
    model_kind: str,
    length: int,
    n_samples: int,
    n_sweeps: int,
    burn_in: int,
    bond_dim: int,
    seed: int,
) -> None:
    hi, hamiltonian, _ = build_heisenberg_square(length, pbc=False)
    sweep_size = int(hi.size * n_sweeps)

    if model_kind == "mps":
        model_seq = SimpleMPS(
            rngs=nnx.Rngs(seed),
            n_sites=hi.size,
            bond_dim=bond_dim,
        )
        model_mh = SimpleMPS(
            rngs=nnx.Rngs(seed + 1),
            n_sites=hi.size,
            bond_dim=bond_dim,
        )
        seq_samples = _sequential_mps_samples(
            model_seq,
            n_samples=n_samples,
            n_sweeps=n_sweeps,
            burn_in=burn_in,
            key=jax.random.key(seed),
        )
    elif model_kind == "peps":
        model_seq = SimplePEPS(
            rngs=nnx.Rngs(seed),
            shape=(length, length),
            bond_dim=bond_dim,
        )
        model_mh = SimplePEPS(
            rngs=nnx.Rngs(seed + 1),
            shape=(length, length),
            bond_dim=bond_dim,
        )
        seq_samples = _sequential_peps_samples(
            model_seq,
            n_samples=n_samples,
            n_sweeps=n_sweeps,
            burn_in=burn_in,
            key=jax.random.key(seed),
        )
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")

    vstate_seq = _build_vstate(
        hi,
        model_seq,
        n_samples=n_samples,
        sweep_size=sweep_size,
        seed=seed,
    )
    vstate_mh = _build_vstate(
        hi,
        model_mh,
        n_samples=n_samples,
        sweep_size=sweep_size,
        seed=seed + 11,
    )
    vstate_mh.parameters = vstate_seq.parameters
    vstate_mh.model_state = vstate_seq.model_state

    seq_stats = _stats_from_samples(vstate_seq, hamiltonian, seq_samples)
    mh_stats = vstate_mh.expect(hamiltonian)

    print(f"MH check ({model_kind}) L={length}")
    print(f"  samples={n_samples} sweeps={n_sweeps} burn_in={burn_in}")
    _print_stats("sequential", seq_stats)
    _print_stats("netket", mh_stats)
    print(f"  seq-netket diff {seq_stats.mean - mh_stats.mean}")
    if hasattr(vstate_mh.sampler_state, "acceptance"):
        print(f"  netket acceptance {float(vstate_mh.sampler_state.acceptance)}")


def _parse_int_list(values: Iterable[str]) -> list[int]:
    return [int(v) for v in values]


def _benchmark_sampling(
    *,
    model_kind: str,
    length: int,
    n_samples: int,
    n_sweeps: int,
    bond_dim: int,
    seed: int,
) -> None:
    hi, _, _ = build_heisenberg_square(length, pbc=False)
    n_sites = int(hi.size)
    random_sweeps = int(n_sweeps * n_sites)

    if model_kind == "mps":
        model = SimpleMPS(
            rngs=nnx.Rngs(seed),
            n_sites=n_sites,
            bond_dim=bond_dim,
        )
        sampler = nk.sampler.MetropolisLocal(hi, n_chains=1)
        vstate = nk.vqs.MCState(
            sampler,
            model,
            n_samples=n_samples,
            n_discard_per_chain=0,
        )
        key = jax.random.key(seed)

        _ = sequential_sample_mps(
            model,
            n_samples=n_samples,
            n_sweeps=n_sweeps,
            key=key,
        )
        _ = jax.block_until_ready(_)
        t0 = time.perf_counter()
        seq_samples = sequential_sample_mps(
            model,
            n_samples=n_samples,
            n_sweeps=n_sweeps,
            key=key,
        )
        _ = jax.block_until_ready(seq_samples)
        seq_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        rand_samples, _, _, accept = random_flip_sample(
            vstate,
            n_samples=n_samples,
            n_sweeps=random_sweeps,
            key=jax.random.key(seed + 1),
        )
        _ = jax.block_until_ready(rand_samples)
        rand_time = time.perf_counter() - t1
    elif model_kind == "peps":
        model = SimplePEPS(
            rngs=nnx.Rngs(seed),
            shape=(length, length),
            bond_dim=bond_dim,
        )
        sampler = nk.sampler.MetropolisLocal(hi, n_chains=1)
        vstate = nk.vqs.MCState(
            sampler,
            model,
            n_samples=n_samples,
            n_discard_per_chain=0,
        )
        key = jax.random.key(seed)

        warm_samples, _ = peps_sequential_sample(
            model,
            n_samples=n_samples,
            n_sweeps=n_sweeps,
            key=key,
        )
        _ = jax.block_until_ready(warm_samples)
        t0 = time.perf_counter()
        seq_samples, _ = peps_sequential_sample(
            model,
            n_samples=n_samples,
            n_sweeps=n_sweeps,
            key=key,
        )
        _ = jax.block_until_ready(seq_samples)
        seq_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        rand_samples, _, _, accept = random_flip_sample(
            vstate,
            n_samples=n_samples,
            n_sweeps=random_sweeps,
            key=jax.random.key(seed + 1),
        )
        _ = jax.block_until_ready(rand_samples)
        rand_time = time.perf_counter() - t1
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")

    print(f"Timing benchmark ({model_kind}) L={length}")
    print(f"  samples={n_samples} sweeps={n_sweeps} random_sweeps={random_sweeps}")
    print(f"  sequential time {seq_time:.3f}s")
    print(f"  random time     {rand_time:.3f}s")
    print(f"  random accept   {float(accept)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify sequential sampling against FullSum and NetKet MH.",
    )
    parser.add_argument(
        "--model",
        choices=("mps", "peps", "both"),
        default="peps",
        help="Which model family to verify.",
    )
    parser.add_argument(
        "--fullsum-size",
        type=int,
        default=3,
        help="Square lattice size for FullSum verification.",
    )
    parser.add_argument(
        "--mh-sizes",
        nargs="+",
        default=["5", "8", "10"],
        help="Square lattice sizes for NetKet MH comparisons.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=4096,
        help="Number of samples per check.",
    )
    parser.add_argument(
        "--sweeps",
        type=int,
        default=10,
        help="Number of sequential sweeps between samples.",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=200,
        help="Number of sequential sweeps used for burn-in.",
    )
    parser.add_argument(
        "--bond-dim",
        type=int,
        default=2,
        help="Bond dimension for the tensor network model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a timing benchmark comparing sequential vs random flips.",
    )
    parser.add_argument(
        "--benchmark-size",
        type=int,
        default=4,
        help="Square lattice size for timing benchmark.",
    )
    args = parser.parse_args()

    mh_sizes = _parse_int_list(args.mh_sizes)

    model_kinds = [args.model]
    if args.model == "both":
        model_kinds = ["mps", "peps"]

    for model_kind in model_kinds:
        _run_fullsum_check(
            model_kind=model_kind,
            length=args.fullsum_size,
            n_samples=args.nsamples,
            n_sweeps=args.sweeps,
            burn_in=args.burn_in,
            bond_dim=args.bond_dim,
            seed=args.seed,
        )
        for length in mh_sizes:
            _run_mh_check(
                model_kind=model_kind,
                length=length,
                n_samples=args.nsamples,
                n_sweeps=args.sweeps,
                burn_in=args.burn_in,
                bond_dim=args.bond_dim,
                seed=args.seed + length,
            )
        if args.benchmark:
            _benchmark_sampling(
                model_kind=model_kind,
                length=args.benchmark_size,
                n_samples=args.nsamples,
                n_sweeps=args.sweeps,
                bond_dim=args.bond_dim,
                seed=args.seed + 100,
            )


if __name__ == "__main__":
    main()
