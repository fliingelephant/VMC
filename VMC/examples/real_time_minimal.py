from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import logging
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import netket as nk
from netket import stats as nkstats
from netket.operator import AbstractOperator
from flax import nnx

from VMC.drivers.custom_driver import CustomTDVP_SR, RealTime
from VMC.models.mps import SimpleMPS
from VMC.models.peps import SimplePEPS
from VMC.preconditioners import SRPreconditioner
from VMC.samplers.sequential import sequential_sample
from VMC.examples.real_time import build_heisenberg_square
from VMC.utils.vmc_utils import flatten_samples, get_apply_fun

logger = logging.getLogger(__name__)


def _log_sampling_preview(
    step: int,
    samples: jax.Array,
    sampler_state,
    *,
    acceptance: float | None = None,
    preview_count: int = 3,
) -> None:
    """Log a compact sampling preview for demos."""
    if not logger.isEnabledFor(logging.INFO):
        return
    flat = flatten_samples(samples)
    preview_len = min(preview_count, int(flat.shape[0]))
    preview = jnp.asarray(flat[:preview_len])
    if acceptance is None and sampler_state is not None and hasattr(
        sampler_state, "acceptance"
    ):
        acceptance = sampler_state.acceptance
    if acceptance is not None:
        acceptance = float(acceptance)
    logger.info(
        "step=%d samples_shape=%s acceptance=%s preview=%s",
        step,
        samples.shape,
        acceptance,
        preview,
    )


def _maybe_log_fullsum_check(
    hi,
    hamiltonian,
    model,
    params: dict,
    model_state: dict,
    sample_stats: nkstats.Statistics,
    *,
    step: int,
    max_states: int = 512,
) -> None:
    """Compare sampled energy with a full-sum estimate when feasible."""
    if step != 0:
        return
    try:
        n_states = int(hi.n_states)
    except Exception:  # pragma: no cover - defensive for non-finite Hilbert spaces.
        n_states = None
    if n_states is None or n_states > max_states:
        logger.info("fullsum_check skipped: n_states=%s", n_states)
        return
    fullsum_state = nk.vqs.FullSumState(hi, model)
    fullsum_state.parameters = params
    if model_state:
        fullsum_state.model_state = model_state
    energy_fullsum = fullsum_state.expect(hamiltonian).mean
    diff = sample_stats.mean - energy_fullsum
    logger.info(
        "fullsum_check step=%d n_states=%s energy_sample=%s energy_fullsum=%s diff=%s err=%s",
        step,
        n_states,
        complex(sample_stats.mean),
        complex(energy_fullsum),
        complex(diff),
        sample_stats.error_of_mean,
    )


def _set_parameters_no_reset(vstate: nk.vqs.MCState, params: dict) -> None:
    """Update parameters without resetting cached samples (RK stages reuse samples)."""
    vstate._parameters = params  # pylint: disable=protected-access


def _time_derivative_with_samples(
    vstate: nk.vqs.MCState,
    hamiltonian,
    preconditioner: SRPreconditioner,
    params: dict,
    t: float,
    *,
    step: int,
    grad_factor: complex,
    stage: int,
) -> tuple[dict, nkstats.Statistics | None]:
    _set_parameters_no_reset(vstate, params)
    if callable(hamiltonian) and not isinstance(hamiltonian, AbstractOperator):
        op_t = hamiltonian(t)
    else:
        op_t = hamiltonian
    local_energies = vstate.local_estimators(op_t)
    stats = nkstats.statistics(local_energies) if stage == 0 else None
    updates = preconditioner.apply(
        vstate,
        local_energies,
        step=step,
        grad_factor=grad_factor,
        stage=stage,
    )
    return updates, stats


def _rk4_update_with_samples(
    vstate: nk.vqs.MCState,
    hamiltonian,
    preconditioner: SRPreconditioner,
    params: dict,
    t: float,
    dt: float,
    *,
    step: int,
    grad_factor: complex,
) -> tuple[dict, nkstats.Statistics]:
    k1, stats = _time_derivative_with_samples(
        vstate,
        hamiltonian,
        preconditioner,
        params,
        t,
        step=step,
        grad_factor=grad_factor,
        stage=0,
    )
    params_k2 = CustomTDVP_SR._tree_add_scaled(params, k1, 0.5 * dt)
    k2, _ = _time_derivative_with_samples(
        vstate,
        hamiltonian,
        preconditioner,
        params_k2,
        t + 0.5 * dt,
        step=step,
        grad_factor=grad_factor,
        stage=1,
    )
    params_k3 = CustomTDVP_SR._tree_add_scaled(params, k2, 0.5 * dt)
    k3, _ = _time_derivative_with_samples(
        vstate,
        hamiltonian,
        preconditioner,
        params_k3,
        t + 0.5 * dt,
        step=step,
        grad_factor=grad_factor,
        stage=2,
    )
    params_k4 = CustomTDVP_SR._tree_add_scaled(params, k3, dt)
    k4, _ = _time_derivative_with_samples(
        vstate,
        hamiltonian,
        preconditioner,
        params_k4,
        t + dt,
        step=step,
        grad_factor=grad_factor,
        stage=3,
    )
    incr = CustomTDVP_SR._tree_weighted_sum(k1, k2, k3, k4)
    params_new = CustomTDVP_SR._tree_add_scaled(params, incr, dt)
    if stats is None:
        raise RuntimeError("RK4 stage-0 statistics missing.")
    return params_new, stats


def random_flip_sample(
    vstate: nk.vqs.MCState,
    *,
    n_samples: int,
    n_sweeps: int,
    key: jax.Array,
    init_samples: jax.Array | None = None,
    _value_and_grad: Callable[[jax.Array], tuple[jax.Array, Any]] | None = None,
    return_gradients: bool = False,
) -> (
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, Any, jax.Array, jax.Array]
):
    """Sample using random single-spin flips with Metropolis acceptance."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if n_sweeps <= 0:
        raise ValueError("n_sweeps must be positive.")
    if return_gradients and _value_and_grad is None:
        raise ValueError("return_gradients requires _value_and_grad.")

    apply_fun, params, model_state, kwargs = get_apply_fun(vstate)
    n_sites = int(vstate.hilbert.size)

    def logpsi_batch(samples: jax.Array) -> jax.Array:
        return jax.vmap(
            lambda s: apply_fun({"params": params, **model_state}, s, **kwargs)
        )(samples)

    if init_samples is None:
        key, subkey = jax.random.split(key)
        bits = jax.random.bernoulli(subkey, p=0.5, shape=(n_samples, n_sites))
        samples = 2 * bits.astype(jnp.int32) - 1
    else:
        samples = init_samples
        if samples.shape != (n_samples, n_sites):
            raise ValueError(
                f"init_samples must have shape {(n_samples, n_sites)}, got {samples.shape}"
            )

    if _value_and_grad is None:
        logpsi = logpsi_batch(samples)
        grads = None
    else:
        value, grads = _value_and_grad(samples)
        logpsi = jnp.log(value)
        grads = jax.tree_util.tree_map(lambda g: g / value, grads)
    batch_idx = jnp.arange(n_samples)

    def _where_accept(
        accept: jax.Array, proposed: jax.Array, current: jax.Array
    ) -> jax.Array:
        mask = accept.reshape((accept.shape[0],) + (1,) * (proposed.ndim - 1))
        return jnp.where(mask, proposed, current)

    def sweep(carry, _):
        if return_gradients:
            samples, logpsi, grads, key = carry
        else:
            samples, logpsi, key = carry
        key, key_site, key_u = jax.random.split(key, 3)
        flip_sites = jax.random.randint(key_site, (n_samples,), 0, n_sites)
        proposed = samples.at[batch_idx, flip_sites].set(
            -samples[batch_idx, flip_sites]
        )
        if _value_and_grad is None:
            logpsi_prop = logpsi_batch(proposed)
            grads_prop = None
        else:
            value_prop, grads_prop = _value_and_grad(proposed)
            logpsi_prop = jnp.log(value_prop)
            grads_prop = jax.tree_util.tree_map(lambda g: g / value_prop, grads_prop)
        log_ratio = 2.0 * jnp.real(logpsi_prop - logpsi)
        accept = jax.random.uniform(key_u, (n_samples,)) < jnp.exp(
            jnp.minimum(0.0, log_ratio)
        )
        samples = jnp.where(accept[:, None], proposed, samples)
        logpsi = jnp.where(accept, logpsi_prop, logpsi)
        if return_gradients:
            grads = jax.tree_util.tree_map(
                lambda new, old: _where_accept(accept, new, old),
                grads_prop,
                grads,
            )
            return (samples, logpsi, grads, key), accept
        return (samples, logpsi, key), accept

    if return_gradients:
        (samples, logpsi, grads, key), accepts = jax.lax.scan(
            sweep, (samples, logpsi, grads, key), None, length=int(n_sweeps)
        )
    else:
        (samples, logpsi, key), accepts = jax.lax.scan(
            sweep, (samples, logpsi, key), None, length=int(n_sweeps)
        )
    acceptance = jnp.mean(accepts)
    if return_gradients:
        return samples, logpsi, grads, key, acceptance
    return samples, logpsi, key, acceptance


def minimal_real_time_mcmc_demo(
    *,
    length: int = 4,
    bond_dim: int = 4,
    n_samples: int = 512,
    n_steps: int = 30,
    T: float = 0.30,
    n_chains: int = 8,
    diag_shift: float = 1e-3,
    seed: int = 0,
    dtype: jnp.dtype = jnp.complex128,
):
    """Minimal real-time TDVP loop with a built-in MCMC sampler."""
    hi, H, _ = build_heisenberg_square(length, pbc=False)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
    model = SimpleMPS(
        rngs=nnx.Rngs(seed),
        n_sites=hi.size,
        bond_dim=bond_dim,
        dtype=dtype,
    )
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=8,
    )
    dt = float(T) / float(n_steps)
    preconditioner = SRPreconditioner(diag_shift=diag_shift)
    grad_factor = RealTime().grad_factor

    logger.info(
        "MCMC real-time demo: L=%d bond_dim=%d samples=%d dt=%.3e",
        length,
        bond_dim,
        n_samples,
        dt,
    )

    sample_time_total = 0.0
    params = vstate.parameters
    model_state = vstate.model_state
    t = 0.0
    for step in range(n_steps):
        t_start = time.perf_counter()
        samples = vstate.samples
        sample_time = time.perf_counter() - t_start
        sample_time_total += sample_time
        _log_sampling_preview(step, samples, vstate.sampler_state)

        params_new, stats = _rk4_update_with_samples(
            vstate,
            H,
            preconditioner,
            params,
            t,
            dt,
            step=step,
            grad_factor=grad_factor,
        )
        _maybe_log_fullsum_check(
            hi,
            H,
            vstate.model,
            params,
            model_state,
            stats,
            step=step,
        )
        energy = stats.mean
        logger.info(
            "t=%.3e energy=%s sample_time=%.3e", t, complex(energy), sample_time
        )
        vstate.parameters = params_new
        params = params_new
        t += dt

    final_energy = vstate.expect(H).mean
    logger.info("t=%.3e energy=%s", n_steps * dt, complex(final_energy))
    logger.info("avg_sample_time=%.3e", sample_time_total / n_steps)
    return vstate


def minimal_real_time_random_flip_demo(
    *,
    length: int = 4,
    bond_dim: int = 4,
    n_samples: int = 512,
    n_steps: int = 30,
    T: float = 0.30,
    n_sweeps: int | None = None,
    diag_shift: float = 1e-3,
    seed: int = 2,
    dtype: jnp.dtype = jnp.complex128,
):
    """Minimal real-time TDVP loop with random-flip Metropolis sampling."""
    hi, H, _ = build_heisenberg_square(length, pbc=False)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=1)
    model = SimpleMPS(
        rngs=nnx.Rngs(seed),
        n_sites=hi.size,
        bond_dim=bond_dim,
        dtype=dtype,
    )
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=0,
    )
    dt = float(T) / float(n_steps)
    preconditioner = SRPreconditioner(diag_shift=diag_shift)
    grad_factor = RealTime().grad_factor
    n_sites = hi.size
    if n_sweeps is None:
        n_sweeps = 4 * n_sites

    logger.info(
        "Random-flip real-time demo: L=%d bond_dim=%d samples=%d sweeps=%d dt=%.3e",
        length,
        bond_dim,
        n_samples,
        n_sweeps,
        dt,
    )

    key = jax.random.key(seed)
    flat_samples = None
    n_chains = int(vstate.sampler.n_chains)
    chain_length = int(vstate.chain_length)
    sample_time_total = 0.0
    accept_total = 0.0
    params = vstate.parameters
    model_state = vstate.model_state
    t = 0.0

    for step in range(n_steps):
        t_start = time.perf_counter()
        flat_samples, _, key, acceptance = random_flip_sample(
            vstate,
            n_samples=vstate.n_samples,
            n_sweeps=n_sweeps,
            key=key,
            init_samples=flat_samples,
        )
        sample_time = time.perf_counter() - t_start
        sample_time_total += sample_time
        accept_total += float(acceptance)

        chain_samples = flat_samples.reshape(n_chains, chain_length, n_sites)
        vstate._samples = chain_samples
        _log_sampling_preview(step, chain_samples, None, acceptance=acceptance)

        params_new, stats = _rk4_update_with_samples(
            vstate,
            H,
            preconditioner,
            params,
            t,
            dt,
            step=step,
            grad_factor=grad_factor,
        )
        _maybe_log_fullsum_check(
            hi,
            H,
            vstate.model,
            params,
            model_state,
            stats,
            step=step,
        )
        energy = stats.mean
        logger.info(
            "t=%.3e energy=%s sample_time=%.3e acceptance=%.3f",
            t,
            complex(energy),
            sample_time,
            float(acceptance),
        )
        vstate.parameters = params_new
        params = params_new
        t += dt

    final_energy = vstate.expect(H).mean
    logger.info("t=%.3e energy=%s", n_steps * dt, complex(final_energy))
    logger.info(
        "avg_sample_time=%.3e avg_acceptance=%.3f",
        sample_time_total / n_steps,
        accept_total / n_steps,
    )
    return vstate


def minimal_real_time_sequential_demo(
    *,
    length: int = 4,
    bond_dim: int = 4,
    n_samples: int = 512,
    n_steps: int = 30,
    T: float = 0.30,
    n_sweeps: int = 1,
    diag_shift: float = 1e-3,
    seed: int = 1,
    dtype: jnp.dtype = jnp.complex128,
):
    """Minimal real-time TDVP loop with sequential Metropolis sampling."""
    hi, H, _ = build_heisenberg_square(length, pbc=False)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=1)
    model = SimpleMPS(
        rngs=nnx.Rngs(seed),
        n_sites=hi.size,
        bond_dim=bond_dim,
        dtype=dtype,
    )
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=0,
    )
    dt = float(T) / float(n_steps)
    preconditioner = SRPreconditioner(diag_shift=diag_shift)
    grad_factor = RealTime().grad_factor

    logger.info(
        "Sequential real-time demo: L=%d bond_dim=%d samples=%d sweeps=%d dt=%.3e",
        length,
        bond_dim,
        n_samples,
        n_sweeps,
        dt,
    )

    key = jax.random.key(seed)
    n_sites = hi.size
    n_chains = int(vstate.sampler.n_chains)
    chain_length = int(vstate.chain_length)
    sample_time_total = 0.0
    params = vstate.parameters
    model_state = vstate.model_state
    t = 0.0

    for step in range(n_steps):
        t_start = time.perf_counter()
        key, subkey = jax.random.split(key)
        seq_samples = sequential_sample(
            vstate.model,
            n_samples=vstate.n_samples,
            key=subkey,
            n_sweeps=n_sweeps,
        )
        sample_time = time.perf_counter() - t_start
        sample_time_total += sample_time
        seq_samples = seq_samples.reshape(n_chains, chain_length, n_sites)
        # Inject samples for the demo; avoids implementing a custom sampler.
        vstate._samples = seq_samples
        _log_sampling_preview(step, seq_samples, None)

        params_new, stats = _rk4_update_with_samples(
            vstate,
            H,
            preconditioner,
            params,
            t,
            dt,
            step=step,
            grad_factor=grad_factor,
        )
        _maybe_log_fullsum_check(
            hi,
            H,
            vstate.model,
            params,
            model_state,
            stats,
            step=step,
        )
        energy = stats.mean
        logger.info(
            "t=%.3e energy=%s sample_time=%.3e", t, complex(energy), sample_time
        )
        vstate.parameters = params_new
        params = params_new
        t += dt

    final_energy = vstate.expect(H).mean
    logger.info("t=%.3e energy=%s", n_steps * dt, complex(final_energy))
    logger.info("avg_sample_time=%.3e", sample_time_total / n_steps)
    return vstate


def minimal_real_time_peps_random_flip_demo(
    *,
    length: int = 4,
    bond_dim: int = 3,
    n_samples: int = 128,
    n_steps: int = 30,
    T: float = 0.30,
    n_sweeps: int | None = None,
    diag_shift: float = 1e-3,
    seed: int = 3,
    dtype: jnp.dtype = jnp.complex128,
):
    """Minimal real-time TDVP loop for PEPS with random-flip Metropolis sampling."""
    hi, H, _ = build_heisenberg_square(length, pbc=False)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=1)
    model = SimplePEPS(
        rngs=nnx.Rngs(seed),
        shape=(length, length),
        bond_dim=bond_dim,
        dtype=dtype,
    )
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=0,
    )
    dt = float(T) / float(n_steps)
    preconditioner = SRPreconditioner(diag_shift=diag_shift)
    grad_factor = RealTime().grad_factor
    n_sites = hi.size
    if n_sweeps is None:
        n_sweeps = 4 * n_sites

    logger.info(
        "PEPS random-flip demo: L=%d bond_dim=%d samples=%d sweeps=%d dt=%.3e",
        length,
        bond_dim,
        n_samples,
        n_sweeps,
        dt,
    )

    key = jax.random.key(seed)
    flat_samples = None
    n_chains = int(vstate.sampler.n_chains)
    chain_length = int(vstate.chain_length)
    sample_time_total = 0.0
    accept_total = 0.0
    params = vstate.parameters
    model_state = vstate.model_state
    t = 0.0

    for step in range(n_steps):
        t_start = time.perf_counter()
        flat_samples, _, key, acceptance = random_flip_sample(
            vstate,
            n_samples=vstate.n_samples,
            n_sweeps=n_sweeps,
            key=key,
            init_samples=flat_samples,
        )
        sample_time = time.perf_counter() - t_start
        sample_time_total += sample_time
        accept_total += float(acceptance)

        chain_samples = flat_samples.reshape(n_chains, chain_length, n_sites)
        vstate._samples = chain_samples
        _log_sampling_preview(step, chain_samples, None, acceptance=acceptance)

        params_new, stats = _rk4_update_with_samples(
            vstate,
            H,
            preconditioner,
            params,
            t,
            dt,
            step=step,
            grad_factor=grad_factor,
        )
        _maybe_log_fullsum_check(
            hi,
            H,
            vstate.model,
            params,
            model_state,
            stats,
            step=step,
        )
        energy = stats.mean
        logger.info(
            "t=%.3e energy=%s sample_time=%.3e acceptance=%.3f",
            t,
            complex(energy),
            sample_time,
            float(acceptance),
        )
        vstate.parameters = params_new
        params = params_new
        t += dt

    final_energy = vstate.expect(H).mean
    logger.info("t=%.3e energy=%s", n_steps * dt, complex(final_energy))
    logger.info(
        "avg_sample_time=%.3e avg_acceptance=%.3f",
        sample_time_total / n_steps,
        accept_total / n_steps,
    )
    return vstate


def minimal_real_time_peps_sequential_demo(
    *,
    length: int = 4,
    bond_dim: int = 3,
    n_samples: int = 128,
    n_steps: int = 30,
    T: float = 0.30,
    n_sweeps: int = 1,
    diag_shift: float = 1e-3,
    seed: int = 4,
    dtype: jnp.dtype = jnp.complex128,
):
    """Minimal real-time TDVP loop for PEPS with sequential Metropolis sampling."""
    hi, H, _ = build_heisenberg_square(length, pbc=False)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=1)
    model = SimplePEPS(
        rngs=nnx.Rngs(seed),
        shape=(length, length),
        bond_dim=bond_dim,
        dtype=dtype,
    )
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=0,
    )
    dt = float(T) / float(n_steps)
    preconditioner = SRPreconditioner(diag_shift=diag_shift)
    grad_factor = RealTime().grad_factor
    n_sites = hi.size

    logger.info(
        "PEPS sequential demo: L=%d bond_dim=%d samples=%d sweeps=%d dt=%.3e",
        length,
        bond_dim,
        n_samples,
        n_sweeps,
        dt,
    )

    key = jax.random.key(seed)
    n_chains = int(vstate.sampler.n_chains)
    chain_length = int(vstate.chain_length)
    sample_time_total = 0.0
    params = vstate.parameters
    model_state = vstate.model_state
    t = 0.0

    for step in range(n_steps):
        t_start = time.perf_counter()
        flat_samples, key = sequential_sample(
            vstate.model,
            n_samples=vstate.n_samples,
            key=key,
            n_sweeps=n_sweeps,
            return_key=True,
        )
        sample_time = time.perf_counter() - t_start
        sample_time_total += sample_time

        chain_samples = flat_samples.reshape(n_chains, chain_length, n_sites)
        vstate._samples = chain_samples
        _log_sampling_preview(step, chain_samples, None)

        params_new, stats = _rk4_update_with_samples(
            vstate,
            H,
            preconditioner,
            params,
            t,
            dt,
            step=step,
            grad_factor=grad_factor,
        )
        _maybe_log_fullsum_check(
            hi,
            H,
            vstate.model,
            params,
            model_state,
            stats,
            step=step,
        )
        energy = stats.mean
        logger.info(
            "t=%.3e energy=%s sample_time=%.3e", t, complex(energy), sample_time
        )
        vstate.parameters = params_new
        params = params_new
        t += dt

    final_energy = vstate.expect(H).mean
    logger.info("t=%.3e energy=%s", n_steps * dt, complex(final_energy))
    logger.info("avg_sample_time=%.3e", sample_time_total / n_steps)
    return vstate


def run_mps_demos() -> None:
    """Run the MPS demo suite."""
    minimal_real_time_mcmc_demo()
    minimal_real_time_random_flip_demo()
    minimal_real_time_sequential_demo()


def run_peps_demos() -> None:
    """Run the PEPS demo suite."""
    minimal_real_time_peps_random_flip_demo()
    minimal_real_time_peps_sequential_demo()


def main() -> None:
    run_mps_demos()
    run_peps_demos()


if __name__ == "__main__":
    main()
