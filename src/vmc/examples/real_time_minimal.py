from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import functools
import logging
import time

import jax
import jax.numpy as jnp
import netket as nk
from netket import stats as nkstats
from flax import nnx

from vmc.core import _value_and_grad
from vmc.drivers import DynamicsDriver, RealTimeUnit
from vmc.examples.real_time import build_heisenberg_square
from vmc.models.mps import MPS
from vmc.models.peps import PEPS
from vmc.preconditioners import SRPreconditioner
from vmc.samplers.sequential import sequential_sample_with_gradients
from vmc.utils.utils import occupancy_to_spin
from vmc.utils.vmc_utils import flatten_samples, model_params

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
    fullsum_state.parameters = model_params(model)
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


def random_flip_sample(
    model,
    *,
    n_samples: int,
    n_updates: int,
    key: jax.Array,
    init_samples: jax.Array | None = None,
    full_gradient: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array, jax.Array]:
    """Sample using random single-spin flips with Metropolis acceptance."""
    n_sites = int(model.n_sites if hasattr(model, "n_sites") else model.shape[0] * model.shape[1])

    if init_samples is None:
        key, subkey = jax.random.split(key)
        bits = jax.random.bernoulli(subkey, p=0.5, shape=(n_samples, n_sites))
        samples = occupancy_to_spin(bits)
    else:
        samples = init_samples

    def batch_value_and_grad(batch: jax.Array):
        return jax.vmap(
            lambda s: _value_and_grad(model, s, full_gradient=full_gradient)
        )(batch)

    amps, grads, p = batch_value_and_grad(samples)
    grads = grads / amps[:, None]

    accepts = []
    for _ in range(int(n_updates)):
        key, subkey = jax.random.split(key)
        flip_sites = jax.random.randint(subkey, (n_samples,), 0, n_sites)
        proposed = samples.at[jnp.arange(n_samples), flip_sites].set(
            -samples[jnp.arange(n_samples), flip_sites]
        )
        amps_prop, grads_prop, p_prop = batch_value_and_grad(proposed)
        grads_prop = grads_prop / amps_prop[:, None]

        weight_cur = jnp.abs(amps) ** 2
        weight_prop = jnp.abs(amps_prop) ** 2
        ratio = jnp.where(
            weight_cur > 0.0,
            weight_prop / weight_cur,
            jnp.where(weight_prop > 0.0, jnp.inf, 0.0),
        )
        key, subkey = jax.random.split(key)
        accept = jax.random.uniform(subkey, (n_samples,)) < jnp.minimum(1.0, ratio)
        accepts.append(accept)

        mask = accept[:, None]
        samples = jnp.where(mask, proposed, samples)
        amps = jnp.where(accept, amps_prop, amps)
        grads = jnp.where(mask, grads_prop, grads)
        if p is not None:
            p = jnp.where(mask, p_prop, p)

    acceptance = jnp.mean(jnp.stack(accepts, axis=0))
    return samples, grads, p, key, acceptance


def minimal_real_time_mcmc_demo(
    *,
    length: int = 4,
    bond_dim: int = 4,
    n_samples: int = 32,
    n_steps: int = 30,
    T: float = 0.30,
    diag_shift: float = 1e-3,
    seed: int = 0,
    dtype: jnp.dtype = jnp.complex128,
):
    """Minimal real-time TDVP loop with sequential sampling."""
    hi, H, _ = build_heisenberg_square(length, pbc=False)
    model = MPS(
        rngs=nnx.Rngs(seed),
        n_sites=hi.size,
        bond_dim=bond_dim,
        dtype=dtype,
    )
    dt = float(T) / float(n_steps)
    sampler = functools.partial(
        sequential_sample_with_gradients,
        n_samples=n_samples,
        burn_in=0,
        full_gradient=False,
    )
    preconditioner = SRPreconditioner(diag_shift=diag_shift)
    driver = DynamicsDriver(
        model,
        H,
        sampler=sampler,
        preconditioner=preconditioner,
        dt=dt,
        time_unit=RealTimeUnit(),
        sampler_key=jax.random.key(seed),
    )

    logger.info(
        "MCMC real-time demo: L=%d bond_dim=%d samples=%d dt=%.3e",
        length,
        bond_dim,
        n_samples,
        dt,
    )

    sample_time_total = 0.0
    for step in range(n_steps):
        t_start = time.perf_counter()
        driver.step()
        sample_time = time.perf_counter() - t_start
        sample_time_total += sample_time
        stats = driver.energy
        _maybe_log_fullsum_check(hi, H, model, stats, step=step)
        if driver.last_samples is not None:
            _log_sampling_preview(step, driver.last_samples, None)
        logger.info(
            "t=%.3e energy=%s sample_time=%.3e",
            driver.t,
            complex(stats.mean),
            sample_time,
        )

    logger.info("t=%.3e energy=%s", n_steps * dt, complex(driver.energy.mean))
    logger.info("avg_sample_time=%.3e", sample_time_total / n_steps)
    return model


def minimal_real_time_random_flip_demo(
    *,
    length: int = 4,
    bond_dim: int = 4,
    n_samples: int = 512,
    n_steps: int = 30,
    T: float = 0.30,
    n_updates: int | None = None,
    diag_shift: float = 1e-3,
    seed: int = 2,
    dtype: jnp.dtype = jnp.complex128,
):
    """Minimal real-time TDVP loop with random-flip Metropolis sampling."""
    hi, H, _ = build_heisenberg_square(length, pbc=False)
    model = MPS(
        rngs=nnx.Rngs(seed),
        n_sites=hi.size,
        bond_dim=bond_dim,
        dtype=dtype,
    )
    dt = float(T) / float(n_steps)
    if n_updates is None:
        n_updates = 4 * hi.size
    sampler = functools.partial(
        random_flip_sample,
        n_samples=n_samples,
        n_updates=n_updates,
        full_gradient=False,
    )
    preconditioner = SRPreconditioner(diag_shift=diag_shift)
    driver = DynamicsDriver(
        model,
        H,
        sampler=sampler,
        preconditioner=preconditioner,
        dt=dt,
        time_unit=RealTimeUnit(),
        sampler_key=jax.random.key(seed),
    )

    logger.info(
        "Random-flip real-time demo: L=%d bond_dim=%d samples=%d updates=%d dt=%.3e",
        length,
        bond_dim,
        n_samples,
        n_updates,
        dt,
    )

    accept_total = 0.0
    for step in range(n_steps):
        driver.step()
        stats = driver.energy
        acceptance = driver.last_sampler_info
        if acceptance is not None:
            accept_total += float(acceptance)
        _maybe_log_fullsum_check(hi, H, model, stats, step=step)
        if driver.last_samples is not None:
            _log_sampling_preview(
                step, driver.last_samples, None, acceptance=acceptance
            )
        logger.info(
            "t=%.3e energy=%s acceptance=%.3f",
            driver.t,
            complex(stats.mean),
            float(acceptance) if acceptance is not None else float("nan"),
        )

    logger.info("t=%.3e energy=%s", n_steps * dt, complex(driver.energy.mean))
    logger.info("avg_acceptance=%.3f", accept_total / n_steps)
    return model


def minimal_real_time_sequential_demo(
    *,
    length: int = 4,
    bond_dim: int = 4,
    n_samples: int = 32,
    n_steps: int = 30,
    T: float = 0.30,
    diag_shift: float = 1e-3,
    seed: int = 1,
    dtype: jnp.dtype = jnp.complex128,
):
    """Minimal real-time TDVP loop with sequential Metropolis sampling."""
    hi, H, _ = build_heisenberg_square(length, pbc=False)
    model = MPS(
        rngs=nnx.Rngs(seed),
        n_sites=hi.size,
        bond_dim=bond_dim,
        dtype=dtype,
    )
    dt = float(T) / float(n_steps)
    sampler = functools.partial(
        sequential_sample_with_gradients,
        n_samples=n_samples,
        burn_in=0,
        full_gradient=False,
    )
    preconditioner = SRPreconditioner(diag_shift=diag_shift)
    driver = DynamicsDriver(
        model,
        H,
        sampler=sampler,
        preconditioner=preconditioner,
        dt=dt,
        time_unit=RealTimeUnit(),
        sampler_key=jax.random.key(seed),
    )

    logger.info(
        "Sequential real-time demo: L=%d bond_dim=%d samples=%d dt=%.3e",
        length,
        bond_dim,
        n_samples,
        dt,
    )

    for step in range(n_steps):
        driver.step()
        stats = driver.energy
        _maybe_log_fullsum_check(hi, H, model, stats, step=step)
        logger.info("t=%.3e energy=%s", driver.t, complex(stats.mean))

    logger.info("t=%.3e energy=%s", n_steps * dt, complex(driver.energy.mean))
    return model


def minimal_real_time_peps_random_flip_demo(
    *,
    length: int = 4,
    bond_dim: int = 3,
    n_samples: int = 128,
    n_steps: int = 30,
    T: float = 0.30,
    n_updates: int | None = None,
    diag_shift: float = 1e-3,
    seed: int = 3,
    dtype: jnp.dtype = jnp.complex128,
):
    """Minimal real-time TDVP loop for PEPS with random-flip Metropolis sampling."""
    hi, H, _ = build_heisenberg_square(length, pbc=False)
    model = PEPS(
        rngs=nnx.Rngs(seed),
        shape=(length, length),
        bond_dim=bond_dim,
        dtype=dtype,
    )
    dt = float(T) / float(n_steps)
    if n_updates is None:
        n_updates = 4 * hi.size
    sampler = functools.partial(
        random_flip_sample,
        n_samples=n_samples,
        n_updates=n_updates,
        full_gradient=False,
    )
    preconditioner = SRPreconditioner(diag_shift=diag_shift)
    driver = DynamicsDriver(
        model,
        H,
        sampler=sampler,
        preconditioner=preconditioner,
        dt=dt,
        time_unit=RealTimeUnit(),
        sampler_key=jax.random.key(seed),
    )

    logger.info(
        "PEPS random-flip demo: L=%d bond_dim=%d samples=%d updates=%d dt=%.3e",
        length,
        bond_dim,
        n_samples,
        n_updates,
        dt,
    )

    accept_total = 0.0
    for step in range(n_steps):
        driver.step()
        stats = driver.energy
        acceptance = driver.last_sampler_info
        if acceptance is not None:
            accept_total += float(acceptance)
        _maybe_log_fullsum_check(hi, H, model, stats, step=step)
        if driver.last_samples is not None:
            _log_sampling_preview(
                step, driver.last_samples, None, acceptance=acceptance
            )
        logger.info(
            "t=%.3e energy=%s acceptance=%.3f",
            driver.t,
            complex(stats.mean),
            float(acceptance) if acceptance is not None else float("nan"),
        )

    logger.info("t=%.3e energy=%s", n_steps * dt, complex(driver.energy.mean))
    logger.info("avg_acceptance=%.3f", accept_total / n_steps)
    return model


def minimal_real_time_peps_sequential_demo(
    *,
    length: int = 4,
    bond_dim: int = 3,
    n_samples: int = 8,
    n_steps: int = 30,
    T: float = 0.30,
    diag_shift: float = 1e-3,
    seed: int = 4,
    dtype: jnp.dtype = jnp.complex128,
):
    """Minimal real-time TDVP loop for PEPS with sequential sampling."""
    hi, H, _ = build_heisenberg_square(length, pbc=False)
    model = PEPS(
        rngs=nnx.Rngs(seed),
        shape=(length, length),
        bond_dim=bond_dim,
        dtype=dtype,
    )
    dt = float(T) / float(n_steps)
    sampler = functools.partial(
        sequential_sample_with_gradients,
        n_samples=n_samples,
        burn_in=0,
        full_gradient=False,
    )
    preconditioner = SRPreconditioner(diag_shift=diag_shift)
    driver = DynamicsDriver(
        model,
        H,
        sampler=sampler,
        preconditioner=preconditioner,
        dt=dt,
        time_unit=RealTimeUnit(),
        sampler_key=jax.random.key(seed),
    )

    logger.info(
        "PEPS sequential demo: L=%d bond_dim=%d samples=%d dt=%.3e",
        length,
        bond_dim,
        n_samples,
        dt,
    )

    for step in range(n_steps):
        driver.step()
        stats = driver.energy
        _maybe_log_fullsum_check(hi, H, model, stats, step=step)
        logger.info("t=%.3e energy=%s", driver.t, complex(stats.mean))

    logger.info("t=%.3e energy=%s", n_steps * dt, complex(driver.energy.mean))
    return model


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
