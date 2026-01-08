"""Independent-set samplers for blockade constraints."""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import logging
from typing import Callable, Sequence

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

__all__ = [
    "IndependentSetSampler",
    "build_neighbor_arrays",
    "independent_set_violations",
    "occupancy_to_spin",
    "spin_to_occupancy",
]


def build_neighbor_arrays(
    num_sites: int, edges: Sequence[tuple[int, int]]
) -> tuple[jax.Array, jax.Array]:
    """Build padded neighbor arrays from an undirected edge list.

    Args:
        num_sites: Number of vertices in the graph.
        edges: Iterable of undirected edges (u, v) with 0-based indices.

    Returns:
        Tuple of (neighbors, mask) arrays with shape (num_sites, max_degree).
    """
    if num_sites <= 0:
        raise ValueError("num_sites must be positive.")
    adjacency = [set() for _ in range(num_sites)]
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"Edge must have two entries, got {edge}.")
        u, v = edge
        if u == v:
            raise ValueError(f"Self-loop detected at {u}.")
        if not (0 <= u < num_sites and 0 <= v < num_sites):
            raise ValueError(f"Edge {edge} out of bounds for num_sites={num_sites}.")
        adjacency[u].add(v)
        adjacency[v].add(u)

    max_degree = max((len(neigh) for neigh in adjacency), default=0)
    neighbors = jnp.zeros((num_sites, max_degree), dtype=jnp.int32)
    mask = jnp.zeros((num_sites, max_degree), dtype=jnp.bool_)
    if max_degree == 0:
        return neighbors, mask

    neighbor_buffer = []
    mask_buffer = []
    for neigh in adjacency:
        padded = sorted(neigh) + [0] * (max_degree - len(neigh))
        neighbor_buffer.append(padded)
        mask_buffer.append([True] * len(neigh) + [False] * (max_degree - len(neigh)))

    neighbors = jnp.asarray(neighbor_buffer, dtype=jnp.int32)
    mask = jnp.asarray(mask_buffer, dtype=jnp.bool_)
    return neighbors, mask


def occupancy_to_spin(occupancies: jax.Array) -> jax.Array:
    """Convert 0/1 occupancy variables into -1/+1 spins."""
    occupancies = jnp.asarray(occupancies)
    return 2 * occupancies - 1


def spin_to_occupancy(spins: jax.Array) -> jax.Array:
    """Convert -1/+1 spins into 0/1 occupancy variables."""
    spins = jnp.asarray(spins)
    return ((spins + 1) // 2).astype(jnp.int32)


def _blocked_counts(
    samples: jax.Array, neighbors: jax.Array, mask: jax.Array
) -> jax.Array:
    """Count occupied neighbors for each site."""
    safe_neighbors = jnp.where(mask, neighbors, 0)
    neighbor_occ = jnp.take(samples, safe_neighbors, axis=-1)
    return jnp.sum(neighbor_occ * mask, axis=-1)


def independent_set_violations(
    samples: jax.Array, neighbors: jax.Array, mask: jax.Array
) -> jax.Array:
    """Check for independent-set violations in a batch of samples."""
    blocked = _blocked_counts(samples, neighbors, mask)
    return jnp.any((samples == 1) & (blocked > 0), axis=-1)


class IndependentSetSampler:
    """Standalone Metropolis-Hastings sampler for independent sets."""

    def __init__(self, num_sites: int, edges: Sequence[tuple[int, int]]):
        if num_sites <= 0:
            raise ValueError("num_sites must be positive.")
        self._num_sites = int(num_sites)
        self._neighbors, self._mask = build_neighbor_arrays(num_sites, edges)

    @property
    def num_sites(self) -> int:
        return self._num_sites

    def sample(
        self,
        log_prob_fn: Callable[[jax.Array], jax.Array] | None,
        *,
        n_samples: int,
        n_sweeps: int,
        key: jax.Array,
        init_samples: jax.Array | None = None,
        log_prob_is_batched: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Run blocked-aware Metropolis-Hastings sweeps.

        Args:
            log_prob_fn: Callable returning log-probabilities for a batch of samples.
                Use log |psi|^2 for VMC. If None, samples uniformly over independent sets.
            n_samples: Number of parallel chains.
            n_sweeps: Number of Metropolis sweeps to run.
            key: JAX PRNG key.
            init_samples: Optional initial samples with shape (n_samples, num_sites).
            log_prob_is_batched: If False, log_prob_fn is vmapped over samples.

        Returns:
            Tuple of (samples, log_prob, key, acceptance).
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if n_sweeps <= 0:
            raise ValueError("n_sweeps must be positive.")

        if log_prob_fn is None:
            def _zero_log_prob(batch_samples: jax.Array) -> jax.Array:
                return jnp.zeros((batch_samples.shape[0],), dtype=jnp.float64)

            log_prob_fn = _zero_log_prob
            log_prob_is_batched = True
        if not log_prob_is_batched:
            log_prob_fn = jax.vmap(log_prob_fn)

        if init_samples is None:
            samples = jnp.zeros((n_samples, self._num_sites), dtype=jnp.int32)
        else:
            samples = jnp.asarray(init_samples, dtype=jnp.int32)
            if samples.ndim == 1:
                samples = jnp.tile(samples[None, :], (n_samples, 1))
            if samples.shape != (n_samples, self._num_sites):
                raise ValueError(
                    "init_samples must have shape "
                    f"({n_samples}, {self._num_sites}), got {samples.shape}."
                )

        violations = independent_set_violations(samples, self._neighbors, self._mask)
        if bool(jnp.any(violations)):
            raise ValueError("init_samples must contain only independent sets.")

        log_prob = jnp.real(jnp.asarray(log_prob_fn(samples), dtype=jnp.float64))
        if log_prob.shape != (n_samples,):
            raise ValueError(
                f"log_prob_fn must return shape ({n_samples},), got {log_prob.shape}."
            )

        batch_idx = jnp.arange(n_samples)

        def flippable_mask(batch_samples: jax.Array) -> jax.Array:
            blocked = _blocked_counts(batch_samples, self._neighbors, self._mask)
            return (batch_samples == 1) | ((batch_samples == 0) & (blocked == 0))

        def sweep(carry, _):
            batch_samples, batch_log_prob, key_in = carry
            key_out, key_site, key_u = jax.random.split(key_in, 3)

            flippable = flippable_mask(batch_samples)
            n_flippable = jnp.sum(flippable, axis=-1).astype(jnp.float64)
            logits = jnp.where(flippable, 0.0, -jnp.inf)
            site_idx = jax.random.categorical(key_site, logits, axis=-1)

            proposed = batch_samples.at[batch_idx, site_idx].set(
                1 - batch_samples[batch_idx, site_idx]
            )
            log_prob_prop = jnp.real(
                jnp.asarray(log_prob_fn(proposed), dtype=jnp.float64)
            )

            flippable_prop = flippable_mask(proposed)
            n_flippable_prop = jnp.sum(flippable_prop, axis=-1).astype(jnp.float64)

            # Proposal is uniform over flippable sites, so include size ratio.
            log_accept = (
                log_prob_prop
                - batch_log_prob
                + jnp.log(n_flippable)
                - jnp.log(n_flippable_prop)
            )
            accept = jax.random.uniform(key_u, (n_samples,)) < jnp.exp(
                jnp.minimum(0.0, log_accept)
            )
            batch_samples = jnp.where(accept[:, None], proposed, batch_samples)
            batch_log_prob = jnp.where(accept, log_prob_prop, batch_log_prob)
            return (batch_samples, batch_log_prob, key_out), accept

        (samples, log_prob, key), accepts = jax.lax.scan(
            sweep, (samples, log_prob, key), None, length=int(n_sweeps)
        )
        acceptance = jnp.mean(accepts)
        return samples, log_prob, key, acceptance
