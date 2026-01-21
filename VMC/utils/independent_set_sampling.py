"""Independent-set samplers for blockade constraints."""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

from typing import Callable, Iterator, Sequence

import jax
import jax.numpy as jnp

from VMC.utils.utils import occupancy_to_spin

__all__ = [
    "DiscardBlockedSampler",
    "IndependentSetSampler",
    "all_config_batches",
    "build_neighbor_arrays",
    "config_codes",
    "enumerate_all_configs",
    "enumerate_independent_sets_grid",
    "grid_edges",
    "independent_set_violations",
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
    adjacency = [set() for _ in range(num_sites)]
    for u, v in edges:
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


def grid_edges(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    """Return undirected edges for a 2D square lattice."""
    edges = []
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            if r + 1 < n_rows:
                edges.append((idx, (r + 1) * n_cols + c))
            if c + 1 < n_cols:
                edges.append((idx, r * n_cols + (c + 1)))
    return edges


def _valid_row_masks(n_cols: int) -> tuple[list[int], jax.Array]:
    masks = []
    bits = []
    for mask in range(1 << n_cols):
        if mask & (mask << 1):
            continue
        masks.append(mask)
        bits.append([(mask >> c) & 1 for c in range(n_cols)])
    return masks, jnp.asarray(bits, dtype=jnp.int32)


def enumerate_independent_sets_grid(n_rows: int, n_cols: int) -> jax.Array:
    """Enumerate all independent sets on a 2D grid as occupancies."""
    masks, row_bits = _valid_row_masks(n_cols)
    sequences: list[tuple[int, ...]] = [()]
    for _ in range(n_rows):
        new_sequences = []
        for seq in sequences:
            prev_mask = masks[seq[-1]] if seq else None
            for idx, mask in enumerate(masks):
                if prev_mask is None or (mask & prev_mask) == 0:
                    new_sequences.append(seq + (idx,))
        sequences = new_sequences

    n_states = len(sequences)
    samples = jnp.zeros((n_states, n_rows * n_cols), dtype=jnp.int32)
    for state_idx, seq in enumerate(sequences):
        for row_idx, mask_idx in enumerate(seq):
            start = row_idx * n_cols
            samples = samples.at[state_idx, start : start + n_cols].set(
                row_bits[mask_idx]
            )
    return samples


def config_codes(samples: jax.Array) -> jax.Array:
    """Encode occupancy configurations as integer codes."""
    bits = samples.astype(jnp.uint32)
    shifts = jnp.arange(bits.shape[1], dtype=jnp.uint32)
    return jnp.sum(bits * (jnp.uint32(1) << shifts), axis=-1)


def enumerate_all_configs(num_sites: int) -> jax.Array:
    """Enumerate all 0/1 configurations for a given number of sites."""
    shifts = jnp.arange(num_sites, dtype=jnp.uint32)
    indices = jnp.arange(1 << num_sites, dtype=jnp.uint32)
    occupancies = (indices[:, None] >> shifts) & jnp.uint32(1)
    return occupancies.astype(jnp.int32)


def all_config_batches(
    num_sites: int, *, batch_size: int
) -> Iterator[tuple[jax.Array, jax.Array]]:
    """Generate all configurations in batches as occupancies and codes."""
    total = 1 << num_sites
    shifts = jnp.arange(num_sites, dtype=jnp.uint32)
    for start in range(0, total, batch_size):
        size = min(batch_size, total - start)
        indices = jnp.arange(start, start + size, dtype=jnp.uint32)
        occupancies = (indices[:, None] >> shifts) & jnp.uint32(1)
        yield occupancies.astype(jnp.int32), indices


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


def _prepare_samples(
    init_samples: jax.Array | None,
    n_samples: int,
    num_sites: int,
    neighbors: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Initialize samples."""
    if init_samples is None:
        samples = jnp.zeros((n_samples, num_sites), dtype=jnp.int32)
    else:
        samples = jnp.asarray(init_samples, dtype=jnp.int32)
        if samples.ndim == 1:
            samples = jnp.tile(samples[None, :], (n_samples, 1))
        if bool(jnp.any(independent_set_violations(samples, neighbors, mask))):
            raise ValueError("init_samples must contain only independent sets.")
    return samples


class IndependentSetSampler:
    """Standalone Metropolis-Hastings sampler for independent sets."""

    def __init__(self, num_sites: int, edges: Sequence[tuple[int, int]]):
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
        n_steps: int,
        key: jax.Array,
        init_samples: jax.Array | None = None,
        log_prob_is_batched: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Run blocked-aware Metropolis-Hastings steps.

        Args:
            log_prob_fn: Callable returning log-probabilities for a batch of samples.
                Use log |psi|^2 for VMC. If None, samples uniformly over independent sets.
            n_samples: Number of parallel chains.
            n_steps: Number of Metropolis steps to run.
            key: JAX PRNG key.
            init_samples: Optional initial samples with shape (n_samples, num_sites).
                Must contain only independent sets.
            log_prob_is_batched: If False, log_prob_fn is vmapped over samples.

        Returns:
            Tuple of (samples, log_prob, key, acceptance).
        """
        if log_prob_fn is None:
            def _zero_log_prob(batch_samples: jax.Array) -> jax.Array:
                return jnp.zeros((batch_samples.shape[0],), dtype=jnp.float64)

            log_prob_fn = _zero_log_prob
            log_prob_is_batched = True
        if not log_prob_is_batched:
            log_prob_fn = jax.vmap(log_prob_fn)

        samples = _prepare_samples(
            init_samples, n_samples, self._num_sites, self._neighbors, self._mask
        )
        log_prob = jnp.real(jnp.asarray(log_prob_fn(samples), dtype=jnp.float64))

        batch_idx = jnp.arange(n_samples)

        def flippable_mask(batch_samples: jax.Array) -> jax.Array:
            blocked = _blocked_counts(batch_samples, self._neighbors, self._mask)
            return (batch_samples == 1) | ((batch_samples == 0) & (blocked == 0))

        def step(carry, _):
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
            step, (samples, log_prob, key), None, length=int(n_steps)
        )
        acceptance = jnp.mean(accepts)
        return samples, log_prob, key, acceptance


class DiscardBlockedSampler:
    """Metropolis sampler that discards blocked proposals."""

    def __init__(self, num_sites: int, edges: Sequence[tuple[int, int]]):
        self._num_sites = int(num_sites)
        self._neighbors, self._mask = build_neighbor_arrays(num_sites, edges)

    @property
    def num_sites(self) -> int:
        return self._num_sites

    def sample(
        self,
        log_prob_fn: Callable[[jax.Array], jax.Array],
        *,
        n_samples: int,
        n_steps: int,
        key: jax.Array,
        init_samples: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Run Metropolis-Hastings steps with invalid moves rejected.

        Args:
            log_prob_fn: Callable returning log-probabilities for a batch of samples.
            n_samples: Number of parallel chains.
            n_steps: Number of Metropolis steps to run.
            key: JAX PRNG key.
            init_samples: Optional initial samples with shape (n_samples, num_sites).
                Must contain only independent sets.

        Returns:
            Tuple of (samples, log_prob, key, acceptance).
        """
        samples = _prepare_samples(
            init_samples, n_samples, self._num_sites, self._neighbors, self._mask
        )
        log_prob = jnp.real(jnp.asarray(log_prob_fn(samples), dtype=jnp.float64))

        batch_idx = jnp.arange(n_samples)

        def step(carry, _):
            batch_samples, batch_log_prob, key_in = carry
            key_out, key_site, key_u = jax.random.split(key_in, 3)
            site_idx = jax.random.randint(
                key_site, (n_samples,), 0, self._num_sites
            )

            proposed = batch_samples.at[batch_idx, site_idx].set(
                1 - batch_samples[batch_idx, site_idx]
            )
            invalid = independent_set_violations(proposed, self._neighbors, self._mask)
            log_prob_prop = jnp.real(
                jnp.asarray(log_prob_fn(proposed), dtype=jnp.float64)
            )

            log_accept = jnp.where(
                invalid, -jnp.inf, log_prob_prop - batch_log_prob
            )
            accept = jax.random.uniform(key_u, (n_samples,)) < jnp.exp(
                jnp.minimum(0.0, log_accept)
            )
            batch_samples = jnp.where(accept[:, None], proposed, batch_samples)
            batch_log_prob = jnp.where(accept, log_prob_prop, batch_log_prob)
            return (batch_samples, batch_log_prob, key_out), accept

        (samples, log_prob, key), accepts = jax.lax.scan(
            step, (samples, log_prob, key), None, length=int(n_steps)
        )
        acceptance = jnp.mean(accepts)
        return samples, log_prob, key, acceptance
