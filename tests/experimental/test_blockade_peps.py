"""Tests for Blockade PEPS implementation."""
import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.core import _sample_counts, _trim_samples, make_mc_sampler
from vmc.operators import LocalHamiltonian
from vmc.peps import (
    DensityMatrix,
    NoTruncation,
    Variational,
    ZipUp,
    build_mc_kernels,
)
from vmc.peps.blockade import (
    BlockadePEPS,
    BlockadePEPSConfig,
    random_independent_set,
    rydberg_hamiltonian,
)


def _sample_with_kernels(
    model: BlockadePEPS,
    operator: LocalHamiltonian,
    *,
    n_samples: int,
    n_chains: int,
    key: jax.Array,
    initial_configuration: jax.Array,
    full_gradient: bool,
) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array, jax.Array, jax.Array, jax.Array]:
    _, num_chains, chain_length, total_samples = _sample_counts(n_samples, n_chains)
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
    (final_configurations, final_keys, _), (samples_hist, estimates) = mc_sampler(
        tensors,
        config_states,
        chain_keys,
        cache,
        n_steps=chain_length,
    )
    samples = _trim_samples(samples_hist, total_samples, n_samples)
    grads = _trim_samples(estimates.local_log_derivatives, total_samples, n_samples)
    p = (
        None
        if full_gradient
        else _trim_samples(estimates.active_slice_indices, total_samples, n_samples)
    )
    amps = _trim_samples(estimates.amp, total_samples, n_samples)
    energies = _trim_samples(estimates.local_estimate, total_samples, n_samples)
    return samples, grads, p, final_keys, final_configurations, amps, energies


class BlockadePEPSConfigTest(unittest.TestCase):
    """Tests for BlockadePEPSConfig."""

    def test_dmax_property(self):
        config = BlockadePEPSConfig(shape=(3, 3), D0=2, D1=3)
        self.assertEqual(config.Dmax, 3)

        config2 = BlockadePEPSConfig(shape=(3, 3), D0=4, D1=2)
        self.assertEqual(config2.Dmax, 4)

    def test_phys_dim_must_be_2(self):
        with self.assertRaises(ValueError):
            BlockadePEPSConfig(shape=(3, 3), D0=2, D1=2, phys_dim=3)


class BlockadePEPSTensorTest(unittest.TestCase):
    """Tests for tensor assembly and shapes."""

    def test_tensor_shapes(self):
        """Test that tensor shapes are correct for different positions."""
        config = BlockadePEPSConfig(shape=(3, 3), D0=2, D1=3)
        model = BlockadePEPS(
            rngs=nnx.Rngs(0),
            config=config,
            contraction_strategy=NoTruncation(),
        )
        Dmax = config.Dmax

        # Corner (0, 0): no incoming edges, so nc=1
        corner = jnp.asarray(model.tensors[0][0])
        self.assertEqual(corner.shape, (2, 1, 1, Dmax, 1, Dmax))

        # Edge (0, 1): 1 incoming edge (left), so nc=2
        edge_top = jnp.asarray(model.tensors[0][1])
        self.assertEqual(edge_top.shape, (2, 2, 1, Dmax, Dmax, Dmax))

        # Edge (1, 0): 1 incoming edge (up), so nc=2
        edge_left = jnp.asarray(model.tensors[1][0])
        self.assertEqual(edge_left.shape, (2, 2, Dmax, Dmax, 1, Dmax))

        # Bulk (1, 1): 2 incoming edges (left, up), so nc=4
        bulk = jnp.asarray(model.tensors[1][1])
        self.assertEqual(bulk.shape, (2, 4, Dmax, Dmax, Dmax, Dmax))


class CfgIdxTest(unittest.TestCase):
    """Tests for configuration index computation."""

    def test_cfg_idx_bulk_n0(self):
        """Test cfg_idx for n=0 at bulk site."""
        from vmc.peps.blockade.model import _assemble_site

        config = BlockadePEPSConfig(shape=(3, 3), D0=2, D1=2)
        model = BlockadePEPS(
            rngs=nnx.Rngs(0),
            config=config,
            contraction_strategy=NoTruncation(),
        )
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]

        # Test different incoming configurations at bulk site (1, 1)
        # kL=0, kU=0 -> cfg_idx=0
        n_config = jnp.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=jnp.int32)
        tensor = _assemble_site(
            tensors,
            config,
            1,
            1,
            n_config[1, 1],
            n_config[1, 0],
            n_config[0, 1],
        )
        expected = tensors[1][1][:, 0]
        self.assertTrue(jnp.allclose(tensor, expected))

        # kL=0, kU=1 -> cfg_idx=1
        n_config = jnp.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=jnp.int32)
        tensor = _assemble_site(
            tensors,
            config,
            1,
            1,
            n_config[1, 1],
            n_config[1, 0],
            n_config[0, 1],
        )
        expected = tensors[1][1][:, 1]
        self.assertTrue(jnp.allclose(tensor, expected))

        # kL=1, kU=0 -> cfg_idx=2
        n_config = jnp.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=jnp.int32)
        tensor = _assemble_site(
            tensors,
            config,
            1,
            1,
            n_config[1, 1],
            n_config[1, 0],
            n_config[0, 1],
        )
        expected = tensors[1][1][:, 2]
        self.assertTrue(jnp.allclose(tensor, expected))

        # kL=1, kU=1 -> cfg_idx=3
        n_config = jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=jnp.int32)
        tensor = _assemble_site(
            tensors,
            config,
            1,
            1,
            n_config[1, 1],
            n_config[1, 0],
            n_config[0, 1],
        )
        expected = tensors[1][1][:, 3]
        self.assertTrue(jnp.allclose(tensor, expected))

    def test_cfg_idx_n1_always_0(self):
        """Test that n=1 always uses cfg_idx=0."""
        from vmc.peps.blockade.model import _assemble_site

        config = BlockadePEPSConfig(shape=(3, 3), D0=2, D1=2)
        model = BlockadePEPS(
            rngs=nnx.Rngs(0),
            config=config,
            contraction_strategy=NoTruncation(),
        )
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]

        # When n=1, cfg_idx should always be 0 (regardless of neighbors)
        n_config = jnp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=jnp.int32)
        tensor = _assemble_site(
            tensors,
            config,
            1,
            1,
            n_config[1, 1],
            n_config[1, 0],
            n_config[0, 1],
        )
        expected = tensors[1][1][:, 0]
        self.assertTrue(jnp.allclose(tensor, expected))


class IndependentSetTest(unittest.TestCase):
    """Tests for independent set configuration generation."""

    def test_random_independent_set_valid(self):
        """Test that generated configurations are valid independent sets."""
        shape = (4, 4)
        key = jax.random.key(42)

        for _ in range(10):
            key, subkey = jax.random.split(key)
            config = random_independent_set(subkey, shape)

            # Check: no adjacent 1s
            config_np = jax.device_get(config)
            for r in range(shape[0]):
                for c in range(shape[1]):
                    if config_np[r, c] == 1:
                        # Check all neighbors
                        if c > 0:
                            self.assertEqual(config_np[r, c - 1], 0)
                        if c < shape[1] - 1:
                            self.assertEqual(config_np[r, c + 1], 0)
                        if r > 0:
                            self.assertEqual(config_np[r - 1, c], 0)
                        if r < shape[0] - 1:
                            self.assertEqual(config_np[r + 1, c], 0)

    def test_random_independent_set_deterministic(self):
        """Test that same key produces same configuration."""
        shape = (3, 3)
        key = jax.random.key(123)
        config1 = random_independent_set(key, shape)
        config2 = random_independent_set(key, shape)
        self.assertTrue(jnp.array_equal(config1, config2))


class BlockadeConstraintTest(unittest.TestCase):
    """Tests for blockade constraint enforcement."""

    def test_violating_config_gives_zero_amplitude(self):
        """Test that blockade-violating configs have zero amplitude."""
        config = BlockadePEPSConfig(shape=(2, 2), D0=2, D1=2)
        model = BlockadePEPS(
            rngs=nnx.Rngs(0),
            config=config,
            contraction_strategy=NoTruncation(),
        )
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]

        # Valid configuration: alternating pattern
        valid_config = jnp.array([[1, 0], [0, 1]], dtype=jnp.int32)
        valid_sample = BlockadePEPS.flatten_sample(valid_config)
        amp_valid = BlockadePEPS.apply(
            tensors, valid_sample, config.shape, config, model.strategy
        )

        # Invalid configuration: adjacent 1s (violation)
        # This should give zero amplitude due to cfg_idx mismatch
        invalid_config = jnp.array([[1, 1], [0, 0]], dtype=jnp.int32)
        invalid_sample = BlockadePEPS.flatten_sample(invalid_config)
        amp_invalid = BlockadePEPS.apply(
            tensors, invalid_sample, config.shape, config, model.strategy
        )

        # Valid amplitude should be non-zero
        self.assertGreater(jnp.abs(amp_valid), 1e-10)

        self.assertLess(jnp.abs(amp_invalid), 1e-10)


class HamiltonianTest(unittest.TestCase):
    """Tests for Rydberg Hamiltonian builder."""

    def test_hamiltonian_terms_count(self):
        """Test that hamiltonian has correct number of terms."""
        shape = (3, 3)
        n_sites = shape[0] * shape[1]

        # Without NNN: n_sites X terms + n_sites n terms
        h = rydberg_hamiltonian(shape, Omega=1.0, Delta=0.5)
        self.assertEqual(len(h.terms), 2 * n_sites)

        # With NNN: adds diagonal NNN terms
        h_nnn = rydberg_hamiltonian(shape, Omega=1.0, Delta=0.5, V_nnn=0.1)
        # Diagonal NNN: (n_rows-1) * (n_cols-1) pairs for diagonal
        # Anti-diagonal NNN: (n_rows-1) * (n_cols-1) pairs for anti-diagonal
        expected_nnn = 2 * (shape[0] - 1) * (shape[1] - 1)
        self.assertEqual(len(h_nnn.terms), 2 * n_sites + expected_nnn)


class SweepTest(unittest.TestCase):
    """Tests for the sweep function."""

    def test_sweep_produces_valid_configs(self):
        """Test that sweep produces valid independent set configurations."""
        config = BlockadePEPSConfig(shape=(3, 3), D0=2, D1=2)
        model = BlockadePEPS(
            rngs=nnx.Rngs(0),
            config=config,
            contraction_strategy=NoTruncation(),
        )

        key = jax.random.key(42)
        key, init_key = jax.random.split(key)
        initial_config = random_independent_set(init_key, config.shape)
        initial_sample = BlockadePEPS.flatten_sample(initial_config).reshape(1, -1)
        samples, _, _, _, _, _, _ = _sample_with_kernels(
            model,
            LocalHamiltonian(shape=config.shape, terms=()),
            n_samples=5,
            n_chains=1,
            key=key,
            initial_configuration=initial_sample,
            full_gradient=False,
        )

        for i in range(samples.shape[0]):
            new_config = BlockadePEPS.unflatten_sample(samples[i], config.shape)

            # Check validity
            config_np = jax.device_get(new_config)
            for r in range(config.shape[0]):
                for c in range(config.shape[1]):
                    if config_np[r, c] == 1:
                        if c > 0:
                            self.assertEqual(config_np[r, c - 1], 0)
                        if c < config.shape[1] - 1:
                            self.assertEqual(config_np[r, c + 1], 0)
                        if r > 0:
                            self.assertEqual(config_np[r - 1, c], 0)
                        if r < config.shape[0] - 1:
                            self.assertEqual(config_np[r + 1, c], 0)


class GradsAndEnergyTest(unittest.TestCase):
    """Tests for grads_and_energy function."""

    def test_grads_and_energy_shapes(self):
        """Test that grads_and_energy returns correct shapes."""
        from vmc.peps.blockade import model as blockade_model
        from vmc.peps.common.contraction import _forward_with_cache

        config = BlockadePEPSConfig(shape=(3, 3), D0=2, D1=2)
        model = BlockadePEPS(
            rngs=nnx.Rngs(0),
            config=config,
            contraction_strategy=NoTruncation(),
        )

        key = jax.random.key(42)
        initial_config = random_independent_set(key, config.shape)
        sample = BlockadePEPS.flatten_sample(initial_config)
        h = rydberg_hamiltonian(config.shape, Omega=1.0, Delta=0.5)

        config_2d = sample.reshape(config.shape)
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        eff_tensors = blockade_model.assemble_tensors(
            tensors,
            config_2d,
            config,
        )
        amp, top_envs = _forward_with_cache(
            eff_tensors,
            config_2d,
            config.shape,
            model.strategy,
        )
        env_grads, energy, _ = blockade_model.estimate(
            tensors,
            sample,
            amp,
            h,
            config.shape,
            config,
            model.strategy,
            top_envs,
        )

        self.assertEqual(len(env_grads), config.shape[0])
        self.assertEqual(len(env_grads[0]), config.shape[1])
        self.assertEqual(energy.shape, ())


class SamplerTest(unittest.TestCase):
    """Tests for sequential sampler dispatches."""

    def test_sequential_sample_shapes(self):
        """Test that sequential_sample returns correct shapes."""
        config = BlockadePEPSConfig(shape=(3, 3), D0=2, D1=2)
        model = BlockadePEPS(
            rngs=nnx.Rngs(0),
            config=config,
            contraction_strategy=NoTruncation(),
        )

        key = jax.random.key(42)
        key, init_key = jax.random.split(key)
        n_chains = 2
        n_samples = 4

        initial_configs = jax.vmap(lambda k: random_independent_set(k, config.shape))(
            jax.random.split(init_key, n_chains)
        )
        initial_flat = initial_configs.reshape(n_chains, -1)

        h = rydberg_hamiltonian(config.shape, Omega=1.0, Delta=0.5)
        samples, _, _, _, _, _, _ = _sample_with_kernels(
            model,
            h,
            n_samples=n_samples,
            n_chains=n_chains,
            key=key,
            initial_configuration=initial_flat,
            full_gradient=False,
        )

        n_sites = config.shape[0] * config.shape[1]
        self.assertEqual(samples.shape, (n_samples, n_sites))

    def test_sequential_sample_with_gradients_shapes(self):
        """Test that sequential_sample_with_gradients returns correct shapes."""
        config = BlockadePEPSConfig(shape=(3, 3), D0=2, D1=2)
        model = BlockadePEPS(
            rngs=nnx.Rngs(0),
            config=config,
            contraction_strategy=NoTruncation(),
        )

        key = jax.random.key(42)
        key, init_key = jax.random.split(key)
        n_chains = 2
        n_samples = 4

        initial_configs = jax.vmap(lambda k: random_independent_set(k, config.shape))(
            jax.random.split(init_key, n_chains)
        )
        initial_flat = initial_configs.reshape(n_chains, -1)

        h = rydberg_hamiltonian(config.shape, Omega=1.0, Delta=0.5)

        samples, grads, p, _, final_configs, amps, energies = _sample_with_kernels(
            model,
            h,
            n_samples=n_samples,
            n_chains=n_chains,
            key=key,
            initial_configuration=initial_flat,
            full_gradient=False,
        )

        n_sites = config.shape[0] * config.shape[1]
        self.assertEqual(samples.shape, (n_samples, n_sites))
        self.assertEqual(amps.shape, (n_samples,))
        self.assertEqual(energies.shape, (n_samples,))


class GradientFiniteDiffTest(unittest.TestCase):
    """Tests for gradient correctness via finite differences."""

    def test_amplitude_gradient_finite_diff(self):
        """Test that JAX gradients match finite differences."""
        import numpy as np
        from jax.flatten_util import ravel_pytree

        config = BlockadePEPSConfig(shape=(2, 2), D0=2, D1=2)
        model = BlockadePEPS(
            rngs=nnx.Rngs(0),
            config=config,
            contraction_strategy=NoTruncation(),
        )

        # Use a valid configuration
        key = jax.random.key(42)
        init_config = random_independent_set(key, config.shape)
        sample = BlockadePEPS.flatten_sample(init_config)

        # Get analytic gradient via JAX
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]

        def amp_fn(ts):
            return BlockadePEPS.apply(ts, sample, config.shape, config, model.strategy)

        amp, grad = jax.value_and_grad(amp_fn, holomorphic=True)(tensors)
        grad_flat, _ = ravel_pytree(grad)

        # Flatten tensors for finite diff
        def flatten_tensors(tensors):
            parts = []
            shapes = []
            for row in tensors:
                for t in row:
                    parts.append(np.asarray(t).ravel())
                    shapes.append(t.shape)
            return np.concatenate(parts), shapes

        def unflatten_tensors(flat, shapes, n_rows, n_cols):
            tensors = []
            offset = 0
            for r in range(n_rows):
                row = []
                for c in range(n_cols):
                    idx = r * n_cols + c
                    size = int(np.prod(shapes[idx]))
                    t = flat[offset:offset + size].reshape(shapes[idx])
                    row.append(jnp.asarray(t))
                    offset += size
                tensors.append(row)
            return tensors

        flat, shapes = flatten_tensors(tensors)

        def amp_from_flat(flat_params):
            ts = unflatten_tensors(flat_params, shapes, config.shape[0], config.shape[1])
            return np.asarray(
                BlockadePEPS.apply(ts, sample, config.shape, config, model.strategy)
            )

        # Central difference
        eps = 1e-6
        grad_fd = np.zeros_like(flat)
        for i in range(len(flat)):
            flat_plus = flat.copy()
            flat_plus[i] += eps
            flat_minus = flat.copy()
            flat_minus[i] -= eps
            grad_fd[i] = (amp_from_flat(flat_plus) - amp_from_flat(flat_minus)) / (2 * eps)

        # Compare
        max_diff = np.max(np.abs(np.asarray(grad_flat) - grad_fd))
        self.assertLess(max_diff, 1e-5)


class DiagonalEnergyTest(unittest.TestCase):
    """Test that diagonal energy is computed correctly."""

    def test_diagonal_energy_direct(self):
        """Test that diagonal energy matches direct computation."""
        from vmc.peps.blockade import model as blockade_model
        from vmc.peps.common.contraction import _forward_with_cache

        shape = (2, 2)
        Omega = 0.0  # No X term for this test
        Delta = 0.5

        h = rydberg_hamiltonian(shape, Omega=Omega, Delta=Delta)

        # Create model and compute energy
        config = BlockadePEPSConfig(shape=shape, D0=2, D1=2)
        model = BlockadePEPS(
            rngs=nnx.Rngs(0),
            config=config,
            contraction_strategy=NoTruncation(),
        )

        config_arr = jnp.array([[1, 0], [0, 1]], dtype=jnp.int32)
        n_ones = jnp.sum(config_arr)
        expected_energy = -Delta * n_ones
        sample = BlockadePEPS.flatten_sample(config_arr)
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        eff_tensors = blockade_model.assemble_tensors(
            tensors,
            config_arr,
            config,
        )
        amp, top_envs = _forward_with_cache(
            eff_tensors,
            config_arr,
            config.shape,
            model.strategy,
        )
        _, energy, _ = blockade_model.estimate(
            tensors,
            sample,
            amp,
            h,
            config.shape,
            config,
            model.strategy,
            top_envs,
        )

        # With Omega=0, energy should just be diagonal
        self.assertAlmostEqual(
            float(jnp.real(energy)),
            float(expected_energy),
            places=5,
        )


class SamplingBalanceTest(unittest.TestCase):
    """Test sampling balance properties."""

    def _enumerate_independent_sets_2x2(self):
        """Enumerate all valid independent sets for 2x2 grid.
        
        Valid configs (flattened, row-major):
        - 0000 (empty)
        - 1000, 0100, 0010, 0001 (single occupied)
        - 1001, 0110 (diagonal pairs - non-adjacent)
        """
        valid = [
            [0, 0, 0, 0],  # empty
            [1, 0, 0, 0],  # top-left
            [0, 1, 0, 0],  # top-right
            [0, 0, 1, 0],  # bottom-left
            [0, 0, 0, 1],  # bottom-right
            [1, 0, 0, 1],  # diagonal TL-BR
            [0, 1, 1, 0],  # diagonal TR-BL
        ]
        return jnp.array(valid, dtype=jnp.int32)

    def test_sampling_explores_configurations(self):
        """Test that sampling explores multiple configurations."""
        config = BlockadePEPSConfig(shape=(2, 2), D0=2, D1=2)
        model = BlockadePEPS(
            rngs=nnx.Rngs(0),
            config=config,
            contraction_strategy=NoTruncation(),
        )

        key = jax.random.key(42)
        key, init_key = jax.random.split(key)
        n_chains = 4
        n_samples = 20

        initial_configs = jax.vmap(lambda k: random_independent_set(k, config.shape))(
            jax.random.split(init_key, n_chains)
        )
        initial_flat = initial_configs.reshape(n_chains, -1)

        h = rydberg_hamiltonian(config.shape, Omega=1.0, Delta=0.5)
        samples, _, _, _, _, _, _ = _sample_with_kernels(
            model,
            h,
            n_samples=n_samples,
            n_chains=n_chains,
            key=key,
            initial_configuration=initial_flat,
            full_gradient=False,
        )

        # Check that we get multiple unique configurations
        unique_samples = jnp.unique(samples, axis=0)
        # Should have at least 2 unique samples (exploration happened)
        self.assertGreater(len(unique_samples), 1)

    def test_detailed_balance_chi_squared_3x3(self):
        """Test that sampling distribution matches |ψ|² on 3x3 via chi-squared test."""
        from scipy import stats
        import numpy as np
        import itertools

        shape = (3, 3)
        n_rows, n_cols = shape
        n_sites = n_rows * n_cols

        config = BlockadePEPSConfig(shape=shape, D0=2, D1=2)
        model = BlockadePEPS(
            rngs=nnx.Rngs(123),
            config=config,
            contraction_strategy=NoTruncation(),
        )

        valid = []
        for bits in itertools.product([0, 1], repeat=n_sites):
            arr = np.asarray(bits, dtype=np.int32).reshape(shape)
            ok = True
            for r in range(n_rows):
                for c in range(n_cols):
                    if arr[r, c] == 1:
                        if r > 0 and arr[r - 1, c] == 1:
                            ok = False
                            break
                        if r < n_rows - 1 and arr[r + 1, c] == 1:
                            ok = False
                            break
                        if c > 0 and arr[r, c - 1] == 1:
                            ok = False
                            break
                        if c < n_cols - 1 and arr[r, c + 1] == 1:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                valid.append(arr.reshape(-1))
        valid_configs = jnp.asarray(valid, dtype=jnp.int32)
        n_configs = len(valid_configs)

        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        amplitudes = jax.vmap(
            lambda s: BlockadePEPS.apply(tensors, s, config.shape, config, model.strategy)
        )(valid_configs)
        probs_exact = jnp.abs(amplitudes) ** 2
        probs_exact = probs_exact / jnp.sum(probs_exact)

        key = jax.random.key(42)
        key, init_key = jax.random.split(key)
        n_chains = 8
        n_samples = 10000

        initial_configs = jax.vmap(lambda k: random_independent_set(k, shape))(
            jax.random.split(init_key, n_chains)
        )
        initial_flat = initial_configs.reshape(n_chains, -1)

        h = rydberg_hamiltonian(shape, Omega=1.0, Delta=0.5)
        samples, _, _, _, _, _, _ = _sample_with_kernels(
            model,
            h,
            n_samples=n_samples,
            n_chains=n_chains,
            key=key,
            initial_configuration=initial_flat,
            full_gradient=False,
        )

        weights = 2 ** np.arange(n_sites - 1, -1, -1, dtype=np.int64)
        valid_codes_np = np.dot(np.asarray(valid_configs), weights)
        sample_codes_np = np.dot(np.asarray(samples), weights)

        counts = np.zeros(n_configs)
        for i, code in enumerate(valid_codes_np):
            counts[i] = np.sum(sample_codes_np == code)

        expected_counts = np.asarray(probs_exact) * n_samples
        mask = expected_counts > 5
        if np.sum(mask) < 2:
            self.skipTest("Not enough bins with expected count > 5")

        observed = counts[mask]
        expected = expected_counts[mask]

        chi2, p_value = stats.chisquare(observed, expected)
        self.assertGreater(
            p_value, 0.01,
            f"Chi-squared test failed: chi2={chi2:.2f}, p={p_value:.4f}\n"
            f"Expected: {expected}\nObserved: {observed}"
        )


if __name__ == "__main__":
    unittest.main()
