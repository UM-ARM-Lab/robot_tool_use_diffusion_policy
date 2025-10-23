"""
Generic random number generator with buffered sampling for improved performance.

Provides pre-generated buffers of random floats and integers to reduce
the overhead of calling numpy's RNG on every sample.
"""

import numpy as np


class RandomNumberGenerator:
    """
    A wrapper class for a random number generator that holds buffers of pre-generated
    random numbers for faster access.

    Instead of calling np.random functions every time we need a random number, this
    class pre-generates large batches and serves them from memory buffers. This is
    significantly faster when you need many random numbers in tight loops.

    Example:
        >>> rng = RandomNumberGenerator(buffer_size=10000, seed=42)
        >>> # Sample random floats
        >>> prob = rng.random()  # Returns single float in [0, 1)
        >>> probs = rng.random(size=100)  # Returns 100 floats
        >>>
        >>> # Sample random integers
        >>> rng.set_max_val(1000)  # Set range for integers
        >>> idx = rng.randint()  # Returns single int in [0, 1000)
        >>> indices = rng.randint(size=100)  # Returns 100 ints
    """

    def __init__(self, buffer_size=10000, seed=12345):
        """
        Initializes the random number generator with a seed and buffer size.

        Args:
            buffer_size: The number of random numbers to pre-generate. Should be
                        large enough that you're not frequently regenerating buffers.
                        Recommended: 10000-100000 for training loops.
            seed: The seed for the random number generator for reproducibility.
        """
        self.buffer_size = buffer_size
        self.seed = seed
        self.max_val = -1

        # Create numpy random generator with seed
        self._generator = np.random.default_rng(seed)

        # Pre-generate buffer of random floats
        self._reset_fp_buffer()

        # Int buffer will be created when set_max_val is called
        self._int_buffer = None
        self._int_counter = 0

    def _reset_fp_buffer(self):
        """Refill the float buffer with new random values."""
        self._fp_buffer = self._generator.random(size=self.buffer_size)
        self._fp_counter = 0

    def _reset_int_buffer(self):
        """Refill the integer buffer with new random values."""
        if self.max_val == -1:
            raise ValueError("Need to call set_max_val before generating int buffer")
        self._int_buffer = self._generator.integers(
            low=0, high=self.max_val, size=self.buffer_size
        )
        self._int_counter = 0

    def random(self, size=1):
        """
        Returns random float value(s) between 0 and 1.

        Args:
            size: Number of random floats to return (default: 1)

        Returns:
            If size=1: single float
            If size>1: numpy array of floats

        Raises:
            ValueError: If size > buffer_size
        """
        if size > self.buffer_size:
            raise ValueError(
                f"Requested size ({size}) exceeds buffer size ({self.buffer_size}). "
                f"Either reduce size or increase buffer_size."
            )

        # Check if we need to refill the buffer
        if self._fp_counter + size > self.buffer_size:
            self._reset_fp_buffer()

        # Get random value(s) from buffer
        if size == 1:
            result = self._fp_buffer[self._fp_counter]
        else:
            result = self._fp_buffer[self._fp_counter:self._fp_counter + size]

        self._fp_counter += size
        return result

    def set_max_val(self, max_val):
        """
        Sets the maximum integer value for randint and creates a buffer.

        Args:
            max_val: Maximum value (exclusive) for randint. Will generate
                    integers in range [0, max_val).
        """
        if max_val <= 0:
            raise ValueError(f"max_val must be positive, got {max_val}")

        self.max_val = max_val
        self._reset_int_buffer()

    def randint(self, size=1):
        """
        Returns random integer value(s) between 0 and self.max_val (exclusive).

        Args:
            size: Number of random integers to return (default: 1)

        Returns:
            If size=1: single integer
            If size>1: numpy array of integers

        Raises:
            ValueError: If set_max_val() hasn't been called
            ValueError: If size > buffer_size
        """
        if self.max_val == -1:
            raise ValueError("Need to call set_max_val before calling randint")

        if size > self.buffer_size:
            raise ValueError(
                f"Requested size ({size}) exceeds buffer size ({self.buffer_size}). "
                f"Either reduce size or increase buffer_size."
            )

        # Check if we need to refill the buffer
        if self._int_counter + size > self.buffer_size:
            self._reset_int_buffer()

        # Get random value(s) from buffer
        if size == 1:
            result = self._int_buffer[self._int_counter]
        else:
            result = self._int_buffer[self._int_counter:self._int_counter + size]

        self._int_counter += size
        return result

    def get_stats(self):
        """
        Get statistics about buffer usage (useful for debugging/monitoring).

        Returns:
            dict with buffer usage information
        """
        return {
            'buffer_size': self.buffer_size,
            'seed': self.seed,
            'max_val': self.max_val,
            'fp_buffer_usage': f"{self._fp_counter}/{self.buffer_size}",
            'int_buffer_usage': f"{self._int_counter}/{self.buffer_size}" if self._int_buffer is not None else "N/A",
            'fp_buffer_remaining': self.buffer_size - self._fp_counter,
            'int_buffer_remaining': self.buffer_size - self._int_counter if self._int_buffer is not None else 0,
        }


def test_random_number_generator():
    """Test the RandomNumberGenerator implementation"""
    print("Testing RandomNumberGenerator...")

    # Test basic initialization
    rng = RandomNumberGenerator(buffer_size=100, seed=42)
    print(f"✓ Initialization successful")

    # Test random float generation
    single_float = rng.random()
    assert 0 <= single_float < 1, "Single float out of range"
    print(f"✓ Single float: {single_float:.4f}")

    # Test bulk float generation
    floats = rng.random(size=10)
    assert len(floats) == 10, "Wrong number of floats returned"
    assert all(0 <= f < 1 for f in floats), "Floats out of range"
    print(f"✓ Bulk floats (10): min={floats.min():.4f}, max={floats.max():.4f}")

    # Test buffer refill for floats
    # Generate enough to trigger refill
    for _ in range(10):
        rng.random(size=10)
    print(f"✓ Float buffer refill works")

    # Test integer generation
    rng.set_max_val(100)
    single_int = rng.randint()
    assert 0 <= single_int < 100, "Single int out of range"
    print(f"✓ Single int: {single_int}")

    # Test bulk integer generation
    ints = rng.randint(size=20)
    assert len(ints) == 20, "Wrong number of ints returned"
    assert all(0 <= i < 100 for i in ints), "Ints out of range"
    print(f"✓ Bulk ints (20): min={ints.min()}, max={ints.max()}")

    # Test buffer refill for ints
    for _ in range(5):
        rng.randint(size=10)
    print(f"✓ Int buffer refill works")

    # Test stats
    stats = rng.get_stats()
    print(f"✓ Stats: {stats}")

    # Test error handling
    try:
        rng.randint(size=200)  # Exceeds buffer size
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Error handling: {e}")

    # Test reproducibility with same seed
    rng1 = RandomNumberGenerator(buffer_size=100, seed=123)
    rng2 = RandomNumberGenerator(buffer_size=100, seed=123)

    vals1 = rng1.random(size=10)
    vals2 = rng2.random(size=10)
    assert np.allclose(vals1, vals2), "Same seed should produce same values"
    print(f"✓ Reproducibility verified")

    # Test alpha-sampling use case (like in cotraining)
    print("\n--- Testing Co-training Use Case ---")
    rng = RandomNumberGenerator(buffer_size=10000, seed=42)
    alpha = 0.3
    n_samples = 10000

    # Simulate alpha-based sampling
    real_count = sum(1 for _ in range(n_samples) if rng.random() < alpha)
    empirical_alpha = real_count / n_samples
    print(f"Target alpha: {alpha}")
    print(f"Empirical alpha: {empirical_alpha:.4f}")
    print(f"Difference: {abs(empirical_alpha - alpha):.4f}")
    assert abs(empirical_alpha - alpha) < 0.02, "Alpha sampling not accurate enough"
    print(f"✓ Alpha-sampling works correctly")

    # Test index sampling for two datasets
    rng.set_max_val(1000)  # Size of larger dataset
    indices = rng.randint(size=100)
    print(f"✓ Index sampling: generated {len(indices)} indices in range [0, 1000)")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_random_number_generator()
