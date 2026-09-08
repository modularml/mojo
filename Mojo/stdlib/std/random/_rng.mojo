# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Internal random number generator implementation.

This module provides a pure Mojo implementation of a high-quality PRNG
using the Philox algorithm, which works efficiently on both CPU and GPU.

Warning:
    This is NOT a cryptographically secure random number generator and should
    not be used for security-sensitive applications such as generating
    passwords, tokens, encryption keys, or any other security-critical values.

This is an internal implementation detail and should not be used directly.
Use the public API in `random.mojo` instead.
"""

from std.math import sqrt, log, cos, pi
from std.os import abort
from std.ffi import _Global
from std.utils.numerics import isnan, max_finite, FPUtils

from .philox import Random as PhiloxRandom


struct _PhiloxWrapper(Copyable):
    """Wrapper around Philox RNG to provide a simpler interface.

    This struct wraps the Philox RNG and caches generated values to provide
    a simple scalar interface (generating one value at a time) while using
    the vectorized Philox generator underneath.

    Warning:
        This is NOT a cryptographically secure random number generator.
        Do not use this for security-sensitive applications such as
        generating passwords, tokens, encryption keys, or any other
        security-critical random values. The internal state can be
        predicted from output, making it unsuitable for cryptographic use.
    """

    var _rng: PhiloxRandom[10]
    """The underlying Philox random number generator."""

    comptime _cache_length = 4

    var _cache: SIMD[.uint32, Self._cache_length]
    """Cache of generated random values."""

    var _cache_index: Int
    """Index into the cache for the next value to return."""

    def __init__(out self, seed: UInt64 = 0x3D30F19CD101):
        """Initializes the Philox wrapper.

        Args:
            seed: The seed value for the generator.
        """
        self._rng = PhiloxRandom(seed=seed)
        self._cache = SIMD[.uint32, Self._cache_length](0)
        self._cache_index = (
            Self._cache_length
        )  # Start empty to trigger generation

    def next_uint32(mut self) -> UInt32:
        """Generates the next 32-bit random number.

        Returns:
            A random UInt32 value.
        """
        if self._cache_index >= Self._cache_length:
            self._cache = self._rng.step()
            self._cache_index = 0

        var result = self._cache[self._cache_index]
        self._cache_index += 1
        return result

    def next_uint64(mut self) -> UInt64:
        """Generates a 64-bit random number from two 32-bit values.

        Returns:
            A random UInt64 value.
        """
        var high = UInt64(self.next_uint32())
        var low = UInt64(self.next_uint32())
        return (high << 32) | low

    def next_float64(mut self) -> Float64:
        """Generates a random Float64 in the range [0, 1).

        Returns:
            A random Float64 value in [0, 1).
        """
        # Use 53 bits of randomness (mantissa precision of Float64)
        var value = self.next_uint64() >> 11
        comptime float64_mantissa_bits = FPUtils[
            DType.float64
        ].mantissa_width() + 1  # 53 (52 mantissa bits + 1 for the sign bit)
        return Float64(value) * (1.0 / Float64(1 << float64_mantissa_bits))


struct _RandomState(Copyable):
    """Global random state manager.

    This struct manages a global _PhiloxWrapper instance for the random module.
    This provides process-wide global storage, NOT thread-local storage.
    All threads share the same random state instance. Concurrent access
    from multiple threads without external synchronization will result in
    race conditions and undefined behavior.
    """

    var _generator: _PhiloxWrapper
    """The underlying Philox wrapper generator."""

    def __init__(out self):
        """Initializes with default seed."""
        self._generator = _PhiloxWrapper(0)

    def __init__(out self, seed: UInt64):
        """Initializes with a specific seed.

        Args:
            seed: The seed value.
        """
        self._generator = _PhiloxWrapper(seed)

    def seed(mut self, value: UInt64):
        """Re-seeds the generator.

        Args:
            value: The new seed value.
        """
        self._generator = _PhiloxWrapper(value)

    def random_uint64(mut self, min: UInt64, max: UInt64) -> UInt64:
        """Generates a random UInt64 in the range [min, max].

        Args:
            min: Minimum value (inclusive).
            max: Maximum value (inclusive).

        Returns:
            A random UInt64 in [min, max].
        """
        if min >= max:
            return min

        var range = max - min + 1

        if range == 0:
            return self._generator.next_uint64()

        # Use rejection sampling for uniform distribution
        var threshold = (~range + 1) % range

        while True:
            var value = self._generator.next_uint64()
            if value >= threshold:
                return min + (value % range)

    def random_int64(mut self, min: Int64, max: Int64) -> Int64:
        """Generates a random Int64 in the range [min, max].

        Args:
            min: Minimum value (inclusive).
            max: Maximum value (inclusive).

        Returns:
            A random Int64 in [min, max].
        """
        if min >= max:
            return min

        var range = UInt64(max - min) + 1

        if range == 0:
            return Int64(self._generator.next_uint64())

        var threshold = (~range + 1) % range

        while True:
            var value = self._generator.next_uint64()
            if value >= threshold:
                return min + Int64(value % range)

    def random_float64(mut self, min: Float64, max: Float64) -> Float64:
        """Generates a random Float64 in the range [min, max).

        Args:
            min: Minimum value (inclusive).
            max: Maximum value.

        Returns:
            A random Float64 in [min, max).
        """
        var unit = self._generator.next_float64()
        var range = max - min

        # Handle edge cases: inf, nan, or very large ranges
        if isnan(range):
            return min
        if range > max_finite[.float64]():
            # For very large ranges (e.g., -inf to +inf), direct multiplication
            # unit * range would overflow. Factor the calculation to avoid
            # intermediate overflow: min + unit * (max - min) becomes
            # min + unit * max - unit * min
            return min + unit * max - unit * min

        return min + unit * range

    def normal_float64(mut self, mean: Float64, std_dev: Float64) -> Float64:
        """Generates a random Float64 from a normal distribution.

        Uses the Box-Muller transform to generate normally distributed values.

        Args:
            mean: The mean of the distribution.
            std_dev: The standard deviation of the distribution.

        Returns:
            A random Float64 sampled from Normal(mean, std_dev).
        """
        # Box-Muller transform: converts two independent uniform [0,1) random
        # variables into two independent standard normal N(0,1) variables.
        # Formula: Given u1, u2 ~ Uniform(0,1):
        #   z0 = sqrt(-2 * ln(u1)) * cos(2π * u2)
        #   z1 = sqrt(-2 * ln(u1)) * sin(2π * u2)
        # Both z0 and z1 are independent standard normal variables.

        # Generate two uniform random values in [0, 1)
        var u1 = self._generator.next_float64()
        var u2 = self._generator.next_float64()

        # Ensure u1 > 0 to avoid log(0) which is undefined (-infinity).
        # This is extremely unlikely but we handle it by rejection sampling
        # to ensure numerical stability.
        while u1 <= 0.0:
            u1 = self._generator.next_float64()

        # Apply Box-Muller transform to get a standard normal N(0,1) sample.
        # We compute only z0 here (could also compute z1 for efficiency, but
        # we'd need to cache it for the next call to avoid waste).
        var mag = std_dev * sqrt(-2.0 * log(u1))
        var z0 = mag * cos(2.0 * pi * u2)

        # Transform from standard normal N(0,1) to N(mean, std_dev^2)
        # using the affine transformation: X = mean + std_dev * Z
        return mean + z0


def _init_random_state() -> _RandomState:
    """Initializes the global random state.

    Returns:
        A new _RandomState instance.
    """
    return _RandomState()


def _get_global_random_state(
    out result: Pointer[_RandomState, MutUntrackedOrigin]
):
    """Gets the global random state.

    Returns:
        A pointer to the global _RandomState instance.
    """
    try:
        result = _global_random_state.get_or_create_ptr()
    except:
        abort("Failed to initialize global random state")


# Global random state (using _Global for proper global storage)
comptime _global_random_state = _Global["random_state", _init_random_state]
