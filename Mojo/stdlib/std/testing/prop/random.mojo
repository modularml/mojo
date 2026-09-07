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
"""Implements random number generation for property-based testing."""

from std.random import random_ui64, seed as seed_fn


struct Rng(Movable):
    """A seeded pseudo-random number generator.

    Users should not need to create this type directly, instead use the `Rng`
    value provided by the `Strategy` trait.
    """

    @doc_hidden
    def __init__(out self, *, seed: Int):
        # TODO: Figure out how to ensure this 'global' seed value is not
        # accidentally overwritten by the user in their test code.
        seed_fn(seed)

    # TODO: Add playback support.
    @doc_hidden
    def _next(mut self, max: UInt64 = UInt64.MAX) raises -> UInt64:
        """If in playback mode, returns the next value in the history, otherwise
        generates a random value and records it.

        All random values are built on top of a random `UInt64` so we can record
        and keep a history of all generated values. When a test failure is
        encountered, this history is used to "shrink" the generated input values
        from Strategies by reordering, reducing, and removing the individual
        `UInt64` values in the history.

        Args:
            max: The maximum value.

        Returns:
            The next value in the history, or a random value if in live mode.

        Raises:
            If in playback mode and the history is exhausted.
        """
        return random_ui64(0, max)

    def _xoshiro_float(mut self) raises -> Float64:
        """Returns a random `Float64` between `[0.0, 1.0]` using the Xoshiro
        algorithm.

        References:
            https://prng.di.unimi.it/#remarks
        """
        var uint64 = self._next()
        # C++ equivalent (uint64 >> 11) * 0x1.0p-53
        var float64 = Float64(uint64 >> 11) * (2.0**-53)
        return float64

    def rand_bool(
        mut self,
        *,
        true_probability: Float64 = 0.5,
    ) raises -> Bool:
        """Returns a random `Bool` with the given probability of being True.

        Args:
            true_probability: The probability of being `True` (between 0.0 and 1.0).

        Returns:
            A random `Bool`.

        Raises:
            If the underlying random number generator raises an error.
        """
        if true_probability < 0.0:
            return False
        if true_probability > 1.0:
            return True

        var percentage = self._xoshiro_float()
        return true_probability > percentage

    # TODO: Revisit when we have a better random module.
    def rand_scalar[
        dtype: DType
    ](
        mut self,
        *,
        min: Scalar[dtype] = Scalar[dtype].MIN_FINITE,
        max: Scalar[dtype] = Scalar[dtype].MAX_FINITE,
    ) raises -> Scalar[dtype]:
        """Returns a random `Scalar` from the given range.

        Parameters:
            dtype: The `DType` of the scalar.

        Args:
            min: The minimum value.
            max: The maximum value.

        Returns:
            A random number in the range [min, max].

        Raises:
            If the minimum value is greater than the maximum value or if the
            underlying random number generator raises an error.
        """
        if min > max:
            raise Error("invalid min/max")

        if min == max:
            return min

        comptime if dtype == .bool:
            return rebind[Scalar[dtype]](Scalar[.bool](self.rand_bool()))
        elif dtype.is_integral():
            var offset = UInt64(0) - UInt64(Scalar[dtype].MIN)
            var a = UInt64(min) + offset
            var b = UInt64(max) + offset
            var diff = a - b if a > b else b - a
            var uint64 = self._next(diff)
            return Scalar[dtype](uint64) + min
        elif dtype.is_floating_point():
            var f = self._xoshiro_float()
            var result = Float64(min) * (1.0 - f) + Float64(max) * f
            return Scalar[dtype](result)
        else:
            comptime assert (
                False
            ), "rand_scalar expected bool, integral, or floating point"

    # TODO (MSTDL-1185): Can remove when UInt and SIMD are unified.
    def rand_uint(
        mut self,
        *,
        min: UInt = UInt.MIN,
        max: UInt = UInt.MAX,
    ) raises -> UInt:
        """Returns a random `UInt` from the given range.

        Args:
            min: The minimum value.
            max: The maximum value.

        Returns:
            A random `UInt` in the range [min, max].

        Raises:
            If the underlying random number generator raises an error.
        """
        return self.rand_scalar(min=min, max=max)

    # TODO (MSTDL-1185): Can remove when Int and SIMD are unified.
    def rand_int(
        mut self,
        *,
        min: Int = Int.MIN,
        max: Int = Int.MAX,
    ) raises -> Int:
        """Returns a random `Int` from the given range.

        Args:
            min: The minimum value.
            max: The maximum value.

        Returns:
            A random `Int` in the range [min, max].

        Raises:
            If the underlying random number generator raises an error.
        """
        return Int(
            self.rand_scalar[.int](
                min=Int(min),
                max=Int(max),
            )
        )
