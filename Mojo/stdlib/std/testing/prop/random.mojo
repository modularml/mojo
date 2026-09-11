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

from std.bit import bit_reverse
from std.builtin.dtype import _uint_type_of_width
from std.random import random_ui64, seed as seed_fn
from std.sys.info import bit_width_of
from ._errors import PLAYBACK_EXHAUSTED


struct Rng(Movable):
    """A seeded pseudo-random number generator.

    Users should not need to create this type directly, instead use the `Rng`
    value provided by the `Strategy` trait.
    """

    var history: List[UInt64]
    """The recorded history of values generated."""

    var playback_mode: Bool
    """Whether this RNG producer is currently in playback mode."""

    var playback_index: Int
    """The current index in the history during playback mode."""

    @doc_hidden
    def __init__(out self, var history: List[UInt64]):
        self.history = history^
        self.playback_mode = True
        self.playback_index = 0

    @doc_hidden
    def __init__(out self, *, seed: Int):
        # TODO: Figure out how to ensure this 'global' seed value is not
        # accidentally overwritten by the user in their test code.
        seed_fn(seed)
        self.history = []
        self.playback_mode = False
        self.playback_index = 0

    @doc_hidden
    def _next(mut self, max: UInt64 = UInt64.MAX, out value: UInt64) raises:
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
        if self.playback_mode:
            if self.playback_index >= len(self.history):
                raise materialize[PLAYBACK_EXHAUSTED]()
            value = self.history[self.playback_index]
            self.playback_index += 1
        else:
            value = random_ui64(0, max)
            self.history.append(value)

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
        return percentage > (1.0 - true_probability)

    def _rand_more(mut self, count: Int, *, min: Int, max: Int) raises -> Bool:
        """Returns whether a collection strategy should draw another element.

        Args:
            count: The number of elements drawn so far.
            min: The minimum number of elements.
            max: The maximum number of elements.

        Returns:
            True if another element should be drawn.

        Raises:
            If the underlying random number generator raises an error.
        """
        if count < min:
            return True
        if count >= max:
            return False

        var average = Float64(min + max) / 2.0
        var probability = 1.0 - 1.0 / (1.0 + average)
        return self.rand_bool(true_probability=probability)

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
            comptime bits = bit_width_of[dtype]()
            comptime U = (
                DType.uint64 if bits
                <= 64 else DType.uint128 if bits
                == 128 else DType.uint256
            )
            comptime N = (bits + 63) / 64

            var span = max.cast[U]() - min.cast[U]()

            # Generate exactly enough 64-bit words to cover `span`. `k` is the
            # position of `span`'s highest nonzero word; capping the top word
            # there keeps the generated range within 2x of `span + 1`, bounding
            # the modular-reduction bias.
            var k = 0
            var hi_cap = span.cast[DType.uint64]()
            for i in range(1, N):
                var w = (span >> Scalar[U](64 * i)).cast[DType.uint64]()
                if w != 0:
                    k = i
                    hi_cap = w

            var result = self._next(hi_cap).cast[U]() << Scalar[U](64 * k)
            for j in range(k):
                result = result | (self._next().cast[U]() << Scalar[U](64 * j))

            if span != Scalar[U].MAX:
                result = result % (span + 1)
            return result.cast[dtype]() + min
        elif dtype.is_floating_point():
            comptime assert dtype in (
                DType.float16,
                DType.bfloat16,
                DType.float32,
                DType.float64,
            ), "rand_scalar supports float16, bfloat16, float32, and float64"
            return self._rand_float[dtype](min=min, max=max)
        else:
            comptime assert (
                False
            ), "rand_scalar expected bool, integral, or floating point"

    def _rand_float[
        dtype: DType
    ](mut self, *, min: Scalar[dtype], max: Scalar[dtype]) raises -> Scalar[
        dtype
    ]:
        """Returns a random float in `[min, max]` from one stream word: the
        sign in the word's top bit, then a magnitude code for
        `_decode_float_magnitude`.

        Parameters:
            dtype: The float dtype.

        Args:
            min: The minimum value.
            max: The maximum value.

        Returns:
            A random float in `[min, max]`.

        Raises:
            If the underlying stream raises an error.
        """
        comptime W = UInt64(bit_width_of[dtype]())
        comptime M = UInt64(DType.mantissa_width[dtype]())
        comptime mantissa_mask = (UInt64(1) << M) - 1
        comptime U = _uint_type_of_width[bit_width_of[dtype]()]()
        comptime sign_bit = Scalar[U](1) << Scalar[U](W - 1)

        if min != min or max != max:
            raise Error("invalid min/max")

        var lo = min.cast[DType.float64]()
        var hi = max.cast[DType.float64]()

        # Bound the word by the width of the target float type. The sign
        # sits in the top bit so that, in the shrinker's order, a positive
        # value is simpler than a negative one of the same magnitude.
        var word = self._next(UInt64.MAX >> (64 - W))
        var negative = (word >> (W - 1)) & 1 == 1
        var magnitude = _decode_float_magnitude[dtype](word)
        var value = magnitude
        # A zero magnitude with the sign applied is `-0.0`, which plain
        # comparisons cannot keep out of a range like `[0.0, 1.0]`.
        if negative and (magnitude != 0 or _neg_zero_allowed(lo, hi)):
            value = Scalar[dtype](from_bits=magnitude.to_bits() | sign_bit)

        var f = value.cast[DType.float64]()
        if lo <= f <= hi:
            return value

        var fraction = Float64(
            magnitude.to_bits[DType.uint64]() & mantissa_mask
        ) / Float64(mantissa_mask + 1)
        var range_size = hi - lo
        if range_size > Float64.MAX_FINITE:
            range_size = Float64.MAX_FINITE
        var folded: Float64
        if abs(hi) < abs(lo):
            folded = hi - range_size * fraction
        else:
            folded = lo + range_size * fraction

        # Guard against rounding in the arithmetic above.
        if folded < lo:
            folded = lo
        elif folded > hi:
            folded = hi
        return folded.cast[dtype]()

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


def _neg_zero_allowed(min: Float64, max: Float64) -> Bool:
    """Returns whether `-0.0` lies within `[min, max]`.

    IEEE comparison cannot tell `-0.0` from `0.0`, so the sign bit of `min`
    decides: `min=0.0` excludes `-0.0`, while `min=-0.0` or any negative
    `min` includes it.
    """
    var min_is_neg_zero = min == 0.0 and (min.to_bits() >> 63) == 1
    return (min < 0.0 or min_is_neg_zero) and max >= 0.0


def _decode_float_magnitude[dtype: DType](word: UInt64) -> Scalar[dtype]:
    """Decodes a non-negative float from the magnitude code in the low
    `W - 1` bits of `word`, ordered so that smaller codes give simpler floats.

    Parameters:
        dtype: The float dtype to decode.

    Args:
        word: The stream word; bits at and above the sign are ignored.

    Returns:
        The decoded non-negative float, possibly infinite.
    """
    comptime W = UInt64(bit_width_of[dtype]())
    comptime M = UInt64(DType.mantissa_width[dtype]())
    comptime E = UInt64(DType.exponent_width[dtype]())
    comptime bias = UInt64(DType.exponent_bias[dtype]())
    comptime max_exponent = (UInt64(1) << E) - 1
    comptime mantissa_mask = (UInt64(1) << M) - 1
    comptime code_mask = UInt64.MAX >> (65 - W)
    comptime U = _uint_type_of_width[bit_width_of[dtype]()]()

    var code = word & code_mask
    if code == code_mask:
        return Scalar[dtype](from_bits=(max_exponent << M).cast[U]())

    var field = code >> M
    var mantissa = code & mantissa_mask
    if field == 0:
        return mantissa.cast[dtype]()

    var index = field - 1
    var exponent: UInt64
    if index <= bias:
        exponent = index + bias
    else:
        exponent = 2 * bias - index

    # Bits below the binary point are stored reversed, so reducing the code
    # strips its high bits, which hold the finest fractions, first.
    if exponent <= bias:
        mantissa = bit_reverse(mantissa) >> (64 - M)
    elif exponent - bias < M:
        var fraction_bits = M - (exponent - bias)
        var fraction_mask = (UInt64(1) << fraction_bits) - 1
        var fraction = bit_reverse(mantissa & fraction_mask) >> (
            64 - fraction_bits
        )
        mantissa = (mantissa & ~fraction_mask) | fraction

    var bits = (exponent << M) | mantissa
    return Scalar[dtype](from_bits=bits.cast[U]())
