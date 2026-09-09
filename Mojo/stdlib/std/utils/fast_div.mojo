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

"""Implements the fast division algorithm.

This method replaces division by constants with a sequence of shifts and
multiplications, significantly optimizing division performance.
"""

from std.bit import log2_ceil
from std.builtin.dtype import _uint_type_of_width
from std._gpu.intrinsics import mulhi
from std.sys.info import bit_width_of


struct FastDiv[dtype: DType](TrivialRegisterPassable, Writable):
    """Implements fast division for a given type.

    This struct provides optimized division by a constant divisor,
    replacing the division operation with a series of shifts and
    multiplications. This approach significantly improves performance,
    especially in scenarios where division is a frequent operation.

    Parameters:
        dtype: The data type for the division operation.
    """

    comptime uint_type = _uint_type_of_width[bit_width_of[Self.dtype]()]()
    """The unsigned integer type used for the fast division algorithm."""

    var _div: Scalar[Self.uint_type]
    var _mprime: Scalar[Self.uint_type]
    var _sh1: UInt8
    var _sh2: UInt8
    var _is_pow2: Bool
    var _log2_shift: UInt8

    @always_inline
    def __init__(out self, divisor: Int = 1):
        """Initializes FastDiv with the divisor.

        Constraints:
            ConstraintError: If the bitwidth of the type is > 64.

        Args:
            divisor: The divisor to use for fast division.
                Defaults to 1.
        """
        comptime assert (
            bit_width_of[Self.dtype]() <= 64
        ), "larger types are not currently supported"
        self._div = Scalar[Self.uint_type](divisor)

        self._is_pow2 = divisor.is_power_of_two()
        self._log2_shift = UInt8(log2_ceil(divisor))

        # Only compute magic number parameters if not power of 2
        if not self._is_pow2:
            comptime wide_type = _uint_type_of_width[
                bit_width_of[Self.dtype]() * 2
            ]()
            self._mprime = (
                (
                    (
                        Scalar[wide_type](1)
                        << Scalar[wide_type](bit_width_of[Self.dtype]())
                    )
                    * (
                        (
                            Scalar[wide_type](1)
                            << self._log2_shift.cast[wide_type]()
                        )
                        - Scalar[wide_type](divisor)
                    )
                    / Scalar[wide_type](divisor)
                )
            ).cast[Self.uint_type]() + 1
            self._sh1 = min(self._log2_shift, 1)
            self._sh2 = max(self._log2_shift - 1, 0)
        else:
            self._mprime = 0
            self._sh1 = 0
            self._sh2 = 0

    @always_inline
    def __rdiv__(self, other: Scalar[Self.uint_type]) -> Scalar[Self.uint_type]:
        """Divides the other scalar by the divisor.

        Args:
            other: The dividend.

        Returns:
            The result of the division.
        """
        return other / self

    @always_inline
    def __rtruediv__(
        self, other: Scalar[Self.uint_type]
    ) -> Scalar[Self.uint_type]:
        """Divides the other scalar by the divisor (true division).

        Uses the fast division algorithm, with optimized path for power-of-2 divisors.

        Args:
            other: The dividend.

        Returns:
            The result of the division.
        """
        if self._is_pow2:
            # For power-of-2 divisors, just use bit shift
            return other >> self._log2_shift.cast[Self.uint_type]()
        else:
            # FastDiv algorithm for non-power-of-2 divisors.
            var t: Scalar[Self.uint_type]

            comptime if bit_width_of[Self.dtype]() <= 32:
                t = mulhi(
                    self._mprime.cast[.uint32](),
                    other.cast[.uint32](),
                ).cast[Self.uint_type]()
            else:
                t = mulhi(
                    self._mprime.cast[.uint64](),
                    other.cast[.uint64](),
                ).cast[Self.uint_type]()
            return (
                t + ((other - t) >> self._sh1.cast[Self.uint_type]())
            ) >> self._sh2.cast[Self.uint_type]()

    @always_inline
    def __rmod__(self, other: Scalar[Self.uint_type]) -> Scalar[Self.uint_type]:
        """Computes the remainder of division.

        Args:
            other: The dividend.

        Returns:
            The remainder.
        """
        var q = other / self
        return other - (q * self._div)

    @always_inline
    def __divmod__(
        self, other: Scalar[Self.uint_type]
    ) -> Tuple[Scalar[Self.uint_type], Scalar[Self.uint_type]]:
        """Computes both quotient and remainder.

        Args:
            other: The dividend.

        Returns:
            A tuple containing the quotient and remainder.
        """
        var q = other / self
        return q, (other - (q * self._div))

    @no_inline
    def write_to[W: Writer](self, mut writer: W):
        """Writes the FastDiv parameters to a writer.

        Parameters:
            W: The type of the writer.

        Args:
            writer: The writer to which the parameters are written.
        """
        writer.write("div: ", self._div, "\n")
        writer.write("mprime: ", self._mprime, "\n")
        writer.write("sh1: ", self._sh1, "\n")
        writer.write("sh2: ", self._sh2, "\n")
        writer.write("is_pow2: ", self._is_pow2, "\n")
        writer.write("log2_shift: ", self._log2_shift, "\n")
