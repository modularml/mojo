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
"""Vendor-neutral element encodings of an OCP microscaling (MX) operand."""

from std.os import abort

from linalg.fp6_utils import FP6Format


@fieldwise_init
struct MXFormat(Equatable, TrivialRegisterPassable):
    """Names the element encoding of an OCP microscaling (MX) operand.

    An MX operand is a block of elements sharing one `float8_e8m0fnu` scale;
    this type names the encoding of the *elements*, which is all that varies
    between MXFP4, MXFP6 and MXFP8.

    Deliberately not tied to any vendor: AMD CDNA4 selects these through its
    `f8f6f4` MFMA selector and NVIDIA through its own block-scaled MMA, so the
    numeric values here are ours and each backend maps them to its own encoding
    (see `CDNA4F8F6F4MatrixFormat.__init__` for the CDNA4 mapping). Kernels that
    are not hardware-specific -- EP dispatch packing, quantization, layout
    math -- should carry this type rather than a vendor selector.

    Prefer deriving it from a dtype with `from_dtype`, so the format travels
    with the data instead of being inferred at each call site.
    """

    var _value: Int32

    comptime FP8_E4M3 = Self(0)
    comptime FP8_E5M2 = Self(1)
    comptime FP6_E2M3 = Self(2)
    comptime FP6_E3M2 = Self(3)
    comptime FP4_E2M1 = Self(4)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @staticmethod
    def from_dtype[dtype: DType]() -> Self:
        """Returns the MX format a quantized operand dtype denotes.

        Parameters:
            dtype: The storage dtype of the quantized elements.

        Returns:
            The corresponding MX element format.
        """
        if dtype == .float6_e2m3fn:
            return Self.FP6_E2M3
        if dtype == .float6_e3m2fn:
            return Self.FP6_E3M2
        if dtype == .float8_e4m3fn:
            return Self.FP8_E4M3
        if dtype == .float8_e5m2:
            return Self.FP8_E5M2
        if dtype == .uint8:
            return Self.FP4_E2M1
        abort("dtype does not denote an MX element format")

    @staticmethod
    def from_name[name: StaticString]() -> Self:
        """Returns the MX format a graph-parameter name denotes.

        Parameters:
            name: One of `mxfp4`, `mxfp6`, `mxfp6_e3m2`, `mxfp8`, `mxfp8_e5m2`.
                The bare `mxfp6` and `mxfp8` select the encoding each format
                defaults to, E2M3 and E4M3.

        Returns:
            The corresponding MX element format.
        """
        if name == "mxfp4":
            return Self.FP4_E2M1
        if name == "mxfp6":
            return Self.FP6_E2M3
        if name == "mxfp6_e3m2":
            return Self.FP6_E3M2
        if name == "mxfp8":
            return Self.FP8_E4M3
        if name == "mxfp8_e5m2":
            return Self.FP8_E5M2
        abort("unknown MX format name: " + String(name))

    def bits_per_element(self) -> Int:
        """Returns how many bits one element of this format occupies.

        This is the payload width, not the fragment width: 32 FP6 elements
        occupy 24 bytes but travel in a wider fragment on some hardware.

        Returns:
            The element width in bits.
        """
        if self == Self.FP8_E4M3 or self == Self.FP8_E5M2:
            return 8
        if self == Self.FP6_E2M3 or self == Self.FP6_E3M2:
            return 6
        if self == Self.FP4_E2M1:
            return 4
        abort("invalid MX format")

    def is_fp6(self) -> Bool:
        """Returns whether this is one of the two six-bit encodings.

        Returns:
            True for FP6_E2M3 and FP6_E3M2.
        """
        return self == Self.FP6_E2M3 or self == Self.FP6_E3M2

    def fp6_format(self) -> FP6Format:
        """Returns the FP6 element encoding, for the FP6 packing routines.

        Both FP6 encodings occupy six bits and pack four codes into three
        bytes, so nothing downstream can tell them apart from the bytes alone
        -- this format is the only record of which one they hold.

        Returns:
            The matching `FP6Format`.
        """
        if self == Self.FP6_E2M3:
            return FP6Format.E2M3
        if self == Self.FP6_E3M2:
            return FP6Format.E3M2
        abort("not an FP6 format")
