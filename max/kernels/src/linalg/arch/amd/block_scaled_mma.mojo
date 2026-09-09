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
"""CDNA4 block-scaled MFMA wrappers for the LLVM `f8f6f4` family.

These wrappers are shared architecture helpers for AMD block-scaled kernels.
The MFMA inputs use LLVM's `f8f6f4` operand-format selector so callers can pick
FP8, FP6, or FP4 encodings per operand, while the scale inputs remain packed
E8M0 words.
"""

from std.memory import bitcast
from std.os import abort
from std.sys import llvm_intrinsic, size_of
from std.sys.info import _cdna_4_or_newer

from linalg.mx_format import MXFormat


@fieldwise_init
struct CDNA4F8F6F4MatrixFormat(Equatable, TrivialRegisterPassable):
    """Represents the CDNA4 `f8f6f4` operand format selector."""

    var _value: Int32

    comptime FLOAT8_E4M3 = Self(0)  # MXFP8 E4M3
    comptime FLOAT8_E5M2 = Self(1)  # MXFP8 E5M2
    comptime FLOAT6_E2M3 = Self(2)  # MXFP6 E2M3
    comptime FLOAT6_E3M2 = Self(3)  # MXFP6 E3M2
    comptime FLOAT4_E2M1 = Self(4)  # MXFP4 E2M1

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __init__(out self, mx_format: MXFormat):
        """Maps a vendor-neutral MX element format onto this CDNA4 selector.

        Written out rather than passing the value through, so the neutral type's
        numbering stays independent of what the hardware happens to encode.

        Args:
            mx_format: The MX element format to select.
        """
        if mx_format == MXFormat.FP8_E4M3:
            self = Self.FLOAT8_E4M3
        elif mx_format == MXFormat.FP8_E5M2:
            self = Self.FLOAT8_E5M2
        elif mx_format == MXFormat.FP6_E2M3:
            self = Self.FLOAT6_E2M3
        elif mx_format == MXFormat.FP6_E3M2:
            self = Self.FLOAT6_E3M2
        elif mx_format == MXFormat.FP4_E2M1:
            self = Self.FLOAT4_E2M1
        else:
            abort("invalid MX format")

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def bits_per_element(self) -> Int:
        """Returns how many bits one operand element occupies.

        Note this is the *payload* width, not the fragment width: 32 FP6
        elements occupy 24 bytes but travel in a 32-byte fragment. Use
        `simd_width` for the fragment.

        Returns:
            The element width in bits.
        """
        if self == CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3:
            return 8
        if self == CDNA4F8F6F4MatrixFormat.FLOAT8_E5M2:
            return 8
        if self == CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3:
            return 6
        if self == CDNA4F8F6F4MatrixFormat.FLOAT6_E3M2:
            return 6
        if self == CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1:
            return 4
        abort("invalid matrix format")

    def simd_width(self) -> SIMDLength:
        """Returns the operand fragment width, in bytes, this format expects.

        FP6 holds only 24 payload bytes, but 24 is not a power of two, so
        it asks for 32. InstCombine narrows the operand back down, so the
        padding costs nothing. Stage FP6 loads as 24 bytes, not 32.

        Returns:
            The fragment width in bytes.
        """
        if self == CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3:
            return 32
        if self == CDNA4F8F6F4MatrixFormat.FLOAT8_E5M2:
            return 32
        if self == CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3:
            return 32
        if self == CDNA4F8F6F4MatrixFormat.FLOAT6_E3M2:
            return 32
        if self == CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1:
            return 16
        abort("invalid matrix format")


@always_inline
def cdna4_block_scaled_mfma[
    a_scale_byte_index: Int32,
    b_scale_byte_index: Int32,
    a_matrix_format: CDNA4F8F6F4MatrixFormat,
    b_matrix_format: CDNA4F8F6F4MatrixFormat,
](
    mut d: SIMD[.float32, _],
    a: SIMD[.uint8, _],
    b: SIMD[.uint8, _],
    packed_scale_word_a: Int32,
    packed_scale_word_b: Int32,
):
    """Executes a CDNA4 `f8f6f4` block-scaled MFMA, inferring the MMA shape
    from the accumulator width (16 lanes -> 32x32x64, 4 lanes -> 16x16x128).

    `a_scale_byte_index` and `b_scale_byte_index` select byte 0..3 from the
    packed 32-bit E8M0 `packed_scale_word_a` and `packed_scale_word_b` inputs.
    """
    comptime assert (
        _cdna_4_or_newer()
    ), "CDNA4 block-scaled MFMA wrappers require CDNA4 or newer"
    comptime assert (
        d.length == 16 or d.length == 4
    ), "accumulator width must be 16 (32x32x64) or 4 (16x16x128)"
    comptime assert a.length == a_matrix_format.simd_width(), "bad a width"
    comptime assert b.length == b_matrix_format.simd_width(), "bad b width"

    comptime intrinsic = (
        "llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4" if d.length
        == 16 else "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4"
    )
    # The ISA names the scale-byte selector {OP_SEL_HI, OP_SEL}. Together
    # they choose which packed E8M0 byte the MFMA consumes from each 32-bit
    # scale word: 0 -> bits 7:0, 1 -> 15:8, 2 -> 23:16, 3 -> 31:24.
    d = llvm_intrinsic[
        intrinsic,
        SIMD[.float32, d.length],
    ](
        bitcast[.int32, a.length // size_of[DType.int32]()](a),
        bitcast[.int32, b.length // size_of[DType.int32]()](b),
        d,
        a_matrix_format,
        b_matrix_format,
        a_scale_byte_index,
        packed_scale_word_a,
        b_scale_byte_index,
        packed_scale_word_b,
    )
