# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Implement UTF-16 utils."""

from std.bit.mask import splat
from std.sys.info import bit_width_of
from std.sys.intrinsics import likely, unlikely
from std.collections.string._unicode import BIGGEST_UNICODE_CODEPOINT


def _decode_utf16[
    dtype: DType,
    //,
    *,
    strict: Bool = True,
    replace: Codepoint = Codepoint.ord("�"),
](*, from_utf16: Span[mut=False, Scalar[dtype], ...]) raises -> String:
    comptime assert dtype.is_integral(), "The dtype must be integral"
    comptime assert bit_width_of[dtype]() > 8, (
        "Decoding UTF-16 from a buffer of bytes is ambiguous "
        "bitcast it to UInt16 or use the codepoints decoder if they are "
        "independent codepoints"
    )
    comptime replace_str = String(replace)
    comptime replace_length = UInt(replace_str.byte_length())

    # TODO: this could use reduce_sum with a function that calculates the utf8
    # length from the utf16 codepoints. Performance needs to be measured
    var utf16_len = UInt(len(from_utf16))
    var result = String(capacity_bytes=2 * Int(utf16_len))
    var utf16_ptr = from_utf16.unsafe_ptr()
    var utf16_idx = UInt(0)
    var utf8_offset = UInt(0)

    while utf16_idx < utf16_len:
        comptime cont_bytes = 0b1000_0000
        var c_og = utf16_ptr[unsafe_offset=utf16_idx]
        var c = c_og.cast[DType.uint32]()
        var ptr = result.unsafe_ptr_mut().unsafe_offset(Int(utf8_offset))
        var num_bytes: UInt
        # high surrogate marker 0xD800 == 0b11_0110 << 10
        var is_high_surrogate = (c >> 10).eq(0b11_0110)
        var is_full_surrogate = Scalar[DType.bool](False)
        comptime is_bigger_than_16bit = bit_width_of[dtype]() > 16
        if unlikely(is_bigger_than_16bit and c_og > 0xFFFF):
            comptime if strict:
                raise Error("Invalid UTF-16 at index: ", utf16_idx)
            else:
                result += replace_str
                num_bytes = replace_length
        elif likely(c < cont_bytes):  # ASCII
            num_bytes = 1
            ptr[unsafe_offset=0] = c.cast[DType.uint8]()
        elif c <= 0xFF:  # latin-1
            comptime utf8_length_2_prefix = 0b1100_0000
            comptime low_6b = 0b0011_1111
            num_bytes = 2
            ptr[unsafe_offset=0] = ((c >> 6) | utf8_length_2_prefix).cast[
                DType.uint8
            ]()
            ptr[unsafe_offset=1] = ((c & low_6b) | cont_bytes).cast[
                DType.uint8
            ]()
        else:
            # get lower 10 bits
            comptime low_10b = UInt32(0b0011_1111_1111)
            var has_more = utf16_idx < utf16_len - 1
            var c2 = utf16_ptr[unsafe_offset=utf16_idx + UInt(has_more)].cast[
                DType.uint32
            ]()
            # low surrogate marker 0xDC00 == 0b0011_0111 << 10
            var is_low_surrogate = (c2 >> 10).eq(0b0011_0111)
            is_full_surrogate = is_high_surrogate & is_low_surrogate
            var any_pair = is_high_surrogate | is_low_surrogate

            if unlikely(Bool(any_pair & ~is_full_surrogate)):
                comptime if strict:
                    raise Error("Unpaired surrogate at index: ", utf16_idx)
                else:
                    result += replace_str
                    num_bytes = replace_length
            else:
                comptime offset = 2**16
                c = c if not is_full_surrogate else (
                    offset + (((c & low_10b) << 10) | (c2 & low_10b))
                )
                var cp = Codepoint(unsafe_unchecked_codepoint=c)
                var num = cp.unsafe_write_utf8[
                    optimize_ascii=False, branchless=True
                ](ptr)
                num_bytes = UInt(num)

        utf8_offset += num_bytes
        utf16_idx += UInt(1) << UInt(is_full_surrogate.cast[DType.uint]())
        result._set_byte_length(Int(utf8_offset))

    return result^
