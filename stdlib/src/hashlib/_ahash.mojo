# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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

from bit import byte_swap
from bit import rotate_bits_left
from memory import UnsafePointer
from ._hasher import _Hasher, _HashableWithHasher

alias U256 = SIMD[DType.uint64, 4]
alias U128 = SIMD[DType.uint64, 2]
alias MULTIPLE = 6364136223846793005
alias ROT = 23


@always_inline
fn _folded_multiply(lhs: UInt64, rhs: UInt64) -> UInt64:
    """A fast function to emulate a folded multiply of two 64 bit uints.
    Used because we don't have UInt128 type.

    Args:
        lhs: 64 bit uint.
        rhs: 64 bit uint.

    Returns:
        A value which is similar in its bitpattern to result of a folded multply.
    """
    var b1 = lhs * byte_swap(rhs)
    var b2 = byte_swap(lhs) * (~rhs)
    return b1 ^ byte_swap(b2)


@always_inline
fn _read_small(data: UnsafePointer[UInt8], length: Int) -> U128:
    """Produce a `SIMD[DType.uint64, 2]` value from data which is smaller than or equal to `8` bytes.

    Args:
        data: Pointer to the byte array.
        length: The byte array length.

    Returns:
        Returns a SIMD[DType.uint64, 2] value.
    """
    if length >= 2:
        if length >= 4:
            # len 4-8
            var a = data.bitcast[DType.uint32]().load().cast[DType.uint64]()
            var b = data.offset(length - 4).bitcast[DType.uint32]().load().cast[
                DType.uint64
            ]()
            return U128(a, b)
        else:
            # len 2-3
            var a = data.bitcast[DType.uint16]().load().cast[DType.uint64]()
            var b = data.offset(length - 1).load().cast[DType.uint64]()
            return U128(a, b)
    else:
        # len 0-1
        if length > 0:
            var a = data.load().cast[DType.uint64]()
            return U128(a, a)
        else:
            return U128(0, 0)


struct AHasher[key: U256](_Hasher):
    """Adopted AHash algorithm which produces fast and high quality hash value by
    implementing `_Hasher` trait.

    References:

    - [AHasher Implementation in Rust](https://github.com/tkaitchuck/aHash)
    """

    var buffer: UInt64
    var pad: UInt64
    var extra_keys: U128

    fn __init__(inout self):
        """Initialize the hasher."""
        alias pi_key = key ^ U256(
            0x243F_6A88_85A3_08D3,
            0x1319_8A2E_0370_7344,
            0xA409_3822_299F_31D0,
            0x082E_FA98_EC4E_6C89,
        )
        self.buffer = pi_key[0]
        self.pad = pi_key[1]
        self.extra_keys = U128(pi_key[2], pi_key[3])

    @always_inline
    fn _update(inout self, new_data: UInt64):
        """Update the buffer value with new data.

        Args:
            new_data: Value used for update.
        """
        self.buffer = _folded_multiply(new_data ^ self.buffer, MULTIPLE)

    @always_inline
    fn _large_update(inout self, new_data: U128):
        """Update the buffer value with new data.

        Args:
            new_data: Value used for update.
        """
        var xored = new_data ^ self.extra_keys
        var combined = _folded_multiply(xored[0], xored[1])
        self.buffer = rotate_bits_left[ROT]((self.buffer + self.pad) ^ combined)

    fn _update_with_bytes(inout self, data: UnsafePointer[UInt8], length: Int):
        """Consume provided data to update the internal buffer.

        Args:
            data: Pointer to the byte array.
            length: The length of the byte array.
        """
        self.buffer = (self.buffer + length) * MULTIPLE
        if length > 8:
            if length > 16:
                var tail = data.offset(length - 16).bitcast[
                    DType.uint64
                ]().load[width=2]()
                self._large_update(tail)
                var offset = 0
                while length - offset > 16:
                    var block = data.offset(offset).bitcast[
                        DType.uint64
                    ]().load[width=2]()
                    self._large_update(block)
                    offset += 16
            else:
                var a = data.bitcast[DType.uint64]().load()
                var b = data.offset(length - 8).bitcast[DType.uint64]().load()
                self._large_update(U128(a, b))
        else:
            var value = _read_small(data, length)
            self._large_update(value)

    fn _update_with_simd(inout self, new_data: SIMD[_, _]):
        """Update the buffer value with new data.

        Args:
            new_data: Value used for update.
        """
        var v64: SIMD[DType.uint64, new_data.size]

        @parameter
        if new_data.type.is_floating_point():
            v64 = new_data._float_to_bits[DType.uint64]()
        else:
            v64 = new_data.cast[DType.uint64]()

        @parameter
        if v64.size == 1:
            self._update(v64[0])
        else:

            @parameter
            for i in range(0, v64.size, 2):
                self._large_update(U128(v64[i], v64[i + 1]))

    fn update[T: _HashableWithHasher](inout self, value: T):
        """Update the buffer value with new hashable value.

        Args:
            value: Value used for update.
        """
        value.__hash__(self)

    @always_inline
    fn finish(owned self) -> UInt64:
        """Computes the hash value based on all the previously provided data.

        Returns:
            Final hash value.
        """
        var rot = self.buffer & 63
        var folded = _folded_multiply(self.buffer, self.pad)
        return (folded << rot) | (folded >> (64 - rot))
