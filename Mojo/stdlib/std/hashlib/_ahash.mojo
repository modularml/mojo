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

from std.sys import size_of
from std.bit import rotate_bits_left
from std.memory import bitcast
from std.collections import Span

from .hasher import Hasher

comptime U256 = SIMD[.uint64, 4]
comptime U128 = SIMD[.uint64, 2]
comptime MULTIPLE = 6364136223846793005
comptime ROT = 23


@always_inline
def _folded_multiply(lhs: UInt64, rhs: UInt64) -> UInt64:
    """A fast function to emulate a folded multiply of two 64 bit uints.

    Args:
        lhs: 64 bit uint.
        rhs: 64 bit uint.

    Returns:
        A value which is similar in its bitpattern to result of a folded multiply.
    """
    # Extend to 128 bits and multiply.
    var m = lhs.cast[.uint128]() * rhs.cast[.uint128]()
    # Extract the high and low 64 bits.
    var res = bitcast[.uint64, 2](m)
    return res[0] ^ res[1]


@always_inline
def _read_small(data: ImmPointer[UInt8, _], length: Int) -> U128:
    """Produce a `SIMD[DType.uint64, 2]` value from data which is smaller than or equal to `8` bytes.

    Args:
        data: Pointer to the byte array.
        length: The byte array length.

    Returns:
        Returns a SIMD[.uint64, 2] value.
    """
    if length >= 2:
        if length >= 4:
            # len 4-8
            var a = data.unsafe_bitcast[UInt32]().unsafe_load().cast[.uint64]()
            var b = (
                data.unsafe_offset(length - 4)
                .unsafe_bitcast[UInt32]()
                .unsafe_load()
                .cast[.uint64]()
            )
            return U128(a, b)
        else:
            # len 2-3
            var a = data.unsafe_bitcast[UInt16]().unsafe_load().cast[.uint64]()
            var b = data.unsafe_offset(length - 1).unsafe_load().cast[.uint64]()
            return U128(a, b)
    else:
        # len 0-1
        if length > 0:
            var a = data.unsafe_load().cast[.uint64]()
            return U128(a, a)
        else:
            return U128(0, 0)


struct AHasher[key: U256](Defaultable, Hasher):
    """Adopted AHash algorithm which produces fast and high quality hash value by
    implementing `Hasher` trait.

    References:

    - [AHasher Implementation in Rust](https://github.com/tkaitchuck/aHash)
    """

    var buffer: UInt64
    var pad: UInt64
    var extra_keys: U128

    def __init__(out self):
        """Initialize the hasher."""
        comptime pi_key = Self.key ^ U256(
            0x243F_6A88_85A3_08D3,
            0x1319_8A2E_0370_7344,
            0xA409_3822_299F_31D0,
            0x082E_FA98_EC4E_6C89,
        )
        self.buffer = pi_key[0]
        self.pad = pi_key[1]
        self.extra_keys = U128(pi_key[2], pi_key[3])

    def __init__(out self, seed: U256):
        """Initializes the hasher with a runtime seed mixed into its keyed state.

        Same derivation as the zero-argument initializer, but XORs in
        `seed` first. An all-zero `seed` reproduces that output exactly.

        Args:
            seed: Runtime bits mixed into the hasher's keyed state, e.g. a
                per-tenant or per-request salt.
        """
        var pi_key = (
            Self.key
            ^ U256(
                0x243F_6A88_85A3_08D3,
                0x1319_8A2E_0370_7344,
                0xA409_3822_299F_31D0,
                0x082E_FA98_EC4E_6C89,
            )
            ^ seed
        )
        self.buffer = pi_key[0]
        self.pad = pi_key[1]
        self.extra_keys = U128(pi_key[2], pi_key[3])

    @always_inline
    def _update(mut self, new_data: UInt64):
        """Update the buffer value with new data.

        Args:
            new_data: Value used for update.
        """
        self.buffer = _folded_multiply(new_data ^ self.buffer, MULTIPLE)

    @always_inline
    def _large_update(mut self, new_data: U128):
        """Update the buffer value with new data.

        Args:
            new_data: Value used for update.
        """
        var xored = new_data ^ self.extra_keys
        var combined = _folded_multiply(xored[0], xored[1])
        self.buffer = rotate_bits_left[ROT]((self.buffer + self.pad) ^ combined)

    def _update_with_bytes(mut self, data: Span[Byte, _]):
        """Consume provided data to update the internal buffer.

        Args:
            data: Span of bytes to hash.
        """
        var length = len(data)
        var ptr = data.unsafe_ptr()
        self.buffer = (self.buffer + UInt64(length)) * MULTIPLE
        if length > 8:
            if length > 16:
                var tail = (
                    ptr.unsafe_offset(length - 16)
                    .unsafe_bitcast[UInt64]()
                    .unsafe_load[width=2]()
                )
                self._large_update(tail)
                var offset = 0
                while length - offset > 16:
                    var block = (
                        ptr.unsafe_offset(offset)
                        .unsafe_bitcast[UInt64]()
                        .unsafe_load[width=2]()
                    )
                    self._large_update(block)
                    offset += 16
            else:
                var a = ptr.unsafe_bitcast[UInt64]().unsafe_load()
                var b = (
                    ptr.unsafe_offset(length - 8)
                    .unsafe_bitcast[UInt64]()
                    .unsafe_load()
                )
                self._large_update(U128(a, b))
        else:
            var value = _read_small(ptr, length)
            self._large_update(value)

    def _update_with_simd(mut self, new_data: SIMD[_, _]):
        """Update the buffer value with new data.

        Args:
            new_data: Value used for update.
        """

        # number of rounds a single vector value will contribute to a hash
        # values smaller than 8 bytes contribute only once
        # values which are multiple of 8 bytes contribute multiple times
        # e.g. int128 is 16 bytes long and evaluates to 2 rounds
        comptime rounds = max(1, size_of[new_data.dtype]() // 8)

        comptime if rounds == 1:
            # vector values are not bigger than 8 bytes each
            var u64 = new_data.to_bits[.uint64]()

            comptime if u64.length == 1:
                self._update(u64[0])
            else:
                comptime for i in range(0, u64.length, 2):
                    self._large_update(U128(u64[i], u64[i + 1]))
        else:
            # vector values will contribute to hash in multiple rounds
            comptime for i in range(new_data.length):
                var v = new_data[i]
                comptime assert size_of[v.dtype]() > 8 and v.dtype.is_integral()

                comptime for r in range(0, rounds, 2):
                    var u64_1 = (v >> Scalar[new_data.dtype](r * 64)).cast[
                        DType.uint64
                    ]()
                    var u64_2 = (
                        v >> Scalar[new_data.dtype]((r + 1) * 64)
                    ).cast[.uint64]()
                    self._large_update(U128(u64_1, u64_2))

    def update(mut self, value: ImmSpan[Byte, _]):
        """Update the buffer value with new hashable value.

        Args:
            value: Value used for update.
        """
        self._update_with_bytes(value)

    @always_inline
    def finish(var self) -> UInt64:
        """Computes the hash value based on all the previously provided data.

        Returns:
            Final hash value.
        """
        var rot = self.buffer & 63
        var folded = _folded_multiply(self.buffer, self.pad)
        return (folded << rot) | (folded >> ((UInt64(64) - rot) & UInt64(63)))


def hash_seeded_bytes(data: ImmPointer[UInt8, _], n: Int, seed: U256) -> UInt64:
    """Hashes a sequence of bytes with a runtime seed mixed into AHash's keyed state.

    Seeded analogue of `hash(bytes, n)`. An all-zero `seed` reproduces
    `hash(data, n)` exactly.

    Args:
        data: Pointer to the byte sequence to hash.
        n: The number of bytes to hash.
        seed: Runtime bits mixed into the hasher's keyed state, e.g. a
            per-tenant or per-request salt.

    Returns:
        A 64-bit integer hash value.
    """
    var hasher = AHasher[U256(0)](seed)
    hasher._update_with_bytes(Span(unsafe_ptr=data, length=n))
    var value = hasher^.finish()
    return value


def hash_seeded[T: Hashable](value: T, seed: U256) -> UInt64:
    """Hashes a `Hashable` value with a runtime seed mixed into AHash's keyed state.

    Seeded analogue of `hash(hashable)`. An all-zero `seed` reproduces
    `hash(value)` exactly.

    Parameters:
        T: Any Hashable type.

    Args:
        value: The input data to hash.
        seed: Runtime bits mixed into the hasher's keyed state, e.g. a
            per-tenant or per-request salt.

    Returns:
        A 64-bit integer hash value.
    """
    var hasher = AHasher[U256(0)](seed)
    value.__hash__(hasher)
    var result = hasher^.finish()
    return result
