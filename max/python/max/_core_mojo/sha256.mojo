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

"""SHA-256 (FIPS 180-4) implementation for KV cache prefix hashing."""

from std.collections import Array, Span
from std.bit import rotate_bits_right
from std.collections.array import Array
from std.builtin.globals import global_constant

# FIPS 180-4 initial hash values

comptime _H0: Array[UInt32, 8] = [
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
]

comptime _K: Array[UInt32, 64] = [
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
]


@always_inline
def _ch(x: UInt32, y: UInt32, z: UInt32) -> UInt32:
    """Choice function: `(x & y) ^ (~x & z)`."""
    return (x & y) ^ (~x & z)


@always_inline
def _maj(x: UInt32, y: UInt32, z: UInt32) -> UInt32:
    """Majority function: `(x & y) ^ (x & z) ^ (y & z)`."""
    return (x & y) ^ (x & z) ^ (y & z)


@always_inline
def _big_sigma0(x: UInt32) -> UInt32:
    return (
        rotate_bits_right[2](x)
        ^ rotate_bits_right[13](x)
        ^ rotate_bits_right[22](x)
    )


@always_inline
def _big_sigma1(x: UInt32) -> UInt32:
    return (
        rotate_bits_right[6](x)
        ^ rotate_bits_right[11](x)
        ^ rotate_bits_right[25](x)
    )


@always_inline
def _small_sigma0(x: UInt32) -> UInt32:
    return rotate_bits_right[7](x) ^ rotate_bits_right[18](x) ^ (x >> 3)


@always_inline
def _small_sigma1(x: UInt32) -> UInt32:
    return rotate_bits_right[17](x) ^ rotate_bits_right[19](x) ^ (x >> 10)


@always_inline
def _load_be_u32(data: Span[Byte, _], offset: Int) -> UInt32:
    """Load a 32-bit big-endian integer from the data."""
    return (
        (data[offset].cast[.uint32]() << 24)
        | (data[offset + 1].cast[.uint32]() << 16)
        | (data[offset + 2].cast[.uint32]() << 8)
        | (data[offset + 3].cast[.uint32]())
    )


def _compress(mut h: Array[UInt32, 8], block: Span[Byte, _]):
    """Process one 64-byte block of data into the hash state."""
    var w = Array[UInt32, 64](fill=UInt32(0))
    # Load the first 16 words from the block
    for t in range(16):
        w[t] = _load_be_u32(block, t * 4)
    # Extend to 64 words
    for t in range(16, 64):
        w[t] = (
            _small_sigma1(w[t - 2])
            + w[t - 7]
            + _small_sigma0(w[t - 15])
            + w[t - 16]
        )
    # Working variables
    var a = h[0]
    var b = h[1]
    var c = h[2]
    var d = h[3]
    var e = h[4]
    var f = h[5]
    var g = h[6]
    var hh = h[7]
    # Compression loop
    for t in range(64):
        var t1 = (
            hh + _big_sigma1(e) + _ch(e, f, g) + global_constant[_K]()[t] + w[t]
        )
        var t2 = _big_sigma0(a) + _maj(a, b, c)
        hh = g
        g = f
        f = e
        e = d + t1
        d = c
        c = b
        b = a
        a = t1 + t2

    # Update state
    h[0] = a + h[0]
    h[1] = b + h[1]
    h[2] = c + h[2]
    h[3] = d + h[3]
    h[4] = e + h[4]
    h[5] = f + h[5]
    h[6] = g + h[6]
    h[7] = hh + h[7]


def sha256(data: Span[Byte, _]) -> Array[UInt8, 32]:
    """Compute the SHA-256 hash of the given data. Returns a 32-byte hash."""
    var h: Array[UInt32, 8] = materialize[_H0]()

    # Process the data in 64-byte blocks
    var n = len(data)
    var num_full = n // 64
    for i in range(num_full):
        _compress(h, data[i * 64 : (i + 1) * 64])

    # Build final padded block
    var rem = n - (num_full * 64)
    var pad = Array[UInt8, 128](fill=UInt8(0))
    for i in range(rem):
        pad[i] = data[num_full * 64 + i]
    pad[rem] = UInt8(0x80)
    var bit_len = UInt64(n * 8)
    var pad_len = 128 if rem >= 56 else 64
    # Write 64-bit length into last 8 bytes
    for i in range(8):
        pad[pad_len - 1 - i] = UInt8((bit_len >> UInt64(i * 8)) & 0xFF)
    var pad_ptr: Pointer[UInt8, origin_of(pad)] = pad.unsafe_ptr()
    _compress(h, Span[Byte, _](unsafe_ptr=pad_ptr, length=64))
    if pad_len == 128:
        _compress(
            h, Span[Byte, _](unsafe_ptr=pad_ptr.unsafe_offset(64), length=64)
        )

    # Serialize the hash to 32 bytes
    var out = Array[UInt8, 32](fill=UInt8(0))
    for i in range(8):
        out[i * 4 + 0] = UInt8((h[i] >> 24) & 0xFF)
        out[i * 4 + 1] = UInt8((h[i] >> 16) & 0xFF)
        out[i * 4 + 2] = UInt8((h[i] >> 8) & 0xFF)
        out[i * 4 + 3] = UInt8(h[i] & 0xFF)
    return out^
