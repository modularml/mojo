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

"""Implements the [Fnv1a 64 bit variant](https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function) algorithm as a Hasher type."""

from std.collections import Span
from std.sys import size_of

from .hasher import Hasher


struct Fnv1a(Defaultable, Hasher):
    """Fnv1a is a very simple algorithm with good quality, but sub optimal runtime for long inputs.
    It can be used for comp time hash value generation.

    References:

    - [Fnv1a 64 bit variant](https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function)
    """

    var _value: UInt64

    def __init__(out self):
        """Initialize the hasher."""
        self._value = 0xCBF29CE484222325

    def _update_with_bytes(mut self, data: Span[Byte, _]):
        """Consume provided data to update the internal buffer.

        Args:
            data: Span of bytes to hash.
        """
        for i in range(len(data)):
            self._value ^= data[i].cast[.uint64]()
            self._value *= 0x100000001B3

    def _update_with_simd(mut self, value: SIMD[_, _]):
        """Update the buffer value with new data.

        Args:
            value: Value used for update.
        """

        # number of rounds a single vector value will contribute to a hash
        # values smaller than 8 bytes contribute only once
        # values which are multiple of 8 bytes contribute multiple times
        # e.g. int128 is 16 bytes long and evaluates to 2 rounds
        comptime rounds = max(1, size_of[value.dtype]() // 8)
        var bits = value.to_bits()

        comptime for i in range(value.length):
            var v = bits[i]

            comptime for r in range(rounds):
                self._value ^= (v >> type_of(v)(r * 64)).cast[.uint64]()
                self._value *= 0x100000001B3

    def update(mut self, value: ImmSpan[Byte, _]):
        """Update the buffer value with new hashable value.

        Args:
            value: Value used for update.
        """
        self._update_with_bytes(value)

    def finish(var self) -> UInt64:
        """Computes the hash value based on all the previously provided data.

        Returns:
            Final hash value.
        """
        return self._value
