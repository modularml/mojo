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
"""Implements the string strategy for generating random `String` values in property tests."""

from . import Strategy
from std.testing.prop.random import Rng


__extension String:
    @staticmethod
    def strategy(
        *,
        min_len: Int = 0,
        max_len: Int = Int.MAX,
        unicode: Bool = False,
        only_printable: Bool = False,
    ) raises -> _StringStrategy:
        """Returns a strategy for generating random strings.

        Args:
            min_len: The minimum length of the string.
            max_len: The maximum length of the string.
            unicode: Whether to include unicode characters.
            only_printable: Whether to only include printable characters.

        Returns:
            A strategy for generating random strings.

        Raises:
            If the minimum length is greater than the maximum length.
        """
        return _StringStrategy(
            min_len=min_len,
            max_len=max_len,
            unicode=unicode,
            only_printable=only_printable,
        )

    @staticmethod
    def ascii_strategy(
        *,
        min_len: Int = 0,
        max_len: Int = Int.MAX,
        only_printable: Bool = False,
    ) raises -> _StringStrategy:
        """Returns a strategy for generating random ascii strings.

        Args:
            min_len: The minimum length of the string.
            max_len: The maximum length of the string.
            only_printable: Whether to only include printable characters.

        Returns:
            A strategy for generating random ascii strings.

        Raises:
            If the minimum length is greater than the maximum length.
        """
        return _StringStrategy(
            min_len=min_len,
            max_len=max_len,
            unicode=False,
            only_printable=only_printable,
        )

    @staticmethod
    def utf8_strategy(
        *, min_len: Int = 0, max_len: Int = Int.MAX
    ) raises -> _StringStrategy:
        """Returns a strategy for generating random UTF-8 encoded strings.

        Args:
            min_len: The minimum length of the string.
            max_len: The maximum length of the string.

        Returns:
            A strategy for generating random UTF-8 encoded strings.

        Raises:
            If the minimum length is greater than the maximum length.
        """
        return _StringStrategy(
            min_len=min_len,
            max_len=max_len,
            unicode=True,
            only_printable=False,
        )


struct _StringStrategy(Strategy):
    comptime Value = String

    # min/max length in characters not bytes
    var min_len: Int
    var max_len: Int
    var unicode: Bool
    var only_printable: Bool

    def __init__(
        out self,
        *,
        min_len: Int = 0,
        max_len: Int = Int.MAX,
        unicode: Bool = False,
        only_printable: Bool = False,
    ) raises:
        if min_len < 0 or min_len > max_len:
            raise Error("Invalid min/max for string length")

        comptime MAX_LIST_SIZE = 100

        self.min_len = min_len
        self.max_len = min(max_len, MAX_LIST_SIZE)
        self.unicode = unicode
        self.only_printable = only_printable

    def value(mut self, mut rng: Rng, out s: Self.Value) raises:
        var size = rng.rand_int(min=self.min_len, max=self.max_len)

        s = String(capacity_bytes=size)

        var char_strategy = _CodepointStrategy(
            self.unicode, self.only_printable
        )

        for _ in range(size):
            s.write(char_strategy.value(rng))


__extension Codepoint:
    @staticmethod
    def strategy(
        *, unicode: Bool = False, only_printable: Bool = False
    ) -> _CodepointStrategy:
        """Returns a strategy for generating random Codepoints.

        Args:
            unicode: Whether to include unicode characters.
            only_printable: Whether to only include printable characters.

        Returns:
            A strategy for generating random Codepoints.
        """
        return {unicode, only_printable}


@fieldwise_init
struct _CodepointStrategy:
    comptime Value = Codepoint

    var unicode: Bool
    var only_printable: Bool

    def value(mut self, mut rng: Rng) raises -> Self.Value:
        if self.unicode:
            while True:
                # TODO: Better unicode coverage
                # TODO: Have an option for generating invalid codepoints
                var code_point = rng.rand_scalar[.uint32](
                    min=UInt32(0x0020), max=UInt32(0xFFFF)
                )
                # Skip surrogate range
                if UInt32(0xD800) <= code_point <= UInt32(0xDFFF):
                    continue

                return Codepoint(unsafe_unchecked_codepoint=code_point)
        else:
            # ascii printable characters
            var start: UInt32 = UInt32(32) if self.only_printable else UInt32(0)
            var end: UInt32 = UInt32(126) + UInt32(not self.only_printable)
            return Codepoint(
                unsafe_unchecked_codepoint=rng.rand_scalar[.uint32](
                    min=start, max=end
                )
            )
