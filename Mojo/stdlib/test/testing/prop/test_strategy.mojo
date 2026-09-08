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

from std.testing import (
    assert_equal,
    assert_true,
    TestSuite,
)
from test_utils import DelCounter
from std.testing.prop import Rng, Strategy
from std.testing.prop.strategy.simd_strategy import *
from std.testing.prop.strategy.string_strategy import *
from std.testing.prop.strategy.list_strategy import *
from std.collections.string._utf8 import _is_valid_utf8


@fieldwise_init
struct TestStruct(Copyable):
    var n: Int

    @staticmethod
    def strategy() -> TestStructStrategy:
        return TestStructStrategy(0, 10)


@fieldwise_init
struct TestStructStrategy(Movable, Strategy):
    comptime Value = TestStruct

    var min: Int
    var max: Int

    def value(mut self, mut rng: Rng) raises -> Self.Value:
        return {rng.rand_int(min=self.min, max=self.max)}


def test_strategy_returns_correct_value() raises:
    var strategy = TestStruct.strategy()
    var rng = Rng(seed=1234)
    for _ in range(10):
        var n = strategy.value(rng).n
        assert_true(n >= 0 and n <= 10)


def test_simd_strategy() raises:
    var min = Int64(-10)
    var max = Int64(30)
    var strat = SIMD[.int64, 8].strategy(min=min, max=max)
    var rng = Rng(seed=1234)
    for _ in range(10):
        var v = strat.value(rng)
        assert_true(all(v.ge(min)))
        assert_true(all(v.le(max)))


def test_string_ascii_strategy() raises:
    var s = String.ascii_strategy(min_len=1, max_len=10, only_printable=True)
    var rng = Rng(seed=1234)
    for _ in range(10):
        var s = s.value(rng)
        assert_true(1 <= s.byte_length() <= 10)
        assert_true(StringSlice(s).is_ascii_printable())

    s = String.ascii_strategy(min_len=1, max_len=10, only_printable=False)
    rng = Rng(seed=1234)
    for _ in range(10):
        var s = s.value(rng)
        assert_true(1 <= s.byte_length() <= 10)
        for c in s.codepoints():
            assert_true(c.to_u32() <= UInt32(127))


def test_string_utf8_strategy() raises:
    var s = String.utf8_strategy(min_len=1, max_len=10)
    var rng = Rng(seed=1234)
    for _ in range(10):
        var s = s.value(rng)
        assert_true(1 <= len(s.codepoints()) <= 10, s)
        assert_true(_is_valid_utf8(s.as_bytes()), s)


def test_list_strategy() raises:
    var s = List[Int8].strategy(
        Int8.strategy(min=0, max=100), min_len=2, max_len=45
    )
    var rng = Rng(seed=1234)

    for _ in range(10):
        var l = s.value(rng)

        assert_true(2 <= len(l) <= 45)

        for el in l:
            assert_true(Int8(0) <= el <= Int8(100))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
