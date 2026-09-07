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
"""Implements the list strategy for generating random `List` values in property tests."""

from . import Strategy
from std.testing.prop.random import Rng
from std.builtin.simd import SIMD
from std.collections import List


__extension List:
    @staticmethod
    def strategy[
        StrategyType: Strategy
    ](
        var strategy: StrategyType,
        *,
        min_len: Int = 0,
        max_len: Int = Int.MAX,
    ) raises -> _ListStrategy[StrategyType]:
        """Returns a strategy for generating lists with random elements.

        Parameters:
            StrategyType: The type of the strategy to use for generating random elements.

        Args:
            strategy: The strategy to use for generating random elements.
            min_len: The minimum length of the list.
            max_len: The maximum length of the list.

        Returns:
            A strategy for generating lists with random elements.

        Raises:
            If the minimum length is greater than the maximum length.
        """
        return _ListStrategy(strategy^, min_len=min_len, max_len=max_len)


struct _ListStrategy[T: Strategy](Movable, Strategy):
    comptime Value = List[Self.T.Value]

    var _strat: Self.T
    var _min_len: Int
    var _max_len: Int

    def __init__(
        out self,
        var strategy: Self.T,
        *,
        min_len: Int = 0,
        max_len: Int = Int.MAX,
    ) raises:
        if min_len < 0 or min_len > max_len:
            raise Error("Invalid min/max for list length")

        # TODO: Make this configurable for other collection types via
        # a property test config value.
        comptime MAX_LIST_SIZE = 100

        self._strat = strategy^
        self._min_len = min_len
        self._max_len = min(max_len, MAX_LIST_SIZE)

    # TODO: Provide more consistent "corner case" values.
    # Empty list, single element list, max size list, etc...
    def value(mut self, mut rng: Rng) raises -> Self.Value:
        var result = List[Self.T.Value](capacity=self._min_len)

        while len(result) < self._min_len:
            result.append(self._strat.value(rng))

        var average_len = Float64(self._min_len + self._max_len) / 2.0

        # geometric distribution
        var probability = 1.0 - 1.0 / (1.0 + average_len)
        while len(result) < self._max_len:
            var should_append = rng.rand_bool(true_probability=probability)
            if not should_append:
                break

            result.append(self._strat.value(rng))

        return result^
