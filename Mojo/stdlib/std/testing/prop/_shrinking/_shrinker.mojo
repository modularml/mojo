# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licenxse is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from std.testing.prop.random import Rng
from std.testing.prop.strategy import Strategy
from std.testing.prop._errors import PLAYBACK_EXHAUSTED
from std.collections import Set
from std.os import abort

from ._passes import DeleteChunkPass, ReduceChunkPass, ZeroChunkPass

comptime Stream = List[UInt64]
comptime StreamChunk[width: Int] = SIMD[DType.uint64, width]


def stream_is_better(stream_a: Stream, stream_b: Stream) -> Bool:
    if len(stream_a) != len(stream_b):
        return len(stream_a) < len(stream_b)

    for a, b in zip(stream_a, stream_b):
        if a != b:
            return a < b

    # probably shouldn't happen that the streams are the same
    return False


@fieldwise_init
struct ShrinkResult[T: Copyable & Deinitable & Writable](Copyable):
    var value: Self.T
    var runs: Int


trait ShrinkOracle(Deinitable, Movable):
    def try_stream(mut self, stream: Stream) -> Bool:
        ...

    def best_stream[o: ImmOrigin, //](ref[o] self) -> ref[o._subtree] Stream:
        ...


struct Shrinker[
    StrategyType: Strategy, f: def(var StrategyType.Value) thin raises
](ShrinkOracle):

    """Property test input reducer.
    When a test fails, try and find a minimal reproduction of the failure.
    """

    var best: Stream
    var expected_error: Error
    var max_reductions: Int
    var cache: Set[UInt64]
    var strategy: Self.StrategyType

    @doc_hidden
    def __init__(
        out self,
        var strategy: Self.StrategyType,
        var stream: Stream,
        var error: Error,
        max_reductions: Int = 1000,
    ):
        self.best = stream^
        self.expected_error = error^
        self.max_reductions = max_reductions
        self.strategy = strategy^

        self.cache = {}

    def shrink(mut self) -> ShrinkResult[Self.StrategyType.Value]:
        """Find the minimal interesting input.

        Returns:
            The minimal interesting input found during shrinking.
        """

        var reductions = 0
        var runs = 0
        var delete_pass = DeleteChunkPass()
        var zero_pass = ZeroChunkPass()
        var reduce_pass = ReduceChunkPass()
        while reductions < self.max_reductions:
            runs += 1

            if not (
                delete_pass.run(self)
                or zero_pass.run(self)
                or reduce_pass.run(self)
            ):
                break

            reductions += 1

        var rng = Rng(self.best.copy())

        try:
            return ShrinkResult(self.strategy.value(rng), runs)
        except:
            # We know best works so this shouldn't happen
            abort("Failed to build value from best stream")

    def try_stream(mut self, stream: Stream) -> Bool:
        var fingerprint = hash(stream)
        if fingerprint in self.cache:
            return False

        self.cache.add(fingerprint)
        var rng = Rng(stream.copy())
        try:
            var val = self.strategy.value(rng)
            Self.f(val^)
            return False
        except e:
            var used = Stream(stream[: rng.playback_index])
            if self._error_is_interesting(e, used):
                self.best = used^
                return True
            return False

    def best_stream[o: ImmOrigin, //](ref[o] self) -> ref[o._subtree] Stream:
        return self.best

    def _error_is_interesting(self, e: Error, used: Stream) -> Bool:
        # TODO: Need a robust way of determining if two errors
        # originate from the same failure condition.
        return String(e) == String(self.expected_error) and stream_is_better(
            used, self.best
        )
