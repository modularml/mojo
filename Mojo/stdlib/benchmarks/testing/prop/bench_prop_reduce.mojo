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

from std.benchmark import Bench, Bencher, BenchId, keep
from std.testing.prop import Rng, Strategy
from std.testing.prop._shrinking._shrinker import Shrinker, Stream

comptime max_shrink_steps = 1000

# ===-----------------------------------------------------------------------===#
# Benchmark harness
# ===-----------------------------------------------------------------------===#


def _failing_error[
    StrategyType: Strategy,
    //,
    make_strategy: def() thin raises -> StrategyType,
    prop: def(var StrategyType.Value) thin raises,
](name: String, stream: Stream) raises -> Error:
    """Returns the error `prop` raises for the value `stream` decodes to."""
    var strategy = make_strategy()
    var rng = Rng(stream.copy())
    var value = strategy.value(rng)
    try:
        prop(value^)
    except e:
        return e
    raise Error("benchmark stream for '", name, "' must fail the property")


def bench_prop_reduce[
    StrategyType: Strategy,
    //,
    make_strategy: def() thin raises -> StrategyType,
    prop: def(var StrategyType.Value) thin raises,
](mut m: Bench, name: String, stream: Stream) raises:
    """Registers a benchmark that shrinks `stream` against `prop`."""
    var expected_error = _failing_error[make_strategy, prop](name, stream)

    var probe = Shrinker[StrategyType, prop](
        make_strategy(), stream.copy(), expected_error, max_shrink_steps
    )

    var steps = probe.shrink().runs

    def do_bench(mut b: Bencher) raises {imm stream, imm expected_error}:
        @always_inline
        def do_shrink() raises {imm}:
            var shrinker = Shrinker[StrategyType, prop](
                make_strategy(),
                stream.copy(),
                expected_error,
                max_shrink_steps,
            )
            var result = shrinker.shrink()
            keep(result)

        b.iter(do_shrink)

    m.bench_function(do_bench, BenchId(name, String("steps=", steps)))


# ===-----------------------------------------------------------------------===#
# Strategies, properties and streams
# ===-----------------------------------------------------------------------===#

comptime UInt64x16Strategy = type_of(SIMD[DType.uint64, 16].strategy())
comptime UInt64ListStrategy = type_of(List[UInt64].strategy(UInt64.strategy()))
comptime AsciiStrategy = type_of(String.ascii_strategy(only_printable=True))


def uint64x16_strategy() raises -> UInt64x16Strategy:
    return SIMD[DType.uint64, 16].strategy()


def uint64_list_strategy() raises -> UInt64ListStrategy:
    return List[UInt64].strategy(UInt64.strategy())


def ascii_strategy() raises -> AsciiStrategy:
    return String.ascii_strategy(only_printable=True)


def fails_when_all_lanes_large(var value: SIMD[DType.uint64, 16]) raises:
    var threshold = SIMD[DType.uint64, 16](1000)
    if value.gt(threshold).reduce_and():
        raise Error("all lanes above 1000")


def fails_when_long_without_small(var values: List[UInt64]) raises:
    if len(values) < 12:
        return
    for value in values:
        if value <= 1000:
            return
    raise Error("at least 12 elements above 1000")


def fails_when_long_and_increasing(var values: List[UInt64]) raises:
    if len(values) < 12:
        return
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            return
    raise Error("at least 12 strictly increasing elements")


def fails_when_long_without_spaces(var s: String) raises:
    if s.byte_length() >= 16 and " " not in s:
        raise Error("at least 16 characters without a space")


def repeated(value: UInt64, count: Int) -> List[UInt64]:
    var values = List[UInt64](capacity=count)
    for _ in range(count):
        values.append(value)
    return values^


def ramp(step: UInt64, count: Int) -> List[UInt64]:
    var values = List[UInt64](capacity=count)
    for i in range(count):
        values.append(UInt64(i + 1) * step)
    return values^


def list_stream(elements: List[UInt64]) -> Stream:
    """Encodes `elements` the way the list strategy consumes a stream: one
    continue word before each element and a stop word at the end."""
    var stream = Stream(capacity=2 * len(elements) + 1)
    for element in elements:
        stream.append(UInt64.MAX)
        stream.append(element)
    stream.append(0)
    return stream^


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#


def main() raises:
    var m = Bench()

    # Sixteen words that each have to be binary searched down on their own;
    # deleting any of them starves the strategy and zeroing any lane passes.
    bench_prop_reduce[uint64x16_strategy, fails_when_all_lanes_large](
        m, "shrink_simd_lanes", repeated(1 << 40, 16)
    )

    # Half the list can go in one deletion, but every survivor still has to
    # be reduced separately because a zeroed element passes the property.
    bench_prop_reduce[uint64_list_strategy, fails_when_long_without_small](
        m, "shrink_list_min_length", list_stream(repeated(1 << 40, 24))
    )

    # Each element's floor depends on its predecessor, so the reductions have
    # to happen in order and the binary searches spend most tries failing.
    bench_prop_reduce[uint64_list_strategy, fails_when_long_and_increasing](
        m, "shrink_list_increasing", list_stream(ramp(1 << 36, 24))
    )

    # The same shape through the string strategy.
    bench_prop_reduce[ascii_strategy, fails_when_long_without_spaces](
        m,
        "shrink_string_no_spaces",
        list_stream(repeated(UInt64(ord("z") - ord(" ")), 64)),
    )

    m.dump_report()
