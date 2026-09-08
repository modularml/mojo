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

from std.random import random_ui64, seed
from std.sys import bit_width_of
from std.sys.intrinsics import likely, unlikely

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep
from std.bit import bit_width, count_leading_zeros
from std.bit.mask import splat

# ===-----------------------------------------------------------------------===#
# Benchmarks
# ===-----------------------------------------------------------------------===#

# ===-----------------------------------------------------------------------===#
# next_power_of_two
# ===-----------------------------------------------------------------------===#


def next_power_of_two_int_v1(val: Int) -> Int:
    if val <= 1:
        return 1

    if val.is_power_of_two():
        return val

    return 1 << bit_width(val - 1)


def next_power_of_two_int_v2(val: Int) -> Int:
    if val <= 1:
        return 1

    return 1 << (bit_width_of[Int]() - count_leading_zeros(val - 1))


def next_power_of_two_int_v3(val: Int) -> Int:
    var v = Int(val)
    return Int(
        mlir_value=v.gt(1)
        .select(
            1 << (Int(bit_width_of[Int]()) - count_leading_zeros(v - 1)),
            1,
        )
        .__mlir_index__()
    )


def next_power_of_two_int_v4(val: Int) -> Int:
    return 1 << (
        (bit_width_of[Int]() - count_leading_zeros(val - 1))
        & splat(likely(val > 1))
    )


def next_power_of_two_uint_v1(
    val: UInt,
) -> UInt:
    if unlikely(val == 0):
        return 1

    return UInt(1 << (bit_width_of[UInt]() - count_leading_zeros(Int(val - 1))))


def next_power_of_two_uint_v2(
    val: UInt,
) -> UInt:
    return UInt(
        val.eq(0).select(
            1 << (UInt(bit_width_of[UInt]()) - count_leading_zeros(val - 1)),
            1,
        )
    )


def next_power_of_two_uint_v3(
    val: UInt,
) -> UInt:
    return UInt(
        1
        << (
            bit_width_of[UInt]()
            - count_leading_zeros(Int(val - UInt(likely(val > 0))))
        )
    )


def next_power_of_two_uint_v4(
    val: UInt,
) -> UInt:
    return UInt(
        1
        << (
            bit_width_of[UInt]()
            - count_leading_zeros(Int((val | UInt(unlikely(val == 0))) - 1))
        )
    )


def _build_list[start: Int, stop: Int]() -> List[Int]:
    var values = List[Int](capacity=10_000)
    for _ in range(10_000):
        values.append(Int(random_ui64(UInt64(start), UInt64(stop))))
    return values^


comptime width = bit_width_of[Int]()


def bench_next_power_of_two_int[
    func: def(Int) thin -> Int
](mut b: Bencher) raises:
    var _values = _build_list[0, 2**width - 1]()

    @always_inline
    def call_fn() raises {imm}:
        for _ in range(10_000):
            for i in range(len(_values)):
                var result = func(_values.unsafe_get(i))
                keep(result)

    b.iter(call_fn)


def bench_next_power_of_two_uint[
    func: def(UInt) thin -> UInt
](mut b: Bencher) raises:
    var _values = _build_list[0, 2**width - 1]()

    @always_inline
    def call_fn() raises {imm}:
        for _ in range(10_000):
            for i in range(len(_values)):
                var result = func(UInt(_values.unsafe_get(i)))
                keep(result)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main() raises:
    seed()
    var m = Bench(BenchConfig(num_repetitions=10))
    m.bench_function(
        bench_next_power_of_two_int[next_power_of_two_int_v1],
        BenchId("bench_next_power_of_two_int_v1"),
    )
    m.bench_function(
        bench_next_power_of_two_int[next_power_of_two_int_v2],
        BenchId("bench_next_power_of_two_int_v2"),
    )
    m.bench_function(
        bench_next_power_of_two_int[next_power_of_two_int_v3],
        BenchId("bench_next_power_of_two_int_v3"),
    )
    m.bench_function(
        bench_next_power_of_two_int[next_power_of_two_int_v4],
        BenchId("bench_next_power_of_two_int_v4"),
    )
    m.bench_function(
        bench_next_power_of_two_uint[next_power_of_two_uint_v1],
        BenchId("bench_next_power_of_two_uint_v1"),
    )
    m.bench_function(
        bench_next_power_of_two_uint[next_power_of_two_uint_v2],
        BenchId("bench_next_power_of_two_uint_v2"),
    )
    m.bench_function(
        bench_next_power_of_two_uint[next_power_of_two_uint_v3],
        BenchId("bench_next_power_of_two_uint_v3"),
    )
    m.bench_function(
        bench_next_power_of_two_uint[next_power_of_two_uint_v4],
        BenchId("bench_next_power_of_two_uint_v4"),
    )

    var results = Dict[String, Tuple[Float64, Int]]()
    for info in m.info_vec:
        var n = info.name
        var time = info.result.mean("ms")
        var avg, amnt = results.get(n, (Float64(0), 0))
        results[n] = (
            (avg * Float64(amnt) + time) / Float64((amnt + 1)),
            amnt + 1,
        )
    print("")
    for k_v in results.items():
        print(k_v.key, k_v.value[0], sep=",")
