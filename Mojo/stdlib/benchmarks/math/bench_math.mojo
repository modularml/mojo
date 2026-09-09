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

from std.math import *
from std.random import *

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep

# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
comptime input_type = Float32


def make_inputs(
    begin: input_type, end: input_type, num: input_type
) -> List[input_type]:
    if num == 1:
        return [begin]

    var step = (end - begin) / (num - 1)

    var result = List[input_type]()
    for i in range(input_type(0), num, input_type(1)):
        result.append(begin + step * i)
    return result^


def make_int_inputs(begin: Int, end: Int, num: Int) -> List[Int]:
    if num == 1:
        return [begin]

    var step = (end - begin) // (num - 1)

    var result = List[Int]()
    for i in range(num):
        result.append(begin + step * i)
    return result^


# ===-----------------------------------------------------------------------===#
# Benchmark math_func
# ===-----------------------------------------------------------------------===#


def bench_math[
    math_f1p: def[dtype: DType, size: SIMDLength](
        SIMD[dtype, size]
    ) thin -> SIMD[dtype, size] where dtype.is_floating_point()
](mut b: Bencher) raises:
    var inputs = make_inputs(0, 10_000, 1_000_000)

    @always_inline
    def call_fn() raises {imm inputs}:
        for input in inputs:
            var result = math_f1p(input)
            keep(result)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark fma
# ===-----------------------------------------------------------------------===#
def bench_math3[
    math_f3p: def[dtype: DType, size: SIMDLength](
        SIMD[dtype, size], SIMD[dtype, size], SIMD[dtype, size]
    ) thin -> SIMD[dtype, size]
](mut b: Bencher) raises:
    var inputs = make_inputs(0, 10_000, 1_000_000)

    @always_inline
    def call_fn() raises {imm inputs}:
        for input in inputs:
            var result = math_f3p(input, input, input)
            keep(result)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark lcm/gcd
# ===-----------------------------------------------------------------------===#
def bench_math2[math_f2p: def(Int, Int, /) thin -> Int](mut b: Bencher) raises:
    var int_inputs = make_int_inputs(0, 10_000_000, 1_000_000)

    @always_inline
    def call_fn() raises {imm int_inputs}:
        for i, input_val in enumerate(List(int_inputs[: len(int_inputs) // 2])):
            var result = keep(
                math_f2p(input_val, int_inputs[len(int_inputs) - (i + 1)])
            )
            keep(result)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main() raises:
    seed()
    var m = Bench(BenchConfig(num_repetitions=1))
    m.bench_function(bench_math[sin], BenchId("bench_math_sin"))
    m.bench_function(bench_math[cos], BenchId("bench_math_cos"))
    m.bench_function(bench_math[tan], BenchId("bench_math_tan"))
    m.bench_function(bench_math[asin], BenchId("bench_math_asin"))
    m.bench_function(bench_math[acos], BenchId("bench_math_acos"))
    m.bench_function(bench_math[atan], BenchId("bench_math_atan"))
    m.bench_function(bench_math[log], BenchId("bench_math_log"))
    m.bench_function(bench_math[log2], BenchId("bench_math_log2"))
    m.bench_function(bench_math[sqrt], BenchId("bench_math_sqrt"))
    m.bench_function(bench_math[exp2], BenchId("bench_math_exp2"))
    m.bench_function(bench_math[exp], BenchId("bench_math_exp"))
    m.bench_function(bench_math[erf], BenchId("bench_math_erf"))
    m.bench_function(bench_math3[fma], BenchId("bench_math_fma"))
    m.bench_function(bench_math2[lcm], BenchId("bench_math_lcm"))
    m.bench_function(bench_math2[gcd], BenchId("bench_math_gcd"))
    m.dump_report()
