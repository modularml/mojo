# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s -t

from benchmark import Bench, Bencher, BenchId, keep, BenchConfig, Unit, run
from random import *
from math import *

# ===----------------------------------------------------------------------===#
# Benchmark Data
# ===----------------------------------------------------------------------===#
alias input_type = Float32


fn make_inputs(
    begin: input_type, end: input_type, num: input_type
) -> List[input_type]:
    if num == 1:
        return List[input_type](begin)

    var step = (end - begin) / (num - 1)

    var result: List[input_type] = List[input_type]()
    for i in range(num):
        result.append(begin + step * i)
    return result


var inputs = make_inputs(0, 10_000, 1_000_000)

# ===----------------------------------------------------------------------===#
# Benchmark math_func
# ===----------------------------------------------------------------------===#


@parameter
fn bench_math[
    math_f1p: fn[type: DType, size: Int] (SIMD[type, size]) -> SIMD[type, size]
](inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for input in inputs:
            _ = math_f1p(input[])

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark fma
# ===----------------------------------------------------------------------===#
@parameter
fn bench_math3[
    math_f3p: fn[type: DType, size: Int] (
        SIMD[type, size], SIMD[type, size], SIMD[type, size]
    ) -> SIMD[type, size]
](inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for input in inputs:
            _ = math_f3p(input[], input[], input[])

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark Main
# ===----------------------------------------------------------------------===#
def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=1, warmup_iters=100000))
    m.bench_function[bench_math[sin]](BenchId("bench_math_sin"))
    m.bench_function[bench_math[cos]](BenchId("bench_math_cos"))
    m.bench_function[bench_math[tan]](BenchId("bench_math_tan"))
    m.bench_function[bench_math[asin]](BenchId("bench_math_asin"))
    m.bench_function[bench_math[acos]](BenchId("bench_math_acos"))
    m.bench_function[bench_math[atan]](BenchId("bench_math_atan"))
    m.bench_function[bench_math[log]](BenchId("bench_math_log"))
    m.bench_function[bench_math[log2]](BenchId("bench_math_log2"))
    m.bench_function[bench_math[sqrt]](BenchId("bench_math_sqrt"))
    m.bench_function[bench_math[exp2]](BenchId("bench_math_exp2"))
    m.bench_function[bench_math[exp]](BenchId("bench_math_exp"))
    m.bench_function[bench_math[erf]](BenchId("bench_math_erf"))
    m.bench_function[bench_math3[fma]](BenchId("bench_math_fma"))
    m.dump_report()
