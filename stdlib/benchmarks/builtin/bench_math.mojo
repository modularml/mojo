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
# RUN: %mojo %s

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
# Benchmark sin
# ===----------------------------------------------------------------------===#
@parameter
fn bench_math_sin(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for input in inputs:
            _ = sin(input[])

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark cos
# ===----------------------------------------------------------------------===#
@parameter
fn bench_math_cos(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for input in inputs:
            _ = cos(input[])

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark tan
# ===----------------------------------------------------------------------===#
@parameter
fn bench_math_tan(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for input in inputs:
            _ = tan(input[])

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark asin
# ===----------------------------------------------------------------------===#
@parameter
fn bench_math_asin(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for input in inputs:
            _ = asin(input[])

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark acos
# ===----------------------------------------------------------------------===#
@parameter
fn bench_math_acos(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for input in inputs:
            _ = acos(input[])

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark atan
# ===----------------------------------------------------------------------===#
@parameter
fn bench_math_atan(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for input in inputs:
            _ = atan(input[])

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark log
# ===----------------------------------------------------------------------===#
@parameter
fn bench_math_log(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for input in inputs:
            _ = log(input[])

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark log2
# ===----------------------------------------------------------------------===#
@parameter
fn bench_math_log2(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for input in inputs:
            _ = log2(input[])

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark sqrt
# ===----------------------------------------------------------------------===#
@parameter
fn bench_math_sqrt(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for input in inputs:
            _ = sqrt(input[])

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark exp2
# ===----------------------------------------------------------------------===#
@parameter
fn bench_math_exp2(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for input in inputs:
            _ = exp2(input[])

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark exp
# ===----------------------------------------------------------------------===#
@parameter
fn bench_math_exp(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for input in inputs:
            _ = exp(input[])

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark erf
# ===----------------------------------------------------------------------===#
@parameter
fn bench_math_erf(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for input in inputs:
            _ = erf(input[])

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark fma
# ===----------------------------------------------------------------------===#
@parameter
fn bench_math_fma(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for input in inputs:
            _ = fma(input[], input[], input[])

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark Main
# ===----------------------------------------------------------------------===#
def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=1, warmup_iters=100000))
    m.bench_function[bench_math_sin](BenchId("bench_math_sin"))
    m.bench_function[bench_math_cos](BenchId("bench_math_cos"))
    m.bench_function[bench_math_tan](BenchId("bench_math_tan"))
    m.bench_function[bench_math_asin](BenchId("bench_math_asin"))
    m.bench_function[bench_math_acos](BenchId("bench_math_acos"))
    m.bench_function[bench_math_atan](BenchId("bench_math_atan"))
    m.bench_function[bench_math_log](BenchId("bench_math_log"))
    m.bench_function[bench_math_log2](BenchId("bench_math_log2"))
    m.bench_function[bench_math_sqrt](BenchId("bench_math_sqrt"))
    m.bench_function[bench_math_exp2](BenchId("bench_math_exp2"))
    m.bench_function[bench_math_exp](BenchId("bench_math_exp"))
    m.bench_function[bench_math_erf](BenchId("bench_math_erf"))
    m.bench_function[bench_math_fma](BenchId("bench_math_fma"))
    m.dump_report()
