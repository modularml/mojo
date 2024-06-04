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
from utils.stringref import _memmem, _memchr, _align_down

from bit import countr_zero
from builtin.dtype import _uint_type_of_width

# ===----------------------------------------------------------------------===#
# Benchmark Data
# ===----------------------------------------------------------------------===#


# ===----------------------------------------------------------------------===#
# Benchmarks
# ===----------------------------------------------------------------------===#
@parameter
fn bench_formatter_int[n: Int](inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn():
        var s1 = String()
        var s1_fmt = Formatter(s1)
        Int(n).format_to(s1_fmt)

    b.iter[call_fn]()


@parameter
fn bench_formatter_simd[n: Int](inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn():
        var s1 = String()
        var s1_fmt = Formatter(s1)
        SIMD[DType.int32](n).format_to(s1_fmt)

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark Main
# ===----------------------------------------------------------------------===#
def main():
    var m = Bench(BenchConfig(num_repetitions=1, warmup_iters=10000))
    m.bench_function[bench_formatter_int[42]](BenchId("bench_formatter_int_42"))
    m.bench_function[bench_formatter_int[2**64]](
        BenchId("bench_formatter_int_2**64")
    )
    m.bench_function[bench_formatter_simd[42]](BenchId("bench_formatter_simd"))
    m.bench_function[bench_formatter_simd[2**16]](
        BenchId("bench_formatter_simd_2**16")
    )
    m.dump_report()
