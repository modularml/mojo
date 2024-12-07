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
# RUN: %mojo-no-debug %s -t
# NOTE: to test changes on the current branch using run-benchmarks.sh, remove
# the -t flag. Remember to replace it again before pushing any code.

from sys import simdwidthof

from benchmark import Bench, BenchConfig, Bencher, BenchId, Unit, keep, run
from bit import count_trailing_zeros
from builtin.dtype import _uint_type_of_width

from utils.stringref import _align_down, _memchr, _memmem

# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Benchmarks
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_writer_int[n: Int](mut b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn():
        var s1 = String()
        s1.write(n)
        _ = s1^

    b.iter[call_fn]()


@parameter
fn bench_writer_simd[n: Int](mut b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn():
        var s1 = String()
        s1.write(SIMD[DType.int32, simdwidthof[DType.int32]()](n))
        _ = s1^

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main():
    var m = Bench(BenchConfig(num_repetitions=1))
    m.bench_function[bench_writer_int[42]](BenchId("bench_writer_int_42"))
    m.bench_function[bench_writer_int[2**64]](
        BenchId("bench_writer_int_2**64")
    )
    m.bench_function[bench_writer_simd[42]](BenchId("bench_writer_simd"))
    m.bench_function[bench_writer_simd[2**16]](
        BenchId("bench_writer_simd_2**16")
    )
    m.dump_report()
