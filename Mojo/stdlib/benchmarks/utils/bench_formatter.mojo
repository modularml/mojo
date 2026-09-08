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

from std.sys import simd_width_of

from std.benchmark import Bench, BenchConfig, Bencher, BenchId

# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Benchmarks
# ===-----------------------------------------------------------------------===#
def bench_writer_int[n: Int](mut b: Bencher) raises:
    @always_inline
    def call_fn():
        var s1 = String()
        s1.write(n)
        _ = s1^

    b.iter(call_fn)


def bench_writer_simd[n: Int](mut b: Bencher) raises:
    @always_inline
    def call_fn():
        var s1 = String()
        s1.write(SIMD[.int32, simd_width_of[DType.int32]()](n))
        _ = s1^

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main() raises:
    var m = Bench(BenchConfig(num_repetitions=1))
    m.bench_function(bench_writer_int[42], BenchId("bench_writer_int_42"))
    m.bench_function(
        bench_writer_int[2**64], BenchId("bench_writer_int_2**64")
    )
    m.bench_function(bench_writer_simd[42], BenchId("bench_writer_simd"))
    m.bench_function(
        bench_writer_simd[2**16], BenchId("bench_writer_simd_2**16")
    )
    m.dump_report()
