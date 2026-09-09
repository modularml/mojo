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

from std.math import isclose
from std.random import rand
from std.sys import simd_width_of, size_of

import std.benchmark
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.gemv import gemv, naive_gemv
from linalg.matmul import matmul
from std.testing import assert_false

comptime alignment = 64


def bench_run(
    func: Some[ImplicitlyCopyable & (def() raises)],
) raises -> std.benchmark.Report:
    return std.benchmark.run(func, 2, 1_000_000, 1, 3)


def test_gemv() raises:
    print("== test_gemv")
    comptime type = DType.float32
    comptime absolute_tolerance = 1e-08
    comptime relative_tolerance = 1e-05

    # alias type = DType.float16
    # alias absolute_tolerance = 5e-02
    # alias relative_tolerance = 5e-01

    comptime simd_width = simd_width_of[type]()
    # alias m = 22016
    # alias k = 4096

    comptime m = 4096
    comptime k = 11008

    var lhs_storage = alloc[Scalar[type],](m * k, alignment=alignment)
    var lhs = TileTensor(lhs_storage.as_unsafe_any_origin(), row_major[m, k]())

    var rhs_storage = alloc[Scalar[type],](k, alignment=alignment)
    var rhs = TileTensor(rhs_storage.as_unsafe_any_origin(), row_major[k]())

    var out_storage = alloc[Scalar[type],](m, alignment=alignment)
    var out = TileTensor(out_storage.as_unsafe_any_origin(), row_major[m]())

    var ref_out_storage = alloc[Scalar[type]](m, alignment=alignment)
    var ref_out = TileTensor(
        ref_out_storage.as_unsafe_any_origin(), row_major[m]()
    )

    rand[type](lhs_storage, m * k)
    rand[type](rhs_storage, k)

    # Compute reference output
    naive_gemv(ref_out, lhs, rhs)

    # Validate results from serial and parallel implementations

    _ = out.fill(0)
    gemv[parallelize=False](out, lhs, rhs)

    # Verify the result
    for i in range(out.num_elements()):
        var expect = ref_out[i]
        var actual = out[i]
        if not isclose(
            expect, actual, atol=absolute_tolerance, rtol=relative_tolerance
        ):
            print(out[i], "!=", ref_out[i], "@", i)
            print("Serial Error")
            assert_false(True)

    comptime threads = 0

    _ = out.fill(0)
    gemv[parallelize=True](out, lhs, rhs)

    # Verify the result
    for i in range(out.num_elements()):
        var expect = ref_out[i]
        var actual = out[i]
        if not isclose(
            expect, actual, atol=absolute_tolerance, rtol=relative_tolerance
        ):
            print(out[i], "!=", ref_out[i], "@", i)
            print("Parallel Error")
            assert_false(True)

    comptime bytes_per_iteration = 2 * m * k * size_of[type]()
    comptime gigabyte = 1024 * 1024 * 1024

    # Serial Gemv
    @always_inline
    def bench_fn_serial() raises {var}:
        gemv[parallelize=False](out, lhs, rhs)

    var serial_perf = bench_run(bench_fn_serial)
    std.benchmark.keep(out[10])
    var serial_bandwidth = (
        Float64(bytes_per_iteration) / serial_perf.mean()
    ) / gigabyte
    print(
        "Serial GEMV Bandwidth: ",
        serial_bandwidth,
        "(",
        serial_perf.mean(),
        ")",
    )
    print("Serial GEMV GFLOP/s", 1e-9 * ((2 * m * k) / serial_perf.mean()))

    # Parallel Gemv
    @always_inline
    def bench_fn_parallel() raises {var}:
        gemv[parallelize=True](out, lhs, rhs)

    var par_perf = bench_run(bench_fn_parallel)
    std.benchmark.keep(out[10])

    var rhs_mat = TileTensor(
        rhs_storage.as_unsafe_any_origin(), row_major[k, 1]()
    )
    var out_mat = TileTensor(
        out_storage.as_unsafe_any_origin(), row_major[m, 1]()
    )

    # Compute speedup and bandwidth stats
    var par_bandwidth = (
        Float64(bytes_per_iteration) / par_perf.mean()
    ) / gigabyte
    print(
        "Parallel GEMV Bandwidth: ",
        par_bandwidth,
        "(",
        par_perf.mean(),
        ")",
    )
    print("Parallel GEMV GFLOP/s", 1e-9 * ((2 * m * k) / par_perf.mean()))

    var bandwidth_increase = par_bandwidth / serial_bandwidth
    print("--> Bandwidth increase: ", bandwidth_increase)

    var speedup = serial_perf.mean() / par_perf.mean()
    print("--> Mean Runtime Speedup: ", speedup)

    @always_inline
    def bench_fn_matmul() raises {var}:
        matmul(out_mat, lhs, rhs_mat)

    bench_fn_matmul()

    var matmul_perf = bench_run(bench_fn_matmul)
    std.benchmark.keep(out[10])
    matmul_perf.print()
    print("Matmul GEMV GFLOP/s", 1e-9 * ((2 * m * k) / matmul_perf.mean()))

    lhs_storage.free()
    rhs_storage.free()
    out_storage.free()
    ref_out_storage.free()


def main() raises:
    test_gemv()
