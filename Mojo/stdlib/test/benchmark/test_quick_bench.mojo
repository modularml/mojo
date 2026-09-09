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

import std.math
from std.memory import alloc, dealloc, Layout, ThinAllocation
from std.random import randint
from std.time import sleep

from std.benchmark import BenchId, BenchMetric, QuickBench, ThroughputMeasure
from std.testing import TestSuite


def vec_reduce[
    N: Int, dtype: DType
](x: Pointer[Scalar[dtype], ImmutAnyOrigin]) -> Scalar[dtype]:
    var total: Scalar[dtype] = 0
    for i in range(N):
        total += x[unsafe_offset=i]
    return total


def vec_add[
    N: Int, dtype: DType
](
    x: Pointer[Scalar[dtype], MutAnyOrigin],
    y: Pointer[Scalar[dtype], ImmutAnyOrigin],
) -> Pointer[Scalar[dtype], MutAnyOrigin]:
    for i in range(N):
        x[unsafe_offset=i] += y[unsafe_offset=i]
    return x


def dummy() -> None:
    sleep(0.5)


def dummy(x0: Int) -> Float32:
    return Float32(x0)


def dummy(x0: Int, x1: Int) -> Float32:
    return Float32(x0 + x1)


def dummy(x0: Int, x1: Int, x2: Int) -> Float32:
    return Float32(x0 + x1 + x2)


def dummy(x0: Int, x1: Int, x2: Int, x3: Int) -> Float32:
    return Float32(x0 + x1 + x2 + x3)


def dummy(x0: Int, x1: Int, x2: Int, x3: Int, x4: Int) -> Float32:
    return Float32(x0 + x1 + x2 + x3 + x4)


def dummy(x0: Int, x1: Int, x2: Int, x3: Int, x4: Int, x5: Int) -> Float32:
    return Float32(x0 + x1 + x2 + x3 + x4 + x5)


def dummy(
    x0: Int, x1: Int, x2: Int, x3: Int, x4: Int, x5: Int, x6: Int
) -> Float32:
    return Float32(x0 + x1 + x2 + x3 + x4 + x5 + x6)


def dummy(
    x0: Int, x1: Int, x2: Int, x3: Int, x4: Int, x5: Int, x6: Int, x7: Int
) -> Float32:
    return Float32(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7)


def dummy(
    x0: Int,
    x1: Int,
    x2: Int,
    x3: Int,
    x4: Int,
    x5: Int,
    x6: Int,
    x7: Int,
    x8: Int,
) -> Float32:
    return Float32(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8)


def dummy(
    x0: Int,
    x1: Int,
    x2: Int,
    x3: Int,
    x4: Int,
    x5: Int,
    x6: Int,
    x7: Int,
    x8: Int,
    x9: Int,
) -> Float32:
    return Float32(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9)


def test_overloaded() raises:
    var qb = QuickBench()

    qb.run[T_out=NoneType._mlir_type](
        dummy,
        bench_id=BenchId("dummy_none"),
        measures=[
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ],
    )
    qb.run[Int, T_out=Float32](
        dummy,
        1,
        bench_id=BenchId("dummy_1"),
        measures=[
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ],
    )

    qb.run[Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        bench_id=BenchId("dummy_2"),
        measures=[
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ],
    )

    qb.run[Int, Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        3,
        bench_id=BenchId("dummy_3"),
        measures=[
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ],
    )

    qb.run[Int, Int, Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        3,
        4,
        bench_id=BenchId("dummy_4"),
        measures=[
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ],
    )

    qb.run[Int, Int, Int, Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        3,
        4,
        5,
        bench_id=BenchId("dummy_5"),
        measures=[
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ],
    )

    qb.run[Int, Int, Int, Int, Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        3,
        4,
        5,
        6,
        bench_id=BenchId("dummy_6"),
        measures=[
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ],
    )

    qb.run[Int, Int, Int, Int, Int, Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        bench_id=BenchId("dummy_7"),
        measures=[
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ],
    )

    qb.run[Int, Int, Int, Int, Int, Int, Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        bench_id=BenchId("dummy_8"),
        measures=[
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ],
    )

    qb.run[Int, Int, Int, Int, Int, Int, Int, Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        bench_id=BenchId("dummy_9"),
        measures=[
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ],
    )

    qb.run[Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, T_out=Float32](
        dummy,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        bench_id=BenchId("dummy_10"),
        measures=[
            ThroughputMeasure(BenchMetric.flops, 1)  # N additions per call
        ],
    )

    qb.dump_report()


@always_inline
def exp(x: SIMD[.float32, 4]) -> type_of(x):
    return std.math.exp(x)


@always_inline
def tanh(x: SIMD[.float32, 4]) -> type_of(x):
    return std.math.tanh(x)


def test_mojo_math() raises:
    var qb = QuickBench()

    qb.run(
        exp,
        1.0,
        bench_id=BenchId("exp"),
        measures=[ThroughputMeasure(BenchMetric.bytes, 4)],  # 4 bytes per call
    )

    qb.run(
        tanh,
        1.0,
        bench_id=BenchId("tanh"),
        measures=[ThroughputMeasure(BenchMetric.bytes, 4)],  # 4 bytes per call
    )
    qb.dump_report()


def test_custom() raises:
    comptime N = 1024
    comptime alignment = 64
    comptime dtype = DType.int32
    var xy_layout = Layout[Scalar[dtype]](count=N, alignment=alignment)
    var x = alloc(xy_layout).unsafe_leak()
    var y = alloc(xy_layout).unsafe_leak()
    randint[dtype](x, N, 0, 255)
    randint[dtype](y, N, 0, 255)

    var qb = QuickBench()

    qb.run(
        vec_reduce[N, dtype],
        x.as_unsafe_any_origin(),
        bench_id=BenchId("vec_reduce"),
        measures=[
            ThroughputMeasure(BenchMetric.flops, N)  # N additions per call
        ],
    )

    qb.run(
        vec_add[N, dtype],
        x.as_unsafe_any_origin(),
        y.as_unsafe_any_origin(),
        bench_id=BenchId("vec_add"),
        measures=[
            ThroughputMeasure(BenchMetric.flops, N)  # N additions per call
        ],
    )

    qb.dump_report()
    dealloc(ThinAllocation(unsafe_owned_ptr=x).unsafe_with_layout(xy_layout))
    dealloc(ThinAllocation(unsafe_owned_ptr=y).unsafe_with_layout(xy_layout))


def test_all() raises:
    # Width of columns is dynamic based on the longest value as a string, so
    # only test the first column.

    # CHECK: name,
    # CHECK: exp ,
    # CHECK: tanh,
    test_mojo_math()

    # CHECK: name      ,
    # CHECK: vec_reduce,
    # CHECK: vec_add   ,
    test_custom()

    # CHECK: name      ,
    # CHECK: dummy_none,
    # CHECK: dummy_1   ,
    # CHECK: dummy_2   ,
    # CHECK: dummy_3   ,
    # CHECK: dummy_4   ,
    # CHECK: dummy_5   ,
    # CHECK: dummy_6   ,
    # CHECK: dummy_7   ,
    # CHECK: dummy_8   ,
    # CHECK: dummy_9   ,
    # CHECK: dummy_10  ,
    test_overloaded()


def main() raises:
    # NOTE: we pass an empty list since the benchmark infra also tries to parse
    # the arguments for its own purposes.
    TestSuite.discover_tests[__functions_in_module()](
        cli_args=List[StaticString]()
    ).run()
