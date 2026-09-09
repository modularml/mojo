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

from std.random import random_float64
from std.sys import get_defined_dtype, get_defined_int
from std.sys.info import size_of

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from nn.argsort import argsort


def bench_argsort[
    dtype: DType
](ctx: DeviceContext, mut m: Bench, N: Int) raises:
    # Allocate and fill host data with random values.
    var input_host_ptr = List(length=N, fill=Scalar[dtype](0))
    for i in range(N):
        input_host_ptr[i] = Scalar[dtype](
            random_float64(-1e6, 1e6).cast[dtype]()
        )

    # Allocate device buffers.
    var device_input = ctx.enqueue_create_buffer[dtype](N)
    var device_indices = ctx.enqueue_create_buffer[.int64](N)
    ctx.enqueue_copy(device_input, input_host_ptr)

    var device_input_tensor = TileTensor(
        device_input,
        row_major(N),
    )
    var device_indices_tensor = TileTensor(
        device_indices,
        row_major(N),
    )

    # Warm up and verify.
    argsort[ascending=True, target="gpu"](
        device_indices_tensor, device_input_tensor, ctx
    )
    # Re-copy input since argsort modifies it in-place.
    ctx.enqueue_copy(device_input, input_host_ptr)
    ctx.synchronize()

    @always_inline
    def bench_ascending(
        mut b: Bencher,
    ) raises {var device_input_tensor, var device_indices_tensor, imm}:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            argsort[ascending=True, target="gpu"](
                device_indices_tensor, device_input_tensor, ctx
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    var num_bytes = N * (size_of[dtype]() + size_of[DType.int64]())
    m.bench_function(
        bench_ascending,
        BenchId("argsort", input_id=String(dtype, "/N=", N)),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )

    ctx.synchronize()

    _ = device_input^
    _ = device_indices^
    _ = input_host_ptr^


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .float32]()
    var N = get_defined_int["N", 131072]()

    var m = Bench()
    with DeviceContext() as ctx:
        bench_argsort[dtype](ctx, m, N)

    m.dump_report()
