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

from std.sys import get_defined_int

from max.benchmark import bencher_iter_custom
from std.benchmark import Bench, Bencher, BenchId
from std.builtin._closure import __ownership_keepalive
from max.gpu import (
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.primitives.grid_controls import (
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from max.gpu.primitives.grid_controls import pdl_launch_attributes
from max.gpu.host import DeviceContext


def copy1(
    a: ImmPointer[Float32, ImmutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n: Int32,
):
    var tmp = Float32()
    for i in range(
        block_idx.x * block_dim.x + thread_idx.x,
        Int(n),
        block_dim.x * grid_dim.x,
    ):
        tmp += b[i]

    launch_dependent_grids()

    for i in range(
        block_idx.x * block_dim.x + thread_idx.x,
        Int(n),
        block_dim.x * grid_dim.x,
    ):
        b[i] = a[i] + tmp


def copy2(
    b: ImmPointer[Float32, ImmutAnyOrigin],
    c: MutPointer[Float32, MutAnyOrigin],
    d: ImmPointer[Float32, ImmutAnyOrigin],
    n: Int32,
):
    var result = Float32()
    for i in range(
        block_idx.x * block_dim.x + thread_idx.x,
        Int(n),
        block_dim.x * grid_dim.x,
    ):
        result += d[i]

    wait_on_dependent_grids()

    for i in range(
        block_idx.x * block_dim.x + thread_idx.x,
        Int(n),
        block_dim.x * grid_dim.x,
    ):
        c[i] = b[i] + result + 2.0


def copy1_n(
    a: ImmPointer[Float32, ImmutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n: Int32,
):
    var tmp = Float32()
    for i in range(
        block_idx.x * block_dim.x + thread_idx.x,
        Int(n),
        block_dim.x * grid_dim.x,
    ):
        tmp += b[i]

    for i in range(
        block_idx.x * block_dim.x + thread_idx.x,
        Int(n),
        block_dim.x * grid_dim.x,
    ):
        b[i] = a[i] + tmp


def copy2_n(
    b: ImmPointer[Float32, ImmutAnyOrigin],
    c: MutPointer[Float32, MutAnyOrigin],
    d: ImmPointer[Float32, ImmutAnyOrigin],
    n: Int32,
):
    var result = Float32()
    for i in range(
        block_idx.x * block_dim.x + thread_idx.x,
        Int(n),
        block_dim.x * grid_dim.x,
    ):
        result += d[i]

    for i in range(
        block_idx.x * block_dim.x + thread_idx.x,
        Int(n),
        block_dim.x * grid_dim.x,
    ):
        c[i] = b[i] + result + 2.0


@no_inline
def bench_pdl_copy(mut b: Bench, *, length: Int, context: DeviceContext) raises:
    comptime dtype = DType.float32
    var a_host = List(length=length, fill=Scalar[dtype](0))
    var b_host = List(length=length, fill=Scalar[dtype](0))
    var c_host = List(length=length, fill=Scalar[dtype](0))
    var d_host = List(length=length, fill=Scalar[dtype](0))

    comptime grid_dim = 16
    comptime block_dim = 256

    for i in range(length):
        a_host[i] = Float32(i)
        b_host[i] = Float32(i)
        d_host[i] = Float32(i)

    var a_device = context.enqueue_create_buffer[dtype](length)
    var b_device = context.enqueue_create_buffer[dtype](length)
    var c_device = context.enqueue_create_buffer[dtype](length)
    var d_device = context.enqueue_create_buffer[dtype](length)

    context.enqueue_copy(a_device, a_host)
    context.enqueue_copy(b_device, b_host)
    context.enqueue_copy(c_device, c_host)
    context.enqueue_copy(d_device, d_host)

    @always_inline
    def kernel_launch(
        ctx: DeviceContext,
    ) raises {mut b_device, mut c_device, imm}:
        for _ in range(10):
            context.enqueue_function[copy1](
                a_device,
                b_device,
                Int32(length),
                grid_dim=(grid_dim),
                block_dim=(block_dim),
                attributes=pdl_launch_attributes(),
            )
            context.enqueue_function[copy2](
                b_device,
                c_device,
                d_device,
                Int32(length),
                grid_dim=(grid_dim),
                block_dim=(block_dim),
                attributes=pdl_launch_attributes(),
            )

    @always_inline
    def bench_func(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, context)

    b.bench_function(
        bench_func,
        BenchId("copy_pdl", input_id=String("length=", length)),
    )
    context.synchronize()
    context.enqueue_copy(c_host, c_device)

    __ownership_keepalive(a_device, b_device, c_device, d_device)
    _ = a_host^
    _ = b_host^
    _ = c_host^
    _ = d_host^


@no_inline
def bench_copy(mut b: Bench, *, length: Int, context: DeviceContext) raises:
    comptime dtype = DType.float32
    var a_host = List(length=length, fill=Scalar[dtype](0))
    var b_host = List(length=length, fill=Scalar[dtype](0))
    var c_host = List(length=length, fill=Scalar[dtype](0))
    var d_host = List(length=length, fill=Scalar[dtype](0))

    comptime grid_dim = 16
    comptime block_dim = 256

    for i in range(length):
        a_host[i] = Float32(i)
        b_host[i] = Float32(i)
        d_host[i] = Float32(i)

    var a_device = context.enqueue_create_buffer[dtype](length)
    var b_device = context.enqueue_create_buffer[dtype](length)
    var c_device = context.enqueue_create_buffer[dtype](length)
    var d_device = context.enqueue_create_buffer[dtype](length)

    context.enqueue_copy(a_device, a_host)
    context.enqueue_copy(b_device, b_host)
    context.enqueue_copy(c_device, c_host)
    context.enqueue_copy(d_device, d_host)

    @always_inline
    def kernel_launch(
        ctx: DeviceContext,
    ) raises {mut b_device, mut c_device, imm}:
        for _ in range(10):
            context.enqueue_function[copy1_n](
                a_device,
                b_device,
                Int32(length),
                grid_dim=(grid_dim),
                block_dim=(block_dim),
            )
            context.enqueue_function[copy2_n](
                b_device,
                c_device,
                d_device,
                Int32(length),
                grid_dim=(grid_dim),
                block_dim=(block_dim),
            )

    @always_inline
    def bench_func(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, context)

    b.bench_function(
        bench_func,
        BenchId("copy_n", input_id=String("length=", length)),
    )
    context.synchronize()
    context.enqueue_copy(c_host, c_device)

    __ownership_keepalive(a_device, b_device, c_device, d_device)
    _ = a_host^
    _ = b_host^
    _ = c_host^
    _ = d_host^


def main() raises:
    comptime length = get_defined_int["length", 4096]()
    var m = Bench()

    with DeviceContext() as ctx:
        bench_pdl_copy(m, length=length, context=ctx)
        bench_copy(m, length=length, context=ctx)

    m.dump_report()
