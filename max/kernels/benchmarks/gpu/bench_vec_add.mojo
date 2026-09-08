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
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.builtin._closure import __ownership_keepalive
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from internal_utils import update_bench_config_args
from std.testing import assert_equal


def vec_func(
    in0: ImmPointer[Float32, ImmutAnyOrigin],
    in1: ImmPointer[Float32, ImmutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    len: Int32,
):
    var tid = global_idx.x
    if tid >= Int(len):
        return
    output[tid] = in0[tid] + in1[tid]


@no_inline
def bench_vec_add(
    mut b: Bench, *, block_dim: Int, length: Int, context: DeviceContext
) raises:
    comptime dtype = DType.float32
    var in0_host = List(length=length, fill=Scalar[dtype](0))
    var in1_host = List(length=length, fill=Scalar[dtype](2))
    var out_host = List(length=length, fill=Scalar[dtype](0))

    for i in range(length):
        in0_host[i] = Float32(i)

    var in0_device = context.enqueue_create_buffer[dtype](length)
    var in1_device = context.enqueue_create_buffer[dtype](length)
    var out_device = context.enqueue_create_buffer[dtype](length)

    context.enqueue_copy(in0_device, in0_host)
    context.enqueue_copy(in1_device, in1_host)

    @always_inline
    def kernel_launch(ctx: DeviceContext) raises {mut out_device, imm}:
        context.enqueue_function[vec_func](
            in0_device,
            in1_device,
            out_device,
            Int32(length),
            grid_dim=(length // block_dim),
            block_dim=(block_dim),
        )

    @always_inline
    def bench_func(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, context)

    b.bench_function(
        bench_func,
        BenchId("vec_add", input_id=String("block_dim=", block_dim)),
        [ThroughputMeasure(BenchMetric.flops, length)],
    )
    context.synchronize()
    context.enqueue_copy(out_host, out_device)

    for i in range(length):
        assert_equal(Scalar[dtype](i + 2), out_host[i])

    __ownership_keepalive(in0_device, in1_device, out_device)
    _ = in0_host^
    _ = in1_host^
    _ = out_host^


def main() raises:
    comptime block_dim = get_defined_int["block_dim", 32]()
    var m = Bench()
    update_bench_config_args(m)

    with DeviceContext() as ctx:
        bench_vec_add(m, block_dim=block_dim, length=32 * 1024, context=ctx)

    m.dump_report()
