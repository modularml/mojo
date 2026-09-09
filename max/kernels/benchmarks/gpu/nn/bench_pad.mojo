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

from std.sys import get_defined_dtype, get_defined_int, size_of

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils import get_defined_shape, int_list_to_tuple
from layout import TileTensor, row_major
from nn.pad_gpu import pad_constant, get_padding_output_shape

from std.utils.index import IndexList


def bench_pad_gpu[
    rank: Int, //, dtype: DType, shape: IndexList[rank], pad_size: Int
](ctx: DeviceContext, mut b: Bench) raises:
    # Create paddings with uniform pre/post padding on all dimensions.
    var paddings_stack = Array[Int, 2 * rank](uninitialized=True)
    var paddings = TileTensor(paddings_stack, row_major[2 * rank]())
    for i in range(rank):
        paddings[2 * i] = Int(pad_size)
        paddings[2 * i + 1] = Int(pad_size)

    var output_shape = get_padding_output_shape(shape, paddings)
    var input_size = shape.flattened_length()
    var output_size = output_shape.flattened_length()

    var in_device = ctx.enqueue_create_buffer[dtype](input_size)
    var out_device = ctx.enqueue_create_buffer[dtype](output_size)
    var constant = Scalar[dtype](0)

    @always_inline
    def bench_fn(mut b: Bencher) raises {mut out_device, imm}:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {mut out_device, imm}:
            pad_constant(
                out_device.unsafe_ptr(),
                output_shape,
                in_device.unsafe_ptr(),
                shape,
                paddings.ptr,
                constant,
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    # Total memory traffic: read input + write output.
    var total_bytes = (input_size + output_size) * size_of[dtype]()

    b.bench_function(
        bench_fn,
        BenchId(
            "pad_constant",
            input_id=String(dtype, "/", shape, "/pad=", pad_size),
        ),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    ctx.synchronize()

    _ = in_device
    _ = out_device


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .float32]()
    comptime shape = int_list_to_tuple[
        get_defined_shape["shape", "256x256"]()
    ]()
    comptime pad_size = get_defined_int["pad_size", 3]()

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        bench_pad_gpu[dtype, shape, pad_size](ctx, m)

    m.dump_report()
