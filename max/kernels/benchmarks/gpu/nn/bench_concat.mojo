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

from std.math import ceildiv
from std.os import abort
from std.random import randn
from std.sys import get_defined_int, size_of

from max.algorithm.functional import elementwise
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import Coord, TileTensor, row_major, coord_to_index_list
from nn.concat import _concat_gpu_elementwise, _concat_inner_most_single_dim

from std.utils import IndexList, StaticTuple


def bench_concat[
    num_inputs: Int, rank: Int
](
    mut b: Bench,
    shapes: List[IndexList[rank]],
    ctx: DeviceContext,
    axis: Int,
) raises:
    comptime type = DType.float32
    if num_inputs != len(shapes):
        raise Error("num_inputs does not match number of shapes provided")

    var out_axis = 0
    var name = String()

    # Create host and device buffers for input 0
    var shape0 = shapes[0]
    var size0 = shape0.flattened_length()
    var input0_host_buffer = ctx.enqueue_create_host_buffer[type](size0)
    var input0_device_buffer = ctx.enqueue_create_buffer[type](size0)
    ctx.synchronize()
    randn(input0_host_buffer.as_span())
    ctx.enqueue_copy(input0_device_buffer, input0_host_buffer)
    name += String(shape0)
    out_axis += shape0[axis]

    # Create host and device buffers for input 1
    var shape1 = shapes[1]
    var size1 = shape1.flattened_length()
    var input1_host_buffer = ctx.enqueue_create_host_buffer[type](size1)
    var input1_device_buffer = ctx.enqueue_create_buffer[type](size1)
    ctx.synchronize()
    randn(input1_host_buffer.as_span())
    ctx.enqueue_copy(input1_device_buffer, input1_host_buffer)
    name += String(shape1)
    out_axis += shape1[axis]

    # Create output buffers
    var out_shape = shapes[0]
    out_shape[axis] = out_axis
    name += String("->", out_shape)
    var output_size = out_shape.flattened_length()
    var output_host_buffer = ctx.enqueue_create_host_buffer[type](output_size)
    var output_device_buffer = ctx.enqueue_create_buffer[type](output_size)
    ctx.synchronize()
    randn(output_host_buffer.as_span())
    ctx.enqueue_copy(output_device_buffer, output_host_buffer)

    # Create TileTensors with dynamic layouts
    var input0_device = TileTensor(
        input0_device_buffer,
        row_major(Coord(shape0)),
    )
    var input1_device = TileTensor(
        input1_device_buffer,
        row_major(Coord(shape1)),
    )
    var output_device = TileTensor(
        output_device_buffer,
        row_major(Coord(out_shape)),
    )

    # Create input tuple for kernel
    var inputs = StaticTuple[
        TileTensor[
            type,
            input0_device.LayoutType,
            ImmutAnyOrigin,
            Engine=input0_device.Engine,
        ],
        num_inputs,
    ](
        input0_device.as_unsafe_any_origin().as_immut(),
        input1_device.as_unsafe_any_origin().as_immut(),
    )

    # Create host TileTensors for verification
    var input0_host = TileTensor(
        input0_host_buffer,
        row_major(Coord(shape0)),
    )
    var input1_host = TileTensor(
        input1_host_buffer,
        row_major(Coord(shape1)),
    )
    var output_host = TileTensor(
        output_host_buffer,
        row_major(Coord(out_shape)),
    )

    var inputs_host = StaticTuple[
        TileTensor[type, input0_host.LayoutType, MutAnyOrigin],
        num_inputs,
    ](
        input0_host.as_unsafe_any_origin(),
        input1_host.as_unsafe_any_origin(),
    )

    @always_inline
    def bench_func(
        mut b: Bencher, shape: IndexList[rank]
    ) raises {mut output_device, imm}:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {mut output_device, imm}:
            _concat_gpu_elementwise[epilogue_fn=None](
                output_device.as_unsafe_any_origin(), axis, inputs, ctx
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_with_input(
        bench_func,
        BenchId("concat", name),
        out_shape,
        # TODO: Pick relevant benchmetric.
        [
            ThroughputMeasure(
                BenchMetric.elements,
                out_shape.flattened_length() * size_of[type]() * 2,
            )
        ],
    )

    ctx.enqueue_copy(output_host_buffer, output_device_buffer)
    ctx.synchronize()

    var offset = 0
    for i in range(num_inputs):
        var input = inputs_host[i]
        var input_shape = shapes[i]

        def check[width: Int, alignment: Int = 1](coords: Coord) {var}:
            var out_coords = coord_to_index_list(coords)
            out_coords[axis] += offset
            var out_coord = Coord(out_coords)
            if output_host.load[width=1](out_coord) != input.load[width=1](
                coords
            ):
                abort(String("mismatch at coords ", out_coords))

        elementwise[1](check, Coord(input_shape), ctx)
        offset += input_shape[axis]

    _ = input0_device_buffer
    _ = input1_device_buffer
    _ = output_device_buffer


# A/B arm for the FusedConcatSlice static-divisor fold. Benches
# `_concat_inner_most_single_dim` (the inner-most-single-dim concat kernel that
# carries the per-thread row -> n-D `divmod`) twice on identical work: with a
# fully static output layout (`row_major[...]()`, all `ComptimeInt` dims, so the
# divisors fold to magic-multiply + shift) and with a dynamic `Coord(IndexList)`
# layout (runtime `IDIV`). Same kernel either way; `static_shape` selects the arm
# so a single function measures both and isolates the fold's perf delta. The
# dims are explicit comptime params (rank-5; dims 1..rank-2 must be non-trivial
# so they actually decompose -- rank-2 would hit the no-divide fast path).
def bench_concat_inner_most_single_dim[
    dtype: DType,
    d0: Int,
    d1: Int,
    d2: Int,
    d3: Int,
    d4: Int,
    num_inputs: Int,
    static_shape: Bool = False,
](mut b: Bench, ctx: DeviceContext) raises:
    comptime rank = 5
    comptime n_rows = d0 * d1 * d2 * d3 * d4
    comptime n_out = n_rows * num_inputs
    comptime B_SIZE = 32

    # One reused input buffer per concat operand.
    var in_dev = List[DeviceBuffer[dtype]]()
    for _ in range(num_inputs):
        in_dev.append(ctx.enqueue_create_buffer[dtype](n_rows))
    var out_dev = ctx.enqueue_create_buffer[dtype](n_out)
    ctx.synchronize()

    @always_inline
    def bench_fn(mut b: Bencher) raises {mut out_dev, imm}:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {mut out_dev, imm}:
            comptime if static_shape:
                # Fully static layouts -> the row -> n-D divisors fold.
                comptime input_layout = row_major[d0, d1, d2, d3, d4]()
                comptime InLayout = type_of(input_layout)
                var output = TileTensor(
                    out_dev, row_major[d0, d1, d2, d3, num_inputs]()
                )
                # Name the input tile type so the kernel's `InputEngine` param
                # matches the `DeviceBuffer` constructor's storage exactly.
                comptime InTile = type_of(
                    TileTensor(in_dev[0], input_layout)
                    .as_unsafe_any_origin()
                    .as_immut()
                )
                var inputs = StaticTuple[InTile, num_inputs]()

                comptime for n in range(num_inputs):
                    inputs[n] = (
                        TileTensor(in_dev[n], input_layout)
                        .as_unsafe_any_origin()
                        .as_immut()
                    )

                comptime kernel = _concat_inner_most_single_dim[
                    OutputLayoutType=output.LayoutType,
                    output_origin=MutAnyOrigin,
                    OutputEngine=output.Engine,
                    InputLayoutType=InLayout,
                    input_origin=ImmutAnyOrigin,
                    InputEngine=InTile.Engine,
                    dtype=dtype,
                    num_inputs=num_inputs,
                    block_size=B_SIZE,
                    epilogue_fn=None,
                ]
                ctx.enqueue_function[kernel](
                    output.as_unsafe_any_origin(),
                    inputs,
                    grid_dim=(ceildiv(n_rows, B_SIZE)),
                    block_dim=(B_SIZE),
                )
            else:
                # All-runtime `Coord(IndexList)` layouts -> runtime `IDIV`.
                var in_shape = IndexList[rank](d0, d1, d2, d3, d4)
                var out_shape = IndexList[rank](d0, d1, d2, d3, num_inputs)
                comptime InLayout = type_of(row_major(Coord(in_shape)))
                var output = TileTensor(out_dev, row_major(Coord(out_shape)))
                # Name the input tile type so the kernel's `InputEngine` param
                # matches the `DeviceBuffer` constructor's storage exactly.
                comptime InTile = type_of(
                    TileTensor(in_dev[0], row_major(Coord(in_shape)))
                    .as_unsafe_any_origin()
                    .as_immut()
                )
                var inputs = StaticTuple[InTile, num_inputs]()

                comptime for n in range(num_inputs):
                    inputs[n] = (
                        TileTensor(in_dev[n], row_major(Coord(in_shape)))
                        .as_unsafe_any_origin()
                        .as_immut()
                    )

                comptime kernel = _concat_inner_most_single_dim[
                    OutputLayoutType=output.LayoutType,
                    output_origin=MutAnyOrigin,
                    OutputEngine=output.Engine,
                    InputLayoutType=InLayout,
                    input_origin=ImmutAnyOrigin,
                    InputEngine=InTile.Engine,
                    dtype=dtype,
                    num_inputs=num_inputs,
                    block_size=B_SIZE,
                    epilogue_fn=None,
                ]
                ctx.enqueue_function[kernel](
                    output.as_unsafe_any_origin(),
                    inputs,
                    grid_dim=(ceildiv(n_rows, B_SIZE)),
                    block_dim=(B_SIZE),
                )

        bencher_iter_custom(b, kernel_launch, ctx)

    comptime shape_tag = "static" if static_shape else "dynamic"
    b.bench_function(
        bench_fn,
        BenchId(
            "concat_inner_most_single_dim",
            input_id=String(shape_tag, dtype, d0, d1, d2, d3, d4, sep="/"),
        ),
        [ThroughputMeasure(BenchMetric.elements, n_out * size_of[dtype]() * 2)],
    )

    ctx.synchronize()
    _ = in_dev^
    _ = out_dev


def main() raises:
    comptime num_inputs = get_defined_int["num_inputs", 2]()
    comptime axis = get_defined_int["axis", 0]()
    comptime W0 = get_defined_int["W0", 1]()
    comptime X0 = get_defined_int["X0", 1]()
    comptime Y0 = get_defined_int["Y0", 1]()
    comptime Z0 = get_defined_int["Z0", 1]()

    comptime W1 = get_defined_int["W1", 1]()
    comptime X1 = get_defined_int["X1", 1]()
    comptime Y1 = get_defined_int["Y1", 1]()
    comptime Z1 = get_defined_int["Z1", 1]()

    var b = Bench()
    with DeviceContext() as ctx:
        bench_concat[num_inputs=num_inputs](
            b,
            [IndexList[4](W0, X0, Y0, Z0), IndexList[4](W1, X1, Y1, Z1)],
            ctx,
            axis=axis,
        )

        # FusedConcatSlice static-divisor fold A/B: rank-5 dims with non-trivial
        # outer dims (rank-2 would hit the no-divide fast path) so dims 1..rank-2
        # actually decompose. Shape is sized to the instruction-bound regime
        # (~65K rows): small enough that the per-thread divide is on the critical
        # path (so the fold shows a ~1.3x delta), but large enough to be stable.
        # Much larger shapes saturate HBM and hide the divide behind memory
        # latency. Static arm folds the divisors; dynamic arm emits the divide.
        bench_concat_inner_most_single_dim[
            DType.float32, 4, 8, 64, 32, 1, num_inputs=4, static_shape=False
        ](b, ctx)
        bench_concat_inner_most_single_dim[
            DType.float32, 4, 8, 64, 32, 1, num_inputs=4, static_shape=True
        ](b, ctx)

        b.dump_report()
