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

from std.collections import Optional
from std.math import ceildiv
from std.sys import size_of

from max.gpu.host import DeviceBuffer, DeviceContext
from layout import Coord, TileTensor, row_major
from nn.concat import (
    _concat_gpu,
    _concat_inner_most_single_dim,
    elementwise_epilogue_type,
)
from std.testing import assert_equal, assert_true

from std.utils import IndexList, StaticTuple


def test_concat_4_inputs_rank5[test_epilogue: Bool](ctx: DeviceContext) raises:
    print("== test_concat_4_inputs_rank5")

    comptime rank = 5
    comptime dtype = DType.float32

    comptime d0 = 1
    comptime d1 = 128
    comptime d2 = 32
    comptime d3 = 64
    comptime d4 = 1

    comptime input_layout = row_major[d0, d1, d2, d3, d4]()
    comptime output_layout = row_major[d0, d1, d2, d3, 4]()

    # Create host buffers
    var input_0_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        input_layout.product()
    )
    var input_1_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        input_layout.product()
    )
    var input_2_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        input_layout.product()
    )
    var input_3_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        input_layout.product()
    )
    var output_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        output_layout.product()
    )
    ctx.synchronize()

    # Create TileTensors from host buffers and fill with arange pattern
    var input_0_host = TileTensor(input_0_host_buffer, input_layout)
    var input_1_host = TileTensor(input_1_host_buffer, input_layout)
    var input_2_host = TileTensor(input_2_host_buffer, input_layout)
    var input_3_host = TileTensor(input_3_host_buffer, input_layout)

    # Fill with arange pattern
    for i in range(input_layout.product()):
        input_0_host_buffer[i] = Float32(i)
        input_1_host_buffer[i] = Float32(i)
        input_2_host_buffer[i] = Float32(i)
        input_3_host_buffer[i] = Float32(i)

    # Create device buffers
    var input_0_device_buffer = ctx.enqueue_create_buffer[dtype](
        input_layout.product()
    )
    var input_1_device_buffer = ctx.enqueue_create_buffer[dtype](
        input_layout.product()
    )
    var input_2_device_buffer = ctx.enqueue_create_buffer[dtype](
        input_layout.product()
    )
    var input_3_device_buffer = ctx.enqueue_create_buffer[dtype](
        input_layout.product()
    )
    var output_device_buffer = ctx.enqueue_create_buffer[dtype](
        output_layout.product()
    )

    # Copy host to device
    ctx.enqueue_copy(input_0_device_buffer, input_0_host_buffer)
    ctx.enqueue_copy(input_1_device_buffer, input_1_host_buffer)
    ctx.enqueue_copy(input_2_device_buffer, input_2_host_buffer)
    ctx.enqueue_copy(input_3_device_buffer, input_3_host_buffer)
    ctx.synchronize()

    # Create TileTensors from device buffers using dynamic layouts
    # Use Coord(IndexList) pattern that benchmark uses to create runtime layouts
    var input_shape = IndexList[rank](d0, d1, d2, d3, d4)
    var output_shape = IndexList[rank](d0, d1, d2, d3, 4)

    var input_0_dyn = TileTensor(
        input_0_device_buffer, row_major(Coord(input_shape))
    )
    var input_1_dyn = TileTensor(
        input_1_device_buffer, row_major(Coord(input_shape))
    )
    var input_2_dyn = TileTensor(
        input_2_device_buffer, row_major(Coord(input_shape))
    )
    var input_3_dyn = TileTensor(
        input_3_device_buffer, row_major(Coord(input_shape))
    )
    var output_dyn = TileTensor(
        output_device_buffer, row_major(Coord(output_shape))
    )

    comptime B_SIZE = 32

    @__parameter
    @always_inline
    @__copy_capture(output_dyn)
    def epilogue_plus_one[
        c_type: DType, _rank: Int, width: SIMDLength, *, alignment: Int
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        var coord = Coord(indices)
        comptime assert output_dyn.flat_rank >= coord.flat_rank
        output_dyn.store[width=width](
            coord,
            rebind[SIMD[dtype, width]](val + 1),
        )

    comptime kernel = _concat_inner_most_single_dim[
        OutputLayoutType=output_dyn.LayoutType,
        output_origin=MutAnyOrigin,
        OutputEngine=output_dyn.Engine,
        InputLayoutType=input_0_dyn.LayoutType,
        input_origin=ImmutAnyOrigin,
        InputEngine=input_0_dyn.Engine,
        dtype=dtype,
        num_inputs=4,
        block_size=B_SIZE,
        epilogue_fn=Optional[elementwise_epilogue_type](
            epilogue_plus_one
        ) if test_epilogue else None,
    ]

    @always_inline
    def run_concat_inner_most_single_dim(
        ctx: DeviceContext,
    ) raises {
        var output_dyn,
        var input_0_dyn,
        var input_1_dyn,
        var input_2_dyn,
        var input_3_dyn,
        imm,
    }:
        ctx.enqueue_function[kernel](
            output_dyn.as_unsafe_any_origin(),
            StaticTuple[
                TileTensor[dtype, input_0_dyn.LayoutType, ImmutAnyOrigin],
                4,
            ](
                input_0_dyn.as_unsafe_any_origin().as_immut(),
                input_1_dyn.as_unsafe_any_origin().as_immut(),
                input_2_dyn.as_unsafe_any_origin().as_immut(),
                input_3_dyn.as_unsafe_any_origin().as_immut(),
            ),
            grid_dim=(d0 * d1 * d2 * d3 * d4 // B_SIZE),
            block_dim=(B_SIZE),
        )

    var nstime_kernel = ctx.execution_time(run_concat_inner_most_single_dim, 1)
    print(
        "concat_inner_most_single_dim time = ",
        Float64(nstime_kernel) * 1e-6,
        " ms",
    )
    print(
        "transfer rate = ",
        Float64(output_dyn.num_elements() * size_of[UInt8]() * 2)
        * 1e9
        / Float64((1024**3))
        / Float64(nstime_kernel),
        "GB/s",
    )

    # Copy output back to host
    var output_host = TileTensor(output_host_buffer, output_layout)
    ctx.enqueue_copy(output_host_buffer, output_device_buffer)
    ctx.synchronize()

    def validate_results() raises {imm}:
        for i in range(d0):
            for j in range(d1):
                for k in range(d2):
                    for l in range(d3):
                        comptime tail_val = 1 if test_epilogue else 0
                        var not_match_0 = output_host[
                            i, j, k, l, 0
                        ] != input_0_host[i, j, k, l, 0] + Float32(tail_val)
                        var not_match_1 = output_host[
                            i, j, k, l, 1
                        ] != input_1_host[i, j, k, l, 0] + Float32(tail_val)
                        var not_match_2 = output_host[
                            i, j, k, l, 2
                        ] != input_2_host[i, j, k, l, 0] + Float32(tail_val)
                        var not_match_3 = output_host[
                            i, j, k, l, 3
                        ] != input_3_host[i, j, k, l, 0] + Float32(tail_val)
                        if (
                            not_match_0
                            or not_match_1
                            or not_match_2
                            or not_match_3
                        ):
                            assert_true(False, msg="❌ Test failed!")
                            return

    validate_results()

    @always_inline
    def run_concat_gpu(
        ctx: DeviceContext,
    ) raises {
        var output_dyn,
        var input_0_dyn,
        var input_1_dyn,
        var input_2_dyn,
        var input_3_dyn,
        imm,
    }:
        # uses default stream
        _concat_gpu[
            epilogue_fn=Optional[elementwise_epilogue_type](
                epilogue_plus_one
            ) if test_epilogue else None
        ](
            output_dyn.as_unsafe_any_origin(),
            4,
            StaticTuple[
                TileTensor[dtype, input_0_dyn.LayoutType, ImmutAnyOrigin],
                4,
            ](
                input_0_dyn.as_unsafe_any_origin().as_immut(),
                input_1_dyn.as_unsafe_any_origin().as_immut(),
                input_2_dyn.as_unsafe_any_origin().as_immut(),
                input_3_dyn.as_unsafe_any_origin().as_immut(),
            ),
            ctx,
        )

    var nstime = ctx.execution_time(run_concat_gpu, 1)
    print("concat_gpu time = ", Float64(nstime) * 1e-6, " ms")
    print(
        "transfer rate = ",
        Float64(output_dyn.num_elements() * size_of[UInt8]() * 2)
        * 1e9
        / Float64((1024**3))
        / Float64(nstime),
        "GB/s",
    )

    ctx.enqueue_copy(output_host_buffer, output_device_buffer)
    ctx.synchronize()

    validate_results()

    _ = input_0_device_buffer
    _ = input_1_device_buffer
    _ = input_2_device_buffer
    _ = input_3_device_buffer
    _ = output_device_buffer


def test_inner_most_single_dim_static_vs_dynamic(ctx: DeviceContext) raises:
    """Static-divisor fold numerical-equivalence check.

    Runs `_concat_inner_most_single_dim` twice on identical inputs: once with a
    fully static output layout (`row_major[...]()`, all `ComptimeInt` dims, so
    the per-thread row -> n-D `divmod` strength-reduces to magic-multiply +
    shift) and once with the dynamic `Coord(IndexList)` output layout (runtime
    `IDIV`). The two device outputs must be bit-identical: the fold only changes
    which instructions the compiler emits, not the arithmetic result.
    """
    comptime rank = 5
    comptime dtype = DType.float32

    # rank-5 with non-trivial outer dims so dims 1..rank-2 actually decompose
    # (rank==2 would hit the fast path and never divide).
    comptime d0 = 2
    comptime d1 = 7
    comptime d2 = 5
    comptime d3 = 9
    comptime d4 = 1
    comptime num_inputs = 4

    comptime input_layout = row_major[d0, d1, d2, d3, d4]()
    comptime static_output_layout = row_major[d0, d1, d2, d3, num_inputs]()
    comptime n_rows = d0 * d1 * d2 * d3 * d4
    comptime n_out = d0 * d1 * d2 * d3 * num_inputs
    comptime B_SIZE = 32

    # One reused host staging buffer; each input gets a disjoint value range
    # (base offset `n * n_rows`) so per-input values stay distinguishable in the
    # concatenated output.
    var input_host = ctx.enqueue_create_host_buffer[dtype](n_rows)
    ctx.synchronize()

    var in_dev = List[DeviceBuffer[dtype]]()
    for n in range(num_inputs):
        var b = ctx.enqueue_create_buffer[dtype](n_rows)
        for i in range(n_rows):
            input_host[i] = Float32(n * n_rows + i)
        ctx.enqueue_copy(b, input_host)
        in_dev.append(b)
    ctx.synchronize()

    # Two output device buffers, identical contents target.
    var out_static_dev = ctx.enqueue_create_buffer[dtype](n_out)
    var out_dynamic_dev = ctx.enqueue_create_buffer[dtype](n_out)

    # Static-layout tensors (fold fires).
    comptime StaticInLayout = type_of(input_layout)
    var out_static = TileTensor(out_static_dev, static_output_layout)
    # Name the input tile type so the kernel's `InputEngine` param matches the
    # `DeviceBuffer` constructor's storage exactly (the tuple elements are built
    # from the same expression); the tuple can't be indexed at comptime.
    comptime StaticInTile = type_of(
        TileTensor(in_dev[0], input_layout).as_unsafe_any_origin().as_immut()
    )
    var ins_static = StaticTuple[StaticInTile, num_inputs]()

    comptime for n in range(num_inputs):
        ins_static[n] = (
            TileTensor(in_dev[n], input_layout)
            .as_unsafe_any_origin()
            .as_immut()
        )

    comptime kernel_static = _concat_inner_most_single_dim[
        OutputLayoutType=out_static.LayoutType,
        output_origin=MutAnyOrigin,
        OutputEngine=out_static.Engine,
        InputLayoutType=StaticInLayout,
        input_origin=ImmutAnyOrigin,
        InputEngine=StaticInTile.Engine,
        dtype=dtype,
        num_inputs=num_inputs,
        block_size=B_SIZE,
        epilogue_fn=None,
    ]
    ctx.enqueue_function[kernel_static](
        out_static.as_unsafe_any_origin(),
        ins_static,
        grid_dim=(ceildiv(n_rows, B_SIZE)),
        block_dim=(B_SIZE),
    )

    # Dynamic-layout tensors (runtime divide).
    var dyn_in_shape = IndexList[rank](d0, d1, d2, d3, d4)
    var dyn_out_shape = IndexList[rank](d0, d1, d2, d3, num_inputs)
    var out_dynamic = TileTensor(
        out_dynamic_dev, row_major(Coord(dyn_out_shape))
    )
    comptime DynInLayout = type_of(row_major(Coord(dyn_in_shape)))
    comptime DynInTile = type_of(
        TileTensor(in_dev[0], row_major(Coord(dyn_in_shape)))
        .as_unsafe_any_origin()
        .as_immut()
    )
    var ins_dynamic = StaticTuple[
        TileTensor[dtype, DynInLayout, ImmutAnyOrigin], num_inputs
    ]()

    comptime for n in range(num_inputs):
        ins_dynamic[n] = (
            TileTensor(in_dev[n], row_major(Coord(dyn_in_shape)))
            .as_unsafe_any_origin()
            .as_immut()
        )

    comptime kernel_dynamic = _concat_inner_most_single_dim[
        OutputLayoutType=out_dynamic.LayoutType,
        output_origin=MutAnyOrigin,
        OutputEngine=out_dynamic.Engine,
        InputLayoutType=DynInLayout,
        input_origin=ImmutAnyOrigin,
        InputEngine=DynInTile.Engine,
        dtype=dtype,
        num_inputs=num_inputs,
        block_size=B_SIZE,
        epilogue_fn=None,
    ]
    ctx.enqueue_function[kernel_dynamic](
        out_dynamic.as_unsafe_any_origin(),
        ins_dynamic,
        grid_dim=(ceildiv(n_rows, B_SIZE)),
        block_dim=(B_SIZE),
    )

    var host_static = ctx.enqueue_create_host_buffer[dtype](n_out)
    var host_dynamic = ctx.enqueue_create_host_buffer[dtype](n_out)
    ctx.enqueue_copy(host_static, out_static_dev)
    ctx.enqueue_copy(host_dynamic, out_dynamic_dev)
    ctx.synchronize()

    for i in range(n_out):
        assert_equal(
            host_static[i],
            host_dynamic[i],
            msg="static-fold output diverged from dynamic-divide output",
        )

    _ = in_dev^
    _ = out_static_dev
    _ = out_dynamic_dev


def main() raises:
    with DeviceContext() as ctx:
        test_concat_4_inputs_rank5[True](ctx)
        test_concat_4_inputs_rank5[False](ctx)
        test_inner_most_single_dim_static_vs_dynamic(ctx)
