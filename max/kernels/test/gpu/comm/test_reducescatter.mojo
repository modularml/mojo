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

from std.math.uutils import ualign_down
from std.sys import size_of, simd_width_of
from std.itertools import product
from std.utils.coord import _coerce_dynamic

from layout import Coord, Idx, TileTensor, row_major
from layout.coord import DynamicCoord
from std.collections import Optional
from comm import Signal, MAX_GPUS
from comm.sync import enable_p2p, init_signal_buffer
from comm.reducescatter import (
    reducescatter,
    ReduceScatterConfig,
    elementwise_epilogue_type,
)
from internal_utils._testing import test_value_for_gpu_element
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceMulticastBuffer,
    get_gpu_target,
    HostBuffer,
)
from std.testing import assert_almost_equal, assert_true
from std.utils import StaticTuple
from std.utils.numerics import get_accum_type

# Test hyperparameters
comptime test_dtypes = (DType.bfloat16, DType.float32)
comptime test_gpu_counts = (2, 4, 8)

# 1D test lengths
comptime test_1d_lengths = (
    8 * 1024,  # Small
    8 * 1024 + 8,  # Ragged: +1 bf16 SIMD vector / +2 f32 SIMD vectors
    8 * 1024 + 24,  # Ragged: +3 bf16 SIMD vectors / +6 f32 SIMD vectors
    256 * 1024,  # Medium
    16 * 1024 * 1024,  # Large
    16 * 1024 * 1024 + 8,  # Ragged: +1 bf16 SIMD vector / +2 f32 SIMD vectors
    16 * 1024 * 1024 + 24,  # Ragged: +3 bf16 SIMD vectors / +6 f32 SIMD vectors
)

# 2D test shapes: (M, D) (tested with axis=0 and axis=1)
comptime test_2d_shapes = (
    (16, 128),
    (15, 128),  # Ragged rows (odd number not divisible by ngpus)
    (9, 7168),  # Ragged, Deepseek V3.1 token dim
    (32, 256),
)


def reducescatter_test[
    dtype: DType,
    ngpus: Int,
    rank: Int,
    axis: Int,
    use_custom_epilogue: Bool = False,
    use_multimem: Bool = False,
](list_of_ctx: List[DeviceContext], shape: Coord) raises where (
    shape.flat_rank == rank and shape.rank == rank
):
    """Test reduce-scatter operation (1D and 2D)."""
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"
    comptime assert axis < rank
    comptime assert axis >= 0
    comptime assert rank <= 2, "Only up to 2D supported currently"
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()

    var num_elements = Int(shape.product())

    var multimem_tag = "-multimem" if use_multimem else ""
    var epilogue_tag = "-custom_epilogue" if use_custom_epilogue else ""
    comptime if rank == 1:
        print(
            String(
                "====reducescatter-flat-",
                dtype,
                "-",
                ngpus,
                "gpus-",
                num_elements,
                multimem_tag,
                epilogue_tag,
            )
        )
    else:
        print(
            String(
                "====reducescatter-axis",
                axis,
                "-",
                dtype,
                "-",
                ngpus,
                "gpus-(",
                shape[0].value(),
                "x",
                shape[1].value(),
                ")",
                multimem_tag,
                epilogue_tag,
            )
        )

    # Compute partitioning config.
    var axis_size: Int
    var unit_numel: Int
    comptime if rank == 1:
        axis_size = num_elements // simd_width
        unit_numel = simd_width
    elif axis == 0:
        axis_size = Int(shape[0].value())
        unit_numel = Int(shape[1].value())
    else:
        axis_size = Int(shape[1].value()) // simd_width
        unit_numel = Int(shape[0].value()) * simd_width
    var config = ReduceScatterConfig[dtype, ngpus](axis_size, unit_numel, 0)

    # Allocate and initialize buffers (shared across all axis values).
    var in_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_in = List[HostBuffer[dtype]](capacity=ngpus)

    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    for gpu_idx in range(ngpus):
        if not use_multimem:
            in_bufs_list.append(
                list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](num_elements)
            )
        out_bufs_list.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](
                config.rank_num_elements(gpu_idx)
            )
        )

        var h = list_of_ctx[gpu_idx].enqueue_create_host_buffer[dtype](
            num_elements
        )
        for j in range(num_elements):
            h[j] = test_value_for_gpu_element[dtype](gpu_idx, j)

        if not use_multimem:
            list_of_ctx[gpu_idx].enqueue_copy(in_bufs_list[gpu_idx], h)

        host_in.append(h^)

        signal_buffers.append(
            list_of_ctx[gpu_idx].create_buffer_sync[.uint8](size_of[Signal]())
        )
        init_signal_buffer(signal_buffers[gpu_idx], list_of_ctx[gpu_idx])
        rank_sigs[gpu_idx] = (
            signal_buffers[gpu_idx]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )

    comptime for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Create input buffers.
    # The input/output tile types are built once (`type_of`) then constructed
    # from pointers into different buffers (per-GPU list entries, multicast
    # buffers). `DeviceBuffer.unsafe_ptr()` now returns a tracked origin, so
    # each buffer's pointer has a distinct origin; opt out to `AnyOrigin` so the
    # tile type is origin-agnostic (and the `StaticTuple` writes don't trip a
    # false-positive aliasing exclusivity check).
    comptime InputTileType = type_of(
        TileTensor[mut=False](
            in_bufs_list[0].unsafe_ptr().as_unsafe_any_origin(),
            row_major(shape),
        )
    )
    comptime num_input_bufs = 1 if use_multimem else ngpus
    var in_bufs = Array[InputTileType, num_input_bufs](uninitialized=True)

    comptime if use_multimem:
        var multicast_buf = DeviceMulticastBuffer[dtype](
            list_of_ctx.copy(), num_elements
        )
        comptime for i in range(ngpus):
            var unicast_buf = multicast_buf.unicast_buffer_for(list_of_ctx[i])
            list_of_ctx[i].enqueue_copy(unicast_buf, host_in[i])
        in_bufs[0] = InputTileType(
            multicast_buf.multicast_buffer_for(list_of_ctx[0])
            .unsafe_ptr()
            .as_unsafe_any_origin(),
            row_major(shape),
        )
    else:
        comptime for i in range(ngpus):
            in_bufs[i] = InputTileType(
                in_bufs_list[i].unsafe_ptr().as_unsafe_any_origin(),
                row_major(shape),
            )

    comptime shape_type = DynamicCoord[.int, rank]

    comptime OutputTileType = type_of(
        TileTensor[mut=True](
            out_bufs_list[0].unsafe_ptr().as_unsafe_any_origin(),
            row_major(shape_type()),
        )
    )
    var out_bufs = StaticTuple[OutputTileType, ngpus]()

    comptime for i in range(ngpus):
        var runtime_shape = shape_type()
        comptime if rank == 1:
            runtime_shape[0] = _coerce_dynamic[runtime_shape.element_types[0]](
                config.rank_num_elements(i)
            )
        elif rank == 2:
            comptime if axis == 0:
                runtime_shape[0] = _coerce_dynamic[
                    runtime_shape.element_types[0]
                ](config.rank_units(i))
                runtime_shape[1] = _coerce_dynamic[
                    runtime_shape.element_types[1]
                ](Int(shape[1].value()))
            else:
                runtime_shape[0] = _coerce_dynamic[
                    runtime_shape.element_types[0]
                ](Int(shape[0].value()))
                runtime_shape[1] = _coerce_dynamic[
                    runtime_shape.element_types[1]
                ](config.rank_units(i) * simd_width)

        out_bufs[i] = OutputTileType(
            out_bufs_list[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(runtime_shape),
        )

    @always_inline
    @__parameter
    @__copy_capture(out_bufs)
    def outputs_lambda[
        input_index: Int,
        _dtype: DType,
        _width: SIMDLength,
        *,
        _alignment: Int,
    ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
        var out_buf = out_bufs[input_index]
        out_buf.store[width=_width, alignment=_alignment](
            coords,
            rebind[SIMD[dtype, _width]](-val),
        )

    # `reducescatter`'s output is a world-view `Array`, not a `StaticTuple`;
    # convert once (the actual write goes through `outputs_lambda` above,
    # which still reads the `StaticTuple` directly).
    var out_bufs_arr = Array[OutputTileType, ngpus](
        fill_with=lambda (k: Int) -> OutputTileType: out_bufs[k]
    )

    comptime for i in range(ngpus):
        reducescatter[
            ngpus=ngpus,
            output_lambda=Optional[elementwise_epilogue_type](
                outputs_lambda[input_index=i, ...]
            ) if use_custom_epilogue else None,
            axis=axis,
            use_multimem=use_multimem,
        ](
            in_bufs,
            out_bufs_arr,
            rank_sigs,
            list_of_ctx[i],
            my_rank=Optional[Int](i),
        )

    comptime for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Create TileTensors, run reduce-scatter, and verify results.
    # Branches on axis at comptime because 1D and 2D use different TileTensor
    # ranks (and therefore different types).
    comptime if rank == 1:
        # --- 1D flat case ---

        for gpu_idx in range(ngpus):
            var out_len = config.rank_num_elements(gpu_idx)
            var result_host = list_of_ctx[gpu_idx].enqueue_create_host_buffer[
                dtype
            ](out_len)
            list_of_ctx[gpu_idx].enqueue_copy(
                result_host, out_bufs_list[gpu_idx]
            )
            list_of_ctx[gpu_idx].synchronize()

            for j in range(out_len):
                comptime accum_t = get_accum_type[dtype]()
                var accum = Scalar[accum_t](0)
                var global_idx = config.rank_start(gpu_idx) + j
                comptime for k in range(ngpus):
                    accum += Scalar[accum_t](
                        test_value_for_gpu_element[dtype](k, global_idx)
                    )
                var expected_sum = Scalar[dtype](accum)
                var expected = (
                    -expected_sum if use_custom_epilogue else expected_sum
                )
                assert_almost_equal(
                    result_host[j],
                    expected,
                    msg=String(
                        "GPU ",
                        gpu_idx,
                        " element ",
                        j,
                        " (global ",
                        global_idx,
                        ") mismatch",
                    ),
                )

    elif rank == 2:
        # --- 2D axis-aware case ---

        for gpu_idx in range(ngpus):
            var out_size = config.rank_num_elements(gpu_idx)
            var result_host = list_of_ctx[gpu_idx].enqueue_create_host_buffer[
                dtype
            ](out_size)
            list_of_ctx[gpu_idx].enqueue_copy(
                result_host, out_bufs_list[gpu_idx]
            )
            list_of_ctx[gpu_idx].synchronize()

            if axis == 0:
                var row_start = config.rank_unit_start(gpu_idx)
                var my_rows = config.rank_units(gpu_idx)
                for r in range(my_rows):
                    for c in range(Int(shape[1].value())):
                        var global_flat = (row_start + r) * Int(
                            shape[1].value()
                        ) + c
                        comptime accum_t = get_accum_type[dtype]()
                        var accum = Scalar[accum_t](0)
                        comptime for k in range(ngpus):
                            accum += Scalar[accum_t](
                                test_value_for_gpu_element[dtype](
                                    k, global_flat
                                )
                            )
                        var expected_sum = Scalar[dtype](accum)
                        var expected = (
                            -expected_sum if use_custom_epilogue else expected_sum
                        )
                        assert_almost_equal(
                            result_host[r * Int(shape[1].value()) + c],
                            expected,
                            msg=String(
                                "GPU ",
                                gpu_idx,
                                " axis=0 (",
                                r,
                                ",",
                                c,
                                ") mismatch",
                            ),
                        )
            else:
                var col_start = config.rank_unit_start(gpu_idx) * simd_width
                var my_cols = config.rank_units(gpu_idx) * simd_width
                for r in range(Int(shape[0].value())):
                    for c in range(my_cols):
                        var global_flat = r * Int(shape[1].value()) + (
                            col_start + c
                        )
                        comptime accum_t = get_accum_type[dtype]()
                        var accum = Scalar[accum_t](0)
                        comptime for k in range(ngpus):
                            accum += Scalar[accum_t](
                                test_value_for_gpu_element[dtype](
                                    k, global_flat
                                )
                            )
                        var expected_sum = Scalar[dtype](accum)
                        var expected = (
                            -expected_sum if use_custom_epilogue else expected_sum
                        )
                        assert_almost_equal(
                            result_host[r * my_cols + c],
                            expected,
                            msg=String(
                                "GPU ",
                                gpu_idx,
                                " axis=1 (",
                                r,
                                ",",
                                c,
                                ") mismatch",
                            ),
                        )

    _ = host_in^


@always_inline
def _residual_value[
    dtype: DType
](group_id: Int, flat_idx: Int) -> Scalar[dtype]:
    """Residual pattern: replicated within a group, distinct between groups.

    The group term is what makes a relay folding its own residual instead of
    the destination's a test failure rather than a coincidence.
    """
    return Scalar[dtype]((flat_idx % 7) + 1 + 8 * group_id)


def grouped_reducescatter_test[
    ngpus: Int = 4,
    group_size: Int = 2,
    axis: Int = 0,
    rows_first: Int = 5,
    rows_second: Int = 3,
    D: Int = 128,
    with_residual: Bool = False,
](list_of_ctx: List[DeviceContext]) raises:
    """Test grouped reduce-scatter with group-local ranks and shapes.

    With exactly two groups this also exercises the relay-assisted path, which
    `reducescatter` selects on its own where the interconnect makes it pay: the
    other group's GPUs reduce a trailing slice of each destination's partition
    and write the finished values in. Ragged row counts -- differing between
    the groups, and not dividing evenly inside one -- are the interesting case
    there, since every partition is split on its own length.
    """
    comptime dtype = DType.float32
    comptime rank = 2

    print(
        "====grouped-reducescatter-axis",
        axis,
        "-float32-",
        ngpus,
        "gpus-group",
        group_size,
        "-rows",
        rows_first,
        "/",
        rows_second,
        sep="",
    )

    var in_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var res_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_in = List[HostBuffer[dtype]](capacity=ngpus)
    var host_res = List[HostBuffer[dtype]](capacity=ngpus)

    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    for gpu_idx in range(ngpus):
        var group_rows = rows_first if gpu_idx < group_size else rows_second
        var num_elements = group_rows * D
        in_bufs_list.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](num_elements)
        )

        var local_rank = gpu_idx % group_size
        var config = ReduceScatterConfig[dtype, group_size](group_rows, D, 0)
        out_bufs_list.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](
                config.rank_num_elements(local_rank)
            )
        )

        var h = list_of_ctx[gpu_idx].enqueue_create_host_buffer[dtype](
            num_elements
        )
        for j in range(num_elements):
            h[j] = test_value_for_gpu_element[dtype](gpu_idx, j)
        list_of_ctx[gpu_idx].enqueue_copy(in_bufs_list[gpu_idx], h)
        host_in.append(h^)

        # Replicated within a group, but NOT across groups -- which is why
        # the collective takes a world view of it.
        comptime if with_residual:
            res_bufs_list.append(
                list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](num_elements)
            )
            var rh = list_of_ctx[gpu_idx].enqueue_create_host_buffer[dtype](
                num_elements
            )
            for j in range(num_elements):
                rh[j] = _residual_value[dtype](gpu_idx // group_size, j)
            list_of_ctx[gpu_idx].enqueue_copy(res_bufs_list[gpu_idx], rh)
            host_res.append(rh^)

        signal_buffers.append(
            list_of_ctx[gpu_idx].create_buffer_sync[.uint8](size_of[Signal]())
        )
        init_signal_buffer(signal_buffers[gpu_idx], list_of_ctx[gpu_idx])
        rank_sigs[gpu_idx] = (
            signal_buffers[gpu_idx]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )

    comptime for i in range(ngpus):
        list_of_ctx[i].synchronize()

    comptime shape_type = DynamicCoord[.int, rank]
    comptime InputTileType = type_of(
        TileTensor[mut=False](
            in_bufs_list[0].unsafe_ptr().as_unsafe_any_origin(),
            row_major(shape_type()),
        )
    )
    comptime OutputTileType = type_of(
        TileTensor[mut=True](
            out_bufs_list[0].unsafe_ptr().as_unsafe_any_origin(),
            row_major(shape_type()),
        )
    )
    var in_bufs = StaticTuple[InputTileType, ngpus]()
    var out_bufs = Array[OutputTileType, ngpus](uninitialized=True)

    for gpu_idx in range(ngpus):
        var group_rows = rows_first if gpu_idx < group_size else rows_second
        var local_rank = gpu_idx % group_size
        var config = ReduceScatterConfig[dtype, group_size](group_rows, D, 0)

        var input_shape = shape_type()
        input_shape[0] = _coerce_dynamic[input_shape.element_types[0]](
            group_rows
        )
        input_shape[1] = _coerce_dynamic[input_shape.element_types[1]](D)
        in_bufs._unsafe_ref(gpu_idx) = InputTileType(
            in_bufs_list[gpu_idx].unsafe_ptr().as_unsafe_any_origin(),
            row_major(input_shape),
        )

        var output_shape = shape_type()
        output_shape[0] = _coerce_dynamic[output_shape.element_types[0]](
            config.rank_units(local_rank)
        )
        output_shape[1] = _coerce_dynamic[output_shape.element_types[1]](D)
        out_bufs[gpu_idx] = OutputTileType(
            out_bufs_list[gpu_idx].unsafe_ptr().as_unsafe_any_origin(),
            row_major(output_shape),
        )

    # World-view input array `reducescatter` expects (indexed by GLOBAL
    # device rank); it does its own group-local slicing internally from
    # `group_size` + `my_rank`. `rank_sigs` is already world-view above.
    var world_in_bufs = Array[InputTileType, ngpus](
        fill_with=lambda (i: Int) -> InputTileType: in_bufs[i]
    )

    comptime ResPtrType = ImmPointer[Scalar[dtype], ImmutAnyOrigin]
    var world_res_ptrs = Array[ResPtrType, ngpus](uninitialized=True)
    comptime if with_residual:
        for i in range(ngpus):
            world_res_ptrs[i] = rebind[ResPtrType](
                res_bufs_list[i].unsafe_ptr()
            )

    comptime for gpu_idx in range(ngpus):
        comptime if with_residual:
            reducescatter[
                ngpus=ngpus,
                group_size=group_size,
                axis=axis,
                has_residual=True,
            ](
                world_in_bufs,
                out_bufs,
                rank_sigs,
                list_of_ctx[gpu_idx],
                my_rank=Optional[Int](gpu_idx),
                residuals=Optional(world_res_ptrs.copy()),
            )
        else:
            reducescatter[
                ngpus=ngpus,
                group_size=group_size,
                axis=axis,
            ](
                world_in_bufs,
                out_bufs,
                rank_sigs,
                list_of_ctx[gpu_idx],
                my_rank=Optional[Int](gpu_idx),
            )

    comptime for i in range(ngpus):
        list_of_ctx[i].synchronize()

    for gpu_idx in range(ngpus):
        var group_start = ualign_down(gpu_idx, group_size)
        var group_rows = rows_first if gpu_idx < group_size else rows_second
        var local_rank = gpu_idx % group_size
        var config = ReduceScatterConfig[dtype, group_size](group_rows, D, 0)
        var out_size = config.rank_num_elements(local_rank)
        var result_host = list_of_ctx[gpu_idx].enqueue_create_host_buffer[
            dtype
        ](out_size)
        list_of_ctx[gpu_idx].enqueue_copy(result_host, out_bufs_list[gpu_idx])
        list_of_ctx[gpu_idx].synchronize()

        var row_start = config.rank_unit_start(local_rank)
        var my_rows = config.rank_units(local_rank)
        for r in range(my_rows):
            for c in range(D):
                var global_flat = (row_start + r) * D + c
                comptime accum_t = get_accum_type[dtype]()
                var accum = Scalar[accum_t](0)
                comptime for local_input_idx in range(group_size):
                    accum += Scalar[accum_t](
                        test_value_for_gpu_element[dtype](
                            group_start + local_input_idx, global_flat
                        )
                    )
                comptime if with_residual:
                    # Matches the kernel's order: round the reduction to
                    # `dtype`, then add the residual and round once more.
                    accum = Scalar[accum_t](Scalar[dtype](accum)) + Scalar[
                        accum_t
                    ](
                        _residual_value[dtype](
                            group_start // group_size, global_flat
                        )
                    )
                assert_almost_equal(
                    result_host[r * D + c],
                    Scalar[dtype](accum),
                    msg=String(
                        "GPU ",
                        gpu_idx,
                        " grouped axis=0 (",
                        r,
                        ",",
                        c,
                        ") mismatch",
                    ),
                )

    _ = host_in^
    _ = host_res^
    _ = res_bufs_list^


@__parameter
def run_reducescatter_sweep[use_multimem: Bool]() raises:
    """Run reduce-scatter tests across 1D and 2D configurations."""
    var list_of_ctx = List[DeviceContext](capacity=MAX_GPUS)
    for i in range(DeviceContext.number_of_devices()):
        list_of_ctx.append(DeviceContext(i))

    # 1D flat tests.
    comptime for dtype_idx, ngpus_idx, length_idx, epilogue_idx in product(
        range(len(test_dtypes)),
        range(len(test_gpu_counts)),
        range(len(test_1d_lengths)),
        range(2),
    ):
        comptime dtype = rebind[DType](test_dtypes[dtype_idx])
        comptime ngpus = rebind[Int](test_gpu_counts[ngpus_idx])
        comptime length = rebind[Int](test_1d_lengths[length_idx])
        comptime use_custom_epilogue = epilogue_idx == 1

        if DeviceContext.number_of_devices() < ngpus:
            continue

        try:
            reducescatter_test[
                dtype=dtype,
                ngpus=ngpus,
                rank=1,
                axis=0,
                use_custom_epilogue=use_custom_epilogue,
                use_multimem=use_multimem,
            ](list_of_ctx, Coord(length))
        except e:
            if (
                use_multimem
                and "multimem is only supported on SM90+ GPUs" in String(e)
            ):
                print(
                    "Skipping multimem test - SM90+ not supported by"
                    " compilation target"
                )
            else:
                raise e^

    # 2D axis tests.
    comptime for dtype_idx, ngpus_idx, shape_idx, epilogue_idx in product(
        range(len(test_dtypes)),
        range(len(test_gpu_counts)),
        range(len(test_2d_shapes)),
        range(2),
    ):
        comptime dtype = rebind[DType](test_dtypes[dtype_idx])
        comptime ngpus = rebind[Int](test_gpu_counts[ngpus_idx])
        comptime M = rebind[type_of(test_2d_shapes[0])](
            test_2d_shapes[shape_idx]
        )[0]
        comptime D = rebind[type_of(test_2d_shapes[0])](
            test_2d_shapes[shape_idx]
        )[1]
        comptime use_custom_epilogue = epilogue_idx == 1

        if DeviceContext.number_of_devices() < ngpus:
            continue

        comptime for axis in range(2):
            try:
                reducescatter_test[
                    dtype=dtype,
                    ngpus=ngpus,
                    rank=2,
                    axis=axis,
                    use_custom_epilogue=use_custom_epilogue,
                    use_multimem=use_multimem,
                ](list_of_ctx, Coord(M, D))
            except e:
                if (
                    use_multimem
                    and "multimem is only supported on SM90+ GPUs" in String(e)
                ):
                    print(
                        "Skipping multimem test - SM90+ not supported by"
                        " compilation target"
                    )
                else:
                    raise e^


def main() raises:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    # Standard (non-multimem) sweep
    run_reducescatter_sweep[use_multimem=False]()
    if DeviceContext.number_of_devices() >= 4:
        var list_of_ctx = List[DeviceContext](capacity=MAX_GPUS)
        for i in range(DeviceContext.number_of_devices()):
            list_of_ctx.append(DeviceContext(i))
        # Two groups of two: ragged between the groups (5 vs 3 rows) and
        # inside one (5 rows over 2 ranks), which is the shape the relay path
        # splits per partition.
        grouped_reducescatter_test(list_of_ctx)
        # A partition small enough that the tuning table declines to relay,
        # so the pair takes the plain path.
        grouped_reducescatter_test[rows_first=2, rows_second=2, D=8](
            list_of_ctx
        )
        # A whole group empty while its partner is not. Only this shape
        # reaches the pair-widened emptiness scan: the empty group has
        # nothing of its own to reduce but still has to launch, on the same
        # grid and into the same barrier, because its GPUs act as reduce
        # relays for the partner. Returning early there hangs the node.
        grouped_reducescatter_test[rows_first=1024, rows_second=0](list_of_ctx)
        # The same with the empty group first, so the relay role is driven
        # from both sides of the pair.
        grouped_reducescatter_test[rows_first=0, rows_second=1024](list_of_ctx)
        # Residual fold at a relay-active size: a relay folds the replicated
        # residual from its OWN copy at the destination's offset, so this is
        # the case that would catch it reading the wrong slice.
        grouped_reducescatter_test[rows_first=1024, rows_second=1024](
            list_of_ctx
        )
        grouped_reducescatter_test[
            rows_first=1024, rows_second=1024, with_residual=True
        ](list_of_ctx)
        # And ragged, so the relays' residual offsets differ per destination.
        grouped_reducescatter_test[rows_first=1024, rows_second=768](
            list_of_ctx
        )
        grouped_reducescatter_test[
            rows_first=1024, rows_second=768, with_residual=True
        ](list_of_ctx)

    if DeviceContext.number_of_devices() == 8:
        var list_of_ctx = List[DeviceContext](capacity=MAX_GPUS)
        for i in range(8):
            list_of_ctx.append(DeviceContext(i))
        # Two groups of four: the shape the relay recipe targets, and the one
        # where the pair spans the whole node, so its barrier takes the
        # full-world counter bank rather than a grouped one.
        grouped_reducescatter_test[ngpus=8, group_size=4](list_of_ctx)
        grouped_reducescatter_test[
            ngpus=8, group_size=4, rows_first=9, rows_second=7, D=512
        ](list_of_ctx)
        # One whole group of four empty, relaying for a non-empty partner.
        grouped_reducescatter_test[
            ngpus=8, group_size=4, rows_first=4096, rows_second=0
        ](list_of_ctx)
        grouped_reducescatter_test[
            ngpus=8,
            group_size=4,
            rows_first=4096,
            rows_second=4096,
            with_residual=True,
        ](list_of_ctx)

    print("All reduce-scatter tests passed!")
