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
"""Implements GPU-specific tensor padding kernels with constant or edge-fill strategies."""

from std.algorithm.functional import vectorize
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer, DeviceAttribute
from layout import Coord, TensorEngine, TensorLayout, TileTensor
from layout.tile_layout import Layout
from std.math import align_down, ceildiv
from std.sys.info import align_of
from std.collections import Array
from std.utils.index import IndexList


def _fill_strides_indexlist[
    rank: Int,
](input_shape: IndexList[rank], mut strides: Array[Int, rank],):
    """
    Fill `strides`, which will be an array of strides indexed by axis, assuming
    `buf` contains contiguous buf.

    Note that `buf` is only used for querying its dimensions.
    """
    comptime assert rank > 0
    strides[rank - 1] = 1

    comptime for idx in range(rank - 1):
        comptime axis = rank - idx - 2
        var next_axis_stride = strides[axis + 1]
        var next_axis_dim = input_shape[axis + 1]
        var curr_axis_stride = next_axis_stride * next_axis_dim
        strides[axis] = curr_axis_stride


@always_inline
def _vectorized_copy_row[
    dtype: DType,
    simd_width: Int,
](
    input_ptr: UnsafePointer[mut=False, Scalar[dtype], _],
    output_ptr: UnsafePointer[mut=True, Scalar[dtype], _],
    row_length: Int,
    threads_per_row: Int,
):
    """Cooperatively copy a row with coalesced, vectorize-unrolled strided
    access and aligned loads/stores.

    Each thread handles every `threads_per_row`-th SIMD block so that threads
    in the same warp touch adjacent blocks each iteration, preserving GPU
    memory coalescing.  `vectorize` drives the per-thread block loop, giving
    automatic remainder handling and instruction-level-parallelism via
    unrolling.  When both pointers satisfy the SIMD alignment requirement,
    aligned loads/stores are emitted for more efficient memory transactions.
    """
    var tid = thread_idx.x % threads_per_row

    var scaled = align_down(row_length, simd_width)
    var stride = threads_per_row * simd_width
    var my_start = tid * simd_width

    # Number of SIMD blocks this thread is responsible for.
    var num_blocks = (
        ceildiv(scaled - my_start, stride) if my_start < scaled else 0
    )

    comptime alignment = align_of[SIMD[dtype, simd_width]]()
    var src_aligned = Int(input_ptr) % alignment == 0
    var dst_aligned = Int(output_ptr) % alignment == 0

    if src_aligned and dst_aligned:

        def _copy_aligned[n: Int](blk: Int) {input_ptr, output_ptr, mut}:
            for j in range(n):
                var off = my_start + (blk + j) * stride
                output_ptr.store[width=simd_width, alignment=alignment](
                    off,
                    input_ptr.load[width=simd_width, alignment=alignment](off),
                )

        vectorize[simd_width](num_blocks, _copy_aligned)
    else:

        def _copy[n: Int](blk: Int) {input_ptr, output_ptr, mut}:
            for j in range(n):
                var off = my_start + (blk + j) * stride
                output_ptr.store[width=simd_width](
                    off, input_ptr.load[width=simd_width](off)
                )

        vectorize[simd_width](num_blocks, _copy)

    # Scalar remainder for tail elements.
    for i in range(scaled + tid, row_length, threads_per_row):
        output_ptr[i] = input_ptr[i]


@__name(t"padded_copy_{dtype}_w{simd_width}")
def padded_copy_kernel[
    InputLayoutType: TensorLayout,
    input_origin: ImmOrigin,
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    dtype: DType,
    simd_width: Int,
    InputEngine: TensorEngine,
    OutputEngine: TensorEngine,
](
    input_tensor: TileTensor[
        dtype, InputLayoutType, input_origin, Engine=InputEngine
    ],
    output_tensor: TileTensor[
        dtype, OutputLayoutType, output_origin, Engine=OutputEngine
    ],
    rows_per_sm: Int32,
    total_rows: Int32,
    row_length: Int32,
):
    var _rows_per_sm = Int(rows_per_sm)
    var _total_rows = Int(total_rows)
    var _row_length = Int(row_length)
    var start_row = block_idx.x * _rows_per_sm
    var threads_per_row = block_dim.x

    var rows_per_iter = block_dim.y
    var end_row = min(start_row + _rows_per_sm, _total_rows)

    start_row += thread_idx.y

    for row in range(start_row, end_row, rows_per_iter):
        var coord = input_tensor.layout.idx2crd(row * _row_length)
        var output_offset = Int(output_tensor.layout(coord))
        var input_offset = row * _row_length

        var output_ptr = output_tensor.ptr + output_offset
        var input_ptr = input_tensor.ptr + input_offset

        _vectorized_copy_row[dtype, simd_width](
            input_ptr,
            output_ptr,
            _row_length,
            threads_per_row,
        )


def _pad_constant_impl[
    dtype: DType,
    simd_width: Int = 4,
    max_threads: Int = 256,
    threads_per_row: Int = 16,
](
    input_tensor: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    output_tensor: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    # Zero-element input (e.g. a ``(B, C, 0, 0)`` tensor padded out to
    # ``(B, C, 1, 1)`` in a diffusion VAE encoder for the text-to-image
    # placeholder): nothing to copy.  The caller -- ``pad_constant`` --
    # has already filled the output buffer with the constant value via
    # ``enqueue_fill`` before reaching this function, which is exactly
    # the correct output (every position is "padded" because there's no
    # input region to copy from).  Without this guard the kernel would
    # divide by zero computing ``total_rows = num_elements // row_length``
    # and launch with ``grid_dim=(0)`` which is undefined.
    if input_tensor.num_elements() == 0:
        return

    var row_length = Int(input_tensor.dim(input_tensor.rank - 1))
    var total_rows = input_tensor.num_elements() // row_length

    comptime assert threads_per_row > 0 and max_threads % threads_per_row == 0

    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)

    var linear_block_count: Int

    var rows_per_block: Int

    if sm_count > total_rows:
        linear_block_count = total_rows
        rows_per_block = 1
    else:
        linear_block_count = sm_count
        rows_per_block = ceildiv(total_rows, sm_count)

    comptime block_rows = max_threads // threads_per_row
    comptime kernel = padded_copy_kernel[
        input_origin=ImmOrigin(input_tensor.origin),
        InputLayoutType=input_tensor.LayoutType,
        output_origin=output_tensor.origin,
        OutputLayoutType=output_tensor.LayoutType,
        dtype=dtype,
        simd_width=simd_width,
        InputEngine=type_of(input_tensor.as_immut()).Engine,
        OutputEngine=output_tensor.Engine,
    ]

    ctx.enqueue_function[kernel](
        input_tensor.as_immut(),
        output_tensor,
        Int32(rows_per_block),
        Int32(total_rows),
        Int32(row_length),
        grid_dim=(linear_block_count),
        block_dim=(threads_per_row, block_rows),
    )


def pad_constant[
    rank: Int, dtype: DType, padding_type: DType
](
    output: UnsafePointer[mut=True, Scalar[dtype], _],
    output_shape: IndexList[rank],
    input: UnsafePointer[mut=False, Scalar[dtype], _],
    input_shape: IndexList[rank],
    paddings: UnsafePointer[mut=False, Scalar[padding_type], _],
    constant: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    """
    Fill `output` with values from `input`, and edges padded with `constant`
    based on `paddings`.

    Args:
        output: The output buffer.
        output_shape: The output shape.
        input: The input buffer.
        input_shape: The input shape.
        paddings: Ordered (before, after) padding sizes for each axis.
        constant: The constant to pad output with.
        ctx: Device context for participating GPU.

    Example:
        ```mojo
        var input_shape = (X, Y, Z)
        var paddings = [x0, x1, y0, y1, z0, z1]

        out[x, y, z] =
          input[x - x0, y - y0, z - z0] if x ∈ [x0, x0 + X] &&
                                           y ∈ [y0, y0 + Y] &&
                                           z ∈ [z0, z0 + Z]
          else constant
        ```
    """

    var input_strides = Array[Int, rank]()
    var output_strides = Array[Int, rank]()

    var output_size: Int = 1

    comptime for i in range(rank):
        output_size *= output_shape[i]

    var output_buffer = DeviceBuffer[dtype](
        ctx, output, output_size, owning=False
    )
    output_buffer.enqueue_fill(constant)

    _fill_strides_indexlist[rank](input_shape, input_strides)
    _fill_strides_indexlist[rank](output_shape, output_strides)

    var input_tensor = TileTensor(
        input,
        Layout(Coord(input_shape), Coord(input_strides)),
    )

    var pre_pad_offset = 0

    for i in range(rank):
        pre_pad_offset += Int(paddings[2 * i]) * output_strides[i]

    var adjusted_output_ptr = output + pre_pad_offset

    var output_tensor = TileTensor(
        adjusted_output_ptr,
        Layout(Coord(input_shape), Coord(output_strides)),
    )

    _pad_constant_impl[dtype](
        input_tensor,
        output_tensor,
        ctx,
    )


def get_padding_output_shape[
    rank: Int
](
    input_shape: IndexList[rank],
    paddings: TileTensor[mut=False, .int, ...],
) -> IndexList[rank]:
    """Computes the output shape produced by padding `input_shape` with the
    before/after amounts given in `paddings`.

    `paddings` is a flat one-dimensional tensor of length `2 * rank` ordered
    as `(before_axis0, after_axis0, before_axis1, after_axis1, ...)`, and the
    returned shape has each axis `i` set to `before[i] + input_shape[i] +
    after[i]`.

    Args:
        input_shape: Shape of the tensor before padding.
        paddings: Flat `(before, after)` padding sizes for each axis.

    Returns:
        The padded output shape with the same rank as `input_shape`.
    """
    comptime assert (
        paddings.flat_rank == 1 and paddings.static_shape[0] == 2 * rank
    )
    var output_shape = IndexList[rank]()
    for i in range(rank):
        var before = paddings[2 * i]
        var after = paddings[2 * i + 1]
        output_shape[i] = Int(before) + input_shape[i] + Int(after)
    return output_shape
