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
"""Implements spatial merge, which compresses vision token grids by merging spatial blocks before attention."""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from layout import (
    Coord,
    Idx,
    TensorLayout,
    TensorEngine,
    TileTensor,
    row_major,
)
from layout.tile_layout import Layout
from std.utils.index import IndexList


@__name(t"spatial_merge_{dtype}")
def spatial_merge_kernel[
    dtype: DType,
    InputLayoutType: TensorLayout,
    input_origin: ImmOrigin,
    InputEngine: TensorEngine,
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    OutputEngine: TensorEngine,
    GridThwLayoutType: TensorLayout,
    grid_thw_origin: ImmOrigin,
    GridThwEngine: TensorEngine,
](
    output: TileTensor[
        dtype, OutputLayoutType, output_origin, Engine=OutputEngine
    ],
    input: TileTensor[dtype, InputLayoutType, input_origin, Engine=InputEngine],
    grid_thw: TileTensor[
        .int64, GridThwLayoutType, grid_thw_origin, Engine=GridThwEngine
    ],
    batch_size: Int32,
    hidden_size: Int32,
    merge_size: Int32,
):
    """
    Spatial merge kernel.

    Grid: 1D over all output patches (one block per output patch).
    Threads: loop over channels (hidden_size x merge_size^2).

    Parameters:
        dtype: Element type of the input and output tensors.
        InputLayoutType: Compile-time `TensorLayout` of the input tensor.
        input_origin: Immutable origin of the input tensor.
        InputEngine: Engine of the input tensor.
        OutputLayoutType: Compile-time `TensorLayout` of the output tensor.
        output_origin: Mutable origin of the output tensor.
        OutputEngine: Engine of the output tensor.
        GridThwLayoutType: Compile-time `TensorLayout` of the `grid_thw`
            tensor.
        grid_thw_origin: Immutable origin of the `grid_thw` tensor.
        GridThwEngine: Engine of the `grid_thw` tensor.

    Args:
        output: Output tensor.
        input: Input tensor.
        grid_thw: Grid dimensions tensor (B, 3) containing [t, h, w] for each item.
        batch_size: Number of items in batch.
        hidden_size: Hidden dimension size.
        merge_size: Size of spatial merge blocks.
    """
    var _batch_size = Int(batch_size)
    var _hidden_size = Int(hidden_size)
    var _merge_size = Int(merge_size)
    comptime assert grid_thw.flat_rank == 2
    # The `.ptr` arithmetic below addresses scalars, so the kernel only
    # supports scalar-element tiles.
    comptime assert input.element_size == 1
    comptime assert output.element_size == 1
    comptime assert grid_thw.element_size == 1

    # Global patch index.
    var patch_idx = block_idx.x

    var offset_in: Int64 = 0
    var offset_out: Int64 = 0

    # Compute input/output offsets on-the-fly by scanning grid_thw.
    # Simultaneously find which batch item this patch belongs to.
    var b = 0
    var found = False
    for i in range(_batch_size):
        var t = grid_thw[i, 0]
        var h = grid_thw[i, 1]
        var w = grid_thw[i, 2]
        var h_out = h // Int64(_merge_size)
        var w_out = w // Int64(_merge_size)
        var num_output_patches = t * h_out * w_out

        # Check if patch_idx falls in this batch item.
        if patch_idx < Int(offset_out + num_output_patches):
            b = i
            found = True
            break

        # Accumulate offsets.
        offset_in += rebind[Int64](h * w)
        offset_out += rebind[Int64](num_output_patches)

    # Skip blocks whose patch index is past the last output patch.
    if not found:
        return

    # Local patch index (i.e., within this batch item).
    var patch_local_idx = patch_idx - Int(offset_out)

    # Get dimensions for this batch item from grid_thw.
    var T = grid_thw[b, 0]
    var H = grid_thw[b, 1]
    var W = grid_thw[b, 2]
    var H_out = H // Int64(_merge_size)
    var W_out = W // Int64(_merge_size)
    var C_out = _hidden_size * _merge_size * _merge_size

    # Create a RuntimeLayout for the patch space [T, H_out, W_out]
    # to convert linear patch_local_idx to (t, ho, wo) coordinates.
    var patch_space_rt_layout = row_major(T, H_out, W_out)

    # Convert linear patch index to 3D coordinates (t, ho, wo).
    var patch_coords = patch_space_rt_layout.idx2crd(Int(patch_local_idx))
    var t, ho, wo = (
        patch_coords[0].value(),
        patch_coords[1].value(),
        patch_coords[2].value(),
    )

    # Create a tiled layout for input representing
    # [H_out, _merge_size, W_out, _merge_size, _hidden_size].
    # This allows us to index the input as a 5D tiled tensor.
    # Physical memory: [H, W, _hidden_size] row-major.
    # Logical view: [H_out, _merge_size, W_out, _merge_size, _hidden_size].
    var input_tiled_shape = IndexList[5](
        Int(H_out), _merge_size, Int(W_out), _merge_size, _hidden_size
    )
    var input_tiled_stride = IndexList[5](
        _merge_size
        * Int(W)
        * _hidden_size,  # stride for H_out: skip _merge_size full rows.
        Int(W) * _hidden_size,  # stride for dh: move one row within block.
        _merge_size
        * _hidden_size,  # stride for W_out: skip _merge_size columns.
        _hidden_size,  # stride for dw: move one column within block.
        1,  # stride for c: move one channel.
    )

    var input_tiled_layout = Layout(
        Coord(input_tiled_shape),
        Coord(input_tiled_stride),
    )

    var input_tensor = TileTensor(
        input.ptr + Int(offset_in * Int64(_hidden_size)),
        input_tiled_layout,
    )

    # Create TileTensor for output: [T, H_out, W_out, C_out].
    # Note: in reality we want 2D flattened to [T * H_out * W_out, C_out], but
    # we use 4D for semantic clarity - internally in memory it is handled correctly.
    var output_runtime_layout = row_major((T, H_out, W_out, C_out))
    var output_tensor = TileTensor(
        output.ptr + Int(offset_out * Int64(C_out)),
        output_runtime_layout,
    )

    # Create layout for the merged channel dimension structure.
    # C_out represents [_merge_size, _merge_size, _hidden_size] flattened row-major.
    var channel_layout = row_major((_merge_size, _merge_size, _hidden_size))

    # Copy patch - threads loop over output channels.
    # Each c_out in [0, C_out) corresponds to [_merge_size, _merge_size, _hidden_size]
    # flattened in the permute(0, 1, 3, 2, 4, 5) order.
    for c_out in range(thread_idx.x, C_out, block_dim.x):
        # Decompose c_out into (dh, dw, c) using the channel layout.
        var channel_coords = channel_layout.idx2crd(c_out)
        var dh, dw, c = (
            channel_coords[0].value(),
            channel_coords[1].value(),
            channel_coords[2].value(),
        )
        output_tensor[Coord(t, ho, wo, c_out)] = input_tensor[
            Coord(ho, dh, wo, dw, c)
        ]


def spatial_merge[
    dtype: DType,
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    input: TileTensor[dtype, address_space=.GENERIC, ...],
    grid_thw: TileTensor[.int64, address_space=.GENERIC, ...],
    hidden_size: Int,
    merge_size: Int,
    ctx: DeviceContext,
) raises:
    """
    Launches the spatial merge kernel that compresses vision token grids by merging spatial blocks before attention.

    Parameters:
        dtype: Element type of the input and output tensors.

    Args:
        output: Output tensor holding the merged patches.
        input: Input tensor holding the original patch grid.
        grid_thw: Grid dimensions tensor of shape `(batch_size, 3)` with `[t, h, w]` per item.
        hidden_size: Hidden dimension size of each patch.
        merge_size: Size of the spatial merge blocks.
        ctx: Device context used to enqueue the kernel.
    """
    comptime threads_per_block = 256
    var batch_size = Int(grid_thw.dim[0]())
    # One block per merged output patch: each block writes
    # merge_size * merge_size * hidden_size elements, so the block count is the
    # output element count divided by that per-patch size.
    var num_blocks = Int(output.dim[0]() * output.dim[1]()) // (
        merge_size * merge_size * hidden_size
    )

    comptime kernel = spatial_merge_kernel[
        dtype,
        input.LayoutType,
        ImmOrigin(input.origin),
        input.Engine,
        output.LayoutType,
        output.origin,
        output.Engine,
        grid_thw.LayoutType,
        ImmOrigin(grid_thw.origin),
        grid_thw.Engine,
    ]

    ctx.enqueue_function[kernel](
        output,
        input.as_immut(),
        grid_thw.as_immut(),
        Int32(batch_size),
        Int32(hidden_size),
        Int32(merge_size),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(threads_per_block, 1, 1),
    )
