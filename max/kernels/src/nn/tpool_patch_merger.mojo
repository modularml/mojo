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
"""Implements the temporal-pool patch-merger kernel that reduces vision token sequences via average pooling."""

from std.math import ceildiv, divmod
from std.sys.info import simd_width_of

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from layout import (
    Coord,
    Idx,
    DefaultEngine,
    TensorLayout,
    TensorEngine,
    TileTensor,
)


# ------------------------------------------------------------------------------
# GPU kernel
# ------------------------------------------------------------------------------


@__name(t"tpool_patch_merger_{dtype}_w{vec_width}")
def tpool_patch_merger_kernel[
    dtype: DType,
    XLayout: TensorLayout,
    x_origin: ImmOrigin,
    XEngine: TensorEngine,
    OutLayout: TensorLayout,
    out_origin: MutOrigin,
    OutEngine: TensorEngine,
    GridThwLayout: TensorLayout,
    grid_thw_origin: ImmOrigin,
    GridThwEngine: TensorEngine,
    vec_width: Int,
    num_threads: Int,
](
    x_tile: TileTensor[dtype, XLayout, x_origin, Engine=XEngine],
    out_tile: TileTensor[dtype, OutLayout, out_origin, Engine=OutEngine],
    grid_thws: TileTensor[
        .int64, GridThwLayout, grid_thw_origin, Engine=GridThwEngine
    ],
    kH: Int32,
    kW: Int32,
    D: Int32,
    n_vids: Int32,
):
    """Temporal pooling patch merger kernel.

    Averages x across the temporal dimension for each video, rearranging
    spatially according to the (kH, kW) merge kernel. Each video's output
    occupies H_i * W_i contiguous rows in the flat output tensor.

    Grid mapping:
        block_idx.z  = video index
        block_idx.y  = patch index within the video (max_pat upper bound)
        block_idx.x  = tile index along D
        thread_idx.x = lane within D tile

    Parameters:
        dtype: Element type of the input and output tensors.
        XLayout: Memory layout of the input tensor `x_tile`.
        x_origin: Immutable origin of the input tensor `x_tile`.
        XEngine: Engine of the input tensor `x_tile`.
        OutLayout: Memory layout of the output tensor `out_tile`.
        out_origin: Mutable origin of the output tensor `out_tile`.
        OutEngine: Engine of the output tensor `out_tile`.
        GridThwLayout: Memory layout of the grid dimensions tensor `grid_thws`.
        grid_thw_origin: Immutable origin of the grid dimensions tensor
            `grid_thws`.
        GridThwEngine: Engine of the grid dimensions tensor
            `grid_thws`.
        vec_width: SIMD vector width for loads and stores along the hidden
            dimension.
        num_threads: Number of threads per block.

    Args:
        x_tile: Input tensor [n_tokens, D].
        out_tile: Contiguous output tensor [total_output_patches, D].
        grid_thws: Grid dimensions tensor [n_vids, 3] with (T, H, W) per video.
        kH: Merge kernel height.
        kW: Merge kernel width.
        D: Hidden dimension.
        n_vids: Number of videos.
    """
    comptime assert x_tile.flat_rank == 2, "x_tile must be rank 2"
    comptime assert out_tile.flat_rank == 2, "out_tile must be rank 2"
    var _kH = Int(kH)
    var _kW = Int(kW)
    var _D = Int(D)
    var _n_vids = Int(n_vids)
    comptime assert grid_thws.flat_rank == 2, "grid_thws must be rank 2"
    # Provide evidence that flat_rank >= 2 for the Coord(..., ...) accesses below.
    comptime assert grid_thws.flat_rank >= 2
    comptime assert x_tile.flat_rank >= 2
    comptime assert out_tile.flat_rank >= 2
    comptime assert x_tile.element_size == 1
    comptime assert out_tile.element_size == 1
    comptime assert grid_thws.element_size == 1

    var vid = block_idx.z
    var pat_idx = block_idx.y
    var d_tile = block_idx.x
    var tid = thread_idx.x

    if vid >= _n_vids:
        return

    var t = Int(grid_thws[Coord(vid, Idx[0])])
    var h = Int(grid_thws[Coord(vid, Idx[1])])
    var w = Int(grid_thws[Coord(vid, Idx[2])])

    var new_H = h // _kH
    var new_W = w // _kW
    var n_patches = new_H * new_W
    var n_kernel = _kH * _kW
    var n_pat_total = n_patches * n_kernel

    if pat_idx >= n_pat_total:
        return

    # Scan grid_thws to compute input and output offsets for this video.
    var in_offset: Int = 0
    var out_offset: Int = 0
    for i in range(vid):
        var ti = Int(grid_thws[Coord(i, Idx[0])])
        var hi = Int(grid_thws[Coord(i, Idx[1])])
        var wi = Int(grid_thws[Coord(i, Idx[2])])
        in_offset += ti * hi * wi
        out_offset += hi * wi

    var sp_idx, ker_idx = divmod(pat_idx, n_kernel)
    var nh, nw = divmod(sp_idx, new_W)
    var ph, pw = divmod(ker_idx, _kW)
    var h_src = nh * _kH + ph
    var w_src = nw * _kW + pw
    var spatial_flat = h_src * w + w_src

    comptime if vec_width == 1:
        var d = d_tile * num_threads + tid
        if d >= _D:
            return
        var acc = Scalar[dtype](0)
        for t_i in range(t):
            var row = in_offset + t_i * (h * w) + spatial_flat
            acc += x_tile[Coord(row, d)][0]
        acc /= Scalar[dtype](t)
        out_tile.store(Coord(out_offset + pat_idx, d), acc)
    else:
        var d_start = (d_tile * num_threads + tid) * vec_width
        if d_start >= _D:
            return
        var acc = SIMD[dtype, vec_width](0)
        for t_i in range(t):
            var row = in_offset + t_i * (h * w) + spatial_flat
            acc += x_tile.load[width=vec_width](Coord(row, d_start))
        acc /= Scalar[dtype](t)
        out_tile.store[width=vec_width](
            Coord(out_offset + pat_idx, d_start), acc
        )


# ------------------------------------------------------------------------------
# Host launch (enqueue)
# ------------------------------------------------------------------------------


def tpool_patch_merger[
    dtype: DType,
    output_layout: TensorLayout,
    x_layout: TensorLayout,
    bounds_layout: TensorLayout,
    OutputEngine: TensorEngine = DefaultEngine[element_width=1],
    XEngine: TensorEngine = DefaultEngine[element_width=1],
    BoundsEngine: TensorEngine = DefaultEngine[element_width=1],
](
    output: TileTensor[dtype, output_layout, MutAnyOrigin, Engine=OutputEngine],
    x: TileTensor[dtype, x_layout, ImmutAnyOrigin, Engine=XEngine],
    bounds: TileTensor[
        .int64, bounds_layout, ImmutAnyOrigin, Engine=BoundsEngine
    ],
    kH: Int,
    kW: Int,
    max_h: Int,
    max_w: Int,
    ctx: DeviceContext,
) raises:
    """Temporal pooling patch merger entry point.

    Parameters:
        dtype: Element type of the input and output tensors.
        output_layout: Memory layout of the output tensor.
        x_layout: Memory layout of the input tensor.
        bounds_layout: Memory layout of the bounds tensor.
        OutputEngine: Engine of the output tensor.
        XEngine: Engine of the input tensor.
        BoundsEngine: Engine of the bounds tensor.

    Args:
        output: Contiguous output tensor [total_output_patches, D].
        x: Input tensor [n_tokens, D].
        bounds: Grid dimensions tensor [n_vids, 3] with (T, H, W) per video.
        kH: Merge kernel height.
        kW: Merge kernel width.
        max_h: Maximum H across all videos (for grid sizing).
        max_w: Maximum W across all videos (for grid sizing).
        ctx: Device context.
    """
    var D = Int(x.dim[1]())
    var n_vids = Int(bounds.dim[0]())
    var max_pat = max_h // kH * max_w // kW * kH * kW

    comptime simd_width = simd_width_of[
        dtype, target=ctx.default_device_info.target()
    ]()
    comptime num_threads = 256

    if D % simd_width == 0:
        var grid_x = ceildiv(D, num_threads * simd_width)
        comptime kernel = tpool_patch_merger_kernel[
            dtype,
            x.LayoutType,
            ImmOrigin(x.origin),
            x.Engine,
            output.LayoutType,
            output.origin,
            output.Engine,
            bounds.LayoutType,
            ImmOrigin(bounds.origin),
            bounds.Engine,
            simd_width,
            num_threads,
        ]
        ctx.enqueue_function[kernel](
            x.as_immut(),
            output,
            bounds.as_immut(),
            Int32(kH),
            Int32(kW),
            Int32(D),
            Int32(n_vids),
            grid_dim=(grid_x, max_pat, n_vids),
            block_dim=(num_threads, 1, 1),
        )
    else:
        var grid_x = ceildiv(D, num_threads * 1)
        comptime kernel = tpool_patch_merger_kernel[
            dtype,
            x.LayoutType,
            ImmOrigin(x.origin),
            x.Engine,
            output.LayoutType,
            output.origin,
            output.Engine,
            bounds.LayoutType,
            ImmOrigin(bounds.origin),
            bounds.Engine,
            1,
            num_threads,
        ]
        ctx.enqueue_function[kernel](
            x.as_immut(),
            output,
            bounds.as_immut(),
            Int32(kH),
            Int32(kW),
            Int32(D),
            Int32(n_vids),
            grid_dim=(grid_x, max_pat, n_vids),
            block_dim=(num_threads, 1, 1),
        )
