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

"""Learnable 2D interpolated position embedding (Kimi K2.5 MoonViT3d).

Equivalent to ``Learnable2DInterpPosEmbDivided_fixed.forward()`` from
``nvidia/Kimi-K2.5-NVFP4/modeling_kimi_k25.py``.

For each video described by ``grid_thws``:
  1. Bicubic-interpolates the learnable 2D weight grid from (H, W) to (h, w).
     When ``(h, w) == (H, W)`` the grid is used directly.
  2. If ``t > 1`` adds a 1D sincos temporal embedding per frame.
  3. Adds the result element-wise to ``x``.

Tensor layout (all row-major):
  - ``x``:           (L, dim)           patch embeddings
  - ``weight``:      (H, W, dim)        learnable 2D grid
  - ``grid_thws``:   (N, 3)   int64     per-video (t, h, w)
  - ``time_weight``: (num_frames, dim)  float32  1D sincos temporal embedding
  - ``output``:      (L, dim)           x + interpolated position embedding
"""

from std.math import clamp, floor

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from layout import TensorLayout, TensorEngine, TileTensor


# ---------------------------------------------------------------------------
# Bicubic interpolation helper
# ---------------------------------------------------------------------------


@always_inline
def _cubic_weight(x: Float32) -> Float32:
    """Catmull-Rom cubic weight (a = -0.75), matching PyTorch F.interpolate."""
    var a: Float32 = -0.75
    var ax = abs(x)
    var ax2 = ax * ax
    var ax3 = ax2 * ax
    if ax <= 1:
        return (a + 2) * ax3 - (a + 3) * ax2 + 1
    elif ax < 2:
        return a * ax3 - 5 * a * ax2 + 8 * a * ax - 4 * a
    else:
        return 0


# ---------------------------------------------------------------------------
# GPU kernel
# ---------------------------------------------------------------------------


@__name(t"learnable_2d_interp_pos_emb_{dtype}")
def _gpu_kernel[
    dtype: DType,
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    OutputEngine: TensorEngine,
    XLayoutType: TensorLayout,
    x_origin: ImmOrigin,
    XEngine: TensorEngine,
    WeightLayoutType: TensorLayout,
    weight_origin: ImmOrigin,
    WeightEngine: TensorEngine,
    GridLayoutType: TensorLayout,
    grid_origin: ImmOrigin,
    GridEngine: TensorEngine,
    TimeLayoutType: TensorLayout,
    time_origin: ImmOrigin,
    TimeEngine: TensorEngine,
](
    output: TileTensor[
        dtype, OutputLayoutType, output_origin, Engine=OutputEngine
    ],
    x: TileTensor[dtype, XLayoutType, x_origin, Engine=XEngine],
    weight: TileTensor[
        dtype, WeightLayoutType, weight_origin, Engine=WeightEngine
    ],
    grid_thws: TileTensor[
        .int64, GridLayoutType, grid_origin, Engine=GridEngine
    ],
    time_weight: TileTensor[
        .float32, TimeLayoutType, time_origin, Engine=TimeEngine
    ],
    N: Int32,
    dim: Int32,
    H: Int32,
    W: Int32,
):
    """GPU kernel: one block per output position, threads stride over dim.

    output[l, d] = x[l, d] + bicubic(weight, hh, ww)[d]
                   + (time_weight[tt, d] if t > 1 else 0)

    Bicubic uses half-pixel coords and Catmull-Rom kernel (a = -0.75),
    or falls back to direct lookup when (h, w) == (H, W).

    Args:
        output: (L, dim) output tensor on device.
        x: (L, dim) input patch embeddings on device.
        weight: (H, W, dim) learnable 2D grid on device.
        grid_thws: (N, 3) per-video (t, h, w) on device.
        time_weight: (num_frames, dim) sincos temporal embedding on device.
        N: Number of videos.
        dim: Embedding dimension.
        H: Height of weight grid.
        W: Width of weight grid.
    """
    comptime assert output.flat_rank == 2
    var _N = Int(N)
    var _dim = Int(dim)
    var _H = Int(H)
    var _W = Int(W)
    comptime assert x.flat_rank == 2
    comptime assert weight.flat_rank == 3
    comptime assert grid_thws.flat_rank == 2
    comptime assert time_weight.flat_rank == 2
    comptime assert output.element_size == 1
    comptime assert x.element_size == 1
    comptime assert weight.element_size == 1
    comptime assert grid_thws.element_size == 1
    comptime assert time_weight.element_size == 1

    var pos_idx = block_idx.x

    # Scan grid_thws to find which video this position belongs to.
    var offset: Int = 0
    var t: Int = 0
    var h: Int = 0
    var w: Int = 0
    for img in range(_N):
        t = Int(grid_thws[img, 0])
        h = Int(grid_thws[img, 1])
        w = Int(grid_thws[img, 2])
        var img_len = t * h * w
        if pos_idx < offset + img_len:
            break
        offset += img_len

    var local_pos = pos_idx - offset
    var tt, hw_pos = divmod(local_pos, h * w)
    var hh, ww = divmod(hw_pos, w)

    var no_interp = h == _H and w == _W
    var scale_h = Float32(_H) / Float32(h)
    var scale_w = Float32(_W) / Float32(w)

    # Precompute bicubic mapping for this spatial position.
    var ih_floor: Int = 0
    var iw_floor: Int = 0
    var dy: Float32 = 0
    var dx: Float32 = 0
    if not no_interp:
        var in_h = (Float32(hh) + 0.5) * scale_h - 0.5
        var in_w = (Float32(ww) + 0.5) * scale_w - 0.5
        ih_floor = Int(floor(in_h))
        iw_floor = Int(floor(in_w))
        dy = in_h - Float32(ih_floor)
        dx = in_w - Float32(iw_floor)

    # Threads stride over _dim channels.
    for d in range(thread_idx.x, _dim, block_dim.x):
        var pos_val: Float32
        if no_interp:
            pos_val = Float32(weight[hh, ww, d][0])
        else:
            var val: Float32 = 0
            comptime for i in range(4):
                var yp = clamp(ih_floor + i - 1, 0, _H - 1)
                var wy = _cubic_weight(Float32(i) - 1.0 - dy)
                comptime for j in range(4):
                    var xp = clamp(iw_floor + j - 1, 0, _W - 1)
                    var wx = _cubic_weight(Float32(j) - 1.0 - dx)
                    val += Float32(weight[yp, xp, d][0]) * wy * wx
            pos_val = val

        var time_val: Float32 = 0
        if t > 1:
            time_val = time_weight[tt, d][0]

        output[pos_idx, d] = Scalar[dtype](
            Float32(x[pos_idx, d][0]) + pos_val + time_val
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def learnable_2d_interp_pos_emb[
    dtype: DType,
](
    output: TileTensor[mut=True, dtype, ...],
    x: TileTensor[dtype, ...],
    weight: TileTensor[dtype, ...],
    grid_thws: TileTensor[.int64, ...],
    time_weight: TileTensor[.float32, ...],
    ctx: DeviceContext,
) raises:
    """Applies learnable 2D interpolated position embedding on GPU.

    For each video described by ``grid_thws``, bicubic-interpolates ``weight``
    from (_H, _W) to (h, w), optionally adds temporal sincos embedding, and
    adds the result element-wise to ``x``.

    Parameters:
        dtype: Element type of ``x``, ``weight``, and ``output``.

    Args:
        output: (L, dim) output tensor.
        x: (L, dim) input patch embeddings.
        weight: (H, W, dim) learnable 2D grid.
        grid_thws: (N, 3) per-video (t, h, w), int64.
        time_weight: (num_frames, dim) 1D sincos temporal embedding, float32.
        ctx: Device context for GPU dispatch.
    """
    comptime assert output.flat_rank == 2
    comptime assert x.flat_rank == 2
    comptime assert weight.flat_rank == 3
    comptime assert grid_thws.flat_rank == 2
    comptime assert time_weight.flat_rank == 2

    var L = Int(x.dim[0]())
    var dim = Int(x.dim[1]())
    var H = Int(weight.dim[0]())
    var W = Int(weight.dim[1]())
    var N = Int(grid_thws.dim[0]())

    comptime BLOCK_SIZE = 256

    comptime kernel = _gpu_kernel[
        dtype,
        output.LayoutType,
        output.origin,
        output.Engine,
        x.LayoutType,
        ImmOrigin(x.origin),
        x.Engine,
        weight.LayoutType,
        ImmOrigin(weight.origin),
        weight.Engine,
        grid_thws.LayoutType,
        ImmOrigin(grid_thws.origin),
        grid_thws.Engine,
        time_weight.LayoutType,
        ImmOrigin(time_weight.origin),
        time_weight.Engine,
    ]
    ctx.enqueue_function[kernel](
        output,
        x.as_immut(),
        weight.as_immut(),
        grid_thws.as_immut(),
        time_weight.as_immut(),
        Int32(N),
        Int32(dim),
        Int32(H),
        Int32(W),
        grid_dim=(L,),
        block_dim=(BLOCK_SIZE,),
    )
