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
"""Test tpool_patch_merger kernel against CPU reference behavior.

Python reference: tpool_path_merger.py
  - x: (total_tokens, D), grid_thws: (n_videos, 3) [T,H,W], merge (kH,kW)
  - Output: contiguous tensor (total_output_patches, D) with temporal mean.
This test implements the same logic in Mojo (CPU reference) and compares
GPU kernel output to it.
"""


from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from nn.tpool_patch_merger import (
    tpool_patch_merger,
)
from std.random import rand, seed
from std.testing import assert_almost_equal


def cpu_reference_one_video[
    dtype: DType,
](
    x: Pointer[Scalar[dtype], _],
    in_offset: Int,
    out_base: MutPointer[Scalar[dtype], _],
    t: Int,
    h: Int,
    w: Int,
    kH: Int,
    kW: Int,
    D: Int,
) -> None:
    """CPU reference for one video: same math as Python tpool_patch_merger."""
    var new_H = h // kH
    var new_W = w // kW
    var n_pat = new_H * new_W * kH * kW
    for pat_idx in range(n_pat):
        var sp_idx = pat_idx // (kH * kW)
        var ker_idx = pat_idx % (kH * kW)
        var nh = sp_idx // new_W
        var nw = sp_idx % new_W
        var ph = ker_idx // kW
        var pw = ker_idx % kW
        var h_src = nh * kH + ph
        var w_src = nw * kW + pw
        var spatial_flat = h_src * w + w_src
        for d in range(D):
            var acc = Scalar[dtype](0)
            for ti in range(t):
                var row = in_offset + ti * (h * w) + spatial_flat
                acc += x[row * D + d]
            acc = acc / Scalar[dtype](t)
            out_base.store(pat_idx * D + d, acc)


def test_tpool_patch_merger(ctx: DeviceContext) raises:
    """Compare GPU kernel output to CPU reference (same math as Python)."""
    comptime dtype = DType.bfloat16
    comptime kH = 2
    comptime kW = 2
    comptime D = 8

    # Two videos: (T=2, H=4, W=4) and (T=3, H=6, W=6)
    var t0: Int = 2
    var h0: Int = 4
    var w0: Int = 4
    var t1: Int = 3
    var h1: Int = 6
    var w1: Int = 6
    var max_h = max(h0, h1)
    var max_w = max(w0, w1)
    var len0 = t0 * h0 * w0
    var len1 = t1 * h1 * w1
    var total_in = len0 + len1
    var n_videos: Int = 2

    # Each video outputs H*W rows (temporal dim averaged out).
    var out0_rows = h0 * w0
    var out1_rows = h1 * w1
    var total_out = out0_rows + out1_rows

    # Host buffers for input and grid_thws
    var x_host = ctx.enqueue_create_host_buffer[dtype](total_in * D)
    var bounds_host = ctx.enqueue_create_host_buffer[.int64](n_videos * 3)
    ctx.synchronize()

    seed(42)
    rand[dtype](x_host.as_span(), min=0.0, max=1.0)

    bounds_host[0] = Int64(t0)
    bounds_host[1] = Int64(h0)
    bounds_host[2] = Int64(w0)
    bounds_host[3] = Int64(t1)
    bounds_host[4] = Int64(h1)
    bounds_host[5] = Int64(w1)

    # Device buffers
    var x_dev = ctx.enqueue_create_buffer[dtype](total_in * D)
    var out_dev = ctx.enqueue_create_buffer[dtype](total_out * D)
    var bounds = ctx.enqueue_create_buffer[.int64](n_videos * 3)
    ctx.enqueue_copy(x_dev, x_host)
    ctx.enqueue_copy(bounds, bounds_host)
    ctx.enqueue_memset(out_dev, 0)
    ctx.synchronize()

    # CPU reference: write to host buffers
    var ref_host = ctx.enqueue_create_host_buffer[dtype](total_out * D)

    cpu_reference_one_video[dtype](
        x_host.as_span().unsafe_ptr(),
        0,
        ref_host.as_span().unsafe_ptr(),
        t0,
        h0,
        w0,
        kH,
        kW,
        D,
    )
    cpu_reference_one_video[dtype](
        x_host.as_span().unsafe_ptr(),
        len0,
        ref_host.as_span().unsafe_ptr().unsafe_offset(out0_rows * D),
        t1,
        h1,
        w1,
        kH,
        kW,
        D,
    )

    # GPU kernel: contiguous output
    var x_tile = TileTensor[mut=False, Engine=_](
        x_dev,
        row_major(Coord(total_in, D)),
    )
    var out_tile = TileTensor(
        out_dev,
        row_major(Coord(total_out, D)),
    )
    var bounds_tensor = TileTensor[mut=False, Engine=_](
        bounds,
        row_major(Coord(n_videos, Idx[3])),
    )

    tpool_patch_merger(
        out_tile,
        x_tile,
        bounds_tensor,
        kH,
        kW,
        max_h,
        max_w,
        ctx,
    )

    ctx.synchronize()

    # Copy GPU output back
    var out_host = ctx.enqueue_create_host_buffer[dtype](total_out * D)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    # Compare to CPU reference; relax for bfloat16 precision
    var atol = 1e-2
    for i in range(total_out * D):
        assert_almost_equal(out_host[i], ref_host[i], atol=atol)

    print("test_tpool_patch_merger: PASSED")


def main() raises:
    with DeviceContext() as ctx:
        test_tpool_patch_merger(ctx)
