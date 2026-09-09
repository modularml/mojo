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
"""Runtime-HW correctness for amd_4wave_conv.

Builds TileTensors with **dynamic NHWC layout** (the shape graph
compilers produce for symbolic image resolution, e.g. FLUX VAE),
invokes the runtime-HW path (`use_runtime_hw=True`), and cross-checks
against the comptime-HW path on the same input. Both paths must
produce byte-identical output.

Covers BF16 (non-swizzle path with single up-front divmod, carry-
increment per iter) and FP8 (swizzle path with per-iter divmod).
"""

from max.gpu.host import DeviceContext
from std.random import rand
from std.utils import IndexList

from layout import Coord, TileTensor, row_major

from nn.conv.gpu.amd.amd_4wave_conv import amd_4wave_conv


def _round_up(x: Int, mod: Int) -> Int:
    return ((x + mod - 1) // mod) * mod


def _run[
    a_type: DType,
    c_type: DType,
    *,
    H: Int,
    W: Int,
    C_in: Int,
    C_out: Int,
    R: Int,
    S: Int,
    pad_h: Int,
    pad_w: Int,
    stride_h: Int = 1,
    stride_w: Int = 1,
](ctx: DeviceContext) raises:
    comptime N = 1
    comptime H_out = (H + 2 * pad_h - R) // stride_h + 1
    comptime W_out = (W + 2 * pad_w - S) // stride_w + 1
    comptime M_total = N * H_out * W_out
    comptime K_real = R * S * C_in
    comptime K_padded = _round_up(K_real, 256)

    print(
        "== rt-hw conv:",
        "dtype=",
        String(a_type),
        " H=",
        H,
        " W=",
        W,
        " C_in=",
        C_in,
        " C_out=",
        C_out,
        " RxS=",
        R,
        "x",
        S,
        " stride=",
        stride_h,
        "x",
        stride_w,
        " pad=",
        pad_h,
        "x",
        pad_w,
    )

    var input_dev = ctx.enqueue_create_buffer[a_type](N * H * W * C_in)
    var filter_dev = ctx.enqueue_create_buffer[a_type](C_out * K_padded)
    var output_static_dev = ctx.enqueue_create_buffer[c_type](M_total * C_out)
    var output_rt_dev = ctx.enqueue_create_buffer[c_type](M_total * C_out)

    var input_host = ctx.enqueue_create_host_buffer[a_type](N * H * W * C_in)
    var filter_host = ctx.enqueue_create_host_buffer[a_type](C_out * K_padded)
    rand(input_host.unsafe_ptr(), N * H * W * C_in)
    rand(filter_host.unsafe_ptr(), C_out * K_padded)
    for f in range(C_out):
        for k in range(K_real, K_padded):
            filter_host[f * K_padded + k] = 0
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)

    # ---- Static path (oracle) ----
    comptime static_in_layout = row_major[N, H, W, C_in]()
    comptime filter_layout = row_major[C_out, K_padded]()
    comptime static_out_layout = row_major[M_total, C_out]()
    var input_static = TileTensor(input_dev, static_in_layout)
    var filter = TileTensor(filter_dev, filter_layout)
    var output_static = TileTensor(output_static_dev, static_out_layout)
    amd_4wave_conv[
        H=H,
        W=W,
        H_out=H_out,
        W_out=W_out,
        R=R,
        S=S,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        C_in=C_in,
    ](input_static, filter, output_static, ctx)

    # ---- Runtime-HW path: build TileTensors with DYNAMIC NHWC dims ----
    comptime dyn_in_layout = row_major(Coord(IndexList[4](N, H, W, C_in)))
    comptime dyn_out_layout = row_major(Coord(IndexList[2](M_total, C_out)))
    var input_dyn = TileTensor(input_dev, dyn_in_layout)
    var output_rt = TileTensor(output_rt_dev, dyn_out_layout)
    # Note: H/W/H_out/W_out kwargs are *ignored* when use_runtime_hw=True
    # — the kernel reads input.dim() at runtime. R/S/stride/pad/C_in
    # still need to be comptime.
    amd_4wave_conv[
        R=R,
        S=S,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        C_in=C_in,
        use_runtime_hw=True,
    ](input_dyn, filter, output_rt, ctx)

    # ---- Compare byte-identical ----
    var static_host = ctx.enqueue_create_host_buffer[c_type](M_total * C_out)
    var rt_host = ctx.enqueue_create_host_buffer[c_type](M_total * C_out)
    ctx.enqueue_copy(static_host, output_static_dev)
    ctx.enqueue_copy(rt_host, output_rt_dev)
    ctx.synchronize()

    var mismatches = 0
    for i in range(M_total * C_out):
        if static_host[i] != rt_host[i]:
            mismatches += 1
    if mismatches > 0:
        print("  FAIL:", mismatches, "/", M_total * C_out, "differ")
        raise Error("runtime-HW conv != static-HW conv")
    print("  PASS:", M_total * C_out, "elements byte-identical")


def main() raises:
    with DeviceContext() as ctx:
        # 1x1 pointwise (BF16)
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=16,
            W=16,
            C_in=256,
            C_out=128,
            R=1,
            S=1,
            pad_h=0,
            pad_w=0,
        ](ctx)
        # 3x3 halo (BF16)
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=16,
            W=16,
            C_in=128,
            C_out=128,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # 3x3 stride=2 (BF16)
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=16,
            W=16,
            C_in=128,
            C_out=128,
            R=3,
            S=3,
            stride_h=2,
            stride_w=2,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # 3x3 halo (FP8)
        _run[
            DType.float8_e4m3fn,
            DType.bfloat16,
            H=16,
            W=16,
            C_in=128,
            C_out=128,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # FLUX VAE-style shapes: dynamic H/W (via the runtime-HW path),
        # static C_in/C_out/R/S. Mirrors the conv shapes the production
        # dispatcher routes through `use_runtime_hw=True` after FLUX2's
        # graph compiler lowers symbolic image_height/image_width.
        # 3x3 same-pad, BF16 — 128ch
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=32,
            W=32,
            C_in=128,
            C_out=128,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # 3x3 same-pad, BF16 — 256ch
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=32,
            W=32,
            C_in=256,
            C_out=256,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # 3x3 same-pad, BF16 — 512ch (FLUX VAE main block size)
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=16,
            W=16,
            C_in=512,
            C_out=512,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # 3x3 BF16 — 256→128 transition
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=32,
            W=32,
            C_in=256,
            C_out=128,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # 3x3 stride=2 BF16 — downsample
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=32,
            W=32,
            C_in=256,
            C_out=512,
            R=3,
            S=3,
            stride_h=2,
            stride_w=2,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # 1x1 pointwise BF16 — 512→128
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=32,
            W=32,
            C_in=512,
            C_out=128,
            R=1,
            S=1,
            pad_h=0,
            pad_w=0,
        ](ctx)
        # 3x3 pad=2 BF16 — non-1-pad
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=16,
            W=16,
            C_in=128,
            C_out=128,
            R=3,
            S=3,
            pad_h=2,
            pad_w=2,
        ](ctx)
