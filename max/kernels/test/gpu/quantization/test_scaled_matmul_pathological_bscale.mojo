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
#
# SM100 (B200) isolated repro for the MLA-prefill up-proj GEMM NaN surface.
#
# The Kimi-K2.5 dense FP8 MLA prefill fuses two FP8 block-scaled up-proj GEMMs
# (matmul_dynamic_scaled_fp8 / batched_matmul_dynamic_scaled_fp8): Q@w_uk and
# raw_out@w_uv. The activation dynamic-quant feeding `a_scales` is now guarded
# (the 0*Inf reciprocal fix), but the WEIGHT scales `b_scales` are loaded from
# disk and applied INSIDE the GEMM as a plain multiply (c += (a*a_scale) @
# (b*b_scale)). This test asks: if a loaded b_scale is pathological (a near-zero
# f32 denormal, or a very large value), does the GEMM output go non-finite?
#
# Two pathological columns are injected into b_scales:
#   - a near-zero f32 DENORMAL scale (1e-40): the dequant `b*b_scale` for that
#     output column underflows; the column's output should be a clean ~0, never
#     a NaN (there is no reciprocal here — b_scale multiplies directly).
#   - a LARGE scale (1e30): `b * 1e30` for full-magnitude fp8 weights (~448)
#     gives ~4e32, accumulated over K — this probes whether the f32 accumulator
#     saturates to +Inf (a real overflow) vs stays finite.
#
# Expected: the denormal column is finite (~0). The large-scale column MAY
# legitimately overflow to +Inf for an adversarial scale — that is arithmetic
# overflow, not a quant bug, and the production weight scales are O(1) trained
# values, so this case is informational (it characterizes the ceiling, it does
# not by itself indict the kernel). The finding that matters for the Kimi NaN is
# whether a *plausible* (denormal/zero) loaded scale can yield NaN: it must not.

from max.gpu.host import DeviceContext
from layout import (
    CoordLike,
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from std.utils.numerics import isinf, isnan, max_finite
from linalg.fp8_quantization import matmul_dynamic_scaled_fp8
from std.testing import assert_equal


def run_pathological_bscale[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    """Runs matmul_dynamic_scaled_fp8 with pathological b_scale columns and
    reports (num_nan, num_inf) in the GEMM output, plus the denormal-column
    finiteness specifically."""
    comptime transpose_b = True
    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    var a_size = M * K
    var b_size = N * K  # transpose_b: b is [N, K]
    var c_size = M * N
    var a_scales_size = M
    var b_scales_size = N

    var a_host_ptr = alloc[Scalar[in_dtype]](a_size)
    var b_host_ptr = alloc[Scalar[in_dtype]](b_size)
    var c_host_ptr = alloc[Scalar[out_dtype]](c_size)
    var a_scales_host_ptr = alloc[Scalar[scales_dtype]](a_scales_size)
    var b_scales_host_ptr = alloc[Scalar[scales_dtype]](b_scales_size)

    var a_layout = row_major(Coord(m, k))
    var b_layout = row_major(Coord(n, k))
    var c_layout = row_major(Coord(m, n))
    var a_scales_layout = row_major(Coord(Idx[1], m))
    var b_scales_layout = row_major(Coord(n, Idx[1]))

    # Fill A and B with full-magnitude fp8 values (worst case for overflow):
    # every element = fp8 max (448). a_scale = 1.0 (benign activation scale).
    comptime fp8_max = max_finite[in_dtype]()
    for i in range(a_size):
        a_host_ptr[i] = fp8_max
    for i in range(b_size):
        b_host_ptr[i] = fp8_max
    for i in range(a_scales_size):
        a_scales_host_ptr[i] = Scalar[scales_dtype](1.0)

    # b_scales: most columns benign (1.0); inject pathologies.
    # column 0: near-zero f32 denormal; column 1 (if exists): large 1e30.
    for j in range(b_scales_size):
        b_scales_host_ptr[j] = Scalar[scales_dtype](1.0)
    var denormal_col = 0
    b_scales_host_ptr[denormal_col] = Scalar[scales_dtype](1e-40)
    var large_col = -1
    if N > 1:
        large_col = 1
        b_scales_host_ptr[large_col] = Scalar[scales_dtype](1e30)

    var a_device = ctx.enqueue_create_buffer[in_dtype](a_size)
    var b_device = ctx.enqueue_create_buffer[in_dtype](b_size)
    var c_device = ctx.enqueue_create_buffer[out_dtype](c_size)
    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](a_scales_size)
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](b_scales_size)

    # Zero-init output so an un-written cell is a clean 0, not garbage.
    c_device.enqueue_fill(Scalar[out_dtype](0))

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    var a_tile = TileTensor(a_device, a_layout)
    var b_tile = TileTensor(b_device, b_layout)
    var c_tile = TileTensor(c_device, c_layout)
    var a_scales_tile = TileTensor(a_scales_device, a_scales_layout)
    var b_scales_tile = TileTensor(b_scales_device, b_scales_layout)

    matmul_dynamic_scaled_fp8[
        input_scale_granularity="colwise",
        weight_scale_granularity="rowwise",
        m_scale_granularity=1,
        n_scale_granularity=1,
        k_scale_granularity=KType.static_value,
        transpose_b=transpose_b,
        target="gpu",
    ](
        c_tile,
        a_tile,
        b_tile,
        a_scales_tile,
        b_scales_tile,
        ctx,
    )
    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.synchronize()

    var c_host = TileTensor(c_host_ptr, c_layout)
    comptime assert c_host.flat_rank == 2

    var num_nan = 0
    var num_inf = 0
    var denormal_col_nonfinite = 0
    var inf_outside_large_col = 0
    for i in range(M):
        for j in range(N):
            var val = c_host[i, j].cast[.float32]()
            if isnan(val):
                num_nan += 1
            elif isinf(val):
                num_inf += 1
                if j != large_col:
                    inf_outside_large_col += 1
            if j == denormal_col and (isnan(val) or isinf(val)):
                denormal_col_nonfinite += 1

    print(
        "  up-proj GEMM pathological b_scale: M=",
        M,
        "N=",
        N,
        "K=",
        K,
        "-> NaN=",
        num_nan,
        "Inf=",
        num_inf,
        "(denormal-col non-finite=",
        denormal_col_nonfinite,
        ", large-col idx=",
        large_col,
        ")",
    )
    # Hard assertions for the NaN this test guards: a near-zero/denormal weight
    # scale must NOT produce NaN, and the denormal column must be finite (~0).
    # Inf in the deliberately-huge large_col is arithmetic overflow of an
    # adversarial scale (informational), but no OTHER column may go infinite.
    assert_equal(num_nan, 0, msg="NaN in up-proj GEMM output (b_scale path)")
    assert_equal(
        denormal_col_nonfinite,
        0,
        msg="denormal-scale column went non-finite (must quantize to ~0)",
    )
    assert_equal(
        inf_outside_large_col,
        0,
        msg="Inf outside the deliberately-large b_scale column",
    )

    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()
    _ = a_device
    _ = b_device
    _ = c_device
    _ = a_scales_device
    _ = b_scales_device


def main() raises:
    with DeviceContext() as ctx:
        # Kimi up-proj-ish dims. Q@w_uk: N=kv_lora_rank(512), K=qk_nope(128).
        # raw_out@w_uv: N=v_head_dim(128), K=kv_lora_rank(512). M = a few
        # query tokens times heads. Use representative M.
        run_pathological_bscale[
            in_dtype=DType.float8_e4m3fn,
            out_dtype=DType.bfloat16,
            scales_dtype=DType.float32,
        ](ctx, Idx[128], Idx[512], Idx[128])
        run_pathological_bscale[
            in_dtype=DType.float8_e4m3fn,
            out_dtype=DType.bfloat16,
            scales_dtype=DType.float32,
        ](ctx, Idx[128], Idx[128], Idx[512])
