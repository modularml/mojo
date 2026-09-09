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
# SM100 (B200) static matmul small-M dispatch coverage (FP8 + BF16 output).
#
# Covers the `m <= 128` decode band of the SM100 static matmul dispatchers for
# the Nemotron-3-Nano-4B weight-proj shapes. For the small-M decode band
# (m in {25..31}) `choose_config` picks mma_n=16 / cta_group=1, a config no
# `build_sm100_matmul_configs` grid sample produces, so
# `select_and_launch_sm100_config` DISPATCH_MISSes and (pre-fix) falls back to
# vendor cuBLASLt.
#
#   * FP8 (a=b=float8_e4m3fn, c=bfloat16, static-scaled): the fp8 dispatcher's
#     `heuristic_and_outliers_dispatch` (bf16 output) has no never-miss, so the
#     band MISSed -> vendor. Fixed in `matmul_dispatch_sm100_fp8`.
#   * BF16 (a=b=c=bfloat16): `select_and_launch_sm100_config`'s never-miss is
#     fp8-OUTPUT-only, so `matmul_dispatch_sm100_bf16` MISSed -> vendor for the
#     same band. Fixed in `matmul_dispatch_sm100_bf16`.
#
# Both fixes mirror the fp8-OUTPUT never-miss: on a MISS launch the guaranteed
# -valid `default_matmul_config_bf16_fp8` config on MAX's own tcgen05 Mojo
# kernel and return DISPATCH_HIT.
#
# This test asserts two things per (dtype, shape, m):
#   1. the dispatcher returns DISPATCH_HIT (MAX Mojo kernel, not vendor MISS);
#   2. the DISPATCH_HIT output matches the vendor-cuBLASLt reference within
#      FP8/BF16-accum tolerance (rtol/atol = 1e-2).

from std.sys import size_of
import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from internal_utils import assert_almost_equal
from std.random import rand, seed
from layout import TileTensor, Coord, row_major, Idx
from linalg.matmul.gpu.sm100_structured.default.dispatch import (
    matmul_dispatch_sm100_fp8,
    matmul_dispatch_sm100_bf16,
    DISPATCH_HIT,
)
from std.utils.index import IndexList
from std.testing import assert_equal


# Static per-tensor scale (input_scale * weight_scale), folded into the compute
# epilogue on the FP8 Mojo path and into the vendor `alpha` on the reference.
comptime STATIC_SCALE = 0.5

# Previously-missing small-M decode band (m in {25..31}) + control m-values.
comptime BAND_MS = [25, 27, 29, 31, 8, 16, 64, 128]


@__parameter
@always_inline
def scaled_compute_fn[
    dtype: DType, width: SIMDLength, *, alignment: Int = 1
](idx: IndexList[2], val: SIMD[dtype, width]) capturing -> SIMD[dtype, width]:
    # Mirror MatmulStaticScaledFloat8's SM100 compute lambda: accumulate in
    # fp32, apply the scalar scale, cast back to output dtype (bf16).
    var scaled = val.cast[.float32]() * Float32(STATIC_SCALE)
    return scaled.cast[dtype]()


def _assert_hit(status: Int, tag: String, N: Int, K: Int, m: Int) raises:
    assert_equal(
        status,
        DISPATCH_HIT,
        String(
            "expected DISPATCH_HIT (MAX Mojo kernel) for ",
            tag,
            " (N=",
            N,
            ", K=",
            K,
            ", m=",
            m,
            "), got ",
            status,
        ),
    )


def check_fp8_band[N: Int, K: Int](ctx: DeviceContext, m: Int) raises:
    comptime a_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime transpose_b = True

    var a_shape = row_major(Coord(m, Idx[K]))
    var b_shape = row_major(Coord(Idx[N], Idx[K]))  # transpose_b: [N, K]
    var c_shape = row_major(Coord(m, Idx[N]))

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](m * K)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = ctx.enqueue_create_host_buffer[a_type](N * K)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](m * N)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_ref_host_ptr = ctx.enqueue_create_host_buffer[c_type](m * N)
    var c_ref_host = TileTensor(c_ref_host_ptr, c_shape)

    var a_device = ctx.enqueue_create_buffer[a_type](m * K)
    var a_tensor = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[a_type](N * K)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[c_type](m * N)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_ref_device = ctx.enqueue_create_buffer[c_type](m * N)
    var c_ref_tensor = TileTensor(c_ref_device, c_shape)

    seed(0)
    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    # MAX Mojo path: the production static-scaled FP8 dispatch entry.
    var status = matmul_dispatch_sm100_fp8[
        transpose_b=transpose_b,
        elementwise_compute_lambda_fn=scaled_compute_fn,
    ](c_tensor, a_tensor, b_tensor, ctx)
    _assert_hit(status, "FP8", N, K, m)

    # Reference: same FP8 GEMM via vendor cuBLASLt, static scale via `alpha`.
    vendor_blas.matmul(
        ctx,
        c_ref_tensor.to_layout_tensor(),
        a_tensor.to_layout_tensor(),
        b_tensor.to_layout_tensor(),
        c_row_major=True,
        transpose_b=transpose_b,
        alpha=Float32(STATIC_SCALE),
    )

    ctx.synchronize()
    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_ref_host_ptr, c_ref_device)
    ctx.synchronize()

    assert_almost_equal(
        c_host._storage, c_ref_host._storage, m * N, atol=0.01, rtol=0.01
    )
    print("=== PASS FP8 (N=", N, ", K=", K, ", m=", m, ") ===")

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_ref_device^


def check_bf16_band[N: Int, K: Int](ctx: DeviceContext, m: Int) raises:
    comptime dt = DType.bfloat16
    comptime transpose_b = True

    var a_shape = row_major(Coord(m, Idx[K]))
    var b_shape = row_major(Coord(Idx[N], Idx[K]))  # transpose_b: [N, K]
    var c_shape = row_major(Coord(m, Idx[N]))

    var a_host_ptr = ctx.enqueue_create_host_buffer[dt](m * K)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = ctx.enqueue_create_host_buffer[dt](N * K)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_ptr = ctx.enqueue_create_host_buffer[dt](m * N)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_ref_host_ptr = ctx.enqueue_create_host_buffer[dt](m * N)
    var c_ref_host = TileTensor(c_ref_host_ptr, c_shape)

    var a_device = ctx.enqueue_create_buffer[dt](m * K)
    var a_tensor = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[dt](N * K)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[dt](m * N)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_ref_device = ctx.enqueue_create_buffer[dt](m * N)
    var c_ref_tensor = TileTensor(c_ref_device, c_shape)

    seed(0)
    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    # MAX Mojo path: the SM100 bf16 static matmul dispatch entry (no epilogue).
    var status = matmul_dispatch_sm100_bf16[transpose_b=transpose_b](
        c_tensor, a_tensor, b_tensor, ctx
    )
    _assert_hit(status, "BF16", N, K, m)

    # Reference: same GEMM via vendor cuBLASLt.
    vendor_blas.matmul(
        ctx,
        c_ref_tensor.to_layout_tensor(),
        a_tensor.to_layout_tensor(),
        b_tensor.to_layout_tensor(),
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.synchronize()
    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_ref_host_ptr, c_ref_device)
    ctx.synchronize()

    assert_almost_equal(
        c_host._storage, c_ref_host._storage, m * N, atol=0.01, rtol=0.01
    )
    print("=== PASS BF16 (N=", N, ", K=", K, ", m=", m, ") ===")

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_ref_device^


def fp8_all[N: Int, K: Int](ctx: DeviceContext) raises:
    comptime for m in BAND_MS:
        check_fp8_band[N, K](ctx, m)


def bf16_all[N: Int, K: Int](ctx: DeviceContext) raises:
    comptime for m in BAND_MS:
        check_bf16_band[N, K](ctx, m)


def main() raises:
    with DeviceContext() as ctx:
        # FP8 static-scaled weight-proj (N, K) shapes.
        fp8_all[17504, 3136](ctx)
        fp8_all[3136, 7680](ctx)
        fp8_all[12544, 3136](ctx)
        fp8_all[3136, 12544](ctx)

        # BF16 Nemotron-3-Nano-4B (N, K) shapes (hidden_size K=3136):
        # gate/up (intermediate=12544), down, attn qkv (N=7168), attn o
        # (K=q_heads*head_dim=5120). The lm_head shape (131072, 3136) exercises
        # the same band code path but its B weight (784MB in bf16) exceeds the
        # bazel GPU-memory sandbox chunk; it is validated by a direct `mojo`
        # run, not gated here.
        bf16_all[12544, 3136](ctx)
        bf16_all[3136, 12544](ctx)
        bf16_all[7168, 3136](ctx)
        bf16_all[3136, 5120](ctx)
