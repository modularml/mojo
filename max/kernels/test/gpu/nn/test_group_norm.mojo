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


import std.math as math
from std.math import rsqrt
from std.sys import simd_width_of

from max.gpu.host import DeviceContext, get_gpu_target
from layout import Coord, Idx, TileTensor, row_major
from nn.normalization import *
from std.testing import assert_almost_equal, assert_true

from std.utils.index import Index, IndexList


def compute_group_stats[
    t: DType
](vec: TileTensor[t, ...], size: Int, eps: Float32) raises -> Tuple[
    Float64,
    Float64,
]:
    """Compute group mean and rsqrt(variance + eps) in float64 for accuracy."""
    comptime assert vec.flat_rank == 1, "vec must be rank 1"
    comptime assert vec.element_size == 1
    var sum_val = Float64(0)
    var sum_sq = Float64(0)
    for i in range(size):
        var v = Float64(vec[i][0])
        sum_val += v
        sum_sq += v * v
    var mean = sum_val / Float64(size)
    var variance = max(sum_sq / Float64(size) - mean * mean, 0.0)
    return (mean, 1.0 / math.sqrt(variance + Float64(eps)))


def run_group_norm_gpu[
    dtype: DType, rank: Int
](
    ctx: DeviceContext,
    shape: IndexList[rank],
    num_groups: Int,
    rtol: Float64 = 1e-4,
    atol: Float64 = 1e-5,
) raises:
    print("== run_group_norm_gpu")

    var N = shape[0]
    var C = shape[1]
    var spatial = shape.flattened_length() // (N * C)
    var group_size = C // num_groups * spatial
    var rows = N * num_groups
    var cols = group_size

    var data_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var res = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](C)
    var beta_h = ctx.enqueue_create_host_buffer[dtype](C)

    for i in range(rows * cols):
        data_h[i] = Scalar[dtype](i % 256)  # bounded range to avoid overflow

    for i in range(C):
        gamma_h[i] = (Float64(i + C) / Float64(C)).cast[dtype]()
        beta_h[i] = (Float64(i) / Float64(C)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](C)
    var beta_d = ctx.enqueue_create_buffer[dtype](C)

    var param_shape = Index(C)
    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    var beta = TileTensor(beta_d, row_major(Coord(param_shape)))
    var epsilon = Float32(1e-5)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @__copy_capture(data_buf)
    @always_inline
    @__parameter
    def input_fn[width: Int](coords: Coord) -> SIMD[dtype, width]:
        var idx = data_buf.layout(coords)

        return data_buf.raw_load[width=width](idx)

    @__copy_capture(gamma)
    @always_inline
    @__parameter
    def gamma_scalar_fn[width: Int](coords: Coord) -> SIMD[dtype, width]:
        var idx = gamma.layout(coords)
        return gamma.raw_load[width=width](idx)

    @__copy_capture(beta)
    @always_inline
    @__parameter
    def beta_scalar_fn[width: Int](coords: Coord) -> SIMD[dtype, width]:
        var idx = beta.layout(coords)
        return beta.raw_load[width=width](idx)

    group_norm[dtype, rank, input_fn, gamma_scalar_fn, beta_scalar_fn, "gpu"](
        Coord(shape), epsilon, Int32(num_groups), data_buf, ctx=ctx
    )
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = TileTensor(
            data_h.unsafe_ptr() + r * cols,
            row_major(cols),
        )
        var stats = compute_group_stats(vec, cols, epsilon)
        var mean_ref = stats[0]
        var norm_factor = stats[1]
        for c in range(cols):
            var g = r % num_groups
            var c_base = g * (C // num_groups)
            var offset = c // spatial
            var true_c = c_base + offset
            var idx = r * cols + c
            # Compute reference in float64 for accuracy, then cast to
            # dtype for comparison (matches GPU's higher-precision accum).
            var val = Scalar[dtype](
                (Float64(vec[c][0]) - mean_ref)
                * norm_factor
                * Float64(gamma_h[true_c])
                + Float64(beta_h[true_c])
            )
            assert_almost_equal(val, res[idx], rtol=rtol, atol=atol)

    _ = data_d^
    _ = gamma_d^
    _ = beta_d^


def main() raises:
    with DeviceContext() as ctx:
        comptime default_simd = simd_width_of[
            DType.float32, target=get_gpu_target()
        ]()

        # === Warp-Tiling Kernel Dispatch (SIMD-aligned, fits warp strategy) ===

        # Small, SIMD-aligned groups
        run_group_norm_gpu[.float32](ctx, Index(2, 8, 2, 2), num_groups=4)
        run_group_norm_gpu[.float32](ctx, Index(2, 8, 4), num_groups=4)

        # Larger, but still small enough for warp tiling
        run_group_norm_gpu[.float32](ctx, Index(2, 32, 2, 2), num_groups=8)
        run_group_norm_gpu[.float32](ctx, Index(2, 32, 4), num_groups=8)

        # SIMD aligned with group boundary, but not aligned with channel boundary
        run_group_norm_gpu[.float32](ctx, Index(2, 32, 1, 10), num_groups=8)

        # === Block Kernel Dispatch (too wide for warp or not divisible by SIMD width) ===

        # Large column count (too wide for warp)
        run_group_norm_gpu[.float32](ctx, Index(1, 128, 1, 64), num_groups=8)
        run_group_norm_gpu[.float32](ctx, Index(1, 128, 64), num_groups=8)

        # Aligned, but still too large for warp strategy
        run_group_norm_gpu[.float32](ctx, Index(1, 64, 1, 64), num_groups=8)
        run_group_norm_gpu[.float32](ctx, Index(1, 64, 64), num_groups=8)

        # === Invalid Case: cols < simd_width → triggers safety assertion ===

        # Misaligned shape
        try:
            run_group_norm_gpu[.float32](ctx, Index(1, 33, 1, 1), num_groups=11)
        except e:
            assert_true(
                "group_norm_gpu requires num_cols >= simd_width" in String(e)
            )
        try:
            run_group_norm_gpu[.float32](ctx, Index(1, 33, 1), num_groups=11)
        except e:
            assert_true(
                "group_norm_gpu requires num_cols >= simd_width" in String(e)
            )

        # Small group sizes result in too few columns
        try:
            run_group_norm_gpu[.float32](ctx, Index(1, 12, 1, 1), num_groups=6)
        except e:
            assert_true(
                "group_norm_gpu requires num_cols >= simd_width" in String(e)
            )
        try:
            run_group_norm_gpu[.float32](ctx, Index(1, 12, 1), num_groups=6)
        except e:
            assert_true(
                "group_norm_gpu requires num_cols >= simd_width" in String(e)
            )

        # === Edge Cases ===

        # Trivial spatial=1 (all channels collapsed to one dimension)
        run_group_norm_gpu[.float32](ctx, Index(2, 128, 1, 1), num_groups=1)
        run_group_norm_gpu[.float32](ctx, Index(2, 128, 1), num_groups=1)

        # Non-multiple of simd_width → scalar fallback block path
        run_group_norm_gpu[.float32](ctx, Index(2, 33, 1, 1), num_groups=1)
        run_group_norm_gpu[.float32](ctx, Index(2, 33, 1), num_groups=1)

        # One-channel, one-group (channel_per_group=1)
        run_group_norm_gpu[.float32](ctx, Index(2, 1, 4, 8), num_groups=1)
        run_group_norm_gpu[.float32](ctx, Index(2, 1, 32), num_groups=1)

        # Edge case from group norm layer tests
        run_group_norm_gpu[.float32](ctx, Index(2, 2, 4, 4), num_groups=1)
        run_group_norm_gpu[.float32](ctx, Index(2, 2, 16), num_groups=1)

        # Zero-spatial input: the FLUX.2 VAE encoder is invoked
        # unconditionally on a ``(0, 0, 3)`` placeholder image for
        # text-to-image, and every ``GroupNorm`` inside the encoder sees
        # a ``(B, C, 0, 0)`` tensor.  The kernel must early-return rather
        # than dispatching with ``num_cols=0`` (which previously failed
        # the ``num_cols >= simd_width`` check).
        run_group_norm_gpu[.float32](ctx, Index(1, 128, 0, 0), num_groups=32)
        run_group_norm_gpu[.float32](ctx, Index(1, 512, 0, 0), num_groups=32)
        run_group_norm_gpu[.bfloat16](ctx, Index(1, 128, 0, 0), num_groups=32)
        run_group_norm_gpu[.bfloat16](ctx, Index(1, 512, 0, 0), num_groups=32)

        # === Multi-Block Kernel Dispatch (large group_size, few groups) ===

        # These shapes trigger the multi-block path because:
        # - group_size > WARP_SIZE * simd_width * max_warps_per_block
        #   (too large for warp tiling)
        # - num_rows (= N * num_groups) < desired_min_grid (= 256)
        # Tolerances are relaxed because the test reference uses a two-pass
        # variance formula (E[X^2]-E[X]^2) while the GPU uses Welford,
        # which diverge more at large group sizes.

        # bfloat16 tests matching FLUX2 VAE decoder dtype
        run_group_norm_gpu[.bfloat16](
            ctx, Index(1, 128, 64, 64), num_groups=32, rtol=2e-3, atol=5e-4
        )
        run_group_norm_gpu[.bfloat16](
            ctx, Index(1, 256, 64, 64), num_groups=32, rtol=2e-3, atol=5e-4
        )
        run_group_norm_gpu[.bfloat16](
            ctx, Index(1, 512, 32, 32), num_groups=32, rtol=2e-3, atol=5e-4
        )
        # Batch > 1 with multi-block
        run_group_norm_gpu[.bfloat16](
            ctx, Index(2, 256, 32, 32), num_groups=32, rtol=2e-3, atol=5e-4
        )
        # 3D multi-block
        run_group_norm_gpu[.bfloat16](
            ctx, Index(1, 128, 16384), num_groups=32, rtol=2e-3, atol=5e-4
        )

        # float32 multi-block tests for coverage
        run_group_norm_gpu[.float32](
            ctx, Index(1, 128, 64, 64), num_groups=32, rtol=2e-3, atol=5e-4
        )
        run_group_norm_gpu[.float32](
            ctx, Index(1, 512, 32, 32), num_groups=32, rtol=2e-3, atol=5e-4
        )

        # Mismatched channels/groups → top-level init error
        try:
            run_group_norm_gpu[.float32](ctx, Index(2, 7, 3, 3), num_groups=3)
        except e:
            assert_true("Invalid num_groups" in String(e))
        try:
            run_group_norm_gpu[.float32](ctx, Index(2, 7, 9), num_groups=3)
        except e:
            assert_true("Invalid num_groups" in String(e))
