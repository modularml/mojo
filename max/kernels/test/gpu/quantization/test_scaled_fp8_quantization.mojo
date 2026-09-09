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

from max.gpu.host import DeviceContext
from layout import Coord, CoordLike, Idx, MixedLayout, TileTensor, row_major
from layout._fillers import random
from linalg.fp8_quantization import (
    batched_quantize_dynamic_scaled_fp8,
    quantize_dynamic_scaled_fp8,
    quantize_static_scaled_fp8,
    quantize_tensor_dynamic_scaled_fp8,
)
from std.sys import has_nvidia_gpu_accelerator
from std.testing import assert_equal, assert_true, assert_almost_equal

from std.utils.numerics import (
    get_accum_type,
    isinf,
    isnan,
    max_finite,
    min_finite,
)


def test_static_scaled_fp8_quant[
    out_dtype: DType,
    in_dtype: DType,
](ctx: DeviceContext, scale: Float32, m: Int, n: Int) raises:
    var shape = row_major(Coord(Int64(m), Int64(n)))
    var total_size = m * n

    var in_host_ptr = alloc[Scalar[in_dtype]](total_size)
    var out_host_ptr = alloc[Scalar[out_dtype]](total_size)

    var in_host = TileTensor(in_host_ptr, shape)
    var out_host = TileTensor(out_host_ptr, shape)

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)

    random(in_host)
    _ = out_host.fill(0)

    ctx.enqueue_copy(in_device, in_host_ptr)
    ctx.enqueue_copy(out_device, out_host_ptr)

    var in_tt = TileTensor(in_device, shape)
    var out_tt = TileTensor(out_device, shape)

    quantize_static_scaled_fp8[out_dtype, in_dtype](out_tt, in_tt, scale, ctx)

    ctx.enqueue_copy(out_host_ptr, out_device)

    ctx.synchronize()

    for i in range(m):
        for j in range(n):
            var in_val_scaled_f32: Float32

            in_val_scaled_f32 = in_host[i, j][0].cast[.float32]() * (
                1.0 / scale
            )

            in_val_scaled_f32 = max(
                Float32(min_finite[out_dtype]()),
                min(Float32(max_finite[out_dtype]()), in_val_scaled_f32),
            )

            assert_equal(
                in_val_scaled_f32.cast[.float8_e4m3fn]().cast[DType.float64](),
                out_host[i, j][0].cast[.float64](),
            )

    in_host_ptr.free()
    out_host_ptr.free()


def test_dynamic_scaled_fp8_quant[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    MType: CoordLike,
    NType: CoordLike,
](ctx: DeviceContext, m: MType, n: NType) raises:
    comptime group_size: Int = NType.static_value
    comptime accum_dtype = get_accum_type[in_dtype]()

    var shape = row_major(Coord(m, n))
    var scales_shape = row_major(
        Coord(Idx[NType.static_value // group_size], m)
    )
    var total_size = Int(m.value()) * Int(n.value())
    var scales_size = (Int(n.value()) // group_size) * Int(m.value())

    var in_host_ptr = alloc[Scalar[in_dtype]](total_size)
    var out_host_ptr = alloc[Scalar[out_dtype]](total_size)
    var scales_host_ptr = alloc[Scalar[scales_dtype]](scales_size)

    var in_host = TileTensor(in_host_ptr, shape)
    var out_host = TileTensor(out_host_ptr, shape)
    var scales_host = TileTensor(scales_host_ptr, scales_shape)

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_size)

    random(in_host, -1.0, 1.0)

    ctx.enqueue_copy(in_device, in_host_ptr)

    var in_tensor = TileTensor(in_device, shape)
    var out_tensor = TileTensor(out_device, shape)
    var scales_tensor = TileTensor(scales_device, scales_shape)

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var in_tensor} -> SIMD[in_dtype, width]:
        return in_tensor.load[width=width, alignment=alignment]((row, col))

    quantize_tensor_dynamic_scaled_fp8[
        in_dtype=in_dtype,
        group_size_or_per_token=-1,
        num_cols=in_tensor.static_shape[1],
    ](
        input_fn,
        out_tensor,
        scales_tensor,
        1200.0,
        ctx,
        Int(in_tensor.dim[0]()),
    )

    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(scales_host_ptr, scales_device)
    ctx.synchronize()

    comptime assert in_host.flat_rank == 2
    comptime assert scales_host.flat_rank == 2

    comptime assert in_host.flat_rank == 2
    comptime assert scales_host.flat_rank == 2

    var max_scale = Scalar[scales_dtype](0)
    for i in range(Int(m.value())):
        for j in range(Int(n.value())):
            max_scale = max(
                max_scale,
                abs(in_host[i, j].cast[scales_dtype]()),
            )

    var scale_factor: Scalar[scales_dtype]

    scale_factor = (
        min(max_scale, 1200.0)
        / Scalar[out_dtype].MAX_FINITE.cast[scales_dtype]()
    )
    var scale_factor_recip = 1.0 / scale_factor.cast[accum_dtype]()

    assert_equal(
        scales_host[0, 0].cast[.float32](),
        scale_factor.cast[.float32](),
    )

    for i in range(Int(m.value())):
        for j in range(Int(n.value())):
            var in_val = in_host[i, j]
            var out_val = out_host[i, j]
            assert_equal(
                out_val.cast[.float32](),
                (in_val.cast[accum_dtype]() * scale_factor_recip)
                .cast[out_dtype]()
                .cast[.float32](),
                msg="At [" + String(i) + ", " + String(j) + "]",
            )

    in_host_ptr.free()
    out_host_ptr.free()
    scales_host_ptr.free()


def test_dynamic_fp8_quant[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    group_size_or_per_token: Int,
    MType: CoordLike,
    NType: CoordLike,
](ctx: DeviceContext, m: MType, n: NType) raises:
    comptime group_size: Int = NType.static_value if group_size_or_per_token == -1 else group_size_or_per_token
    comptime accum_dtype = get_accum_type[in_dtype]()

    var shape = row_major(Coord(m, n))
    var scales_shape = row_major(
        Coord(Idx[NType.static_value // group_size], m)
    )
    var total_size = Int(m.value()) * Int(n.value())
    var scales_size = (Int(n.value()) // group_size) * Int(m.value())

    var in_host_ptr = alloc[Scalar[in_dtype]](total_size)
    var out_host_ptr = alloc[Scalar[out_dtype]](total_size)
    var scales_host_ptr = alloc[Scalar[scales_dtype]](scales_size)

    var in_host = TileTensor(in_host_ptr, shape)
    var out_host = TileTensor(out_host_ptr, shape)
    var scales_host = TileTensor(scales_host_ptr, scales_shape)

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_size)

    random(in_host, -1.0, 1.0)

    ctx.enqueue_copy(in_device, in_host_ptr)

    var in_tensor = TileTensor(in_device, shape)
    var out_tensor = TileTensor(out_device, shape)
    var scales_tensor = TileTensor(scales_device, scales_shape)

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var in_tensor} -> SIMD[in_dtype, width]:
        return in_tensor.load[width=width, alignment=alignment]((row, col))

    quantize_dynamic_scaled_fp8[
        in_dtype=in_dtype,
        group_size_or_per_token=group_size_or_per_token,
        num_cols=in_tensor.static_shape[1],
    ](
        input_fn,
        out_tensor,
        scales_tensor,
        1200.0,
        ctx,
        Int(in_tensor.dim[0]()),
    )

    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(scales_host_ptr, scales_device)
    ctx.synchronize()

    comptime assert in_host.flat_rank == 2
    comptime assert scales_host.flat_rank == 2
    for i in range(Int(m.value())):
        for group_idx in range(Int(n.value()) // group_size):
            var group_max = Scalar[in_dtype](0)
            for j in range(group_size):
                group_max = max(
                    group_max,
                    abs(in_host[i, j + group_idx * group_size]),
                )

            var scale_factor: Scalar[scales_dtype]

            comptime if scales_dtype == .float8_e8m0fnu:
                scale_factor = max(
                    group_max.cast[accum_dtype]()
                    / Scalar[out_dtype].MAX_FINITE.cast[accum_dtype](),
                    Scalar[accum_dtype](1e-10),
                ).cast[scales_dtype]()
            else:
                scale_factor = (
                    min(group_max.cast[scales_dtype](), 1200.0)
                    / Scalar[out_dtype].MAX_FINITE.cast[scales_dtype]()
                )
            var scale_factor_recip = 1.0 / scale_factor.cast[accum_dtype]()

            assert_equal(
                scales_host[group_idx, i].cast[.float32](),
                scale_factor.cast[.float32](),
            )

            for j in range(group_size):
                var in_val = in_host[i, j + group_idx * group_size]
                var out_val = out_host[i, j + group_idx * group_size]
                assert_equal(
                    out_val.cast[.float32](),
                    (in_val.cast[accum_dtype]() * scale_factor_recip)
                    .cast[out_dtype]()
                    .cast[.float32](),
                    msg="At ["
                    + String(i)
                    + ", "
                    + String(j + group_idx * group_size)
                    + "]",
                )

    in_host_ptr.free()
    out_host_ptr.free()
    scales_host_ptr.free()


def test_batched_dynamic_fp8_quant[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    group_size_or_per_token: Int,
    BSType: CoordLike,
    MType: CoordLike,
    KType: CoordLike,
](ctx: DeviceContext, bs: BSType, m: MType, k: KType) raises:
    comptime group_size: Int = KType.static_value if group_size_or_per_token == -1 else group_size_or_per_token
    comptime accum_dtype = get_accum_type[in_dtype]()

    var shape = row_major(Coord(bs, m, k))
    var scales_shape = row_major(
        Coord(bs, Idx[KType.static_value // group_size], m)
    )
    var total_size = Int(bs.value()) * Int(m.value()) * Int(k.value())
    var scales_size = (
        Int(bs.value()) * (Int(k.value()) // group_size) * Int(m.value())
    )

    var in_host_ptr = alloc[Scalar[in_dtype]](total_size)
    var out_host_ptr = alloc[Scalar[out_dtype]](total_size)
    var scales_host_ptr = alloc[Scalar[scales_dtype]](scales_size)

    var in_host = TileTensor(in_host_ptr, shape)
    var out_host = TileTensor(out_host_ptr, shape)
    var scales_host = TileTensor(scales_host_ptr, scales_shape)

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_size)

    random(in_host, -1.0, 1.0)

    ctx.enqueue_copy(in_device, in_host_ptr)

    var in_tensor = TileTensor(in_device, shape)
    var out_tensor = TileTensor(out_device, shape)
    var scales_tensor = TileTensor(scales_device, scales_shape)

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](batch: Int, row: Int, col: Int) {var in_tensor} -> SIMD[in_dtype, width]:
        return in_tensor.load[width=width, alignment=alignment](
            (batch, row, col)
        )

    batched_quantize_dynamic_scaled_fp8[
        in_dtype=in_dtype,
        group_size_or_per_token=group_size_or_per_token,
        num_cols=in_tensor.static_shape[2],
    ](
        input_fn,
        out_tensor,
        scales_tensor,
        1200.0,
        ctx,
        num_rows=Int(m.value()),
        batch_size=Int(bs.value()),
    )

    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(scales_host_ptr, scales_device)
    ctx.synchronize()

    comptime assert in_host.flat_rank == 3
    comptime assert scales_host.flat_rank == 3
    for batch_idx in range(Int(bs.value())):
        for i in range(Int(m.value())):
            for group_idx in range(Int(k.value()) // group_size):
                var group_max = Scalar[in_dtype](0)
                for j in range(group_size):
                    group_max = max(
                        group_max,
                        abs(in_host[batch_idx, i, j + group_idx * group_size]),
                    )

                var scale_factor = (
                    min(group_max, 1200.0)
                    / Scalar[out_dtype].MAX_FINITE.cast[in_dtype]()
                )
                var scale_factor_recip = 1.0 / scale_factor.cast[accum_dtype]()

                assert_equal(
                    scales_host[batch_idx, group_idx, i].cast[.float64](),
                    scale_factor.cast[.float64](),
                )

                for j in range(group_size):
                    var in_val = in_host[
                        batch_idx, i, j + group_idx * group_size
                    ]
                    var out_val = out_host[
                        batch_idx, i, j + group_idx * group_size
                    ]

                    assert_equal(
                        out_val.cast[.float32](),
                        (in_val.cast[accum_dtype]() * scale_factor_recip)
                        .cast[out_dtype]()
                        .cast[.float32](),
                        msg="At ["
                        + String(i)
                        + ", "
                        + String(j + group_idx * group_size)
                        + "]",
                    )

    in_host_ptr.free()
    out_host_ptr.free()
    scales_host_ptr.free()


def test_dynamic_fp8_quant_near_zero[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    group_size: Int,
    MType: CoordLike,
    NType: CoordLike,
](ctx: DeviceContext, m: MType, n: NType) raises:
    """Regression for the FP8 dynamic-quant `0*Inf` NaN on a near-zero group.

    When a group's max-abs is a tiny f32 denormal (~1e-38), the dynamic scale
    `scale_factor = group_max / fp8_max` underflows to a NONZERO denormal, so
    `scale_factor_recip = 1/scale_factor` OVERFLOWS to +Inf (the `==0` guard
    misses it). `fp8_quantize` then computes `value * Inf` (NaN on a zero lane,
    Inf otherwise) and casts to fp8 — and on NVIDIA `use_clamp` is False, so the
    NaN/Inf flows into the output. A correct quant emits finite fp8 (the group
    is effectively zero → ~0). Asserts FINITENESS of every output element.
    """
    var shape = row_major(Coord(m, n))
    var scales_shape = row_major(
        Coord(Idx[NType.static_value // group_size], m)
    )
    var total_size = Int(m.value()) * Int(n.value())
    var scales_size = (Int(n.value()) // group_size) * Int(m.value())

    var in_host_ptr = alloc[Scalar[in_dtype]](total_size)
    var out_host_ptr = alloc[Scalar[out_dtype]](total_size)
    var scales_host_ptr = alloc[Scalar[scales_dtype]](scales_size)

    # Fill the whole input with a near-zero magnitude (~1e-38) and a zero lane
    # at the start of each group — so every group's max-abs is a tiny denormal
    # and each group has an exactly-zero element (the 0*Inf lane).
    for i in range(total_size):
        in_host_ptr[i] = Scalar[in_dtype](1e-38)
    for i in range(Int(m.value())):
        for g in range(Int(n.value()) // group_size):
            in_host_ptr[i * Int(n.value()) + g * group_size] = Scalar[in_dtype](
                0
            )

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_size)
    out_device.enqueue_fill(
        Scalar[out_dtype](0)
    )  # so unwritten cells are finite
    ctx.enqueue_copy(in_device, in_host_ptr)

    var in_tensor = TileTensor(in_device, shape)
    var out_tensor = TileTensor(out_device, shape)
    var scales_tensor = TileTensor(scales_device, scales_shape)

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var in_tensor} -> SIMD[in_dtype, width]:
        return in_tensor.load[width=width, alignment=alignment]((row, col))

    quantize_dynamic_scaled_fp8[
        in_dtype=in_dtype,
        group_size_or_per_token=group_size,
        num_cols=in_tensor.static_shape[1],
    ](input_fn, out_tensor, scales_tensor, 1200.0, ctx, Int(in_tensor.dim[0]()))

    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.synchronize()

    # Post-#87813 the cast clamp saturates the 0*Inf/+Inf, so a finiteness-only
    # check passes even WITHOUT the guard (vacuous — see
    # test_dynamic_tensor_fp8_quant_near_zero). Assert the VALUES are a clean fp8
    # zero, which the guard produces (scale_recip = 0) and the unguarded +Inf
    # path does NOT (it saturates to +-max_finite).
    var n_nonfinite = 0
    var n_nonzero = 0
    for i in range(total_size):
        var v = out_host_ptr[i].cast[.float32]()
        if isnan(v) or isinf(v):
            n_nonfinite += 1
        if v != 0.0:
            n_nonzero += 1
    print(
        "  near-zero group: fp8 out num_nonfinite =",
        n_nonfinite,
        " num_nonzero =",
        n_nonzero,
    )
    assert_equal(n_nonfinite, 0)
    assert_equal(n_nonzero, 0)

    in_host_ptr.free()
    out_host_ptr.free()
    scales_host_ptr.free()


def test_dynamic_tensor_fp8_quant_near_zero[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    group_size: Int,
    MType: CoordLike,
    NType: CoordLike,
](ctx: DeviceContext, m: MType, n: NType) raises:
    """Regression for the FP8 dynamic-quant `0*Inf` NaN on the PER-TENSOR path.

    Same denormal-scale reciprocal overflow as test_dynamic_fp8_quant_near_zero,
    but routed through `quantize_tensor_dynamic_scaled_fp8` with num_rows > 1, so
    it exercises the two-launch per-tensor reduction (compute_scales_fp8_kernel +
    quantize_fp8_kernel_per_tensor). That second kernel re-derives the scale from
    the tensor-wide max and previously open-coded a `== 0`-only reciprocal guard,
    so the denormal-max overflow could still NaN here (the per-group near-zero
    test above does not reach it). Asserts FINITENESS of every output element.
    """
    var shape = row_major(Coord(m, n))
    var scales_shape = row_major(
        Coord(Idx[NType.static_value // group_size], m)
    )
    var total_size = Int(m.value()) * Int(n.value())
    var scales_size = (Int(n.value()) // group_size) * Int(m.value())

    var in_host_ptr = alloc[Scalar[in_dtype]](total_size)
    var out_host_ptr = alloc[Scalar[out_dtype]](total_size)
    var scales_host_ptr = alloc[Scalar[scales_dtype]](scales_size)

    # Every group's max-abs is a tiny f32 denormal (~1e-38) with an exactly-zero
    # lane at each group start (the 0*Inf lane). With num_rows > 1 the per-tensor
    # path reduces all group maxes to one tensor-wide denormal scale.
    for i in range(total_size):
        in_host_ptr[i] = Scalar[in_dtype](1e-38)
    for i in range(Int(m.value())):
        for g in range(Int(n.value()) // group_size):
            in_host_ptr[i * Int(n.value()) + g * group_size] = Scalar[in_dtype](
                0
            )

    var in_device = ctx.enqueue_create_buffer[in_dtype](total_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](total_size)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_size)
    out_device.enqueue_fill(
        Scalar[out_dtype](0)
    )  # so unwritten cells are finite
    ctx.enqueue_copy(in_device, in_host_ptr)

    var in_tensor = TileTensor(in_device, shape)
    var out_tensor = TileTensor(out_device, shape)
    var scales_tensor = TileTensor(scales_device, scales_shape)

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var in_tensor} -> SIMD[in_dtype, width]:
        return in_tensor.load[width=width, alignment=alignment]((row, col))

    quantize_tensor_dynamic_scaled_fp8[
        in_dtype=in_dtype,
        group_size_or_per_token=group_size,
        num_cols=in_tensor.static_shape[1],
    ](input_fn, out_tensor, scales_tensor, 1200.0, ctx, Int(in_tensor.dim[0]()))

    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.synchronize()

    # WITH the reciprocal guard a near-zero group quantizes to a CLEAN fp8 zero
    # (scale_recip = 0 -> value*0 = 0). WITHOUT it, scale_recip = +Inf and every
    # lane saturates to +-max_finite at the cast clamp (use_clamp on since
    # #87813): finite, but WRONG. A finiteness-only check is therefore MASKED by
    # the clamp and would not catch a missing guard, so assert the VALUES are 0.
    var n_nonfinite = 0
    var n_nonzero = 0
    for i in range(total_size):
        var v = out_host_ptr[i].cast[.float32]()
        if isnan(v) or isinf(v):
            n_nonfinite += 1
        if v != 0.0:
            n_nonzero += 1
    print(
        "  near-zero per-tensor: fp8 out num_nonfinite =",
        n_nonfinite,
        " num_nonzero =",
        n_nonzero,
    )
    assert_equal(n_nonfinite, 0)
    assert_equal(n_nonzero, 0)

    in_host_ptr.free()
    out_host_ptr.free()
    scales_host_ptr.free()


def main() raises:
    with DeviceContext() as ctx:
        # Regression: FP8 dynamic-quant must not emit NaN on a near-zero group
        # (the 0*Inf / denormal-scale-reciprocal bug). Run first.
        #
        # The scales dtype is load-bearing for whether the bug even manifests:
        # for a near-zero group `scale_factor = group_max / fp8_max` is ~2e-41.
        # With FLOAT32 scales that value is preserved (a nonzero f32 denormal),
        # so `1/scale_factor` overflows to +Inf and the unguarded `==0`-only
        # check misses it — this is the case the fix exists for. With BFLOAT16
        # scales the same ~2e-41 underflows below bf16's min denormal (~9.2e-41)
        # and FLUSHES to exactly 0, so the `==0` guard already catches it and no
        # overflow occurs. Cover BOTH so the regression actually exercises the
        # overflow path (float32) and not just the benign bf16-flush path.
        test_dynamic_fp8_quant_near_zero[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.float32,
            group_size=128,
        ](ctx, Int(4), Idx[512])
        test_dynamic_fp8_quant_near_zero[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            group_size=128,
        ](ctx, Int(4), Idx[512])
        # Same near-zero denormal-scale regression on the PER-TENSOR path
        # (quantize_tensor_dynamic_scaled_fp8 -> quantize_fp8_kernel_per_tensor).
        # num_rows > 1 (m=4) forces the two-launch reduce-then-requantize path,
        # whose reciprocal was previously only `== 0`-guarded. Float32 scales so
        # the denormal survives to overflow `1/scale` (see the dtype note above).
        test_dynamic_tensor_fp8_quant_near_zero[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.float32,
            group_size=128,
        ](ctx, Int(4), Idx[512])
        test_static_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
        ](ctx, 0.5, 32, 16)
        test_static_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.float16,
        ](ctx, 0.33, 31, 15)
        test_static_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
        ](ctx, 0.3323, 31, 15)

        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
        ](ctx, Idx[800], Idx[8192])
        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
        ](ctx, Idx[1000], Idx[128])
        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
        ](ctx, Int(1), Idx[256])
        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
        ](ctx, Int(1), Idx[1024])
        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
        ](ctx, Int(1), Idx[16384])
        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
        ](ctx, Int(4), Idx[16384])
        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
        ](ctx, Int(4), Idx[576])

        # Test different alignments of the group_size to exercise the computation of simd_width.
        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
        ](ctx, Int(2), Idx[260])
        test_dynamic_scaled_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
        ](ctx, Int(2), Idx[264])

        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Int(1), Idx[256])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Int(1), Idx[1024])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Int(1), Idx[16384])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            128,
        ](ctx, Int(4), Idx[16384])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            128,
        ](ctx, Int(4), Idx[576])

        # Test different alignments of the group_size to exercise the computation of simd_width.
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Int(2), Idx[260])
        test_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Int(2), Idx[264])

        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Int(2), Int(1), Idx[256])
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Int(3), Int(1), Idx[1024])
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            -1,
        ](ctx, Int(4), Int(1), Idx[16384])
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            128,
        ](ctx, Int(128), Int(400), Idx[512])
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            128,
        ](ctx, Int(128), Int(1024), Idx[128])

        # Test different alignments of the group_size to exercise the computation of simd_width.
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            132,
        ](ctx, Int(128), Int(400), Idx[528])
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            136,
        ](ctx, Int(128), Int(1024), Idx[544])
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            128,
        ](ctx, Int(128), Int(1024), Idx[192])
        test_batched_dynamic_fp8_quant[
            DType.float8_e4m3fn,
            DType.float32,
            DType.float32,
            128,
        ](ctx, Int(7), Int(1000), Idx[576])

        # DType.float8_e8m0fnu is only supported on NVIDIA GPUs
        comptime if has_nvidia_gpu_accelerator():
            test_dynamic_fp8_quant[
                DType.float8_e4m3fn,
                DType.bfloat16,
                DType.float8_e8m0fnu,
                128,
            ](ctx, Int(43), Idx[1024])
            test_dynamic_fp8_quant[
                DType.float8_e4m3fn,
                DType.bfloat16,
                DType.float8_e8m0fnu,
                128,
            ](ctx, Int(3), Idx[16384])
            test_dynamic_fp8_quant[
                DType.float8_e4m3fn,
                DType.float32,
                DType.float8_e8m0fnu,
                128,
            ](ctx, Int(1), Idx[576])
