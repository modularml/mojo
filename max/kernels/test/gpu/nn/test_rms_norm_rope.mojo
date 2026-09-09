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

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from nn.normalization import rms_norm_rope
from std.sys import align_of
from std.testing import assert_almost_equal
from std.utils.numerics import get_accum_type

from std.utils.index import Index, IndexList


def compute_rms_norm_rope_ref[
    dtype: DType, output_dtype: DType, cos_sin_dtype: DType
](
    input_h: Pointer[Scalar[dtype], _],
    gamma_h: Pointer[Scalar[dtype], _],
    cos_h: Pointer[Scalar[cos_sin_dtype], _],
    sin_h: Pointer[Scalar[cos_sin_dtype], _],
    output_ref: MutPointer[Scalar[output_dtype], _],
    rows: Int,
    cols: Int,
    epsilon: Float32,
    weight_offset: Scalar[dtype],
) raises:
    """CPU reference: RMS norm followed by RoPE rotation.

    The reduction and gamma scaling run in `dtype`'s accumulation precision and
    the normed value is rounded to `output_dtype` *before* RoPE, mirroring the
    fused kernel's `multiply_before_cast=True` / output-dtype rounding.
    """
    comptime accum_type = get_accum_type[dtype]()

    var half_cols = cols // 2

    for r in range(rows):
        # Compute RMS norm factor for this row.
        var sum_sq = Scalar[accum_type](0)
        for c in range(cols):
            var v = input_h[r * cols + c].cast[accum_type]()
            sum_sq += v * v
        var rms = math.sqrt(
            sum_sq / Scalar[accum_type](cols) + epsilon.cast[accum_type]()
        )
        var inv_rms = Scalar[accum_type](1) / rms

        # Apply norm with gamma (multiply_before_cast=True: all ops in
        # accum_type), rounding the normed value to output_dtype before RoPE.
        var normed = List(length=cols, fill=Scalar[output_dtype](0))
        for c in range(cols):
            var v = input_h[r * cols + c].cast[accum_type]()
            var g = (
                gamma_h[c].cast[accum_type]() + weight_offset.cast[accum_type]()
            )
            normed[c] = (v * inv_rms * g).cast[output_dtype]()

        # Apply RoPE: rotated[c] = -normed[c+half] if c < half else normed[c-half]
        for c in range(cols):
            var normed_val = normed[c].cast[accum_type]()
            var cos_val = cos_h[r * cols + c].cast[accum_type]()
            var sin_val = sin_h[r * cols + c].cast[accum_type]()
            var paired_c = c + half_cols if c < half_cols else c - half_cols
            var paired_val = normed[paired_c].cast[accum_type]()
            var rotated = -paired_val if c < half_cols else paired_val
            output_ref[r * cols + c] = (
                normed_val * cos_val + rotated * sin_val
            ).cast[output_dtype]()


def run_rms_norm_rope_gpu[
    rank: Int,
    //,
    dtype: DType,
    output_dtype: DType = dtype,
    cos_sin_dtype: DType = dtype,
](ctx: DeviceContext, shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    print("== run_rms_norm_rope_gpu")

    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var cos_h = ctx.enqueue_create_host_buffer[cos_sin_dtype](rows * cols)
    var sin_h = ctx.enqueue_create_host_buffer[cos_sin_dtype](rows * cols)
    var result_gpu = ctx.enqueue_create_host_buffer[output_dtype](rows * cols)
    var result_ref = ctx.enqueue_create_host_buffer[output_dtype](rows * cols)

    # Initialize with diverse deterministic values.
    for i in range(rows * cols):
        data_h[i] = (Float64((i % 37) - 18) / 10.0).cast[dtype]()
        cos_h[i] = (Float64((i % 19) - 9) / 10.0).cast[cos_sin_dtype]()
        sin_h[i] = (Float64((i % 23) - 11) / 10.0).cast[cos_sin_dtype]()

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()

    var epsilon = Float32(0.001)
    var weight_offset = Scalar[dtype](0.0)

    # Compute CPU reference.
    compute_rms_norm_rope_ref[dtype, output_dtype, cos_sin_dtype](
        data_h.unsafe_ptr(),
        gamma_h.unsafe_ptr(),
        cos_h.unsafe_ptr(),
        sin_h.unsafe_ptr(),
        result_ref.unsafe_ptr(),
        rows,
        cols,
        epsilon,
        weight_offset,
    )

    # Run GPU kernel.
    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var cos_d = ctx.enqueue_create_buffer[cos_sin_dtype](rows * cols)
    var sin_d = ctx.enqueue_create_buffer[cos_sin_dtype](rows * cols)
    var output_d = ctx.enqueue_create_buffer[output_dtype](rows * cols)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(cos_d, cos_h)
    ctx.enqueue_copy(sin_d, sin_h)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var output_buf = TileTensor(output_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(cols))
    var cos_vals = TileTensor(cos_d, row_major(Coord(shape)))
    var sin_vals = TileTensor(sin_d, row_major(Coord(shape)))

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        var idx = data_buf.layout(coords)
        return data_buf.raw_load[
            width=width, alignment=alignment * align_of[dtype]()
        ](idx)

    @always_inline
    def cos_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var cos_vals} -> SIMD[cos_sin_dtype, width]:
        var idx = cos_vals.layout(coords)
        return cos_vals.raw_load[
            width=width, alignment=alignment * align_of[cos_sin_dtype]()
        ](idx)

    @always_inline
    def sin_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var sin_vals} -> SIMD[cos_sin_dtype, width]:
        var idx = sin_vals.layout(coords)
        return sin_vals.raw_load[
            width=width, alignment=alignment * align_of[cos_sin_dtype]()
        ](idx)

    @always_inline
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[output_dtype, width]) {var output_buf} -> None:
        var idx = output_buf.layout(coords)
        output_buf.raw_store[
            width=width, alignment=alignment * align_of[output_dtype]()
        ](idx, val)

    rms_norm_rope[
        dtype,
        output_dtype,
        cos_sin_dtype,
        rank,
        target="gpu",
    ](
        input_fn,
        cos_fn,
        sin_fn,
        output_fn,
        Coord(shape),
        Int(cols),
        gamma,
        epsilon.cast[dtype](),
        weight_offset,
        ctx,
    )

    ctx.enqueue_copy(result_gpu, output_d)
    ctx.synchronize()

    # Compare GPU result with CPU reference.
    for i in range(rows * cols):
        assert_almost_equal(result_gpu[i], result_ref[i], rtol=rtol)

    _ = data_d
    _ = gamma_d
    _ = cos_d
    _ = sin_d
    _ = output_d


def main() raises:
    with DeviceContext() as ctx:
        # Basic shapes
        run_rms_norm_rope_gpu[.float32](ctx, Index(2, 4))
        run_rms_norm_rope_gpu[.float32](ctx, Index(3, 8))
        run_rms_norm_rope_gpu[.float32](ctx, Index(5, 16))
        # Higher rank
        run_rms_norm_rope_gpu[.float32](ctx, Index(2, 3, 8))
        run_rms_norm_rope_gpu[.float32](ctx, Index(1, 5, 6, 16))
        # Larger cols
        run_rms_norm_rope_gpu[.float32](ctx, Index(4, 128))
        run_rms_norm_rope_gpu[.float32](ctx, Index(2, 256))
        run_rms_norm_rope_gpu[.float32](ctx, Index(2, 4096))
        # bfloat16
        # BFloat16 accumulates rounding from both RMSNorm and RoPE; use 5%.
        run_rms_norm_rope_gpu[.bfloat16](ctx, Index(3, 128), rtol=5e-2)
        run_rms_norm_rope_gpu[.bfloat16](ctx, Index(2, 4096), rtol=5e-2)
        # Mixed cos/sin dtype
        run_rms_norm_rope_gpu[.bfloat16, cos_sin_dtype=DType.float32](
            ctx, Index(2, 128), rtol=5e-2
        )
        # Decoupled output dtype: f32 input/weight, bf16 output (the JSC-32
        # "float32 RMSNorm sandwich" the GC now fuses). Exercises the
        # warp-tiling (cols <= regs), block (large cols), and rank>2 paths.
        run_rms_norm_rope_gpu[
            DType.float32,
            output_dtype=DType.bfloat16,
            cos_sin_dtype=DType.bfloat16,
        ](ctx, Index(2, 256), rtol=5e-2)
        run_rms_norm_rope_gpu[
            DType.float32,
            output_dtype=DType.bfloat16,
            cos_sin_dtype=DType.bfloat16,
        ](ctx, Index(2, 4096), rtol=5e-2)
        run_rms_norm_rope_gpu[
            DType.float32,
            output_dtype=DType.bfloat16,
            cos_sin_dtype=DType.bfloat16,
        ](ctx, Index(2, 6, 256), rtol=5e-2)
