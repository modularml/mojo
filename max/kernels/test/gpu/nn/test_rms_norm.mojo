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

from std.math import sqrt
from std.random import rand

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from nn.normalization import *
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


def compute_rms[
    dtype: DType
](data: TileTensor[dtype, ...], size: Int, eps: Float32) -> Scalar[dtype]:
    comptime assert data.flat_rank == 1, "data.rank must be 1"
    comptime assert data.element_size == 1

    comptime accum_type = get_accum_type[dtype]()
    var sum_of_squares = Scalar[accum_type]()
    for i in range(size):
        var val = data[i][0].cast[accum_type]()
        sum_of_squares += val * val
    var result = sqrt(
        (sum_of_squares / Scalar[accum_type](data.num_elements()))
        + eps.cast[accum_type]()
    )
    return result.cast[dtype]()


def run_rms_norm_gpu[
    rank: Int,
    //,
    dtype: DType,
    *,
    static_cols: Int = -1,
    multiply_before_cast: Bool = True,
](ctx: DeviceContext, shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    print("== run_rms_norm_gpu")

    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var res = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)

    rand[dtype](data_h.as_span())

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    var epsilon = Float32(0.001)
    var weight_offset = Scalar[dtype](0.0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)

    @always_inline
    @__copy_capture(data_buf)
    @__parameter
    def input_fn[width: Int](coords: Coord) -> SIMD[dtype, width]:
        var idx = data_buf.layout(coords)
        return data_buf.raw_load[width=width](idx)

    @always_inline
    @__copy_capture(data_buf)
    @__parameter
    def identity_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) -> None:
        var idx = data_buf.layout(coords)
        data_buf.raw_store[width=width, alignment=alignment](idx, val)

    # `rms_norm_gpu` migrated to a `Coord` shape boundary (softmax PR #88203);
    # `rank` is now an explicit parameter (no longer inferred from `shape`).
    rms_norm_gpu[
        rank,
        input_fn,
        identity_output_fn,
        multiply_before_cast=multiply_before_cast,
    ](Coord(shape), gamma, epsilon, weight_offset, ctx)
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = TileTensor(
            data_h.unsafe_ptr() + r * cols,
            row_major(cols),
        )
        var rms_ref = compute_rms(vec, cols, epsilon)
        for c in range(cols):
            var idx = r * cols + c
            var val = (data_h[idx] / rms_ref) * (gamma_h[c] + weight_offset)
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_d
    _ = gamma_d


def run_rms_norm_gpu_zero_rows[dtype: DType](ctx: DeviceContext) raises:
    """A zero-row shape must be a no-op, not a rejected zero-sized launch.

    A rank owns an empty shard when rows < TP group size (bs=1 decode at TP4).
    """
    print("== run_rms_norm_gpu_zero_rows")

    comptime cols = 128
    comptime sentinel = Scalar[dtype](-7.0)

    var data_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)
    for i in range(cols):
        data_h[i] = sentinel
        gamma_h[i] = Scalar[dtype](1.0)

    var data_d = ctx.enqueue_create_buffer[dtype](cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)

    var data_buf = TileTensor(data_d, row_major(Coord(Index(0, cols))))
    var gamma = TileTensor(gamma_d, row_major(Coord(Index(cols))))

    @always_inline
    @__copy_capture(data_buf)
    @__parameter
    def input_fn[width: Int](coords: Coord) -> SIMD[dtype, width]:
        return data_buf.raw_load[width=width](data_buf.layout(coords))

    @always_inline
    @__copy_capture(data_buf)
    @__parameter
    def identity_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) -> None:
        data_buf.raw_store[width=width, alignment=alignment](
            data_buf.layout(coords), val
        )

    rms_norm_gpu[2, input_fn, identity_output_fn, multiply_before_cast=True](
        Coord(Index(0, cols)),
        gamma,
        Float32(0.001),
        Scalar[dtype](0.0),
        ctx,
    )
    var res = ctx.enqueue_create_host_buffer[dtype](cols)
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for i in range(cols):
        assert_almost_equal(res[i], sentinel)

    _ = data_d
    _ = gamma_d


def run_rms_norm_row_based[
    dtype: DType, rank: Int, *, multiply_before_cast: Bool = True
](ctx: DeviceContext, shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    """Drives the row-based `rms_norm` entry point, not `rms_norm_gpu`.

    `mo.reduce.rms_norm` lowers through this overload, so its epilogue-closure
    contract needs coverage of its own; the `run_rms_norm_gpu` cases above
    exercise a different function.
    """
    print("== run_rms_norm_row_based")

    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    comptime eps_f32 = Float32(0.001)

    var data_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var res = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)

    rand[dtype](data_h.as_span())

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    # Distinct output buffer: input_fn (reads) and output_fn (writes) are
    # separate value-closure args, so they must reference distinct buffer
    # origins (writing in place into `data_d` would alias the read).
    var out_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var out_buf = TileTensor(out_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    var epsilon = eps_f32.cast[dtype]()
    var weight_offset = Scalar[dtype](0.0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        var idx = data_buf.layout(coords)
        return data_buf.raw_load[width=width, alignment=alignment](idx)

    @always_inline
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var out_buf}:
        var idx = out_buf.layout(coords)
        out_buf.raw_store[width=width, alignment=alignment](
            idx, rebind[SIMD[dtype, width]](val)
        )

    rms_norm[
        dtype, rank, target="gpu", multiply_before_cast=multiply_before_cast
    ](
        input_fn,
        output_fn,
        Coord(shape),
        Scalar[DType.int](cols),
        gamma,
        epsilon,
        weight_offset,
        ctx,
    )
    ctx.enqueue_copy(res, out_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = TileTensor(
            data_h.unsafe_ptr() + r * cols,
            row_major(cols),
        )
        var rms_ref = compute_rms(vec, cols, eps_f32)
        for c in range(cols):
            var idx = r * cols + c
            var val = (data_h[idx] / rms_ref) * (gamma_h[c] + weight_offset)
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_d
    _ = out_d
    _ = gamma_d


def main() raises:
    with DeviceContext() as ctx:
        run_rms_norm_gpu_zero_rows[.bfloat16](ctx)
        run_rms_norm_gpu_zero_rows[.float32](ctx)
        run_rms_norm_gpu[.float32](ctx, Index(5))
        run_rms_norm_gpu[.float32](ctx, Index(3, 4, 10, 20, 8))
        run_rms_norm_gpu[.float32](ctx, Index(1, 5, 6, 10, 128))
        run_rms_norm_gpu[.float32](ctx, Index(2, 5))
        run_rms_norm_gpu[.float32](ctx, Index(2, 55))
        run_rms_norm_gpu[.float32](ctx, Index(7, 557))
        run_rms_norm_gpu[.float32](ctx, Index(2, 8191))
        run_rms_norm_gpu[.float32](ctx, Index(2, 8192))
        run_rms_norm_gpu[.float32](ctx, Index(2, 16384))
        run_rms_norm_gpu[.float32](ctx, Index(2, 16385))
        run_rms_norm_gpu[.bfloat16](ctx, Index(3000, 32, 128), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16](ctx, Index(2999, 31, 128), rtol=2e-2)

        # Rank-3 `[B, S, H]` cases (added with the `IndexList`->`Coord` boundary
        # migration). These drive a non-trivial multi-axis
        # `_get_start_indices_of_nth_subvolume` per-row translation over the
        # outer `[B, S]` dims, exercising the divmod path the migration targets.
        # f32 cols=256 enters the single-pass warp-per-row regime; the wider
        # f32/bf16 cols cover the warp-tiling path.
        run_rms_norm_gpu[.float32](ctx, Index(8, 3072, 256))
        run_rms_norm_gpu[.float32](ctx, Index(4, 1024, 2048))
        run_rms_norm_gpu[.bfloat16](ctx, Index(4, 1024, 4096), rtol=2e-2)

        run_rms_norm_gpu[.float32](ctx, Index(32768, 1536))
        run_rms_norm_gpu[.bfloat16](ctx, Index(32768, 1536), rtol=2e-2)
        run_rms_norm_gpu[.float32](ctx, Index(4095, 1536))
        run_rms_norm_gpu[.float32](ctx, Index(64, 256))
        run_rms_norm_gpu[.bfloat16](ctx, Index(64, 256), rtol=2e-2)
        run_rms_norm_gpu[.float32](ctx, Index(32768, 2048))
        run_rms_norm_gpu[.bfloat16](ctx, Index(32768, 4096), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16](ctx, Index(8, 2048), rtol=2e-2)
        run_rms_norm_gpu[.float32](ctx, Index(8, 8193))

        # Test static shape dispatch.
        run_rms_norm_gpu[.bfloat16, static_cols=4096](
            ctx, Index(2, 4096), rtol=2e-2
        )
        run_rms_norm_gpu[.bfloat16, static_cols=16384](
            ctx, Index(2, 16384), rtol=2e-2
        )

        # High-row-count, register-resident widths: exercises the CDNA4
        # wide-SIMD warp-tiling path (gated on row count). cols are multiples
        # of 16 in the (128, 8192] warp-tiling range.
        run_rms_norm_gpu[.bfloat16](ctx, Index(1, 4096, 4096), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16](ctx, Index(1, 8192, 2880), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16](ctx, Index(1, 8192, 5120), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16](ctx, Index(1, 8192, 8192), rtol=2e-2)

        # Single-pass warp-per-row path: narrow exact-fit rows of 1..4 SIMD
        # vectors per lane (cols == chunks * WARP_SIZE * simd_width) with enough
        # rows to enter the warp-per-row region. f32 sp_stride=128 so cols
        # 128/256/384/512 => chunks 1..4; bf16 sp_stride=256 so cols
        # 256/512/768/1024 => chunks 1..4. The rank-4 case is a
        # production-representative normalized shape [4096,6,1,256] (rows=24576).
        # Cover both `multiply_before_cast` values.
        run_rms_norm_gpu[.float32](ctx, Index(24576, 128))
        run_rms_norm_gpu[.float32](ctx, Index(24576, 256))
        run_rms_norm_gpu[.float32](ctx, Index(24576, 384))
        run_rms_norm_gpu[.float32](ctx, Index(24576, 512))
        run_rms_norm_gpu[.float32](ctx, Index(4096, 6, 1, 256))
        run_rms_norm_gpu[.float32, multiply_before_cast=False](
            ctx, Index(24576, 512)
        )
        # bf16 narrow exact-fit rows now also use single-pass (previously
        # f32-only); chunks 1..4 across both `multiply_before_cast` values.
        run_rms_norm_gpu[.bfloat16](ctx, Index(24576, 256), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16](ctx, Index(24576, 512), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16](ctx, Index(24576, 768), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16](ctx, Index(24576, 1024), rtol=2e-2)
        run_rms_norm_gpu[.bfloat16, multiply_before_cast=False](
            ctx, Index(24576, 768), rtol=2e-2
        )

        # Cover the `multiply_before_cast=False` path (used by e.g. Llama)
        # across the multi-chunk warp-tiling dispatch: exact-fit and ragged,
        # narrow (simd) and wide (simd*2) branches.
        run_rms_norm_gpu[.float32, multiply_before_cast=False](
            ctx, Index(2, 8192)
        )
        run_rms_norm_gpu[.float32, multiply_before_cast=False](
            ctx, Index(7, 557)
        )
        run_rms_norm_gpu[.bfloat16, multiply_before_cast=False](
            ctx, Index(4, 4096), rtol=2e-2
        )
        run_rms_norm_gpu[.bfloat16, multiply_before_cast=False](
            ctx, Index(4, 8192), rtol=2e-2
        )
        run_rms_norm_gpu[.bfloat16, multiply_before_cast=False](
            ctx, Index(4, 5120), rtol=2e-2
        )

        # Row-based `rms_norm` (the `mo.reduce.rms_norm` lowering path), across
        # both `multiply_before_cast` branches and the narrow/wide dispatch.
        run_rms_norm_row_based[DType.float32](ctx, Index(2, 5))
        run_rms_norm_row_based[DType.float32](ctx, Index(7, 557))
        run_rms_norm_row_based[DType.float32](ctx, Index(24576, 256))
        run_rms_norm_row_based[DType.float32](ctx, Index(2, 8192))
        run_rms_norm_row_based[DType.float32](ctx, Index(3, 4, 10, 20, 8))
        run_rms_norm_row_based[DType.bfloat16](ctx, Index(64, 256), rtol=2e-2)
        run_rms_norm_row_based[DType.bfloat16](
            ctx, Index(4, 1024, 4096), rtol=2e-2
        )
        run_rms_norm_row_based[DType.float32, multiply_before_cast=False](
            ctx, Index(7, 557)
        )
        run_rms_norm_row_based[DType.float32, multiply_before_cast=False](
            ctx, Index(24576, 512)
        )
        run_rms_norm_row_based[DType.bfloat16, multiply_before_cast=False](
            ctx, Index(4, 4096), rtol=2e-2
        )
