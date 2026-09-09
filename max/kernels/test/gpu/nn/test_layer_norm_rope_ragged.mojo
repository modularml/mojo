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
from nn.normalization import layer_norm_rope_ragged
from std.sys import align_of
from std.testing import assert_almost_equal
from std.utils.numerics import get_accum_type

from std.utils.index import Index, IndexList


def compute_layer_norm_rope_ragged_ref[
    dtype: DType, output_dtype: DType, freq_dtype: DType
](
    input_h: Pointer[Scalar[dtype], _],
    gamma_h: Pointer[Scalar[dtype], _],
    beta_h: Pointer[Scalar[dtype], _],
    row_offsets_h: Pointer[UInt32, _],
    start_pos_h: Pointer[UInt32, _],
    freqs_h: Pointer[Scalar[freq_dtype], _],
    output_ref: MutPointer[Scalar[output_dtype], _],
    rows: Int,
    cols: Int,
    rope_dim: Int,
    num_batches: Int,
    epsilon: Float32,
) raises:
    """CPU reference: LayerNorm followed by ragged, interleaved RoPE applied
    to the leading `rope_dim` columns; the rest pass through unrotated.

    Mirrors the fused kernel: LayerNorm's reduction/affine run in `dtype`'s
    accumulation precision and the normed value is rounded to `output_dtype`
    before RoPE.
    """
    comptime accum_type = get_accum_type[dtype]()

    for r in range(rows):
        # LayerNorm: mean + variance over the full row.
        var sum_val = Scalar[accum_type](0)
        for c in range(cols):
            sum_val += input_h[r * cols + c].cast[accum_type]()
        var mean = sum_val / Scalar[accum_type](cols)

        var sum_sq_diff = Scalar[accum_type](0)
        for c in range(cols):
            var d = input_h[r * cols + c].cast[accum_type]() - mean
            sum_sq_diff += d * d
        var variance = sum_sq_diff / Scalar[accum_type](cols)
        var inv = Scalar[accum_type](1) / math.sqrt(
            variance + epsilon.cast[accum_type]()
        )

        var normed = List(length=cols, fill=Scalar[output_dtype](0))
        for c in range(cols):
            var v = input_h[r * cols + c].cast[accum_type]()
            var g = gamma_h[c].cast[accum_type]()
            var b = beta_h[c].cast[accum_type]()
            normed[c] = ((v - mean) * inv * g + b).cast[output_dtype]()

        # Ragged absolute position for this row.
        var batch_idx = 0
        for b in range(num_batches):
            if (
                UInt32(r) >= row_offsets_h[b]
                and UInt32(r) < row_offsets_h[b + 1]
            ):
                batch_idx = b
                break
        var token_idx = r - Int(row_offsets_h[batch_idx])
        var pos = Int(start_pos_h[batch_idx]) + token_idx

        # Interleaved complex-multiply RoPE on [0, rope_dim); passthrough on
        # [rope_dim, cols). Matches `_rope`'s own precision (the multiply
        # runs in `freq_dtype`, not the accumulation type), so a narrow
        # `freq_dtype` doesn't produce a spurious reference/kernel mismatch.
        for c in range(cols):
            if c >= rope_dim:
                output_ref[r * cols + c] = normed[c]
                continue
            var pair_base = (c // 2) * 2
            var re_c = pair_base
            var im_c = pair_base + 1
            var x_re = normed[re_c].cast[freq_dtype]()
            var x_im = normed[im_c].cast[freq_dtype]()
            var f_re = freqs_h[pos * rope_dim + re_c]
            var f_im = freqs_h[pos * rope_dim + im_c]
            var result_re = x_re * f_re - x_im * f_im
            var result_im = x_re * f_im + x_im * f_re
            var result = result_re if c == re_c else result_im
            output_ref[r * cols + c] = result.cast[output_dtype]()


def run_layer_norm_rope_ragged_gpu[
    rank: Int,
    //,
    dtype: DType,
    rope_dim: Int,
    max_seq_len: Int,
    output_dtype: DType = dtype,
    freq_dtype: DType = dtype,
](
    ctx: DeviceContext,
    shape: IndexList[rank],
    row_offsets: List[UInt32],
    start_pos_vals: List[UInt32],
    rtol: Float64 = 0.01,
    atol: Float64 = 1e-8,
) raises:
    print("== run_layer_norm_rope_ragged_gpu")

    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols
    var num_batches = len(start_pos_vals)

    var data_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var beta_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var row_offsets_h = ctx.enqueue_create_host_buffer[.uint32](num_batches + 1)
    var start_pos_h = ctx.enqueue_create_host_buffer[.uint32](num_batches)
    var freqs_h = ctx.enqueue_create_host_buffer[freq_dtype](
        max_seq_len * rope_dim
    )
    var result_gpu = ctx.enqueue_create_host_buffer[output_dtype](rows * cols)
    var result_ref = ctx.enqueue_create_host_buffer[output_dtype](rows * cols)

    # Initialize with diverse deterministic values.
    for i in range(rows * cols):
        data_h[i] = (Float64((i % 37) - 18) / 10.0).cast[dtype]()
    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()
        beta_h[i] = (Float64(i % 5) / 10.0).cast[dtype]()
    for i in range(num_batches + 1):
        row_offsets_h[i] = row_offsets[i]
    for i in range(num_batches):
        start_pos_h[i] = start_pos_vals[i]
    for i in range(max_seq_len * rope_dim):
        freqs_h[i] = (Float64((i % 17) - 8) / 10.0).cast[freq_dtype]()

    var epsilon = Float32(0.001)

    # Compute CPU reference.
    compute_layer_norm_rope_ragged_ref[dtype, output_dtype, freq_dtype](
        data_h.unsafe_ptr(),
        gamma_h.unsafe_ptr(),
        beta_h.unsafe_ptr(),
        row_offsets_h.unsafe_ptr(),
        start_pos_h.unsafe_ptr(),
        freqs_h.unsafe_ptr(),
        result_ref.unsafe_ptr(),
        rows,
        cols,
        rope_dim,
        num_batches,
        epsilon,
    )

    # Run GPU kernel.
    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var beta_d = ctx.enqueue_create_buffer[dtype](cols)
    var row_offsets_d = ctx.enqueue_create_buffer[.uint32](num_batches + 1)
    var start_pos_d = ctx.enqueue_create_buffer[.uint32](num_batches)
    var freqs_d = ctx.enqueue_create_buffer[freq_dtype](max_seq_len * rope_dim)
    var output_d = ctx.enqueue_create_buffer[output_dtype](rows * cols)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)
    ctx.enqueue_copy(row_offsets_d, row_offsets_h)
    ctx.enqueue_copy(start_pos_d, start_pos_h)
    ctx.enqueue_copy(freqs_d, freqs_h)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var output_buf = TileTensor(output_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(cols))
    var beta = TileTensor(beta_d, row_major(cols))
    var row_offsets_buf = TileTensor(row_offsets_d, row_major(num_batches + 1))
    var start_pos_buf = TileTensor(start_pos_d, row_major(num_batches))
    var freqs_buf = TileTensor(freqs_d, row_major[max_seq_len, rope_dim]())

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        var idx = data_buf.layout(coords)
        return data_buf.raw_load[
            width=width, alignment=alignment * align_of[dtype]()
        ](idx)

    @always_inline
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[output_dtype, width]) {var output_buf} -> None:
        var idx = output_buf.layout(coords)
        output_buf.raw_store[
            width=width, alignment=alignment * align_of[output_dtype]()
        ](idx, val)

    layer_norm_rope_ragged[
        dtype,
        output_dtype,
        freq_dtype,
        rank,
        target="gpu",
        interleaved=True,
    ](
        input_fn,
        output_fn,
        Coord(shape),
        Int(cols),
        gamma,
        beta,
        epsilon.cast[dtype](),
        row_offsets_buf,
        start_pos_buf,
        freqs_buf,
        ctx,
    )

    ctx.enqueue_copy(result_gpu, output_d)
    ctx.synchronize()

    # Compare GPU result with CPU reference.
    for i in range(rows * cols):
        assert_almost_equal(result_gpu[i], result_ref[i], rtol=rtol, atol=atol)

    _ = data_d
    _ = gamma_d
    _ = beta_d
    _ = row_offsets_d
    _ = start_pos_d
    _ = freqs_d
    _ = output_d


def main() raises:
    with DeviceContext() as ctx:
        # Basic shapes: 2 sequences of 3 tokens each (rows=6), full-width RoPE
        # (rope_dim == cols).
        run_layer_norm_rope_ragged_gpu[
            dtype=DType.float32, rope_dim=8, max_seq_len=16
        ](
            ctx,
            Index(6, 8),
            row_offsets=[UInt32(0), 3, 6],
            start_pos_vals=[UInt32(0), 5],
        )
        # Partial RoPE matching the DeepSeekV3.2/GLM-5.2 Indexer shape
        # (head_dim=128, rope_dim=64, roped prefix + passthrough suffix).
        run_layer_norm_rope_ragged_gpu[
            dtype=DType.float32, rope_dim=64, max_seq_len=32
        ](
            ctx,
            Index(6, 128),
            row_offsets=[UInt32(0), 3, 6],
            start_pos_vals=[UInt32(0), 5],
        )
        # Uneven batch sizes and nonzero cache offsets for both sequences.
        run_layer_norm_rope_ragged_gpu[
            dtype=DType.float32, rope_dim=32, max_seq_len=32
        ](
            ctx,
            Index(9, 64),
            row_offsets=[UInt32(0), 2, 9],
            start_pos_vals=[UInt32(3), 11],
        )
        # bfloat16 end to end.
        run_layer_norm_rope_ragged_gpu[
            dtype=DType.bfloat16, rope_dim=64, max_seq_len=32
        ](
            ctx,
            Index(6, 128),
            row_offsets=[UInt32(0), 3, 6],
            start_pos_vals=[UInt32(0), 5],
            rtol=5e-2,
            atol=5e-3,
        )
        # Decoupled output dtype: f32 input/weight, bf16 output (the
        # "LayerNorm sandwich" the GC fuses through).
        run_layer_norm_rope_ragged_gpu[
            dtype=DType.float32,
            rope_dim=64,
            max_seq_len=32,
            output_dtype=DType.bfloat16,
            freq_dtype=DType.bfloat16,
        ](
            ctx,
            Index(6, 128),
            row_offsets=[UInt32(0), 3, 6],
            start_pos_vals=[UInt32(0), 5],
            rtol=5e-2,
            atol=5e-3,
        )
