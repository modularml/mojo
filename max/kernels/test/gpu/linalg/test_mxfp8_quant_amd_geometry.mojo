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
"""Launch-geometry gates for `quantize_mx_amd`'s MXFP8 path (MI355X / CDNA4).

Four threads cooperate through `lane_group_max` to scale one 32-element MX
block, so every geometry the launcher derives must keep each lane group covering
exactly one block -- an invariant nothing in the kernel enforces, and one the
MXFP4 round-trip coverage cannot see: a two-lane permutation of the
thread-to-column map leaves `test_quantize_roundtrip[4, 7168]` at its usual 0.2
max relative error, that tolerance being wider than the error a wrong block
scale introduces.
"""

from std.math import ceildiv
from std.memory import bitcast
from std.random import random_float64, seed
from std.testing import assert_equal, assert_true
from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from linalg.block_scaled_utils import compute_mxfp8_block_scale
from linalg.block_scaled_quantization import quantize_mx_amd
from linalg.fp4_utils import MXFP4_SF_VECTOR_SIZE, MXFP8_SF_VECTOR_SIZE

comptime SEED = 0xC0FFEE


def _block_magnitude(block_idx: Int) -> Float64:
    """Returns a power of two in [2^-10, 2^10], cycling over the block index.

    Every test here needs the ladder: at a flat magnitude a straddling lane
    group picks the right scale by accident and passes while broken.
    """
    return Float64(1 << (block_idx % 21)) / 1024.0


def _fill_input(mut values: List[Float32], M: Int, K: Int):
    """Builds the deterministic BF16 input, widened to f32 for the oracle."""
    seed(SEED)
    values.clear()
    for _ in range(M):
        for col in range(K):
            var mag = _block_magnitude(col // MXFP8_SF_VECTOR_SIZE)
            var v = (random_float64(-4.0, 4.0) * mag).cast[.bfloat16]()
            values.append(v.cast[.float32]())


def _quantize_mxfp8[
    M: Int, K: Int, num_max_threads: Int
](
    ctx: DeviceContext,
    values: List[Float32],
    mut data: List[UInt8],
    mut scales: List[UInt8],
) raises:
    comptime scale_K = ceildiv(K, MXFP8_SF_VECTOR_SIZE)

    var input_dev = ctx.enqueue_create_buffer[.bfloat16](M * K)
    with input_dev.map_to_host() as h:
        for i in range(M * K):
            h[i] = values[i].cast[.bfloat16]()

    var data_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](M * K)
    var scales_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](M * scale_K)
    var input_tt = TileTensor(input_dev, row_major((Idx[M], Idx[K])))
    var data_tt = TileTensor(data_dev, row_major((Idx[M], Idx[K])))
    var scales_tt = TileTensor(scales_dev, row_major((Idx[M], Idx[scale_K])))
    quantize_mx_amd[num_max_threads=num_max_threads](
        ctx, data_tt, scales_tt, input_tt
    )

    var data_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](M * K)
    var scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        M * scale_K
    )
    ctx.enqueue_copy(data_host, data_dev)
    ctx.enqueue_copy(scales_host, scales_dev)
    ctx.synchronize()

    data.clear()
    for i in range(M * K):
        data.append(bitcast[.uint8](data_host[i]))
    scales.clear()
    for i in range(M * scale_K):
        scales.append(bitcast[.uint8](scales_host[i]))


def test_mxfp8_matches_host_oracle[M: Int, K: Int](ctx: DeviceContext) raises:
    comptime scale_K = ceildiv(K, MXFP8_SF_VECTOR_SIZE)
    print("  host oracle M=", M, " K=", K)

    var values = List[Float32]()
    _fill_input(values, M, K)
    var data = List[UInt8]()
    var scales = List[UInt8]()
    _quantize_mxfp8[M, K, 512](ctx, values, data, scales)

    var distinct_scales = List[UInt8]()
    var nonzero_bytes = 0
    for row in range(M):
        for block in range(scale_K):
            var base = row * K + block * MXFP8_SF_VECTOR_SIZE
            var group_max = Float32(0)
            for e in range(MXFP8_SF_VECTOR_SIZE):
                group_max = max(group_max, abs(values[base + e]))

            var want_scale: Float8_e8m0fnu
            var multiplier: Float32
            var block_is_dead: Bool
            want_scale, multiplier, block_is_dead = compute_mxfp8_block_scale[
                .float8_e8m0fnu
            ](group_max)
            # Host and device `recip` disagree at E8M0's subnormal floor, so an
            # oracle over a dead block would compare two different functions.
            assert_true(
                not block_is_dead,
                (
                    "test input produced a dead MX block; the host oracle does"
                    " not model that path"
                ),
            )

            var want_bits = bitcast[.uint8](want_scale)
            assert_equal(
                scales[row * scale_K + block],
                want_bits,
                String("scale mismatch at row ")
                + String(row)
                + " block "
                + String(block),
            )
            if want_bits not in distinct_scales:
                distinct_scales.append(want_bits)

            for e in range(MXFP8_SF_VECTOR_SIZE):
                var want = (values[base + e] * multiplier).cast[
                    .float8_e4m3fn
                ]()
                var got = data[base + e]
                assert_equal(
                    got,
                    bitcast[.uint8](want),
                    String("data mismatch at row ")
                    + String(row)
                    + " col "
                    + String(block * MXFP8_SF_VECTOR_SIZE + e),
                )
                if got != UInt8(0):
                    nonzero_bytes += 1

    print(
        "    distinct scales=",
        len(distinct_scales),
        " nonzero bytes=",
        nonzero_bytes,
    )
    # Non-vacuity: an all-zero output or a single scale for the whole tensor
    # would satisfy the comparisons above without exercising the grouping.
    assert_true(
        len(distinct_scales) >= min(8, scale_K),
        String("input exercised only ")
        + String(len(distinct_scales))
        + " distinct scales",
    )
    assert_true(nonzero_bytes > M * K // 2, "most output bytes are zero")


def test_mxfp8_geometry_invariant[M: Int, K: Int](ctx: DeviceContext) raises:
    comptime scale_K = ceildiv(K, MXFP8_SF_VECTOR_SIZE)
    print("  geometry invariance M=", M, " K=", K)

    var values = List[Float32]()
    _fill_input(values, M, K)

    var data = List[UInt8]()
    var scales = List[UInt8]()
    _quantize_mxfp8[M, K, 512](ctx, values, data, scales)

    var alt_data = List[UInt8]()
    var alt_scales = List[UInt8]()
    comptime for threads in [256, 128, 64]:
        _quantize_mxfp8[M, K, threads](ctx, values, alt_data, alt_scales)
        for i in range(M * K):
            assert_equal(
                alt_data[i],
                data[i],
                String("data differs at ")
                + String(i)
                + " for num_max_threads="
                + String(threads),
            )
        for i in range(M * scale_K):
            assert_equal(
                alt_scales[i],
                scales[i],
                String("scale differs at ")
                + String(i)
                + " for num_max_threads="
                + String(threads),
            )


def test_mxfp4_geometry_invariant[M: Int, K: Int](ctx: DeviceContext) raises:
    """The launcher is shared, so MXFP4 must not move either."""
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, MXFP4_SF_VECTOR_SIZE)
    print("  mxfp4 geometry invariance M=", M, " K=", K)

    var values = List[Float32]()
    _fill_input(values, M, K)

    var input_dev = ctx.enqueue_create_buffer[.bfloat16](M * K)
    with input_dev.map_to_host() as h:
        for i in range(M * K):
            h[i] = values[i].cast[.bfloat16]()

    var reference = List[UInt8]()
    comptime for threads in [512, 256, 64]:
        var data_dev = ctx.enqueue_create_buffer[.uint8](M * packed_K)
        var scales_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](M * scale_K)
        var input_tt = TileTensor(input_dev, row_major((Idx[M], Idx[K])))
        var data_tt = TileTensor(data_dev, row_major((Idx[M], Idx[packed_K])))
        var scales_tt = TileTensor(
            scales_dev, row_major((Idx[M], Idx[scale_K]))
        )
        quantize_mx_amd[num_max_threads=threads](
            ctx, data_tt, scales_tt, input_tt
        )

        var data_host = ctx.enqueue_create_host_buffer[.uint8](M * packed_K)
        var scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
            M * scale_K
        )
        ctx.enqueue_copy(data_host, data_dev)
        ctx.enqueue_copy(scales_host, scales_dev)
        ctx.synchronize()

        var flat = List[UInt8]()
        for i in range(M * packed_K):
            flat.append(data_host[i])
        for i in range(M * scale_K):
            flat.append(bitcast[.uint8](scales_host[i]))

        if len(reference) == 0:
            reference = flat^
        else:
            for i in range(len(reference)):
                assert_equal(
                    flat[i],
                    reference[i],
                    String("mxfp4 output differs at ")
                    + String(i)
                    + " for num_max_threads="
                    + String(threads),
                )


def main() raises:
    with DeviceContext() as ctx:
        print("test_mxfp8_geometry_invariant:")
        test_mxfp8_geometry_invariant[144, 6144](ctx)
        test_mxfp8_geometry_invariant[16, 6144](ctx)
        test_mxfp8_geometry_invariant[33, 6208](ctx)

        print("test_mxfp8_matches_host_oracle:")
        # MiniMax-M3 decode: 72 x (num_speculative_tokens + 1) rows, at the
        # three activation widths.
        test_mxfp8_matches_host_oracle[144, 6144](ctx)
        test_mxfp8_matches_host_oracle[144, 3072](ctx)
        test_mxfp8_matches_host_oracle[144, 4096](ctx)
        # Fewer rows than CUs, then more rows than the grid holds so the row
        # loop strides.
        test_mxfp8_matches_host_oracle[16, 6144](ctx)
        test_mxfp8_matches_host_oracle[2048, 3072](ctx)
        # An odd row count with a non-round (still 32-aligned, as the launcher
        # requires) column count, then the smallest legal shape.
        test_mxfp8_matches_host_oracle[33, 6208](ctx)
        test_mxfp8_matches_host_oracle[1, 32](ctx)

        print("test_mxfp4_geometry_invariant:")
        test_mxfp4_geometry_invariant[144, 6144](ctx)
        test_mxfp4_geometry_invariant[7, 7168](ctx)

        print("PASS")
