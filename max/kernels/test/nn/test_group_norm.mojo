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
from nn.normalization import *
from std.testing import assert_almost_equal

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


def run_group_norm_cpu[
    dtype: DType, rank: Int
](
    shape: IndexList[rank],
    num_groups: Int,
    rtol: Float64 = 1e-4,
    atol: Float64 = 1e-5,
) raises:
    print("== run_group_norm_cpu")

    var N = shape[0]
    var C = shape[1]
    var spatial = shape.flattened_length() // (N * C)
    var group_size = C // num_groups * spatial
    var rows = N * num_groups
    var cols = group_size

    var total = shape.flattened_length()
    var data_ptr = List(length=total, fill=Scalar[dtype](0))
    var output_ptr = List(length=total, fill=Scalar[dtype](0))
    var gamma_ptr = List(length=C, fill=Scalar[dtype](0))
    var beta_ptr = List(length=C, fill=Scalar[dtype](0))

    for i in range(total):
        data_ptr[i] = Scalar[dtype](i % 256)  # bounded range to avoid overflow

    for i in range(C):
        gamma_ptr[i] = (Float64(i + C) / Float64(C)).cast[dtype]()
        beta_ptr[i] = (Float64(i) / Float64(C)).cast[dtype]()

    var param_shape = Index(C)
    var data_buf = TileTensor(data_ptr, row_major(Coord(shape)))
    var output_buf = TileTensor(output_ptr, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_ptr, row_major(Coord(param_shape)))
    var beta = TileTensor(beta_ptr, row_major(Coord(param_shape)))
    var epsilon = Float32(1e-5)

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

    group_norm_cpu[
        dtype=dtype,
        rank=rank,
        input_fn=input_fn,
        gamma_fn=gamma_scalar_fn,
        beta_fn=beta_scalar_fn,
    ](Coord(shape), epsilon, output_buf, num_groups)

    var data_ptr_ptr: MutPointer[
        data_ptr.T, origin_of(data_ptr)
    ] = data_ptr.unsafe_ptr()
    for r in range(rows):
        var vec = TileTensor(
            data_ptr_ptr + r * cols,
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
            # Compute reference in float64 for accuracy, then cast to dtype
            # for comparison (matches the kernel's higher-precision accum).
            var val = Scalar[dtype](
                (Float64(vec[c][0]) - mean_ref)
                * norm_factor
                * Float64(gamma_ptr[true_c])
                + Float64(beta_ptr[true_c])
            )
            assert_almost_equal(val, output_ptr[idx], rtol=rtol, atol=atol)


def main() raises:
    # Basic shapes, rank 3 and 4.
    run_group_norm_cpu[.float32](Index(2, 8, 2, 2), num_groups=4)
    run_group_norm_cpu[.float32](Index(2, 8, 4), num_groups=4)
    run_group_norm_cpu[.float32](Index(2, 32, 2, 2), num_groups=8)
    run_group_norm_cpu[.float32](Index(2, 32, 4), num_groups=8)

    # Group boundary not aligned with spatial-simd boundaries (only matters
    # for the GPU kernel's SIMD path; exercised here for CPU coverage too).
    run_group_norm_cpu[.float32](Index(2, 32, 1, 10), num_groups=8)

    # Larger column counts.
    run_group_norm_cpu[.float32](Index(1, 128, 1, 64), num_groups=8)
    run_group_norm_cpu[.float32](Index(1, 128, 64), num_groups=8)
    run_group_norm_cpu[.float32](Index(1, 64, 1, 64), num_groups=8)
    run_group_norm_cpu[.float32](Index(1, 64, 64), num_groups=8)

    # Trivial spatial=1 (all channels collapsed to one dimension).
    run_group_norm_cpu[.float32](Index(2, 128, 1, 1), num_groups=1)
    run_group_norm_cpu[.float32](Index(2, 128, 1), num_groups=1)

    # Odd column counts (no SIMD-alignment requirement on CPU).
    run_group_norm_cpu[.float32](Index(2, 33, 1, 1), num_groups=1)
    run_group_norm_cpu[.float32](Index(2, 33, 1), num_groups=1)

    # One-channel, one-group (channels_per_group=1).
    run_group_norm_cpu[.float32](Index(2, 1, 4, 8), num_groups=1)
    run_group_norm_cpu[.float32](Index(2, 1, 32), num_groups=1)

    # Edge case from group norm layer tests.
    run_group_norm_cpu[.float32](Index(2, 2, 4, 4), num_groups=1)
    run_group_norm_cpu[.float32](Index(2, 2, 16), num_groups=1)

    # Zero-spatial input: the FLUX.2 VAE encoder is invoked unconditionally
    # on a ``(0, 0, 3)`` placeholder image for text-to-image, and every
    # ``GroupNorm`` inside the encoder sees a ``(B, C, 0, 0)`` tensor. The
    # kernel must early-return rather than dividing by a zero group size.
    run_group_norm_cpu[.float32](Index(1, 128, 0, 0), num_groups=32)
    run_group_norm_cpu[.bfloat16](Index(1, 128, 0, 0), num_groups=32)

    # bfloat16 coverage, larger group sizes. Tolerances are relaxed because
    # the test reference uses a two-pass variance formula (E[X^2]-E[X]^2)
    # while the kernel uses Welford, which diverge more at bfloat16
    # precision (mirrors the GPU test's multi-block tolerance choice).
    run_group_norm_cpu[.bfloat16](
        Index(1, 128, 8, 8), num_groups=32, rtol=1e-2, atol=1e-2
    )
    run_group_norm_cpu[.bfloat16](
        Index(2, 256, 4, 4), num_groups=32, rtol=1e-2, atol=1e-2
    )
    run_group_norm_cpu[.bfloat16](
        Index(1, 128, 64), num_groups=32, rtol=1e-2, atol=1e-2
    )
