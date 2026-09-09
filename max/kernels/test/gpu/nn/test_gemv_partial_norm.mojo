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
# Test: gemv_and_partial_norm (M=1, bf16) on B200.
#
# Validates both the `fused=True` single-kernel path and the `fused=False`
# 2-launch baseline (matmul + RMS-norm; unnormed tail is a view into the
# matmul output) against a vendor BLAS + host-side partial RMS-norm
# reference.
# ===----------------------------------------------------------------------=== #

from std.math import rsqrt, sqrt
from std.memory import alloc
from std.random import rand

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from layout import TileTensor, Coord, CoordLike, Idx, row_major

from internal_utils import assert_almost_equal
from nn.gemv_partial_norm import gemv_and_partial_norm


def _host_reference[
    c_type: DType, a_type: DType
](
    y_ref_ptr: ImmPointer[Scalar[c_type], _],
    gamma_ptr: ImmPointer[Scalar[a_type], _],
    normed_ref: MutPointer[Scalar[c_type], _],
    unnormed_ref: MutPointer[Scalar[c_type], _],
    n: Int,
    n_normed: Int,
    eps: Float32,
):
    """Reference partial RMS norm on a [1, n] row."""
    var n_unnormed = n - n_normed
    var sumsq: Float64 = 0.0
    for i in range(n_normed):
        var v = y_ref_ptr[i].cast[.float64]()
        sumsq += v * v
    var mean_sq = sumsq / Float64(n_normed)
    var norm_factor = Float64(1) / sqrt(mean_sq + eps.cast[.float64]())

    for i in range(n_normed):
        var v = y_ref_ptr[i].cast[.float64]()
        var g = gamma_ptr[i].cast[.float64]()
        normed_ref[i] = (v * norm_factor * g).cast[c_type]()

    for i in range(n_unnormed):
        unnormed_ref[i] = y_ref_ptr[n_normed + i]


def test_gemv_partial_norm[
    NType: CoordLike,
    KType: CoordLike,
    NNormedType: CoordLike,
    //,
    c_type: DType,
    a_type: DType,
    *,
    fused: Bool,
    transpose_b: Bool = True,
](ctx: DeviceContext, n: NType, k: KType, n_normed: NNormedType,) raises:
    var M = 1
    var N = Int(n.value())
    var K = Int(k.value())
    var N_NORMED = Int(n_normed.value())
    var N_UNNORMED = N - N_NORMED

    print(
        t"gemv_and_partial_norm: fused={fused} dtype={a_type}"
        t" shape=(M=1, N={N}, K={K}, N_normed={N_NORMED})"
    )

    comptime ak_shape = row_major(Coord(Idx[1], Idx[KType.static_value]))
    comptime b_shape = row_major(
        Coord(Idx[NType.static_value], Idx[KType.static_value])
    )
    comptime c_shape = row_major(Coord(Idx[1], Idx[NType.static_value]))
    comptime normed_shape = row_major(
        Coord(Idx[1], Idx[NNormedType.static_value])
    )
    var unnormed_shape = row_major(Coord(Idx[1], N_UNNORMED))
    comptime gamma_shape = row_major(Idx[NNormedType.static_value])

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](M * K)
    var b_host_ptr = ctx.enqueue_create_host_buffer[a_type](N * K)
    var gamma_host_ptr = ctx.enqueue_create_host_buffer[a_type](N_NORMED)
    var y_ref_host_ptr = ctx.enqueue_create_host_buffer[c_type](M * N)
    var normed_ref_ptr = ctx.enqueue_create_host_buffer[c_type](M * N_NORMED)
    var unnormed_ref_ptr = ctx.enqueue_create_host_buffer[c_type](
        M * N_UNNORMED
    )
    var normed_ours_ptr = ctx.enqueue_create_host_buffer[c_type](M * N_NORMED)
    var unnormed_ours_ptr = ctx.enqueue_create_host_buffer[c_type](
        M * N_UNNORMED
    )

    rand(a_host_ptr.as_span())
    rand(b_host_ptr.as_span())
    for i in range(N_NORMED):
        gamma_host_ptr[i] = (
            Float64(0.75) + Float64(i) / Float64(N_NORMED) * Float64(0.5)
        ).cast[a_type]()

    var a_dev = ctx.enqueue_create_buffer[a_type](M * K)
    var a_tensor = TileTensor(a_dev, ak_shape)
    var b_dev = ctx.enqueue_create_buffer[a_type](N * K)
    var b_tensor = TileTensor(b_dev, b_shape)
    var gamma_dev = ctx.enqueue_create_buffer[a_type](N_NORMED)
    var gamma_tensor = TileTensor(gamma_dev, gamma_shape)

    var y_ref_dev = ctx.enqueue_create_buffer[c_type](M * N)
    var y_ref_tensor = TileTensor(y_ref_dev, c_shape)

    var normed_dev = ctx.enqueue_create_buffer[c_type](M * N_NORMED)
    var normed_tensor = TileTensor(normed_dev, normed_shape)
    var unnormed_dev = ctx.enqueue_create_buffer[c_type](M * N_UNNORMED)
    var unnormed_tensor = TileTensor(unnormed_dev, unnormed_shape)

    ctx.enqueue_copy(a_dev, a_host_ptr)
    ctx.enqueue_copy(b_dev, b_host_ptr)
    ctx.enqueue_copy(gamma_dev, gamma_host_ptr)

    var eps = Float32(0.001)

    vendor_blas.matmul(
        ctx,
        y_ref_tensor.to_layout_tensor(),
        a_tensor.to_layout_tensor(),
        b_tensor.to_layout_tensor(),
        c_row_major=True,
        transpose_b=transpose_b,
    )

    gemv_and_partial_norm[
        transpose_b=transpose_b,
        fused=fused,
    ](
        normed_tensor,
        unnormed_tensor,
        a_tensor,
        b_tensor,
        gamma_tensor,
        eps,
        ctx,
    )

    ctx.enqueue_copy(y_ref_host_ptr, y_ref_dev)
    ctx.enqueue_copy(normed_ours_ptr, normed_dev)
    ctx.enqueue_copy(unnormed_ours_ptr, unnormed_dev)
    ctx.synchronize()

    _host_reference[c_type, a_type](
        y_ref_host_ptr.unsafe_ptr(),
        gamma_host_ptr.unsafe_ptr(),
        normed_ref_ptr.unsafe_ptr(),
        unnormed_ref_ptr.unsafe_ptr(),
        N,
        N_NORMED,
        eps,
    )

    assert_almost_equal(
        normed_ours_ptr.unsafe_ptr(),
        normed_ref_ptr.unsafe_ptr(),
        M * N_NORMED,
        atol=5e-2,
        rtol=5e-2,
    )
    # The unfused path does not populate `unnormed_output`: the
    # unnormed tail is a view into the matmul scratch. Only check
    # this output for the fused path.
    comptime if fused:
        assert_almost_equal(
            unnormed_ours_ptr.unsafe_ptr(),
            unnormed_ref_ptr.unsafe_ptr(),
            M * N_UNNORMED,
            atol=1e-2,
            rtol=1e-2,
        )
    print("\n=== TEST PASSED ===\n")

    _ = a_dev^
    _ = b_dev^
    _ = gamma_dev^
    _ = y_ref_dev^
    _ = normed_dev^
    _ = unnormed_dev^


def main() raises:
    with DeviceContext() as ctx:
        # Primary shape: N=2112, K=7168, N_normed=1536.
        test_gemv_partial_norm[.bfloat16, DType.bfloat16, fused=False](
            ctx, Idx[2112], Idx[7168], Idx[1536]
        )
        test_gemv_partial_norm[.bfloat16, DType.bfloat16, fused=True](
            ctx, Idx[2112], Idx[7168], Idx[1536]
        )

        # Smaller shape exercising the same path.
        test_gemv_partial_norm[.bfloat16, DType.bfloat16, fused=False](
            ctx, Idx[512], Idx[1024], Idx[256]
        )
        test_gemv_partial_norm[.bfloat16, DType.bfloat16, fused=True](
            ctx, Idx[512], Idx[1024], Idx[256]
        )
