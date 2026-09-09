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
"""Dispatch coverage test for the high-level MXFP8 `block_scaled_matmul`.

`block_scaled_matmul` (the entry point behind the
`mo.matmul.dynamic.block.scaled` op) routes every MXFP8 shape through the Mojo
`heuristic_and_outliers_dispatch`. That heuristic only enumerates configs for
M up to 8192, so a large-M shape such as the 30k-token prefill in KERN-3024
(M=30000, N=128, K=6144) produced a config the dispatcher never built and used
to hard-error with "heuristic dispatch found no config for this shape (the
vendor fallback is NVFP4-only)."

This test exercises both regimes against a `naive_block_scaled_matmul`
reference:
  * a heuristic-covered shape (M <= 8192) that hits the Mojo kernel, and
  * the KERN-3024 large-M shape that must now fall back to the vendor
    (cuBLASLt) MXFP8 block-scaled matmul.
"""

from std.math import ceildiv, align_up
from std.random import rand, random_ui64
from max.gpu.host import DeviceContext
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind

from internal_utils import assert_almost_equal
from layout import CoordLike, Coord, Idx, TileTensor, row_major

from linalg.block_scaled_quantization import (
    block_scaled_matmul,
    naive_block_scaled_matmul,
)
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    MXFP8_SF_DTYPE,
    set_scale_factor,
)


def test_mxfp8_dispatch[
    NType: CoordLike,
    KType: CoordLike,
    //,
    input_type: DType,
    output_type: DType,
](ctx: DeviceContext, m: Int, n: NType, k: KType) raises:
    """Run the high-level `block_scaled_matmul` for an MXFP8 shape and compare
    against the naive reference. N and K are static; M is dynamic (matching the
    dynamic token dimension of the failing op)."""
    var M = m
    var N = Int(n.value())
    var K = Int(k.value())

    comptime scales_type = MXFP8_SF_DTYPE
    comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE

    print(
        t"in/out dtypes=({input_type}, {output_type}, {scales_type})  problem"
        t" shape=(M={M}, N={N}, K={K})"
    )

    var a_shape = row_major(Coord(M, Idx[KType.static_value]))
    var b_shape = row_major(
        Coord(Idx[NType.static_value], Idx[KType.static_value])
    )
    var c_shape = row_major(Coord(M, Idx[NType.static_value]))

    var a_scales_shape = row_major(
        Coord(
            ceildiv(M, SF_MN_GROUP_SIZE),
            Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var b_scales_shape = row_major(
        Coord(
            Idx[ceildiv(NType.static_value, SF_MN_GROUP_SIZE)],
            Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )

    var a_size = M * K
    var b_size = N * K
    var c_size = M * N
    var a_scales_size = a_scales_shape.product()
    var b_scales_size = b_scales_shape.product()

    var a_host_ptr = ctx.enqueue_create_host_buffer[input_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[input_type](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[output_type](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[output_type](c_size)

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        a_scales_size
    )
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        b_scales_size
    )

    var a_device = ctx.enqueue_create_buffer[input_type](a_size)
    var b_device = ctx.enqueue_create_buffer[input_type](b_size)
    var c_device = ctx.enqueue_create_buffer[output_type](c_size)
    var c_device_ref = ctx.enqueue_create_buffer[output_type](c_size)
    var a_scales_device = ctx.enqueue_create_buffer[scales_type](a_scales_size)
    var b_scales_device = ctx.enqueue_create_buffer[scales_type](b_scales_size)

    rand(a_host_ptr.unsafe_ptr(), a_size)
    rand(b_host_ptr.unsafe_ptr(), b_size)

    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    # NOTE: unused scales must be 0.0 or we hit accuracy issues.
    for idx0 in range(align_up(M, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0, align_up(K, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            if idx0 < M and idx1 < K:
                var scale_value = (
                    (1 << random_ui64(0, 3))
                    .cast[.float32]()
                    .cast[scales_type]()
                )
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_host, idx0, idx1, scale_value
                )
            else:
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_host, idx0, idx1, Scalar[scales_type](0.0)
                )

    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)
    for idx0 in range(align_up(N, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0, align_up(K, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            if idx0 < N and idx1 < K:
                var scale_value = (
                    (1 << random_ui64(0, 3))
                    .cast[.float32]()
                    .cast[scales_type]()
                )
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    b_scales_host, idx0, idx1, scale_value
                )
            else:
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    b_scales_host, idx0, idx1, Scalar[scales_type](0.0)
                )

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    var a = TileTensor(a_device, a_shape)
    var b = TileTensor(b_device, b_shape)
    var c = TileTensor(c_device, c_shape)
    var c_ref = TileTensor(c_device_ref, c_shape)
    var a_scales = TileTensor(a_scales_device, a_scales_shape)
    var b_scales = TileTensor(b_scales_device, b_scales_shape)

    # High-level dispatch: this is the path that hard-errored for large M.
    block_scaled_matmul[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        transpose_b=True,
        target="gpu",
    ](c, a, b, a_scales, b_scales, Float32(1.0), ctx)

    naive_block_scaled_matmul[
        scaling_kind=UMMAKind.KIND_MXF8F6F4,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        transpose_b=True,
    ](
        c_ref.to_layout_tensor(),
        a.to_layout_tensor(),
        b.to_layout_tensor(),
        a_scales.to_layout_tensor(),
        b_scales.to_layout_tensor(),
        ctx,
    )

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    assert_almost_equal(
        c_host_ptr.unsafe_ptr(),
        c_host_ref_ptr.unsafe_ptr(),
        c_size,
        atol=1e-2,
        rtol=1e-2,
    )
    print("  -> PASSED")

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^


def main() raises:
    with DeviceContext() as ctx:
        # Heuristic-covered shape (M <= 8192): exercises the Mojo block-scaled
        # kernel path and guards against a regression there.
        test_mxfp8_dispatch[.float8_e4m3fn, .bfloat16](
            ctx, 4096, Idx[128], Idx[6144]
        )

        # KERN-3024: 30k-token prefill. The heuristic has no config for this
        # M, so the dispatcher must fall back to the vendor (cuBLASLt) MXFP8
        # block-scaled matmul instead of raising.
        test_mxfp8_dispatch[.float8_e4m3fn, .bfloat16](
            ctx, 30000, Idx[128], Idx[6144]
        )

    print("\n=== ALL TESTS PASSED ===\n")
