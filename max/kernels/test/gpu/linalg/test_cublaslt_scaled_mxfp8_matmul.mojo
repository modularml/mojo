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

from std.math import ceildiv, align_up
from std.random import random_ui64
from max.gpu.host import DeviceContext
from internal_utils import assert_almost_equal
from std.random import rand
from linalg.matmul.vendor.blas import matmul
from _cublas.cublaslt import cublasLtGetVersion
from layout import CoordLike, Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    MXFP8_SF_DTYPE,
    set_scale_factor,
)
from linalg.block_scaled_quantization import naive_block_scaled_matmul
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind


def test_scaled_mxfp8_cublaslt[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    input_type: DType,
    output_type: DType,
    transpose_b: Bool,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    comptime assert (
        transpose_b == True
    ), "Only transpose_b = True is supported for scaled FP8 matmul"

    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    var cublaslt_version = cublasLtGetVersion()

    if cublaslt_version < 120901:
        raise Error(
            "This test needs cublasLt version 120901 or higher",
            " cublasLt version: ",
            cublaslt_version,
        )

    comptime scales_type = MXFP8_SF_DTYPE

    print(
        t"in/out dtypes=({input_type}, {output_type}, {scales_type})  problem"
        t" shape=({M}, {N}, {K}) "
    )

    var a_shape = Coord(m, k)
    var b_shape = Coord(n, k)
    var c_shape = Coord(m, n)

    var a_scales_shape = Coord(
        ceildiv(M, SF_MN_GROUP_SIZE),
        ceildiv(K, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K),
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var b_scales_shape = Coord(
        ceildiv(N, SF_MN_GROUP_SIZE),
        ceildiv(K, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K),
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )

    var a_scales_size = (
        ceildiv(M, SF_MN_GROUP_SIZE)
        * ceildiv(K, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )
    var b_scales_size = (
        ceildiv(N, SF_MN_GROUP_SIZE)
        * ceildiv(K, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        a_scales_size
    )
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        b_scales_size
    )

    var a_scales_device = ctx.enqueue_create_buffer[scales_type](a_scales_size)
    var b_scales_device = ctx.enqueue_create_buffer[scales_type](b_scales_size)

    var a_size = M * K
    var b_size = N * K
    var c_size = M * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[input_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[input_type](b_size)
    var c_host_ptr = alloc[Scalar[output_type]](c_size)
    var c_host_ref_ptr = alloc[Scalar[output_type]](c_size)

    var a_device = ctx.enqueue_create_buffer[input_type](a_size)
    var b_device = ctx.enqueue_create_buffer[input_type](b_size)
    var c_device = ctx.enqueue_create_buffer[output_type](c_size)
    var c_device_ref = ctx.enqueue_create_buffer[output_type](c_size)

    var a_scales_host_tt = TileTensor(
        a_scales_host_ptr, row_major(a_scales_shape)
    )
    # NOTE: It is very important that we set unused scales to 0.0 otherwise we will hit accuracy issues
    for idx0 in range(align_up(M, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0,
            align_up(K, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K),
            MXFP8_SF_VECTOR_SIZE,
        ):
            if idx0 < M and idx1 < K:
                var scale_value = (
                    (1 << random_ui64(0, 3))
                    .cast[.float32]()
                    .cast[scales_type]()
                )
                set_scale_factor[SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE](
                    a_scales_host_tt, idx0, idx1, scale_value
                )
            else:
                set_scale_factor[SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE](
                    a_scales_host_tt, idx0, idx1, Scalar[scales_type](0.0)
                )
    var b_scales_host_tt = TileTensor(
        b_scales_host_ptr, row_major(b_scales_shape)
    )
    for idx0 in range(align_up(N, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0,
            align_up(K, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K),
            MXFP8_SF_VECTOR_SIZE,
        ):
            if idx0 < N and idx1 < K:
                var scale_value = (
                    (1 << random_ui64(0, 3))
                    .cast[.float32]()
                    .cast[scales_type]()
                )
                set_scale_factor[SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE](
                    b_scales_host_tt, idx0, idx1, scale_value
                )
            else:
                set_scale_factor[SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE](
                    b_scales_host_tt, idx0, idx1, Scalar[scales_type](0.0)
                )

    rand(a_host_ptr.unsafe_ptr(), a_size)
    rand(b_host_ptr.unsafe_ptr(), b_size)

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    var a = TileTensor(a_device, row_major(a_shape))
    var b = TileTensor(b_device, row_major(b_shape))
    var c = TileTensor(c_device, row_major(c_shape))
    var a_scales = TileTensor(a_scales_device, row_major(a_scales_shape))
    var b_scales = TileTensor(b_scales_device, row_major(b_scales_shape))

    matmul(
        ctx,
        c,
        a,
        b,
        a_scales=a_scales.as_immut(),
        b_scales=b_scales.as_immut(),
        transpose_b=True,
        c_row_major=True,
    )

    ctx.enqueue_copy(c_host_ptr, c_device)

    var c_ref = TileTensor(c_device_ref, row_major(c_shape))
    naive_block_scaled_matmul[
        scaling_kind=UMMAKind.KIND_MXF8F6F4,
        SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE,
        transpose_b=transpose_b,
    ](
        c_ref.to_layout_tensor(),
        a.to_layout_tensor(),
        b.to_layout_tensor(),
        a_scales.to_layout_tensor(),
        b_scales.to_layout_tensor(),
        ctx,
    )

    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)

    ctx.synchronize()

    assert_almost_equal(
        c_host_ptr,
        c_host_ref_ptr,
        c_size,
        atol=0.01,
        rtol=0.01,
    )


def main() raises:
    with DeviceContext() as ctx:
        test_scaled_mxfp8_cublaslt[
            .float8_e4m3fn,
            .bfloat16,
            True,
        ](ctx, Idx[128], Idx[128], Idx[128])

        test_scaled_mxfp8_cublaslt[
            .float8_e4m3fn,
            .bfloat16,
            True,
        ](ctx, Idx[256], Idx[256], Idx[256])

        test_scaled_mxfp8_cublaslt[
            .float8_e4m3fn,
            .bfloat16,
            True,
        ](ctx, Idx[128], Idx[3 * 128], Idx[256])

        test_scaled_mxfp8_cublaslt[
            .float8_e4m3fn,
            .bfloat16,
            True,
        ](ctx, 3 * 128, Idx[128], Idx[3 * 128])

        test_scaled_mxfp8_cublaslt[
            .float8_e4m3fn,
            .bfloat16,
            True,
        ](ctx, Idx[2560], Idx[4096], Idx[1024])

        test_scaled_mxfp8_cublaslt[
            .float8_e4m3fn,
            .bfloat16,
            True,
        ](ctx, Idx[1000], Idx[4096], Idx[1024])

        test_scaled_mxfp8_cublaslt[
            .float8_e4m3fn,
            .bfloat16,
            True,
        ](ctx, Idx[1000], Idx[4096 + 64], Idx[1024])

        test_scaled_mxfp8_cublaslt[
            .float8_e4m3fn,
            .bfloat16,
            True,
        ](ctx, Idx[1000], Idx[4096 + 64], Idx[1024 + 64])
