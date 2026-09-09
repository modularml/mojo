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

from std.math import ceildiv

from max.gpu.host import DeviceContext
from internal_utils import assert_almost_equal
from std.random import rand
from linalg.matmul.vendor.blas import matmul
from _cublas.cublaslt import cublasLtGetVersion
from layout import TileTensor, Coord, CoordLike, Idx, row_major
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
)
from linalg.block_scaled_quantization import naive_block_scaled_matmul
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind


def test_block_scaled_nvfp4_cublaslt[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    out_dtype: DType,
    in_dtype: DType,
    transpose_b: Bool,
](
    ctx: DeviceContext,
    m: MType,
    n: NType,
    k: KType,
    tensor_sf: Float32 = 1.0,
) raises:
    comptime assert (
        transpose_b == True
    ), "Only transpose_b = True is supported for scaled NVFP4 matmul"

    comptime assert in_dtype == .float4_e2m1fn and out_dtype in (
        DType.float32,
        DType.bfloat16,
    ), (
        "Only float4-e2m1fn input type and float32 or bfloat16 output type"
        " are supported for NVFP4."
    )

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

    comptime scales_dtype = NVFP4_SF_DTYPE
    # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
    # Replace this with float4-e2m1fn when GENAI-337 is fixed.
    comptime input_dtype = DType.uint8

    var a_shape = row_major(Coord(m, Idx[KType.static_value // 2]))
    var b_shape = row_major(
        Coord(Idx[NType.static_value], Idx[KType.static_value // 2])
    )
    var c_shape = row_major(Coord(m, n))
    var a_scales_shape = row_major(
        Coord(
            Int(ceildiv(M, SF_MN_GROUP_SIZE)),
            Idx[ceildiv(KType.static_value, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var b_scales_shape = row_major(
        Coord(
            Idx[ceildiv(NType.static_value, SF_MN_GROUP_SIZE)],
            Idx[ceildiv(KType.static_value, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )

    var a_scales_size = (
        ceildiv(M, SF_MN_GROUP_SIZE)
        * ceildiv(K, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )
    var b_scales_size = (
        ceildiv(N, SF_MN_GROUP_SIZE)
        * ceildiv(K, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](
        a_scales_size
    )
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](
        b_scales_size
    )
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)

    rand(a_scales_host._storage, a_scales_host.num_elements())
    rand(b_scales_host._storage, b_scales_host.num_elements())

    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](a_scales_size)
    var a_scales_device_nd = TileTensor(a_scales_device, a_scales_shape)
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](b_scales_size)
    var b_scales_device_nd = TileTensor(b_scales_device, b_scales_shape)

    var a_size = M * (K // 2)
    var b_size = N * (K // 2)
    var c_size = M * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[input_dtype](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = ctx.enqueue_create_host_buffer[input_dtype](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_ptr = ctx.enqueue_create_host_buffer[out_dtype](c_size)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[out_dtype](c_size)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)

    var a_device = ctx.enqueue_create_buffer[input_dtype](a_size)
    var a_device_nd = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[input_dtype](b_size)
    var b_device_nd = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[out_dtype](c_size)
    var c_device_nd = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[out_dtype](c_size)
    var c_device_ref_nd = TileTensor(c_device_ref, c_shape)

    rand(a_host._storage, a_host.num_elements(), min=0, max=255)
    rand(b_host._storage, b_host.num_elements(), min=0, max=255)

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    matmul[scales_type=scales_dtype,](
        ctx,
        c_device_nd,
        a_device_nd,
        b_device_nd,
        a_scales=a_scales_device_nd,
        b_scales=b_scales_device_nd,
        transpose_b=True,
        c_row_major=True,
    )

    var c_ref = c_device_ref_nd.to_layout_tensor()
    var a_lt = a_device_nd.to_layout_tensor()
    var b_lt = b_device_nd.to_layout_tensor()
    var a_scales_lt = a_scales_device_nd.to_layout_tensor()
    var b_scales_lt = b_scales_device_nd.to_layout_tensor()
    naive_block_scaled_matmul[
        scaling_kind=UMMAKind.KIND_MXF4NVF4,
        SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
        transpose_b=transpose_b,
    ](
        c_ref,
        a_lt,
        b_lt,
        a_scales_lt,
        b_scales_lt,
        ctx,
    )

    ctx.enqueue_copy(c_host._storage, c_device)
    ctx.enqueue_copy(c_host_ref._storage, c_device_ref)

    ctx.synchronize()

    assert_almost_equal(
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        atol=0.01,
        rtol=0.01,
    )


def main() raises:
    with DeviceContext() as ctx:
        test_block_scaled_nvfp4_cublaslt[
            .bfloat16,
            .float4_e2m1fn,
            True,
        ](ctx, Int(128), Idx[128], Idx[64])

        test_block_scaled_nvfp4_cublaslt[
            .bfloat16,
            .float4_e2m1fn,
            True,
        ](ctx, Int(256), Idx[256], Idx[64 - 32])

        test_block_scaled_nvfp4_cublaslt[
            .bfloat16,
            .float4_e2m1fn,
            True,
        ](ctx, Int(128), Idx[3 * 128], Idx[256 + 32])

        test_block_scaled_nvfp4_cublaslt[
            .bfloat16,
            .float4_e2m1fn,
            True,
        ](ctx, Int(3 * 128), Idx[128], Idx[3 * 64])

        test_block_scaled_nvfp4_cublaslt[
            .bfloat16,
            .float4_e2m1fn,
            True,
        ](ctx, Int(2560), Idx[4096], Idx[1024])

        test_block_scaled_nvfp4_cublaslt[
            .bfloat16,
            .float4_e2m1fn,
            True,
        ](ctx, Int(1000), Idx[4096], Idx[1024])

        test_block_scaled_nvfp4_cublaslt[
            .bfloat16,
            .float4_e2m1fn,
            True,
        ](ctx, Int(1000), Idx[4096 + 64], Idx[1024])

        test_block_scaled_nvfp4_cublaslt[
            .bfloat16,
            .float4_e2m1fn,
            True,
        ](ctx, Int(1000), Idx[4096 + 64], Idx[1024 + 64])
