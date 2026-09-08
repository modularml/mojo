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

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.memory import bitcast
from std.random import rand
from std.sys.intrinsics import llvm_intrinsic
from std.utils import IndexList

from internal_utils import assert_almost_equal
from layout import (
    Coord,
    CoordLike,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    Idx,
    row_major,
)
from linalg.matmul.vendor.blas import matmul
from linalg.fp4_utils import MXFP4_SF_VECTOR_SIZE


def block_scaled_matmul_ref[
    output_dtype: DType
](
    a_ptr: ImmPointer[UInt8, ImmutAnyOrigin],
    b_ptr: ImmPointer[UInt8, ImmutAnyOrigin],
    a_scales_ptr: ImmPointer[Float8_e8m0fnu, ImmutAnyOrigin],
    b_scales_ptr: ImmPointer[Float8_e8m0fnu, ImmutAnyOrigin],
    c_ptr: MutPointer[Scalar[output_dtype], MutAnyOrigin],
    M_dev: Int32,
    N_dev: Int32,
    K_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width args.
    var M = Int(M_dev)
    var N = Int(N_dev)
    var K = Int(K_dev)

    @always_inline
    def cast_fp2em1x2_to_fp32x2[
        byte_select: Int
    ](packed: Int32, scale: Float32) -> SIMD[.float32, 2]:
        return llvm_intrinsic[
            "llvm.amdgcn.cvt.scalef32.pk.f32.fp4",
            SIMD[.float32, 2],
        ](packed, scale, Int32(byte_select))

    var m = global_idx.x
    var n = global_idx.y

    if m >= M or n >= N:
        return

    var k_groups = K // MXFP4_SF_VECTOR_SIZE

    var am_scales_ptr = a_scales_ptr + m * k_groups
    var bn_scales_ptr = b_scales_ptr + n * k_groups

    var am_ptr = a_ptr + m * (K // 2)
    var bn_ptr = b_ptr + n * (K // 2)

    var accum = SIMD[.float32, 2](0)

    for ko in range(k_groups):
        var a_scale = am_scales_ptr[ko].cast[.float32]()
        var b_scale = bn_scales_ptr[ko].cast[.float32]()

        for ki in range(0, MXFP4_SF_VECTOR_SIZE // 2, 4):
            var a_data = bitcast[.int32, 1](am_ptr.load[width=4](ki))
            var b_data = bitcast[.int32, 1](bn_ptr.load[width=4](ki))

            comptime for byte_select in range(4):
                accum += cast_fp2em1x2_to_fp32x2[byte_select](
                    a_data, a_scale
                ) * cast_fp2em1x2_to_fp32x2[byte_select](b_data, b_scale)

        am_ptr += MXFP4_SF_VECTOR_SIZE // 2
        bn_ptr += MXFP4_SF_VECTOR_SIZE // 2

    c_ptr[m * N + n] = accum.reduce_add().cast[output_dtype]()


def test_block_scaled_mxfp4_hipblaslt[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    output_dtype: DType,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    comptime assert output_dtype in (
        DType.float32,
        DType.bfloat16,
    ), "Only BF16/FP32 output type are supported for MXFP4."

    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
    # Replace this with float4-e2m1fn when GENAI-337 is fixed.
    comptime input_dtype = DType.uint8
    comptime scales_dtype = DType.float8_e8m0fnu

    comptime static_k_scales_dim = ceildiv(
        KType.static_value, MXFP4_SF_VECTOR_SIZE
    )
    var k_scales_dim = ceildiv(K, MXFP4_SF_VECTOR_SIZE)

    var a_shape = row_major(Coord(m, Idx[KType.static_value // 2]))
    var b_shape = row_major(
        Coord(Idx[NType.static_value], Idx[KType.static_value // 2])
    )
    var c_shape = row_major(Coord(m, n))
    var a_scales_shape = row_major(Coord(m, Idx[static_k_scales_dim]))
    var b_scales_shape = row_major(Coord(n, Idx[static_k_scales_dim]))

    var a_scales_size = M * k_scales_dim
    var b_scales_size = N * k_scales_dim

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
    var c_host_ptr = ctx.enqueue_create_host_buffer[output_dtype](c_size)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[output_dtype](c_size)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)

    var a_device = ctx.enqueue_create_buffer[input_dtype](a_size)
    var a_device_nd = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[input_dtype](b_size)
    var b_device_nd = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[output_dtype](c_size)
    var c_device_nd = TileTensor[mut=True](c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[output_dtype](c_size)

    rand(a_host._storage, a_host.num_elements(), min=0, max=255)
    rand(b_host._storage, b_host.num_elements(), min=0, max=255)

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    comptime a_scales_layout = Layout.row_major(
        a_scales_device_nd.static_shape[0], a_scales_device_nd.static_shape[1]
    )
    comptime b_scales_layout = Layout.row_major(
        b_scales_device_nd.static_shape[0], b_scales_device_nd.static_shape[1]
    )

    var a_scales_lt = LayoutTensor[
        scales_dtype, a_scales_layout, ImmutAnyOrigin
    ](
        rebind[ImmPointer[Scalar[scales_dtype], ImmutAnyOrigin]](
            a_scales_device_nd._storage
        ),
        RuntimeLayout[a_scales_layout].row_major(
            IndexList[2](
                Int(a_scales_device_nd.dim[0]()),
                Int(a_scales_device_nd.dim[1]()),
            )
        ),
    )
    var b_scales_lt = LayoutTensor[
        scales_dtype, b_scales_layout, ImmutAnyOrigin
    ](
        rebind[ImmPointer[Scalar[scales_dtype], ImmutAnyOrigin]](
            b_scales_device_nd._storage
        ),
        RuntimeLayout[b_scales_layout].row_major(
            IndexList[2](
                Int(b_scales_device_nd.dim[0]()),
                Int(b_scales_device_nd.dim[1]()),
            )
        ),
    )

    matmul(
        ctx,
        c_device_nd.to_layout_tensor(),
        a_device_nd.to_layout_tensor(),
        b_device_nd.to_layout_tensor(),
        a_scales=a_scales_lt,
        b_scales=b_scales_lt,
        transpose_b=True,
        c_row_major=True,
    )

    comptime BLOCK_DIM = 32

    ctx.enqueue_function[block_scaled_matmul_ref[output_dtype]](
        a_device,
        b_device,
        a_scales_device,
        b_scales_device,
        c_device_ref,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
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
        test_block_scaled_mxfp4_hipblaslt[.bfloat16](
            ctx, Idx[128], Idx[128], Idx[256]
        )
        test_block_scaled_mxfp4_hipblaslt[.bfloat16](
            ctx, Idx[64], Idx[224], Idx[512]
        )
