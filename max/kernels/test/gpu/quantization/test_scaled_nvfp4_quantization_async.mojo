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
from std.testing import assert_equal
from layout import CoordLike, Coord, Idx, TileTensor, row_major
from layout._fillers import random
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_VECTOR_SIZE,
    cast_uint_to_fp4e2m1,
)
from linalg.block_scaled_quantization import (
    quantize_dynamic_scaled_fp4fp8,
    quantize_dynamic_scaled_fp4_async,
)


def test_nvfp4_quantization[
    BType: CoordLike,
    MType: CoordLike,
    NType: CoordLike,
    //,
    dtype: DType,
    scales_dtype: DType,
    SF_VECTOR_SIZE: Int,
](
    ctx: DeviceContext,
    batch: BType,
    m: MType,
    n: NType,
    tensor_sf: Float32,
) raises:
    comptime out_dtype = DType.uint8

    var _B = Int(batch.value())
    var M = Int(m.value())
    var N = Int(n.value())

    var input_shape = Coord(m, n)
    comptime output_n = ceildiv(NType.static_value, 2)
    var output_shape = Coord(m, Idx[output_n])

    var host_ptr = alloc[Scalar[dtype]](M * N)
    var host_tensor = TileTensor(host_ptr, row_major(input_shape))

    var host_ptr_output = alloc[Scalar[out_dtype]](M * ceildiv(N, 2))
    var host_ptr_output_ref = alloc[Scalar[out_dtype]](M * ceildiv(N, 2))

    var device_buffer = ctx.enqueue_create_buffer[dtype](M * N)

    var device_buffer_output = ctx.enqueue_create_buffer[out_dtype](
        M * ceildiv(N, 2)
    )

    var device_buffer_output_ref = ctx.enqueue_create_buffer[out_dtype](
        M * ceildiv(N, 2)
    )

    random(host_tensor, min=-1.0, max=1.0)

    ctx.enqueue_copy(device_buffer, host_ptr)

    var scales_shape = Coord(
        ceildiv(M, SF_MN_GROUP_SIZE),
        ceildiv(N, SF_VECTOR_SIZE * SF_ATOM_K),
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )

    var scales_total = (
        ceildiv(M, SF_MN_GROUP_SIZE)
        * ceildiv(N, SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )

    var scales_host_ptr = alloc[Scalar[scales_dtype]](scales_total)
    var scales_host_ptr_ref = alloc[Scalar[scales_dtype]](scales_total)

    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_total)
    var scales_device_ref = ctx.enqueue_create_buffer[scales_dtype](
        scales_total
    )

    var input_tensor = TileTensor(device_buffer, row_major(input_shape))
    var output_tensor = TileTensor(
        device_buffer_output, row_major(output_shape)
    )
    var scales_tensor = TileTensor(scales_device, row_major(scales_shape))
    var output_tensor_ref = TileTensor(
        device_buffer_output_ref, row_major(output_shape)
    )
    var scales_tensor_ref = TileTensor(
        scales_device_ref, row_major(scales_shape)
    )

    quantize_dynamic_scaled_fp4_async[SF_VECTOR_SIZE=SF_VECTOR_SIZE,](
        ctx,
        output_tensor,
        scales_tensor,
        input_tensor,
        tensor_sf,
    )

    quantize_dynamic_scaled_fp4fp8[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
        ctx,
        output_tensor_ref,
        scales_tensor_ref,
        input_tensor,
        num_cols=N,
        num_cols_padded=N,
        tensor_sf=tensor_sf,
    )

    ctx.enqueue_copy(host_ptr_output, device_buffer_output)
    ctx.enqueue_copy(host_ptr_output_ref, device_buffer_output_ref)

    ctx.enqueue_copy(scales_host_ptr, scales_device)
    ctx.enqueue_copy(scales_host_ptr_ref, scales_device_ref)

    ctx.synchronize()

    var scales_tensor_host_ref = TileTensor(
        scales_host_ptr_ref, row_major(scales_shape)
    )
    var scales_tensor_host = TileTensor(
        scales_host_ptr, row_major(scales_shape)
    )

    # check scalers
    for i in range(ceildiv(M, SF_MN_GROUP_SIZE)):
        for j in range(ceildiv(N, SF_VECTOR_SIZE * SF_ATOM_K)):
            for k in range(SF_ATOM_M[0]):
                for l in range(SF_ATOM_M[1]):
                    for m in range(SF_ATOM_K):
                        var coord = Coord(i, j, k, l, m)
                        assert_equal(
                            scales_tensor_host_ref[coord].cast[.float64](),
                            scales_tensor_host[coord].cast[.float64](),
                        )

    var output_tensor_host_ref = TileTensor(
        host_ptr_output_ref, row_major(output_shape)
    )
    var output_tensor_host = TileTensor(
        host_ptr_output, row_major(output_shape)
    )

    # check output
    for row_idx in range(0, M):
        for col_idx in range(0, N // 2, SF_VECTOR_SIZE // 2):
            comptime assert output_tensor_host.flat_rank >= 2
            var output_vector = output_tensor_host.load[
                width=SF_VECTOR_SIZE // 2
            ](Coord(row_idx, col_idx))
            var output_vector_ref = output_tensor_host_ref.load[
                width=SF_VECTOR_SIZE // 2
            ](Coord(row_idx, col_idx))

            var output_fp32 = cast_uint_to_fp4e2m1[
                out_dtype=DType.float32,
                out_width=SF_VECTOR_SIZE,
            ](output_vector)
            var output_fp32_ref = cast_uint_to_fp4e2m1[
                out_dtype=DType.float32,
                out_width=SF_VECTOR_SIZE,
            ](output_vector_ref)

            assert_equal(
                output_fp32,
                output_fp32_ref,
            )

    host_ptr.free()
    host_ptr_output.free()
    scales_host_ptr.free()


def main() raises:
    with DeviceContext() as ctx:
        test_nvfp4_quantization[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
        ](
            ctx,
            Idx[1],
            2 * 128,
            Idx[11 * 64],
            tensor_sf=1.0,
        )
        test_nvfp4_quantization[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
        ](
            ctx,
            Idx[1],
            Idx[999],
            Idx[576],
            tensor_sf=1.0,
        )
        test_nvfp4_quantization[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
        ](
            ctx,
            Idx[1],
            Idx[129],
            Idx[23 * 128],
            tensor_sf=1.0,
        )
        test_nvfp4_quantization[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
        ](
            ctx,
            Idx[1],
            27 * 128,
            Idx[23 * 128],
            tensor_sf=0.43,
        )
        test_nvfp4_quantization[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
        ](
            ctx,
            Idx[1],
            Idx[13],
            Idx[17 * 128],
            tensor_sf=0.5,
        )
