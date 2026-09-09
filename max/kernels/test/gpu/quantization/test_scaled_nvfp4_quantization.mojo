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

from max.gpu.host import DeviceContext
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    UNKNOWN_VALUE,
    lt_to_tt,
)
from layout._fillers import random
from linalg.block_scaled_quantization import quantize_dynamic_scaled_fp4fp8
from std.testing import assert_almost_equal
from std.math import ceildiv, recip
from linalg.fp4_utils import (
    cast_fp_to_fp4e2m1,
    cast_uint_to_fp4e2m1,
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    get_scale_factor,
)
from std.utils import IndexList


def test_dynamic_fp4_quant[
    in_dtype: DType,
    scales_dtype: DType,
    SF_VECTOR_SIZE: Int,
    M: Optional[Int],
    N: Optional[Int],
](ctx: DeviceContext, m: Int, n: Int, tensor_sf: Float32 = 1.0) raises:
    if N.or_else(n) % (SF_VECTOR_SIZE // 2) != 0:
        raise Error(
            "n must be a multiple of (SF_VECTOR_SIZE // 2) due to kernel"
            " constraints"
        )

    comptime out_dtype = DType.uint8

    # Input tensor layout and buffer
    comptime input_static_shape = Layout.row_major(
        M.or_else(UNKNOWN_VALUE), N.or_else(UNKNOWN_VALUE)
    )
    var input_dynamic_shape = IndexList[2](M.or_else(m), N.or_else(n))
    var input_runtime_layout = RuntimeLayout[input_static_shape].row_major(
        input_dynamic_shape
    )
    var in_device = ctx.enqueue_create_buffer[in_dtype](
        input_dynamic_shape.flattened_length()
    )
    var input_tensor = LayoutTensor[in_dtype, input_static_shape](
        in_device, input_runtime_layout
    )

    # Output tensor layout and buffer
    comptime output_static_shape = Layout.row_major(
        M.or_else(UNKNOWN_VALUE),
        ceildiv(N.or_else(UNKNOWN_VALUE), 2),
    )
    var output_dynamic_shape = IndexList[2](
        M.or_else(m), ceildiv(N.or_else(n), 2)
    )
    var output_runtime_layout = RuntimeLayout[output_static_shape].row_major(
        output_dynamic_shape
    )
    var out_device = ctx.enqueue_create_buffer[out_dtype](
        output_dynamic_shape.flattened_length()
    )
    var output_tensor = LayoutTensor[out_dtype, output_static_shape](
        out_device, output_runtime_layout
    )

    # Scales tensor layout and buffer
    var scales_shape = IndexList[5](
        ceildiv(m, SF_MN_GROUP_SIZE),
        ceildiv(n, SF_VECTOR_SIZE * SF_ATOM_K),
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )
    comptime scales_static_layout = Layout.row_major(
        UNKNOWN_VALUE,
        UNKNOWN_VALUE,
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )
    var scales_runtime_layout = RuntimeLayout[scales_static_layout].row_major(
        scales_shape
    )
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](
        scales_shape.flattened_length()
    )
    var scales_tensor = LayoutTensor[scales_dtype, scales_static_layout](
        scales_device, scales_runtime_layout
    )

    # Initialize input with random data and output with zeros on host
    with in_device.map_to_host() as in_host:
        var in_host_tensor = LayoutTensor[in_dtype, input_static_shape](
            in_host, input_runtime_layout
        )
        random(in_host_tensor)

    with out_device.map_to_host() as out_host:
        for i in range(len(out_host)):
            out_host[i] = 0

    # Run the quantization kernel
    quantize_dynamic_scaled_fp4fp8[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
        ctx,
        lt_to_tt(output_tensor).as_unsafe_any_origin(),
        lt_to_tt(scales_tensor).as_unsafe_any_origin(),
        lt_to_tt(input_tensor).as_unsafe_any_origin(),
        num_cols=n,
        num_cols_padded=n,
        tensor_sf=tensor_sf,
    )

    ctx.synchronize()

    # Verify results by reading back from device
    with in_device.map_to_host() as in_host:
        with out_device.map_to_host() as out_host:
            with scales_device.map_to_host() as scales_host:
                var input_tensor_host = LayoutTensor[
                    in_dtype, input_static_shape
                ](in_host, input_runtime_layout)
                var output_tensor_host = LayoutTensor[
                    out_dtype, output_static_shape
                ](out_host, output_runtime_layout)
                var scales_tensor_host = LayoutTensor[
                    scales_dtype, scales_static_layout
                ](scales_host, scales_runtime_layout)

                for row_idx in range(0, m):
                    for col_idx in range(0, n, SF_VECTOR_SIZE):
                        var vec_max: Float32
                        # kernel support N shapes that are multiples of (SF_VECTOR_SIZE // 2).
                        # Here we handle the oob case by loading only the first half of the SF_VECTOR_SIZE.
                        if (n % SF_VECTOR_SIZE != 0) and (
                            col_idx + SF_VECTOR_SIZE > n
                        ):
                            var input_vector = input_tensor_host.load[
                                SF_VECTOR_SIZE // 2
                            ](row_idx, col_idx)
                            vec_max = (
                                abs(input_vector).reduce_max().cast[.float32]()
                            )
                        else:
                            var input_vector = input_tensor_host.load[
                                SF_VECTOR_SIZE
                            ](row_idx, col_idx)
                            vec_max = (
                                abs(input_vector).reduce_max().cast[.float32]()
                            )

                        var sf_value = tensor_sf * (
                            vec_max * recip(Float32(6.0))
                        )
                        var ref_fp8_sf = sf_value.cast[scales_dtype]()

                        var fp8_sf = get_scale_factor[
                            SF_VECTOR_SIZE=SF_VECTOR_SIZE
                        ](
                            scales_tensor_host.as_unsafe_any_origin(),
                            row_idx,
                            col_idx,
                        )

                        # verify the scale factors
                        assert_almost_equal(
                            ref_fp8_sf.cast[.float64](),
                            fp8_sf.cast[.float64](),
                            rtol=1e-1,
                            atol=1e-1,
                        )

                        var output_scale = Float32(0.0)
                        if vec_max != 0:
                            output_scale = recip(
                                ref_fp8_sf.cast[.float32]() * recip(tensor_sf)
                            )

                        # verify the output values
                        if (n % SF_VECTOR_SIZE != 0) and (
                            col_idx + SF_VECTOR_SIZE > n
                        ):
                            var input_f32 = (
                                input_tensor_host.load[SF_VECTOR_SIZE // 2](
                                    row_idx, col_idx
                                ).cast[.float32]()
                                * output_scale
                            )
                            var ref_output_e2m1 = cast_fp_to_fp4e2m1(input_f32)
                            var output_e2m1 = cast_uint_to_fp4e2m1[
                                out_dtype=DType.float32,
                                out_width=SF_VECTOR_SIZE // 2,
                            ](
                                output_tensor_host.load[
                                    (SF_VECTOR_SIZE // 2) // 2
                                ](row_idx, col_idx // 2)
                            )
                            assert_almost_equal(
                                ref_output_e2m1,
                                output_e2m1,
                                rtol=1e0,
                                atol=1e-1,
                            )
                        else:
                            var input_f32 = (
                                input_tensor_host.load[SF_VECTOR_SIZE](
                                    row_idx, col_idx
                                ).cast[.float32]()
                                * output_scale
                            )
                            var ref_output_e2m1 = cast_fp_to_fp4e2m1(input_f32)
                            var output_e2m1 = cast_uint_to_fp4e2m1[
                                out_dtype=DType.float32,
                                out_width=SF_VECTOR_SIZE,
                            ](
                                output_tensor_host.load[(SF_VECTOR_SIZE // 2)](
                                    row_idx, col_idx // 2
                                )
                            )
                            assert_almost_equal(
                                ref_output_e2m1,
                                output_e2m1,
                                rtol=1e0,
                                atol=1e-1,
                            )


def main() raises:
    with DeviceContext() as ctx:
        # Zero-row inputs should not launch kernels.
        test_dynamic_fp4_quant[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            NVFP4_SF_VECTOR_SIZE,
            M=None,
            N=Int(128),
        ](ctx, 0, 128)
        test_dynamic_fp4_quant[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            NVFP4_SF_VECTOR_SIZE,
            M=None,
            N=Int(128),
        ](ctx, 256, 128)
        test_dynamic_fp4_quant[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            NVFP4_SF_VECTOR_SIZE,
            M=None,
            N=Int(128 + 8),
        ](ctx, 258, 128 + 8)
        test_dynamic_fp4_quant[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            NVFP4_SF_VECTOR_SIZE,
            M=None,
            N=Int(128 + 64 - 8),
        ](ctx, 258, 128 + 64 - 8)
        test_dynamic_fp4_quant[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            NVFP4_SF_VECTOR_SIZE,
            M=None,
            N=Int(8192 + 8),
        ](ctx, 1000, 8192 + 8, tensor_sf=0.43)
        test_dynamic_fp4_quant[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            NVFP4_SF_VECTOR_SIZE,
            M=None,
            N=Int(16384 + 8),
        ](ctx, 2048, 16384 + 8, tensor_sf=0.5)
