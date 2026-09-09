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


from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, coord_to_index_list, row_major
from layout._fillers import random
from nn.normalization import *
from std.sys import align_of
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


def run_rms_norm_fused_residual_add_gpu[
    rank: Int,
    //,
    dtype: DType,
](ctx: DeviceContext, shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var runtime_layout = row_major(Coord(shape))
    var param_shape = Index(cols)
    var param_runtime_layout = row_major(Coord(param_shape))

    # Allocate device buffers
    var data_device = ctx.enqueue_create_buffer[dtype](shape.flattened_length())
    # Separate residual buffer (a copy of `data_device`): the fused kernel's two
    # input loaders are distinct value-closure arguments and must reference
    # distinct buffer origins. The unfused reference reads `data_device` for the
    # residual, so the copy keeps residual == input numerically.
    var residual_device = ctx.enqueue_create_buffer[dtype](
        shape.flattened_length()
    )
    var unfused_intermediate_device = ctx.enqueue_create_buffer[dtype](
        shape.flattened_length()
    )
    var result_unfused_device = ctx.enqueue_create_buffer[dtype](
        shape.flattened_length()
    )
    var result_fused_device = ctx.enqueue_create_buffer[dtype](
        shape.flattened_length()
    )
    var residual_fused_output_device = ctx.enqueue_create_buffer[dtype](
        shape.flattened_length()
    )
    var gamma1_device = ctx.enqueue_create_buffer[dtype](cols)
    var gamma2_device = ctx.enqueue_create_buffer[dtype](cols)

    # Initialize input data on host
    with data_device.map_to_host() as data_host:
        var data_host_tensor = TileTensor(data_host, runtime_layout)
        random(data_host_tensor)

    # Mirror the input into the residual buffer (residual == input).
    ctx.enqueue_copy(residual_device, data_device)

    with gamma1_device.map_to_host() as gamma1_host:
        var gamma1_host_tensor = TileTensor(gamma1_host, param_runtime_layout)
        random(gamma1_host_tensor)

    with gamma2_device.map_to_host() as gamma2_host:
        var gamma2_host_tensor = TileTensor(gamma2_host, param_runtime_layout)
        random(gamma2_host_tensor)

    # Initialize output buffers with zeros
    with unfused_intermediate_device.map_to_host() as host:
        for i in range(len(host)):
            host[i] = 0

    with result_unfused_device.map_to_host() as host:
        for i in range(len(host)):
            host[i] = 0

    with result_fused_device.map_to_host() as host:
        for i in range(len(host)):
            host[i] = 0

    with residual_fused_output_device.map_to_host() as host:
        for i in range(len(host)):
            host[i] = 0

    # Create device layout tensors
    var data_buf = TileTensor(data_device, runtime_layout)
    var data_buf_res = TileTensor(residual_device, runtime_layout)
    var gamma1 = TileTensor(gamma1_device, param_runtime_layout)
    var gamma2 = TileTensor(gamma2_device, param_runtime_layout)
    var result_fused_buf = TileTensor(result_fused_device, runtime_layout)
    var result_unfused_buf = TileTensor(result_unfused_device, runtime_layout)
    var unfused_intermediate_buf = TileTensor(
        unfused_intermediate_device, runtime_layout
    )
    var residual_fused_output_buf = TileTensor(
        residual_fused_output_device, runtime_layout
    )

    var epsilon1 = Float32(0.001)
    var epsilon2 = Float32(0.002)
    var weight_offset1 = Scalar[dtype](0.0)
    var weight_offset2 = Scalar[dtype](0.0)

    # Test fused operation
    @always_inline
    def input_fn[
        width: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        var idx = data_buf.layout(coords)
        return data_buf.raw_load[width=width](idx)

    @always_inline
    def residual_input_fn[
        width: Int
    ](coords: Coord) {var data_buf_res} -> SIMD[dtype, width]:
        var idx = data_buf_res.layout(coords)
        return data_buf_res.raw_load[width=width](idx)

    @always_inline
    def fused_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var result_fused_buf} -> None:
        var idx = result_fused_buf.layout(coords)
        result_fused_buf.raw_store[
            width=width, alignment=alignment * align_of[dtype]()
        ](idx, val)

    @always_inline
    def fused_residual_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {
        var residual_fused_output_buf
    } -> None:
        var idx = residual_fused_output_buf.layout(coords)
        residual_fused_output_buf.raw_store[
            width=width, alignment=alignment * align_of[dtype]()
        ](idx, val)

    # Call fused kernel
    rms_norm_fused_residual_add[
        dtype,
        rank,
        target="gpu",
        multiply_before_cast=True,
    ](
        input_fn,
        residual_input_fn,
        fused_output_fn,
        fused_residual_output_fn,
        Coord(shape),
        Int(cols),
        gamma1,
        epsilon1.cast[dtype](),
        weight_offset1,
        gamma2,
        epsilon2.cast[dtype](),
        weight_offset2,
        ctx,
    )

    # Test unfused operations for comparison
    @always_inline
    @__copy_capture(unfused_intermediate_buf)
    @__parameter
    def unfused_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) -> None:
        var idx = unfused_intermediate_buf.layout(coords)
        unfused_intermediate_buf.raw_store[width=width, alignment=alignment](
            idx, val
        )

    @always_inline
    @__copy_capture(data_buf)
    @__parameter
    def rms_input_fn_coord[width: Int](coords: Coord) -> SIMD[dtype, width]:
        var idx = data_buf.layout(coords)
        return data_buf.raw_load[width=width](idx)

    # Step 1: First RMS norm
    rms_norm_gpu[
        rank, rms_input_fn_coord, unfused_output_fn, multiply_before_cast=True
    ](Coord(shape), gamma1, epsilon1, weight_offset1, ctx)

    @always_inline
    def sum_fn[width: Int, alignment: Int = 1](coords: Coord) {var}:
        var data_idx = data_buf.layout(coords)
        var residual_val = data_buf.raw_load[width=width](data_idx)
        var unfused_idx = unfused_intermediate_buf.layout(coords)
        var result_val = unfused_intermediate_buf.raw_load[width=width](
            unfused_idx
        )
        unfused_intermediate_buf.raw_store[width=width](
            unfused_idx, residual_val + result_val
        )

    elementwise[simd_width_of[dtype](), target="gpu"](
        sum_fn,
        unfused_intermediate_buf.layout.shape_coord(),
        ctx,
    )

    @__parameter
    @always_inline
    @__copy_capture(unfused_intermediate_buf)
    def unfused_input2_fn[width: Int](coords: Coord) -> SIMD[dtype, width]:
        var idx = unfused_intermediate_buf.layout(coords)
        return unfused_intermediate_buf.raw_load[width=width](idx)

    # Test unfused operations for comparison
    @always_inline
    @__copy_capture(result_unfused_buf)
    @__parameter
    def unfused_output2_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) -> None:
        var idx = result_unfused_buf.layout(coords)
        result_unfused_buf.raw_store[width=width, alignment=alignment](idx, val)

    rms_norm_gpu[
        rank,
        unfused_input2_fn,
        unfused_output2_fn,
        multiply_before_cast=True,
    ](Coord(shape), gamma2, epsilon2, weight_offset2, ctx)

    ctx.synchronize()

    # Verify results
    var flattened_size = rows * cols
    with result_fused_device.map_to_host() as result_fused_host:
        with result_unfused_device.map_to_host() as result_unfused_host:
            with residual_fused_output_device.map_to_host() as residual_fused_host:
                with unfused_intermediate_device.map_to_host() as unfused_intermediate_host:
                    for i in range(flattened_size):
                        assert_almost_equal(
                            result_fused_host[i],
                            result_unfused_host[i],
                            rtol=rtol,
                        )
                        assert_almost_equal(
                            residual_fused_host[i],
                            unfused_intermediate_host[i],
                            rtol=rtol,
                        )


def main() raises:
    with DeviceContext() as ctx:
        # Test various shapes similar to test_rms_norm.mojo
        run_rms_norm_fused_residual_add_gpu[.float32](ctx, Index(5))
        run_rms_norm_fused_residual_add_gpu[.float32](
            ctx, Index(3, 4, 10, 20, 8)
        )
        run_rms_norm_fused_residual_add_gpu[.bfloat16](
            ctx, Index(1, 5, 6, 10, 128)
        )
        run_rms_norm_fused_residual_add_gpu[.float32](ctx, Index(2, 5))
        run_rms_norm_fused_residual_add_gpu[.bfloat16](ctx, Index(2, 55))
        run_rms_norm_fused_residual_add_gpu[.float32](ctx, Index(7, 557))
        run_rms_norm_fused_residual_add_gpu[.bfloat16](ctx, Index(2, 8191))
        run_rms_norm_fused_residual_add_gpu[.float32](ctx, Index(2, 8192))
        run_rms_norm_fused_residual_add_gpu[.bfloat16](ctx, Index(2, 16384))
        run_rms_norm_fused_residual_add_gpu[.bfloat16](ctx, Index(2, 16385))

        # TODO(KERN-1951): the following fails with CUDA_ERROR_INVALID_VALUE, not sure why
        # run_rms_norm_fused_residual_add_gpu[DType.float32](ctx, Index(2, 16384))
        # run_rms_norm_fused_residual_add_gpu[DType.float32](ctx, Index(2, 16385))
