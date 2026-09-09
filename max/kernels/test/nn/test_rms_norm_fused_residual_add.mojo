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

from std.sys.info import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, coord_to_index_list, row_major
from layout._fillers import random
from nn.normalization import rms_norm_cpu, rms_norm_fused_residual_add
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


def run_rms_norm_fused_residual_add_gpu[
    rank: Int,
    //,
    dtype: DType,
](shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    # Allocate host memory
    var data_heap = List(length=rows * cols, fill=Scalar[dtype](0))
    var data_h = TileTensor(data_heap, row_major(Coord(shape)))
    var unfused_intermediate_heap = List(
        length=rows * cols, fill=Scalar[dtype](0)
    )
    var unfused_intermediate_h = TileTensor(
        unfused_intermediate_heap, row_major(Coord(shape))
    )
    var result_unfused_heap = List(length=rows * cols, fill=Scalar[dtype](0))
    var result_unfused_h = TileTensor(
        result_unfused_heap, row_major(Coord(shape))
    )
    var result_fused_heap = List(length=rows * cols, fill=Scalar[dtype](0))
    var result_fused_h = TileTensor(result_fused_heap, row_major(Coord(shape)))
    var residual_fused_output_heap = List(
        length=rows * cols, fill=Scalar[dtype](0)
    )
    var residual_fused_output_h = TileTensor(
        residual_fused_output_heap, row_major(Coord(shape))
    )
    var gamma1_heap = List(length=cols, fill=Scalar[dtype](0))
    var gamma1_h = TileTensor(gamma1_heap, row_major(cols))
    var gamma2_heap = List(length=cols, fill=Scalar[dtype](0))
    var gamma2_h = TileTensor(gamma2_heap, row_major(cols))

    # Initialize input data
    random(data_h)
    random(gamma1_h)
    random(gamma2_h)

    var _ = Index(cols)

    var data_buf = data_h
    # Separate residual buffer (a copy of the input): the fused kernel's two
    # input loaders are distinct value-closure args and must reference distinct
    # buffer origins. Copying keeps residual == input numerically.
    var residual_heap = List(length=rows * cols, fill=Scalar[dtype](0))
    var data_buf_res = TileTensor(residual_heap, row_major(Coord(shape)))
    for i in range(rows * cols):
        residual_heap[i] = data_heap[i]
    var gamma1 = gamma1_h
    var gamma2 = gamma2_h
    var result_fused_buf = result_fused_h
    var result_unfused_buf = result_unfused_h
    var unfused_intermediate_buf = unfused_intermediate_h
    var residual_fused_output_buf = residual_fused_output_h
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
        result_fused_buf.raw_store[width=width, alignment=alignment](idx, val)

    @always_inline
    def fused_residual_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {
        var residual_fused_output_buf
    } -> None:
        var idx = residual_fused_output_buf.layout(coords)
        residual_fused_output_buf.raw_store[width=width, alignment=alignment](
            idx, val
        )

    # Call fused kernel
    rms_norm_fused_residual_add[
        dtype,
        rank,
        target="cpu",
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

    # Legacy `rms_norm_cpu` still takes a comptime `capturing` closure, so the
    # unfused reference uses its own comptime input loader (the fused kernel's
    # `input_fn` is now a value closure and cannot bind a comptime parameter).
    @always_inline
    @__parameter
    @__copy_capture(data_buf)
    def rms_input_fn[width: Int](coords: Coord) -> SIMD[dtype, width]:
        var idx = data_buf.layout(coords)
        return data_buf.raw_load[width=width](idx)

    # Step 1: First RMS norm
    rms_norm_cpu[rms_input_fn, unfused_output_fn, multiply_before_cast=True](
        Coord(shape), gamma1, epsilon1, weight_offset1
    )

    @always_inline
    def sum_fn[width: Int, alignment: Int = 1](coords: Coord) {var}:
        var data_buf_idx = data_buf.layout(coords)
        var residual_val = data_buf.raw_load[width=width](data_buf_idx)
        var unfused_intermediate_buf_idx = unfused_intermediate_buf.layout(
            coords
        )
        var result_val = unfused_intermediate_buf.raw_load[width=width](
            unfused_intermediate_buf_idx
        )

        var residual_add_val = residual_val + result_val
        unfused_intermediate_buf.raw_store[width=width](
            unfused_intermediate_buf_idx, residual_add_val
        )

    elementwise[simd_width_of[dtype](), target="cpu"](
        sum_fn,
        unfused_intermediate_buf.layout.shape_coord(),
        DeviceContext(api="cpu"),
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

    rms_norm_cpu[
        unfused_input2_fn,
        unfused_output2_fn,
        multiply_before_cast=True,
    ](Coord(shape), gamma2, epsilon2, weight_offset2)

    var flattened_size = rows * cols
    for i in range(flattened_size):
        assert_almost_equal(
            result_fused_h.raw_load(i),
            result_unfused_h.raw_load(i),
            rtol=rtol,
        )
        assert_almost_equal(
            residual_fused_output_h.raw_load(i),
            unfused_intermediate_h.raw_load(i),
            rtol=rtol,
        )
    _ = gamma2_heap^
    _ = gamma1_heap^
    _ = residual_fused_output_heap^
    _ = result_fused_heap^
    _ = result_unfused_heap^
    _ = unfused_intermediate_heap^
    _ = data_heap^


def main() raises:
    # Test various shapes similar to test_rms_norm.mojo
    run_rms_norm_fused_residual_add_gpu[.float32](Index(5))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(3, 4, 10, 20, 8))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(1, 5, 6, 10, 128))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(2, 5))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(2, 55))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(7, 557))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(2, 8191))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(2, 8192))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(2, 16384))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(2, 16385))

    run_rms_norm_fused_residual_add_gpu[.float32](Index(2, 16384))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(2, 16385))
