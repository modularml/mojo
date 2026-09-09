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

from max.algorithm.backend.gpu.reduction import reduce_launch
from max.gpu.host import DeviceContext
from std.testing import assert_equal, TestSuite

from std.utils import IndexList, StaticTuple

comptime num_reductions = 2


def fused_reduce_inner_test[
    reduce_fn: def[ty: DType, width: SIMDLength, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    rank: Int,
    dtype: DType,
    *,
    axis: Int = rank - 1,
](
    shape: IndexList[rank],
    init: StaticTuple[Scalar[dtype], num_reductions],
    expected_vals0: List[Float32],
    expected_vals1: List[Float32],
    ctx: DeviceContext,
    offset: Int = 1,
) raises:
    var out_shape = shape
    out_shape[axis] = 1

    var in_size = shape.flattened_length()
    var out_size = out_shape.flattened_length()

    assert_equal(
        len(expected_vals0),
        out_size,
        "expected vals must match output shape",
    )
    assert_equal(
        len(expected_vals1),
        out_size,
        "expected vals must match output shape",
    )

    var vec_device = ctx.enqueue_create_buffer[dtype](in_size)
    with vec_device.map_to_host() as vec_host:
        for i in range(in_size):
            vec_host[i] = Scalar[dtype](i // shape[axis] + offset)

    var res_device0 = ctx.enqueue_create_buffer[dtype](out_size)
    var res_device1 = ctx.enqueue_create_buffer[dtype](out_size)
    var input_buf_device = Span(
        unsafe_ptr=vec_device.unsafe_ptr(), length=shape.flattened_length()
    )
    var output_buf_device0 = Span(
        unsafe_ptr=res_device0.unsafe_ptr(), length=out_shape.flattened_length()
    )
    var output_buf_device1 = Span(
        unsafe_ptr=res_device1.unsafe_ptr(), length=out_shape.flattened_length()
    )

    @__copy_capture(input_buf_device, shape)
    @__parameter
    def input_fn[
        dtype: DType,
        width: Int,
        _rank: Int,
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var c = rebind[IndexList[rank]](coords)
        var linear_idx = 0
        var stride = 1

        comptime for i in reversed(range(rank)):
            linear_idx += c[i] * stride
            stride *= shape[i]
        return rebind[SIMD[dtype, width]](
            input_buf_device.unsafe_ptr().unsafe_load[width=width](linear_idx)
        )

    @__copy_capture(output_buf_device0, output_buf_device1, out_shape)
    @__parameter
    def output_fn[
        _dtype: DType, width: SIMDLength, _rank: Int
    ](
        coords: IndexList[_rank],
        val: StaticTuple[SIMD[_dtype, width], num_reductions],
    ):
        var c = rebind[IndexList[rank]](coords)
        var linear_idx = 0
        var stride = 1

        comptime for i in reversed(range(rank)):
            linear_idx += c[i] * stride
            stride *= out_shape[i]
        output_buf_device0.unsafe_ptr().unsafe_store[width=width](
            linear_idx, rebind[SIMD[dtype, width]](val[0])
        )
        output_buf_device1.unsafe_ptr().unsafe_store[width=width](
            linear_idx, rebind[SIMD[dtype, width]](val[1])
        )

    reduce_launch[
        num_reductions,
        input_fn,
        output_fn,
        reduce_fn,
        rank,
        dtype,
        reduce_dim=axis,
    ](shape, init, ctx)

    with res_device0.map_to_host() as res_host0:
        for i in range(out_shape.flattened_length()):
            assert_equal(
                String(res_host0[i].cast[.float32]()),
                String(expected_vals0[i]),
            )

    with res_device1.map_to_host() as res_host1:
        for i in range(out_shape.flattened_length()):
            assert_equal(
                String(res_host1[i].cast[.float32]()),
                String(expected_vals1[i]),
            )

    _ = vec_device
    _ = res_device0
    _ = res_device1


def reduce_inner_test[
    reduce_fn: def[dtype: DType, width: SIMDLength](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    rank: Int,
    dtype: DType,
    expected_vals_type: DType,
    *,
    axis: Int = rank - 1,
](
    shape: IndexList[rank],
    init: Scalar[dtype],
    expected_vals: List[Scalar[expected_vals_type]],
    ctx: DeviceContext,
    offset: Int = 1,
) raises:
    comptime num_reductions = 1

    var out_shape = shape
    out_shape[axis] = 1

    var in_size = shape.flattened_length()
    var out_size = shape.flattened_length() // shape[axis]
    assert_equal(
        len(expected_vals), out_size, "expected vals must match output shape"
    )

    var vec_device = ctx.enqueue_create_buffer[dtype](in_size)

    with vec_device.map_to_host() as vec_host:
        for i in range(in_size):
            vec_host[i] = Scalar[dtype](i // shape[axis] + offset)

    var res_device = ctx.enqueue_create_buffer[dtype](out_size)
    var input_buf_device = Span(
        unsafe_ptr=vec_device.unsafe_ptr(), length=shape.flattened_length()
    )
    var output_buf_device = Span(
        unsafe_ptr=res_device.unsafe_ptr(), length=out_shape.flattened_length()
    )

    @always_inline
    @__parameter
    def reduce_wrapper[
        dtype: DType, width: SIMDLength, reduction_idx: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert reduction_idx < num_reductions, "invalid reduction idx"

        return reduce_fn[dtype, width](lhs, rhs)

    @__copy_capture(input_buf_device, shape)
    @__parameter
    def input_fn[
        dtype: DType,
        width: Int,
        _rank: Int,
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var c = rebind[IndexList[rank]](coords)
        var linear_idx = 0
        var stride = 1

        comptime for i in reversed(range(rank)):
            linear_idx += c[i] * stride
            stride *= shape[i]
        return rebind[SIMD[dtype, width]](
            input_buf_device.unsafe_ptr().unsafe_load[width=width](linear_idx)
        )

    @__copy_capture(output_buf_device, out_shape)
    @__parameter
    def output_fn[
        _dtype: DType, width: SIMDLength, _rank: Int
    ](
        coords: IndexList[_rank],
        val: StaticTuple[SIMD[_dtype, width], num_reductions],
    ):
        var c = rebind[IndexList[rank]](coords)
        var linear_idx = 0
        var stride = 1

        comptime for i in reversed(range(rank)):
            linear_idx += c[i] * stride
            stride *= out_shape[i]
        output_buf_device.unsafe_ptr().unsafe_store[width=width](
            linear_idx, rebind[SIMD[dtype, width]](val[0])
        )

    reduce_launch[
        num_reductions,
        input_fn,
        output_fn,
        reduce_wrapper,
        rank,
        dtype,
        reduce_dim=axis,
    ](shape, StaticTuple[_, num_reductions](init), ctx)

    with res_device.map_to_host() as res_host:
        for i in range(out_shape.flattened_length()):
            print(res_host[i], expected_vals[i])
            assert_equal(String(res_host[i]), String(expected_vals[i]))

    _ = vec_device
    _ = res_device


def test_reduce() raises:
    @__parameter
    def reduce_add[
        dtype: DType,
        width: SIMDLength,
    ](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return x + y

    @__parameter
    def reduce_max[
        dtype: DType,
        width: SIMDLength,
    ](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return max(x, y)

    @__parameter
    def fused_reduce_add_max[
        dtype: DType,
        width: SIMDLength,
        reduction_idx: Int,
    ](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert reduction_idx < 2, "reduction idx OOB"

        comptime func = reduce_max if reduction_idx == 0 else reduce_add
        return func(x, y)

    with DeviceContext() as ctx:
        reduce_inner_test[reduce_add](
            IndexList[3](2, 3, 257),
            Float32(0),
            [Float32(257.0), 514.0, 771.0, 1028.0, 1285.0, 1542.0],
            ctx,
        )

        reduce_inner_test[reduce_add](
            IndexList[2](5, 257),
            Float32(0),
            [Float32(257.0), 514.0, 771.0, 1028.0, 1285.0],
            ctx,
        )

        reduce_inner_test[reduce_add](
            IndexList[4](2, 2, 2, 1029),
            Float32(0),
            [
                Float32(1029.0),
                2058.0,
                3087.0,
                4116.0,
                5145.0,
                6174.0,
                7203.0,
                8232.0,
            ],
            ctx,
        )

        reduce_inner_test[reduce_add, axis=0](
            IndexList[3](5, 3, 2),
            Float32(0),
            [
                Float32(15.0),
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
            ],
            ctx,
        )

        reduce_inner_test[reduce_add, axis=1](
            IndexList[3](5, 3, 2),
            Float32(0),
            [
                Float32(4.0),
                5.0,
                10.0,
                11.0,
                16.0,
                17.0,
                22.0,
                23.0,
                28.0,
                29.0,
            ],
            ctx,
        )

        reduce_inner_test[reduce_max](
            IndexList[2](5, 3),
            Float32.MIN,
            [Float32(1.0), 2.0, 3.0, 4.0, 5.0],
            ctx,
        )

        fused_reduce_inner_test[fused_reduce_add_max, 2, DType.float32](
            IndexList[2](5, 3),
            StaticTuple[Float32, 2](Float32.MIN, 0.0),
            [Float32(1.0), 2.0, 3.0, 4.0, 5.0],
            [Float32(3.0), 6.0, 9.0, 12.0, 15.0],
            ctx,
        )

        # bf16 tests
        reduce_inner_test[reduce_max](
            IndexList[2](5, 5),
            BFloat16.MIN,
            [Float32(1.0), 2.0, 3.0, 4.0, 5.0],
            ctx,
        )

        fused_reduce_inner_test[fused_reduce_add_max, 2, DType.bfloat16](
            IndexList[2](5, 3),
            StaticTuple[BFloat16, 2](BFloat16.MIN, 0.0),
            [Float32(1.0), 2.0, 3.0, 4.0, 5.0],
            [Float32(3.0), 6.0, 9.0, 12.0, 15.0],
            ctx,
        )

        # fp16 tests
        reduce_inner_test[reduce_max](
            IndexList[2](5, 5),
            Float16.MIN,
            [Float32(1.0), 2.0, 3.0, 4.0, 5.0],
            ctx,
        )

        fused_reduce_inner_test[fused_reduce_add_max, 2, DType.float16](
            IndexList[2](5, 3),
            StaticTuple[Float16, 2](Float16.MIN, 0.0),
            [Float32(1.0), 2.0, 3.0, 4.0, 5.0],
            [Float32(3.0), 6.0, 9.0, 12.0, 15.0],
            ctx,
        )

        # int64 tests
        reduce_inner_test[reduce_max](
            IndexList[2](5, 5),
            Int64.MIN,
            [Int64(1), 2, 3, 4, 5],
            ctx,
        )
        fused_reduce_inner_test[fused_reduce_add_max, 2, DType.int64](
            IndexList[2](5, 3),
            StaticTuple[Int64, 2](Int64.MIN, 0),
            [Float32(1.0), 2.0, 3.0, 4.0, 5.0],
            [Float32(3.0), 6.0, 9.0, 12.0, 15.0],
            ctx,
        )
        # Add offset to ensure upper and lower 32 bits of element are non-zero
        var offset: Int = 0xDEADBEEF
        reduce_inner_test[reduce_max](
            IndexList[2](5, 5),
            Int64.MIN,
            [
                Int64(offset),
                Int64(offset + 1),
                Int64(offset + 2),
                Int64(offset + 3),
                Int64(offset + 4),
            ],
            ctx,
            offset=offset,
        )
        fused_reduce_inner_test[fused_reduce_add_max, 2, DType.int64](
            IndexList[2](5, 3),
            StaticTuple[Int64, 2](Int64.MIN, 0),
            [
                Float32(offset),
                Float32(Float64(offset) + 1.0),
                Float32(Float64(offset) + 2.0),
                Float32(Float64(offset) + 3.0),
                Float32(Float64(offset) + 4.0),
            ],
            [
                Float32(Float64(offset) * 3 + 3.0),
                Float32(Float64(offset) * 3 + 6.0),
                Float32(Float64(offset) * 3 + 9.0),
                Float32(Float64(offset) * 3 + 12.0),
                Float32(Float64(offset) * 3 + 15.0),
            ],
            ctx,
            offset=offset,
        )


def test_multiblock_reduce() raises:
    """Tests the multiblock_reduce_kernel path for under-saturated cases
    where num_rows is small but the reduction axis is large."""

    @__parameter
    def reduce_add[
        dtype: DType,
        width: SIMDLength,
    ](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return x + y

    @__parameter
    def reduce_max[
        dtype: DType,
        width: SIMDLength,
    ](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return max(x, y)

    with DeviceContext() as ctx:
        # Large 1D reduction: single row, exercises multiblock path.
        # Shape [8192], reduce axis 0. Each element = 1, so sum = 8192.
        reduce_inner_test[reduce_add, axis=0](
            IndexList[1](8192),
            Float32(0),
            [Float32(8192.0)],
            ctx,
            offset=1,
        )

        # Larger 1D reduction to stress the two-phase coordination.
        reduce_inner_test[reduce_add, axis=0](
            IndexList[1](131072),
            Float32(0),
            [Float32(131072.0)],
            ctx,
            offset=1,
        )

        # Low-row 2D reduction: few rows with large reduction axis.
        # Shape [4, 8192], reduce axis 1. Each row has constant value
        # (row_idx + 1), so sum = value * 8192.
        reduce_inner_test[reduce_add](
            IndexList[2](4, 8192),
            Float32(0),
            [Float32(8192.0), 16384.0, 24576.0, 32768.0],
            ctx,
        )

        # Max reduction on large 1D tensor.
        # Elements are i // shape[axis] + offset = 0 + 1 = 1 for all i.
        reduce_inner_test[reduce_max, axis=0](
            IndexList[1](8192),
            Float32.MIN,
            [Float32(1.0)],
            ctx,
            offset=1,
        )


def test_thread_saturated_contiguous_reduce() raises:
    """Regression test for a contiguous last-axis reduction that saturates the
    device.

    An N-D last-axis reduction is normalized to a rank-3 (outer, reduce, 1)
    shape with reduce_dim=1, so the reduce dim is still physically contiguous
    even though it is not the final index. `reduce_launch` previously decided
    contiguity with `reduce_dim == rank - 1`, so the trailing unit dim made it
    treat the reduction as non-contiguous and, once `num_rows` reached
    `256 * sm_count`, dispatch to `saturated_reduce_kernel`. That kernel packs
    adjacent rows into SIMD lanes, which is only valid when a real inner dim
    supplies those rows; with inner == 1 it summed the wrong elements and
    produced wrong results. Concretely, `Tensor.mean(-1)` on a (37888, 64)
    tensor was wrong on a 148-SM B200, since 37888 == 256 * 148. This pins the
    fix: the reduction must stay correct at and beyond the saturation
    boundary.
    """

    @__parameter
    def reduce_add[
        dtype: DType,
        width: SIMDLength,
    ](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return x + y

    with DeviceContext() as ctx:
        var sm_count = ctx.default_device_info.sm_count
        comptime reduce_size = 64

        # Exactly hit (and clear) the `num_rows >= 256 * sm_count`
        # thread-saturation boundary that selects `saturated_reduce_kernel`.
        for outer in [256 * sm_count, 256 * sm_count + 1]:
            var expected = List[Float32](capacity=outer)
            for o in range(outer):
                # Each element of row `o` is `o + 1`, so the row sums to
                # reduce_size * (o + 1).
                expected.append(Float32(reduce_size * (o + 1)))

            reduce_inner_test[reduce_add, axis=1](
                IndexList[3](outer, reduce_size, 1),
                Float32(0),
                expected,
                ctx,
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
