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
# Issue #23536

from max.algorithm.functional import stencil
from std.testing import TestSuite

from std.utils import IndexList
from std.utils.numerics import min_or_neg_inf

comptime _map_fn_type = def[rank: Int](IndexList[rank]) capturing -> Tuple[
    IndexList[rank],
    IndexList[rank],
]
comptime load_fn_type = def[dtype: DType, rank: Int, simd_width: Int](
    IndexList[rank]
) capturing -> SIMD[dtype, simd_width]


def _linear_index[
    rank: Int
](coords: IndexList[rank, ...], shape: IndexList[rank]) -> Int:
    """Convert multi-dimensional coordinates to linear index (row-major)."""
    var linear_idx = 0
    var stride = 1

    comptime for i in reversed(range(rank)):
        linear_idx += coords[i] * stride
        stride *= shape[i]
    return linear_idx


def fill_span[
    dtype: DType, origin: MutOrigin
](buf: Span[Scalar[dtype], origin], size: Int):
    var buf_ptr: Pointer[buf.T, buf.origin] = buf.unsafe_ptr()
    for j in range(size):
        buf_ptr[unsafe_offset=j] = Scalar[dtype](j) + 1


# TODO: Refactor tests
# CHECK-LABEL: test_stencil_avg_pool
def test_stencil_avg_pool() raises:
    print("== test_stencil_avg_pool")
    comptime rank = 4
    comptime stencil_rank = 2
    comptime dtype = DType.float32
    comptime simd_with = 1

    comptime input_width = 5
    comptime input_height = 5

    comptime stride = 1
    comptime pool_window_h = 3
    comptime pool_window_w = 3
    comptime dilation = 1

    comptime input_shape_dims = IndexList[4](1, input_height, input_width, 1)

    comptime output_height = input_height - pool_window_h + 1
    comptime output_width = input_width - pool_window_w + 1

    comptime output_shape_dims = IndexList[4](1, output_height, output_width, 1)

    var input_stack = Array[
        Scalar[dtype], Int(input_shape_dims.flattened_length())
    ](uninitialized=True)
    var input = Span(input_stack)
    var input_shape = IndexList[rank](1, input_height, input_width, 1)
    var output_stack = Array[
        Scalar[dtype], Int(output_shape_dims.flattened_length())
    ](uninitialized=True)
    var output = Span(output_stack)
    var output_shape = IndexList[rank](1, output_height, output_width, 1)
    var output_ptr: Pointer[output.T, output.origin] = output.unsafe_ptr()

    fill_span(input, Int(input_shape_dims.flattened_length()))
    for i in range(Int(output_shape_dims.flattened_length())):
        output_ptr[unsafe_offset=i] = 0

    def map_fn(
        point: IndexList[stencil_rank, ...],
    ) -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](point[0], point[1])
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h, point[1] + pool_window_w
        )
        return lower_bound, upper_bound

    @always_inline
    def load_fn[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {
        imm input_stack,
        var input_shape,
    } -> SIMD[
        dtype, simd_width
    ]:
        var linear_idx = _linear_index(point, input_shape)
        var input_stack_ptr: Pointer[
            input_stack.T, origin_of(input_stack)
        ] = input_stack.unsafe_ptr()
        var r = input_stack_ptr.unsafe_load[width=simd_width](
            linear_idx
        )._refine[dtype]()

        return r

    @always_inline
    def avg_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    @always_inline
    def avg_pool_compute[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return val + result

    def dilation_fn(dim: Int) -> Int:
        return 1

    @always_inline
    def avg_pool_compute_finalize[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var output,
        var output_shape,
    }:
        var res = val / (pool_window_h * pool_window_w)
        var linear_idx = _linear_index(point, output_shape)
        var output_ptr: Pointer[output.T, output.origin] = output.unsafe_ptr()
        output_ptr.unsafe_store(linear_idx, res)

    comptime stencil_axis = IndexList[stencil_rank](1, 2)
    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_with,
        dtype,
    ](
        output_shape,
        input_shape,
        map_fn,
        dilation_fn,
        load_fn,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize,
    )

    # CHECK: 7.0    8.0     9.0
    # CHECK: 12.0    13.0    14.0
    # CHECK: 17.0    18.0    19.0
    for i in range(0, output_height):
        for j in range(0, output_width):
            var idx = _linear_index(IndexList[rank](0, i, j, 0), output_shape)
            print(output_ptr[unsafe_offset=idx], "\t", end="")
        print("")


# CHECK-LABEL: test_stencil_avg_pool_padded
def test_stencil_avg_pool_padded() raises:
    print("== test_stencil_avg_pool_padded")
    comptime rank = 4
    comptime stencil_rank = 2
    comptime dtype = DType.float32
    comptime simd_with = 1

    comptime input_width = 5
    comptime input_height = 5

    comptime stride = 1
    comptime pool_window_h = 5
    comptime pool_window_w = 5
    comptime dilation = 1
    comptime pad_h = 2
    comptime pad_w = 2

    comptime input_shape_dims = IndexList[4](1, input_height, input_width, 1)

    comptime output_height = input_height - pool_window_h + pad_h * 2 + 1
    comptime output_width = input_width - pool_window_w + pad_w * 2 + 1

    comptime output_shape_dims = IndexList[4](1, output_height, output_width, 1)

    var input_stack = Array[Scalar[dtype], input_shape_dims.flattened_length()](
        fill={}
    )
    var input = Span(input_stack)
    var input_shape = IndexList[rank](1, input_height, input_width, 1)
    var output_stack = Array[
        Scalar[dtype], output_shape_dims.flattened_length()
    ](uninitialized=True)
    var output = Span(output_stack)
    var output_shape = IndexList[rank](1, output_height, output_width, 1)
    var output_ptr: Pointer[output.T, output.origin] = output.unsafe_ptr()

    fill_span(input, input_shape_dims.flattened_length())
    for i in range(output_shape_dims.flattened_length()):
        output_ptr[unsafe_offset=i] = 0

    def map_fn(
        point: IndexList[stencil_rank, ...],
    ) -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] - pad_h, point[1] - pad_w
        )
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h - pad_h, point[1] + pool_window_w - pad_w
        )
        return lower_bound, upper_bound

    def load_fn[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {
        imm input_stack,
        var input_shape,
    } -> SIMD[
        dtype, simd_width
    ]:
        var linear_idx = _linear_index(point, input_shape)
        var input_stack_ptr: Pointer[
            input_stack.T, origin_of(input_stack)
        ] = input_stack.unsafe_ptr()
        return input_stack_ptr.unsafe_load[width=simd_width](
            linear_idx
        )._refine[dtype]()

    @always_inline
    def avg_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    @always_inline
    def avg_pool_compute[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return val + result

    def avg_pool_compute_finalize[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var output,
        var output_shape,
    }:
        var res = val / (pool_window_h * pool_window_w)
        var linear_idx = _linear_index(point, output_shape)
        var output_ptr: Pointer[output.T, output.origin] = output.unsafe_ptr()
        output_ptr.unsafe_store(linear_idx, res)

    def dilation_fn(dim: Int) -> Int:
        return 1

    comptime stencil_axis = IndexList[stencil_rank](1, 2)
    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_with,
        dtype,
    ](
        output_shape,
        input_shape,
        map_fn,
        dilation_fn,
        load_fn,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize,
    )

    # CHECK: 2.52 3.6 4.8 4.08 3.24
    # CHECK: 4.56 6.4 8.4 7.04 5.52
    # CHECK: 7.2 10.0 13.0 10.8 8.4
    # CHECK: 6.96 9.6 12.4 10.24 7.92
    # CHECK: 6.12 8.4 10.8 8.88 6.84
    for i in range(0, output_height):
        for j in range(0, output_width):
            var idx = _linear_index(IndexList[rank](0, i, j, 0), output_shape)
            print(output_ptr[unsafe_offset=idx], "\t", end="")
        print("")


# CHECK-LABEL: test_stencil_avg_pool_stride_2
def test_stencil_avg_pool_stride_2() raises:
    print("== test_stencil_avg_pool_stride_2")
    comptime rank = 4
    comptime stencil_rank = 2
    comptime dtype = DType.float32
    comptime simd_with = 1

    comptime input_width = 7
    comptime input_height = 7

    comptime stride = 2
    comptime pool_window_h = 3
    comptime pool_window_w = 3
    comptime dilation = 1

    comptime input_shape_dims = IndexList[4](1, input_height, input_width, 1)

    comptime output_height = (input_height - pool_window_h) // stride + 1
    comptime output_width = (input_width - pool_window_w) // stride + 1

    comptime output_shape_dims = IndexList[4](1, output_height, output_width, 1)

    var input_stack = Array[
        Scalar[dtype], Int(input_shape_dims.flattened_length())
    ](uninitialized=True)
    var input = Span(input_stack)
    var input_shape = IndexList[rank](1, input_height, input_width, 1)
    var output_stack = Array[
        Scalar[dtype], Int(output_shape_dims.flattened_length())
    ](uninitialized=True)
    var output = Span(output_stack)
    var output_shape = IndexList[rank](1, output_height, output_width, 1)
    var output_ptr: Pointer[output.T, output.origin] = output.unsafe_ptr()

    fill_span(input, Int(input_shape_dims.flattened_length()))
    for i in range(Int(output_shape_dims.flattened_length())):
        output_ptr[unsafe_offset=i] = 0

    def map_fn(
        point: IndexList[stencil_rank, ...],
    ) -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride, point[1] * stride
        )
        var upper_bound = IndexList[stencil_rank](
            (point[0] * stride + pool_window_h),
            (point[1] * stride + pool_window_w),
        )
        return lower_bound, upper_bound

    def load_fn[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {
        imm input_stack,
        var input_shape,
    } -> SIMD[
        dtype, simd_width
    ]:
        var linear_idx = _linear_index(point, input_shape)
        var input_stack_ptr: Pointer[
            input_stack.T, origin_of(input_stack)
        ] = input_stack.unsafe_ptr()
        return input_stack_ptr.unsafe_load[width=simd_width](
            linear_idx
        )._refine[dtype]()

    @always_inline
    def avg_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    @always_inline
    def avg_pool_compute[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return val + result

    def avg_pool_compute_finalize[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var output,
        var output_shape,
    }:
        var res = val / (pool_window_h * pool_window_w)
        var linear_idx = _linear_index(point, output_shape)
        var output_ptr: Pointer[output.T, output.origin] = output.unsafe_ptr()
        output_ptr.unsafe_store(linear_idx, res)

    def dilation_fn(dim: Int) -> Int:
        return 1

    comptime stencil_axis = IndexList[stencil_rank](1, 2)
    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_with,
        dtype,
    ](
        output_shape,
        input_shape,
        map_fn,
        dilation_fn,
        load_fn,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize,
    )

    # CHECK: 9.0     11.0    13.0
    # CHECK: 23.0    25.0    27.0
    # CHECK: 37.0    39.0    41.0
    for i in range(0, output_height):
        for j in range(0, output_width):
            var idx = _linear_index(IndexList[rank](0, i, j, 0), output_shape)
            print(output_ptr[unsafe_offset=idx], "\t", end="")
        print("")


# CHECK-LABEL: test_stencil_max_pool_dilation_2
def test_stencil_max_pool_dilation_2() raises:
    print("== test_stencil_max_pool_dilation_2")
    comptime rank = 4
    comptime stencil_rank = 2
    comptime dtype = DType.float32
    comptime simd_with = 1

    comptime input_width = 7
    comptime input_height = 7

    comptime stride = 1
    comptime pool_window_h = 3
    comptime pool_window_w = 3
    comptime dilation = 2

    comptime input_shape_dims = IndexList[4](1, input_height, input_width, 1)

    comptime output_height = (
        input_height - pool_window_h - (pool_window_h - 1) * (dilation - 1)
    ) // stride + 1
    comptime output_width = (
        input_width - pool_window_w - (pool_window_w - 1) * (dilation - 1)
    ) // stride + 1

    comptime output_shape_dims = IndexList[4](1, output_height, output_width, 1)

    var input_stack = Array[
        Scalar[dtype], Int(input_shape_dims.flattened_length())
    ](uninitialized=True)
    var input = Span(input_stack)
    var input_shape = IndexList[rank](1, input_height, input_width, 1)
    var output_stack = Array[
        Scalar[dtype], Int(output_shape_dims.flattened_length())
    ](uninitialized=True)
    var output = Span(output_stack)
    var output_shape = IndexList[rank](1, output_height, output_width, 1)
    var output_ptr: Pointer[output.T, output.origin] = output.unsafe_ptr()

    fill_span(input, Int(input_shape_dims.flattened_length()))
    for i in range(Int(output_shape_dims.flattened_length())):
        output_ptr[unsafe_offset=i] = 0

    def map_fn(
        point: IndexList[stencil_rank, ...],
    ) -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride, point[1] * stride
        )
        var upper_bound = IndexList[stencil_rank](
            (point[0] * stride + pool_window_h * dilation),
            (point[1] * stride + pool_window_w * dilation),
        )
        return lower_bound, upper_bound

    def load_fn[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {
        imm input_stack,
        var input_shape,
    } -> SIMD[
        dtype, simd_width
    ]:
        var linear_idx = _linear_index(point, input_shape)
        var input_stack_ptr: Pointer[
            input_stack.T, origin_of(input_stack)
        ] = input_stack.unsafe_ptr()
        return input_stack_ptr.unsafe_load[width=simd_width](
            linear_idx
        )._refine[dtype]()

    @always_inline
    def max_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return min_or_neg_inf[dtype]()

    @always_inline
    def max_pool_compute[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return max(val, result)

    def max_pool_compute_finalize[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var output,
        var output_shape,
    }:
        var linear_idx = _linear_index(point, output_shape)
        var output_ptr: Pointer[output.T, output.origin] = output.unsafe_ptr()
        output_ptr.unsafe_store(linear_idx, val)

    @always_inline
    def dilation_fn(dim: Int) -> Int:
        return dilation

    comptime stencil_axis = IndexList[stencil_rank](1, 2)
    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_with,
        dtype,
    ](
        output_shape,
        input_shape,
        map_fn,
        dilation_fn,
        load_fn,
        max_pool_compute_init,
        max_pool_compute,
        max_pool_compute_finalize,
    )

    # CHECK: 33.0    34.0    35.0
    # CHECK: 40.0    41.0    42.0
    # CHECK: 47.0    48.0    49.0
    for i in range(0, output_height):
        for j in range(0, output_width):
            var idx = _linear_index(IndexList[rank](0, i, j, 0), output_shape)
            print(output_ptr[unsafe_offset=idx], "\t", end="")
        print("")


# CHECK-LABEL: test_stencil_size_0
def test_stencil_size_0() raises:
    comptime rank = 4
    comptime stencil_rank = 2

    # An output shape whose last dimension is 0 means we should do zero work.
    var output_shape = IndexList[rank](1, 1, 1, 0)
    var input_shape = IndexList[rank](1, 1, 1, 1)

    def map_fn(
        point: IndexList[stencil_rank, ...],
    ) -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        return IndexList[stencil_rank](0, 0), IndexList[stencil_rank](1, 1)

    def map_strides(dim: Int) -> Int:
        return 1

    def load_fn[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) -> SIMD[dtype, simd_width]:
        return 0

    def init_fn[simd_width: Int]() -> SIMD[.float32, simd_width]:
        return 0

    def compute_fn[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        a: SIMD[.float32, simd_width],
        b: SIMD[.float32, simd_width],
    ) -> SIMD[.float32, simd_width]:
        return a + b

    def finalize_fn[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[.float32, simd_width],):
        pass

    comptime stencil_axis = IndexList[stencil_rank](1, 2)
    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        1,
        DType.float32,
    ](
        output_shape,
        input_shape,
        map_fn,
        map_strides,
        load_fn,
        init_fn,
        compute_fn,
        finalize_fn,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
