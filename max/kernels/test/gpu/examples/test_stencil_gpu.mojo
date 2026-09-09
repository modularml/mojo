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

from max.algorithm.functional import stencil, stencil_gpu
from max.gpu.host import DeviceContext
from layout import Layout, TileTensor, coord_to_index_list, row_major
from layout._utils import ManagedLayoutTensor
from layout.tile_tensor import stack_allocation
from std.testing import assert_almost_equal

from std.utils import IndexList
from std.utils.numerics import min_or_neg_inf


def fill_buffer[dtype: DType](buf: TileTensor[mut=True, dtype, ...]):
    for j in range(buf.num_elements()):
        buf.raw_store(j, Scalar[dtype](j) + 1)


def assert_allclose[
    dtype: DType
](
    h_output_ref: TileTensor[dtype, ...],
    h_output_gpu: TileTensor[dtype, ...],
) raises:
    for i in range(h_output_ref.num_elements()):
        assert_almost_equal(h_output_ref.raw_load(i), h_output_gpu.raw_load(i))


def test_stencil_avg_pool(ctx: DeviceContext) raises:
    print("== test_stencil_avg_pool")
    comptime rank = 4
    comptime stencil_rank = 2
    comptime dtype = DType.float32
    comptime simd_width = 1

    comptime input_width = 5
    comptime input_height = 5

    comptime stride = 1
    comptime pool_window_h = 3
    comptime pool_window_w = 3
    comptime dilation = 1

    comptime output_height = input_height - pool_window_h + 1
    comptime output_width = input_width - pool_window_w + 1
    var input_shape_dyn = IndexList[4](1, input_height, input_width, 1)
    var output_shape_dyn = IndexList[4](1, output_height, output_width, 1)

    var d_input_managed = ManagedLayoutTensor[
        dtype, Layout.row_major(1, input_height, input_width, 1)
    ](ctx)
    var d_output_managed = ManagedLayoutTensor[
        dtype, Layout.row_major(1, output_height, output_width, 1)
    ](ctx)

    var h_input = TileTensor(
        d_input_managed.tensor[update=False]().ptr,
        row_major[1, input_height, input_width, 1](),
    )
    var h_output = TileTensor(
        d_output_managed.tensor[update=False]().ptr,
        row_major[1, output_height, output_width, 1](),
    )
    var h_output_ref = stack_allocation[dtype=dtype](
        row_major[1, output_height, output_width, 1]()
    )

    fill_buffer(h_input)
    _ = h_output.fill(0)
    _ = h_output_ref.fill(0)

    var d_input = TileTensor(
        d_input_managed.device_tensor().ptr,
        row_major[1, input_height, input_width, 1](),
    )
    var d_output = TileTensor(
        d_output_managed.device_tensor().ptr,
        row_major[1, output_height, output_width, 1](),
    )

    def map_fn_gpu(
        point: IndexList[stencil_rank, ...],
    ) {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](point[0], point[1])
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h, point[1] + pool_window_w
        )
        return lower_bound, upper_bound

    def dilation_fn_gpu(dim: Int) {} -> Int:
        return 1

    @always_inline
    def load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {var d_input,} -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load_linear[width=simd_width](point)
        )

    def avg_pool_compute_init_gpu[
        simd_width: Int
    ]() {} -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    def avg_pool_compute_gpu[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) {} -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    def avg_pool_compute_finalize_gpu[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var d_output,
    }:
        var res = val / (pool_window_h * pool_window_w)
        d_output.store_linear(point, res)

    comptime stencil_axis = IndexList[stencil_rank](1, 2)
    stencil_gpu[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
    ](
        ctx,
        rebind[IndexList[rank]](
            coord_to_index_list(d_output.layout.shape_coord())
        ),
        rebind[IndexList[rank]](
            coord_to_index_list(d_input.layout.shape_coord())
        ),
        map_fn_gpu,
        dilation_fn_gpu,
        load_fn_gpu,
        avg_pool_compute_init_gpu,
        avg_pool_compute_gpu,
        avg_pool_compute_finalize_gpu,
    )

    # Refresh host view; tensor() handles device-to-host transfer and sync.
    h_output = TileTensor(
        d_output_managed.tensor().ptr,
        row_major[1, output_height, output_width, 1](),
    )

    def map_fn_cpu(
        point: IndexList[stencil_rank, ...],
    ) {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](point[0], point[1])
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h, point[1] + pool_window_w
        )
        return lower_bound, upper_bound

    def dilation_fn_cpu(dim: Int) {} -> Int:
        return 1

    @always_inline
    def load_fn_ref[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {var h_input,} -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load_linear[width=simd_width](point)
        )

    def avg_pool_compute_init_cpu[
        simd_width: Int
    ]() {} -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    def avg_pool_compute_cpu[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) {} -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    def avg_pool_compute_finalize_ref[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var h_output_ref,
    }:
        var res = val / (pool_window_h * pool_window_w)
        h_output_ref.store_linear(point, res)

    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
    ](
        output_shape_dyn,
        input_shape_dyn,
        map_fn_cpu,
        dilation_fn_cpu,
        load_fn_ref,
        avg_pool_compute_init_cpu,
        avg_pool_compute_cpu,
        avg_pool_compute_finalize_ref,
    )

    assert_allclose(h_output_ref, h_output)

    for i in range(0, output_height):
        for j in range(0, output_width):
            print(h_output[0, i, j, 0], "\t", end="")
        print("")


def test_stencil_avg_pool_padded(ctx: DeviceContext) raises:
    print("== test_stencil_avg_pool_padded")
    comptime rank = 4
    comptime stencil_rank = 2
    comptime dtype = DType.float32
    comptime simd_width = 1

    comptime input_width = 5
    comptime input_height = 5

    comptime stride = 1
    comptime pool_window_h = 5
    comptime pool_window_w = 5
    comptime dilation = 1
    comptime pad_h = 2
    comptime pad_w = 2

    comptime output_height = input_height - pool_window_h + pad_h * 2 + 1
    comptime output_width = input_width - pool_window_w + pad_w * 2 + 1
    var input_shape_dyn = IndexList[4](1, input_height, input_width, 1)
    var output_shape_dyn = IndexList[4](1, output_height, output_width, 1)

    var d_input_managed = ManagedLayoutTensor[
        dtype, Layout.row_major(1, input_height, input_width, 1)
    ](ctx)
    var d_output_managed = ManagedLayoutTensor[
        dtype, Layout.row_major(1, output_height, output_width, 1)
    ](ctx)

    var h_input = TileTensor(
        d_input_managed.tensor[update=False]().ptr,
        row_major[1, input_height, input_width, 1](),
    )
    var h_output = TileTensor(
        d_output_managed.tensor[update=False]().ptr,
        row_major[1, output_height, output_width, 1](),
    )
    var h_output_ref = stack_allocation[dtype=dtype](
        row_major[1, output_height, output_width, 1]()
    )
    _ = h_output_ref.fill(0)

    fill_buffer(h_input)
    _ = h_output.fill(0)

    var d_input = TileTensor(
        d_input_managed.device_tensor().ptr,
        row_major[1, input_height, input_width, 1](),
    )
    var d_output = TileTensor(
        d_output_managed.device_tensor().ptr,
        row_major[1, output_height, output_width, 1](),
    )

    def map_fn_gpu(
        point: IndexList[stencil_rank, ...],
    ) {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] - pad_h, point[1] - pad_w
        )
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h - pad_h, point[1] + pool_window_w - pad_w
        )
        return lower_bound, upper_bound

    def dilation_fn_gpu(dim: Int) {} -> Int:
        return 1

    @always_inline
    def load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {var d_input,} -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load_linear[width=simd_width](point)
        )

    def avg_pool_compute_init_gpu[
        simd_width: Int
    ]() {} -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    def avg_pool_compute_gpu[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) {} -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    def avg_pool_compute_finalize_gpu[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var d_output,
    }:
        var res = val / (pool_window_h * pool_window_w)
        d_output.store_linear(point, res)

    comptime stencil_axis = IndexList[stencil_rank](1, 2)
    stencil_gpu[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
    ](
        ctx,
        rebind[IndexList[rank]](
            coord_to_index_list(d_output.layout.shape_coord())
        ),
        rebind[IndexList[rank]](
            coord_to_index_list(d_input.layout.shape_coord())
        ),
        map_fn_gpu,
        dilation_fn_gpu,
        load_fn_gpu,
        avg_pool_compute_init_gpu,
        avg_pool_compute_gpu,
        avg_pool_compute_finalize_gpu,
    )

    # Refresh host view; tensor() handles device-to-host transfer and sync.
    h_output = TileTensor(
        d_output_managed.tensor().ptr,
        row_major[1, output_height, output_width, 1](),
    )

    def map_fn_cpu(
        point: IndexList[stencil_rank, ...],
    ) {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] - pad_h, point[1] - pad_w
        )
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h - pad_h, point[1] + pool_window_w - pad_w
        )
        return lower_bound, upper_bound

    def dilation_fn_cpu(dim: Int) {} -> Int:
        return 1

    @always_inline
    def load_fn_ref[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {var h_input,} -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load_linear[width=simd_width](point)
        )

    def avg_pool_compute_init_cpu[
        simd_width: Int
    ]() {} -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    def avg_pool_compute_cpu[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) {} -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    def avg_pool_compute_finalize_ref[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var h_output_ref,
    }:
        var res = val / (pool_window_h * pool_window_w)
        h_output_ref.store_linear(point, res)

    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
    ](
        output_shape_dyn,
        input_shape_dyn,
        map_fn_cpu,
        dilation_fn_cpu,
        load_fn_ref,
        avg_pool_compute_init_cpu,
        avg_pool_compute_cpu,
        avg_pool_compute_finalize_ref,
    )

    # Ensure results match expected
    assert_allclose(h_output_ref, h_output)

    for i in range(0, output_height):
        for j in range(0, output_width):
            print(h_output[0, i, j, 0], "\t", end="")
        print("")


def test_stencil_avg_pool_stride_2(ctx: DeviceContext) raises:
    print("== test_stencil_avg_pool_stride_2")
    comptime rank = 4
    comptime stencil_rank = 2
    comptime dtype = DType.float32
    comptime simd_width = 1

    comptime input_width = 7
    comptime input_height = 7

    comptime stride = 2
    comptime pool_window_h = 3
    comptime pool_window_w = 3
    comptime dilation = 1

    comptime output_height = (input_height - pool_window_h) // stride + 1
    comptime output_width = (input_width - pool_window_w) // stride + 1
    var input_shape_dyn = IndexList[4](1, input_height, input_width, 1)
    var output_shape_dyn = IndexList[4](1, output_height, output_width, 1)

    var d_input_managed = ManagedLayoutTensor[
        dtype, Layout.row_major(1, input_height, input_width, 1)
    ](ctx)
    var d_output_managed = ManagedLayoutTensor[
        dtype, Layout.row_major(1, output_height, output_width, 1)
    ](ctx)

    var h_input = TileTensor(
        d_input_managed.tensor[update=False]().ptr,
        row_major[1, input_height, input_width, 1](),
    )
    var h_output = TileTensor(
        d_output_managed.tensor[update=False]().ptr,
        row_major[1, output_height, output_width, 1](),
    )
    var h_output_ref = stack_allocation[dtype=dtype](
        row_major[1, output_height, output_width, 1]()
    )
    _ = h_output_ref.fill(0)

    fill_buffer(h_input)
    _ = h_output.fill(0)

    var d_input = TileTensor(
        d_input_managed.device_tensor().ptr,
        row_major[1, input_height, input_width, 1](),
    )
    var d_output = TileTensor(
        d_output_managed.device_tensor().ptr,
        row_major[1, output_height, output_width, 1](),
    )

    def map_fn_gpu(
        point: IndexList[stencil_rank, ...],
    ) {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride, point[1] * stride
        )
        var upper_bound = IndexList[stencil_rank](
            point[0] * stride + pool_window_h,
            point[1] * stride + pool_window_w,
        )
        return lower_bound, upper_bound

    def dilation_fn_gpu(dim: Int) {} -> Int:
        return 1

    @always_inline
    def load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {var d_input,} -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load_linear[width=simd_width](point)
        )

    def avg_pool_compute_init_gpu[
        simd_width: Int
    ]() {} -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    def avg_pool_compute_gpu[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) {} -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    def avg_pool_compute_finalize_gpu[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var d_output,
    }:
        var res = val / (pool_window_h * pool_window_w)
        d_output.store_linear(point, res)

    comptime stencil_axis = IndexList[stencil_rank](1, 2)
    stencil_gpu[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
    ](
        ctx,
        rebind[IndexList[rank]](
            coord_to_index_list(d_output.layout.shape_coord())
        ),
        rebind[IndexList[rank]](
            coord_to_index_list(d_input.layout.shape_coord())
        ),
        map_fn_gpu,
        dilation_fn_gpu,
        load_fn_gpu,
        avg_pool_compute_init_gpu,
        avg_pool_compute_gpu,
        avg_pool_compute_finalize_gpu,
    )

    # Refresh host view; tensor() handles device-to-host transfer and sync.
    h_output = TileTensor(
        d_output_managed.tensor().ptr,
        row_major[1, output_height, output_width, 1](),
    )

    def map_fn_cpu(
        point: IndexList[stencil_rank, ...],
    ) {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride, point[1] * stride
        )
        var upper_bound = IndexList[stencil_rank](
            point[0] * stride + pool_window_h,
            point[1] * stride + pool_window_w,
        )
        return lower_bound, upper_bound

    def dilation_fn_cpu(dim: Int) {} -> Int:
        return 1

    @always_inline
    def load_fn_ref[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {var h_input,} -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load_linear[width=simd_width](point)
        )

    def avg_pool_compute_init_cpu[
        simd_width: Int
    ]() {} -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    def avg_pool_compute_cpu[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) {} -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    def avg_pool_compute_finalize_ref[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var h_output_ref,
    }:
        var res = val / (pool_window_h * pool_window_w)
        h_output_ref.store_linear(point, res)

    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
    ](
        output_shape_dyn,
        input_shape_dyn,
        map_fn_cpu,
        dilation_fn_cpu,
        load_fn_ref,
        avg_pool_compute_init_cpu,
        avg_pool_compute_cpu,
        avg_pool_compute_finalize_ref,
    )

    # Ensure results match expected
    assert_allclose(h_output_ref, h_output)

    for i in range(0, output_height):
        for j in range(0, output_width):
            print(h_output[0, i, j, 0], "\t", end="")
        print("")


def test_stencil_gpu_max_pool(ctx: DeviceContext) raises:
    print("== test_stencil_gpu_max_pool")
    comptime rank = 4
    comptime stencil_rank = 2
    comptime dtype = DType.float32
    comptime simd_width = 1

    comptime input_width = 7
    comptime input_height = 7

    comptime stride = 1
    comptime pool_window_h = 3
    comptime pool_window_w = 3
    comptime dilation = 1

    var input_shape_dyn = IndexList[4](1, input_height, input_width, 1)
    comptime output_height = (
        input_height - pool_window_h - (pool_window_h - 1) * (dilation - 1)
    ) // stride + 1
    comptime output_width = (
        input_width - pool_window_w - (pool_window_w - 1) * (dilation - 1)
    ) // stride + 1

    var output_shape_dyn = IndexList[4](1, output_height, output_width, 1)

    var pad_value = 0

    var d_input_managed = ManagedLayoutTensor[
        dtype, Layout.row_major(1, input_height, input_width, 1)
    ](ctx)
    var d_output_managed = ManagedLayoutTensor[
        dtype, Layout.row_major(1, output_height, output_width, 1)
    ](ctx)

    var h_input = TileTensor(
        d_input_managed.tensor[update=False]().ptr,
        row_major[1, input_height, input_width, 1](),
    )
    var h_output = TileTensor(
        d_output_managed.tensor[update=False]().ptr,
        row_major[1, output_height, output_width, 1](),
    )
    var h_output_ref = stack_allocation[dtype=dtype](
        row_major[1, output_height, output_width, 1]()
    )
    _ = h_output_ref.fill(0)

    fill_buffer(h_input)
    _ = h_output.fill(0)

    var d_input = TileTensor(
        d_input_managed.device_tensor().ptr,
        row_major[1, input_height, input_width, 1](),
    )
    var d_output = TileTensor(
        d_output_managed.device_tensor().ptr,
        row_major[1, output_height, output_width, 1](),
    )

    def map_fn_gpu(
        point: IndexList[stencil_rank, ...],
    ) {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride, point[1] * stride
        )
        var upper_bound = IndexList[stencil_rank](
            (point[0] * stride + pool_window_h * dilation),
            (point[1] * stride + pool_window_w * dilation),
        )
        return lower_bound, upper_bound

    def dilation_fn_gpu(dim: Int) {} -> Int:
        return dilation

    @always_inline
    def load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {var d_input,} -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load_linear[width=simd_width](point)
        )

    def max_pool_compute_init_gpu[
        simd_width: Int
    ]() {} -> SIMD[dtype, simd_width]:
        return min_or_neg_inf[dtype]()

    def max_pool_compute_gpu[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) {} -> SIMD[dtype, simd_width]:
        return max(val, result)

    @always_inline
    def max_pool_compute_finalize_gpu[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var d_output,
    }:
        d_output.store_linear(point, val)

    comptime stencil_axis = IndexList[stencil_rank](1, 2)
    stencil_gpu[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
    ](
        ctx,
        rebind[IndexList[rank]](
            coord_to_index_list(d_output.layout.shape_coord())
        ),
        rebind[IndexList[rank]](
            coord_to_index_list(d_input.layout.shape_coord())
        ),
        map_fn_gpu,
        dilation_fn_gpu,
        load_fn_gpu,
        max_pool_compute_init_gpu,
        max_pool_compute_gpu,
        max_pool_compute_finalize_gpu,
    )

    # Refresh host view; tensor() handles device-to-host transfer and sync.
    h_output = TileTensor(
        d_output_managed.tensor().ptr,
        row_major[1, output_height, output_width, 1](),
    )

    def map_fn_cpu(
        point: IndexList[stencil_rank, ...],
    ) {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride, point[1] * stride
        )
        var upper_bound = IndexList[stencil_rank](
            (point[0] * stride + pool_window_h * dilation),
            (point[1] * stride + pool_window_w * dilation),
        )
        return lower_bound, upper_bound

    def dilation_fn_cpu(dim: Int) {} -> Int:
        return dilation

    @always_inline
    def load_fn_ref[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {var h_input,} -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load_linear[width=simd_width](point)
        )

    def max_pool_compute_init_cpu[
        simd_width: Int
    ]() {} -> SIMD[dtype, simd_width]:
        return min_or_neg_inf[dtype]()

    def max_pool_compute_cpu[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) {} -> SIMD[dtype, simd_width]:
        return max(val, result)

    @always_inline
    def max_pool_compute_finalize_ref[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var h_output_ref,
    }:
        h_output_ref.store_linear(point, val)

    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
    ](
        output_shape_dyn,
        input_shape_dyn,
        map_fn_cpu,
        dilation_fn_cpu,
        load_fn_ref,
        max_pool_compute_init_cpu,
        max_pool_compute_cpu,
        max_pool_compute_finalize_ref,
    )

    # Ensure results match expected
    assert_allclose(h_output_ref, h_output)

    for i in range(0, output_height):
        for j in range(0, output_width):
            print(h_output[0, i, j, 0], "\t", end="")
        print("")


def main() raises:
    with DeviceContext() as ctx:
        test_stencil_avg_pool(ctx)
        test_stencil_avg_pool_padded(ctx)
        test_stencil_avg_pool_stride_2(ctx)
        test_stencil_gpu_max_pool(ctx)
