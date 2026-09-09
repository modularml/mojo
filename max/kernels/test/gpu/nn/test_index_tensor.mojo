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

# GPU smoke test for the advanced-indexing ops (`advanced_indexing_getitem` /
# `advanced_indexing_setitem_inplace` in `nn/index_tensor.mojo`). These ops were
# migrated to unified `{var}` closures (GPUA-23, PR #91716); the whole
# `{var}` + `ImplicitlyCopyable & RegisterPassable` machinery exists so those
# closures can be marshaled to the GPU via `elementwise` with `target="gpu"`.
# The CPU test (`test/nn/test_index_tensor.mojo`) only runs `target="cpu"`, so
# this test closes the device-path coverage gap. Shapes/data/index-patterns and
# reference outputs mirror the CPU test EXACTLY.
#
# Target: any GPU (single-aggregate TileTensor captures only; index rebuilt
# in-kernel from the `IndexList` arg, so no bare-`Coord`-across-launch trap).

from max.gpu.host import DeviceContext
from std.math import align_up
from std.sys import simd_width_of
from std.testing import assert_equal
from std.utils import IndexList

from layout import Coord, TileTensor, row_major
from nn.index_tensor import (
    advanced_indexing_getitem,
    advanced_indexing_getitem_shape,
    advanced_indexing_setitem_inplace,
)


# Matches equivalent numpy: input[:, :, index_a, index_b]
def test_advanced_indexing_getitem_gpu(ctx: DeviceContext) raises:
    print("== test_advanced_indexing_getitem_gpu")

    comptime input_type = DType.int32
    comptime input_rank = 4
    comptime input_shape = IndexList[input_rank](2, 3, 5, 6)
    comptime input_static_layout = row_major[2, 3, 5, 6]()

    comptime index_rank = 2
    comptime index_shape = IndexList[index_rank](2, 3)
    comptime index_type = DType.uint64
    comptime index_static_layout = row_major[2, 3]()

    comptime start_axis = 2
    comptime num_index_tensors = 2
    comptime output_shape = advanced_indexing_getitem_shape[
        start_axis=start_axis, num_index_tensors=num_index_tensors
    ](input_shape, index_shape)
    comptime output_static_layout = row_major[2, 3, 2, 3]()

    # ===== Create host + device buffers =====
    var input_host = ctx.enqueue_create_host_buffer[input_type](
        input_static_layout.static_product
    )
    var input_dev = ctx.enqueue_create_buffer[input_type](
        input_static_layout.static_product
    )
    var a_host = ctx.enqueue_create_host_buffer[index_type](
        index_static_layout.static_product
    )
    var a_dev = ctx.enqueue_create_buffer[index_type](
        index_static_layout.static_product
    )
    var b_host = ctx.enqueue_create_host_buffer[index_type](
        index_static_layout.static_product
    )
    var b_dev = ctx.enqueue_create_buffer[index_type](
        index_static_layout.static_product
    )
    var out_dev = ctx.enqueue_create_buffer[input_type](
        output_static_layout.static_product
    )
    var out_host = ctx.enqueue_create_host_buffer[input_type](
        output_static_layout.static_product
    )
    ctx.synchronize()

    # ===== Fill host buffers (same data as the CPU test) =====
    for i in range(input_shape.flattened_length()):
        input_host[i] = Int32(i)
    for i in range(index_shape.flattened_length()):
        a_host[i] = UInt64(i % 5)
        b_host[i] = UInt64((i + 1) % 5)

    # ===== Copy inputs to device =====
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.synchronize()

    # ===== Build dynamic device TileTensor views =====
    var input_dyn = TileTensor(input_dev, input_static_layout).make_dynamic[
        DType.int64
    ]()
    var index_a_dyn = TileTensor(a_dev, index_static_layout).make_dynamic[
        DType.int64
    ]()
    var index_b_dyn = TileTensor(b_dev, index_static_layout).make_dynamic[
        DType.int64
    ]()
    var output_dyn = TileTensor(out_dev, output_static_layout).make_dynamic[
        DType.int64
    ]()

    # Copy-capture the DEVICE views (`{var ...}`) so the GPU kernel dereferences
    # device memory; the index is rebuilt in-kernel from the `IndexList` arg.
    @always_inline
    def input_tensor_fn[
        dtype: DType, width: Int
    ](idx: IndexList[input_rank]) {var input_dyn} -> SIMD[dtype, width]:
        return rebind[SIMD[dtype, width]](
            input_dyn.load[width=width, alignment=1](Coord(idx))
        )

    @always_inline
    def indices_fn[
        indices_index: Int,
    ](coordinates: IndexList[index_rank]) {
        var index_a_dyn, var index_b_dyn
    } -> Int:
        comptime if indices_index == 0:
            return Int(index_a_dyn.load[width=1](Coord(coordinates)))
        else:
            return Int(index_b_dyn.load[width=1](Coord(coordinates)))

    var in_strides = IndexList[input_rank](
        Int(input_dyn.dynamic_stride(0)),
        Int(input_dyn.dynamic_stride(1)),
        Int(input_dyn.dynamic_stride(2)),
        Int(input_dyn.dynamic_stride(3)),
    )

    advanced_indexing_getitem[
        input_rank=input_rank,
        start_axis=start_axis,
        num_index_tensors=num_index_tensors,
        target="gpu",
        trace_description="test_advanced_indexing_getitem_gpu",
    ](
        output_dyn,
        in_strides,
        ctx,
        input_tensor_fn,
        indices_fn,
    )

    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    # ===== Reference output (identical to CPU test) =====
    var ref_stack = Array[
        Scalar[input_type],
        align_up(output_shape.flattened_length(), simd_width_of[input_type]()),
    ](uninitialized=True)
    var reference_output = TileTensor(ref_stack, output_static_layout)

    reference_output[0, 0, 0, 0] = 1
    reference_output[0, 0, 0, 1] = 8
    reference_output[0, 0, 0, 2] = 15
    reference_output[0, 0, 1, 0] = 22
    reference_output[0, 0, 1, 1] = 24
    reference_output[0, 0, 1, 2] = 1

    reference_output[0, 1, 0, 0] = 31
    reference_output[0, 1, 0, 1] = 38
    reference_output[0, 1, 0, 2] = 45
    reference_output[0, 1, 1, 0] = 52
    reference_output[0, 1, 1, 1] = 54
    reference_output[0, 1, 1, 2] = 31

    reference_output[0, 2, 0, 0] = 61
    reference_output[0, 2, 0, 1] = 68
    reference_output[0, 2, 0, 2] = 75
    reference_output[0, 2, 1, 0] = 82
    reference_output[0, 2, 1, 1] = 84
    reference_output[0, 2, 1, 2] = 61

    reference_output[1, 0, 0, 0] = 91
    reference_output[1, 0, 0, 1] = 98
    reference_output[1, 0, 0, 2] = 105
    reference_output[1, 0, 1, 0] = 112
    reference_output[1, 0, 1, 1] = 114
    reference_output[1, 0, 1, 2] = 91

    reference_output[1, 1, 0, 0] = 121
    reference_output[1, 1, 0, 1] = 128
    reference_output[1, 1, 0, 2] = 135
    reference_output[1, 1, 1, 0] = 142
    reference_output[1, 1, 1, 1] = 144
    reference_output[1, 1, 1, 2] = 121

    reference_output[1, 2, 0, 0] = 151
    reference_output[1, 2, 0, 1] = 158
    reference_output[1, 2, 0, 2] = 165
    reference_output[1, 2, 1, 0] = 172
    reference_output[1, 2, 1, 1] = 174
    reference_output[1, 2, 1, 2] = 151

    for i in range(output_shape.flattened_length()):
        assert_equal(out_host[i], ref_stack[i])

    _ = ref_stack^


# Matches equivalent numpy: input[:, :, index_a, index_b] = updates
def test_advanced_indexing_setitem_inplace_gpu(ctx: DeviceContext) raises:
    print("== test_advanced_indexing_setitem_inplace_gpu")

    comptime input_type = DType.int32
    comptime input_rank = 4
    comptime input_shape = IndexList[input_rank](2, 2, 4, 4)
    comptime input_static_layout = row_major[2, 2, 4, 4]()

    comptime index_rank = 2
    comptime num_index_tensors = 2
    comptime index_shape = IndexList[index_rank](2, 2)
    comptime index_type = DType.uint64
    comptime index_static_layout = row_major[2, 2]()

    comptime updates_rank = 4
    comptime updates_shape = IndexList[updates_rank](2, 2, 2, 2)
    comptime updates_static_layout = row_major[2, 2, 2, 2]()

    comptime start_axis = 2

    # ===== Create host + device buffers =====
    var input_host = ctx.enqueue_create_host_buffer[input_type](
        input_static_layout.static_product
    )
    var input_dev = ctx.enqueue_create_buffer[input_type](
        input_static_layout.static_product
    )
    var a_host = ctx.enqueue_create_host_buffer[index_type](
        index_static_layout.static_product
    )
    var a_dev = ctx.enqueue_create_buffer[index_type](
        index_static_layout.static_product
    )
    var b_host = ctx.enqueue_create_host_buffer[index_type](
        index_static_layout.static_product
    )
    var b_dev = ctx.enqueue_create_buffer[index_type](
        index_static_layout.static_product
    )
    var updates_host = ctx.enqueue_create_host_buffer[input_type](
        updates_static_layout.static_product
    )
    var updates_dev = ctx.enqueue_create_buffer[input_type](
        updates_static_layout.static_product
    )
    var out_host = ctx.enqueue_create_host_buffer[input_type](
        input_static_layout.static_product
    )
    ctx.synchronize()

    # ===== Fill host buffers (same data as the CPU test) =====
    # input starts as all-zeros.
    for i in range(input_shape.flattened_length()):
        input_host[i] = 0
    # Indices: no pair points to the same location to avoid nondeterminism.
    for i in range(index_shape.flattened_length()):
        a_host[i] = UInt64(i % 4)
        b_host[i] = UInt64((i + 1) % 4)
    for i in range(updates_shape.flattened_length()):
        updates_host[i] = Int32(1 + i)

    # ===== Copy inputs to device =====
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(updates_dev, updates_host)
    ctx.synchronize()

    # ===== Build dynamic device TileTensor views =====
    var input_dyn = TileTensor(input_dev, input_static_layout).make_dynamic[
        DType.int64
    ]()
    var index_a_dyn = TileTensor(a_dev, index_static_layout).make_dynamic[
        DType.int64
    ]()
    var index_b_dyn = TileTensor(b_dev, index_static_layout).make_dynamic[
        DType.int64
    ]()
    var updates_dyn = TileTensor(
        updates_dev, updates_static_layout
    ).make_dynamic[.int64]()

    @always_inline
    def updates_tensor_fn[
        dtype: DType, width: Int
    ](idx: IndexList[updates_rank]) {var updates_dyn} -> SIMD[dtype, width]:
        return rebind[SIMD[dtype, width]](
            updates_dyn.load[width=width, alignment=1](Coord(idx))
        )

    @always_inline
    def indices_fn[
        indices_index: Int,
    ](coordinates: IndexList[index_rank]) {
        var index_a_dyn, var index_b_dyn
    } -> Int:
        comptime if indices_index == 0:
            return Int(index_a_dyn.load[width=1](Coord(coordinates)))
        else:
            return Int(index_b_dyn.load[width=1](Coord(coordinates)))

    var idx_shape = IndexList[index_rank](
        Int(index_a_dyn.dim(0)), Int(index_a_dyn.dim(1))
    )
    var upd_strides = IndexList[updates_rank](
        Int(updates_dyn.dynamic_stride(0)),
        Int(updates_dyn.dynamic_stride(1)),
        Int(updates_dyn.dynamic_stride(2)),
        Int(updates_dyn.dynamic_stride(3)),
    )

    advanced_indexing_setitem_inplace[
        index_rank=index_rank,
        start_axis=start_axis,
        num_index_tensors=num_index_tensors,
        target="gpu",
        trace_description="test_advanced_indexing_setitem_inplace_gpu",
    ](
        input_dyn,
        idx_shape,
        upd_strides,
        ctx,
        updates_tensor_fn,
        indices_fn,
    )

    # `input_dev` was mutated in place on device; copy it back.
    ctx.enqueue_copy(out_host, input_dev)
    ctx.synchronize()

    # ===== Reference output (identical to CPU test) =====
    var ref_stack = Array[
        Scalar[input_type],
        align_up(input_shape.flattened_length(), simd_width_of[input_type]()),
    ](uninitialized=True)
    var reference_output = TileTensor(ref_stack, input_static_layout)
    for i in range(input_shape.flattened_length()):
        ref_stack[i] = 0

    reference_output[0, 0, 0, 1] = 1
    reference_output[0, 0, 1, 2] = 2
    reference_output[0, 0, 2, 3] = 3
    reference_output[0, 0, 3, 0] = 4

    reference_output[0, 1, 0, 1] = 5
    reference_output[0, 1, 1, 2] = 6
    reference_output[0, 1, 2, 3] = 7
    reference_output[0, 1, 3, 0] = 8

    reference_output[1, 0, 0, 1] = 9
    reference_output[1, 0, 1, 2] = 10
    reference_output[1, 0, 2, 3] = 11
    reference_output[1, 0, 3, 0] = 12

    reference_output[1, 1, 0, 1] = 13
    reference_output[1, 1, 1, 2] = 14
    reference_output[1, 1, 2, 3] = 15
    reference_output[1, 1, 3, 0] = 16

    for i in range(input_shape.flattened_length()):
        assert_equal(out_host[i], ref_stack[i])

    _ = ref_stack^


def main() raises:
    with DeviceContext() as ctx:
        test_advanced_indexing_getitem_gpu(ctx)
        test_advanced_indexing_setitem_inplace_gpu(ctx)
