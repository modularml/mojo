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

"""
Comprehensive test coverage for GPU concat kernel.

This file tests various code paths in nn/concat.mojo:
1. Device-to-device copy path (outer_dims == 1, no epilogue)
2. Inner most single dim specialized kernel
3. General elementwise path (various axes and shapes)
4. Fused concat with epilogue functions
5. Different data types and tensor ranks
"""

from std.collections import Optional

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.concat import (
    _concat_gpu,
    concat,
    elementwise_epilogue_type,
    fused_concat,
)

from std.testing import assert_equal

from std.utils import IndexList, StaticTuple
from std.utils.index import product


def test_concat_d2d_copy_path(ctx: DeviceContext) raises:
    """Test the device-to-device copy optimization path.

    This path is taken when outer_dims == 1 and no epilogue function.
    """
    print("== test_concat_d2d_copy_path")

    comptime dtype = DType.float32
    comptime rank = 2

    # Shape that ensures outer_dims == 1 (concatenating along axis 0)
    comptime l1 = row_major[2, 128]()
    comptime l2 = row_major[3, 128]()
    comptime l3 = row_major[5, 128]()
    comptime out_layout = row_major[10, 128]()

    var input_0_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        l1.product()
    )
    var input_1_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        l2.product()
    )
    var input_2_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        l3.product()
    )
    var output_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        out_layout.product()
    )

    # Fill with different values
    for i in range(l1.product()):
        input_0_host_buffer[i] = Float32(i)
    for i in range(l2.product()):
        input_1_host_buffer[i] = Float32(i + 1000)
    for i in range(l3.product()):
        input_2_host_buffer[i] = Float32(i + 2000)

    var input_0_device_buffer = ctx.enqueue_create_buffer[dtype](l1.product())
    var input_1_device_buffer = ctx.enqueue_create_buffer[dtype](l2.product())
    var input_2_device_buffer = ctx.enqueue_create_buffer[dtype](l3.product())
    var output_device_buffer = ctx.enqueue_create_buffer[dtype](
        out_layout.product()
    )

    ctx.enqueue_copy(input_0_device_buffer, input_0_host_buffer)
    ctx.enqueue_copy(input_1_device_buffer, input_1_host_buffer)
    ctx.enqueue_copy(input_2_device_buffer, input_2_host_buffer)
    ctx.synchronize()

    var input_0_dyn = TileTensor(
        input_0_device_buffer,
        row_major(Coord(IndexList[rank](2, 128))),
    )
    var input_1_dyn = TileTensor(
        input_1_device_buffer,
        row_major(Coord(IndexList[rank](3, 128))),
    )
    var input_2_dyn = TileTensor(
        input_2_device_buffer,
        row_major(Coord(IndexList[rank](5, 128))),
    )
    var output_dyn = TileTensor(
        output_device_buffer,
        row_major(Coord(IndexList[rank](10, 128))),
    )

    # This should take the d2d copy path
    _concat_gpu[epilogue_fn=None](
        output_dyn.as_unsafe_any_origin(),
        0,  # axis=0 makes outer_dims=1
        StaticTuple[
            TileTensor[
                dtype,
                input_0_dyn.LayoutType,
                ImmutAnyOrigin,
                Engine=input_0_dyn.Engine,
            ],
            3,
        ](
            input_0_dyn.as_unsafe_any_origin().as_immut(),
            input_1_dyn.as_unsafe_any_origin().as_immut(),
            input_2_dyn.as_unsafe_any_origin().as_immut(),
        ),
        ctx,
    )

    ctx.enqueue_copy(output_host_buffer, output_device_buffer)
    ctx.synchronize()

    # Verify results
    var input_0_host = TileTensor(input_0_host_buffer, l1)
    var input_1_host = TileTensor(input_1_host_buffer, l2)
    var input_2_host = TileTensor(input_2_host_buffer, l3)
    var output_host = TileTensor(output_host_buffer, out_layout)

    var n0 = Int(input_0_host.dim[0]())
    var n1 = Int(input_1_host.dim[0]())
    var n2 = Int(input_2_host.dim[0]())
    var inner = Int(input_0_host.dim[1]())
    var off1 = n0
    var off2 = n0 + n1

    # Rows [0, n0) should match input_0
    for i in range(n0):
        for j in range(inner):
            assert_equal(
                output_host[i, j],
                input_0_host[i, j],
                msg="Mismatch in first input section",
            )

    # Rows [n0, n0+n1) should match input_1
    for i in range(n1):
        for j in range(inner):
            assert_equal(
                output_host[i + off1, j],
                input_1_host[i, j],
                msg="Mismatch in second input section",
            )

    # Rows [n0+n1, ...) should match input_2
    for i in range(n2):
        for j in range(inner):
            assert_equal(
                output_host[i + off2, j],
                input_2_host[i, j],
                msg="Mismatch in third input section",
            )

    print("✅ Test passed!")

    _ = input_0_device_buffer
    _ = input_1_device_buffer
    _ = input_2_device_buffer
    _ = output_device_buffer


def test_concat_non_last_axis(ctx: DeviceContext) raises:
    """Test concatenation along non-last axis (general elementwise path)."""
    print("== test_concat_non_last_axis")

    comptime dtype = DType.float32
    comptime rank = 3
    comptime axis = 1

    comptime l1 = row_major[2, 3, 4]()
    comptime l2 = row_major[2, 5, 4]()
    comptime out_layout = row_major[2, 8, 4]()

    var input_0_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        l1.product()
    )
    var input_1_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        l2.product()
    )
    var output_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        out_layout.product()
    )

    for i in range(l1.product()):
        input_0_host_buffer[i] = Float32(i)
    for i in range(l2.product()):
        input_1_host_buffer[i] = Float32(i + 100)

    var input_0_device_buffer = ctx.enqueue_create_buffer[dtype](l1.product())
    var input_1_device_buffer = ctx.enqueue_create_buffer[dtype](l2.product())
    var output_device_buffer = ctx.enqueue_create_buffer[dtype](
        out_layout.product()
    )

    ctx.enqueue_copy(input_0_device_buffer, input_0_host_buffer)
    ctx.enqueue_copy(input_1_device_buffer, input_1_host_buffer)
    ctx.synchronize()

    var input_0_dyn = TileTensor(
        input_0_device_buffer,
        row_major(Coord(IndexList[rank](2, 3, 4))),
    )
    var input_1_dyn = TileTensor(
        input_1_device_buffer,
        row_major(Coord(IndexList[rank](2, 5, 4))),
    )
    var output_dyn = TileTensor(
        output_device_buffer,
        row_major(Coord(IndexList[rank](2, 8, 4))),
    )

    _concat_gpu[epilogue_fn=None](
        output_dyn.as_unsafe_any_origin(),
        axis,
        StaticTuple[
            TileTensor[
                dtype,
                input_0_dyn.LayoutType,
                ImmutAnyOrigin,
                Engine=input_0_dyn.Engine,
            ],
            2,
        ](
            input_0_dyn.as_unsafe_any_origin().as_immut(),
            input_1_dyn.as_unsafe_any_origin().as_immut(),
        ),
        ctx,
    )

    ctx.enqueue_copy(output_host_buffer, output_device_buffer)
    ctx.synchronize()

    var input_0_host = TileTensor(input_0_host_buffer, l1)
    var input_1_host = TileTensor(input_1_host_buffer, l2)
    var output_host = TileTensor(output_host_buffer, out_layout)

    # Verify concatenation along axis 1
    var d0 = Int(input_0_host.dim[0]())
    var seg0 = Int(input_0_host.dim[1]())
    var seg1 = Int(input_1_host.dim[1]())
    var inner = Int(input_0_host.dim[2]())
    for i in range(d0):
        for j in range(seg0):
            for k in range(inner):
                assert_equal(
                    output_host[i, j, k],
                    input_0_host[i, j, k],
                    msg="Mismatch in first input section",
                )
        for j in range(seg1):
            for k in range(inner):
                assert_equal(
                    output_host[i, j + seg0, k],
                    input_1_host[i, j, k],
                    msg="Mismatch in second input section",
                )

    print("✅ Test passed!")

    _ = input_0_device_buffer
    _ = input_1_device_buffer
    _ = output_device_buffer


def test_concat_last_axis_vectorized(ctx: DeviceContext) raises:
    """Test concatenation along last axis with SIMD vectorization."""
    print("== test_concat_last_axis_vectorized")

    comptime dtype = DType.float32
    comptime rank = 3
    comptime axis = 2

    # Use dimensions aligned to SIMD width for vectorization
    comptime l1 = row_major[2, 3, 8]()  # 8 is aligned
    comptime l2 = row_major[2, 3, 16]()  # 16 is aligned
    comptime out_layout = row_major[2, 3, 24]()

    var input_0_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        l1.product()
    )
    var input_1_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        l2.product()
    )
    var output_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        out_layout.product()
    )

    for i in range(l1.product()):
        input_0_host_buffer[i] = Float32(i)
    for i in range(l2.product()):
        input_1_host_buffer[i] = Float32(i + 200)

    var input_0_device_buffer = ctx.enqueue_create_buffer[dtype](l1.product())
    var input_1_device_buffer = ctx.enqueue_create_buffer[dtype](l2.product())
    var output_device_buffer = ctx.enqueue_create_buffer[dtype](
        out_layout.product()
    )

    ctx.enqueue_copy(input_0_device_buffer, input_0_host_buffer)
    ctx.enqueue_copy(input_1_device_buffer, input_1_host_buffer)
    ctx.synchronize()

    var input_0_dyn = TileTensor(
        input_0_device_buffer,
        row_major(Coord(IndexList[rank](2, 3, 8))),
    )
    var input_1_dyn = TileTensor(
        input_1_device_buffer,
        row_major(Coord(IndexList[rank](2, 3, 16))),
    )
    var output_dyn = TileTensor(
        output_device_buffer,
        row_major(Coord(IndexList[rank](2, 3, 24))),
    )

    _concat_gpu[epilogue_fn=None](
        output_dyn.as_unsafe_any_origin(),
        axis,
        StaticTuple[
            TileTensor[
                dtype,
                input_0_dyn.LayoutType,
                ImmutAnyOrigin,
                Engine=input_0_dyn.Engine,
            ],
            2,
        ](
            input_0_dyn.as_unsafe_any_origin().as_immut(),
            input_1_dyn.as_unsafe_any_origin().as_immut(),
        ),
        ctx,
    )

    ctx.enqueue_copy(output_host_buffer, output_device_buffer)
    ctx.synchronize()

    var input_0_host = TileTensor(input_0_host_buffer, l1)
    var input_1_host = TileTensor(input_1_host_buffer, l2)
    var output_host = TileTensor(output_host_buffer, out_layout)

    var d0 = Int(input_0_host.dim[0]())
    var d1 = Int(input_0_host.dim[1]())
    var w0 = Int(input_0_host.dim[2]())
    var w1 = Int(input_1_host.dim[2]())
    for i in range(d0):
        for j in range(d1):
            for k in range(w0):
                assert_equal(
                    output_host[i, j, k],
                    input_0_host[i, j, k],
                    msg="Mismatch in first input section",
                )
            for k in range(w1):
                assert_equal(
                    output_host[i, j, k + w0],
                    input_1_host[i, j, k],
                    msg="Mismatch in second input section",
                )

    print("✅ Test passed!")

    _ = input_0_device_buffer
    _ = input_1_device_buffer
    _ = output_device_buffer


def test_concat_last_axis_unaligned(ctx: DeviceContext) raises:
    """Test concatenation along last axis with unaligned dimensions (scalar path).
    """
    print("== test_concat_last_axis_unaligned")

    comptime dtype = DType.float32
    comptime rank = 3
    comptime axis = 2

    # Use dimensions NOT aligned to SIMD width to force scalar path
    comptime l1 = row_major[2, 3, 7]()  # 7 is unaligned
    comptime l2 = row_major[2, 3, 11]()  # 11 is unaligned
    comptime out_layout = row_major[2, 3, 18]()

    var input_0_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        l1.product()
    )
    var input_1_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        l2.product()
    )
    var output_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        out_layout.product()
    )

    for i in range(l1.product()):
        input_0_host_buffer[i] = Float32(i)
    for i in range(l2.product()):
        input_1_host_buffer[i] = Float32(i + 300)

    var input_0_device_buffer = ctx.enqueue_create_buffer[dtype](l1.product())
    var input_1_device_buffer = ctx.enqueue_create_buffer[dtype](l2.product())
    var output_device_buffer = ctx.enqueue_create_buffer[dtype](
        out_layout.product()
    )

    ctx.enqueue_copy(input_0_device_buffer, input_0_host_buffer)
    ctx.enqueue_copy(input_1_device_buffer, input_1_host_buffer)
    ctx.synchronize()

    var input_0_dyn = TileTensor(
        input_0_device_buffer,
        row_major(Coord(IndexList[rank](2, 3, 7))),
    )
    var input_1_dyn = TileTensor(
        input_1_device_buffer,
        row_major(Coord(IndexList[rank](2, 3, 11))),
    )
    var output_dyn = TileTensor(
        output_device_buffer,
        row_major(Coord(IndexList[rank](2, 3, 18))),
    )

    _concat_gpu[epilogue_fn=None](
        output_dyn.as_unsafe_any_origin(),
        axis,
        StaticTuple[
            TileTensor[
                dtype,
                input_0_dyn.LayoutType,
                ImmutAnyOrigin,
                Engine=input_0_dyn.Engine,
            ],
            2,
        ](
            input_0_dyn.as_unsafe_any_origin().as_immut(),
            input_1_dyn.as_unsafe_any_origin().as_immut(),
        ),
        ctx,
    )

    ctx.enqueue_copy(output_host_buffer, output_device_buffer)
    ctx.synchronize()

    var input_0_host = TileTensor(input_0_host_buffer, l1)
    var input_1_host = TileTensor(input_1_host_buffer, l2)
    var output_host = TileTensor(output_host_buffer, out_layout)

    var d0 = Int(input_0_host.dim[0]())
    var d1 = Int(input_0_host.dim[1]())
    var w0 = Int(input_0_host.dim[2]())
    var w1 = Int(input_1_host.dim[2]())
    for i in range(d0):
        for j in range(d1):
            for k in range(w0):
                assert_equal(
                    output_host[i, j, k],
                    input_0_host[i, j, k],
                    msg="Mismatch in first input section",
                )
            for k in range(w1):
                assert_equal(
                    output_host[i, j, k + w0],
                    input_1_host[i, j, k],
                    msg="Mismatch in second input section",
                )

    print("✅ Test passed!")

    _ = input_0_device_buffer
    _ = input_1_device_buffer
    _ = output_device_buffer


def test_fused_concat_gpu(ctx: DeviceContext) raises:
    """Test fused concat with epilogue function on GPU."""
    print("== test_fused_concat_gpu")

    comptime dtype = DType.float32
    comptime rank = 3
    comptime axis = 1

    comptime input_shape_0 = IndexList[rank](2, 3, 4)
    comptime input_shape_1 = IndexList[rank](2, 5, 4)
    comptime output_shape = IndexList[rank](2, 8, 4)

    var output_device_buffer = ctx.enqueue_create_buffer[dtype](
        product(output_shape, rank)
    )
    var output_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        product(output_shape, rank)
    )

    var output_dyn = TileTensor(
        output_device_buffer, row_major(Coord(output_shape))
    )

    # Input lambda: generates data on-the-fly
    @__parameter
    @always_inline
    def input_fn[
        input_index: Int, width: Int, _rank: Int, alignment: Int = 1
    ](indices: IndexList[_rank]) -> SIMD[dtype, width]:
        comptime if input_index == 0:
            # First input: return constant 1.0
            return SIMD[dtype, width](1.0)
        else:
            # Second input: return constant 2.0
            return SIMD[dtype, width](2.0)

    # Output epilogue: add 10 to every value
    @__parameter
    @always_inline
    @__copy_capture(output_dyn)
    def output_fn[
        c_type: DType, _rank: Int, width: SIMDLength, *, alignment: Int
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        var coord = Coord(indices)
        comptime assert output_dyn.flat_rank >= coord.flat_rank
        output_dyn.store[width=width](
            coord, rebind[SIMD[dtype, width]](val + 10)
        )

    fused_concat[
        dtype,
        rank,
        input_fn,
        output_fn,
        output_dyn.LayoutType,
        axis=axis,
        target="gpu",
    ](
        StaticTuple[IndexList[rank], 2](input_shape_0, input_shape_1),
        output_dyn.as_unsafe_any_origin(),
        ctx,
    )

    ctx.enqueue_copy(output_host_buffer, output_device_buffer)
    ctx.synchronize()

    var output_host = TileTensor(
        output_host_buffer, row_major(Coord(output_shape))
    )

    # Verify epilogue was applied (+10); bounds follow input_shape_* / output_shape
    var d0 = Int(output_shape[0])
    var seg0 = Int(input_shape_0[1])
    var seg1 = Int(input_shape_1[1])
    var inner = Int(input_shape_0[2])
    for i in range(d0):
        # First input: lambda returns 1.0, epilogue adds 10
        for j in range(seg0):
            for k in range(inner):
                assert_equal(
                    output_host[i, j, k],
                    11.0,  # 1.0 + 10
                    msg="Mismatch in first input section",
                )
        # Second input: lambda returns 2.0, epilogue adds 10
        for j in range(seg1):
            for k in range(inner):
                assert_equal(
                    output_host[i, j + seg0, k],
                    12.0,  # 2.0 + 10
                    msg="Mismatch in second input section",
                )

    print("✅ Test passed!")

    _ = output_device_buffer


def test_concat_with_epilogue(ctx: DeviceContext) raises:
    """Test concat with epilogue function (scales values by 2)."""
    print("== test_concat_with_epilogue")

    comptime dtype = DType.float32
    comptime rank = 2
    comptime axis = 0

    comptime l1 = row_major[3, 16]()
    comptime l2 = row_major[5, 16]()
    comptime out_layout = row_major[8, 16]()

    var input_0_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        l1.product()
    )
    var input_1_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        l2.product()
    )
    var output_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        out_layout.product()
    )

    for i in range(l1.product()):
        input_0_host_buffer[i] = Float32(i)
    for i in range(l2.product()):
        input_1_host_buffer[i] = Float32(i + 100)

    var input_0_device_buffer = ctx.enqueue_create_buffer[dtype](l1.product())
    var input_1_device_buffer = ctx.enqueue_create_buffer[dtype](l2.product())
    var output_device_buffer = ctx.enqueue_create_buffer[dtype](
        out_layout.product()
    )

    ctx.enqueue_copy(input_0_device_buffer, input_0_host_buffer)
    ctx.enqueue_copy(input_1_device_buffer, input_1_host_buffer)
    ctx.synchronize()

    var input_0_dyn = TileTensor(
        input_0_device_buffer,
        row_major(Coord(IndexList[rank](3, 16))),
    )
    var input_1_dyn = TileTensor(
        input_1_device_buffer,
        row_major(Coord(IndexList[rank](5, 16))),
    )
    var output_dyn = TileTensor(
        output_device_buffer,
        row_major(Coord(IndexList[rank](8, 16))),
    )

    @__parameter
    @always_inline
    @__copy_capture(output_dyn)
    def epilogue_scale_by_2[
        c_type: DType, _rank: Int, width: SIMDLength, *, alignment: Int
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        var coord = Coord(indices)
        comptime assert output_dyn.flat_rank >= coord.flat_rank
        output_dyn.store[width=width](
            coord, rebind[SIMD[dtype, width]](val * 2)
        )

    _concat_gpu[
        epilogue_fn=Optional[elementwise_epilogue_type](epilogue_scale_by_2)
    ](
        output_dyn.as_unsafe_any_origin(),
        axis,
        StaticTuple[
            TileTensor[
                dtype,
                input_0_dyn.LayoutType,
                ImmutAnyOrigin,
                Engine=input_0_dyn.Engine,
            ],
            2,
        ](
            input_0_dyn.as_unsafe_any_origin().as_immut(),
            input_1_dyn.as_unsafe_any_origin().as_immut(),
        ),
        ctx,
    )

    ctx.enqueue_copy(output_host_buffer, output_device_buffer)
    ctx.synchronize()

    var input_0_host = TileTensor(input_0_host_buffer, l1)
    var input_1_host = TileTensor(input_1_host_buffer, l2)
    var output_host = TileTensor(output_host_buffer, out_layout)

    # Verify values are scaled by 2
    var r0 = Int(input_0_host.dim[0]())
    var r1 = Int(input_1_host.dim[0]())
    var w = Int(input_0_host.dim[1]())
    for i in range(r0):
        for j in range(w):
            assert_equal(
                output_host[i, j],
                input_0_host[i, j] * 2,
                msg="Mismatch in first input section",
            )

    for i in range(r1):
        for j in range(w):
            assert_equal(
                output_host[i + r0, j],
                input_1_host[i, j] * 2,
                msg="Mismatch in second input section",
            )

    print("✅ Test passed!")

    _ = input_0_device_buffer
    _ = input_1_device_buffer
    _ = output_device_buffer


def test_concat_different_dtypes(ctx: DeviceContext) raises:
    """Test concat with different data types."""
    print("== test_concat_different_dtypes")

    # Test with float16
    comptime dtype = DType.float16
    comptime rank = 2
    comptime axis = 1

    comptime l1 = row_major[4, 8]()
    comptime l2 = row_major[4, 12]()
    comptime out_layout = row_major[4, 20]()

    var input_0_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        l1.product()
    )
    var input_1_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        l2.product()
    )
    var output_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        out_layout.product()
    )

    for i in range(l1.product()):
        input_0_host_buffer[i] = Float16(i)
    for i in range(l2.product()):
        input_1_host_buffer[i] = Float16(i + 50)

    var input_0_device_buffer = ctx.enqueue_create_buffer[dtype](l1.product())
    var input_1_device_buffer = ctx.enqueue_create_buffer[dtype](l2.product())
    var output_device_buffer = ctx.enqueue_create_buffer[dtype](
        out_layout.product()
    )

    ctx.enqueue_copy(input_0_device_buffer, input_0_host_buffer)
    ctx.enqueue_copy(input_1_device_buffer, input_1_host_buffer)
    ctx.synchronize()

    var input_0_dyn = TileTensor(
        input_0_device_buffer,
        row_major(Coord(IndexList[rank](4, 8))),
    )
    var input_1_dyn = TileTensor(
        input_1_device_buffer,
        row_major(Coord(IndexList[rank](4, 12))),
    )
    var output_dyn = TileTensor(
        output_device_buffer,
        row_major(Coord(IndexList[rank](4, 20))),
    )

    _concat_gpu[epilogue_fn=None](
        output_dyn.as_unsafe_any_origin(),
        axis,
        StaticTuple[
            TileTensor[
                dtype,
                input_0_dyn.LayoutType,
                ImmutAnyOrigin,
                Engine=input_0_dyn.Engine,
            ],
            2,
        ](
            input_0_dyn.as_unsafe_any_origin().as_immut(),
            input_1_dyn.as_unsafe_any_origin().as_immut(),
        ),
        ctx,
    )

    ctx.enqueue_copy(output_host_buffer, output_device_buffer)
    ctx.synchronize()

    var input_0_host = TileTensor(input_0_host_buffer, l1)
    var input_1_host = TileTensor(input_1_host_buffer, l2)
    var output_host = TileTensor(output_host_buffer, out_layout)

    var d0 = Int(input_0_host.dim[0]())
    var w0 = Int(input_0_host.dim[1]())
    var w1 = Int(input_1_host.dim[1]())
    for i in range(d0):
        for j in range(w0):
            assert_equal(
                output_host[i, j],
                input_0_host[i, j],
                msg="Mismatch in first input section",
            )
        for j in range(w1):
            assert_equal(
                output_host[i, j + w0],
                input_1_host[i, j],
                msg="Mismatch in second input section",
            )

    print("✅ Test passed!")

    _ = input_0_device_buffer
    _ = input_1_device_buffer
    _ = output_device_buffer


def test_concat_high_rank(ctx: DeviceContext) raises:
    """Test concat with high-rank tensors (rank 5)."""
    print("== test_concat_high_rank")

    comptime dtype = DType.float32
    comptime rank = 5
    comptime axis = 2

    # Use dimensions aligned to SIMD width (4) for innermost dimension
    # Inner product: 4*8=32 elements, 128 bytes (32*4), 16-byte aligned
    comptime l1 = row_major[2, 3, 4, 4, 8]()
    comptime l2 = row_major[2, 3, 7, 4, 8]()
    comptime out_layout = row_major[2, 3, 11, 4, 8]()

    var input_0_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        l1.product()
    )
    var input_1_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        l2.product()
    )
    var output_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        out_layout.product()
    )

    for i in range(l1.product()):
        input_0_host_buffer[i] = Float32(i % 100)
    for i in range(l2.product()):
        input_1_host_buffer[i] = Float32((i + 50) % 100)

    var input_0_device_buffer = ctx.enqueue_create_buffer[dtype](l1.product())
    var input_1_device_buffer = ctx.enqueue_create_buffer[dtype](l2.product())
    var output_device_buffer = ctx.enqueue_create_buffer[dtype](
        out_layout.product()
    )

    ctx.enqueue_copy(input_0_device_buffer, input_0_host_buffer)
    ctx.enqueue_copy(input_1_device_buffer, input_1_host_buffer)
    ctx.synchronize()

    var input_0_dyn = TileTensor(
        input_0_device_buffer,
        row_major(Coord(IndexList[rank](2, 3, 4, 4, 8))),
    )
    var input_1_dyn = TileTensor(
        input_1_device_buffer,
        row_major(Coord(IndexList[rank](2, 3, 7, 4, 8))),
    )
    var output_dyn = TileTensor(
        output_device_buffer,
        row_major(Coord(IndexList[rank](2, 3, 11, 4, 8))),
    )

    _concat_gpu[epilogue_fn=None](
        output_dyn.as_unsafe_any_origin(),
        axis,
        StaticTuple[
            TileTensor[
                dtype,
                input_0_dyn.LayoutType,
                ImmutAnyOrigin,
                Engine=input_0_dyn.Engine,
            ],
            2,
        ](
            input_0_dyn.as_unsafe_any_origin().as_immut(),
            input_1_dyn.as_unsafe_any_origin().as_immut(),
        ),
        ctx,
    )

    ctx.enqueue_copy(output_host_buffer, output_device_buffer)
    ctx.synchronize()

    var input_0_host = TileTensor(input_0_host_buffer, l1)
    var input_1_host = TileTensor(input_1_host_buffer, l2)
    var output_host = TileTensor(output_host_buffer, out_layout)

    # Sample verification (full verification would be too verbose)
    var d0 = Int(input_0_host.dim[0]())
    var d1 = Int(input_0_host.dim[1]())
    var seg0 = Int(input_0_host.dim[2]())
    var seg1 = Int(input_1_host.dim[2]())
    for i in range(d0):
        for j in range(d1):
            for k in range(seg0):
                assert_equal(
                    output_host[i, j, k, 0, 0],
                    input_0_host[i, j, k, 0, 0],
                    msg="Mismatch in first input section",
                )
            for k in range(seg1):
                assert_equal(
                    output_host[i, j, k + seg0, 0, 0],
                    input_1_host[i, j, k, 0, 0],
                    msg="Mismatch in second input section",
                )

    print("✅ Test passed!")

    _ = input_0_device_buffer
    _ = input_1_device_buffer
    _ = output_device_buffer


def main() raises:
    with DeviceContext() as ctx:
        test_concat_d2d_copy_path(ctx)
        test_concat_non_last_axis(ctx)
        test_concat_last_axis_vectorized(ctx)
        test_concat_last_axis_unaligned(ctx)
        test_fused_concat_gpu(ctx)
        test_concat_with_epilogue(ctx)
        test_concat_different_dtypes(ctx)
        test_concat_high_rank(ctx)

    print("\n🎉 All tests passed!")
