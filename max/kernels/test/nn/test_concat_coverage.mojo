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
Comprehensive test coverage for CPU concat kernel.

This file tests various code paths in nn/concat.mojo:
1. Serial concat path (_concat_serial)
2. Parallel concat path (_concat_parallel)
3. Inner concat path (_concat_inner)
4. Fused concat with input/output lambdas
5. Different axes, ranks, and tensor shapes
6. Edge cases: empty outer dims, single element inputs, etc.
"""

from max.gpu.host import DeviceContext
from layout import Coord, TensorLayout, TileTensor, row_major
from nn.concat import (
    _concat_inner,
    _concat_parallel,
    _concat_serial,
    concat,
    concat_shape,
    fused_concat,
)

from std.testing import assert_equal
from std.utils import IndexList, StaticTuple
from std.utils.index import product


def _tuple_to_list[
    LayoutType: TensorLayout,
    //,
    dtype: DType,
](
    elems: StaticTuple[
        TileTensor[dtype, LayoutType, ImmutAnyOrigin],
        ...,
    ]
) -> List[TileTensor[dtype, LayoutType, ImmutAnyOrigin]]:
    var output = List[TileTensor[dtype, LayoutType, ImmutAnyOrigin]](
        capacity=len(elems)
    )
    for i in range(len(elems)):
        output.append(elems[i].as_immut())
    return output^


def test_concat_inner_all_outer_dims_singleton() raises:
    """Test inner concat path when all outer dims are 1."""
    print("== test_concat_inner_all_outer_dims_singleton")

    comptime dtype = DType.float32
    comptime rank = 4
    comptime axis = 2  # All dims before this are 1

    # All outer dims are 1, triggering inner concat path
    comptime l1 = row_major[1, 1, 3, 4]()
    comptime l2 = row_major[1, 1, 5, 4]()
    comptime out_layout = row_major[1, 1, 8, 4]()

    var x1_stack = Array[Scalar[dtype], l1.product()](
        fill_with=lambda (i: Int) -> Scalar[dtype]: Float32(i)
    )
    var x2_stack = Array[Scalar[dtype], l2.product()](
        fill_with=lambda (i: Int) -> Scalar[dtype]: Float32(i + 400)
    )
    var out_stack = Array[Scalar[dtype], out_layout.product()](fill=-1)

    var x1 = TileTensor(x1_stack, l1)
    var x2 = TileTensor(x2_stack, l2)
    var output = TileTensor(out_stack, out_layout)

    var x1_dyn = x1.make_dynamic[.int64]()
    var x2_dyn = x2.make_dynamic[.int64]()

    var input_vec = List[TileTensor[dtype, x1_dyn.LayoutType, ImmutAnyOrigin]](
        capacity=2
    )
    input_vec.append(x1_dyn.as_unsafe_any_origin().as_immut())
    input_vec.append(x2_dyn.as_unsafe_any_origin().as_immut())

    _concat_inner[dtype, None](output.make_dynamic[.int64](), input_vec)

    # Verify contiguous concatenation
    for i in range(l1.product()):
        assert_equal(
            output.raw_load(i), x1.raw_load(i), msg="Mismatch in first input"
        )
    for i in range(l2.product()):
        assert_equal(
            output.raw_load(l1.product() + i),
            x2.raw_load(i),
            msg="Mismatch in second input",
        )

    print("✅ Test passed!")


def test_concat_serial_general_case() raises:
    """Test serial concat with general tensor shapes."""
    print("== test_concat_serial_general_case")

    comptime dtype = DType.float32
    comptime rank = 3
    comptime axis = 1

    comptime l1 = row_major[4, 2, 5]()
    comptime l2 = row_major[4, 3, 5]()
    comptime l3 = row_major[4, 4, 5]()
    comptime out_layout = row_major[4, 9, 5]()

    var x1_stack = Array[Scalar[dtype], l1.product()](
        fill_with=lambda (i: Int) -> Scalar[dtype]: Float32(i)
    )
    var x2_stack = Array[Scalar[dtype], l2.product()](
        fill_with=lambda (i: Int) -> Scalar[dtype]: Float32(i + 500)
    )
    var x3_stack = Array[Scalar[dtype], l3.product()](
        fill_with=lambda (i: Int) -> Scalar[dtype]: Float32(i + 600)
    )
    var out_stack = Array[Scalar[dtype], out_layout.product()](fill=-1)

    var x1 = TileTensor(x1_stack, l1)
    var x2 = TileTensor(x2_stack, l2)
    var x3 = TileTensor(x3_stack, l3)
    var output = TileTensor(out_stack, out_layout)

    var x1_dyn = x1.make_dynamic[.int64]()
    var x2_dyn = x2.make_dynamic[.int64]()
    var x3_dyn = x3.make_dynamic[.int64]()

    var input_vec = List[TileTensor[dtype, x1_dyn.LayoutType, ImmutAnyOrigin]](
        capacity=3
    )
    input_vec.append(x1_dyn.as_unsafe_any_origin().as_immut())
    input_vec.append(x2_dyn.as_unsafe_any_origin().as_immut())
    input_vec.append(x3_dyn.as_unsafe_any_origin().as_immut())

    _concat_serial[dtype, None](output.make_dynamic[.int64](), axis, input_vec)

    # Verify concatenation
    for i in range(4):
        for j in range(2):
            for k in range(5):
                assert_equal(output[i, j, k], x1[i, j, k], msg="Mismatch in x1")
        for j in range(3):
            for k in range(5):
                assert_equal(
                    output[i, j + 2, k], x2[i, j, k], msg="Mismatch in x2"
                )
        for j in range(4):
            for k in range(5):
                assert_equal(
                    output[i, j + 5, k], x3[i, j, k], msg="Mismatch in x3"
                )

    print("✅ Test passed!")


def test_concat_parallel_large() raises:
    """Test parallel concat with large enough data to trigger parallelization.
    """
    print("== test_concat_parallel_large")

    comptime dtype = DType.float32
    comptime rank = 2
    comptime axis = 0

    # Large enough to exceed min_work_for_parallel threshold (128KB)
    # 256 * 512 * 4 bytes = 512KB per input
    comptime l1 = row_major[256, 512]()
    comptime l2 = row_major[256, 512]()
    comptime out_layout = row_major[512, 512]()

    var x1_stack = Array[Scalar[dtype], l1.product()](
        fill_with=lambda (idx: Int) -> Scalar[dtype]: Float32(idx // 512)
    )
    var x2_stack = Array[Scalar[dtype], l2.product()](
        fill_with=lambda (idx: Int) -> Scalar[dtype]: Float32(idx // 512 + 256)
    )
    var out_stack = Array[Scalar[dtype], out_layout.product()](fill=-1)

    var x1 = TileTensor(x1_stack, l1)
    var x2 = TileTensor(x2_stack, l2)
    var output = TileTensor(out_stack, out_layout)

    var x1_dyn = x1.make_dynamic[.int64]()
    var x2_dyn = x2.make_dynamic[.int64]()

    var input_vec = List[TileTensor[dtype, x1_dyn.LayoutType, ImmutAnyOrigin]](
        capacity=2
    )
    input_vec.append(x1_dyn.as_unsafe_any_origin().as_immut())
    input_vec.append(x2_dyn.as_unsafe_any_origin().as_immut())

    _concat_parallel[dtype, None](
        output.make_dynamic[.int64](), axis, input_vec
    )

    # Sample verification (checking all elements would be too slow)
    for i in range(0, 256, 32):  # Sample every 32nd row
        for j in range(0, 512, 64):  # Sample every 64th column
            assert_equal(output[i, j], x1[i, j], msg="Mismatch in first input")
            assert_equal(
                output[i + 256, j], x2[i, j], msg="Mismatch in second input"
            )

    print("✅ Test passed!")


def test_fused_concat_cpu() raises:
    """Test fused concat with input/output lambdas on CPU."""
    print("== test_fused_concat_cpu")

    comptime dtype = DType.float32
    comptime rank = 3
    comptime axis = 1

    comptime input_shape_0 = IndexList[rank](2, 3, 4)
    comptime input_shape_1 = IndexList[rank](2, 5, 4)
    comptime output_shape = IndexList[rank](2, 8, 4)

    var out_stack = Array[Scalar[dtype], product(output_shape, rank)](fill=-999)
    var output = TileTensor(out_stack, row_major(Coord(output_shape)))

    # Input lambda: generates data based on input index
    @__parameter
    @always_inline
    def input_fn[
        input_index: Int, width: Int, _rank: Int, alignment: Int = 1
    ](indices: IndexList[_rank]) -> SIMD[dtype, width]:
        comptime if input_index == 0:
            return SIMD[dtype, width](
                Float32(indices[0] * 100 + indices[1] * 10 + indices[2])
            )
        else:
            return SIMD[dtype, width](
                Float32(1000 + indices[0] * 100 + indices[1] * 10 + indices[2])
            )

    # Output epilogue: multiply by 2
    @__parameter
    @always_inline
    @__copy_capture(output)
    def output_fn[
        c_type: DType, _rank: Int, width: SIMDLength, *, alignment: Int
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        var coord = Coord(indices)
        comptime assert output.flat_rank >= coord.flat_rank
        output.store[width=width](coord, rebind[SIMD[dtype, width]](val * 2))

    var output_dyn = output.make_dynamic[.int64]()

    fused_concat[
        dtype,
        rank,
        input_fn,
        output_fn,
        output_dyn.LayoutType,
        axis=axis,
        target="cpu",
    ](
        StaticTuple[IndexList[rank], 2](input_shape_0, input_shape_1),
        output_dyn,
        DeviceContext(api="cpu"),
    )

    # Verify results
    for i in range(2):
        # First input section
        for j in range(3):
            for k in range(4):
                var expected = Float32(i * 100 + j * 10 + k) * 2
                assert_equal(
                    output[i, j, k], expected, msg="Mismatch in first input"
                )
        # Second input section
        for j in range(5):
            for k in range(4):
                var expected = Float32(1000 + i * 100 + j * 10 + k) * 2
                assert_equal(
                    output[i, j + 3, k],
                    expected,
                    msg="Mismatch in second input",
                )

    print("✅ Test passed!")


def test_concat_shape() raises:
    """Test concat_shape utility function."""
    print("== test_concat_shape")

    comptime dtype = DType.float32
    comptime rank = 3
    comptime axis = 1

    comptime l1 = row_major[2, 3, 4]()
    comptime l2 = row_major[2, 5, 4]()

    var x1_stack = Array[Scalar[dtype], l1.product()](fill={})
    var x2_stack = Array[Scalar[dtype], l2.product()](fill={})

    var x1 = TileTensor(x1_stack, l1)
    var x2 = TileTensor(x2_stack, l2)

    var x1_dyn = x1.make_dynamic[.int64]()
    var x2_dyn = x2.make_dynamic[.int64]()

    var input_vec = List[TileTensor[dtype, x1_dyn.LayoutType, ImmutAnyOrigin]](
        capacity=2
    )
    input_vec.append(x1_dyn.as_unsafe_any_origin().as_immut())
    input_vec.append(x2_dyn.as_unsafe_any_origin().as_immut())

    var output_shape = concat_shape[dtype](input_vec, axis)

    assert_equal(output_shape[0], 2, msg="Wrong dim 0")
    assert_equal(output_shape[1], 8, msg="Wrong concat dim (3+5=8)")
    assert_equal(output_shape[2], 4, msg="Wrong dim 2")

    print("✅ Test passed!")


def test_concat_with_epilogue() raises:
    """Test concat with epilogue function."""
    print("== test_concat_with_epilogue")

    comptime dtype = DType.float32
    comptime rank = 2
    comptime axis = 0

    comptime l1 = row_major[3, 8]()
    comptime l2 = row_major[5, 8]()
    comptime out_layout = row_major[8, 8]()

    var x1_stack = Array[Scalar[dtype], l1.product()](
        fill_with=lambda (i: Int) -> Scalar[dtype]: Float32(i)
    )
    var x2_stack = Array[Scalar[dtype], l2.product()](
        fill_with=lambda (i: Int) -> Scalar[dtype]: Float32(i + 100)
    )
    var out_stack = Array[Scalar[dtype], out_layout.product()](fill=-1)

    var x1 = TileTensor(x1_stack, l1)
    var x2 = TileTensor(x2_stack, l2)
    var output = TileTensor(out_stack, out_layout)

    var x1_dyn = x1.make_dynamic[.int64]()
    var x2_dyn = x2.make_dynamic[.int64]()

    var input_tuple = StaticTuple[
        TileTensor[dtype, x1_dyn.LayoutType, ImmutAnyOrigin],
        2,
    ](
        x1_dyn.as_unsafe_any_origin().as_immut(),
        x2_dyn.as_unsafe_any_origin().as_immut(),
    )

    @__parameter
    @always_inline
    @__copy_capture(output)
    def epilogue_add_10[
        c_type: DType, _rank: Int, width: SIMDLength, *, alignment: Int
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        var coord = Coord(indices)
        comptime assert output.flat_rank >= coord.flat_rank
        output.store[width=width](coord, rebind[SIMD[dtype, width]](val + 10))

    concat[dtype, epilogue_fn=epilogue_add_10](
        output.make_dynamic[.int64](),
        axis,
        input_tuple,
        DeviceContext(api="cpu"),
    )

    # Verify epilogue was applied
    for i in range(3):
        for j in range(8):
            assert_equal(
                output[i, j], x1[i, j] + 10, msg="Epilogue not applied to x1"
            )
    for i in range(5):
        for j in range(8):
            assert_equal(
                output[i + 3, j],
                x2[i, j] + 10,
                msg="Epilogue not applied to x2",
            )

    print("✅ Test passed!")


def test_concat_many_inputs() raises:
    """Test concat with many inputs (5 inputs)."""
    print("== test_concat_many_inputs")

    comptime dtype = DType.float32
    comptime rank = 2
    comptime axis = 1

    comptime l1 = row_major[4, 2]()
    comptime l2 = row_major[4, 3]()
    comptime l3 = row_major[4, 1]()
    comptime l4 = row_major[4, 4]()
    comptime l5 = row_major[4, 5]()
    comptime out_layout = row_major[4, 15]()

    var x1_stack = Array[Scalar[dtype], l1.product()](fill=1)
    var x2_stack = Array[Scalar[dtype], l2.product()](fill=2)
    var x3_stack = Array[Scalar[dtype], l3.product()](fill=3)
    var x4_stack = Array[Scalar[dtype], l4.product()](fill=4)
    var x5_stack = Array[Scalar[dtype], l5.product()](fill=5)
    var out_stack = Array[Scalar[dtype], out_layout.product()](fill=-1)

    var x1 = TileTensor(x1_stack, l1)
    var x2 = TileTensor(x2_stack, l2)
    var x3 = TileTensor(x3_stack, l3)
    var x4 = TileTensor(x4_stack, l4)
    var x5 = TileTensor(x5_stack, l5)
    var output = TileTensor(out_stack, out_layout)

    var x1_dyn = x1.make_dynamic[.int64]()

    var input_tuple = StaticTuple[
        TileTensor[dtype, x1_dyn.LayoutType, ImmutAnyOrigin],
        5,
    ](
        x1_dyn.as_unsafe_any_origin().as_immut(),
        x2.make_dynamic[.int64]().as_unsafe_any_origin().as_immut(),
        x3.make_dynamic[.int64]().as_unsafe_any_origin().as_immut(),
        x4.make_dynamic[.int64]().as_unsafe_any_origin().as_immut(),
        x5.make_dynamic[.int64]().as_unsafe_any_origin().as_immut(),
    )

    concat[dtype](
        output.make_dynamic[.int64](),
        axis,
        input_tuple,
        DeviceContext(api="cpu"),
    )

    # Verify each section
    for i in range(4):
        for j in range(2):
            assert_equal(output[i, j], Float32(1), msg="Mismatch in x1")
        for j in range(3):
            assert_equal(output[i, j + 2], Float32(2), msg="Mismatch in x2")
        assert_equal(output[i, 5], Float32(3), msg="Mismatch in x3")
        for j in range(4):
            assert_equal(output[i, j + 6], Float32(4), msg="Mismatch in x4")
        for j in range(5):
            assert_equal(output[i, j + 10], Float32(5), msg="Mismatch in x5")

    print("✅ Test passed!")


def main() raises:
    test_concat_inner_all_outer_dims_singleton()
    test_concat_serial_general_case()
    test_concat_parallel_large()
    test_fused_concat_cpu()
    test_concat_shape()
    test_concat_with_epilogue()
    test_concat_many_inputs()

    print("\n🎉 All CPU concat coverage tests passed!")
