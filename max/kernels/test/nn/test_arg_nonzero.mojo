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

from std.itertools import product
from layout import TileTensor, row_major
from nn.arg_nonzero import arg_nonzero, arg_nonzero_shape
from std.testing import assert_equal


# CHECK-LABEL: test_where_size
def test_where_size() raises:
    print("== test_where_size")
    comptime rank = 3
    comptime values_shape = row_major[3, 2, 1]()
    var values_stack = Array[Float32, values_shape.product()](fill={})
    var values = TileTensor(values_stack, values_shape)

    values[0, 0, 0] = 1.0
    values[0, 1, 0] = 2.0
    values[1, 0, 0] = 0.0
    values[1, 1, 0] = 0.0
    values[2, 0, 0] = 0.0
    values[2, 1, 0] = -3.0

    var output_shape = arg_nonzero_shape[.float32](
        values.make_dynamic[.int64]()
    )

    assert_equal(output_shape[0], 3)
    assert_equal(output_shape[1], 3)


# CHECK-LABEL: test_where_size_bool
def test_where_size_bool() raises:
    print("== test_where_size_bool")
    comptime rank = 3
    comptime values_shape = row_major[3, 2, 1]()
    var values_stack = Array[Scalar[.bool], values_shape.product()](fill={})
    var values = TileTensor(values_stack, values_shape)

    values[0, 0, 0] = True
    values[0, 1, 0] = True
    values[1, 0, 0] = False
    values[1, 1, 0] = False
    values[2, 0, 0] = Scalar[.bool](False)
    values[2, 1, 0] = Scalar[.bool](True)

    var output_shape = arg_nonzero_shape[.bool](values.make_dynamic[.int64]())

    assert_equal(output_shape[0], 3)
    assert_equal(output_shape[1], 3)


# CHECK-LABEL: test_where
def test_where() raises:
    print("== test_where")
    comptime rank = 3
    comptime values_shape = row_major[3, 2, 1]()
    var values_stack = Array[Float32, values_shape.product()](fill={})
    var values = TileTensor(values_stack, values_shape)

    values[0, 0, 0] = 1.0
    values[0, 1, 0] = 2.0
    values[1, 0, 0] = 0.0
    values[1, 1, 0] = 0.0
    values[2, 0, 0] = 0.0
    values[2, 1, 0] = -3.0

    var computed_stack = Array[Int, 9](fill={})
    var computed_outputs = TileTensor[.int](
        computed_stack,
        row_major[3, 3](),
    )

    var golden_stack = Array[Int, 9](fill={})
    var golden_outputs = TileTensor[.int](
        golden_stack,
        row_major[3, 3](),
    )

    golden_outputs[0, 0] = 0
    golden_outputs[0, 1] = 0
    golden_outputs[0, 2] = 0
    golden_outputs[1, 0] = 0
    golden_outputs[1, 1] = 1
    golden_outputs[1, 2] = 0
    golden_outputs[2, 0] = 2
    golden_outputs[2, 1] = 1
    golden_outputs[2, 2] = 0

    arg_nonzero(
        values.make_dynamic[.int64](),
        computed_outputs.make_dynamic[.int64](),
    )

    for i, j in product(range(3), range(3)):
        assert_equal(computed_outputs[i, j], golden_outputs[i, j])


# CHECK-LABEL: test_where_1d
def test_where_1d() raises:
    print("== test_where_1d")
    comptime num_elements = 12
    comptime num_indices = 6

    var values_stack = Array[Float32, num_elements](
        fill_with=lambda (i: Int) -> Float32: Float32(i % 2)
    )
    var values = TileTensor(values_stack, row_major[num_elements]())

    var computed_stack = Array[Int, num_indices](fill={})
    var computed_outputs = TileTensor(
        computed_stack, row_major[num_indices, 1]()
    )

    var golden_stack = Array[Int, num_indices](
        fill_with=lambda (i: Int) -> Int: 2 * i + 1
    )
    var golden_outputs = TileTensor(golden_stack, row_major[num_indices]())

    arg_nonzero(
        values.make_dynamic[.int64](),
        computed_outputs.make_dynamic[.int64](),
    )

    for i in range(num_indices):
        assert_equal(computed_outputs[i, 0], golden_outputs[i])


# CHECK-LABEL: test_where_bool
def test_where_bool() raises:
    print("== test_where_bool")
    comptime rank = 3
    comptime values_shape = row_major[3, 2, 1]()
    var values_stack = Array[Scalar[.bool], Int(values_shape.product())](
        fill={}
    )
    var values = TileTensor(values_stack, values_shape)

    values[0, 0, 0] = True
    values[0, 1, 0] = True
    values[1, 0, 0] = False
    values[1, 1, 0] = False
    values[2, 0, 0] = False
    values[2, 1, 0] = True

    var computed_stack = Array[Int, 9](fill={})
    var computed_outputs = TileTensor(computed_stack, row_major[3, 3]())

    var golden_stack = Array[Int, 9](fill={})
    var golden_outputs = TileTensor(golden_stack, row_major[3, 3]())

    golden_outputs[0, 0] = 0
    golden_outputs[0, 1] = 0
    golden_outputs[0, 2] = 0
    golden_outputs[1, 0] = 0
    golden_outputs[1, 1] = 1
    golden_outputs[1, 2] = 0
    golden_outputs[2, 0] = 2
    golden_outputs[2, 1] = 1
    golden_outputs[2, 2] = 0

    arg_nonzero(
        values.make_dynamic[.int64](),
        computed_outputs.make_dynamic[.int64](),
    )

    for i, j in product(range(3), range(3)):
        assert_equal(computed_outputs[i, j], golden_outputs[i, j])


def main() raises:
    test_where_size()
    test_where_size_bool()
    test_where()
    test_where_1d()
    test_where_bool()
