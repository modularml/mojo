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

from layout import TileTensor, row_major
from nn.tile import tile
from std.testing import assert_equal


# CHECK-LABEL: test_tile_eg1
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
def test_tile_eg1() raises:
    print("== test_tile_eg1")
    comptime type = DType.float32

    var input_stack: Array[Scalar[type], _] = [0, 1, 2, 3]
    var input = TileTensor(input_stack, row_major[2, 2]())

    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack = Array[Scalar[type_repeats], 2](fill=2)
    var repeats = TileTensor(repeats_stack, row_major[2]())

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = Array[Scalar[type], 16](fill={})
    var output = TileTensor(output_stack, row_major[4, 4]())

    tile[type, type_repeats](
        input.make_dynamic[.int64](),
        repeats.make_dynamic[.int64](),
        output.make_dynamic[.int64](),
    )

    print()
    for i in range(4):
        for j in range(4):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg2
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
def test_tile_eg2() raises:
    print("== test_tile_eg2")
    comptime type = DType.float32

    var input_stack: Array[Scalar[type], _] = [0, 1, 2, 3]
    var input = TileTensor(input_stack, row_major[2, 2]())

    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack: Array[Scalar[type_repeats], _] = [3, 2]
    var repeats = TileTensor(repeats_stack, row_major[2]())

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = Array[Scalar[type], 6 * 4](fill={})
    var output = TileTensor(output_stack, row_major[6, 4]())

    tile[type, type_repeats](
        input.make_dynamic[.int64](),
        repeats.make_dynamic[.int64](),
        output.make_dynamic[.int64](),
    )

    print()
    for i in range(6):
        for j in range(4):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg3
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
def test_tile_eg3() raises:
    print("== test_tile_eg3")
    comptime type = DType.float32

    var input_stack: Array[Scalar[type], _] = [0, 1, 2, 3]
    var input = TileTensor(input_stack, row_major[2, 2]())

    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack: Array[Scalar[type_repeats], _] = [2, 3]
    var repeats = TileTensor(repeats_stack, row_major[2]())

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = Array[Scalar[type], 4 * 6](fill={})
    var output = TileTensor(output_stack, row_major[4, 6]())

    tile[type, type_repeats](
        input.make_dynamic[.int64](),
        repeats.make_dynamic[.int64](),
        output.make_dynamic[.int64](),
    )

    print()
    for i in range(4):
        for j in range(6):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg4
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
def test_tile_eg4() raises:
    print("== test_tile_eg4")
    comptime type = DType.float32

    var input_stack = Array[Scalar[type], 2 * 2 * 2](
        fill_with=lambda (i: Int) -> Scalar[type]: Scalar[type](i)
    )
    var input = TileTensor(input_stack, row_major[2, 2, 2]())

    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack: Array[Scalar[type_repeats], _] = [2, 1, 1]
    var repeats = TileTensor(repeats_stack, row_major[3]())

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = Array[Scalar[type], 4 * 2 * 2](fill={})
    var output = TileTensor(output_stack, row_major[4, 2, 2]())

    tile[type, type_repeats](
        input.make_dynamic[.int64](),
        repeats.make_dynamic[.int64](),
        output.make_dynamic[.int64](),
    )

    print()
    for i in range(4):
        for j in range(2):
            for k in range(2):
                print(output[i, j, k], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg5
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,
def test_tile_eg5() raises:
    print("== test_tile_eg5")
    comptime type = DType.float32

    var input_stack = Array[Scalar[type], 2 * 2 * 2](
        fill_with=lambda (i: Int) -> Scalar[type]: Scalar[type](i)
    )
    var input = TileTensor(input_stack, row_major[2, 2, 2]())

    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack: Array[Scalar[type_repeats], _] = [2, 1, 2]
    var repeats = TileTensor(repeats_stack, row_major[3]())

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = Array[Scalar[type], 4 * 2 * 4](fill={})
    var output = TileTensor(output_stack, row_major[4, 2, 4]())

    tile[type, type_repeats](
        input.make_dynamic[.int64](),
        repeats.make_dynamic[.int64](),
        output.make_dynamic[.int64](),
    )

    print()
    for i in range(4):
        for j in range(2):
            for k in range(4):
                print(output[i, j, k], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg6
# CHECK: 1.0 ,2.0 ,1.0 ,2.0 ,
# CHECK: 3.0 ,4.0 ,3.0 ,4.0 ,
def test_tile_eg6() raises:
    print("== test_tile_eg6")
    comptime type = DType.float32

    var input_stack = Array[Scalar[type], 2 * 2](
        fill_with=lambda (i: Int) -> Scalar[type]: Scalar[type](i + 1)
    )
    var input = TileTensor(input_stack, row_major[2, 2]())

    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack: Array[Scalar[type_repeats], _] = [1, 2]
    var repeats = TileTensor(repeats_stack, row_major[2]())

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = Array[Scalar[type], 2 * 4](fill={})
    var output = TileTensor(output_stack, row_major[2, 4]())

    tile[type, type_repeats](
        input.make_dynamic[.int64](),
        repeats.make_dynamic[.int64](),
        output.make_dynamic[.int64](),
    )

    print()
    for i in range(2):
        for j in range(4):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg7
# CHECK: 1.0 ,2.0 ,
# CHECK: 3.0 ,4.0 ,
# CHECK: 1.0 ,2.0 ,
# CHECK: 3.0 ,4.0 ,
def test_tile_eg7() raises:
    print("== test_tile_eg7")
    comptime type = DType.float32

    var input_stack = Array[Scalar[type], 2 * 2](
        fill_with=lambda (i: Int) -> Scalar[type]: Scalar[type](i + 1)
    )
    var input = TileTensor(input_stack, row_major[2, 2]())

    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack: Array[Scalar[type_repeats], _] = [2, 1]
    var repeats = TileTensor(repeats_stack, row_major[2]())

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = Array[Scalar[type], 4 * 2](fill={})
    var output = TileTensor(output_stack, row_major[4, 2]())

    tile[type, type_repeats](
        input.make_dynamic[.int64](),
        repeats.make_dynamic[.int64](),
        output.make_dynamic[.int64](),
    )

    print()
    for i in range(4):
        for j in range(2):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg8
# CHECK: 1.0 ,2.0 ,3.0 ,4.0 ,
# CHECK: 1.0 ,2.0 ,3.0 ,4.0 ,
# CHECK: 1.0 ,2.0 ,3.0 ,4.0 ,
# CHECK: 1.0 ,2.0 ,3.0 ,4.0 ,
def test_tile_eg8() raises:
    print("== test_tile_eg8")
    comptime type = DType.float32

    var input_stack: Array[Scalar[type], _] = [1, 2, 3, 4]
    var input = TileTensor(input_stack, row_major[1, 4]())

    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack: Array[Scalar[type_repeats], _] = [4, 1]
    var repeats = TileTensor(repeats_stack, row_major[2]())

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = Array[Scalar[type], 4 * 4](fill=0)
    var output = TileTensor(output_stack, row_major[4, 4]())

    tile[type, type_repeats](
        input.make_dynamic[.int64](),
        repeats.make_dynamic[.int64](),
        output.make_dynamic[.int64](),
    )

    print()
    for i in range(4):
        for j in range(4):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg9
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
def test_tile_eg9() raises:
    print("== test_tile_eg9")
    comptime type = DType.float32

    var input_stack = Array[Scalar[type], 2 * 2 * 2](
        fill_with=lambda (i: Int) -> Scalar[type]: Scalar[type](i)
    )
    var input = TileTensor(input_stack, row_major[2, 2, 2]())

    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack: Array[Scalar[type_repeats], _] = [2, 2, 1]
    var repeats = TileTensor(repeats_stack, row_major[3]())

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = Array[Scalar[type], 4 * 4 * 2](fill=0)
    var output = TileTensor(output_stack, row_major[4, 4, 2]())

    tile[type, type_repeats](
        input.make_dynamic[.int64](),
        repeats.make_dynamic[.int64](),
        output.make_dynamic[.int64](),
    )

    print()
    for i in range(4):
        for j in range(4):
            for k in range(2):
                print(output[i, j, k], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg10
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
def test_tile_eg10() raises:
    print("== test_tile_eg10")
    comptime type = DType.float32

    var input_stack = Array[Scalar[type], 2 * 2 * 2](
        fill_with=lambda (i: Int) -> Scalar[type]: Scalar[type](i)
    )
    var input = TileTensor(input_stack, row_major[2, 2, 2]())

    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack: Array[Scalar[type_repeats], _] = [3, 2, 3]
    var repeats = TileTensor(repeats_stack, row_major[3]())

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = Array[Scalar[type], 6 * 4 * 6](fill={})
    var output = TileTensor(output_stack, row_major[6, 4, 6]())

    tile[type, type_repeats](
        input.make_dynamic[.int64](),
        repeats.make_dynamic[.int64](),
        output.make_dynamic[.int64](),
    )

    print()
    for i in range(6):
        for j in range(4):
            for k in range(6):
                print(output[i, j, k], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg11
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
def test_tile_eg11() raises:
    print("== test_tile_eg11")
    comptime type = DType.float32

    var input_stack = Array[Scalar[type], 3 * 2 * 2](
        fill_with=lambda (i: Int) -> Scalar[type]: Scalar[type](i)
    )
    var input = TileTensor(input_stack, row_major[3, 2, 2]())

    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack = Array[Scalar[type_repeats], 3](fill={})
    var repeats = TileTensor(repeats_stack, row_major[3]())

    repeats[0] = 2
    repeats[1] = 3
    repeats[2] = 1

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = Array[Scalar[type], 6 * 6 * 2](fill=0)
    var output = TileTensor(output_stack, row_major[6, 6, 2]())

    tile[type, type_repeats](
        input.make_dynamic[.int64](),
        repeats.make_dynamic[.int64](),
        output.make_dynamic[.int64](),
    )

    print()
    for i in range(6):
        for j in range(6):
            for k in range(2):
                print(output[i, j, k], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg12
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
def test_tile_eg12() raises:
    print("== test_tile_eg12")
    comptime type = DType.float32

    var input_stack = Array[Scalar[type], 2 * 2](
        fill_with=lambda (i: Int) -> Scalar[type]: Scalar[type](i)
    )
    var input = TileTensor(input_stack, row_major[1, 1, 2, 2]())

    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack = Array[Scalar[type_repeats], 4](fill={})
    var repeats = TileTensor(repeats_stack, row_major[4]())

    repeats[0] = 1
    repeats[1] = 1
    repeats[2] = 2
    repeats[3] = 3

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = Array[Scalar[type], 4 * 6](fill=0)
    var output = TileTensor(output_stack, row_major[1, 1, 4, 6]())

    tile[type, type_repeats](
        input.make_dynamic[.int64](),
        repeats.make_dynamic[.int64](),
        output.make_dynamic[.int64](),
    )

    print()
    for i in range(1):
        for j in range(1):
            for k in range(4):
                for l in range(6):
                    print(output[i, j, k, l], ",", end="")
                print()
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg13
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
def test_tile_eg13() raises:
    print("== test_tile_eg13")
    comptime type = DType.float32

    var input_stack = Array[Scalar[type], 2 * 2 * 2 * 2](
        fill_with=lambda (i: Int) -> Scalar[type]: Scalar[type](i)
    )
    var input = TileTensor(input_stack, row_major[2, 2, 2, 2]())

    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack = Array[Scalar[type_repeats], 4](fill={})
    var repeats = TileTensor(repeats_stack, row_major[4]())

    repeats[0] = 1
    repeats[1] = 2
    repeats[2] = 2
    repeats[3] = 3

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = Array[Scalar[type], 2 * 4 * 4 * 6](fill=0)
    var output = TileTensor(output_stack, row_major[2, 4, 4, 6]())

    tile[type, type_repeats](
        input.make_dynamic[.int64](),
        repeats.make_dynamic[.int64](),
        output.make_dynamic[.int64](),
    )

    print()
    for i in range(2):
        for j in range(4):
            for k in range(4):
                for l in range(6):
                    print(output[i, j, k, l], ",", end="")
                print()
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg14
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
def test_tile_eg14() raises:
    print("== test_tile_eg14")
    comptime type = DType.float32

    var input_stack = Array[Scalar[type], 2 * 2 * 2 * 2](
        fill_with=lambda (i: Int) -> Scalar[type]: Scalar[type](i)
    )
    var input = TileTensor(input_stack, row_major[2, 2, 2, 2]())

    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack: Array[Scalar[type_repeats], _] = [2, 2, 2, 3]
    var repeats = TileTensor(repeats_stack, row_major[4]())

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = Array[Scalar[type], 4 * 4 * 4 * 6](fill=0)
    var output = TileTensor(output_stack, row_major[4, 4, 4, 6]())

    tile[type, type_repeats](
        input.make_dynamic[.int64](),
        repeats.make_dynamic[.int64](),
        output.make_dynamic[.int64](),
    )

    print()
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(6):
                    print(output[i, j, k, l], ",", end="")
                print()
            print()
        print()
    print()


def test_tile_1d() raises:
    """Test tiling a 1D tensor.

    This tests the edge case where input.rank == 1, which previously would
    cause negative index access (repeats[-1], repeats[-2]) in the tile function.
    """
    comptime type = DType.float32
    comptime type_repeats = DType.int64

    var input_stack: Array[Scalar[type], _] = [1, 2, 3]
    var input = TileTensor(input_stack, row_major[3]())

    # Tile 3 times along the single dimension
    var repeats_stack: Array[Scalar[type_repeats], _] = [3]
    var repeats = TileTensor(repeats_stack, row_major[1]())

    # Output: 3 * 3 = 9 elements
    var output_stack = Array[Scalar[type], 9](fill={})
    var output = TileTensor(output_stack, row_major[9]())

    tile[type, type_repeats](
        input.make_dynamic[.int64](),
        repeats.make_dynamic[.int64](),
        output.make_dynamic[.int64](),
    )

    # Expected: [1, 2, 3] repeated 3 times
    var expected: Array[Scalar[type], 9] = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    for i in range(9):
        assert_equal(output[i], expected[i])


def main() raises:
    test_tile_1d()
    test_tile_eg1()
    test_tile_eg2()
    test_tile_eg3()
    test_tile_eg4()
    test_tile_eg5()
    test_tile_eg6()
    test_tile_eg7()
    test_tile_eg8()
    test_tile_eg9()
    test_tile_eg10()
    test_tile_eg11()
    test_tile_eg12()
    test_tile_eg13()
    test_tile_eg14()
