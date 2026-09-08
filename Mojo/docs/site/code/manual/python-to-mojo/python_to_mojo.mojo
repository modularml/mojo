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
from std.testing import *
from std.math import isclose, abs


def test_mut() raises:
    def foo(mut value: Int):
        value += 1

    var y = 20
    foo(y)
    assert_true(y == 21)


def test_math() raises:
    var a: Int = 7
    var b: Int = 2

    assert_true(
        a / b == 3
    )  # Integer division, result is truncated towards zero

    var c = -7
    var d = 2
    assert_true(
        c / d == -3
    )  # -3, not -4, because division truncates towards zero
    assert_true(c // d == -4)  # -4, biased towards negative infinity

    var e: Float64 = 7.0
    var f: Float64 = 2.0
    assert_almost_equal(e / f, 3.5)  # Floating-point division, result is 3.5
    assert_almost_equal(e // f, 3.0)  # biased towards negative infinity

    var g: Float64 = -7.0
    assert_almost_equal(
        g / f, -3.5
    )  # zero bias does not affect floating-point division
    assert_almost_equal(g // f, -4.0)  # biased towards negative infinity


def test_collections() raises:
    var nums: List[Int] = [1, 2, 3]
    nums.append(4)  # [1, 2, 3, 4]
    assert_true(len(nums) == 4)
    assert_true(nums[0] == 1)
    assert_true(nums[3] == 4)

    var counts: Dict[String, Int] = {"a": 1, "b": 2}
    counts["c"] = 3  # {a: 1, b: 2, c: 3}
    assert_true(counts["a"] == 1)
    assert_true(counts["c"] == 3)

    var list_squares = [
        x * x for x in [0, 1, 2, 3, 4] if x % 2 == 0
    ]  # [0, 4, 16]
    assert_true(
        len(list_squares) == 3
        and list_squares[0] == 0
        and list_squares[1] == 4
        and list_squares[2] == 16
    )

    var positive_numbers = [x for x in range(-3, 3) if x > 0]  # [1, 2]
    assert_true(
        len(positive_numbers) == 2
        and positive_numbers[0] == 1
        and positive_numbers[1] == 2
    )

    var dict_squares = {x: x * x for x in range(3)}  #  {0: 0, 1: 1, 2: 4}, dict
    var upper_case = {k: v.upper() for k, v in [(1, "one"), (2, "two")]}
    # {1: ONE, 2: TWO}, dict
    assert_true(
        dict_squares[0] == 0 and dict_squares[1] == 1 and dict_squares[2] == 4
    )
    assert_true(upper_case[1] == "ONE" and upper_case[2] == "TWO")

    var number_set = {x for x in range(5)}  # x in [0..5) {0, 1, 2, 3, 4}
    assert_true(
        len(number_set) == 5
        and 0 in number_set
        and 1 in number_set
        and 2 in number_set
        and 3 in number_set
        and 4 in number_set
    )


def test_loops() raises:
    var nums: List = [0, 1, 2, 3, 4]
    var squares2: List[Int] = []

    for x in nums:
        if x % 2 == 0:
            squares2.append(x * x)
    assert_true(
        len(squares2) == 3
        and squares2[0] == 0
        and squares2[1] == 4
        and squares2[2] == 16
    )

    var squares3: List[Int] = []
    var idx = 0

    while idx < 3:
        squares3.append(idx * idx)
        idx += 1

    assert_true(
        len(squares3) == 3
        and squares3[0] == 0
        and squares3[1] == 1
        and squares3[2] == 4
    )


def test_raises() raises:
    try:
        raise Error("bad input")
    except e:
        assert_true(String(e) == "bad input")


@fieldwise_init
struct MyCustomError(Writable):
    var message: String


def test_typed_error() raises MyCustomError:  # Typed error
    raise MyCustomError("custom error occurred")


def test_typed_error_catch() raises:
    try:
        test_typed_error()
    except e:
        assert_true(String(e) == "MyCustomError(message=custom error occurred)")


struct Point:
    var x: Int
    var y: Int

    def __init__(out self, x: Int, y: Int):
        self.x = x
        self.y = y


def test_structs() raises:
    var p = Point(3, 4)
    assert_true(p.x == 3 and p.y == 4)


def test_typebound() raises:
    var a = 1

    def foo() raises:
        var a = "hello"
        assert_true(a == "hello")

    foo()
    assert_true(a == 1)


trait Drawable:
    def draw(self):
        ...  # required


@fieldwise_init
struct Circle(Drawable):
    def draw(self):
        # Implementation of drawing a circle
        pass


def test_draw[T: Drawable](shape: T):
    shape.draw()  # Compiler guarantees shape has draw()


def test_traitbound():
    var c = Circle()
    test_draw(c)  # Works because Circle implements Drawable


def another_raising_function() raises:
    raise Error("Message")  # Error raised here


def raising_function() raises:
    another_raising_function()  # Error continues to pass


def handles_errors() raises:  # Raises appears here for testing assertion
    try:
        raising_function()  # Error handled in this function
    except e:
        assert_true(String(e) == "Message")  # Error message is correct


def test_scoped() raises:
    from std.utils import Variant

    comptime StringOrInt = Variant[String, Int]
    var a: StringOrInt = 1
    if a.isa[Int]():
        assert_true(a.unsafe_get[Int]() == 1)
    a = "string"  # Not an error
    if a.isa[String]():
        assert_true(a.unsafe_get[String]() == "string")


def test_mixed_types() raises:
    from std.utils import Variant

    comptime MixedType = Variant[Int, Float64, String, Bool]
    var mixed_list = List[MixedType]()
    mixed_list.append(MixedType(42))
    mixed_list.append(MixedType(3.14))
    mixed_list.append(MixedType("hello"))
    mixed_list.append(MixedType(True))
    # for item in mixed_list:
    #     print(item, end=" ")
    # print()  # Output: 42 3.14 hello True
    assert_true(
        mixed_list[0].unsafe_get[Int]() == 42
        and isclose(mixed_list[1].unsafe_get[Float64](), 3.14)
        and mixed_list[2].unsafe_get[String]() == "hello"
        and mixed_list[3].unsafe_get[Bool]() == True
    )


def main() raises:
    test_mut()
    test_math()
    # size_of[Int]() can't be tested. Depends on target architecture.
    test_collections()
    test_loops()
    test_raises()
    test_typed_error_catch()
    test_structs()
    # Can't test conversion errors
    test_typebound()
    test_traitbound()
    handles_errors()
    test_scoped()
    test_mixed_types()
