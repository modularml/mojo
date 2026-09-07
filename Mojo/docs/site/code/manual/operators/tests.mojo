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
# test_operators.mojo
# Tests for operators.mdx code examples.
# Skip: hypothetical chain, String() constructor (undefined vars),
#        walrus (requires input()), precedence (undefined vars),
#        matrix multiplication (no built-in matrix type).
from std.testing import assert_equal


def test_arithmetic() raises:
    assert_equal(7 + 3, 10)  # add
    assert_equal(7 - 3, 4)  # subtract
    assert_equal(7 * 3, 21)  # multiply


def test_exponentiation_basic() raises:
    assert_equal(2**8, 256)  # 256 (exponentiation)


def test_unary_symbols() raises:
    assert_equal(-7, -7)  # -7
    assert_equal(+7, 7)  # 7
    var a: Int8 = -128
    assert_equal(~a, 127)  # 127


def test_division_and_remainder() raises:
    var a = -7
    var b = 4
    assert_equal(a / b, -1)  # -1 (truncates toward zero)
    assert_equal(a // b, -2)  # -2 (rounds toward negative infinity)


def test_modulo() raises:
    assert_equal(7 % 3, 1)  # 1
    assert_equal(-7 % 4, 1)  # 1
    assert_equal(7 % -4, -1)  # -1


def test_exponentiation_right_assoc() raises:
    assert_equal(2**3**2, 512)  # 512, same as 2 ** (3 ** 2)


def test_comparisons() raises:
    assert_equal(10 > 5, True)  # True
    assert_equal(10 == 10, True)  # True
    assert_equal(10 != 10, False)  # False


def test_float_comparison() raises:
    from std.math import isclose

    var total: Float64 = 0.0
    for _ in range(10):
        total += 0.1
    assert_equal(total == 1.0, False)  # False
    assert_equal(isclose(total, 1.0), True)  # True


def test_chained_comparisons() raises:
    var x = 5
    assert_equal(1 < x < 10, True)  # True
    assert_equal(1 < x < 3, False)  # False
    assert_equal(1 < x <= 5 < 9, True)  # True


def test_chained_len() raises:
    var short_item_list: List[Int] = [1, 2, 3, 4, 5]
    var ok = 0 < len(short_item_list) <= 10
    assert_equal(ok, True)  # True


def test_bitwise() raises:
    var flags: UInt8 = 0b0000_0101
    var mask: UInt8 = 0b0000_0011
    assert_equal(flags & mask, 1)  # 1 (only bit 0 set in both)
    assert_equal(flags | mask, 7)  # 7 (bits 0, 1, and 2)
    assert_equal(flags ^ mask, 6)  # 6 (bits 1 and 2 differ)


def test_shifts() raises:
    assert_equal(1 << 4, 16)  # 16
    assert_equal(16 >> 2, 4)  # 4
    assert_equal(-16 >> 2, -4)  # -4


def test_boolean_logic() raises:
    # constant value on left side of 'True and ...'
    # unreachable code on right side of 'True or ...'
    # assert_equal(True and False, False)  # False. Not testable
    # assert_equal(True or False, True)  # True. Not testable
    assert_equal(not True, False)  # False


def test_short_circuit() raises:
    def always_true() -> Bool:
        print("called")
        return True

    # "called" never prints because the left side
    # of `and` is already False:
    # assert_equal(always_true(), True)  # True. Not testable
    # assert_equal(False and always_true(), False)  # False. Not testable


def test_truthiness() raises:
    # var name = "Mojo"
    # assert_equal(bool(name), True)  # True. Not testable
    pass  # Not testable.


def test_membership() raises:
    var colors: List[String] = ["red", "green", "blue"]
    assert_equal("red" in colors, True)  # True
    assert_equal("yellow" not in colors, True)  # True


def test_substring() raises:
    var food = "peanut butter"
    assert_equal("nut" in food, True)  # True


def test_identity() raises:
    var opt: Optional[Int] = None
    assert_equal(opt is None, True)  # True

    opt = 42
    assert_equal(opt is not None, True)  # True


def test_string_concat_repeat() raises:
    var greeting = "Hello" + " " + "Mojo"
    assert_equal(greeting, "Hello Mojo")  # Hello Mojo
    assert_equal("ha" * 3, "hahaha")  # hahaha
    assert_equal("=" * 5, "=====")  # a line of 5 equals signs


def test_string_comparison() raises:
    assert_equal("apple" < "banana", True)  # True
    assert_equal("apple" == "apple", True)  # True
    assert_equal("apple" != "banana", True)  # True
    assert_equal("Zebra" < "ant", True)  # True
    assert_equal("bird" == "bird", True)  # True


def test_conditional_expression() raises:
    var score = 80
    var result = "pass" if score > 65 else "fail"
    assert_equal(result, "pass")  # pass


def test_conditional_in_function_arg() raises:
    def greet(name: String):
        print("Hello,", name)

    # assert_equal("Sami" if True else "Cass", "Sami")
    # Not testable due to constant condition

    # Fails CI: dead branch due to constant condition
    # greet("Sami" if True else "Cass")  # Hello, Sami


def test_chained_ternary() raises:
    var value = 50
    var label = "low" if value < 10 else "high" if value > 100 else "mid"
    assert_equal(label, "mid")  # mid
    value = 0
    assert_equal(
        "low" if value < 10 else "high" if value > 100 else "mid", "low"
    )  # low
    value = 150
    assert_equal(
        "low" if value < 10 else "high" if value > 100 else "mid", "high"
    )  # high


def test_inplace_assignment() raises:
    var count = 0
    count += 1
    count += 1
    assert_equal(count, 2)  # 2

    var flags: UInt8 = 0b0000_0001
    flags |= 0b0000_0100
    assert_equal(flags, 5)  # 5 (bits 0 and 2 set)

    flags &= 0b0000_0110
    assert_equal(flags, 4)  # 4 (only bit 2 remains set)

    flags ^= 0b0000_0010
    assert_equal(flags, 6)  # 6 (bits 1 and 2 differ)


def main() raises:
    test_arithmetic()
    test_exponentiation_basic()
    test_unary_symbols()
    test_division_and_remainder()
    test_modulo()
    test_exponentiation_right_assoc()
    test_comparisons()
    test_float_comparison()
    test_chained_comparisons()
    test_chained_len()
    test_bitwise()
    test_shifts()
    test_boolean_logic()
    test_short_circuit()
    test_truthiness()
    test_membership()
    test_substring()
    test_identity()
    test_string_concat_repeat()
    test_string_comparison()
    test_conditional_expression()
    test_conditional_in_function_arg()
    test_chained_ternary()
    test_inplace_assignment()
