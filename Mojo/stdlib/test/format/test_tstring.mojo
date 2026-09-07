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

from std.format.tstring import _encode_format_string
from std.testing import assert_equal, assert_raises, assert_true, TestSuite


@fieldwise_init
struct Point(Writable):
    var x: Int
    var y: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(t"({self.x}, {self.y})")


def test_basic_tstring() raises:
    assert_equal(String(t"Hello, World!"), "Hello, World!")


def test_single_interpolation() raises:
    var name = "Alice"
    assert_equal(String(t"Hello, {name}!"), "Hello, Alice!")


def test_multiple_interpolations() raises:
    var x = 10
    var y = 20
    assert_equal(String(t"{x} + {y} = {x + y}"), "10 + 20 = 30")


def test_expression_interpolation() raises:
    assert_equal(String(t"Result: {2 * 3 + 1}"), "Result: 7")


def test_empty_tstring() raises:
    var s = t""
    assert_equal(String(s), "")


def test_tstring_only_expression() raises:
    assert_equal(String(t"{42}"), "42")


def test_escaped_braces() raises:
    assert_equal(String(t"Use {{braces}} like this"), "Use {braces} like this")


def test_mixed_escaped_and_interpolation() raises:
    var value = 123
    assert_equal(
        String(t"The value {{value}} = {value}"), "The value {value} = 123"
    )


def test_deeply_nested_escape_braces() raises:
    var x = 42
    assert_equal(String(t"{{{{{x}}}}}"), "{{42}}")


def test_adjacent_interpolations() raises:
    var a = "A"
    var b = "B"
    var c = "C"
    assert_equal(String(t"{a}{b}{c}"), "ABC")


def test_boolean_interpolation() raises:
    assert_equal(
        String(t"True: {True}, False: {False}"), "True: True, False: False"
    )


def test_integer_interpolation() raises:
    var i8 = Int8(127)
    var i16 = Int16(32767)
    var i32 = Int32(2147483647)
    var i64 = Int64(9223372036854775807)
    assert_equal(String(t"Int8: {i8}"), "Int8: 127")
    assert_equal(String(t"Int16: {i16}"), "Int16: 32767")
    assert_equal(String(t"Int32: {i32}"), "Int32: 2147483647")
    assert_equal(String(t"Int64: {i64}"), "Int64: 9223372036854775807")


def test_string_interpolation() raises:
    var msg = "world"
    # TODO(KGEN): Same bug as test_single_interpolation
    assert_equal(String(t"Hello, {msg}"), "Hello, world")


def test_writable_type() raises:
    var p = Point(10, 20)
    assert_equal(String(t"Point: {p}"), "Point: (10, 20)")


def test_nested_expressions() raises:
    var x = 10
    var y = 5
    assert_equal(String(t"Calc: {(x + y) * 2}"), "Calc: 30")


def test_multiple_same_variable() raises:
    var num = 5
    assert_equal(String(t"{num} * {num} = {num * num}"), "5 * 5 = 25")


def test_complex_expression() raises:
    var a = 2
    var b = 3
    var c = 4
    assert_equal(String(t"Result: {a * b * c + b * 2 + a * 5}"), "Result: 40")


def test_tstring_in_variable() raises:
    var x = 100
    var message = t"The value is {x}"
    assert_equal(String(message), "The value is 100")


def test_method_calls() raises:
    var s = String("hello")
    assert_equal(String(t"Uppercase: {s.upper()}"), "Uppercase: HELLO")
    assert_equal(String(t"Length: {s.byte_length()}"), "Length: 5")


def test_list_subscripting() raises:
    var numbers = [10, 20, 30, 40, 50]
    assert_equal(String(t"First: {numbers[0]}"), "First: 10")
    assert_equal(String(t"Third: {numbers[2]}"), "Third: 30")
    assert_equal(String(t"Last: {numbers[4]}"), "Last: 50")


def test_attribute_access() raises:
    var p = Point(15, 25)
    assert_equal(String(t"X coordinate: {p.x}"), "X coordinate: 15")
    assert_equal(String(t"Y coordinate: {p.y}"), "Y coordinate: 25")


def test_chained_method_calls() raises:
    var text = String("  hello world  ")
    assert_equal(
        String(t"Stripped and upper: {text.strip().upper()}"),
        "Stripped and upper: HELLO WORLD",
    )


def test_subscript_with_expression() raises:
    var data = [100, 200, 300, 400]
    var index = 2
    assert_equal(
        String(t"Value at index {index}: {data[index]}"),
        "Value at index 2: 300",
    )
    assert_equal(
        String(t"Value at computed index: {data[index + 1]}"),
        "Value at computed index: 400",
    )


def test_method_on_literal() raises:
    assert_equal(
        String(t"Upper case: {String('mojo').upper()}"), "Upper case: MOJO"
    )


def test_complex_nested_expression() raises:
    var values = [5, 10, 15, 20]
    var multiplier = 3
    assert_equal(
        String(t"Computed: {values[1] * multiplier + values[2]}"),
        "Computed: 45",
    )


def test_conditional_expression() raises:
    var x = 10
    var y = 20
    var max_val = x if x > y else y
    assert_equal(String(t"Maximum: {max_val}"), "Maximum: 20")


def test_comparison_in_interpolation() raises:
    var a = 5
    var b = 10
    assert_equal(String(t"{a} < {b}: {a < b}"), "5 < 10: True")
    assert_equal(String(t"{a} > {b}: {a > b}"), "5 > 10: False")


def test_arithmetic_with_subscript() raises:
    var nums = [2, 4, 6, 8]
    assert_equal(
        String(t"Sum of first two: {nums[0] + nums[1]}"), "Sum of first two: 6"
    )
    assert_equal(
        String(t"Product of last two: {nums[2] * nums[3]}"),
        "Product of last two: 48",
    )


def test_string_method_with_args() raises:
    var text = String("hello-world-test")
    assert_equal(
        String(t"Split count: {len(text.split('-'))}"), "Split count: 3"
    )


def test_type_conversion_in_interpolation() raises:
    var num = 42
    var float_num = Float64(num)
    assert_equal(String(t"As float: {float_num}"), "As float: 42.0")


def test_same_quote_nested_string_double() raises:
    assert_equal(String(t"hello {"world"}"), "hello world")


def test_same_quote_nested_string_single() raises:
    assert_equal(String(t'hello {'world'}'), "hello world")


def test_same_quote_multiple_nested() raises:
    assert_equal(String(t"a {"b"} c {"d"}"), "a b c d")


def test_same_quote_triple_quoted() raises:
    assert_equal(String(t"""hello {"world"}"""), "hello world")


def test_mixed_quotes_double_outer() raises:
    assert_equal(String(t"outer {'inner'}"), "outer inner")


def test_mixed_quotes_single_outer() raises:
    assert_equal(String(t'outer {"inner"}'), "outer inner")


def test_nested_string_with_expression() raises:
    var count = 5
    assert_equal(String(t"Found {"item"} {count} times"), "Found item 5 times")


def test_escaped_quote_in_nested_string() raises:
    assert_equal(String(t'test {"say \"hello\""}'), 'test say "hello"')


def test_mutating_tsring_interpolated_value_before_written() raises:
    var x = "C++"
    var s = t"{x} is the best language"
    x = "Mojo"
    assert_equal(String(s), "Mojo is the best language")


def test_materialized_value_in_tstring() raises:
    comptime world = "World"
    assert_equal(String(t"Hello {world}"), "Hello World")


# =============================================================================
# Nested t-string tests (t-strings inside t-string interpolations)
# =============================================================================


def test_nested_tstring_different_quotes() raises:
    var x = 10
    assert_equal(String(t"Outer: {t'Inner: {x}'}"), "Outer: Inner: 10")


def test_nested_tstring_same_quote_double() raises:
    var value = 42
    assert_equal(String(t"Result: {t"{value}"}"), "Result: 42")


def test_nested_tstring_same_quote_single() raises:
    var num = 99
    assert_equal(String(t'Value: {t'{num}'}'), "Value: 99")


def test_nested_tstring_multiple() raises:
    var a = 1
    var b = 2
    assert_equal(
        String(t"First: {t'{a}'}, Second: {t'{b}'}"), "First: 1, Second: 2"
    )


def test_nested_tstring_triple_level() raises:
    var val = 7
    assert_equal(
        String(t"L1: {t'L2: {t"L3: {val}"}'}"),
        "L1: L2: L3: 7",
    )


def test_nested_tstring_triple_level_same_quotes() raises:
    var n = 3
    assert_equal(
        String(t"A {t"B {t"C {n}"}"}"),
        "A B C 3",
    )


def test_nested_tstring_with_expression() raises:
    var x = 5
    assert_equal(String(t"Double: {t'{x * 2}'}"), "Double: 10")


def test_nested_tstring_adjacent() raises:
    var a = 1
    var b = 2
    assert_equal(String(t"{t'{a}'}{t'{b}'}"), "12")


def test_nested_tstring_with_escaped_braces() raises:
    var x = 10
    assert_equal(
        String(t"Outer {{brace}} {t'Inner {x}'}"),
        "Outer {brace} Inner 10",
    )


def test_nested_tstring_both_escaped_braces() raises:
    var y = 20
    assert_equal(
        String(t"Out {{1}} {t'In {{2}} {y}'}"),
        "Out {1} In {2} 20",
    )


def test_nested_tstring_empty_outer() raises:
    var x = 123
    assert_equal(String(t"{t'{x}'}"), "123")


def test_tstring_with_escape_character() raises:
    var x = 123
    assert_equal(String(t"abc\t{x}"), "abc\t123")


def test_tstring_with_newline_escape() raises:
    var val = 42
    assert_equal(String(t"line1\n{val}"), "line1\n42")


def test_tstring_with_multiple_escapes() raises:
    var num = 99
    assert_equal(
        String(t"tab\there\nnewline\r{num}\t"), "tab\there\nnewline\r99\t"
    )


def test_tstring_with_punctuation_at_end() raises:
    var x = 10
    assert_equal(String(t"value: {x}!"), "value: 10!")


def test_tstring_with_backslash_escape() raises:
    var x = 10
    assert_equal(String(t"path\\to\\{x}"), "path\\to\\10")


def test_tstring_concatenation() raises:
    var x = 10
    var y = 20
    # fmt: off
    assert_equal(String(t"{x}" t"{y}"), "1020")
    # fmt: on


def test_tstring_multiline_concatenation() raises:
    var x = 10
    var y = 20
    # fmt: off
    var tstring = (
        t"This is a multiline {x}"
        t" tstring expression that will "
        t"concatenate, {y}!"
    )
    # fmt: on

    assert_equal(
        String(tstring),
        "This is a multiline 10 tstring expression that will concatenate, 20!",
    )


# =============================================================================
# Raw t-string tests (rt"..." and tr"...")
# =============================================================================


def test_raw_tstring_basic() raises:
    assert_equal(String(rt"Hello, World!"), "Hello, World!")


def test_raw_tstring_backslash_n_literal() raises:
    var name = "Alice"
    assert_equal(String(rt"Hello\n{name}"), r"Hello\nAlice")


def test_raw_tstring_backslash_t_literal() raises:
    var x = 42
    assert_equal(String(rt"value\t{x}"), r"value\t42")


def test_raw_tstring_interpolation() raises:
    var x = 10
    var y = 20
    assert_equal(String(rt"{x} + {y} = {x + y}"), "10 + 20 = 30")


def test_raw_tstring_backslash_path() raises:
    var name = "docs"
    assert_equal(String(rt"C:\Users\{name}"), r"C:\Users\docs")


def test_raw_tstring_escaped_braces() raises:
    assert_equal(String(rt"Use {{braces}} like this"), "Use {braces} like this")


def test_raw_tstring_mixed_escaped_and_interpolation() raises:
    var value = 123
    assert_equal(
        String(rt"The value {{value}} = {value}"),
        "The value {value} = 123",
    )


def test_raw_tstring_hex_escape_literal() raises:
    var x = 1
    assert_equal(String(rt"A\x42{x}"), r"A\x421")


def test_raw_tstring_double_backslash() raises:
    var x = 1
    assert_equal(String(rt"AB\\{x}"), r"AB\\1")


def test_raw_tstring_tr_prefix() raises:
    var name = "world"
    assert_equal(String(rt"Hello\n{name}"), r"Hello\nworld")


def test_raw_tstring_prefix_variants() raises:
    var x = 1
    # fmt: off
    assert_equal(String(rt"v{x}"), "v1")
    assert_equal(String(rT"v{x}"), "v1")
    assert_equal(String(Rt"v{x}"), "v1")
    assert_equal(String(RT"v{x}"), "v1")
    assert_equal(String(tr"v{x}"), "v1")
    assert_equal(String(tR"v{x}"), "v1")
    assert_equal(String(Tr"v{x}"), "v1")
    assert_equal(String(TR"v{x}"), "v1")
    # fmt: on


def test_raw_tstring_empty() raises:
    assert_equal(String(rt""), "")


def test_raw_tstring_only_expression() raises:
    assert_equal(String(rt"{42}"), "42")


def test_raw_tstring_adjacent_interpolations() raises:
    var a = "A"
    var b = "B"
    assert_equal(String(rt"{a}{b}"), "AB")


def test_raw_tstring_triple_quoted() raises:
    var x = 42
    assert_equal(
        String(rt"""raw\n{x}"""),
        r"raw\n42",
    )


def test_raw_tstring_vs_regular_tstring() raises:
    # Verify raw and regular t-strings differ on escape handling
    var x = 1
    assert_equal(String(t"tab\there{x}"), "tab\there1")
    assert_equal(String(rt"tab\there{x}"), r"tab\there1")


def test_raw_tstring_nested_in_tstring() raises:
    var x = 42
    assert_equal(String(t"outer {rt'raw\n{x}'}"), r"outer raw\n42")


def test_raw_tstring_multiline_concatenation() raises:
    var x = 10
    var y = 20
    # fmt: off
    var tstring = (
        rt"raw\n{x}"
        rt" also raw\t{y}"
        rt" end"
    )
    # fmt: on
    assert_equal(String(tstring), r"raw\n10 also raw\t20 end")


def test_raw_tstring_mixed_concat_raw_then_cooked() raises:
    var x = 42
    # fmt: off
    var tstring = (
        rt"raw\n{x}"
        t" cooked\n{x}"
    )
    # fmt: on
    assert_equal(String(tstring), r"raw\n" + "42 cooked\n42")


def test_raw_tstring_mixed_concat_cooked_then_raw() raises:
    var x = 42
    # fmt: off
    var tstring = (
        t"cooked\n{x}"
        rt" raw\n{x}"
    )
    # fmt: on
    assert_equal(String(tstring), "cooked\n42" + r" raw\n42")


def test_raw_tstring_mixed_concat_three_way() raises:
    var x = 1
    # fmt: off
    var tstring = (
        t"cooked\t{x}"
        rt" raw\t{x}"
        t" cooked\t{x}"
    )
    # fmt: on
    assert_equal(String(tstring), "cooked\t1" + r" raw\t" + "1 cooked\t1")


def test_raw_tstring_in_tstring() raises:
    var x = 42
    var tstring = t"hello \t{x}, {rt"world \t{x}"}"
    assert_equal(String(tstring), "hello \t42, " + r"world \t42")


# =============================================================================
# _encode_format_string tests
# =============================================================================


def _encode(*strings: String) -> List[Byte]:
    var result = List[Byte]()
    for i in range(len(strings)):
        for byte in strings[i].as_bytes():
            result.append(byte)
        result.append(0)
    return result^


def _encode_format_string_to_list(format: StaticString) -> List[Byte]:
    var result = List[Byte]()

    def append(byte: Byte) {mut}:
        result.append(byte)

    _encode_format_string(format, append)
    return result^


def test_encode_plain_text() raises:
    var got = _encode_format_string_to_list("Hello!")
    var want = _encode("Hello!")
    assert_equal(got, want)


def test_encode_empty_string() raises:
    # "" -> "\0"
    assert_equal(_encode_format_string_to_list(""), _encode(""))


def test_encode_single_replacement() raises:
    # "Hello, {}!" -> "Hello, \0!\0"
    assert_equal(
        _encode_format_string_to_list("Hello, {}!"), _encode("Hello, ", "!")
    )


def test_encode_multiple_replacements() raises:
    # "{} + {} = {}" -> "\0 + \0 = \0\0"
    assert_equal(
        _encode_format_string_to_list("{} + {} = {}"),
        _encode("", " + ", " = ", ""),
    )


def test_encode_escaped_braces() raises:
    # "{{braces}}" -> "{braces}\0"
    assert_equal(
        _encode_format_string_to_list("{{braces}}"), _encode("{braces}")
    )


def test_encode_blog_post_example() raises:
    # "Hello, {}, {{I'm}} {}!" -> "Hello, \0, {I'm} \0!\0"
    assert_equal(
        _encode_format_string_to_list("Hello, {}, {{I'm}} {}!"),
        _encode("Hello, ", ", {I'm} ", "!"),
    )


def test_encode_only_replacement_field() raises:
    # "{}" -> "\0\0"
    assert_equal(_encode_format_string_to_list("{}"), _encode("", ""))


def test_encode_adjacent_replacements() raises:
    # "{}{}{}" -> "\0\0\0\0"
    assert_equal(
        _encode_format_string_to_list("{}{}{}"), _encode("", "", "", "")
    )


def test_encode_only_escaped_braces() raises:
    # "{{}}" -> "{}\0"
    assert_equal(_encode_format_string_to_list("{{}}"), _encode("{}"))


def test_encode_adjacent_escaped_braces() raises:
    # "{{}}{{}}" -> "{}{}\0"
    assert_equal(_encode_format_string_to_list("{{}}{{}}"), _encode("{}{}"))


def test_encode_deeply_nested_escaped_braces() raises:
    # "{{{{{}}}}}" -> "{{" + NUL (from {}) + "}}\0"
    assert_equal(
        _encode_format_string_to_list("{{{{{}}}}}"), _encode("{{", "}}")
    )


def test_encode_escaped_braces_adjacent_to_replacement() raises:
    # "{{}}{}{{}}": escaped pair + replacement + escaped pair -> "{}\0{}\0"
    assert_equal(
        _encode_format_string_to_list("{{}}{}{{}}"), _encode("{}", "{}")
    )


def test_encode_replacement_at_start() raises:
    # "{}tail" -> "\0tail\0"
    assert_equal(_encode_format_string_to_list("{}tail"), _encode("", "tail"))


def test_encode_replacement_at_end() raises:
    # "head{}" -> "head\0\0"
    assert_equal(_encode_format_string_to_list("head{}"), _encode("head", ""))


# TODO(https://github.com/modular/modular/issues/6913)
# Re-enable with `assert_aborts` works correctly in a loop.
#
# def test_encode_errors() raises:
#     for invalid in ["hello }", "}}}", "}"]:
#         with assert_raises(contains="single '}'"):
#             _ = _encode_format_string(invalid)

#     for invalid in ["{0}", "{name}", "{abc", "{!r}", "{:.2f}", "{{{", "{}{"]:
#         with assert_raises(
#             contains="unclosed/non-empty replacement field in format string"
#         ):
#             _ = _encode_format_string(invalid)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
