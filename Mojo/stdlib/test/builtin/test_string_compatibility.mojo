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

from std.testing import assert_equal, assert_true, TestSuite


def test_literals() raises:
    """Test string literal operations and type inference.

    String literals now materialize to String by default.
    """

    var cond = True

    # Concatenation with literals
    var concat_static = "foo" + StaticString("bar")
    var concat_static2 = StringSlice("foo") + "bar"
    assert_equal(concat_static, "foobar")
    assert_equal(concat_static2, "foobar")

    # Conditional expressions with literals
    var if_string1 = "foo" if cond else StaticString("bar")
    var if_string2 = StringSlice("foo") if cond else "bar"
    assert_equal(if_string1, "foo")
    assert_equal(if_string2, "foo")

    # Logical or with literals
    var or_string1 = "" or StaticString("foo")
    var or_string2 = StringSlice("foo") or ""
    assert_equal(or_string1, "foo")
    assert_equal(or_string2, "foo")

    # In-place operations
    var in_place = "this is a string"
    in_place += " in-place operation"
    assert_equal(in_place, "this is a string in-place operation")


def test_alias_expressions() raises:
    """Test string alias expressions."""
    comptime alias_concat = "foo" + "bar"
    comptime alias_concat_string = "foo" + String("bar")
    comptime alias_concat_static = StaticString("foo") + "bar"

    comptime alias_or = "foo" or "bar"
    comptime alias_or_string = "" or String("bar")
    comptime alias_or_static = StaticString("foo") or "bar"

    comptime alias_if = "foo" if True else "bar"
    comptime alias_if_string = "foo" if False else String("bar")
    comptime alias_if_static = StaticString("foo") if True else "bar"

    # Test materialization of alias expressions
    assert_equal(alias_concat, "foobar")
    assert_equal(alias_concat_string, "foobar")
    assert_equal(alias_concat_static, "foobar")

    assert_equal(alias_or, "foo")
    assert_equal(alias_or_string, "bar")
    assert_equal(alias_or_static, "foo")

    assert_equal(alias_if, "foo")
    assert_equal(alias_if_string, "bar")
    assert_equal(alias_if_static, "foo")


def _test_string_types_compatibility(
    string: String,
    static_string: StaticString,
    string_slice: StringSlice,
    cond: Bool = True,
) raises:
    """Test compatibility between String, StaticString, and StringSlice in various operations.
    Focus on String and StringSlice compatibility.
    """

    # String concatenation tests
    var add1 = string + static_string
    var add2 = static_string + string
    var add3 = static_string + string_slice
    # Test concatenation results by comparing with expected strings
    assert_equal(add1, String(string) + String(static_string))
    assert_equal(add2, String(static_string) + String(string))
    assert_equal(add3, String(static_string) + String(string_slice))

    # String multiplication tests
    var mul1 = string * 2
    var mul2 = static_string * 2
    var mul3 = string_slice * 2
    assert_equal(mul1, String(string) * 2)
    assert_equal(mul2, String(static_string) * 2)
    assert_equal(mul3, String(string_slice) * 2)

    # Conditional expression tests - fix type compatibility issues
    var if1 = string if cond else static_string
    var if2 = static_string if cond else string
    # Convert StringSlice to String for compatibility
    var if3 = string if cond else String(string_slice)
    # Since cond is True, we expect the first operands
    assert_equal(if1, string)
    assert_equal(if2, String(static_string))
    assert_equal(if3, string)

    # Logical or tests - fix type compatibility issues
    var or1 = string or static_string
    var or2 = static_string or string
    # Convert StringSlice to String for compatibility
    var or3 = string or String(string_slice)
    # Since inputs are truthy, we expect the first operands
    assert_equal(or1, string)
    assert_equal(or2, String(static_string))
    assert_equal(or3, string)


def _test_string_slice_conversions(
    string: String,
    string_slice: StringSlice,
) raises:
    """Test explicit conversions between String and StringSlice."""
    # Convert String to StringSlice
    var slice_from_string = StringSlice(string)
    assert_equal(String(slice_from_string), string)

    # Convert StringSlice to String
    var string_from_slice = String(string_slice)
    assert_equal(string_from_slice, String(string_slice))

    # Test round-trip conversion
    var round_trip = String(StringSlice(string))
    assert_equal(round_trip, string)

    # Test that StringSlice content matches when converted to String
    assert_equal(String(string_slice), String(string_slice))


def _test_equality_operations(
    string: String, static_string: StaticString, string_slice: StringSlice
) raises:
    """Test equality operations between different string types."""
    var string_a = String("hello")
    var static_a = "hello"
    # Use a separate string for the slice to avoid aliasing issues
    var string_for_slice = String("hello")
    var slice_a = StringSlice(string_for_slice)

    var string_b = String("world")
    var static_b = "world"

    # Equality tests
    assert_true(string_a == static_a)
    assert_true(static_a == string_a)
    assert_true(string_a == slice_a)
    assert_true(slice_a == static_a)

    # Inequality tests
    assert_true(string_a != string_b)
    assert_true(static_a != static_b)
    assert_true(string_a != static_b)


def _test_chained_operations(
    string: String,
    static_string: StaticString,
    string_slice: StringSlice,
    cond: Bool = True,
) raises:
    """Test chained operations with mixed string types."""
    # Chained concatenation
    var chained = string + static_string + string_slice
    var expected_chained = (
        String(string) + String(static_string) + String(string_slice)
    )
    assert_equal(chained, expected_chained)

    # Mixed operations
    var mixed = (string + static_string) * 2
    var expected_mixed = (String(string) + static_string) * 2
    assert_equal(mixed, expected_mixed)

    # Complex conditional - fix type compatibility
    var complex_cond = (string + static_string) if cond else (
        static_string + String(string_slice)
    )
    var expected_complex = (String(string) + static_string) if cond else (
        static_string + String(string_slice)
    )
    assert_equal(complex_cond, expected_complex)


def _test_with_collections(string: String, static_string: StaticString) raises:
    """Test string types in collection contexts."""
    # List operations with mixed string types
    var string_list = List[String]()
    string_list.append(string)
    string_list.append(static_string)  # Convert StaticString to String

    # Verify list contents
    assert_equal(string_list[0], string)
    assert_equal(string_list[1], static_string)

    # Dict operations with mixed string types (if Dict is available)
    # Note: This section may need adjustment based on Mojo's Dict implementation
    var string_dict = Dict[String, String]()
    string_dict[static_string + "_key"] = string
    string_dict[string + "_key"] = static_string

    assert_equal(string_dict[static_string + "_key"], string)
    assert_equal(string_dict[string + "_key"], static_string)

    # Test constructing a List with `StaticString` and `String` types
    var list_of_strings: List[String] = [string, static_string]
    assert_equal(list_of_strings[0], string)
    assert_equal(list_of_strings[1], static_string)


def _test_dict_literals(
    string: String,
    static_string: StaticString,
    cond: Bool = True,
) raises:
    """Test dict literal initialization with different string types."""
    # Mixed dict literals - StaticString keys with String values
    var mixed_dict1 = {
        "static_key": String("string_value"),
        "another_key": static_string,
    }
    assert_equal(mixed_dict1["static_key"], String("string_value"))
    assert_equal(mixed_dict1["another_key"], static_string)

    # Mixed dict literals with `String` and `StaticString`
    var mixed_dict2 = {
        String("string_key"): "static_value",
        string: static_string,
    }
    assert_equal(mixed_dict2[String("string_key")], "static_value")
    assert_equal(mixed_dict2[string], static_string)

    # Complex dict literals with expressions as keys and values
    var complex_dict = {
        string + "_suffix": static_string + "_value",
        static_string + "_key": string + "_result",
    }
    assert_equal(complex_dict[string + "_suffix"], static_string + "_value")
    assert_equal(complex_dict[static_string + "_key"], string + "_result")
    # Dict literal with computed keys and values
    var computed_dict = {
        string if cond else static_string: static_string if cond else string,
        "conditional": string if string else static_string,
    }
    assert_equal(computed_dict[string], static_string)
    assert_equal(computed_dict["conditional"], string)


def _test_list_comprehensions(
    string: String,
    static_string: StaticString,
) raises:
    """Test list comprehensions with mixed string types."""

    # List comprehension with mixed string types and operations
    var static_strings: List[String] = ["one", "two", "three"]
    var mixed_ops = [static_string + String("_") + s for s in static_strings]
    assert_equal(mixed_ops[0], String(static_string) + "_one")
    assert_equal(mixed_ops[1], String(static_string) + "_two")
    assert_equal(mixed_ops[2], String(static_string) + "_three")

    # List comprehension with complex conditional logic and mixed types
    var test_strings: List[String] = ["short", "medium", "a", "long_string"]
    var categorized = [
        "short" if s.byte_length()
        < 6 else ("medium" if s.byte_length() < 11 else String("long"))
        for s in test_strings
    ]
    assert_equal(categorized[0], "short")
    assert_equal(categorized[1], "medium")
    assert_equal(categorized[2], "short")
    assert_equal(categorized[3], "long")

    # List comprehension combining different string types
    var string_types_list: List[String] = [string, static_string]
    var combined = [s + "_" + static_string for s in string_types_list]
    assert_equal(combined[0], string + "_" + static_string)
    assert_equal(combined[1], static_string + "_" + static_string)


def _test_conditional_edge_cases(
    string: String,
    static_string: StaticString,
    cond1: Bool = True,
    cond2: Bool = False,
) raises:
    """Test edge cases with conditional expressions."""
    var empty_string = String("")

    # Nested conditionals
    var nested = string if cond1 else (static_string if cond2 else empty_string)
    assert_equal(nested, string)

    # Conditionals with empty strings
    var empty_cond1 = empty_string if cond1 else static_string
    var empty_cond2 = static_string if cond2 else empty_string
    assert_equal(empty_cond1, empty_string)
    assert_equal(empty_cond2, empty_string)


def test_string_types_compatibility() raises:
    var string = "string"
    var static_string = StaticString("static_string")
    var string_slice = StringSlice("string_slice")
    _test_string_types_compatibility(string, static_string, string_slice)


def test_string_slice_conversions() raises:
    var string = "string"
    var string_slice = StringSlice("string_slice")
    _test_string_slice_conversions(string, string_slice)


def test_equality_operations() raises:
    var string = "string"
    var static_string = StaticString("static_string")
    var string_slice = StringSlice("string_slice")
    _test_equality_operations(string, static_string, string_slice)


def test_chained_operations() raises:
    var string = "string"
    var static_string = StaticString("static_string")
    var string_slice = StringSlice("string_slice")
    _test_chained_operations(string, static_string, string_slice)


def test_with_collections() raises:
    var string = "string"
    var static_string = StaticString("static_string")
    _test_with_collections(string, static_string)


def test_dict_literals() raises:
    var string = "string"
    var static_string = StaticString("static_string")
    _test_dict_literals(string, static_string)


def test_list_comprehensions() raises:
    var string = "string"
    var static_string = StaticString("static_string")
    _test_list_comprehensions(string, static_string)


def test_conditional_edge_cases() raises:
    var string = "string"
    var static_string = StaticString("static_string")
    _test_conditional_edge_cases(string, static_string)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
