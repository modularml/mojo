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
"""Tests for format._utils helper types and functions.

This module tests the internal formatting utilities used by the standard library
for implementing Writable and formatting operations.
"""

from std.format._utils import FormatStruct, Named, Repr, write_sequence_to
from std.testing import assert_equal, TestSuite


# ===----------------------------------------------------------------------=== #
# Test Fixtures (shared across multiple tests)
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct TestWritable(ImplicitlyCopyable, Writable):
    """A simple writable type for testing."""

    var value: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("TestWritable(", self.value, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("TestWritable[repr](", self.value, ")")


# ===----------------------------------------------------------------------=== #
# write_sequence_to Tests
# ===----------------------------------------------------------------------=== #


def test_write_sequence_empty() raises:
    var result = String()
    write_sequence_to(result, start="[", end="]")
    assert_equal(result, "[]")


def test_write_sequence_single_element() raises:
    var result = String()
    write_sequence_to(result, 42, start="[", end="]")
    assert_equal(result, "[42]")


def test_write_sequence_multiple_elements() raises:
    var result = String()
    write_sequence_to(result, 1, 2, 3, start="[", end="]")
    assert_equal(result, "[1, 2, 3]")


def test_write_sequence_custom_delimiters() raises:
    var result = String()
    write_sequence_to(result, 1, 2, 3, start="(", end=")")
    assert_equal(result, "(1, 2, 3)")


def test_write_sequence_custom_separator() raises:
    var result = String()
    write_sequence_to(result, 1, 2, 3, start="[", end="]", sep="; ")
    assert_equal(result, "[1; 2; 3]")


def test_write_sequence_custom_all() raises:
    var result = String()
    write_sequence_to(result, "a", "b", "c", start="<", end=">", sep=" | ")
    assert_equal(result, "<a | b | c>")


def test_write_sequence_writable_objects() raises:
    var result = String()
    write_sequence_to(
        result, TestWritable(10), TestWritable(20), start="[", end="]"
    )
    assert_equal(result, "[TestWritable(10), TestWritable(20)]")


def test_write_sequence_mixed_types() raises:
    var result = String()
    write_sequence_to(result, 1, "hello", TestWritable(42), start="[", end="]")
    assert_equal(result, "[1, hello, TestWritable(42)]")


def test_write_sequence_empty_separator() raises:
    var result = String()
    write_sequence_to(result, 1, 2, 3, start="[", end="]", sep="")
    assert_equal(result, "[123]")


# ===----------------------------------------------------------------------=== #
# Repr Tests
# ===----------------------------------------------------------------------=== #


def test_repr_basic() raises:
    var v = TestWritable(42)
    # Repr should call write_repr_to
    assert_equal(String(Repr(v)), "TestWritable[repr](42)")


def test_repr_vs_direct() raises:
    var v = TestWritable(100)
    var repr_result = String(Repr(v))
    var direct_result = String(v)

    assert_equal(repr_result, "TestWritable[repr](100)")
    assert_equal(direct_result, "TestWritable(100)")


# ===----------------------------------------------------------------------=== #
# Named Tests
# ===----------------------------------------------------------------------=== #


def test_named() raises:
    # basic
    assert_equal(String(Named("value", 42)), "value=42")

    # with writable
    assert_equal(
        String(Named("field", TestWritable(100))), "field=TestWritable(100)"
    )

    # empty name
    assert_equal(String(Named("", 42)), "=42")


# ===----------------------------------------------------------------------=== #
# FormatStruct Tests
# ===----------------------------------------------------------------------=== #


def test_format_struct_name_only() raises:
    var result = String()
    FormatStruct(result, "MyStruct").fields()
    assert_equal(result, "MyStruct()")


def test_format_struct_with_fields() raises:
    var result = String()
    FormatStruct(result, "Point").fields(10, 20)
    assert_equal(result, "Point(10, 20)")


def test_format_struct_with_params() raises:
    var result = String()
    FormatStruct(result, "Array").params("Int", 10).fields()
    assert_equal(result, "Array[Int, 10]()")


def test_format_struct_with_params_and_fields() raises:
    var result = String()
    FormatStruct(result, "Container").params("String").fields("value", 42)
    assert_equal(result, "Container[String](value, 42)")


def test_format_struct_multiple_params() raises:
    var result = String()
    FormatStruct(result, "Pair").params("Int", "String").fields(1, "test")
    assert_equal(result, "Pair[Int, String](1, test)")


def test_format_struct_single_param() raises:
    var result = String()
    FormatStruct(result, "Optional").params("String").fields()
    assert_equal(result, "Optional[String]()")


def test_format_struct_single_field() raises:
    var result = String()
    FormatStruct(result, "Wrapper").fields(42)
    assert_equal(result, "Wrapper(42)")


def test_format_struct_empty() raises:
    var result = String()
    FormatStruct(result, "Empty").fields()
    assert_equal(result, "Empty()")


# ===----------------------------------------------------------------------=== #
# Integration Tests
# ===----------------------------------------------------------------------=== #


def test_struct_repr() raises:
    var result = String()
    FormatStruct(result, "MyPoint").fields(
        Named("x", Repr(10)),
        Named("y", Repr(20)),
        Named("name", Repr(String("point"))),
    )
    assert_equal(result, "MyPoint(x=Int(10), y=Int(20), name='point')")


def test_parametric_struct() raises:
    var result = String()
    FormatStruct(result, "Array").params("Int", 10).fields(
        Named("data", 42), Named("len", 5)
    )
    assert_equal(result, "Array[Int, 10](data=42, len=5)")


def test_nested_formatting() raises:
    var inner_result = String()
    FormatStruct(inner_result, "Inner").fields(Named("x", 1), Named("y", 2))

    # Use that in outer struct
    var outer_result = String()
    FormatStruct(outer_result, "Outer").fields(
        Named("inner", inner_result), Named("count", 5)
    )
    assert_equal(outer_result, "Outer(inner=Inner(x=1, y=2), count=5)")


# ===----------------------------------------------------------------------=== #
# Main Test Runner
# ===----------------------------------------------------------------------=== #


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
