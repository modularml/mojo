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
from test_utils.reflection import SimplePoint, NestedStruct, EmptyStruct
from std.benchmark import keep
from std.compile import compile_info
from std.collections.string.format import _FormatUtils
from std.format._utils import write_sequence_to


@fieldwise_init
struct TestWritable(Writable):
    var x: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("write_to: ", self.x)

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("write_repr_to: ", self.x)


def test_repr() raises:
    var t = TestWritable(42)
    assert_equal(repr(t), "write_repr_to: 42")


def test_string_constructor() raises:
    var s = String(TestWritable(42))
    assert_equal(s, "write_to: 42")


def test_format_string() raises:
    assert_equal("{}".format(TestWritable(42)), "write_to: 42")
    assert_equal(String("{}").format(TestWritable(42)), "write_to: 42")
    assert_equal(StringSlice("{}").format(TestWritable(42)), "write_to: 42")

    assert_equal("{!r}".format(TestWritable(42)), "write_repr_to: 42")
    assert_equal(String("{!r}").format(TestWritable(42)), "write_repr_to: 42")
    assert_equal(
        StringSlice("{!r}").format(TestWritable(42)), "write_repr_to: 42"
    )


def test_default_write_to_simple() raises:
    """Test the reflection-based default write_to with a simple struct."""
    var p = SimplePoint(1, 2)
    # Note: get_type_name returns module-qualified names
    assert_equal(String(p), "SimplePoint(x=1, y=2)")
    assert_equal(repr(p), "SimplePoint(x=Int(1), y=Int(2))")


def test_default_write_to_nested() raises:
    """Test the reflection-based default write_to with nested structs."""
    var s = NestedStruct(SimplePoint(3, 4), "test")
    # Note: String's write_repr_to doesn't add quotes (write_to is same as write_repr_to for String)
    assert_equal(
        String(s),
        "NestedStruct(point=SimplePoint(x=3, y=4), name=test)",
    )
    assert_equal(
        repr(s),
        "NestedStruct(point=SimplePoint(x=Int(3), y=Int(4)), name='test')",
    )


def test_default_write_to_empty() raises:
    """Test the reflection-based default write_to with an empty struct."""
    var e = EmptyStruct()
    assert_equal(String(e), "EmptyStruct()")
    assert_equal(repr(e), "EmptyStruct()")


def test_write_sequence_to_with_element_fn_counter() raises:
    """Test write_sequence_to with ElementFn using a simple counter.

    This demonstrates the basic usage of ElementFn: a closure that writes
    elements and raises StopIteration when done.
    """
    var output = String()

    var count = 0

    def write_numbers[T: Writer](mut w: T) raises StopIteration {mut count}:
        if count >= 3:
            raise StopIteration()
        w.write(count)
        count += 1

    write_sequence_to(output, write_numbers)
    assert_equal(output, "[0, 1, 2]")

    _ = count


def test_write_sequence_to_empty_sequence() raises:
    """Test write_sequence_to with ElementFn that immediately raises StopIteration.
    """
    var output = String()

    def write_nothing[T: Writer](mut w: T) raises StopIteration {}:
        raise StopIteration()

    write_sequence_to(output, write_nothing)
    assert_equal(output, "[]")


def test_write_sequence_to_single_element() raises:
    """Test write_sequence_to with ElementFn that writes one element."""
    var output = String()

    var written = False

    def write_once[T: Writer](mut w: T) raises StopIteration {mut written}:
        if written:
            raise StopIteration()
        w.write("only")
        written = True

    write_sequence_to(output, write_once)
    assert_equal(output, "[only]")

    _ = written


def test_write_sequence_to_custom_delimiters() raises:
    """Test write_sequence_to with custom opening, closing, and separator."""
    var output = String()

    var index = 0

    def write_items[T: Writer](mut w: T) raises StopIteration {mut index}:
        if index >= 3:
            raise StopIteration()
        w.write("item", index)
        index += 1

    write_sequence_to(output, write_items, start="{", end="}", sep="; ")
    assert_equal(output, "{item0; item1; item2}")

    _ = index


struct NullWriter(Writer):
    def write_string(mut self, string: StringSlice):
        keep(string)


comptime ALLOC_FUNC = "KGEN_CompilerRT_AlignedAlloc"


def test_format_runtime_does_allocate() raises:
    def runtime_format[
        *Ts: Writable,
    ](mut writer: NullWriter, *args: *Ts) raises:
        _FormatUtils.format_to_runtime(writer, "Hello, {}, {}, {}", *args)

    var info = compile_info[
        runtime_format[Int, String, List[Float32]],
        emission_kind="llvm-opt",
    ]()
    assert_true(ALLOC_FUNC in info, info.asm)


def test_format_comptime_does_not_allocate() raises:
    def comptime_format[
        *Ts: Writable,
    ](mut writer: NullWriter, *args: *Ts):
        _FormatUtils.format_to_comptime["Hello, {}, {}, {}"](writer, *args)

    var info = compile_info[
        comptime_format[Int, String, List[Float32]],
        emission_kind="llvm-opt",
    ]()
    assert_false(ALLOC_FUNC in info)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
