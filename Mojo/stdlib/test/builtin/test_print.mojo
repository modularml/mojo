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

from std.tempfile import NamedTemporaryFile

from std.reflection import call_location, SourceLocation
from std.testing import TestSuite

from std.utils import IndexList


@always_inline
def _assert_error[T: Writable](msg: T, loc: SourceLocation) -> Error:
    return Error(loc.prefix(String("AssertionError: ", msg)))


def _assert_equal_error(
    lhs: String, rhs: String, msg: String, loc: SourceLocation
) -> Error:
    var err = (
        "`left == right` comparison failed:\n   left: "
        + lhs
        + "\n  right: "
        + rhs
    )
    if msg:
        err += "\n  reason: " + msg
    return _assert_error(err, loc)


struct PrintChecker(Movable):
    var tmp: NamedTemporaryFile
    var cursor: Int
    var call_location: SourceLocation

    @always_inline
    def __init__(out self) raises:
        self.tmp = NamedTemporaryFile("rw")
        self.call_location = call_location()
        self.cursor = 0

    def __enter__(var self) -> Self:
        return self^

    def stream(self) -> FileDescriptor:
        return FileDescriptor(self.tmp._file_handle._get_raw_fd())

    def check_line(mut self, expected: String, msg: String = "") raises:
        print(end="", file=self.stream(), flush=True)
        _ = self.tmp.seek(self.cursor)
        var contents = self.tmp.read()
        # An empty read means the expected line was never printed; clamp so
        # the comparison below reports the mismatch instead of aborting.
        var result = contents[byte = : max(contents.byte_length() - 1, 0)]
        if result != expected:
            raise _assert_equal_error(
                String(result), expected, msg, self.call_location
            )
        self.cursor += result.byte_length() + 1

    def check_line_starts_with(
        mut self, prefix: String, msg: String = ""
    ) raises:
        print(end="", file=self.stream(), flush=True)
        _ = self.tmp.seek(self.cursor)
        var contents = self.tmp.read()
        # An empty read means the expected line was never printed; clamp so
        # the comparison below reports the mismatch instead of aborting.
        var result = contents[byte = : max(contents.byte_length() - 1, 0)]
        var prefix_len = prefix.byte_length()
        if result.byte_length() < prefix_len:
            raise _assert_error(msg, self.call_location)
        if result[byte=:prefix_len] != prefix:
            raise _assert_equal_error(
                String(result[byte=:prefix_len]),
                prefix,
                msg,
                self.call_location,
            )
        self.cursor += result.byte_length() + 1


def test_print() raises:
    with PrintChecker() as checker:
        print("Hello", file=checker.stream())
        checker.check_line("Hello")

        print("World", flush=True, file=checker.stream())
        checker.check_line("World")

        var hello: StaticString = "Hello,"
        var world: String = "world!"
        var f: Bool = False
        print(">", hello, world, 42, True, f, file=checker.stream())
        checker.check_line("> Hello, world! 42 True False")

        var float32: Float32 = 99.9
        var float64: Float64 = -129.2901823
        print(">", 3.14, file=checker.stream())
        checker.check_line("> 3.14")
        print(">", float32, file=checker.stream())
        checker.check_line("> 99.9")
        print(">", float64, file=checker.stream())
        checker.check_line("> -129.2901823")
        print(">", IndexList[3](1, 2, 3), file=checker.stream())
        checker.check_line_starts_with("> (1, 2, 3)")

        print(">", 9223372036854775806, file=checker.stream())
        checker.check_line("> 9223372036854775806")

        var pi = 3.1415916535897743
        print(">", pi, file=checker.stream())
        checker.check_line("> 3.1415916535897743")
        var x = (pi - 3.141591) * 1e6
        print(">", x, file=checker.stream())
        checker.check_line_starts_with("> 0.6535")

        print("Hello world", file=checker.stream())
        checker.check_line("Hello world")


def test_print_end() raises:
    # Test string literals
    with PrintChecker() as checker:
        print("Hello", end=" World\n", file=checker.stream())
        checker.check_line("Hello World")

    # Test runtime strings
    with PrintChecker() as checker:
        var dyn_string: String = "End\n"
        print("Start", end=dyn_string, file=checker.stream())
        checker.check_line("StartEnd")


def test_print_sep() raises:
    # Test string literals
    with PrintChecker() as checker:
        print("a", "b", "c", sep="/", file=checker.stream())
        checker.check_line("a/b/c")

        print("a", 1, 2, sep="/", end="xx\n", file=checker.stream())
        checker.check_line("a/1/2xx")

    # Test runtime strings
    with PrintChecker() as checker:
        var dyn_string: String = ":"
        print("1", "2", sep=dyn_string, file=checker.stream())
        checker.check_line("1:2")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
