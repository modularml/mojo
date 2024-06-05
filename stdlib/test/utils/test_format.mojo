# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo -debug-level full %s

from testing import assert_equal

from utils._format import Formattable, Formatter, write_to
from utils.inline_string import _FixedString


fn main() raises:
    test_formatter_of_string()
    test_string_format_seq()
    test_stringable_based_on_format()

    test_formatter_of_fixed_string()


@value
struct Point(Formattable, Stringable):
    var x: Int
    var y: Int

    fn format_to(self, inout writer: Formatter):
        write_to(writer, "Point(", self.x, ", ", self.y, ")")

    fn __str__(self) -> String:
        return String.format_sequence(self)


fn test_formatter_of_string() raises:
    #
    # Test format_to(String)
    #
    var s1 = String()
    var s1_fmt = Formatter(s1)
    Point(2, 7).format_to(s1_fmt)
    assert_equal(s1, "Point(2, 7)")

    #
    # Test write_to(String, ..)
    #
    var s2 = String()
    var s2_fmt = Formatter(s2)
    write_to(s2_fmt, Point(3, 8))
    assert_equal(s2, "Point(3, 8)")


fn test_string_format_seq() raises:
    var s1 = String.format_sequence("Hello, ", "World!")
    assert_equal(s1, "Hello, World!")

    var s2 = String.format_sequence("point = ", Point(2, 7))
    assert_equal(s2, "point = Point(2, 7)")

    var s3 = String.format_sequence()
    assert_equal(s3, "")


fn test_stringable_based_on_format() raises:
    assert_equal(str(Point(10, 11)), "Point(10, 11)")


fn test_formatter_of_fixed_string() raises:
    var s1 = _FixedString[100]()
    var s1_fmt = Formatter(s1)
    write_to(s1_fmt, "Hello, World!")
    assert_equal(str(s1), "Hello, World!")
