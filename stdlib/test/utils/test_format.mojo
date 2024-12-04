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

from utils import Writable, Writer
from utils.inline_string import _FixedString


fn main() raises:
    test_writer_of_string()
    test_string_format_seq()
    test_stringable_based_on_format()

    test_writer_of_fixed_string()

    test_write_int_padded()


@value
struct Point(Writable, Stringable):
    var x: Int
    var y: Int

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Point(", self.x, ", ", self.y, ")")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)


fn test_writer_of_string() raises:
    #
    # Test write_to(String)
    #
    var s1 = String()
    Point(2, 7).write_to(s1)
    assert_equal(s1, "Point(2, 7)")

    #
    # Test writer.write(String, ..)
    #
    var s2 = String()
    s2.write(Point(3, 8))
    assert_equal(s2, "Point(3, 8)")


fn test_string_format_seq() raises:
    var s1 = String.write("Hello, ", "World!")
    assert_equal(s1, "Hello, World!")

    var s2 = String.write("point = ", Point(2, 7))
    assert_equal(s2, "point = Point(2, 7)")

    var s3 = String.write()
    assert_equal(s3, "")


fn test_stringable_based_on_format() raises:
    assert_equal(str(Point(10, 11)), "Point(10, 11)")


fn test_writer_of_fixed_string() raises:
    var s1 = _FixedString[100]()
    s1.write("Hello, World!")
    assert_equal(str(s1), "Hello, World!")


fn test_write_int_padded() raises:
    var s1 = String()

    Int(5).write_padded(s1, width=5)

    assert_equal(s1, "    5")

    Int(123).write_padded(s1, width=5)

    assert_equal(s1, "    5  123")

    # ----------------------------------
    # Test writing int larger than width
    # ----------------------------------

    var s2 = String()

    Int(12345).write_padded(s2, width=3)

    assert_equal(s2, "12345")
