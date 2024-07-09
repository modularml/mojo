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
# RUN: %mojo %s

from testing import assert_equal, assert_true, assert_false

from utils import Span
from utils.string_slice import is_valid_utf8


fn test_string_literal_byte_slice() raises:
    alias string: StringLiteral = "Hello"
    alias slc = string.as_bytes_slice()

    assert_equal(len(slc), 5)
    assert_equal(slc[0], ord("H"))
    assert_equal(slc[1], ord("e"))
    assert_equal(slc[2], ord("l"))
    assert_equal(slc[3], ord("l"))
    assert_equal(slc[4], ord("o"))


fn test_string_byte_slice() raises:
    var string = String("Hello")
    var str_slice = string.as_bytes_slice()

    assert_equal(len(str_slice), 5)
    assert_equal(str_slice[0], ord("H"))
    assert_equal(str_slice[1], ord("e"))
    assert_equal(str_slice[2], ord("l"))
    assert_equal(str_slice[3], ord("l"))
    assert_equal(str_slice[4], ord("o"))

    # ----------------------------------
    # Test subslicing
    # ----------------------------------

    # Slice the whole thing
    var sub1 = str_slice[:5]
    assert_equal(len(sub1), 5)
    assert_equal(sub1[0], ord("H"))
    assert_equal(sub1[1], ord("e"))
    assert_equal(sub1[2], ord("l"))
    assert_equal(sub1[3], ord("l"))
    assert_equal(sub1[4], ord("o"))

    # Slice the end
    var sub2 = str_slice[2:5]
    assert_equal(len(sub2), 3)
    assert_equal(sub2[0], ord("l"))
    assert_equal(sub2[1], ord("l"))
    assert_equal(sub2[2], ord("o"))

    # Slice the first element
    var sub3 = str_slice[0:1]
    assert_equal(len(sub3), 1)
    assert_equal(sub3[0], ord("H"))

    #
    # Test mutation through slice
    #

    sub1[0] = ord("J")
    assert_equal(string, "Jello")

    sub2[2] = ord("y")
    assert_equal(string, "Jelly")

    # ----------------------------------
    # Test empty subslicing
    # ----------------------------------

    var sub4 = str_slice[0:0]
    assert_equal(len(sub4), 0)

    var sub5 = str_slice[2:2]
    assert_equal(len(sub5), 0)

    # Empty slices still have a pointer value
    assert_equal(int(sub5.unsafe_ptr()) - int(sub4.unsafe_ptr()), 2)

    # ----------------------------------
    # Test invalid slicing
    # ----------------------------------

    # TODO: Improve error reporting for invalid slice bounds.

    # assert_equal(
    #     # str_slice[3:6]
    #     str_slice._try_slice(slice(3, 6)).unwrap[String](),
    #     String("Slice end is out of bounds"),
    # )

    # assert_equal(
    #     # str_slice[5:6]
    #     str_slice._try_slice(slice(5, 6)).unwrap[String](),
    #     String("Slice start is out of bounds"),
    # )

    # assert_equal(
    #     # str_slice[5:5]
    #     str_slice._try_slice(slice(5, 5)).unwrap[String](),
    #     String("Slice start is out of bounds"),
    # )


fn test_heap_string_from_string_slice() raises:
    alias string_lit: StringLiteral = "Hello"

    alias static_str: StringSlice[
        ImmutableStaticLifetime
    ] = string_lit.as_string_slice()

    alias heap_string = String(static_str)

    assert_equal(heap_string, "Hello")


fn test_slice_len() raises:
    alias str1: StringLiteral = "12345"
    alias str2: StringLiteral = "1234"
    alias str3: StringLiteral = "123"
    alias str4: StringLiteral = "12"
    alias str5: StringLiteral = "1"

    alias slice1: StringSlice[ImmutableStaticLifetime] = str1.as_string_slice()
    alias slice2: StringSlice[ImmutableStaticLifetime] = str2.as_string_slice()
    alias slice3: StringSlice[ImmutableStaticLifetime] = str3.as_string_slice()
    alias slice4: StringSlice[ImmutableStaticLifetime] = str4.as_string_slice()
    alias slice5: StringSlice[ImmutableStaticLifetime] = str5.as_string_slice()

    assert_equal(5, len(slice1))
    assert_equal(4, len(slice2))
    assert_equal(3, len(slice3))
    assert_equal(2, len(slice4))
    assert_equal(1, len(slice5))


fn test_slice_eq() raises:
    var str1: String = "12345"
    var str2: String = "12345"
    var str3: StringLiteral = "12345"
    var str4: String = "abc"
    var str5: String = "abcdef"
    var str6: StringLiteral = "abcdef"

    # eq

    assert_true(str1.as_string_slice().__eq__(str1))
    assert_true(str1.as_string_slice().__eq__(str2))
    assert_true(str2.as_string_slice().__eq__(str2.as_string_slice()))
    assert_true(str1.as_string_slice().__eq__(str3))

    # ne

    assert_true(str1.as_string_slice().__ne__(str4))
    assert_true(str1.as_string_slice().__ne__(str5))
    assert_true(str1.as_string_slice().__ne__(str5.as_string_slice()))
    assert_true(str1.as_string_slice().__ne__(str6))


fn test_slice_bool() raises:
    var str1: String = "abc"
    assert_true(str1.as_string_slice().__bool__())
    var str2: String = ""
    assert_true(not str2.as_string_slice().__bool__())


fn test_utf8_validation() raises:
    var positive = List[List[UInt8]](
        List[UInt8](0x0),
        List[UInt8](0x00),
        List[UInt8](0x66),
        List[UInt8](0x7F),
        List[UInt8](0x00, 0x7F),
        List[UInt8](0x7F, 0x00),
        List[UInt8](0xC2, 0x80),
        List[UInt8](0xDF, 0xBF),
        List[UInt8](0xE0, 0xA0, 0x80),
        List[UInt8](0xE0, 0xA0, 0xBF),
        List[UInt8](0xED, 0x9F, 0x80),
        List[UInt8](0xEF, 0x80, 0xBF),
        List[UInt8](0xF0, 0x90, 0xBF, 0x80),
        List[UInt8](0xF2, 0x81, 0xBE, 0x99),
        List[UInt8](0xF4, 0x8F, 0x88, 0xAA),
    )
    for item in positive:
        assert_true(is_valid_utf8(item[].unsafe_ptr(), len(item[])))
    _ = positive
    var negative = List[List[UInt8]](
        List[UInt8](0x80),
        List[UInt8](0xBF),
        List[UInt8](0xC0, 0x80),
        List[UInt8](0xC1, 0x00),
        List[UInt8](0xC2, 0x7F),
        List[UInt8](0xDF, 0xC0),
        List[UInt8](0xE0, 0x9F, 0x80),
        List[UInt8](0xE0, 0xC2, 0x80),
        List[UInt8](0xED, 0xA0, 0x80),
        List[UInt8](0xED, 0x7F, 0x80),
        List[UInt8](0xEF, 0x80, 0x00),
        List[UInt8](0xF0, 0x8F, 0x80, 0x80),
        List[UInt8](0xF0, 0xEE, 0x80, 0x80),
        List[UInt8](0xF2, 0x90, 0x91, 0x7F),
        List[UInt8](0xF4, 0x90, 0x88, 0xAA),
        List[UInt8](0xF4, 0x00, 0xBF, 0xBF),
        List[UInt8](
            0xC2, 0x80, 0x00, 0x00, 0xE1, 0x80, 0x80, 0x00, 0xC2, 0xC2, 0x80
        ),
        List[UInt8](0x00, 0xC2, 0xC2, 0x80, 0x00, 0x00, 0xE1, 0x80, 0x80),
        List[UInt8](0x00, 0x00, 0x00, 0xF1, 0x80),
        List[UInt8](0x00, 0xF1),
        List[UInt8](0x00, 0x00, 0x00, 0x00, 0xF1, 0x80, 0x80),
        List[UInt8](0x00, 0xF1, 0x80, 0xC2, 0x80),
        List[UInt8](0x00, 0x00, 0xF0, 0x80, 0x80, 0x80),
    )
    for item in negative:
        print(item[].__str__())
        assert_false(is_valid_utf8(item[].unsafe_ptr(), len(item[])))
    _ = negative


fn main() raises:
    test_string_literal_byte_slice()
    test_string_byte_slice()
    test_heap_string_from_string_slice()
    test_slice_len()
    test_slice_eq()
    test_slice_bool()
    test_utf8_validation()
