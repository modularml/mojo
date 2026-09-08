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

from std.testing import (
    TestSuite,
    assert_equal,
    assert_raises,
    assert_true,
    assert_false,
)
from std.ffi import CStringSlice, external_call
from test_utils import check_write_to


def test_init_from_invalid_string() raises:
    with assert_raises(contains="not nul-terminated"):
        _ = CStringSlice(String(""))

    with assert_raises(contains="not nul-terminated"):
        _ = CStringSlice(String("mojo!"))

    with assert_raises(contains="interior nul byte"):
        _ = CStringSlice(String("mojo\0mojo"))


def test_init_from_invalid_byte_span() raises:
    with assert_raises(contains="not nul-terminated"):
        _ = CStringSlice(Span[Byte, ImmUntrackedOrigin]())

    with assert_raises(contains="not nul-terminated"):
        _ = CStringSlice([Byte(1), Byte(2)])

    with assert_raises(contains="interior nul byte"):
        _ = CStringSlice([Byte(1), Byte(0), Byte(2)])


def test_c_string_slice_from_ptr() raises:
    var string = String("mojo!\0")
    var ptr = string.as_bytes().unsafe_ptr().unsafe_bitcast[Int8]()
    var cslice = CStringSlice(unsafe_from_ptr=ptr)
    assert_equal(len(cslice), 5)
    assert_equal(String(cslice), "mojo!")
    assert_equal(Int(cslice.ptr()), Int(ptr))


def test_c_string_slice_from_nul_string() raises:
    var string = String("\0")
    var cslice = CStringSlice(string)
    assert_equal(len(cslice), 0)
    assert_equal(String(cslice), "")
    assert_equal(Int(cslice.ptr()), Int(string.as_bytes().unsafe_ptr()))


def test_c_string_slice_from_nul_span() raises:
    var span: List[Byte] = [Byte(0)]
    var cslice = CStringSlice(span)
    assert_equal(len(cslice), 0)
    assert_equal(String(cslice), "")
    assert_equal(Int(cslice.ptr()), Int(span.unsafe_ptr()))


def test_c_string_slice_from_string() raises:
    var string = String("mojo!\0")
    var cslice = CStringSlice(string)
    assert_equal(len(cslice), 5)
    assert_equal(String(cslice), "mojo!")
    assert_equal(Int(cslice.ptr()), Int(string.as_bytes().unsafe_ptr()))


def test_c_string_slice_from_span() raises:
    var string: List[Byte] = [
        Byte(109),
        Byte(111),
        Byte(106),
        Byte(111),
        Byte(33),
        Byte(0),
    ]
    var cslice = CStringSlice(string)
    assert_equal(len(cslice), 5)
    assert_equal(String(cslice), "mojo!")
    assert_equal(Int(cslice.ptr()), Int(string.unsafe_ptr()))


def test_c_string_copy() raises:
    var string = String("mojo!\0")
    var cslice = CStringSlice(string)

    var copy = cslice
    assert_true(copy == cslice)
    assert_equal(Int(copy.ptr()), Int(cslice.ptr()))


def test_c_string_eq() raises:
    var first = CStringSlice(String("mojo!\0"))
    var second = CStringSlice(String("mojo!\0"))
    var third = CStringSlice(String("not mojo\0"))

    assert_true(first == first)
    assert_true(first == second)
    assert_true(first != third)
    assert_true(second != third)


def test_c_string_write_to() raises:
    var string = String("mojo\0")
    var cslice = CStringSlice(string)
    check_write_to(cslice, expected="mojo", is_repr=False)
    check_write_to(
        cslice, expected="CStringSlice([109, 111, 106, 111, 0])", is_repr=True
    )

    var empty = String("\0")
    var emptyslice = CStringSlice(empty)
    check_write_to(emptyslice, expected="", is_repr=False)
    check_write_to(emptyslice, expected="CStringSlice([0])", is_repr=True)


def test_c_string_as_bytes() raises:
    var string = String("mojo!\0")
    var cslice = CStringSlice(string)

    assert_equal(len(cslice.as_bytes()), 5)
    assert_equal(len(cslice.as_bytes_with_nul()), 6)

    assert_true(string.as_bytes() != cslice.as_bytes())
    assert_true(string.as_bytes() == cslice.as_bytes_with_nul())


def test_c_string_external_call() raises:
    var string = "THIS-ENV-VAR-DOES-NOT-EXIST-MOJO-IS-COOL"
    var result = external_call[
        "getenv",
        Optional[CStringSlice[ImmStaticOrigin]],
    ](string.as_c_string_slice())
    assert_false(result)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
