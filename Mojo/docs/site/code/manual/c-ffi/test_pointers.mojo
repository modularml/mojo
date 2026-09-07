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
# test_pointers.mojo
# Tests for c-ffi.mdx: "Passing pointers", the frexp out-parameter example,
# the qsort opaque-pointer + callback example, and the getcwd Array/Span
# example.
from std.ffi import c_char, c_double, c_int, c_size_t, external_call
from std.sys import size_of
from std.testing import assert_equal, assert_true


# --- bitcast a typed pointer to an opaque void* ---


def test_opaque_pointer() raises:
    var value: c_int = 42
    var p = Pointer(to=value)  # Pointer to a C int
    var opaque: OpaquePointer[origin_of(value)] = p.unsafe_bitcast[NoneType]()
    # Reinterpreting the pointer doesn't change the address it holds.
    assert_equal(Int(opaque), Int(p))
    _ = value


# --- typed pointer as a C out-parameter (frexp) ---


def test_frexp_out_parameter() raises:
    var exponent: c_int = 0
    var mantissa = external_call["frexp", c_double](
        c_double(12.0), Pointer(to=exponent)
    )
    # 12.0 == 0.75 * 2**4, so C wrote 4 through the pointer.
    assert_equal(mantissa, c_double(0.75))
    assert_equal(exponent, c_int(4))


# --- opaque pointers plus a callback (qsort) ---


def compare(
    a: OpaquePointer[mut=False, _],
    b: OpaquePointer[mut=False, _],
) abi("C") -> c_int:
    var a_value = a.unsafe_bitcast[c_int]()[]
    var b_value = b.unsafe_bitcast[c_int]()[]
    # `qsort` only needs to know which value is larger. Compare the values
    # instead of subtracting them. Large differences can overflow, producing
    # the wrong comparison result and sorting the values incorrectly.
    if a_value < b_value:
        return c_int(-1)
    return c_int(a_value > b_value)


def test_qsort() raises:
    # The two extremes are far enough apart that a subtracting comparator
    # overflows `c_int` and strands them at the front, so this fixture fails
    # against the `a - b` form the example used to show.
    var numbers: List[c_int] = [5, 2_000_000_000, 9, -2_000_000_000, 1, 5]
    external_call["qsort", NoneType](
        numbers.unsafe_ptr(),
        c_size_t(len(numbers)),
        c_size_t(size_of[c_int]()),
        compare,
    )
    var expected: List[c_int] = [-2_000_000_000, 1, 5, 5, 9, 2_000_000_000]
    for i in range(len(numbers)):
        assert_equal(numbers[i], expected[i])


# --- C fills a Mojo-owned Array, then a Span wraps the result (getcwd) ---


def test_getcwd_array_span() raises:
    comptime CAPACITY = 256
    var buf = Array[c_char, CAPACITY](uninitialized=True)

    var filled = external_call[
        "getcwd", Optional[Pointer[c_char, origin_of(buf)]]
    ](buf.unsafe_ptr(), c_size_t(CAPACITY))
    assert_true(Bool(filled), "getcwd failed")

    var length = external_call["strlen", c_size_t](buf.unsafe_ptr())
    var span = Span(
        unsafe_ptr=buf.unsafe_ptr().unsafe_bitcast[Byte](), length=Int(length)
    )
    # Asserting the real path would tie the test to the machine, so check
    # invariants instead.
    assert_true(len(span) > 0)
    # `strlen` stops before C's terminator, so the byte past the span is nul.
    # This is what ties the span's length to C's idea of where the data ends.
    assert_equal(buf[Int(length)], c_char(0))
    var cwd = String(from_utf8=span)
    assert_equal(len(cwd.as_bytes()), Int(length))
    assert_equal(cwd.as_bytes()[0], Byte(ord("/")))


def main() raises:
    test_opaque_pointer()
    test_frexp_out_parameter()
    test_qsort()
    test_getcwd_array_span()
