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
"""Tests for the _memchr, _memmem, _memrchr, and _memrmem functions."""

from std.collections.string.string_span import (
    _memchr,
    _memmem,
    _memrchr,
    _memrmem,
)

from std.testing import assert_equal, assert_false, assert_true
from std.testing import TestSuite

# ===----------------------------------------------------------------------=== #
# _memchr
# ===----------------------------------------------------------------------=== #


def test_memchr_found() raises:
    var s = String("abcdef")
    var span = s.as_bytes()
    var base = span.unsafe_ptr()

    var r1 = _memchr(span, Byte(ord("a")))
    assert_equal(Int(r1[]) - Int(base), 0)

    var r2 = _memchr(span, Byte(ord("c")))
    assert_equal(Int(r2[]) - Int(base), 2)

    var r3 = _memchr(span, Byte(ord("f")))
    assert_equal(Int(r3[]) - Int(base), 5)


def test_memchr_not_found() raises:
    var s = String("abcdef")
    assert_false(_memchr(s.as_bytes(), Byte(ord("z"))))


def test_memchr_empty_source() raises:
    var s = String()
    assert_false(_memchr(s.as_bytes(), Byte(ord("a"))))


def test_memchr_first_occurrence() raises:
    var s = String("abcabc")
    var span = s.as_bytes()
    var r = _memchr(span, Byte(ord("a")))
    assert_equal(Int(r[]) - Int(span.unsafe_ptr()), 0)


def test_memchr_vectorized() raises:
    # Long string to exercise the SIMD path in _memchr_impl.
    var s = String("a") * 200 + "z"
    var span = s.as_bytes()
    var base = span.unsafe_ptr()

    var r1 = _memchr(span, Byte(ord("z")))
    assert_equal(Int(r1[]) - Int(base), 200)
    assert_false(_memchr(span, Byte(ord("x"))))

    # Match in the scalar tail after the vectorized region.
    var s2 = String("a") * 200 + "bcz"
    var span2 = s2.as_bytes()
    var r2 = _memchr(span2, Byte(ord("z")))
    assert_equal(Int(r2[]) - Int(span2.unsafe_ptr()), 202)


# ===----------------------------------------------------------------------=== #
# _memmem
# ===----------------------------------------------------------------------=== #


def test_memmem_single_char_needle() raises:
    # needle_len == 1 delegates to _memchr.
    var s = String("abcdef")
    var span = s.as_bytes()
    var needle = String("c")
    var r = _memmem(span, needle.as_bytes())
    assert_equal(Int(r[]) - Int(span.unsafe_ptr()), 2)


def test_memmem_multi_char() raises:
    var s = String("abcdefgh")
    var span = s.as_bytes()
    var base = span.unsafe_ptr()

    var n1 = String("abc")
    var r1 = _memmem(span, n1.as_bytes())
    assert_equal(Int(r1[]) - Int(base), 0)

    var n2 = String("def")
    var r2 = _memmem(span, n2.as_bytes())
    assert_equal(Int(r2[]) - Int(base), 3)

    var n3 = String("fgh")
    var r3 = _memmem(span, n3.as_bytes())
    assert_equal(Int(r3[]) - Int(base), 5)


def test_memmem_not_found() raises:
    var s = String("abcdef")
    var needle = String("xyz")
    assert_false(_memmem(s.as_bytes(), needle.as_bytes()))


def test_memmem_needle_longer_than_haystack() raises:
    var s = String("abc")
    var needle = String("abcdef")
    assert_false(_memmem(s.as_bytes(), needle.as_bytes()))


def test_memmem_first_occurrence() raises:
    var s = String("abcabc")
    var span = s.as_bytes()
    var needle = String("abc")
    var r = _memmem(span, needle.as_bytes())
    assert_equal(Int(r[]) - Int(span.unsafe_ptr()), 0)


def test_memmem_vectorized() raises:
    # Long haystack to exercise _memmem_impl's SIMD path.
    var s = String("a") * 200 + "xyz"
    var span = s.as_bytes()
    var base = span.unsafe_ptr()

    var n1 = String("xyz")
    var r1 = _memmem(span, n1.as_bytes())
    assert_equal(Int(r1[]) - Int(base), 200)

    var n2 = String("xyw")
    assert_false(_memmem(span, n2.as_bytes()))

    # Needle at start of long string.
    var s2 = String("xyz") + String("a") * 200
    var span2 = s2.as_bytes()
    var r2 = _memmem(span2, n1.as_bytes())
    assert_equal(Int(r2[]) - Int(span2.unsafe_ptr()), 0)


# ===----------------------------------------------------------------------=== #
# _memrchr
# ===----------------------------------------------------------------------=== #


def test_memrchr_last_occurrence() raises:
    var s = String("abcabc")
    var span = s.as_bytes()
    var base = span.unsafe_ptr()

    var r1 = _memrchr(span, Byte(ord("a")))
    assert_equal(Int(r1[]) - Int(base), 3)

    var r2 = _memrchr(span, Byte(ord("c")))
    assert_equal(Int(r2[]) - Int(base), 5)


def test_memrchr_single_occurrence() raises:
    var s = String("abcdef")
    var span = s.as_bytes()
    var r = _memrchr(span, Byte(ord("d")))
    assert_equal(Int(r[]) - Int(span.unsafe_ptr()), 3)


def test_memrchr_not_found() raises:
    var s = String("abcdef")
    assert_false(_memrchr(s.as_bytes(), Byte(ord("z"))))


def test_memrchr_empty_source() raises:
    var s = String()
    assert_false(_memrchr(s.as_bytes(), Byte(ord("a"))))


# ===----------------------------------------------------------------------=== #
# _memrmem
# ===----------------------------------------------------------------------=== #


def test_memrmem_empty_needle() raises:
    # Empty needle returns pointer to start of haystack.
    var s = String("abc")
    var span = s.as_bytes()
    var empty = String()
    var r = _memrmem(span, empty.as_bytes())
    assert_true(r.__bool__())
    assert_equal(Int(r[]) - Int(span.unsafe_ptr()), 0)


def test_memrmem_single_char_needle() raises:
    # needle_len == 1 delegates to _memrchr.
    var s = String("abcabc")
    var span = s.as_bytes()
    var needle = String("a")
    var r = _memrmem(span, needle.as_bytes())
    assert_equal(Int(r[]) - Int(span.unsafe_ptr()), 3)


def test_memrmem_multi_char() raises:
    var s = String("abcabc")
    var span = s.as_bytes()

    var n1 = String("abc")
    var r1 = _memrmem(span, n1.as_bytes())
    assert_equal(Int(r1[]) - Int(span.unsafe_ptr()), 3)

    # Needle that spans the whole string.
    var n2 = String("abcabc")
    var r2 = _memrmem(span, n2.as_bytes())
    assert_equal(Int(r2[]) - Int(span.unsafe_ptr()), 0)


def test_memrmem_not_found() raises:
    var s = String("abcdef")
    var needle = String("xyz")
    assert_false(_memrmem(s.as_bytes(), needle.as_bytes()))


def test_memrmem_needle_longer_than_haystack() raises:
    var s = String("abc")
    var needle = String("abcdef")
    assert_false(_memrmem(s.as_bytes(), needle.as_bytes()))


def test_memrmem_empty_haystack() raises:
    var s = String()
    var needle = String("a")
    assert_false(_memrmem(s.as_bytes(), needle.as_bytes()))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
