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

from std.reflection import SourceLocation
from std.python import PythonObject
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from std.utils.numerics import inf, nan


def test_assert_messages() raises:
    var assertion = "test_assertion.mojo:"
    var assertion_error = ": AssertionError:"
    try:
        assert_true(False)
    except e:
        assert_true(assertion in String(e) and assertion_error in String(e))

    try:
        assert_false(True)
    except e:
        assert_true(assertion in String(e) and assertion_error in String(e))

    try:
        assert_equal(1, 0)
    except e:
        assert_true(assertion in String(e) and assertion_error in String(e))

    try:
        assert_not_equal(0, 0)
    except e:
        assert_true(assertion in String(e) and assertion_error in String(e))


@fieldwise_init
struct DummyStruct(Equatable, Writable):
    """Test struct using default reflection-based __eq__."""

    var value: Int

    # Uses default reflection-based __eq__ from Equatable trait


def test_assert_equal_is_generic() raises:
    assert_equal(DummyStruct(1), DummyStruct(1))

    with assert_raises():
        assert_equal(DummyStruct(1), DummyStruct(2))


def test_assert_not_equal_is_generic() raises:
    assert_not_equal(DummyStruct(1), DummyStruct(2))

    with assert_raises():
        assert_not_equal(DummyStruct(1), DummyStruct(1))


def test_assert_equal_with_simd() raises:
    assert_equal(SIMD[.uint8, 2](1, 1), SIMD[.uint8, 2](1, 1))

    with assert_raises():
        assert_equal(SIMD[.uint8, 2](1, 1), SIMD[.uint8, 2](1, 2))


def test_assert_equal_with_list() raises:
    assert_equal(
        List(["This", "is", "Mojo"]),
        List[String](["This", "is", "Mojo"]),
    )

    with assert_raises():
        assert_equal(
            List(["This", "is", "Mojo"]),
            List[String](["This", "is", "mojo"]),
        )


def test_assert_not_equal_with_list() raises:
    assert_not_equal([3, 2, 1], [3, 1, 0])

    with assert_raises():
        assert_not_equal([3, 2, 1], [3, 2, 1])


def test_assert_almost_equal() raises:
    comptime float_type = DType.float32
    comptime _inf = inf[float_type]()
    comptime _nan = nan[float_type]()

    def _should_succeed[
        dtype: DType, size: SIMDLength
    ](
        lhs: SIMD[dtype, size],
        rhs: SIMD[dtype, size],
        *,
        atol: Float64 = 0,
        rtol: Float64 = 0,
        equal_nan: Bool = False,
    ) raises:
        var msg = "`test_assert_almost_equal` should have succeeded"
        assert_almost_equal(
            lhs, rhs, msg=msg, atol=atol, rtol=rtol, equal_nan=equal_nan
        )

    _should_succeed(
        SIMD[float_type, 2](-_inf, _inf), SIMD[float_type, 2](-_inf, _inf)
    )
    _should_succeed(
        SIMD[float_type, 2](-_nan, _nan),
        SIMD[float_type, 2](-_nan, _nan),
        equal_nan=True,
    )
    _should_succeed(
        SIMD[float_type, 2](1.0, -1.1),
        SIMD[float_type, 2](1.1, -1.0),
        atol=0.11,
    )
    _should_succeed(
        SIMD[float_type, 2](1.0, -1.1),
        SIMD[float_type, 2](1.1, -1.0),
        rtol=0.10,
    )

    def _should_fail[
        dtype: DType, size: SIMDLength
    ](
        lhs: SIMD[dtype, size],
        rhs: SIMD[dtype, size],
        *,
        atol: Float64 = 0,
        rtol: Float64 = 0,
        equal_nan: Bool = False,
    ) raises:
        var msg = "`test_assert_almost_equal` should have failed"
        with assert_raises(contains=msg):
            assert_almost_equal(
                lhs, rhs, msg=msg, atol=atol, rtol=rtol, equal_nan=equal_nan
            )

    _should_fail(
        SIMD[float_type, 2](-_inf, 0.0),
        SIMD[float_type, 2](_inf, 0.0),
        rtol=0.1,
    )
    _should_fail(
        SIMD[float_type, 2](_inf, 0.0),
        SIMD[float_type, 2](0.0, 0.0),
        rtol=0.1,
    )
    _should_fail(
        SIMD[float_type, 2](_nan, 0.0),
        SIMD[float_type, 2](_nan, 0.0),
        equal_nan=False,
    )
    _should_fail(
        SIMD[float_type, 2](_nan, 0.0),
        SIMD[float_type, 2](0.0, 0.0),
        equal_nan=False,
    )
    _should_fail(
        SIMD[float_type, 2](_nan, 0.0),
        SIMD[float_type, 2](0.0, 0.0),
        equal_nan=True,
    )
    _should_fail(
        SIMD[float_type, 2](1.0, 0.0),
        SIMD[float_type, 2](1.1, 0.0),
        atol=0.05,
    )
    _should_fail(
        SIMD[float_type, 2](-1.0, 0.0),
        SIMD[float_type, 2](-1.1, 0.0),
        rtol=0.05,
    )


def test_assert_is() raises:
    var a = PythonObject("mojo")
    var b = a
    assert_true(a is b)


def test_assert_is_not() raises:
    var a = PythonObject("mojo")
    var b = PythonObject("mojo")
    assert_true(a is not b)


def test_assert_custom_location() raises:
    var location = SourceLocation(2, 0, "my_file_location.mojo")
    try:
        assert_true(
            False,
            msg="always_false",
            location=location,
        )
    except e:
        assert_true(String(location) in String(e))
        assert_true("always_false" in String(e))


def test_assert_equal_stringslice() raises:
    var str1 = StaticString("This is Mojo")
    var str2 = "This is Mojo"
    var str3 = StaticString("This is mojo")

    def _build(value: StaticString, start: Int, end: Int) -> StaticString:
        return StaticString(
            unsafe_from_utf8=Span[Byte, ImmStaticOrigin](
                unsafe_ptr=value.as_bytes().unsafe_ptr().unsafe_offset(start),
                length=end - start,
            )
        )

    def _build(
        imm value: String, start: Int, end: Int
    ) -> StringSlice[origin_of(value)._get_owned_interior["bytes"]]:
        return StringSlice(
            unsafe_from_utf8=Span(
                unsafe_ptr=value.as_bytes().unsafe_ptr().unsafe_offset(start),
                length=end - start,
            )
        )

    var l1: List = [_build(str1, 0, 4), _build(str1, 5, 7), _build(str1, 8, 12)]
    var l2: List = [_build(str2, 0, 4), _build(str2, 5, 7), _build(str2, 8, 12)]
    var l3: List = [_build(str3, 0, 4), _build(str3, 5, 7), _build(str3, 8, 12)]
    assert_equal(l1, l1)
    assert_equal(l2, l2)
    assert_equal(l1, l2)

    with assert_raises():
        assert_equal(l1, l3)

    with assert_raises():
        assert_equal(l2, l3)


@fieldwise_init
struct SomeWritable(Equatable, Writable):
    var value: Int


def test_assert_equal_with_writable() raises:
    assert_equal(SomeWritable(1), SomeWritable(1))
    with assert_raises():
        assert_equal(SomeWritable(1), SomeWritable(2))


def test_assert_not_equal_with_writable() raises:
    assert_not_equal(SomeWritable(1), SomeWritable(2))
    with assert_raises():
        assert_not_equal(SomeWritable(1), SomeWritable(1))


def test_assert_equal_with_unicode() raises:
    # Verify assert_equal works correctly with multi-byte Unicode codepoints.

    # Emoji (4 bytes each in UTF-8)
    assert_equal("Hello 🌍", "Hello 🌍")
    with assert_raises():
        assert_equal("Hello 🌍", "Hello 🌎")

    # Chinese characters (3 bytes each in UTF-8)
    assert_equal("你好世界", "你好世界")
    with assert_raises():
        assert_equal("你好世界", "你好地球")

    # Different length Unicode strings
    with assert_raises():
        assert_equal("🎉🎊", "🎉🎊🎁")

    # Mixed ASCII and Unicode
    with assert_raises():
        assert_equal("abc中文def", "abc英文def")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
