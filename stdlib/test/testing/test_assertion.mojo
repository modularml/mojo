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

from testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
)

from utils.numerics import inf, nan


@value
struct DummyStruct:
    var value: Int

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    fn __str__(self) -> String:
        return "Dummy"  # Can't be used for equality


def test_assert_equal_is_generic():
    assert_equal(DummyStruct(1), DummyStruct(1))

    with assert_raises():
        assert_equal(DummyStruct(1), DummyStruct(2))


def test_assert_not_equal_is_generic():
    assert_not_equal(DummyStruct(1), DummyStruct(2))

    with assert_raises():
        assert_not_equal(DummyStruct(1), DummyStruct(1))


def test_assert_equal_with_simd():
    assert_equal(SIMD[DType.uint8, 2](1, 1), SIMD[DType.uint8, 2](1, 1))

    with assert_raises():
        assert_equal(SIMD[DType.uint8, 2](1, 1), SIMD[DType.uint8, 2](1, 2))


def test_assert_messages():
    try:
        assert_true(False)
    except e:
        assert_true("test_assertion.mojo:64:20: AssertionError:" in str(e))

    try:
        assert_false(True)
    except e:
        assert_true("test_assertion.mojo:69:21: AssertionError:" in str(e))

    try:
        assert_equal(1, 0)
    except e:
        assert_true("test_assertion.mojo:74:21: AssertionError:" in str(e))

    try:
        assert_not_equal(0, 0)
    except e:
        assert_true("test_assertion.mojo:79:25: AssertionError:" in str(e))


def test_assert_almost_equal():
    alias float_type = DType.float32
    alias _inf = inf[float_type]()
    alias _nan = nan[float_type]()

    @parameter
    def _should_succeed[
        type: DType, size: Int
    ](
        lhs: SIMD[type, size],
        rhs: SIMD[type, size],
        *,
        atol: Scalar[type] = 0,
        rtol: Scalar[type] = 0,
        equal_nan: Bool = False,
    ):
        var msg = "`test_assert_almost_equal` should have succeeded"
        assert_almost_equal(
            lhs, rhs, msg=msg, atol=atol, rtol=rtol, equal_nan=equal_nan
        )

    _should_succeed[DType.bool, 1](True, True)
    _should_succeed(SIMD[DType.int32, 2](0, 1), SIMD[DType.int32, 2](0, 1))
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

    @parameter
    def _should_fail[
        type: DType, size: Int
    ](
        lhs: SIMD[type, size],
        rhs: SIMD[type, size],
        *,
        atol: Scalar[type] = 0,
        rtol: Scalar[type] = 0,
        equal_nan: Bool = False,
    ):
        var msg = "`test_assert_almost_equal` should have failed"
        with assert_raises(contains=msg):
            assert_almost_equal(
                lhs, rhs, msg=msg, atol=atol, rtol=rtol, equal_nan=equal_nan
            )

    _should_fail[DType.bool, 1](True, False)
    _should_fail(
        SIMD[DType.int32, 2](0, 1), SIMD[DType.int32, 2](0, -1), atol=5.0
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


def main():
    test_assert_equal_is_generic()
    test_assert_not_equal_is_generic()
    test_assert_equal_with_simd()
    test_assert_messages()
    test_assert_almost_equal()
