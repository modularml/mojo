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
    assert_equal,
    assert_not_equal,
    assert_raises,
    assert_true,
    assert_false,
    assert_almost_equal,
)


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
        assert_true("test_assertion.mojo:62:20: AssertionError:" in str(e))

    try:
        assert_false(True)
    except e:
        assert_true("test_assertion.mojo:67:21: AssertionError:" in str(e))

    try:
        assert_equal(1, 0)
    except e:
        assert_true("test_assertion.mojo:72:21: AssertionError:" in str(e))

    try:
        assert_not_equal(0, 0)
    except e:
        assert_true("test_assertion.mojo:77:25: AssertionError:" in str(e))


def main():
    test_assert_equal_is_generic()
    test_assert_not_equal_is_generic()
    test_assert_equal_with_simd()
    test_assert_messages()
