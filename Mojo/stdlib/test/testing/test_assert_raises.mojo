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

from std.testing import assert_equal, assert_raises, assert_true, TestSuite


@fieldwise_init
struct CustomError(Movable, Writable):
    var message: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CustomError: ", self.message)


def test_assert_raises_catches_error() raises:
    with assert_raises():
        raise Error("SomeError")
    # The assert_raises should catch the error and not propagate it.
    # Hence the test will succeed.


def test_assert_raises_catches_matched_error() raises:
    with assert_raises(contains="Some"):
        raise Error("SomeError")

    with assert_raises(contains="Error"):
        raise Error("SomeError")

    with assert_raises(contains="eE"):
        raise Error("SomeError")


def test_assert_raises_no_error() raises:
    try:
        with assert_raises():  # col 27
            pass
        raise Error("This should not be reachable.")
    except e:
        assert_true(String(e).startswith("AssertionError: Didn't raise"))
        assert_true(String(e).endswith(":27"))  # col 27
        assert_true(String(e) != "This should not be reachable.")


def test_assert_raises_no_match() raises:
    try:
        with assert_raises(contains="Some"):
            raise Error("OtherError")
        raise Error("This should not be reachable.")
    except e:
        assert_equal(String(e), "OtherError")


def test_assert_raises_catches_custom_error() raises:
    with assert_raises():
        raise CustomError("something broke")


def test_assert_raises_catches_matched_custom_error() raises:
    with assert_raises(contains="something"):
        raise CustomError("something broke")


def test_assert_raises_no_match_custom_error() raises:
    try:
        with assert_raises(contains="other"):
            raise CustomError("something broke")
        raise Error("This should not be reachable.")
    except e:
        assert_equal(String(e), "CustomError: something broke")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
