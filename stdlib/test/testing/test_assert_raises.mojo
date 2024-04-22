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
# RUN: %mojo-no-debug %s

from testing import assert_raises, assert_equal


fn test_assert_raises_catches_error() raises:
    with assert_raises():
        raise "SomeError"
    # The assert_raises should catch the error and not propagate it.
    # Hence the test will succeed.


fn test_assert_raises_catches_matched_error() raises:
    with assert_raises(contains="Some"):
        raise "SomeError"

    with assert_raises(contains="Error"):
        raise "SomeError"

    with assert_raises(contains="eE"):
        raise "SomeError"


fn test_assert_raises_no_error() raises:
    try:
        with assert_raises():
            pass
        raise Error("This should not be reachable.")
    except e:
        assert_equal(str(e), "AssertionError: Didn't raise")


fn test_assert_raises_no_match() raises:
    try:
        with assert_raises(contains="Some"):
            raise "OtherError"
        raise Error("This should not be reachable.")
    except e:
        assert_equal(str(e), "OtherError")


def main():
    test_assert_raises_catches_error()
    test_assert_raises_catches_matched_error()
    test_assert_raises_no_error()
    test_assert_raises_no_match()
