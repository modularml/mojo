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
# REQUIRES: linux || darwin
# RUN: TEST_MYVAR=MyValue %mojo %s

from os import getenv, setenv

from testing import assert_equal


def test_getenv():
    assert_equal(getenv("TEST_MYVAR"), "MyValue")

    assert_equal(getenv("TEST_MYVAR", "DefaultValue"), "MyValue")

    assert_equal(getenv("NON_EXISTENT_VAR", "DefaultValue"), "DefaultValue")


# CHECK-OK-LABEL: test_setenv
def test_setenv():
    assert_equal(setenv("NEW_VAR", "FOO", True), True)
    assert_equal(getenv("NEW_VAR"), "FOO")

    assert_equal(setenv("NEW_VAR", "BAR", False), True)
    assert_equal(getenv("NEW_VAR"), "FOO")

    assert_equal(setenv("NEW_VAR", "BAR", True), True)
    assert_equal(getenv("NEW_VAR", "BAR"), "BAR")

    assert_equal(setenv("=", "INVALID", True), False)


def main():
    test_getenv()
    test_setenv()
