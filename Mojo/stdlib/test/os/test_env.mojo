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

from std.os import getenv, setenv, unsetenv

from std.testing import TestSuite, assert_equal


def test_getenv() raises:
    assert_equal(getenv("TEST_MYVAR"), "MyValue")

    assert_equal(getenv("TEST_MYVAR", "DefaultValue"), "MyValue")

    assert_equal(getenv("NON_EXISTENT_VAR", "DefaultValue"), "DefaultValue")


def test_setenv() raises:
    assert_equal(setenv("NEW_VAR", "FOO", True), True)
    assert_equal(getenv("NEW_VAR"), "FOO")

    assert_equal(setenv("NEW_VAR", "BAR", False), True)
    assert_equal(getenv("NEW_VAR"), "FOO")

    assert_equal(setenv("NEW_VAR", "BAR", True), True)
    assert_equal(getenv("NEW_VAR", "BAR"), "BAR")

    assert_equal(setenv("=", "INVALID", True), False)


def test_unsetenv() raises:
    assert_equal(setenv("NEW_VAR", "FOO", True), True)
    assert_equal(getenv("NEW_VAR"), "FOO")
    assert_equal(unsetenv("NEW_VAR"), True)
    assert_equal(getenv("NEW_VAR"), "")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
