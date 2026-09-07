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

import std.os
import std.pwd

from std.testing import assert_equal, assert_raises, assert_true, TestSuite


def test_pwuid() raises:
    # Test current process user works
    var passwd = std.pwd.getpwuid(std.os.getuid())
    assert_true(passwd.pw_dir.byte_length() > 2)
    assert_true(passwd.pw_uid >= 0)
    assert_true(passwd.pw_name.byte_length() > 0)
    # Test root user works
    passwd = std.pwd.getpwuid(0)
    assert_true(passwd.pw_dir.byte_length() > 2)
    assert_equal(passwd.pw_uid, 0)
    assert_equal(passwd.pw_name, "root")

    # Ensure incorrect ID fails
    with assert_raises():
        _ = std.pwd.getpwuid(456789431974)


def test_pwnam() raises:
    # Test root user works
    var passwd = std.pwd.getpwnam("root")
    assert_true(passwd.pw_dir.byte_length() > 2)
    assert_equal(passwd.pw_uid, 0)
    assert_equal(passwd.pw_name, "root")

    # Ensure incorrect name fails
    with assert_raises():
        _ = std.pwd.getpwnam("zxcvarahoijewklvnab")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
