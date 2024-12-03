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
# RUN: %mojo %s

import os
import pwd

from testing import assert_equal, assert_raises, assert_true


def test_pwuid():
    # Test current process user works
    passwd = pwd.getpwuid(os.getuid())
    assert_true(len(passwd.pw_dir) > 2)
    assert_true(passwd.pw_uid >= 0)
    assert_true(len(passwd.pw_name) > 0)
    # Test root user works
    passwd = pwd.getpwuid(0)
    assert_true(len(passwd.pw_dir) > 2)
    assert_equal(passwd.pw_uid, 0)
    assert_equal(passwd.pw_name, "root")

    # Ensure incorrect ID fails
    with assert_raises():
        _ = pwd.getpwuid(456789431974)


def test_pwnam():
    # Test root user works
    passwd = pwd.getpwnam("root")
    assert_true(len(passwd.pw_dir) > 2)
    assert_equal(passwd.pw_uid, 0)
    assert_equal(passwd.pw_name, "root")

    # Ensure incorrect name fails
    with assert_raises():
        _ = pwd.getpwnam("zxcvarahoijewklvnab")


def main():
    test_pwuid()
    test_pwnam()
