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
from os.env import getenv, setenv
from os.path import expanduser, join
from sys.info import os_is_windows

from testing import assert_equal, assert_raises, assert_true


fn get_user_path() -> String:
    @parameter
    if os_is_windows():
        return join("C:", "Users", "user")
    return "/home/user"


fn get_current_home() -> String:
    @parameter
    if os_is_windows():
        return getenv("USERPROFILE")
    return getenv("HOME")


def set_home(path: String):
    @parameter
    if os_is_windows():
        _ = os.env.setenv("USERPROFILE", path)
    else:
        _ = os.env.setenv("HOME", path)


fn main() raises:
    alias user_path = get_user_path()
    var original_home = get_current_home()
    set_home(user_path)

    # Single `~`
    assert_equal(user_path, expanduser("~"))

    # Path with home directory
    assert_equal(join(user_path, "folder"), expanduser("~/folder"))

    # Path with trailing slash
    assert_equal(join(user_path, "folder/"), expanduser("~/folder/"))

    # Path without user home directory
    assert_equal("/usr/bin", expanduser("/usr/bin"))

    # Relative path
    assert_equal("../folder", expanduser("../folder"))

    # Empty string
    assert_equal("", expanduser(""))

    # Path with multiple tildes
    assert_equal(join(user_path, "~folder"), expanduser("~/~folder"))

    # Tests with empty "HOME" and "USERPROFILE"
    set_home("")
    if os_is_windows():
        # Don't expand on windows if home isn't set
        assert_equal("~/folder", expanduser("~/folder"))
    else:
        # Test fallback to `/etc/passwd` works on linux
        alias folder = "~/folder"
        assert_true(len(expanduser(folder)) > len(folder))

        # Test expanding user name on Unix
        assert_true(expanduser("~root") != "~root")

        # Test path is returned unchanged on missing user
        var missing_user = "~asdfasdzvxewr/user"
        assert_equal(expanduser(missing_user), missing_user)

    set_home(original_home)
