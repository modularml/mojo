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
# REQUIRES: !windows
# RUN: %mojo %s


import os
from os.path import expanduser, join
from os.env import setenv, getenv
from testing import assert_equal, assert_raises
from sys.info import os_is_windows


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

    # Malformed path should raise
    with assert_raises():
        _ = expanduser("~badpath/folder")

    # Path with multiple tildes
    assert_equal(join(user_path, "~folder"), expanduser("~/~folder"))

    # Test that empty HOME returns `~`
    set_home("")
    assert_equal(expanduser("~/test/path"), "~/test/path")
    assert_equal(expanduser("~"), "~")

    set_home(original_home)
