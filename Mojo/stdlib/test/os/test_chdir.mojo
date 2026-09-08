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
from std.os import chdir, mkdir, rmdir
from std.io import FileDescriptor, FileHandle
from std.pathlib import cwd
from std.testing import TestSuite, assert_equal, assert_not_equal


def test_chdir() raises:
    try:
        std.os.rmdir("test_chdir")
    except:
        pass
    mkdir("test_chdir")
    assert_not_equal(cwd().name(), "test_chdir")
    chdir("test_chdir")
    assert_equal(cwd().name(), "test_chdir")
    chdir("..")


def test_fchdir() raises:
    try:
        std.os.rmdir("test_fchdir")
    except:
        pass
    mkdir("test_fchdir")
    assert_not_equal(cwd().name(), "test_fchdir")
    var f = FileHandle("test_fchdir", "r")
    FileDescriptor(f).fchdir()
    assert_equal(cwd().name(), "test_fchdir")
    chdir("..")
    _ = f


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
