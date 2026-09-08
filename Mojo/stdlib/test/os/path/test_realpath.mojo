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

from std.os.path import realpath
from std.pathlib import Path, cwd
from std.sys import CompilationTarget

from std.python import Python
from std.testing import TestSuite, assert_equal, assert_raises, assert_true


def test_realpath() raises:
    print("test resolution of: .. . ./")
    var cwd_realpath = realpath("../.././.")
    var os_cwd = String(realpath(".././.././."))
    assert_equal(cwd_realpath, realpath(os_cwd))

    print("test current directory resolution")
    var current_dir_1 = cwd()
    var current_dir_2 = realpath("./")
    assert_equal(current_dir_1, current_dir_2)

    print("test absolute path starts from root")
    var abs_path = realpath(".")
    assert_true(abs_path.startswith("/"))

    print("test multiple consecutive separators")
    var multi_sep = realpath("..//..//./")
    var normal_sep = realpath("../../.")
    assert_equal(multi_sep, normal_sep)

    print("test trailing separators are normalized")
    var with_trailing = realpath("../.")
    var without_trailing = realpath("..")
    assert_equal(with_trailing, without_trailing)

    print("test empty relative path components")
    var empty_components = realpath("././../././.")
    var simple_parent = realpath("..")
    assert_equal(empty_components, simple_parent)

    print("error handling for non-existent paths")
    with assert_raises(
        contains="realpath failed to resolve: No such file or directory"
    ):
        _ = realpath("does-not-exist")

    print("test root directory behavior")
    var root_path = realpath("/")
    assert_equal(root_path, "/")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
