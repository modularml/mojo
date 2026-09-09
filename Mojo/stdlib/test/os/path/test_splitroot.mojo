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

from std.os.path import splitroot

from std.testing import TestSuite, assert_equal


def _splitroot_test(
    path: String,
    expected_drive: String,
    expected_root: String,
    expected_tail: String,
) raises:
    var drive, root, tail = splitroot(path)
    assert_equal(drive, expected_drive)
    assert_equal(root, expected_root)
    assert_equal(tail, expected_tail)


def test_absolute_path() raises:
    _splitroot_test("/usr/lib/file.txt", "", "/", "usr/lib/file.txt")
    _splitroot_test("//usr/lib/file.txt", "", "//", "usr/lib/file.txt")
    _splitroot_test("///usr/lib/file.txt", "", "/", "//usr/lib/file.txt")
    _splitroot_test("/a", "", "/", "a")
    _splitroot_test("/a/b", "", "/", "a/b")
    _splitroot_test("/a/b/", "", "/", "a/b/")


def test_relative_path() raises:
    _splitroot_test("usr/lib/file.txt", "", "", "usr/lib/file.txt")
    _splitroot_test(".", "", "", ".")
    _splitroot_test("..", "", "", "..")
    _splitroot_test(
        "entire/.//.tail/..//captured////",
        "",
        "",
        "entire/.//.tail/..//captured////",
    )
    _splitroot_test("a", "", "", "a")
    _splitroot_test("a/b", "", "", "a/b")
    _splitroot_test("a/b/", "", "", "a/b/")


def test_root_directory() raises:
    _splitroot_test("/", "", "/", "")
    _splitroot_test("//", "", "//", "")
    _splitroot_test("///", "", "/", "//")


def test_empty_path() raises:
    _splitroot_test("", "", "", "")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
