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

from std.os.path.path import _split_extension, split_extension

from std.testing import TestSuite, assert_equal


def _split_extension_test(
    path: String, expected_head: String, expected_extension: String
) raises:
    var head, extension = split_extension(path)
    assert_equal(head, expected_head)
    assert_equal(extension, expected_extension)


def test_absolute_file_path() raises:
    _split_extension_test("/usr/lib/file.txt", "/usr/lib/file", ".txt")
    _split_extension_test("//usr/lib/file.txt", "//usr/lib/file", ".txt")
    _split_extension_test("///usr/lib/file.txt", "///usr/lib/file", ".txt")


def test_relative_file_path() raises:
    _split_extension_test("usr/lib/file.txt", "usr/lib/file", ".txt")
    _split_extension_test("./file.txt", "./file", ".txt")
    _split_extension_test(".././.././file.txt", ".././.././file", ".txt")


def test_relative_directories() raises:
    _split_extension_test("", "", "")
    _split_extension_test(".", ".", "")
    _split_extension_test("..", "..", "")
    _split_extension_test("........", "........", "")
    _split_extension_test("usr/lib", "usr/lib", "")


def test_file_names() raises:
    _split_extension_test("foo.bar", "foo", ".bar")
    _split_extension_test("foo.boo.bar", "foo.boo", ".bar")
    _split_extension_test("foo.boo.biff.bar", "foo.boo.biff", ".bar")
    _split_extension_test(".csh.rc", ".csh", ".rc")
    _split_extension_test("nodots", "nodots", "")
    _split_extension_test(".cshrc", ".cshrc", "")
    _split_extension_test("...manydots", "...manydots", "")
    _split_extension_test("...manydots.ext", "...manydots", ".ext")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
