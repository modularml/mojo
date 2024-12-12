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

from os.path import basename
from pathlib import Path

from builtin._location import __source_location
from testing import assert_equal


def main():
    # Root directories
    assert_equal("", basename("/"))

    # Empty strings
    assert_equal("", basename(""))

    # Current directory (matching behavior of python, doesn't resolve `..` etc.)
    assert_equal(".", basename("."))

    # Parent directory
    assert_equal("..", basename(".."))

    # Absolute paths
    assert_equal("file", basename("/file"))
    assert_equal("file.txt", basename("/file.txt"))
    assert_equal("file", basename("/dir/file"))
    assert_equal("file", basename("/dir/subdir/file"))

    # Relative paths
    assert_equal("file", basename("dir/file"))
    assert_equal("file", basename("dir/subdir/file"))
    assert_equal("file", basename("file"))

    # Trailing slashes
    assert_equal("", basename("/path/to/"))
    assert_equal("", basename("/path/to/dir/"))

    # Multiple slashes
    assert_equal("file", basename("/path/to//file"))
    assert_equal("to", basename("/path//to"))

    # Paths with spaces
    assert_equal("file", basename("/path to/file"))
    assert_equal("file", basename("/path to/dir/file"))

    # Paths with special characters
    assert_equal("file", basename("/path-to/file"))
    assert_equal("file", basename("/path_to/dir/file"))

    # Paths with dots
    assert_equal("file", basename("/path/./to/file"))
    assert_equal("file", basename("/path/../to/file"))

    # Paths with double dots
    assert_equal("file", basename("/path/../file"))
    assert_equal("file", basename("/path/to/../file"))

    # Root and relative mixed
    assert_equal("file", basename("/dir/./file"))
    assert_equal("file", basename("/dir/subdir/../file"))

    # Edge cases
    assert_equal("file", basename("/./file"))
    assert_equal("file", basename("/../file"))

    # Unix hidden files
    assert_equal(".hiddenfile", basename("/path/to/.hiddenfile"))
    assert_equal(".hiddenfile", basename("/path/to/dir/.hiddenfile"))

    assert_equal("test_basename.mojo", basename(__source_location().file_name))
    assert_equal(
        "some_file.txt", basename(Path.home() / "dir" / "some_file.txt")
    )
