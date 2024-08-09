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


from os.path import dirname

from testing import assert_equal


fn main() raises:
    # Root directories
    assert_equal(dirname("/"), "/")

    # Empty strings
    assert_equal(dirname(""), "")

    # Current directory (matching behavior of python, doesn't resolve `..` etc.)
    assert_equal(dirname("."), "")

    # Parent directory
    assert_equal(dirname(".."), "")

    # Absolute paths
    assert_equal(dirname("/file"), "/")
    assert_equal(dirname("/dir/file"), "/dir")
    assert_equal(dirname("/dir/subdir/file"), "/dir/subdir")

    # Relative paths
    assert_equal(dirname("dir/file"), "dir")
    assert_equal(dirname("dir/subdir/file"), "dir/subdir")
    assert_equal(dirname("file"), "")

    # Trailing slashes
    assert_equal(dirname("/path/to/"), "/path/to")
    assert_equal(dirname("/path/to/dir/"), "/path/to/dir")

    # Multiple slashes
    assert_equal(dirname("/path/to//file"), "/path/to")
    assert_equal(dirname("/path//to"), "/path")

    # Paths with spaces
    assert_equal(dirname("/path to/file"), "/path to")
    assert_equal(dirname("/path to/dir/file"), "/path to/dir")

    # Paths with special characters
    assert_equal(dirname("/path-to/file"), "/path-to")
    assert_equal(dirname("/path_to/dir/file"), "/path_to/dir")

    # Paths with dots
    assert_equal(dirname("/path/./to/file"), "/path/./to")
    assert_equal(dirname("/path/../to/file"), "/path/../to")

    # Paths with double dots
    assert_equal(dirname("/path/../file"), "/path/..")
    assert_equal(dirname("/path/to/../file"), "/path/to/..")

    # Root and relative mixed
    assert_equal(dirname("/dir/./file"), "/dir/.")
    assert_equal(dirname("/dir/subdir/../file"), "/dir/subdir/..")

    # Edge cases
    assert_equal(dirname("/./file"), "/.")
    assert_equal(dirname("/../file"), "/..")

    # Unix hidden files
    assert_equal(dirname("/path/to/.hiddenfile"), "/path/to")
    assert_equal(dirname("/path/to/dir/.hiddenfile"), "/path/to/dir")
