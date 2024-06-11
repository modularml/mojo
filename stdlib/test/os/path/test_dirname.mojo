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
    assert_equal("/", dirname("/"))

    # Empty strings
    assert_equal("", dirname(""))

    # Current directory (matching behavior of python, doesn't resolve `..` etc.)
    assert_equal("", dirname("."))

    # Parent directory
    assert_equal("", dirname(".."))

    # Absolute paths
    assert_equal("/", dirname("/file"))
    assert_equal("/dir", dirname("/dir/file"))
    assert_equal("/dir/subdir", dirname("/dir/subdir/file"))

    # Relative paths
    assert_equal("dir", dirname("dir/file"))
    assert_equal("dir/subdir", dirname("dir/subdir/file"))
    assert_equal("", dirname("file"))

    # Trailing slashes
    assert_equal("/path/to", dirname("/path/to/"))
    assert_equal("/path/to/dir", dirname("/path/to/dir/"))

    # Multiple slashes
    assert_equal("/path/to", dirname("/path/to//file"))
    assert_equal("/path", dirname("/path//to"))

    # Paths with spaces
    assert_equal("/path to", dirname("/path to/file"))
    assert_equal("/path to/dir", dirname("/path to/dir/file"))

    # Paths with special characters
    assert_equal("/path-to", dirname("/path-to/file"))
    assert_equal("/path_to/dir", dirname("/path_to/dir/file"))

    # Paths with dots
    assert_equal("/path/./to", dirname("/path/./to/file"))
    assert_equal("/path/../to", dirname("/path/../to/file"))

    # Paths with double dots
    assert_equal("/path/..", dirname("/path/../file"))
    assert_equal("/path/to/..", dirname("/path/to/../file"))

    # Root and relative mixed
    assert_equal("/dir/.", dirname("/dir/./file"))
    assert_equal("/dir/subdir/..", dirname("/dir/subdir/../file"))

    # Edge cases
    assert_equal("/.", dirname("/./file"))
    assert_equal("/..", dirname("/../file"))

    # Unix hidden files
    assert_equal("/path/to", dirname("/path/to/.hiddenfile"))
    assert_equal("/path/to/dir", dirname("/path/to/dir/.hiddenfile"))
