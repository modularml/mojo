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
from os.path import normpath

from testing import assert_equal


def main():
    # Root directories
    assert_equal("/", normpath("/"))
    assert_equal("//", normpath("//"))
    assert_equal("/", normpath("///"))
    assert_equal("/dir", normpath("/dir"))
    assert_equal("//dir", normpath("//dir"))
    assert_equal("/dir", normpath("///dir"))

    # Empty strings
    assert_equal(".", normpath(""))

    # Dots
    assert_equal(".", normpath("."))
    assert_equal("..", normpath(".."))
    assert_equal(".....", normpath("....."))
    assert_equal("../../..", normpath("../../.."))
    assert_equal("../../..", normpath("../..//../"))
    assert_equal("..", normpath("..../..//../"))

    # Absolute paths
    assert_equal("/file", normpath("/file"))
    assert_equal("/dir/file", normpath("/dir/file"))
    assert_equal("/dir/subdir/file", normpath("/dir/subdir/file"))

    # Relative paths
    assert_equal("dir/file", normpath("dir/file"))
    assert_equal("dir/subdir/file", normpath("dir/subdir/file"))

    # Trailing slashes
    assert_equal("/path/to", normpath("/path/to/"))
    assert_equal("/path/to/dir", normpath("/path/to/dir/"))

    # Multiple slashes
    assert_equal("/path/to/file", normpath("/path/to//file"))
    assert_equal("/path/to", normpath("/path//to"))

    # Paths with spaces
    assert_equal("/path to/file", normpath("/path to/file"))
    assert_equal("/path to/dir/file", normpath("/path to/dir/file"))

    # Paths with special characters
    assert_equal("/path-to/file", normpath("/path-to/file"))
    assert_equal("/path_to/dir/file", normpath("/path_to/dir/file"))

    # Paths with dots
    assert_equal("/path/to/file", normpath("/path/./to/file"))
    assert_equal("/to/file", normpath("/path/../to/file"))
    assert_equal("/file", normpath("/path/../file"))
    assert_equal("/path/file", normpath("/path/to/../file"))
    assert_equal("file", normpath("path/../to/../file"))

    # Unix hidden files
    assert_equal("/path/to/.hiddenfile", normpath("/path/to/.hiddenfile"))
    assert_equal(
        "/path/to/.dir/.hiddenfile", normpath("/path/to/.dir/.hiddenfile")
    )
