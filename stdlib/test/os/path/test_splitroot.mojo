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
from os.path import splitroot

from testing import assert_equal


def main():
    # Root directories
    var s = splitroot("/")
    assert_equal("/", s[1])
    assert_equal("", s[2])
    s = splitroot("//")
    assert_equal("//", s[1])
    assert_equal("", s[2])
    s = splitroot("///")
    assert_equal("/", s[1])
    assert_equal("//", s[2])

    # Empty strings
    s = splitroot("")
    assert_equal("", s[1])
    assert_equal("", s[2])

    # Absolute paths
    s = splitroot("/file")
    assert_equal("/", s[1])
    assert_equal("file", s[2])
    s = splitroot("//file")
    assert_equal("//", s[1])
    assert_equal("file", s[2])
    s = splitroot("///file")
    assert_equal("/", s[1])
    assert_equal("//file", s[2])
    s = splitroot("/dir/file")
    assert_equal("/", s[1])
    assert_equal("dir/file", s[2])

    # Relative paths
    s = splitroot("file")
    assert_equal("", s[1])
    assert_equal("file", s[2])
    s = splitroot("file/dir")
    assert_equal("", s[1])
    assert_equal("file/dir", s[2])
    s = splitroot(".")
    assert_equal("", s[1])
    assert_equal(".", s[2])
    s = splitroot("..")
    assert_equal("", s[1])
    assert_equal("..", s[2])
    s = splitroot("entire/.//.tail/..//captured////")
    assert_equal("", s[1])
    assert_equal("entire/.//.tail/..//captured////", s[2])
