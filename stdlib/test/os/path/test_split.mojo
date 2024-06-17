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

from os.path import split
from testing import assert_equal


def main():
    var tail: String
    var head: String
    tail, head = split("a/path/to/file.txt")
    assert_equal(tail, "a/path/to")
    assert_equal(head, "file.txt")

    tail, head = split("///file.txt")
    assert_equal(tail, "///")
    assert_equal(head, "file.txt")

    tail, head = split("/a/path/to/file.txt")
    assert_equal(tail, "/a/path/to")
    assert_equal(head, "file.txt")

    tail, head = split("a/path/to/")
    assert_equal(tail, "a/path/to")
    assert_equal(head, "")

    tail, head = split("a/path/to/dir")
    assert_equal(tail, "a/path/to")
    assert_equal(head, "dir")

    tail, head = split("")
    assert_equal(tail, "")
    assert_equal(head, "")

    tail, head = split("just_file.txt")
    assert_equal(tail, "")
    assert_equal(head, "just_file.txt")
