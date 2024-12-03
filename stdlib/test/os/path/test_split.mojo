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
from os.path import expanduser, split
from pathlib import Path

from builtin._location import __source_location
from testing import assert_equal


def main():
    # Normal case
    head, tail = split(os.path.join("a", "b", "c.txt"))
    assert_equal(head, os.path.join("a", "b"))
    assert_equal(tail, "c.txt")

    # Absolute and empty tail
    head, tail = split(Path.home() / "a" / "b" / "")
    assert_equal(head, expanduser(os.path.join("~", "a", "b")))
    assert_equal(tail, "")

    # Empty head
    head, tail = split("c.txt")
    assert_equal(head, "")
    assert_equal(tail, "c.txt")

    # Empty head and tail
    head, tail = split("")
    assert_equal(head, "")
    assert_equal(tail, "")

    # Single separator
    head, tail = split(os.sep)
    assert_equal(head, os.sep)
    assert_equal(tail, "")

    # Two chars, absolute on Linux.
    head, tail = split(os.path.join(os.sep, "a"))
    assert_equal(head, os.sep)
    assert_equal(tail, "a")

    # Two chars relative, empty tail
    head, tail = split(os.path.join("a", ""))
    assert_equal(head, "a")
    assert_equal(tail, "")

    # Test with Path objects
    head, tail = split(Path("a") / "b" / "c.txt")
    assert_equal(head, os.path.join("a", "b"))
    assert_equal(tail, "c.txt")

    # Test with __source_location()
    source_location = __source_location().file_name
    head, tail = split(source_location)
    assert_equal(head + os.sep + tail, source_location)
