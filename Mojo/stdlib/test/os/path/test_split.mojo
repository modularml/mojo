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

import std.os
from std.os.path import expanduser, split
from std.pathlib import Path

from std.reflection import source_location
from std.testing import TestSuite, assert_equal


def test_split() raises:
    # Normal case
    var head, tail = split(std.os.path.join("a", "b", "c.txt"))
    assert_equal(head, std.os.path.join("a", "b"))
    assert_equal(tail, "c.txt")

    # Absolute and empty tail
    head, tail = split(Path.home() / "a" / "b" / "")
    assert_equal(head, expanduser(std.os.path.join("~", "a", "b")))
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
    head, tail = split(std.os.sep)
    assert_equal(head, String(std.os.sep))
    assert_equal(tail, "")

    # Two chars, absolute on Linux.
    head, tail = split(std.os.path.join(String(std.os.sep), "a"))
    assert_equal(head, String(std.os.sep))
    assert_equal(tail, "a")

    # Two chars relative, empty tail
    head, tail = split(std.os.path.join("a", ""))
    assert_equal(head, "a")
    assert_equal(tail, "")

    # Test with Path objects
    head, tail = split(Path("a") / "b" / "c.txt")
    assert_equal(head, std.os.path.join("a", "b"))
    assert_equal(tail, "c.txt")

    # Test with source_location()
    var source_location = String(source_location().file_name())
    head, tail = split(source_location)
    assert_equal(head + std.os.sep + tail, source_location)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
