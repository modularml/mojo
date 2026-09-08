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

from std.os.path import join
from std.pathlib import Path

from std.testing import TestSuite, assert_equal


def test_join() raises:
    # TODO uncomment lines using Path when unpacking is supported
    assert_equal("path/to/file", join("path", "to", "file"))
    # assert_equal("path/to/file", join(Path("path"), Path("to"), Path("file")))
    assert_equal("path/to/file", join("path", "to/file"))
    # assert_equal("path/to/file", join(Path("path"), Path("to/file")))
    assert_equal("path/to/file", join("path/to", "file"))
    # assert_equal("path/to/file", join(Path("path/to"), Path("file")))
    assert_equal("path/to/file", join("path/", "to/", "file"))

    assert_equal("/a/b", join("/", "a", "b"))
    assert_equal("a/b/c", join("a", "b/", "c"))
    assert_equal("a/b/c", join("a", "b", "", "c"))

    assert_equal("path/", join("path", ""))
    # assert_equal("path/", join(Path("path"), Path("")))
    assert_equal("path", join("path"))
    # assert_equal("path", join(Path("path")))
    assert_equal("", join(""))
    assert_equal("path", join("", "path"))

    assert_equal("/path/to/file", join("ignored", "/path/to", "file"))
    # assert_equal("/path/to/file", join(Path("ignored"), Path("/path/to/file")))
    assert_equal(
        "/absolute/path",
        join("ignored", "/ignored/absolute/path", "/absolute", "path"),
    )

    # Separator insertion after absolute path reset.
    assert_equal("/abs/next", join("first/", "/abs", "next"))
    # assert_equal(
    #     "/path/to/file",
    #     join(
    #         Path("ignored"),
    #         Path("/path/to/file/but/ignored/again"),
    #         Path("/path/to/file"),
    #     ),
    # )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
