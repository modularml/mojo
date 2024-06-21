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
from os.path import join
from pathlib import Path

from testing import assert_equal


fn main() raises:
    # TODO uncomment lines using Path when unpacking is supported
    assert_equal("path/to/file", join("path", "to", "file"))
    # assert_equal("path/to/file", join(Path("path"), Path("to"), Path("file")))
    assert_equal("path/to/file", join("path", "to/file"))
    # assert_equal("path/to/file", join(Path("path"), Path("to/file")))
    assert_equal("path/to/file", join("path/to", "file"))
    # assert_equal("path/to/file", join(Path("path/to"), Path("file")))
    assert_equal("path/to/file", join("path/", "to/", "file"))

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
    # assert_equal(
    #     "/path/to/file",
    #     join(
    #         Path("ignored"),
    #         Path("/path/to/file/but/ignored/again"),
    #         Path("/path/to/file"),
    #     ),
    # )
