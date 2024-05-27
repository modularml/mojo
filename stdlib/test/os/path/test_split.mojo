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
from os.path import split
from pathlib import Path

from testing import assert_equal


fn main() raises:
    var head: String
    var tail: String
    head, tail = split("/path/to/file")
    assert_equal(head, "/path/to")
    assert_equal(tail, "file")
    head, tail = split(Path("/path/to/file"))
    assert_equal(head, "/path/to")
    assert_equal(tail, "file")

    head, tail = split("/path/to/dir/")
    assert_equal(head, "/path/to/dir")
    assert_equal(tail, "")
    head, tail = split(Path("/path/to/dir/"))
    assert_equal(head, "/path/to/dir")
    assert_equal(tail, "")

    head, tail = split("/path")
    assert_equal(head, "/")
    assert_equal(tail, "path")
    head, tail = split(Path("/path"))
    assert_equal(head, "/")
    assert_equal(tail, "path")

    head, tail = split("/")
    assert_equal(head, "/")
    assert_equal(tail, "")
    head, tail = split(Path("/"))
    assert_equal(head, "/")
    assert_equal(tail, "")

    head, tail = split("")
    assert_equal(head, "")
    assert_equal(tail, "")
    head, tail = split(Path(""))
    assert_equal(head, "")
    assert_equal(tail, "")

    head, tail = split("/path/to///file")
    assert_equal(head, "/path/to")
    assert_equal(tail, "file")
    head, tail = split(Path("/path/to///file"))
    assert_equal(head, "/path/to")
    assert_equal(tail, "file")

    head, tail = split("/path///")
    assert_equal(head, "/path")
    assert_equal(tail, "")
    head, tail = split(Path("/path///"))
    assert_equal(head, "/path")
    assert_equal(tail, "")

    head, tail = split("file")
    assert_equal(head, "")
    assert_equal(tail, "file")
    head, tail = split(Path("file"))
    assert_equal(head, "")
    assert_equal(tail, "file")

    head, tail = split("///")
    assert_equal(head, "///")
    assert_equal(tail, "")
    head, tail = split(Path("///"))
    assert_equal(head, "///")
    assert_equal(tail, "")
