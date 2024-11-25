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


def test_absolute_path():
    drive, root, tail = splitroot("/usr/lib/file.txt")
    assert_equal(drive, "")
    assert_equal(root, "/")
    assert_equal(tail, "usr/lib/file.txt")

    drive, root, tail = splitroot("//usr/lib/file.txt")
    assert_equal(drive, "")
    assert_equal(root, "//")
    assert_equal(tail, "usr/lib/file.txt")

    drive, root, tail = splitroot("///usr/lib/file.txt")
    assert_equal(drive, "")
    assert_equal(root, "/")
    assert_equal(tail, "//usr/lib/file.txt")


def test_relative_path():
    drive, root, tail = splitroot("usr/lib/file.txt")
    assert_equal(drive, "")
    assert_equal(root, "")
    assert_equal(tail, "usr/lib/file.txt")

    drive, root, tail = splitroot(".")
    assert_equal(drive, "")
    assert_equal(root, "")
    assert_equal(tail, ".")

    drive, root, tail = splitroot("..")
    assert_equal(drive, "")
    assert_equal(root, "")
    assert_equal(tail, "..")

    drive, root, tail = splitroot("entire/.//.tail/..//captured////")
    assert_equal(drive, "")
    assert_equal(root, "")
    assert_equal(tail, "entire/.//.tail/..//captured////")


def test_root_directory():
    drive, root, tail = splitroot("/")
    assert_equal(drive, "")
    assert_equal(root, "/")
    assert_equal(tail, "")

    drive, root, tail = splitroot("//")
    assert_equal(drive, "")
    assert_equal(root, "//")
    assert_equal(tail, "")

    drive, root, tail = splitroot("///")
    assert_equal(drive, "")
    assert_equal(root, "/")
    assert_equal(tail, "//")


def test_empty_path():
    drive, root, tail = splitroot("")
    assert_equal(drive, "")
    assert_equal(root, "")
    assert_equal(tail, "")


def main():
    test_absolute_path()
    test_relative_path()
    test_root_directory()
    test_empty_path()
