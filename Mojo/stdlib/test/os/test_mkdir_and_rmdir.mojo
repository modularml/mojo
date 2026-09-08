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
from std.os.path import exists
from std.pathlib import Path

from std.testing import TestSuite, assert_false, assert_raises, assert_true


def create_and_delete(path: String) raises:
    # verify that the test dir does not exist before starting the test
    assert_false(
        exists(path),
        "Unexpected dir " + path + " it should not exist",
    )

    std.os.mkdir(path, 0o777)
    assert_true(exists(path))

    std.os.rmdir(path)
    # trying to delete non existing dir
    with assert_raises(contains="Can not remove directory: "):
        std.os.rmdir(path)


def _test_mkdir_and_rmdir_str(path: String) raises:
    try:
        std.os.rmdir(path)
    except:
        pass
    # verify that the test dir does not exist before starting the test
    assert_false(
        exists(path),
        String("Unexpected dir ", path, " it should not exist"),
    )

    std.os.mkdir(path, 0o777)
    assert_true(exists(path))

    std.os.rmdir(path)
    # trying to delete non existing dir
    with assert_raises(contains="Can not remove directory: "):
        std.os.rmdir(path)


def _test_mkdir_and_rmdir_path(path: Path) raises:
    try:
        std.os.rmdir(path)
    except:
        pass
    # verify that the test dir does not exist before starting the test
    assert_false(
        exists(path), String("Unexpected dir ", path, " it should not exist")
    )

    std.os.mkdir(path, 0o777)
    assert_true(exists(path))

    std.os.rmdir(path)
    # trying to delete non existing dir
    with assert_raises(contains="Can not remove directory: "):
        std.os.rmdir(path)


def _test_makedirs_and_removedirs(path: Path) raises:
    try:
        std.os.removedirs(path)
    except:
        pass
    # verify that the test dir does not exist before starting the test
    assert_false(
        exists(path), String("Unexpected dir ", path, " it should not exist")
    )
    std.os.makedirs(path, exist_ok=True)
    assert_true(exists(path))
    with assert_raises():
        std.os.makedirs(path)
    # Make sure this doesn't throw error
    std.os.makedirs(path, exist_ok=True)
    std.os.removedirs(path)


def test_mkdir_mode() raises:
    var my_dir_path = Path("my_dir")

    assert_false(
        exists(my_dir_path),
        "Unexpected dir " + my_dir_path.__fspath__() + " it should not exist",
    )

    # creating dir without writing permission
    std.os.mkdir(my_dir_path, 0o111)

    # TODO: This test is failing on Graviton internally in CI, revisit.
    # with assert_raises(contains="Permission denied"):
    #     var file = open(file_name, "w")
    #     file.close()
    #     if exists(file_name):
    #         remove(file_name)

    if exists(my_dir_path):
        std.os.rmdir(my_dir_path)


def test_rmdir_not_empty() raises:
    var my_dir_path = Path("my_dir")
    var file_name = my_dir_path / "file.txt"

    assert_false(
        exists(my_dir_path),
        "Unexpected dir " + my_dir_path.__fspath__() + " it should not exist",
    )

    std.os.mkdir(my_dir_path)
    with open(file_name, "w"):
        pass

    with assert_raises(contains="Can not remove directory: "):
        std.os.rmdir(my_dir_path)

    std.os.remove(file_name)
    std.os.rmdir(my_dir_path)
    assert_false(exists(my_dir_path), "Failed to remove dir")


def test_all_mkdir_and_rmdir() raises:
    _test_mkdir_and_rmdir_str("my_dir")
    _test_mkdir_and_rmdir_path(Path("my_dir"))
    if std.os.getenv("HOME") or std.os.getenv("USERPROFILE"):
        _test_mkdir_and_rmdir_path(Path("~/my_dir").expanduser())


def test_all_makedirs_and_removedirs() raises:
    _test_makedirs_and_removedirs(std.os.path.join("dir1", "dir2", "dir3"))
    _test_makedirs_and_removedirs(Path("dir1") / "dir2" / "dir3")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
