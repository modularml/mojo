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
from os.path import exists
from pathlib import Path

from testing import assert_false, assert_raises, assert_true


fn create_and_delete(path: String) raises:
    # verify that the test dir does not exist before starting the test
    assert_false(
        exists(path),
        "Unexpected dir " + path + " it should not exist",
    )

    os.mkdir(path, 0o777)
    assert_true(exists(path))

    os.rmdir(path)
    # trying to delete non existing dir
    with assert_raises(contains="Can not remove directory: "):
        os.rmdir(path)


fn test_mkdir_and_rmdir(path: String) raises:
    try:
        os.rmdir(path)
    except:
        pass
    # verify that the test dir does not exist before starting the test
    assert_false(
        exists(path),
        "Unexpected dir " + str(path) + " it should not exist",
    )

    os.mkdir(path, 0o777)
    assert_true(exists(path))

    os.rmdir(path)
    # trying to delete non existing dir
    with assert_raises(contains="Can not remove directory: "):
        os.rmdir(path)


fn test_mkdir_and_rmdir(path: Path) raises:
    try:
        os.rmdir(path)
    except:
        pass
    # verify that the test dir does not exist before starting the test
    assert_false(
        exists(path),
        "Unexpected dir " + str(path) + " it should not exist",
    )

    os.mkdir(path, 0o777)
    assert_true(exists(path))

    os.rmdir(path)
    # trying to delete non existing dir
    with assert_raises(contains="Can not remove directory: "):
        os.rmdir(path)


fn test_makedirs_and_removedirs(path: Path) raises:
    try:
        os.removedirs(path)
    except:
        pass
    # verify that the test dir does not exist before starting the test
    assert_false(
        exists(path),
        "Unexpected dir " + str(path) + " it should not exist",
    )
    os.makedirs(path, exist_ok=True)
    assert_true(exists(path))
    with assert_raises():
        os.makedirs(path)
    # Make sure this doesn't throw error
    os.makedirs(path, exist_ok=True)
    os.removedirs(path)


fn test_mkdir_mode() raises:
    var my_dir_path = Path("my_dir")

    assert_false(
        exists(my_dir_path),
        "Unexpected dir " + my_dir_path.__fspath__() + " it should not exist",
    )

    # creating dir without writing permission
    os.mkdir(my_dir_path, 0o111)

    # TODO: This test is failing on Graviton internally in CI, revisit.
    # with assert_raises(contains="Permission denied"):
    #     var file = open(file_name, "w")
    #     file.close()
    #     if exists(file_name):
    #         remove(file_name)

    if exists(my_dir_path):
        os.rmdir(my_dir_path)


fn test_rmdir_not_empty() raises:
    var my_dir_path = Path("my_dir")
    var file_name = my_dir_path / "file.txt"

    assert_false(
        exists(my_dir_path),
        "Unexpected dir " + my_dir_path.__fspath__() + " it should not exist",
    )

    os.mkdir(my_dir_path)
    with open(file_name, "w"):
        pass

    with assert_raises(contains="Can not remove directory: "):
        os.rmdir(my_dir_path)

    os.remove(file_name)
    os.rmdir(my_dir_path)
    assert_false(exists(my_dir_path), "Failed to remove dir")


def main():
    test_mkdir_and_rmdir("my_dir")
    test_mkdir_and_rmdir(Path("my_dir"))
    if os.env.getenv("HOME") or os.env.getenv("USERPROFILE"):
        test_mkdir_and_rmdir(Path("~/my_dir").expanduser())

    test_makedirs_and_removedirs(os.path.join("dir1", "dir2", "dir3"))
    test_makedirs_and_removedirs(Path("dir1") / "dir2" / "dir3")

    test_mkdir_mode()
    test_rmdir_not_empty()
