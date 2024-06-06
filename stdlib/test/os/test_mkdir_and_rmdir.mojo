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

from os import mkdir, remove, rmdir
from os.path import exists
from pathlib import Path

from testing import assert_false, assert_raises, assert_true


fn create_dir_and_test_delete_string[
    func_create: fn (String, Int) raises -> None,
    func_delete: fn (String) raises -> None,
](dir_name: String) raises:
    # verify that the test dir does not exist before starting the test
    assert_false(
        exists(dir_name),
        "Unexpected dir " + dir_name + " it should not exist",
    )

    func_create(dir_name, 0o777)
    assert_true(exists(dir_name))

    func_delete(dir_name)
    # trying to delete non existing dir
    with assert_raises(contains="Can not remove directory: "):
        func_delete(dir_name)


fn create_dir_and_test_delete_path[
    func_create: fn[pathlike: PathLike] (pathlike, Int) raises -> None,
    func_delete: fn[pathlike: PathLike] (pathlike) raises -> None,
](dir_path: Path) raises:
    # verify that the test dir does not exist before starting the test
    assert_false(
        exists(dir_path),
        "Unexpected dir " + dir_path.__fspath__() + " it should not exist",
    )

    func_create(dir_path, 0o777)
    assert_true(exists(dir_path))

    func_delete(dir_path)
    # trying to delete non existing dir
    with assert_raises(contains="Can not remove directory: "):
        func_delete(dir_path)


fn test_mkdir_and_rmdir() raises:
    var cwd_path = Path()
    var my_dir_path = cwd_path / "my_dir"
    var my_dir_name = str(my_dir_path)

    create_dir_and_test_delete_path[mkdir, rmdir](my_dir_path)
    create_dir_and_test_delete_string[mkdir, rmdir](my_dir_name)

    # test relative path
    create_dir_and_test_delete_string[mkdir, rmdir]("my_relative_dir")
    create_dir_and_test_delete_path[mkdir, rmdir](Path("my_relative_dir"))


fn test_mkdir_mode() raises:
    var cwd_path = Path()
    var my_dir_path = cwd_path / "my_dir"
    var file_name = my_dir_path / "file.txt"

    assert_false(
        exists(my_dir_path),
        "Unexpected dir " + my_dir_path.__fspath__() + " it should not exist",
    )

    # creating dir without writing permission
    mkdir(my_dir_path, 0o111)

    # TODO: This test is failing on Graviton internally in CI, revisit.
    # with assert_raises(contains="Permission denied"):
    #     var file = open(file_name, "w")
    #     file.close()
    #     if exists(file_name):
    #         remove(file_name)

    if exists(my_dir_path):
        rmdir(my_dir_path)


fn test_rmdir_not_empty() raises:
    var cwd_path = Path()
    var my_dir_path = cwd_path / "my_dir"
    var file_name = my_dir_path / "file.txt"

    assert_false(
        exists(my_dir_path),
        "Unexpected dir " + my_dir_path.__fspath__() + " it should not exist",
    )

    mkdir(my_dir_path)
    with open(file_name, "w"):
        pass

    with assert_raises(contains="Can not remove directory: "):
        rmdir(my_dir_path)

    remove(file_name)
    rmdir(my_dir_path)
    assert_false(exists(my_dir_path), "Failed to remove dir")


def main():
    test_mkdir_and_rmdir()
    test_mkdir_mode()
    test_rmdir_not_empty()
