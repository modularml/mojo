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
from testing import assert_true, assert_false, assert_equal
from tempfile import gettempdir, mkdtemp


fn test_mkdtemp() raises:
    var dir_name: String

    dir_name = mkdtemp()
    assert_true(exists(dir_name), "Failed to create temporary directory")
    os.rmdir(dir_name)
    assert_false(exists(dir_name), "Failed to delete temporary directory")

    dir_name = mkdtemp(prefix="my_prefix", suffix="my_suffix")
    assert_true(exists(dir_name), "Failed to create temporary directory")
    var name = dir_name.split("/")[-1]
    assert_true(name.startswith("my_prefix"))
    assert_true(name.endswith("my_suffix"))

    os.rmdir(dir_name)
    assert_false(exists(dir_name), "Failed to delete temporary directory")

    dir_name = mkdtemp(dir=Path().__fspath__())
    assert_true(exists(dir_name), "Failed to create temporary directory")
    assert_true(
        exists(Path() / dir_name.split("/")[-1]),
        "Expected directory to be created in cwd",
    )
    os.rmdir(dir_name)
    assert_false(exists(dir_name), "Failed to delete temporary directory")


struct _TempEnvForTest:
    var tmpdir_value: String
    var temp_value: String
    var tmp_value: String
    var clean_up_function: fn () raises -> None

    var tmpdir_back: String
    var temp_back: String
    var tmp_back: String

    fn __init__(
        inout self,
        tmpdir_value: String,
        temp_value: String,
        tmp_value: String,
        clean_up_function: fn () raises -> None,
    ):
        self.tmpdir_value = tmpdir_value
        self.temp_value = temp_value
        self.tmp_value = tmp_value

        self.clean_up_function = clean_up_function

        self.tmpdir_back = os.getenv("TMPDIR")
        self.temp_back = os.getenv("TEMP")
        self.tmp_back = os.getenv("TMP")

    fn __enter__(inout self) raises:
        _ = os.setenv("TMPDIR", self.tmpdir_value, overwrite=True)
        _ = os.setenv("TEMP", self.temp_value, overwrite=True)
        _ = os.setenv("TMP", self.tmp_value, overwrite=True)

    fn __exit__(inout self):
        _ = os.setenv("TMPDIR", self.tmpdir_back, overwrite=True)
        _ = os.setenv("TEMP", self.temp_back, overwrite=True)
        _ = os.setenv("TMP", self.tmp_back, overwrite=True)

    fn __exit__(inout self, error: Error) raises -> Bool:
        self.__exit__()
        self.clean_up_function()
        return False


fn _clean_up_gettempdir_test() raises:
    var dir_without_writing_access = Path() / "dir_without_writing_access"
    if exists(dir_without_writing_access):
        os.rmdir(dir_without_writing_access)
    var dir_with_writing_access = Path() / "dir_with_writing_access"
    if exists(dir_with_writing_access):
        os.rmdir(dir_with_writing_access)


fn _set_up_gettempdir_test(
    dir_with_writing_access: Path, dir_without_writing_access: Path
) raises:
    os.mkdir(dir_with_writing_access, mode=0o700)
    try:
        os.mkdir(dir_without_writing_access, mode=0o100)
    except:
        os.rmdir(dir_with_writing_access)
        raise Error("Failed to setup test")


fn test_gettempdir() raises:
    var non_existing_dir = Path() / "non_existing_dir"
    assert_false(
        exists(non_existing_dir),
        "Unexpected dir" + String(non_existing_dir),
    )
    var dir_without_writing_access = Path() / "dir_without_writing_access"
    var dir_with_writing_access = Path() / "dir_with_writing_access"
    _set_up_gettempdir_test(dir_with_writing_access, dir_without_writing_access)

    var tmpdir_result: Optional[String]
    # test TEMPDIR is used first
    with _TempEnvForTest(
        dir_with_writing_access,
        non_existing_dir,
        non_existing_dir,
        _clean_up_gettempdir_test,
    ):
        tmpdir_result = gettempdir()
        assert_true(tmpdir_result, "Failed to get temporary directory")
        assert_equal(
            tmpdir_result.value()[],
            String(dir_with_writing_access),
            "expected to get:" + String(dir_with_writing_access),
        )

    # test gettempdir falls back to TEMP
    with _TempEnvForTest(
        non_existing_dir,
        dir_with_writing_access,
        non_existing_dir,
        _clean_up_gettempdir_test,
    ):
        tmpdir_result = gettempdir()
        assert_true(tmpdir_result, "Failed to get temporary directory")
        assert_equal(
            tmpdir_result.value()[],
            String(dir_with_writing_access),
            "expected to get:" + String(dir_with_writing_access),
        )

    # test gettempdir falls back to TMP
    with _TempEnvForTest(
        dir_without_writing_access,
        non_existing_dir,
        dir_with_writing_access,
        _clean_up_gettempdir_test,
    ):
        tmpdir_result = gettempdir()
        assert_true(tmpdir_result, "Failed to get temporary directory")
        assert_equal(
            tmpdir_result.value()[],
            String(dir_with_writing_access),
            "expected to get:" + String(dir_with_writing_access),
        )

    _clean_up_gettempdir_test()


fn main() raises:
    test_mkdtemp()
    test_gettempdir()
