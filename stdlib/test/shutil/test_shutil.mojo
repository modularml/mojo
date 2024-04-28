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

from shutil import rmtree
from os import mkdir
from os.path import exists
from pathlib import Path

from testing import assert_false, assert_raises


fn _create_some_files_and_dirs(path: Path) raises:
    mkdir(path / "nested")
    mkdir(path / "nested" / "more_nested")
    mkdir(path / "nested" / "more_nested" / "even_more_nested")

    _ = open(path / "nested.txt", "w")
    _ = open(path / "nested" / "more_nested.txt", "w")
    _ = open(path / "nested" / "more_nested" / "even_more_nested.txt", "w")


fn test_rmtree() raises:
    var cwd_path = Path()
    var my_dir_path = cwd_path / "my_dir"

    assert_false(
        my_dir_path.exists(),
        "Unexpected dir " + my_dir_path.__fspath__() + " it should not exist",
    )

    with assert_raises():
        rmtree(my_dir_path)

    mkdir(my_dir_path)
    _create_some_files_and_dirs(my_dir_path)

    rmtree(my_dir_path)

    assert_false(my_dir_path.exists())


fn main() raises:
    test_rmtree()
