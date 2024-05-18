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
    assert_true(dir_name.split("/")[-1].startswith("my_prefix"))
    assert_true(dir_name.split("/")[-1].endswith("my_suffix"))
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


fn main() raises:
    test_mkdtemp()
