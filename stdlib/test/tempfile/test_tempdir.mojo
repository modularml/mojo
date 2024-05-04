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
from testing import assert_true, assert_false, assert_equal
from tempfile import TemporaryDirectory


fn test_temporary_directory() raises -> None:
    var tmp_dir: String = ""
    with TemporaryDirectory() as tmp_dir:
        assert_true(exists(tmp_dir), "Failed to create temp dir " + tmp_dir)
    assert_false(exists(tmp_dir), "Failed to delete temp dir " + tmp_dir)

    with TemporaryDirectory() as tmp_dir:
        assert_true(exists(tmp_dir), "Failed to create temp dir " + tmp_dir)
        _ = open(tmp_dir + "/test_file", "w")
        os.mkdir(tmp_dir + "/test_dir")
        _ = open(tmp_dir + "/test_dir/test_file2", "w")
    assert_false(exists(tmp_dir), "Failed to delete temp dir " + tmp_dir)


fn main() raises:
    test_temporary_directory()
