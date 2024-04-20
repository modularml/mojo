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
# RUN: %mojo-no-debug %s

import os
from os.path import exists
from testing import assert_true, assert_false, assert_equal
from tempfile import NamedTemporaryFile


fn test_named_temporary_file_deletion() raises:
    var tmp_file: NamedTemporaryFile
    var file_name: String

    with NamedTemporaryFile(prefix="my_prefix") as my_tmp_file:
        file_name = my_tmp_file.name
        assert_true(exists(file_name), "Failed to create file " + file_name)
        assert_true(file_name.split("/")[-1].startswith("my_prefix"))
    assert_false(exists(file_name), "Failed to delete file " + file_name)

    with NamedTemporaryFile(delete=False) as my_tmp_file:
        file_name = my_tmp_file.name
        assert_true(exists(file_name), "Failed to create file " + file_name)
    assert_true(exists(file_name), "File " + file_name + " should still exist")
    os.remove(file_name)

    tmp_file = NamedTemporaryFile()
    file_name = tmp_file.name
    assert_true(exists(file_name), "Failed to create file " + file_name)
    tmp_file.close()
    assert_false(exists(file_name), "Failed to delete file " + file_name)

    tmp_file = NamedTemporaryFile(delete=False)
    file_name = tmp_file.name
    assert_true(exists(file_name), "Failed to create file " + file_name)
    tmp_file.close()
    assert_true(exists(file_name), "File " + file_name + " should still exist")
    os.remove(file_name)


fn test_named_temporary_file_write() raises:
    var file_name: String
    var contents: String

    with NamedTemporaryFile(delete=False) as my_tmp_file:
        file_name = my_tmp_file.name
        my_tmp_file.write("hello world")

    with open(file_name, "r") as my_file:
        contents = my_file.read()
    assert_equal("hello world", contents)
    os.remove(file_name)


fn main() raises:
    test_named_temporary_file_deletion()
    test_named_temporary_file_write()
