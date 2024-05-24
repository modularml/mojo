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
from os.path import getsize

from testing import assert_equal, assert_false


fn main() raises:
    # TODO: use `NamedTemporaryFile` once we implement it.
    alias file_name = "test_file"
    assert_false(os.path.exists(file_name), "File should not exist")
    with open(file_name, "w"):
        pass
    assert_equal(getsize(file_name), 0)
    with open(file_name, "w") as my_file:
        my_file.write(String("test"))
    assert_equal(getsize(file_name), 4)
    os.remove(file_name)
