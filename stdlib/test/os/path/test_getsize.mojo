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

from os.path import getsize
from tempfile import NamedTemporaryFile

from testing import assert_equal


def main():
    with NamedTemporaryFile(delete=False) as tmp_file:
        file_path = tmp_file.name
        # No bytes written yet, 0 size.
        assert_equal(getsize(file_path), 0)
        var data_to_write = "test"
        tmp_file.write(data_to_write)
        assert_equal(getsize(file_path), len(data_to_write))
