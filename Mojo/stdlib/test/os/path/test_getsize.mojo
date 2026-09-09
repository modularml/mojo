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

from std.os.path import getsize
from std.tempfile import NamedTemporaryFile

from std.testing import TestSuite, assert_equal


def test_getsize() raises:
    with NamedTemporaryFile(delete=False) as tmp_file:
        var file_path = tmp_file.name
        # No bytes written yet, 0 size.
        assert_equal(getsize(file_path), 0)
        var data_to_write = "test"
        tmp_file.write(data_to_write)
        assert_equal(getsize(file_path), data_to_write.byte_length())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
