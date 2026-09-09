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

from asyncrt_test_utils import create_test_device_context
from std.testing import TestSuite, assert_equal


def test_host_mapped() raises:
    var ctx = create_test_device_context()

    comptime length = 20

    var in_buf = ctx.enqueue_create_buffer[.int64](length)
    var out_buf = ctx.enqueue_create_buffer[.int64](length)

    with in_buf.map_to_host() as in_map:
        for i in range(length):
            in_map[i] = Int64(i)

    in_buf.enqueue_copy_to(out_buf)

    with out_buf.map_to_host() as out_map:
        for i in range(length):
            assert_equal(out_map[i], Int64(i))

    print("Done")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
