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

from linalg.utils import partial_simd_load, partial_simd_store
from std.testing import TestSuite


# CHECK-LABEL: test_partial_load_store
def test_partial_load_store() raises:
    print("== test_partial_load_store")
    # The total amount of data to allocate
    comptime total_buffer_size: Int = 32

    var read_data = Array[Int, total_buffer_size](
        fill_with=lambda (idx: Int) -> Int: idx
    )
    var write_data = Array[Int, total_buffer_size](fill=0)

    var read_ptr = read_data.unsafe_ptr()
    var write_ptr = write_data.unsafe_ptr()

    # Test partial load:
    var partial_load_data = partial_simd_load[4](
        read_ptr + 1,
        1,
        3,
        99,  # idx  # lbound  # rbound  # pad value
    )
    # CHECK: [99, 2, 3, 99]
    print(partial_load_data)

    # Test partial store:
    partial_simd_store[4](
        write_ptr + 1,
        2,
        4,
        partial_load_data,  # idx  # lbound  # rbound
    )
    var partial_store_data = (write_ptr + 2).load[width=4]()
    # CHECK: [0, 3, 99, 0]
    print(partial_store_data)

    # Test 2D partial load store using pointer offsets.
    # Treat as 8x4 matrix (row-major, width=4).
    comptime width = 4

    # Test partial load at row 3, col 2:
    var nd_partial_load_data = partial_simd_load[4](
        read_ptr + 3 * width + 2,
        0,
        2,
        123,  # lbound  # rbound  # pad value
    )
    # CHECK: [14, 15, 123, 123]
    print(nd_partial_load_data)

    # Test partial store at row 3, col 1:
    partial_simd_store[4](
        write_ptr + 3 * width + 1,
        0,  # lbound
        3,  # rbound
        nd_partial_load_data,  # value
    )
    var nd_partial_store_data = (write_ptr + 3 * width).load[width=4]()

    # CHECK: [0, 14, 15, 123]
    print(nd_partial_store_data)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
