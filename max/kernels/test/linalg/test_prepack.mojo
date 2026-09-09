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

from std.math import ceildiv

from layout import Idx, TileTensor
from layout.tile_layout import row_major
from linalg.packing import pack_b
from std.memory import unsafe_memset_zero, unsafe_stack_allocation


# CHECK-LABEL: test_prepack
def test_prepack():
    print("== test_prepack")

    comptime k = 10
    comptime tile_k = 4
    comptime simd_size = 2
    comptime inner_size = 2 * simd_size
    comptime n = 12
    comptime tile_n = 8
    comptime type = DType.float32

    comptime k_padded = ceildiv(k, tile_k) * tile_k
    comptime n_padded = ceildiv(n, tile_n) * tile_n

    var src_ptr = unsafe_stack_allocation[n * k, type, alignment=64]()
    unsafe_memset_zero(src_ptr, n * k)
    var dst_ptr = unsafe_stack_allocation[
        n_padded * k_padded, type, alignment=64
    ]()
    unsafe_memset_zero(dst_ptr, n_padded * k_padded)

    for i in range(n * k):
        src_ptr[i] = Float32(i)

    var src_tt = TileTensor(src_ptr, row_major(k, n))
    var dst_tt = TileTensor(dst_ptr, row_major(k_padded, n_padded))

    pack_b[
        False,
        simd_size,
        inner_size,
        type,
        type,
        type,
    ](
        dst_tt,
        src_tt,
        tile_n,
        tile_k,
    )
    # CHECK: 0.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 2.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 12.0
    # CHECK-NEXT: 13.0
    # CHECK-NEXT: 14.0
    # CHECK-NEXT: 15.0
    # CHECK-NEXT: 24.0
    # CHECK-NEXT: 25.0
    # CHECK-NEXT: 26.0
    # CHECK-NEXT: 27.0
    # CHECK-NEXT: 36.0
    # CHECK-NEXT: 37.0
    # CHECK-NEXT: 38.0
    # CHECK-NEXT: 39.0
    # CHECK-NEXT: 4.0
    # CHECK-NEXT: 5.0
    # CHECK-NEXT: 6.0
    # CHECK-NEXT: 7.0
    # CHECK-NEXT: 16.0
    # CHECK-NEXT: 17.0
    # CHECK-NEXT: 18.0
    # CHECK-NEXT: 19.0
    # CHECK-NEXT: 28.0
    # CHECK-NEXT: 29.0
    # CHECK-NEXT: 30.0
    # CHECK-NEXT: 31.0
    # CHECK-NEXT: 40.0
    # CHECK-NEXT: 41.0
    # CHECK-NEXT: 42.0
    # CHECK-NEXT: 43.0
    # CHECK-NEXT: 8.0
    # CHECK-NEXT: 9.0
    # CHECK-NEXT: 10.0
    # CHECK-NEXT: 11.0
    # CHECK-NEXT: 20.0
    # CHECK-NEXT: 21.0
    # CHECK-NEXT: 22.0
    # CHECK-NEXT: 23.0
    # CHECK-NEXT: 32.0
    # CHECK-NEXT: 33.0
    # CHECK-NEXT: 34.0
    # CHECK-NEXT: 35.0
    # CHECK-NEXT: 44.0
    # CHECK-NEXT: 45.0
    # CHECK-NEXT: 46.0
    # CHECK-NEXT: 47.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 48.0
    # CHECK-NEXT: 49.0
    # CHECK-NEXT: 50.0
    # CHECK-NEXT: 51.0
    # CHECK-NEXT: 60.0
    # CHECK-NEXT: 61.0
    # CHECK-NEXT: 62.0
    # CHECK-NEXT: 63.0
    # CHECK-NEXT: 72.0
    # CHECK-NEXT: 73.0
    # CHECK-NEXT: 74.0
    # CHECK-NEXT: 75.0
    # CHECK-NEXT: 84.0
    # CHECK-NEXT: 85.0
    # CHECK-NEXT: 86.0
    # CHECK-NEXT: 87.0
    # CHECK-NEXT: 52.0
    # CHECK-NEXT: 53.0
    # CHECK-NEXT: 54.0
    # CHECK-NEXT: 55.0
    # CHECK-NEXT: 64.0
    # CHECK-NEXT: 65.0
    # CHECK-NEXT: 66.0
    # CHECK-NEXT: 67.0
    # CHECK-NEXT: 76.0
    # CHECK-NEXT: 77.0
    # CHECK-NEXT: 78.0
    # CHECK-NEXT: 79.0
    # CHECK-NEXT: 88.0
    # CHECK-NEXT: 89.0
    # CHECK-NEXT: 90.0
    # CHECK-NEXT: 91.0
    # CHECK-NEXT: 56.0
    # CHECK-NEXT: 57.0
    # CHECK-NEXT: 58.0
    # CHECK-NEXT: 59.0
    # CHECK-NEXT: 68.0
    # CHECK-NEXT: 69.0
    # CHECK-NEXT: 70.0
    # CHECK-NEXT: 71.0
    # CHECK-NEXT: 80.0
    # CHECK-NEXT: 81.0
    # CHECK-NEXT: 82.0
    # CHECK-NEXT: 83.0
    # CHECK-NEXT: 92.0
    # CHECK-NEXT: 93.0
    # CHECK-NEXT: 94.0
    # CHECK-NEXT: 95.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 96.0
    # CHECK-NEXT: 97.0
    # CHECK-NEXT: 98.0
    # CHECK-NEXT: 99.0
    # CHECK-NEXT: 108.0
    # CHECK-NEXT: 109.0
    # CHECK-NEXT: 110.0
    # CHECK-NEXT: 111.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 100.0
    # CHECK-NEXT: 101.0
    # CHECK-NEXT: 102.0
    # CHECK-NEXT: 103.0
    # CHECK-NEXT: 112.0
    # CHECK-NEXT: 113.0
    # CHECK-NEXT: 114.0
    # CHECK-NEXT: 115.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 104.0
    # CHECK-NEXT: 105.0
    # CHECK-NEXT: 106.0
    # CHECK-NEXT: 107.0
    # CHECK-NEXT: 116.0
    # CHECK-NEXT: 117.0
    # CHECK-NEXT: 118.0
    # CHECK-NEXT: 119.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 0.0

    for i in range(k_padded):
        for j in range(n_padded):
            print(dst_ptr[i * n_padded + j])


def main():
    test_prepack()
