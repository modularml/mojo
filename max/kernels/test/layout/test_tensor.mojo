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
from std.sys import align_of

from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout import Layout, UNKNOWN_VALUE
from layout.int_tuple import product
from layout.layout_tensor import *
from std.testing import assert_equal


def print_raw_major_tensor(tensor: LayoutTensor):
    for i in range(tensor.shape[0]()):
        for j in range(tensor.shape[1]()):
            print(tensor[i, j], "\t", end="")
        print("")


def print_tile_tensor(tensor: LayoutTensor):
    for i in range(tensor.shape[0]()):
        for j in range(tensor.shape[1]()):
            print(tensor[i, j], "\t", end="")
        print("")


# Print for shape ((m, n), (p, q)) in a 2D format
def print_mode2_shape2_tensor[
    layout: Layout, dtype: DType
](tensor: LayoutTensor[dtype, layout, MutAnyOrigin]):
    comptime assert (
        len(layout) == 2
        and len(layout.shape[0]) == 2
        and len(layout.shape[1]) == 2
    )
    var materialized_layout = materialize[layout]()
    for i in range(product(layout.shape[0])):
        for j in range(product(layout.shape[1])):
            var idx = materialized_layout(IntTuple(i, j))
            print(tensor.ptr[idx], "\t", end="")
        print("")


# CHECK-LABEL: test_basic_tensor_ops
def test_basic_tensor_ops() raises:
    print("== test_basic_tensor_ops")

    var managed_tensor = ManagedLayoutTensor[.float32, Layout(IntTuple(8, 4))]()
    var tensor = managed_tensor.tensor()
    arange(tensor)

    # CHECK: ----original matrix----
    # CHECK: 0.0      1.0     2.0     3.0
    # CHECK: 4.0      5.0     6.0     7.0
    # CHECK: 8.0      9.0     10.0    11.0
    # CHECK: 12.0     13.0    14.0    15.0
    # CHECK: 16.0     17.0    18.0    19.0
    # CHECK: 20.0     21.0    22.0    23.0
    # CHECK: 24.0     25.0    26.0    27.0
    # CHECK: 28.0     29.0    30.0    31.0
    print("----original matrix----")
    print_raw_major_tensor(tensor)

    # CHECK: ----transposed matrix----
    # CHECK: 0.0     4.0     8.0     12.0    16.0    20.0    24.0    28.0
    # CHECK: 1.0     5.0     9.0     13.0    17.0    21.0    25.0    29.0
    # CHECK: 2.0     6.0     10.0    14.0    18.0    22.0    26.0    30.0
    # CHECK: 3.0     7.0     11.0    15.0    19.0    23.0    27.0    31.0
    var transposed_tensor = tensor.transpose()
    print("----transposed matrix----")
    print_raw_major_tensor(transposed_tensor)

    # CHECK: ----tile[ 0 , 0 ]----
    # CHECK: 0.0     4.0
    # CHECK: 1.0     5.0
    # CHECK: ----tile[ 0 , 1 ]----
    # CHECK: 8.0     12.0
    # CHECK: 9.0     13.0
    # CHECK: ----tile[ 0 , 2 ]----
    # CHECK: 16.0    20.0
    # CHECK: 17.0    21.0
    # CHECK: ----tile[ 0 , 3 ]----
    # CHECK: 24.0    28.0
    # CHECK: 25.0    29.0
    # CHECK: ----tile[ 1 , 0 ]----
    # CHECK: 2.0     6.0
    # CHECK: 3.0     7.0
    # CHECK: ----tile[ 1 , 1 ]----
    # CHECK: 10.0    14.0
    # CHECK: 11.0    15.0
    # CHECK: ----tile[ 1 , 2 ]----
    # CHECK: 18.0    22.0
    # CHECK: 19.0    23.0
    # CHECK: ----tile[ 1 , 3 ]----
    # CHECK: 26.0    30.0
    # CHECK: 27.0    31.0
    for tile_i in range(2):
        for tile_j in range(4):
            print("----tile[", tile_i, ",", tile_j, "]----")
            var tile_2x2 = transposed_tensor.tile[2, 2](tile_i, tile_j)
            print_tile_tensor(tile_2x2)

    print("----1d-tensor-tiles----")
    var tensor_8 = LayoutTensor[
        .float32, Layout(8), MutAnyOrigin
    ].stack_allocation[stack_alignment=16]()
    arange(tensor_8)
    # CHECK: ----tile[ 0 ]----
    # CHECK: 0.0
    # CHECK: 1.0
    # CHECK: ----tile[ 1 ]----
    # CHECK: 2.0
    # CHECK: 3.0
    # CHECK: ----tile[ 2 ]----
    # CHECK: 4.0
    # CHECK: 5.0
    # CHECK: ----tile[ 3 ]----
    # CHECK: 6.0
    # CHECK: 7.0
    for tile_i in range(4):
        print("----tile[", tile_i, "]----")
        var tile = tensor_8.tile[2](tile_i)
        print(tile)

    _ = managed_tensor^


# CHECK-LABEL: test_tesnsor_fragments
#   Get fragments of the following layout
#   TH_0    TH_2    TH_0    TH_2
#   TH_1    TH_3    TH_1    TH_3
#   TH_0    TH_2    TH_0    TH_2
#   TH_1    TH_3    TH_1    TH_3
#   TH_0    TH_2    TH_0    TH_2
#   TH_1    TH_3    TH_1    TH_3
#   TH_0    TH_2    TH_0    TH_2
#   TH_1    TH_3    TH_1    TH_3
def test_tesnsor_fragments() raises:
    print("== test_tesnsor_fragments")

    var managed_tensor = ManagedLayoutTensor[.float32, Layout(IntTuple(8, 4))]()

    var tensor = managed_tensor.tensor()
    arange(tensor)

    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 0.0     2.0
    # CHECK: 8.0     10.0
    # CHECK: 16.0    18.0
    # CHECK: 24.0    26.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 4.0     6.0
    # CHECK: 12.0    14.0
    # CHECK: 20.0    22.0
    # CHECK: 28.0    30.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 1.0     3.0
    # CHECK: 9.0     11.0
    # CHECK: 17.0    19.0
    # CHECK: 25.0    27.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 5.0     7.0
    # CHECK: 13.0    15.0
    # CHECK: 21.0    23.0
    # CHECK: 29.0    31.0
    for th_i in range(4):
        print("----fragments-data[", th_i, "]----")
        var fragment_4x2 = tensor.distribute[Layout(IntTuple(2, 2))](th_i)
        print_tile_tensor(fragment_4x2)

    _ = managed_tensor^


# CHECK-LABEL: test_tensor_tile_and_distribute
def test_tensor_tile_and_distribute() raises:
    print("== test_tensor_tile_and_distribute")

    var managed_tensor = ManagedLayoutTensor[.float32, Layout(IntTuple(8, 8))]()

    var tensor = managed_tensor.tensor()
    arange(tensor)

    # CHECK: ----tile-data[ 0 , 0 ]----
    # CHECK: 0.0     1.0     2.0     3.0
    # CHECK: 8.0     9.0     10.0    11.0
    # CHECK: 16.0    17.0    18.0    19.0
    # CHECK: 24.0    25.0    26.0    27.0
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 0.0     2.0
    # CHECK: 16.0    18.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 1.0     3.0
    # CHECK: 17.0    19.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 8.0     10.0
    # CHECK: 24.0    26.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 9.0     11.0
    # CHECK: 25.0    27.0
    # CHECK: ----tile-data[ 0 , 1 ]----
    # CHECK: 4.0     5.0     6.0     7.0
    # CHECK: 12.0    13.0    14.0    15.0
    # CHECK: 20.0    21.0    22.0    23.0
    # CHECK: 28.0    29.0    30.0    31.0
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 4.0     6.0
    # CHECK: 20.0    22.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 5.0     7.0
    # CHECK: 21.0    23.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 12.0    14.0
    # CHECK: 28.0    30.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 13.0    15.0
    # CHECK: 29.0    31.0
    # CHECK: ----tile-data[ 1 , 0 ]----
    # CHECK: 32.0    33.0    34.0    35.0
    # CHECK: 40.0    41.0    42.0    43.0
    # CHECK: 48.0    49.0    50.0    51.0
    # CHECK: 56.0    57.0    58.0    59.0
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 32.0    34.0
    # CHECK: 48.0    50.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 33.0    35.0
    # CHECK: 49.0    51.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 40.0    42.0
    # CHECK: 56.0    58.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 41.0    43.0
    # CHECK: 57.0    59.0
    # CHECK: ----tile-data[ 1 , 1 ]----
    # CHECK: 36.0    37.0    38.0    39.0
    # CHECK: 44.0    45.0    46.0    47.0
    # CHECK: 52.0    53.0    54.0    55.0
    # CHECK: 60.0    61.0    62.0    63.0
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 36.0    38.0
    # CHECK: 52.0    54.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 37.0    39.0
    # CHECK: 53.0    55.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 44.0    46.0
    # CHECK: 60.0    62.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 45.0    47.0
    # CHECK: 61.0    63.0
    for tile_i in range(2):
        for tile_j in range(2):
            var tile_4x4 = tensor.tile[4, 4](tile_i, tile_j)
            print("----tile-data[", tile_i, ",", tile_j, "]----")
            print_tile_tensor(tile_4x4)
            for th_i in range(4):
                var fragment_2x2 = tile_4x4.distribute[
                    Layout(IntTuple(2, 2), IntTuple(2, 1))
                ](th_i)
                print("----fragments-data[", th_i, "]----")
                print_tile_tensor(fragment_2x2)
    _ = managed_tensor^


# CHECK-LABEL: test_tensor_tile_and_distribute_custom_layout
def test_tensor_tile_and_distribute_custom_layout() raises:
    print("== test_tensor_tile_and_distribute_custom_layout")
    var managed_tensor = ManagedLayoutTensor[.float32, Layout(IntTuple(2, 4))]()
    var tensor = managed_tensor.tensor()
    arange(tensor)
    # CHECK: 0.0   1.0   2.0   3.0
    # CHECK: 4.0   5.0   6.0   7.0
    print(tensor)

    # CHECK: row-major-thread-layout
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 0.0   2.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 1.0   3.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 4.0   6.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 5.0   7.0
    print("row-major-thread-layout")
    for th_i in range(4):
        var fragments_1x2 = tensor.distribute[
            Layout(IntTuple(2, 2), IntTuple(2, 1))
        ](th_i)
        print("----fragments-data[", th_i, "]----")
        print(fragments_1x2)

    # CHECK: col-major-thread-layout
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 0.0   2.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 4.0   6.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 1.0   3.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 5.0   7.0
    print("col-major-thread-layout")
    for th_i in range(4):
        var fragments_1x2 = tensor.distribute[
            Layout(IntTuple(2, 2), IntTuple(1, 2))
        ](th_i)
        print("----fragments-data[", th_i, "]----")
        print(fragments_1x2)

    _ = managed_tensor^


# CHECK-LABEL: test_copy_to_tile_major_layout
def test_copy_to_tile_major_layout():
    print("== test_copy_to_tile_major_layout")
    var mat_4x4_row_major = LayoutTensor[
        .float32, Layout(IntTuple(4, 4), IntTuple(4, 1)), MutAnyOrigin
    ].stack_allocation[stack_alignment=16]()
    arange(mat_4x4_row_major)

    # CHECK: (((2, 2), (2, 2)):((1, 8), (2, 4)))
    # CHECK:        0    1    2    3
    # CHECK:     +----+----+----+----+
    # CHECK:  0  |  0 |  2 |  4 |  6 |
    # CHECK:     +----+----+----+----+
    # CHECK:  1  |  1 |  3 |  5 |  7 |
    # CHECK:     +----+----+----+----+
    # CHECK:  2  |  8 | 10 | 12 | 14 |
    # CHECK:     +----+----+----+----+
    # CHECK:  3  |  9 | 11 | 13 | 15 |
    # CHECK:     +----+----+----+----+
    comptime tiled_major_layout = Layout(
        IntTuple(IntTuple(2, 2), IntTuple(2, 2)),
        IntTuple(IntTuple(1, 8), IntTuple(2, 4)),
    )
    var mat_4x4_tiled_2x2 = LayoutTensor[
        .float32, tiled_major_layout, MutAnyOrigin
    ].stack_allocation[stack_alignment=16]()
    print_layout(materialize[tiled_major_layout]())

    mat_4x4_tiled_2x2.copy_from(mat_4x4_row_major)

    # CHECK: mat_4x4_row_major:
    # CHECK: row: 0 data 0.0         1.0     2.0     3.0
    # CHECK: row: 1 data 4.0         5.0     6.0     7.0
    # CHECK: row: 2 data 8.0         9.0     10.0    11.0
    # CHECK: row: 3 data 12.0        13.0    14.0    15.0
    print("mat_4x4_row_major:")
    for i in range(4):
        print("row:", i, "data ", end="")
        for j in range(4):
            print(mat_4x4_row_major.ptr[i * 4 + j], "\t", end="")
        print("")

    # CHECK: mat_4x4_tiled_2x2:
    # CHECK: row: 0 data 0.0         4.0     1.0     5.0
    # CHECK: row: 1 data 2.0         6.0     3.0     7.0
    # CHECK: row: 2 data 8.0         12.0    9.0     13.0
    # CHECK: row: 3 data 10.0        14.0    11.0    15.0
    print("mat_4x4_tiled_2x2:")
    for i in range(4):
        print("row:", i, "data ", end="")
        for j in range(4):
            print(mat_4x4_tiled_2x2.ptr[i * 4 + j], "\t", end="")
        print("")


# This test repeats the following layout into the 4x8 matrix resulting:
# TH_0 TH_2 TH_4 TH_6 TH_0 TH_2 TH_4 TH_6
# TH_1 TH_3 TH_5 TH_7 TH_1 TH_3 TH_5 TH_7
# TH_0 TH_2 TH_4 TH_6 TH_0 TH_2 TH_4 TH_6
# TH_1 TH_3 TH_5 TH_7 TH_1 TH_3 TH_5 TH_7
def test_distribute_tiled_layout():
    print("== test_distribute_tiled_layout")
    var tensor = LayoutTensor[
        .float32, Layout(IntTuple(4, 8), IntTuple(8, 1)), MutAnyOrigin
    ].stack_allocation[stack_alignment=16]()
    arange(tensor)
    comptime threads_2x4_layout = Layout(
        IntTuple(2, IntTuple(2, 2)), IntTuple(1, IntTuple(2, 4))
    )
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 0.0   4.0
    # CHECK: 16.0   20.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 8.0   12.0
    # CHECK: 24.0   28.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 1.0   5.0
    # CHECK: 17.0   21.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 9.0   13.0
    # CHECK: 25.0   29.0
    # CHECK: ----fragments-data[ 4 ]----
    # CHECK: 2.0   6.0
    # CHECK: 18.0   22.0
    # CHECK: ----fragments-data[ 5 ]----
    # CHECK: 10.0   14.0
    # CHECK: 26.0   30.0
    # CHECK: ----fragments-data[ 6 ]----
    # CHECK: 3.0   7.0
    # CHECK: 19.0   23.0
    # CHECK: ----fragments-data[ 7 ]----
    # CHECK: 11.0   15.0
    # CHECK: 27.0   31.0
    for th_i in range(8):
        var thread_tile = tensor.distribute[threads_2x4_layout](th_i)
        print("----fragments-data[", th_i, "]----")
        print(thread_tile)


# CHECK-LABEL: test_distribute_with_tile_size
def test_distribute_with_tile_size():
    print("== test_distribute_with_tile_size")

    var tensor0 = LayoutTensor[
        .float32, Layout(IntTuple(16, 8), IntTuple(8, 1)), MutAnyOrigin
    ].stack_allocation[stack_alignment=16]()
    arange(tensor0)

    # Thread layout:
    # TH_0 TH_2
    # TH_1 TH_3
    # TH_4 TH_6
    # TH_5 TH_7
    comptime thread_layout = Layout(
        IntTuple(IntTuple(2, 2), 2), IntTuple(IntTuple(1, 4), 2)
    )

    # Each thread should have a rank-2 layout. The first mode is a tile to work
    # on. The second mode is how this tile repeats in the domain. The tiles are
    # distributed as follow:
    # +-----------+-----------+-----------+-----------+
    # | TH_0 TH_0 | TH_2 TH_2 | TH_0 TH_0 | TH_2 TH_2 |
    # | TH_0 TH_0 | TH_2 TH_2 | TH_0 TH_0 | TH_2 TH_2 |
    # |-----------+-----------+-----------+-----------+
    # | TH_1 TH_1 | TH_3 TH_3 | TH_1 TH_1 | TH_3 TH_3 |
    # | TH_1 TH_1 | TH_3 TH_3 | TH_1 TH_1 | TH_3 TH_3 |
    # |-----------+-----------+-----------+-----------+
    # | TH_4 TH_4 | TH_6 TH_6 | TH_4 TH_4 | TH_6 TH_6 |
    # | TH_4 TH_4 | TH_6 TH_6 | TH_4 TH_4 | TH_6 TH_6 |
    # |-----------+-----------+-----------+-----------+
    # | TH_5 TH_5 | TH_7 TH_7 | TH_5 TH_5 | TH_7 TH_7 |
    # | TH_5 TH_5 | TH_7 TH_7 | TH_5 TH_5 | TH_7 TH_7 |
    # |-----------+-----------+-----------+-----------+
    # | TH_0 TH_0 | TH_2 TH_2 | TH_0 TH_0 | TH_2 TH_2 |
    # | TH_0 TH_0 | TH_2 TH_2 | TH_0 TH_0 | TH_2 TH_2 |
    # |-----------+-----------+-----------+-----------+
    # | TH_1 TH_1 | TH_3 TH_3 | TH_1 TH_1 | TH_3 TH_3 |
    # | TH_1 TH_1 | TH_3 TH_3 | TH_1 TH_1 | TH_3 TH_3 |
    # |-----------+-----------+-----------+-----------+
    # | TH_4 TH_4 | TH_6 TH_6 | TH_4 TH_4 | TH_6 TH_6 |
    # | TH_4 TH_4 | TH_6 TH_6 | TH_4 TH_4 | TH_6 TH_6 |
    # |-----------+-----------+-----------+-----------+
    # | TH_5 TH_5 | TH_7 TH_7 | TH_5 TH_5 | TH_7 TH_7 |
    # | TH_5 TH_5 | TH_7 TH_7 | TH_5 TH_5 | TH_7 TH_7 |
    # +-----------+-----------+-----------+-----------+

    # CHECK: ----thread[ 0 ]----
    # CHECK: [0.0, 1.0, 8.0, 9.0] [4.0, 5.0, 12.0, 13.0]
    # CHECK: [64.0, 65.0, 72.0, 73.0] [68.0, 69.0, 76.0, 77.0]
    # CHECK: ----thread[ 1 ]----
    # CHECK: [16.0, 17.0, 24.0, 25.0] [20.0, 21.0, 28.0, 29.0]
    # CHECK: [80.0, 81.0, 88.0, 89.0] [84.0, 85.0, 92.0, 93.0]
    # CHECK: ----thread[ 2 ]----
    # CHECK: [2.0, 3.0, 10.0, 11.0] [6.0, 7.0, 14.0, 15.0]
    # CHECK: [66.0, 67.0, 74.0, 75.0] [70.0, 71.0, 78.0, 79.0]
    # CHECK: ----thread[ 3 ]----
    # CHECK: [18.0, 19.0, 26.0, 27.0] [22.0, 23.0, 30.0, 31.0]
    # CHECK: [82.0, 83.0, 90.0, 91.0] [86.0, 87.0, 94.0, 95.0]
    # CHECK: ----thread[ 4 ]----
    # CHECK: [32.0, 33.0, 40.0, 41.0] [36.0, 37.0, 44.0, 45.0]
    # CHECK: [96.0, 97.0, 104.0, 105.0] [100.0, 101.0, 108.0, 109.0]
    # CHECK: ----thread[ 5 ]----
    # CHECK: [48.0, 49.0, 56.0, 57.0] [52.0, 53.0, 60.0, 61.0]
    # CHECK: [112.0, 113.0, 120.0, 121.0] [116.0, 117.0, 124.0, 125.0]
    # CHECK: ----thread[ 6 ]----
    # CHECK: [34.0, 35.0, 42.0, 43.0] [38.0, 39.0, 46.0, 47.0]
    # CHECK: [98.0, 99.0, 106.0, 107.0] [102.0, 103.0, 110.0, 111.0]
    # CHECK: ----thread[ 7 ]----
    # CHECK: [50.0, 51.0, 58.0, 59.0] [54.0, 55.0, 62.0, 63.0]
    # CHECK: [114.0, 115.0, 122.0, 123.0] [118.0, 119.0, 126.0, 127.0]

    for tid in range(comptime (thread_layout.size())):
        print("----thread[", tid, "]----")
        var tile = tensor0.vectorize[2, 2]().distribute[thread_layout](tid)
        print(tile)
    var tensor8x1 = LayoutTensor[
        .float32, Layout(IntTuple(8, 1)), MutAnyOrigin
    ].stack_allocation[stack_alignment=16]()
    arange(tensor8x1)

    # +-------------+
    # | TH_0 | TH_2 |
    # | TH_0 | TH_2 |
    # |------+------+
    # | TH_1 | TH_3 |
    # | TH_1 | TH_3 |
    # |------+------+
    # | TH_4 | TH_6 |
    # | TH_4 | TH_6 |
    # |------+------+
    # | TH_5 | TH_7 |
    # | TH_5 | TH_7 |
    # |------+------+

    # Threads 2, 3, 6, 7 should have the same fragments as thread 0, 1, 4, 5.
    # CHECK: ----thread[ 0 ]----
    # CHECK: [0.0, 1.0]
    # CHECK: ----thread[ 1 ]----
    # CHECK: [2.0, 3.0]
    # CHECK: ----thread[ 2 ]----
    # CHECK: [0.0, 1.0]
    # CHECK: ----thread[ 3 ]----
    # CHECK: [2.0, 3.0]
    # CHECK: ----thread[ 4 ]----
    # CHECK: [4.0, 5.0]
    # CHECK: ----thread[ 5 ]----
    # CHECK: [6.0, 7.0]
    # CHECK: ----thread[ 6 ]----
    # CHECK: [4.0, 5.0]
    # CHECK: ----thread[ 7 ]----
    # CHECK: [6.0, 7.0]
    for tid in range(comptime (thread_layout.size())):
        print("----thread[", tid, "]----")
        var tile = tensor8x1.vectorize[2, 1]().distribute[
            thread_layout, axis=0
        ](tid)
        print(tile)


# CHECK-LABEL: test_vectorize_reads
def test_vectorize_reads():
    print("== test_vectorize_reads")
    var tensor = LayoutTensor[
        .float32, Layout(IntTuple(8, 8), IntTuple(1, 8)), MutAnyOrigin
    ].stack_allocation[stack_alignment=16]()
    arange(tensor)
    var tensor_8_2_vec = tensor.vectorize[1, 4]()
    # CHECK: ((8, 2):(1, 32))
    print(materialize[tensor_8_2_vec.layout]())
    # CHECK: ((1, 4):(0, 8))
    print(materialize[tensor_8_2_vec.element_layout]())
    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    print(tensor_8_2_vec)

    var tensor_2_8_vec = tensor.vectorize[4, 1]()
    # CHECK: ((2, 8):(4, 8))
    print(materialize[tensor_2_8_vec.layout]())
    # CHECK: ((4, 1):(1, 0))
    print(materialize[tensor_2_8_vec.element_layout]())
    # CHECK: [0.0, 8.0, 16.0, 24.0] [1.0, 9.0, 17.0, 25.0] [2.0, 10.0, 18.0, 26.0] [3.0, 11.0, 19.0, 27.0] [4.0, 12.0, 20.0, 28.0] [5.0, 13.0, 21.0, 29.0] [6.0, 14.0, 22.0, 30.0] [7.0, 15.0, 23.0, 31.0]
    # CHECK: [32.0, 40.0, 48.0, 56.0] [33.0, 41.0, 49.0, 57.0] [34.0, 42.0, 50.0, 58.0] [35.0, 43.0, 51.0, 59.0] [36.0, 44.0, 52.0, 60.0] [37.0, 45.0, 53.0, 61.0] [38.0, 46.0, 54.0, 62.0] [39.0, 47.0, 55.0, 63.0]
    print(tensor_2_8_vec)

    var tensor_2_vec = tensor.vectorize[4, 4]()
    # CHECK: ((2, 2):(4, 32))
    print(materialize[tensor_2_vec.layout]())
    # CHECK: (4, 4):(1, 8))
    print(materialize[tensor_2_vec.element_layout]())
    # CHECK: [0.0, 8.0, 16.0, 24.0, 1.0, 9.0, 17.0, 25.0, 2.0, 10.0, 18.0, 26.0, 3.0, 11.0, 19.0, 27.0] [4.0, 12.0, 20.0, 28.0, 5.0, 13.0, 21.0, 29.0, 6.0, 14.0, 22.0, 30.0, 7.0, 15.0, 23.0, 31.0]
    # CHECK: [32.0, 40.0, 48.0, 56.0, 33.0, 41.0, 49.0, 57.0, 34.0, 42.0, 50.0, 58.0, 35.0, 43.0, 51.0, 59.0] [36.0, 44.0, 52.0, 60.0, 37.0, 45.0, 53.0, 61.0, 38.0, 46.0, 54.0, 62.0, 39.0, 47.0, 55.0, 63.0]
    print(tensor_2_vec)


# CHECK-LABEL: test_vectorize_writes
def test_vectorize_writes():
    print("== test_vectorize_writes")
    var tensor = (
        LayoutTensor[
            .float32,
            Layout(IntTuple(4, 4), IntTuple(1, 4)),
            MutAnyOrigin,
        ]
        .stack_allocation[stack_alignment=16]()
        .fill(0)
    )

    var tensor_2x2 = tensor.vectorize[2, 2]()
    tensor_2x2[0, 0] = rebind[tensor_2x2.element_type](SIMD[.float32, 4](1))
    tensor_2x2[0, 1] = rebind[tensor_2x2.element_type](SIMD[.float32, 4](2))
    tensor_2x2[1, 0] = rebind[tensor_2x2.element_type](SIMD[.float32, 4](3))
    tensor_2x2[1, 1] = rebind[tensor_2x2.element_type](SIMD[.float32, 4](4))
    # CHECK: 1.0 1.0 2.0 2.0
    # CHECK: 1.0 1.0 2.0 2.0
    # CHECK: 3.0 3.0 4.0 4.0
    # CHECK: 3.0 3.0 4.0 4.0
    print(tensor)


# CHECK-LABEL: test_slice
def test_slice():
    print("==test_slice")
    var tensor = LayoutTensor[
        .float32, Layout(IntTuple(4, 4), IntTuple(1, 4)), MutAnyOrigin
    ].stack_allocation[stack_alignment=16]()
    arange(tensor)
    # CHECK: row_slice_sub_column
    # CHECK: 0.0 1.0
    # CHECK: 4.0 5.0
    # CHECK: 8.0 9.0
    # CHECK: 12.0 13.0
    print("row_slice_sub_column")
    print(tensor.slice[:, :2]())
    # CHECK: col_slice_sub_row
    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 4.0 5.0 6.0 7.0
    print("col_slice_sub_row")
    print(tensor.slice[:2, :]())
    # sub_slice
    # 5.0 6.0
    # 9.0 10.0
    # 13.0 14.0
    print("sub_slice")
    print(tensor.slice[1:, 1:3]())
    # CHECK: bottom_right
    # CHECK: 10.0 11.0
    # CHECK: 14.0 15.0
    print("bottom_right")
    print(tensor.slice[2:, 2:]())
    # CHECK: top_left
    # CHECK: 0.0 1.0
    # CHECK: 4.0 5.0
    print("top_left")
    print(tensor.slice[:2, :2]())

    print("slice_of_slice")
    # CHECK: slice_of_slice
    # CHECK: 6.0 7.0
    # CHECK: 10.0 11.0
    print(tensor.slice[1:, 1:]().slice[:2, 1:]())


# CHECK-LABEL: test_copy_vectorized
def test_copy_vectorized():
    print("== test_copy_vectorized")
    var tensor_8_8 = LayoutTensor[
        .float32, Layout(IntTuple(8, 8), IntTuple(8, 1)), MutAnyOrigin
    ].stack_allocation[stack_alignment=16]()
    arange(tensor_8_8)
    var vec_8_1 = tensor_8_8.vectorize[1, 4]()
    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    print(vec_8_1)
    var tensor_8_8_zeros = (
        LayoutTensor[
            .float32,
            Layout(IntTuple(8, 8), IntTuple(8, 1)),
            MutAnyOrigin,
        ]
        .stack_allocation[stack_alignment=align_of[SIMD[.float32, 4]]()]()
        .vectorize[1, 4]()
        .fill(0)
    )

    tensor_8_8_zeros.copy_from(vec_8_1)
    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    print(tensor_8_8_zeros)

    var tensor_8_8_zeros_4_1 = (
        stack_allocation_like(tensor_8_8).vectorize[4, 1]().fill(0)
    )

    tensor_8_8_zeros_4_1.copy_from(vec_8_1)
    # CHECK: [0.0, 1.0, 2.0, 3.0] [16.0, 17.0, 18.0, 19.0] [32.0, 33.0, 34.0, 35.0] [48.0, 49.0, 50.0, 51.0] [4.0, 5.0, 6.0, 7.0] [20.0, 21.0, 22.0, 23.0] [36.0, 37.0, 38.0, 39.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [24.0, 25.0, 26.0, 27.0] [40.0, 41.0, 42.0, 43.0] [56.0, 57.0, 58.0, 59.0] [12.0, 13.0, 14.0, 15.0] [28.0, 29.0, 30.0, 31.0] [44.0, 45.0, 46.0, 47.0] [60.0, 61.0, 62.0, 63.0]
    print(tensor_8_8_zeros_4_1)

    var tensor_8_8_zeros_1_4 = (
        stack_allocation_like(tensor_8_8).vectorize[1, 4]().fill(0)
    )

    tensor_8_8_zeros_1_4.copy_from(tensor_8_8_zeros_4_1)
    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    print(tensor_8_8_zeros_1_4)

    var tensor_8_8_zeros_4_4 = (
        LayoutTensor[
            .float32,
            Layout(IntTuple(8, 8), IntTuple(8, 1)),
            MutAnyOrigin,
        ]
        .stack_allocation[stack_alignment=align_of[SIMD[.float32, 4]]()]()
        .vectorize[4, 4]()
        .fill(0)
    )

    tensor_8_8_zeros_4_4.copy_from(tensor_8_8.vectorize[4, 4]())
    # CHECK: [0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0, 16.0, 17.0, 18.0, 19.0, 24.0, 25.0, 26.0, 27.0] [4.0, 5.0, 6.0, 7.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0, 22.0, 23.0, 28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0, 40.0, 41.0, 42.0, 43.0, 48.0, 49.0, 50.0, 51.0, 56.0, 57.0, 58.0, 59.0] [36.0, 37.0, 38.0, 39.0, 44.0, 45.0, 46.0, 47.0, 52.0, 53.0, 54.0, 55.0, 60.0, 61.0, 62.0, 63.0]
    print(tensor_8_8_zeros_4_4)


# CHECK-LABEL: test_distribute_vectorized
def test_distribute_vectorized():
    print("== test_distribute_vectorized")
    var tensor_8_8 = LayoutTensor[
        .float32, Layout(IntTuple(8, 8), IntTuple(8, 1)), MutAnyOrigin
    ].stack_allocation[stack_alignment=16]()
    arange(tensor_8_8)

    var tensor_8_2xv4 = tensor_8_8.vectorize[1, 4]()
    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    print(tensor_8_2xv4)

    # CHECK: ----thread[ 0 ]----
    # CHECK: [0.0, 1.0, 2.0, 3.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0]
    # CHECK: ----thread[ 1 ]----
    # CHECK: [8.0, 9.0, 10.0, 11.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0]
    # CHECK: ----thread[ 2 ]----
    # CHECK: [4.0, 5.0, 6.0, 7.0]
    # CHECK: [20.0, 21.0, 22.0, 23.0]
    # CHECK: [36.0, 37.0, 38.0, 39.0]
    # CHECK: [52.0, 53.0, 54.0, 55.0]
    # CHECK: ----thread[ 3 ]----
    # CHECK: [12.0, 13.0, 14.0, 15.0]
    # CHECK: [28.0, 29.0, 30.0, 31.0]
    # CHECK: [44.0, 45.0, 46.0, 47.0]
    # CHECK: [60.0, 61.0, 62.0, 63.0]
    for tid in range(4):
        var fragments = tensor_8_2xv4.distribute[Layout(IntTuple(2, 2))](tid)
        print("----thread[", tid, "]----")
        print(fragments)

    # Fill the buffer first because we can't fill a vectorized tensor.
    # This will become easier when we can vectorize nested layout.
    var ptr = unsafe_stack_allocation[64 * 32, DType.float32, alignment=16]()
    for i in range(64 * 32):
        ptr[i] = Float32(i)

    var tensor_4x16x64 = LayoutTensor[
        .float32,
        Layout(IntTuple(IntTuple(16, 16), 4), IntTuple(IntTuple(32, 2), 512)),
        element_layout=Layout(2),
    ](ptr)

    comptime thread_layout = Layout(
        IntTuple(IntTuple(8, 4), 4), IntTuple(IntTuple(4, 1), 32)
    )

    # CHECK: ----thread[ 0 ]----
    # CHECK: [0.0, 1.0] [8.0, 9.0] [16.0, 17.0] [24.0, 25.0]
    # CHECK: [256.0, 257.0] [264.0, 265.0] [272.0, 273.0] [280.0, 281.0]
    print("----thread[", 0, "]----")
    print(tensor_4x16x64.distribute[thread_layout](0))
    # CHECK: ----thread[ 37 ]----
    # CHECK: [546.0, 547.0] [554.0, 555.0] [562.0, 563.0] [570.0, 571.0]
    # CHECK: [802.0, 803.0] [810.0, 811.0] [818.0, 819.0] [826.0, 827.0]
    print("----thread[", 37, "]----")
    print(tensor_4x16x64.distribute[thread_layout](37))
    # CHECK: ----thread[ 74 ]----
    # CHECK: [1092.0, 1093.0] [1100.0, 1101.0] [1108.0, 1109.0] [1116.0, 1117.0]
    # CHECK: [1348.0, 1349.0] [1356.0, 1357.0] [1364.0, 1365.0] [1372.0, 1373.0]
    print("----thread[", 74, "]----")
    print(tensor_4x16x64.distribute[thread_layout](74))
    # CHECK: ----thread[ 111 ]----
    # CHECK: [1638.0, 1639.0] [1646.0, 1647.0] [1654.0, 1655.0] [1662.0, 1663.0]
    # CHECK: [1894.0, 1895.0] [1902.0, 1903.0] [1910.0, 1911.0] [1918.0, 1919.0]
    print("----thread[", 111, "]----")
    print(tensor_4x16x64.distribute[thread_layout](111))


def test_distribute_axis_projection():
    var tensor_4x4 = LayoutTensor[
        .float32, Layout(IntTuple(4, 4), IntTuple(4, 1)), MutAnyOrigin
    ].stack_allocation[stack_alignment=16]()
    arange(tensor_4x4)

    # CHECK: th_id 0
    # CHECK: [0.0, 1.0, 2.0, 3.0]
    # CHECK: =====
    # CHECK: th_id 1
    # CHECK: [0.0, 1.0, 2.0, 3.0]
    # CHECK: =====
    # CHECK: th_id 2
    # CHECK: [0.0, 1.0, 2.0, 3.0]
    # CHECK: =====
    # CHECK: th_id 3
    # CHECK: [0.0, 1.0, 2.0, 3.0]
    # CHECK: =====
    # CHECK: th_id 4
    # CHECK: [4.0, 5.0, 6.0, 7.0]
    # CHECK: =====
    # CHECK: th_id 5
    # CHECK: [4.0, 5.0, 6.0, 7.0]
    # CHECK: =====
    # CHECK: th_id 6
    # CHECK: [4.0, 5.0, 6.0, 7.0]
    # CHECK: =====
    # CHECK: th_id 7
    # CHECK: [4.0, 5.0, 6.0, 7.0]
    # CHECK: =====
    # CHECK: th_id 8
    # CHECK: [8.0, 9.0, 10.0, 11.0]
    # CHECK: =====
    # CHECK: th_id 9
    # CHECK: [8.0, 9.0, 10.0, 11.0]
    # CHECK: =====
    # CHECK: th_id 10
    # CHECK: [8.0, 9.0, 10.0, 11.0]
    # CHECK: =====
    # CHECK: th_id 11
    # CHECK: [8.0, 9.0, 10.0, 11.0]
    # CHECK: =====
    # CHECK: th_id 12
    # CHECK: [12.0, 13.0, 14.0, 15.0]
    # CHECK: =====
    # CHECK: th_id 13
    # CHECK: [12.0, 13.0, 14.0, 15.0]
    # CHECK: =====
    # CHECK: th_id 14
    # CHECK: [12.0, 13.0, 14.0, 15.0]
    # CHECK: =====
    # CHECK: th_id 15
    # CHECK: [12.0, 13.0, 14.0, 15.0]
    for th_id in range(16):
        print("th_id", th_id)
        var tensor = tensor_4x4.vectorize[1, 4]().distribute[
            Layout.row_major(4, 4), axis=0
        ](th_id)
        print(tensor)
        print("=====")

    # CHECK: th_id 0
    # CHECK: [0.0, 4.0, 8.0, 12.0]
    # CHECK: =====
    # CHECK: th_id 1
    # CHECK: [1.0, 5.0, 9.0, 13.0]
    # CHECK: =====
    # CHECK: th_id 2
    # CHECK: [2.0, 6.0, 10.0, 14.0]
    # CHECK: =====
    # CHECK: th_id 3
    # CHECK: [3.0, 7.0, 11.0, 15.0]
    # CHECK: =====
    # CHECK: th_id 4
    # CHECK: [0.0, 4.0, 8.0, 12.0]
    # CHECK: =====
    # CHECK: th_id 5
    # CHECK: [1.0, 5.0, 9.0, 13.0]
    # CHECK: =====
    # CHECK: th_id 6
    # CHECK: [2.0, 6.0, 10.0, 14.0]
    # CHECK: =====
    # CHECK: th_id 7
    # CHECK: [3.0, 7.0, 11.0, 15.0]
    # CHECK: =====
    # CHECK: th_id 8
    # CHECK: [0.0, 4.0, 8.0, 12.0]
    # CHECK: =====
    # CHECK: th_id 9
    # CHECK: [1.0, 5.0, 9.0, 13.0]
    # CHECK: =====
    # CHECK: th_id 10
    # CHECK: [2.0, 6.0, 10.0, 14.0]
    # CHECK: =====
    # CHECK: th_id 11
    # CHECK: [3.0, 7.0, 11.0, 15.0]
    # CHECK: =====
    # CHECK: th_id 12
    # CHECK: [0.0, 4.0, 8.0, 12.0]
    # CHECK: =====
    # CHECK: th_id 13
    # CHECK: [1.0, 5.0, 9.0, 13.0]
    # CHECK: =====
    # CHECK: th_id 14
    # CHECK: [2.0, 6.0, 10.0, 14.0]
    # CHECK: =====
    # CHECK: th_id 15
    # CHECK: [3.0, 7.0, 11.0, 15.0]
    for th_id in range(16):
        print("th_id", th_id)
        var tensor = tensor_4x4.vectorize[4, 1]().distribute[
            Layout.row_major(4, 4), axis=1
        ](th_id)
        print(tensor)
        print("=====")


def test_split():
    var tensor_4x4 = LayoutTensor[
        .float32, Layout(IntTuple(4, 4), IntTuple(4, 1)), MutAnyOrigin
    ].stack_allocation[stack_alignment=16]()
    arange(tensor_4x4)

    var tiles_axis0 = tensor_4x4.split[2]()
    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 4.0 5.0 6.0 7.0
    print(tiles_axis0[0])
    # CHECK: 8.0 9.0 10.0 11.0
    # CHECK: 12.0 13.0 14.0 15.0
    print(tiles_axis0[1])

    var tiles_axis1 = tensor_4x4.split[2, axis=1]()
    # CHECK: 0.0 1.0
    # CHECK: 4.0 5.0
    # CHECK: 8.0 9.0
    # CHECK: 12.0 13.0
    print(tiles_axis1[0])
    # CHECK: 2.0 3.0
    # CHECK: 6.0 7.0
    # CHECK: 10.0 11.0
    # CHECK: 14.0 15.0
    print(tiles_axis1[1])

    var tiles_vec2_axis0 = tensor_4x4.vectorize[1, 2]().split[2]()
    # CHECK: [0.0, 1.0] [2.0, 3.0]
    # CHECK: [4.0, 5.0] [6.0, 7.0]
    print(tiles_vec2_axis0[0])
    # CHECK: [8.0, 9.0] [10.0, 11.0]
    # CHECK: [12.0, 13.0] [14.0, 15.0]
    print(tiles_vec2_axis0[1])

    _ = tensor_4x4


# DISABLED-CHECK-LABEL: test_copy_subtiles_scalars
# def test_copy_subtiles_scalars():
#    print("== test_copy_subtiles_scalars")
#    var tensor_13x7 = LayoutTensor[
#        .float32, Layout.row_major(13, 7)
#    ].stack_allocation[stack_alignment=16]()
#    arange(tensor_13x7)
#    print(tensor_13x7)
#
#    alias tile_m_size = 4
#    alias tile_n_size = 2
#
#    # DISABLED-CHECK: ----tile-data[ 0 , 0 ]----
#    # DISABLED-CHECK: 0.0 1.0
#    # DISABLED-CHECK: 7.0 8.0
#    # DISABLED-CHECK: 14.0 15.0
#    # DISABLED-CHECK: 21.0 22.0
#    # DISABLED-CHECK: ----tile-data[ 0 , 1 ]----
#    # DISABLED-CHECK: 2.0 3.0
#    # DISABLED-CHECK: 9.0 10.0
#    # DISABLED-CHECK: 16.0 17.0
#    # DISABLED-CHECK: 23.0 24.0
#    # DISABLED-CHECK: ----tile-data[ 0 , 2 ]----
#    # DISABLED-CHECK: 4.0 5.0
#    # DISABLED-CHECK: 11.0 12.0
#    # DISABLED-CHECK: 18.0 19.0
#    # DISABLED-CHECK: 25.0 26.0
#    # DISABLED-CHECK: ----tile-data[ 0 , 3 ]----
#    # DISABLED-CHECK: 6.0 0.0
#    # DISABLED-CHECK: 13.0 0.0
#    # DISABLED-CHECK: 20.0 0.0
#    # DISABLED-CHECK: 27.0 0.0
#    # DISABLED-CHECK: ----tile-data[ 1 , 0 ]----
#    # DISABLED-CHECK: 28.0 29.0
#    # DISABLED-CHECK: 35.0 36.0
#    # DISABLED-CHECK: 42.0 43.0
#    # DISABLED-CHECK: 49.0 50.0
#    # DISABLED-CHECK: ----tile-data[ 1 , 1 ]----
#    # DISABLED-CHECK: 30.0 31.0
#    # DISABLED-CHECK: 37.0 38.0
#    # DISABLED-CHECK: 44.0 45.0
#    # DISABLED-CHECK: 51.0 52.0
#    # DISABLED-CHECK: ----tile-data[ 1 , 2 ]----
#    # DISABLED-CHECK: 32.0 33.0
#    # DISABLED-CHECK: 39.0 40.0
#    # DISABLED-CHECK: 46.0 47.0
#    # DISABLED-CHECK: 53.0 54.0
#    # DISABLED-CHECK: ----tile-data[ 1 , 3 ]----
#    # DISABLED-CHECK: 34.0 0.0
#    # DISABLED-CHECK: 41.0 0.0
#    # DISABLED-CHECK: 48.0 0.0
#    # DISABLED-CHECK: 55.0 0.0
#    # DISABLED-CHECK: ----tile-data[ 2 , 0 ]----
#    # DISABLED-CHECK: 56.0 57.0
#    # DISABLED-CHECK: 63.0 64.0
#    # DISABLED-CHECK: 70.0 71.0
#    # DISABLED-CHECK: 77.0 78.0
#    # DISABLED-CHECK: ----tile-data[ 2 , 1 ]----
#    # DISABLED-CHECK: 58.0 59.0
#    # DISABLED-CHECK: 65.0 66.0
#    # DISABLED-CHECK: 72.0 73.0
#    # DISABLED-CHECK: 79.0 80.0
#    # DISABLED-CHECK: ----tile-data[ 2 , 2 ]----
#    # DISABLED-CHECK: 60.0 61.0
#    # DISABLED-CHECK: 67.0 68.0
#    # DISABLED-CHECK: 74.0 75.0
#    # DISABLED-CHECK: 81.0 82.0
#    # DISABLED-CHECK: ----tile-data[ 2 , 3 ]----
#    # DISABLED-CHECK: 62.0 0.0
#    # DISABLED-CHECK: 69.0 0.0
#    # DISABLED-CHECK: 76.0 0.0
#    # DISABLED-CHECK: 83.0 0.0
#    # DISABLED-CHECK: ----tile-data[ 3 , 0 ]----
#    # DISABLED-CHECK: 84.0 85.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: ----tile-data[ 3 , 1 ]----
#    # DISABLED-CHECK: 86.0 87.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: ----tile-data[ 3 , 2 ]----
#    # DISABLED-CHECK: 88.0 89.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: ----tile-data[ 3 , 3 ]----
#    # DISABLED-CHECK: 90.0 0.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0
#    for tile_m in range(ceildiv(13, tile_m_size)):
#        for tile_n in range(ceildiv(7, tile_n_size)):
#            var tile_4x2 = tensor_13x7.tile[tile_m_size, tile_n_size](
#                tile_m, tile_n
#            )
#            print("----tile-data[", tile_m, ",", tile_n, "]----")
#            var tile_4x2_cache = LayoutTensor[
#                .float32, Layout.row_major(tile_m_size, tile_n_size)
#            ].stack_allocation[stack_alignment=16]().fill(0)
#            tile_4x2_cache.copy_from[
#                dst_coords_bound = rebind[
#                    IndexList[tile_4x2_cache.layout.rank()]
#                ](IndexList[2](13, 7))
#            ](tile_4x2)
#            print(tile_4x2_cache)


# DISABLED-CHECK-LABEL: test_copy_distributed_subtiles_scalars
# def test_copy_distributed_subtiles_scalars():
#    print("== test_copy_distributed_subtiles_scalars")
#    var tensor_13x7 = LayoutTensor[
#        .float32, Layout.row_major(13, 7)
#    ].stack_allocation[stack_alignment=16]()
#    arange(tensor_13x7)
#
#    alias tile_m_size = 4
#    alias tile_n_size = 4
#
#    # DISABLED-CHECK: ----tile-data[ 0 , 0 ]----
#    # DISABLED-CHECK: 0.0 1.0 2.0 3.0
#    # DISABLED-CHECK: 7.0 8.0 9.0 10.0
#    # DISABLED-CHECK: 14.0 15.0 16.0 17.0
#    # DISABLED-CHECK: 21.0 22.0 23.0 24.0
#    # DISABLED-CHECK: ----fragments-data[ 0 ]----
#    # DISABLED-CHECK: 0.0 2.0
#    # DISABLED-CHECK: 14.0 16.0
#    # DISABLED-CHECK: ----fragments-data[ 1 ]----
#    # DISABLED-CHECK: 1.0 3.0
#    # DISABLED-CHECK: 15.0 17.0
#    # DISABLED-CHECK: ----fragments-data[ 2 ]----
#    # DISABLED-CHECK: 7.0 9.0
#    # DISABLED-CHECK: 21.0 23.0
#    # DISABLED-CHECK: ----fragments-data[ 3 ]----
#    # DISABLED-CHECK: 8.0 10.0
#    # DISABLED-CHECK: 22.0 24.0
#    # DISABLED-CHECK: ----tile-data[ 0 , 1 ]----
#    # DISABLED-CHECK: 4.0 5.0 6.0 0.0
#    # DISABLED-CHECK: 11.0 12.0 13.0 0.0
#    # DISABLED-CHECK: 18.0 19.0 20.0 0.0
#    # DISABLED-CHECK: 25.0 26.0 27.0 0.0
#    # DISABLED-CHECK: ----fragments-data[ 0 ]----
#    # DISABLED-CHECK: 4.0 6.0
#    # DISABLED-CHECK: 18.0 20.0
#    # DISABLED-CHECK: ----fragments-data[ 1 ]----
#    # DISABLED-CHECK: 5.0 0.0
#    # DISABLED-CHECK: 19.0 0.0
#    # DISABLED-CHECK: ----fragments-data[ 2 ]----
#    # DISABLED-CHECK: 11.0 13.0
#    # DISABLED-CHECK: 25.0 27.0
#    # DISABLED-CHECK: ----fragments-data[ 3 ]----
#    # DISABLED-CHECK: 12.0 0.0
#    # DISABLED-CHECK: 26.0 0.0
#    # DISABLED-CHECK: ----tile-data[ 1 , 0 ]----
#    # DISABLED-CHECK: 28.0 29.0 30.0 31.0
#    # DISABLED-CHECK: 35.0 36.0 37.0 38.0
#    # DISABLED-CHECK: 42.0 43.0 44.0 45.0
#    # DISABLED-CHECK: 49.0 50.0 51.0 52.0
#    # DISABLED-CHECK: ----fragments-data[ 0 ]----
#    # DISABLED-CHECK: 28.0 30.0
#    # DISABLED-CHECK: 42.0 44.0
#    # DISABLED-CHECK: ----fragments-data[ 1 ]----
#    # DISABLED-CHECK: 29.0 31.0
#    # DISABLED-CHECK: 43.0 45.0
#    # DISABLED-CHECK: ----fragments-data[ 2 ]----
#    # DISABLED-CHECK: 35.0 37.0
#    # DISABLED-CHECK: 49.0 51.0
#    # DISABLED-CHECK: ----fragments-data[ 3 ]----
#    # DISABLED-CHECK: 36.0 38.0
#    # DISABLED-CHECK: 50.0 52.0
#    # DISABLED-CHECK: ----tile-data[ 1 , 1 ]----
#    # DISABLED-CHECK: 32.0 33.0 34.0 0.0
#    # DISABLED-CHECK: 39.0 40.0 41.0 0.0
#    # DISABLED-CHECK: 46.0 47.0 48.0 0.0
#    # DISABLED-CHECK: 53.0 54.0 55.0 0.0
#    # DISABLED-CHECK: ----fragments-data[ 0 ]----
#    # DISABLED-CHECK: 32.0 34.0
#    # DISABLED-CHECK: 46.0 48.0
#    # DISABLED-CHECK: ----fragments-data[ 1 ]----
#    # DISABLED-CHECK: 33.0 0.0
#    # DISABLED-CHECK: 47.0 0.0
#    # DISABLED-CHECK: ----fragments-data[ 2 ]----
#    # DISABLED-CHECK: 39.0 41.0
#    # DISABLED-CHECK: 53.0 55.0
#    # DISABLED-CHECK: ----fragments-data[ 3 ]----
#    # DISABLED-CHECK: 40.0 0.0
#    # DISABLED-CHECK: 54.0 0.0
#    # DISABLED-CHECK: ----tile-data[ 2 , 0 ]----
#    # DISABLED-CHECK: 56.0 57.0 58.0 59.0
#    # DISABLED-CHECK: 63.0 64.0 65.0 66.0
#    # DISABLED-CHECK: 70.0 71.0 72.0 73.0
#    # DISABLED-CHECK: 77.0 78.0 79.0 80.0
#    # DISABLED-CHECK: ----fragments-data[ 0 ]----
#    # DISABLED-CHECK: 56.0 58.0
#    # DISABLED-CHECK: 70.0 72.0
#    # DISABLED-CHECK: ----fragments-data[ 1 ]----
#    # DISABLED-CHECK: 57.0 59.0
#    # DISABLED-CHECK: 71.0 73.0
#    # DISABLED-CHECK: ----fragments-data[ 2 ]----
#    # DISABLED-CHECK: 63.0 65.0
#    # DISABLED-CHECK: 77.0 79.0
#    # DISABLED-CHECK: ----fragments-data[ 3 ]----
#    # DISABLED-CHECK: 64.0 66.0
#    # DISABLED-CHECK: 78.0 80.0
#    # DISABLED-CHECK: ----tile-data[ 2 , 1 ]----
#    # DISABLED-CHECK: 60.0 61.0 62.0 0.0
#    # DISABLED-CHECK: 67.0 68.0 69.0 0.0
#    # DISABLED-CHECK: 74.0 75.0 76.0 0.0
#    # DISABLED-CHECK: 81.0 82.0 83.0 0.0
#    # DISABLED-CHECK: ----fragments-data[ 0 ]----
#    # DISABLED-CHECK: 60.0 62.0
#    # DISABLED-CHECK: 74.0 76.0
#    # DISABLED-CHECK: ----fragments-data[ 1 ]----
#    # DISABLED-CHECK: 61.0 0.0
#    # DISABLED-CHECK: 75.0 0.0
#    # DISABLED-CHECK: ----fragments-data[ 2 ]----
#    # DISABLED-CHECK: 67.0 69.0
#    # DISABLED-CHECK: 81.0 83.0
#    # DISABLED-CHECK: ----fragments-data[ 3 ]----
#    # DISABLED-CHECK: 68.0 0.0
#    # DISABLED-CHECK: 82.0 0.0
#    # DISABLED-CHECK: ----tile-data[ 3 , 0 ]----
#    # DISABLED-CHECK: 84.0 85.0 86.0 87.0
#    # DISABLED-CHECK: 0.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0 0.0 0.0
#    # DISABLED-CHECK: ----fragments-data[ 0 ]----
#    # DISABLED-CHECK: 84.0 86.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: ----fragments-data[ 1 ]----
#    # DISABLED-CHECK: 85.0 87.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: ----fragments-data[ 2 ]----
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: ----fragments-data[ 3 ]----
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: ----tile-data[ 3 , 1 ]----
#    # DISABLED-CHECK: 88.0 89.0 90.0 0.0
#    # DISABLED-CHECK: 0.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0 0.0 0.0
#    # DISABLED-CHECK: ----fragments-data[ 0 ]----
#    # DISABLED-CHECK: 88.0 90.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: ----fragments-data[ 1 ]----
#    # DISABLED-CHECK: 89.0 0.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: ----fragments-data[ 2 ]----
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: ----fragments-data[ 3 ]----
#    # DISABLED-CHECK: 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0
#
#    for tile_m in range(ceildiv(13, tile_m_size)):
#        for tile_n in range(ceildiv(7, tile_n_size)):
#            print("----tile-data[", tile_m, ",", tile_n, "]----")
#            var tile_4x4 = tensor_13x7.tile[tile_m_size, tile_n_size](
#                tile_m, tile_n
#            )
#            var tile_4x4_cache = LayoutTensor[
#                .float32, Layout.row_major(tile_m_size, tile_n_size)
#            ].stack_allocation[stack_alignment=16]().fill(0)
#            tile_4x4_cache.copy_from[
#                dst_coords_bound = rebind[
#                    IndexList[tile_4x4_cache.layout.rank()]
#                ](IndexList[2](13, 7))
#            ](tile_4x4)
#            print(tile_4x4_cache)
#
#            for th_id in range(4):
#                print("----fragments-data[", th_id, "]----")
#                var tile_2x2 = tile_4x4.distribute[Layout.row_major(2, 2)](
#                    th_id
#                )
#                var tile_2x2_cache = LayoutTensor[
#                    .float32, Layout.row_major(2, 2)
#                ].stack_allocation[stack_alignment=16]().fill(0)
#                tile_2x2_cache.copy_from[
#                    dst_coords_bound = rebind[
#                        IndexList[tile_2x2_cache.layout.rank()]
#                    ](IndexList[2](13, 7))
#                ](tile_2x2)
#                print(tile_2x2_cache)
#


def test_copy_subtiles_scalars_back():
    print("== test_copy_subtiles_scalars_back")

    var tensor_13x7 = (
        LayoutTensor[.float32, Layout.row_major(13, 7), MutAnyOrigin]
        .stack_allocation[stack_alignment=16]()
        .fill(-1)
    )

    comptime tile_m_size = 4
    comptime tile_n_size = 4

    # TODO(#38547) re-enable the checks when the non-deterministic behavior is addressed.
    # CHECK-FIXME: ----tile-data[ 0 , 0 ]----
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: ----tile-data[ 0 , 1 ]----
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: ----tile-data[ 1 , 0 ]----
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: ----tile-data[ 1 , 1 ]----
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: ----tile-data[ 2 , 0 ]----
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: ----tile-data[ 2 , 1 ]----
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: ----tile-data[ 3 , 0 ]----
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: ----tile-data[ 3 , 1 ]----
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0

    for tile_m in range(ceildiv(13, tile_m_size)):
        for tile_n in range(ceildiv(7, tile_n_size)):
            print("----tile-data[", tile_m, ",", tile_n, "]----")
            var tensor_4x4 = tensor_13x7.tile[tile_m_size, tile_n_size](
                tile_m, tile_n
            )
            var tile_4x4_cache = LayoutTensor[
                .float32,
                Layout.row_major(tile_m_size, tile_n_size),
                MutAnyOrigin,
            ].stack_allocation[stack_alignment=16]()
            arange(tile_4x4_cache)
            tensor_4x4.copy_from(tile_4x4_cache)
            print(tensor_13x7)


# CHECK-LABEL: test_slice_with_offsets
def test_slice_with_offsets():
    print("== test_slice_with_offsets")

    var tensor_4x3x2_row_major = LayoutTensor[
        .float32, Layout.row_major(4, 3, 2), MutAnyOrigin
    ].stack_allocation[stack_alignment=16]()

    for i in range(4 * 3 * 2):
        tensor_4x3x2_row_major.ptr[i] = Float32(i)

    # CHECK: slice-of[0:3,:2,0]
    # CHECK: 0.0 2.0
    # CHECK: 6.0 8.0
    # CHECK: 12.0 14.0
    print("slice-of[0:3,:2,0]")
    print(
        tensor_4x3x2_row_major.slice[0:3, 0:2, slice_indices=(0, 1)](
            IndexList[1](0)
        )
    )

    # CHECK: slice-of-[0:3,:2,1]
    # CHECK: 1.0 3.0
    # CHECK: 7.0 9.0
    # CHECK: 13.0 15.0
    print("slice-of-[0:3,:2,1]")
    print(
        tensor_4x3x2_row_major.slice[0:3, 0:2, slice_indices=(0, 1)](
            IndexList[1](1)
        )
    )

    # CHECK: slice-of-[2,:,:]
    # CHECK: 12.0 13.0
    # CHECK: 14.0 15.0
    # CHECK: 16.0 17.0
    print("slice-of-[2,:,:]")
    print(
        tensor_4x3x2_row_major.slice[:, :, slice_indices=(1, 2)](
            IndexList[1](2)
        )
    )

    print("slice-of-[:,1,:]")
    # CHECK: slice-of-[:,1,:]
    # CHECK: 2.0 3.0
    # CHECK: 8.0 9.0
    # CHECK: 14.0 15.0
    # CHECK: 20.0 21.0
    print(
        tensor_4x3x2_row_major.slice[:, :, slice_indices=(0, 2)](
            IndexList[1](1)
        )
    )

    # CHECK: slice-of-[:,0,0]
    # CHECK: 0.0
    # CHECK: 6.0
    # CHECK: 12.0
    # CHECK: 18.0
    print("slice-of-[:,0,0]")
    print(
        tensor_4x3x2_row_major.slice_1d[:, slice_indices=IndexList[1](0)](
            IndexList[2](0, 0)
        )
    )

    # CHECK: slice-of-[2,:,1]
    # CHECK: 13.0
    # CHECK: 15.0
    # CHECK: 17.0
    print("slice-of-[2,:,1]")
    print(
        tensor_4x3x2_row_major.slice_1d[:, slice_indices=IndexList[1](1)](
            IndexList[2](2, 1)
        )
    )


# CHECK-LABEL: test_layout_tensor_iterator
def test_layout_tensor_iterator():
    print("== test_layout_tensor_iterator")

    comptime size = 64
    comptime type = DType.float32

    var arr = Array[Scalar[type], size](
        fill_with=lambda (i: Int) -> Scalar[type]: Float32(i)
    )

    comptime layout_2x2_8x1 = Layout(IntTuple(2, 2), IntTuple(8, 1))

    # Non circular iterator.
    # CHECK: 0.0 1.0
    # CHECK: 8.0 9.0
    # CHECK: 4.0 5.0
    # CHECK: 12.0 13.0
    var iter2x2 = LayoutTensorIter[type, layout_2x2_8x1](arr.unsafe_ptr(), size)
    for _ in range(2):
        print(iter2x2.get())
        iter2x2 += 1

    # Non circular iterator with stride
    # CHECK: 0.0 1.0
    # CHECK: 8.0 9.0
    # CHECK: 16.0 17.0
    # CHECK: 24.0 25.0
    iter2x2 = LayoutTensorIter[type, layout_2x2_8x1](
        arr.unsafe_ptr(), size, stride=16
    )
    for _ in range(2):
        print(iter2x2.get())
        iter2x2 += 1

    # Non circular iterator with offset and stride
    # CHECK: 4.0 5.0
    # CHECK: 12.0 13.0
    # CHECK: 20.0 21.0
    # CHECK: 28.0 29.0
    iter2x2 = LayoutTensorIter[type, layout_2x2_8x1](
        arr.unsafe_ptr(), size, stride=16, offset=4
    )
    for _ in range(2):
        print(iter2x2.get())
        iter2x2 += 1

    # Circular iterator with offset and stride
    # CHECK: 32.0 33.0
    # CHECK: 40.0 41.0
    # CHECK: 48.0 49.0
    # CHECK: 56.0 57.0
    # CHECK: 0.0 1.0
    # CHECK: 8.0 9.0
    # CHECK: 16.0 17.0
    # CHECK: 24.0 25.0
    var iter2x2_circular = LayoutTensorIter[
        type, layout_2x2_8x1, circular=True
    ](arr.unsafe_ptr(), size, stride=16, offset=32)
    for _ in range(4):
        print(iter2x2_circular.get())
        iter2x2_circular += 1

    # Tiled iterator.
    var tensor = LayoutTensor[type, Layout.row_major(8, 8)](arr)
    # CHECK: 32.0 33.0
    # CHECK: 40.0 41.0
    # CHECK: 34.0 35.0
    # CHECK: 42.0 43.0
    # CHECK: 36.0 37.0
    # CHECK: 44.0 45.0
    # CHECK: 38.0 39.0
    # CHECK: 46.0 47.0
    var iter_axis1 = tensor.tiled_iterator[2, 2, axis=1](2, 0)
    for _ in range(4):
        print(iter_axis1.get())
        iter_axis1 += 1
    # CHECK: 38.0 39.0
    # CHECK: 46.0 47.0
    # CHECK: 54.0 55.0
    # CHECK: 62.0 63.0
    var iter_axis0 = tensor.tiled_iterator[2, 2, axis=0](2, 3)
    for _ in range(2):
        print(iter_axis0.get())
        iter_axis0 += 1

    # Reshape iterator.
    # CHECK: 12.0 13.0
    # CHECK: 14.0 15.0
    # CHECK: 16.0 17.0
    var iter2x3 = LayoutTensorIter[type, Layout.row_major(2, 3)](
        arr.unsafe_ptr(), size
    )
    iter2x3 += 1
    var iter3x2 = iter2x3.reshape[Layout.row_major(3, 2)]()
    iter3x2 += 1
    print(iter3x2[])


# CHECK-LABEL: test_nested_layout_tensor_iterator
def test_nested_layout_tensor_iterator():
    print("== test_nested_layout_tensor_iterator")
    comptime N = 128
    comptime K = 8

    comptime size = N * K
    comptime type = DType.float32

    var arr = Array[Scalar[type], size](
        fill_with=lambda (i: Int) -> Scalar[type]: Float32(i)
    )

    # Here we define a float32 tensor (64 * TN, 2 * TK):
    #              K
    #     0       2      4
    #     0+------+------+-----+
    #      | 64x2 | 64x2 | ... |
    #      | tile | tile |     |
    # N  64+------+------+-----+    float32 Matrix
    #      | 64x2 | 64x2 | ... |
    #      | tile | tile |     |
    #   128+------+------+-----+
    #
    # Elements within each tile are stored continuously
    #                  K
    #      0     1     2     3     4
    #     0+-----+-----+-----+-----+
    #      |  0  |  1  | 128 | 129 |
    #     1+-----+-----+-----+-----+
    #      |  2  |  3  | 130 | 131 |
    #     2+-----+-----+-----+-----+
    #      |  4  |  5  | 132 | 133 |
    # N   3+-----+-----+-----+-----+
    #      | ... | ... | ... | ... |
    #    64+-----+-----+-----+-----+
    #      |TK*  |TK*  | ... | ... |
    #      |128  |128+1| ... | ... |
    #    65+-----+-----+-----+-----+
    #      |TK*  |TK*  | ... | ... |
    #      |128+2|128+3| ... | ... |
    #    65+-----+-----+-----+-----+
    # This data layout can be expressed by UInt32 LayoutTensor
    # with shape = IntTuple(IntTuple(64, TN),IntTuple(2, TK))
    # and stride = IntTuple(IntTuple(2, TK * 128),IntTuple(1, 128))

    comptime nested_layout = Layout(
        IntTuple(
            IntTuple(64, N // 64),
            IntTuple(2, K // 2),
        ),
        IntTuple(
            IntTuple(2, 128 * K // 2),
            IntTuple(1, 128),
        ),
    )

    # View a row in plain_tensor as an array of [64, 2] tiles.
    var nested_tensor = LayoutTensor[
        .float32,
        nested_layout,
    ](arr)

    var tiled_nested_tensor_iter = nested_tensor.tiled_iterator[64, 2, axis=1](
        0, 0
    )

    # CHECK: 0.0 1.0
    print(tiled_nested_tensor_iter[][0, 0], tiled_nested_tensor_iter[][0, 1])
    # CHECK: 2.0 3.0
    print(tiled_nested_tensor_iter[][1, 0], tiled_nested_tensor_iter[][1, 1])

    # each tile are stored continuously
    tiled_nested_tensor_iter._incr()

    # CHECK: 128.0 129.0
    print(tiled_nested_tensor_iter[][0, 0], tiled_nested_tensor_iter[][0, 1])
    # CHECK: 130.0 131.0
    print(tiled_nested_tensor_iter[][1, 0], tiled_nested_tensor_iter[][1, 1])


# DISABLED-CHECK-LABEL: test_copy_from_bigger_tensor
# def test_copy_from_bigger_tensor():
#    print("== test_copy_from_bigger_tensor")
#    var tensor_5x7 = LayoutTensor[
#        .float32, Layout.row_major(8, 8)
#    ].stack_allocation().fill(0)
#
#    var tensor_8x8 = LayoutTensor[
#        .float32, Layout.row_major(8, 8)
#    ].stack_allocation()
#    arange(tensor_8x8)
#
#    tensor_5x7.copy_from[
#        rebind[IndexList[tensor_5x7.layout.rank()]](IndexList[2](5, 7))
#    ](tensor_8x8)
#    # DISABLED-CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 0.0
#    # DISABLED-CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 0.0
#    # DISABLED-CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 0.0
#    # DISABLED-CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 0.0
#    # DISABLED-CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 0.0
#    # DISABLED-CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#    print(tensor_5x7)


# DISABLED-CHECK-LABEL: test_copy_from_smaller_tensor
# def test_copy_from_smaller_tensor():
#     print("== test_copy_from_smaller_tensor")
#     var tensor_5x7 = LayoutTensor[
#         .float32, Layout.row_major(8, 8)
#     ].stack_allocation()
#     arange(tensor_5x7)
#
#     var tensor_8x8 = LayoutTensor[
#         .float32, Layout.row_major(8, 8)
#     ].stack_allocation().fill(0)
#
#     tensor_8x8.copy_from[
#         rebind[IndexList[tensor_8x8.layout.rank()]](IndexList[2](5, 7))
#     ](tensor_5x7)
#     # DISABLED-CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 0.0
#     # DISABLED-CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 0.0
#     # DISABLED-CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 0.0
#     # DISABLED-CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 0.0
#     # DISABLED-CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 0.0
#     # DISABLED-CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#     # DISABLED-CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#     # DISABLED-CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#     print(tensor_8x8)


# DISABLED-CHECK-LABEL: test_copy_from_vectorized_masked_write
# def test_copy_from_vectorized_masked_write():
#    print("== test_copy_from_vectorized_masked_write")
#
#    var tensor_4x8 = LayoutTensor[
#        .float32, Layout.row_major(4, 8)
#    ].stack_allocation()
#    arange(tensor_4x8)
#
#    var tensor_8x8 = LayoutTensor[
#        .float32, Layout.row_major(8, 8)
#    ].stack_allocation()
#    arange(tensor_8x8)
#
#    var tensor_8x8_data = LayoutTensor[
#        .float32, Layout.row_major(8, 8)
#    ].stack_allocation()
#
#    var tensor_1x5 = LayoutTensor[.float32, Layout.row_major(1, 5)](
#        tensor_8x8_data.ptr
#    )
#
#    _ = tensor_8x8_data.fill(-1)
#
#    var tensor_1x8 = LayoutTensor[
#        .float32, Layout.row_major(1, 8)
#    ].stack_allocation()
#    arange(tensor_1x8)
#
#    var tensor_1x5_v1_4 = tensor_1x5.vectorize[1, 4]()
#    tensor_1x5_v1_4.copy_from[
#        dst_coords_bound = rebind[IndexList[tensor_1x5_v1_4.layout.rank()]](
#            IndexList[2](1, 5)
#        )
#    ](tensor_1x8.vectorize[1, 4]())
#
#    # DISABLED-CHECK: write-1x5:
#    # DISABLED-CHECK: 0.0 1.0 2.0 3.0 4.0 -1.0 -1.0 -1.0
#    # DISABLED-CHECK: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
#    # DISABLED-CHECK: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
#    # DISABLED-CHECK: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
#    # DISABLED-CHECK: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
#    # DISABLED-CHECK: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
#    # DISABLED-CHECK: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
#    # DISABLED-CHECK: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
#    print("write-1x5:")
#    print(tensor_8x8_data)
#
#    var tensor_3x8 = LayoutTensor[.float32, Layout.row_major(3, 8)](
#        tensor_8x8_data.ptr
#    )
#
#    _ = tensor_8x8_data.fill(-1)
#
#    var tensor_3x8_v_4_4 = tensor_3x8.vectorize[4, 4]()
#    tensor_3x8_v_4_4.copy_from[
#        dst_coords_bound = rebind[IndexList[tensor_3x8_v_4_4.layout.rank()]](
#            IndexList[2](3, 8)
#        )
#    ](tensor_4x8.vectorize[4, 4]())
#
#    # DISABLED-CHECK: write-3x8:
#    # DISABLED-CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
#    # DISABLED-CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
#    # DISABLED-CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
#    # DISABLED-CHECK: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
#    # DISABLED-CHECK: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
#    # DISABLED-CHECK: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
#    # DISABLED-CHECK: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
#    # DISABLED-CHECK: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
#    print("write-3x8:")
#    print(tensor_8x8_data)
#
#    var tensor_5x8 = LayoutTensor[.float32, Layout.row_major(5, 8)](
#        tensor_8x8_data.ptr
#    )
#
#    _ = tensor_8x8_data.fill(-1)
#
#    var tensor5x8_v_4_1 = tensor_5x8.vectorize[4, 1]()
#    tensor5x8_v_4_1.copy_from[
#        dst_coords_bound = rebind[IndexList[tensor5x8_v_4_1.layout.rank()]](
#            IndexList[2](5, 8)
#        )
#    ](tensor_8x8.vectorize[4, 1]())
#
#    # DISABLED-CHECK: write-5x8:
#    # DISABLED-CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
#    # DISABLED-CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
#    # DISABLED-CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
#    # DISABLED-CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
#    # DISABLED-CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
#    # DISABLED-CHECK: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
#    # DISABLED-CHECK: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
#    # DISABLED-CHECK: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
#    print("write-5x8:")
#    print(tensor_8x8_data)
#


# def test_copy_from_vectorized_masked_read():
#    print("== test_copy_from_vectorized_masked_read")
#    var tensor_8x8 = LayoutTensor[
#        .float32, Layout.row_major(8, 8)
#    ].stack_allocation().fill(-1)
#
#    var tensor_8x5 = LayoutTensor[
#        .float32, Layout.row_major(8, 5)
#    ].stack_allocation()
#    arange(tensor_8x5)
#
#    var tensor_8x8_v_1_4 = tensor_8x8.vectorize[1, 4]()
#    tensor_8x8_v_1_4.copy_from[
#        src_coords_bound = rebind[IndexList[tensor_8x8_v_1_4.layout.rank()]](
#            IndexList[2](8, 5)
#        )
#    ](tensor_8x5.vectorize[1, 4]())
#
#    # DISABLED-CHECK: read-8x5:
#    # DISABLED-CHECK: 0.0 1.0 2.0 3.0 4.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 5.0 6.0 7.0 8.0 9.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 10.0 11.0 12.0 13.0 14.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 15.0 16.0 17.0 18.0 19.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 20.0 21.0 22.0 23.0 24.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 25.0 26.0 27.0 28.0 29.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 30.0 31.0 32.0 33.0 34.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 35.0 36.0 37.0 38.0 39.0 0.0 0.0 0.0
#    print("read-8x5:")
#    print(tensor_8x8)
#
#    var tensor_5x8 = LayoutTensor[
#        .float32, Layout.row_major(5, 8)
#    ].stack_allocation()
#    arange(tensor_5x8)
#
#    _ = tensor_8x8.fill(-1)
#    var tensor_8x8_v_4_1 = tensor_8x8.vectorize[4, 1]()
#    tensor_8x8_v_4_1.copy_from[
#        src_coords_bound = rebind[IndexList[tensor_8x8_v_4_1.layout.rank()]](
#            IndexList[2](5, 8)
#        )
#    ](tensor_5x8.vectorize[4, 1]())
#
#    # DISABLED-CHECK: read-5x8:
#    # DISABLED-CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
#    # DISABLED-CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
#    # DISABLED-CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
#    # DISABLED-CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
#    # DISABLED-CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
#    # DISABLED-CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#    print("read-5x8:")
#    print(tensor_8x8)
#
#    _ = tensor_8x8.fill(-1)
#    var tensor_8x8_v_4_4 = tensor_8x8.vectorize[4, 4]()
#    tensor_8x8_v_4_4.copy_from[
#        src_coords_bound = rebind[IndexList[tensor_8x8_v_4_4.layout.rank()]](
#            IndexList[2](5, 8)
#        )
#    ](tensor_5x8.vectorize[4, 4]())
#
#    # DISABLED-CHECK: read-5x8_v_4_4:
#    # DISABLED-CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
#    # DISABLED-CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
#    # DISABLED-CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
#    # DISABLED-CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
#    # DISABLED-CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
#    # DISABLED-CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#    # DISABLED-CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#    print("read-5x8_v_4_4:")
#    print(tensor_8x8)


# CHECK-LABEL: test_binary_math_ops
def test_binary_math_ops() raises:
    print("== test_binary_math_ops")

    var managed_tensor_a = ManagedLayoutTensor[
        .float32, Layout(IntTuple(8, 4))
    ]()
    var tensor_a = managed_tensor_a.tensor()
    arange(tensor_a, start=1)

    var managed_tensor_b = ManagedLayoutTensor[
        .float32, Layout(IntTuple(8, 4))
    ]()
    var tensor_b = managed_tensor_b.tensor()
    arange(tensor_b, start=32, step=-1)

    # CHECK: ----add matrix----
    # CHECK: 33.0 	33.0 	33.0 	33.0
    # CHECK: 33.0 	33.0 	33.0 	33.0
    # CHECK: 33.0 	33.0 	33.0 	33.0
    # CHECK: 33.0 	33.0 	33.0 	33.0
    # CHECK: 33.0 	33.0 	33.0 	33.0
    # CHECK: 33.0 	33.0 	33.0 	33.0
    # CHECK: 33.0 	33.0 	33.0 	33.0
    # CHECK: 33.0 	33.0 	33.0 	33.0
    print("----add matrix----")
    var add = tensor_a + tensor_b
    print_raw_major_tensor(add)

    # CHECK: ----sub matrix----
    # CHECK: 31.0 	29.0 	27.0 	25.0
    # CHECK: 23.0 	21.0 	19.0 	17.0
    # CHECK: 15.0 	13.0 	11.0 	9.0
    # CHECK: 7.0 	5.0 	3.0 	1.0
    # CHECK: -1.0 	-3.0 	-5.0 	-7.0
    # CHECK: -9.0 	-11.0 	-13.0 	-15.0
    # CHECK: -17.0 	-19.0 	-21.0 	-23.0
    # CHECK: -25.0 	-27.0 	-29.0 	-31.0
    print("----sub matrix----")
    var sub = tensor_b - tensor_a
    print_raw_major_tensor(sub)

    # CHECK: ----div matrix----
    # CHECK: 32.0    15.5    10.0    7.25
    # CHECK: 5.6     4.5     3.7142856       3.125
    # CHECK: 2.6666667       2.3     2.0     1.75
    # CHECK: 1.5384616       1.3571428       1.2     1.0625
    # CHECK: 0.9411765       0.8333333       0.7368421       0.65
    # CHECK: 0.5714286       0.5     0.4347826       0.375
    # CHECK: 0.32    0.26923078      0.22222222      0.17857143
    # CHECK: 0.13793103      0.1     0.06451613      0.03125
    print("----div matrix----")
    var div = tensor_b / tensor_a
    print_raw_major_tensor(div)

    # CHECK: ----mul matrix----
    # CHECK: 32.0 	62.0 	90.0 	116.0
    # CHECK: 140.0 	162.0 	182.0 	200.0
    # CHECK: 216.0 	230.0 	242.0 	252.0
    # CHECK: 260.0 	266.0 	270.0 	272.0
    # CHECK: 272.0 	270.0 	266.0 	260.0
    # CHECK: 252.0 	242.0 	230.0 	216.0
    # CHECK: 200.0 	182.0 	162.0 	140.0
    # CHECK: 116.0 	90.0 	62.0 	32.0
    print("----mul matrix----")
    var mul = tensor_b * tensor_a
    print_raw_major_tensor(mul)

    _ = managed_tensor_a^
    _ = managed_tensor_b^


def test_vectorized_tile() raises:
    var managed_tensor_a = ManagedLayoutTensor[
        .float32, Layout(IntTuple(8, 4))
    ]()
    var tensor_a = managed_tensor_a.tensor()
    var vt = tensor_a.vectorize[1, 2]().tile[4, 2](0, 0)
    _ = vt  # silence warning.
    assert_equal(Int(vt.layout.shape[0]), 4)
    assert_equal(Int(vt.layout.shape[1]), 2)
    assert_equal(Int(vt.element_layout.shape[0]), 1)
    assert_equal(Int(vt.element_layout.shape[1]), 2)


def test_nested_tile() raises:
    comptime base_layout = Layout.row_major(8, 8)
    comptime tiler_layout = Layout.row_major(2, 2)
    comptime layout = blocked_product(base_layout, tiler_layout)
    var managed_tensor_a = ManagedLayoutTensor[.float32, layout]()
    arange(managed_tensor_a.tensor())
    var tensor_a = managed_tensor_a.tensor()
    var vt = tensor_a.tile[4, 4](0, 0)
    for i in range(4):
        for j in range(4):
            assert_equal(vt[i, j], tensor_a[i, j])


def test_tensor_size() raises:
    comptime layout = Layout.row_major(4, 4)
    var stack = Array[UInt32, layout.size()](fill={})
    var tensor = LayoutTensor[.uint32, layout](stack)
    assert_equal(tensor.size(), 16)
    comptime layout2 = Layout.row_major(4, UNKNOWN_VALUE)
    var runtime_tensor = LayoutTensor[.uint32, layout2](
        stack,
        RuntimeLayout[layout2].row_major(IndexList[2](4, 4)),
    )
    assert_equal(runtime_tensor.size(), 16)


# This test doesn't need to run, it just needs to compile
def test_merge():
    comptime layout = Layout.row_major(4, 4)
    var stack = Array[UInt32, layout.size()](fill={})
    var tensor = LayoutTensor[.uint32, layout](stack)
    var stack2 = Array[UInt32, layout.size()](fill={})
    var tensor2 = LayoutTensor[.uint32, layout](stack2)
    var a = tensor if tensor.size() > 1 else tensor2
    print(a)


def test_flatten_vectorize() raises:
    """Regression test: flatten().vectorize[N]() must report correct dims
    and allow element access. Previously, the compile-time layout of the
    vectorized view had shape 0 (from UNKNOWN_VALUE / N = -1 / N = 0),
    causing bounds-check assertions to fire on valid accesses.
    """
    comptime W = 4
    var managed = ManagedLayoutTensor[.float32, Layout.row_major(W, W)]()
    var t = managed.tensor()
    for i in range(W * W):
        t.ptr[i] = Float32(i)
    var v = t.flatten().vectorize[W]()
    # The vectorized view should have dim[0] == W (not 0).
    assert_equal(v.dim[0](), W)
    assert_equal(v.size(), W)
    # Verify element reads produce correct values.
    assert_equal(v[0][0], Float32(0))
    assert_equal(v[0][3], Float32(3))
    assert_equal(v[3][0], Float32(12))
    assert_equal(v[3][3], Float32(15))


def main() raises:
    test_basic_tensor_ops()
    test_tesnsor_fragments()
    test_tensor_tile_and_distribute()
    test_tensor_tile_and_distribute_custom_layout()
    test_copy_to_tile_major_layout()
    test_distribute_tiled_layout()
    test_distribute_with_tile_size()
    test_vectorize_reads()
    test_vectorize_writes()
    test_slice()
    test_copy_vectorized()
    test_distribute_vectorized()
    test_distribute_axis_projection()
    test_split()
    # test_copy_subtiles_scalars()
    # test_copy_distributed_subtiles_scalars()
    # # TODO(#38547) re-enable the following test once the non-deterministic behavior is addressed.
    # # test_copy_subtiles_scalars_back()
    test_slice_with_offsets()
    test_layout_tensor_iterator()
    test_nested_layout_tensor_iterator()
    # test_element_coords_vectorized()
    # test_element_coords_tile_and_distribute()
    # test_element_coords_tiles_do_not_div()
    # test_copy_from_bigger_tensor()
    # test_copy_from_smaller_tensor()
    # test_copy_from_vectorized_masked_write()
    # test_copy_from_vectorized_masked_read()
    test_binary_math_ops()
    test_vectorized_tile()
    test_nested_tile()
    test_tensor_size()
    test_flatten_vectorize()
