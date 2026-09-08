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


from layout import Layout, LayoutTensor, RuntimeLayout
from layout._fillers import arange
from std.utils import Index


def test_tile_masked():
    print("== test_tile_masked")
    comptime layout = Layout.row_major(11, 7)
    var stack = Array[Float32, layout.size()](fill={})
    var tensor_11x7 = LayoutTensor[.float32, layout](stack)
    arange(tensor_11x7)
    for tile_i in range(3):
        for tile_j in range(2):
            print("--tile[", tile_i, ", ", tile_j, "]--")
            var tensor_4x4_masked = tensor_11x7.tile[4, 4](tile_i, tile_j)
            print(
                tensor_4x4_masked.runtime_layout,
                tensor_4x4_masked.masked,
                tensor_4x4_masked.runtime_layout.bound_check_required(),
            )
            print(tensor_4x4_masked)


def test_subtile_masked():
    print("== test_subtile_masked")
    comptime layout = Layout.row_major(15, 9)
    var stack = Array[Float32, layout.size()](fill={})
    var tensor_15x9 = LayoutTensor[.float32, layout](stack)
    arange(tensor_15x9)
    for tile_i in range(2):
        for tile_j in range(2):
            print("--tile[", tile_i, ", ", tile_j, "]--")
            var tensor_8x8_masked = tensor_15x9.tile[8, 8](tile_i, tile_j)
            print(
                tensor_8x8_masked.runtime_layout,
                tensor_8x8_masked.masked,
                tensor_8x8_masked.runtime_layout.bound_check_required(),
            )
            for tile_ii in range(2):
                for tile_jj in range(2):
                    print("--subtile[", tile_ii, ", ", tile_jj, "]--")
                    var subtile_4x4_masked = tensor_8x8_masked.tile[4, 4](
                        tile_ii, tile_jj
                    )
                    print(
                        subtile_4x4_masked.runtime_layout,
                        subtile_4x4_masked.masked,
                        subtile_4x4_masked.runtime_layout.bound_check_required(),
                    )
                    print(subtile_4x4_masked)


def test_tile_dynamic_no_bounds():
    print("== test_tile_dynamic_no_bounds")
    comptime layout = Layout.row_major(4, 4)
    var stack = Array[Float32, layout.size()](fill={})
    var tensor_UxU = LayoutTensor[.float32, Layout.row_major[2]()](
        stack, RuntimeLayout[Layout.row_major[2]()].row_major(Index(4, 4))
    )
    arange(tensor_UxU)
    for tile_i in range(2):
        for tile_j in range(2):
            var tensor_2x2_masked = tensor_UxU.tile[2, 2](tile_i, tile_j)
            print("--tile[", tile_i, ", ", tile_j, "]--")
            print(
                tensor_2x2_masked.runtime_layout,
                tensor_2x2_masked.masked,
                tensor_2x2_masked.runtime_layout.bound_check_required(),
            )
            print(tensor_2x2_masked)


def test_tile_dynamic_with_bounds():
    print("== test_tile_dynamic_with_bounds")
    comptime layout = Layout.row_major(5, 3)
    var stack = Array[Float32, layout.size()](fill={})
    var tensor_UxU = LayoutTensor[.float32, Layout.row_major[2]()](
        stack, RuntimeLayout[Layout.row_major[2]()].row_major(Index(5, 3))
    )
    arange(tensor_UxU)
    for tile_i in range(3):
        for tile_j in range(2):
            var tensor_2x2_masked = tensor_UxU.tile[2, 2](tile_i, tile_j)
            print("--tile[", tile_i, ", ", tile_j, "]--")
            print(
                tensor_2x2_masked.runtime_layout,
                tensor_2x2_masked.masked,
                tensor_2x2_masked.runtime_layout.bound_check_required(),
            )
            print(tensor_2x2_masked)


def test_tile_and_distribute():
    print("== test_tile_and_distribute")
    comptime layout = Layout.row_major(5, 3)
    var stack = Array[Float32, layout.size()](fill={})
    var tensor_UxU = LayoutTensor[.float32, Layout.row_major[2]()](
        stack, RuntimeLayout[Layout.row_major[2]()].row_major(Index(5, 3))
    )
    arange(tensor_UxU)
    for tile_i in range(3):
        for tile_j in range(2):
            print("--tile[", tile_i, ", ", tile_j, "]--")
            var tensor_2x2_masked = tensor_UxU.tile[2, 2](tile_i, tile_j)
            print(
                tensor_2x2_masked.runtime_layout,
                tensor_2x2_masked.masked,
                tensor_2x2_masked.runtime_layout.bound_check_required(),
            )
            print(tensor_2x2_masked)
            for thread_id in range(4):
                print("----thread[", thread_id, "]----")
                var distributed_masked_tensor = tensor_2x2_masked.distribute[
                    Layout.row_major(2, 2)
                ](thread_id)
                print(distributed_masked_tensor)


def test_tile_iterator_masked():
    print("== test_tile_iterator_masked")
    comptime layout = Layout.row_major(5, 3)
    var stack = Array[Float32, layout.size()](fill={})
    var tensor_UxU = LayoutTensor[.float32, Layout.row_major[2]()](
        stack, RuntimeLayout[Layout.row_major[2]()].row_major(Index(5, 3))
    )
    arange(tensor_UxU)
    print(tensor_UxU)
    for tile_j in range(2):
        var tensor_iter_2x2_masked = tensor_UxU.tiled_iterator[2, 2](0, tile_j)
        for tile_i in range(3):
            print("--tile[", tile_i, ", ", tile_j, "]--")
            print(tensor_iter_2x2_masked.runtime_layout)
            print(tensor_iter_2x2_masked[])
            tensor_iter_2x2_masked += 1


def test_tile_and_vectorize():
    print("== test_tile_and_vectorize")
    comptime layout = Layout.row_major(3, 4)
    var stack = Array[Float32, layout.size()](fill={})
    var tensor_UxU = LayoutTensor[.float32, Layout.row_major[2]()](
        stack, RuntimeLayout[Layout.row_major[2]()].row_major(Index(3, 4))
    )
    arange(tensor_UxU)
    print(tensor_UxU)
    for tile_i in range(2):
        for tile_j in range(2):
            print("--tile[", tile_i, ", ", tile_j, "]--")
            print(tensor_UxU.tile[2, 2](tile_i, tile_j).vectorize[1, 2]())


def main():
    # CHECK-LABEL: test_tile_masked
    # CHECK: --tile[ 0 ,  0 ]--
    # CHECK: ((4, 4):(7, 1)) True False
    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 7.0 8.0 9.0 10.0
    # CHECK: 14.0 15.0 16.0 17.0
    # CHECK: 21.0 22.0 23.0 24.0
    # CHECK: --tile[ 0 ,  1 ]--
    # CHECK: ((4, 3):(7, 1)) True True
    # CHECK: 4.0 5.0 6.0
    # CHECK: 11.0 12.0 13.0
    # CHECK: 18.0 19.0 20.0
    # CHECK: 25.0 26.0 27.0
    # CHECK: --tile[ 1 ,  0 ]--
    # CHECK: ((4, 4):(7, 1)) True False
    # CHECK: 28.0 29.0 30.0 31.0
    # CHECK: 35.0 36.0 37.0 38.0
    # CHECK: 42.0 43.0 44.0 45.0
    # CHECK: 49.0 50.0 51.0 52.0
    # CHECK: --tile[ 1 ,  1 ]--
    # CHECK: ((4, 3):(7, 1)) True True
    # CHECK: 32.0 33.0 34.0
    # CHECK: 39.0 40.0 41.0
    # CHECK: 46.0 47.0 48.0
    # CHECK: 53.0 54.0 55.0
    # CHECK: --tile[ 2 ,  0 ]--
    # CHECK: ((3, 4):(7, 1)) True True
    # CHECK: 56.0 57.0 58.0 59.0
    # CHECK: 63.0 64.0 65.0 66.0
    # CHECK: 70.0 71.0 72.0 73.0
    # CHECK: --tile[ 2 ,  1 ]--
    # CHECK: ((3, 3):(7, 1)) True True
    # CHECK: 60.0 61.0 62.0
    # CHECK: 67.0 68.0 69.0
    # CHECK: 74.0 75.0 76.0
    test_tile_masked()

    # CHECK-LABEL: test_subtile_masked
    # CHECK: --tile[ 0 ,  0 ]--
    # CHECK: ((8, 8):(9, 1)) True False
    # CHECK: --subtile[ 0 ,  0 ]--
    # CHECK: ((4, 4):(9, 1)) True False
    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 9.0 10.0 11.0 12.0
    # CHECK: 18.0 19.0 20.0 21.0
    # CHECK: 27.0 28.0 29.0 30.0
    # CHECK: --subtile[ 0 ,  1 ]--
    # CHECK: ((4, 4):(9, 1)) True False
    # CHECK: 4.0 5.0 6.0 7.0
    # CHECK: 13.0 14.0 15.0 16.0
    # CHECK: 22.0 23.0 24.0 25.0
    # CHECK: 31.0 32.0 33.0 34.0
    # CHECK: --subtile[ 1 ,  0 ]--
    # CHECK: ((4, 4):(9, 1)) True False
    # CHECK: 36.0 37.0 38.0 39.0
    # CHECK: 45.0 46.0 47.0 48.0
    # CHECK: 54.0 55.0 56.0 57.0
    # CHECK: 63.0 64.0 65.0 66.0
    # CHECK: --subtile[ 1 ,  1 ]--
    # CHECK: ((4, 4):(9, 1)) True False
    # CHECK: 40.0 41.0 42.0 43.0
    # CHECK: 49.0 50.0 51.0 52.0
    # CHECK: 58.0 59.0 60.0 61.0
    # CHECK: 67.0 68.0 69.0 70.0
    # CHECK: --tile[ 0 ,  1 ]--
    # CHECK: ((8, 1):(9, 1)) True True
    # CHECK: --subtile[ 0 ,  0 ]--
    # CHECK: ((4, 1):(9, 1)) True True
    # CHECK: 8.0
    # CHECK: 17.0
    # CHECK: 26.0
    # CHECK: 35.0
    # CHECK: --subtile[ 0 ,  1 ]--
    # CHECK: ((4, 0):(9, 1)) True True
    # CHECK: --subtile[ 1 ,  0 ]--
    # CHECK: ((4, 1):(9, 1)) True True
    # CHECK: 44.0
    # CHECK: 53.0
    # CHECK: 62.0
    # CHECK: 71.0
    # CHECK: --subtile[ 1 ,  1 ]--
    # CHECK: ((4, 0):(9, 1)) True True
    # CHECK: --tile[ 1 ,  0 ]--
    # CHECK: ((7, 8):(9, 1)) True True
    # CHECK: --subtile[ 0 ,  0 ]--
    # CHECK: ((4, 4):(9, 1)) True False
    # CHECK: 72.0 73.0 74.0 75.0
    # CHECK: 81.0 82.0 83.0 84.0
    # CHECK: 90.0 91.0 92.0 93.0
    # CHECK: 99.0 100.0 101.0 102.0
    # CHECK: --subtile[ 0 ,  1 ]--
    # CHECK: ((4, 4):(9, 1)) True False
    # CHECK: 76.0 77.0 78.0 79.0
    # CHECK: 85.0 86.0 87.0 88.0
    # CHECK: 94.0 95.0 96.0 97.0
    # CHECK: 103.0 104.0 105.0 106.0
    # CHECK: --subtile[ 1 ,  0 ]--
    # CHECK: ((3, 4):(9, 1)) True True
    # CHECK: 108.0 109.0 110.0 111.0
    # CHECK: 117.0 118.0 119.0 120.0
    # CHECK: 126.0 127.0 128.0 129.0
    # CHECK: --subtile[ 1 ,  1 ]--
    # CHECK: ((3, 4):(9, 1)) True True
    # CHECK: 112.0 113.0 114.0 115.0
    # CHECK: 121.0 122.0 123.0 124.0
    # CHECK: 130.0 131.0 132.0 133.0
    # CHECK: --tile[ 1 ,  1 ]--
    # CHECK: ((7, 1):(9, 1)) True True
    # CHECK: --subtile[ 0 ,  0 ]--
    # CHECK: ((4, 1):(9, 1)) True True
    # CHECK: 80.0
    # CHECK: 89.0
    # CHECK: 98.0
    # CHECK: 107.0
    # CHECK: --subtile[ 0 ,  1 ]--
    # CHECK: ((4, 0):(9, 1)) True True
    # CHECK: --subtile[ 1 ,  0 ]--
    # CHECK: ((3, 1):(9, 1)) True True
    # CHECK: 116.0
    # CHECK: 125.0
    # CHECK: 134.0
    # CHECK: --subtile[ 1 ,  1 ]--
    # CHECK: ((3, 0):(9, 1)) True True
    test_subtile_masked()

    # CHECK-LABEL: test_tile_dynamic_no_bounds
    # CHECK: --tile[ 0 ,  0 ]--
    # CHECK: ((2, 2):(4, 1)) True False
    # CHECK: 0.0 1.0
    # CHECK: 4.0 5.0
    # CHECK: --tile[ 0 ,  1 ]--
    # CHECK: ((2, 2):(4, 1)) True False
    # CHECK: 2.0 3.0
    # CHECK: 6.0 7.0
    # CHECK: --tile[ 1 ,  0 ]--
    # CHECK: ((2, 2):(4, 1)) True False
    # CHECK: 8.0 9.0
    # CHECK: 12.0 13.0
    # CHECK: --tile[ 1 ,  1 ]--
    # CHECK: ((2, 2):(4, 1)) True False
    # CHECK: 10.0 11.0
    # CHECK: 14.0 15.0
    test_tile_dynamic_no_bounds()

    # CHECK-LABEL: test_tile_dynamic_with_bounds
    # CHECK: --tile[ 0 ,  0 ]--
    # CHECK: ((2, 2):(3, 1)) True False
    # CHECK: 0.0 1.0
    # CHECK: 3.0 4.0
    # CHECK: --tile[ 0 ,  1 ]--
    # CHECK: ((2, 1):(3, 1)) True True
    # CHECK: 2.0
    # CHECK: 5.0
    # CHECK: --tile[ 1 ,  0 ]--
    # CHECK: ((2, 2):(3, 1)) True False
    # CHECK: 6.0 7.0
    # CHECK: 9.0 10.0
    # CHECK: --tile[ 1 ,  1 ]--
    # CHECK: ((2, 1):(3, 1)) True True
    # CHECK: 8.0
    # CHECK: 11.0
    # CHECK: --tile[ 2 ,  0 ]--
    # CHECK: ((1, 2):(3, 1)) True True
    # CHECK: 12.0 13.0
    # CHECK: --tile[ 2 ,  1 ]--
    # CHECK: ((1, 1):(3, 1)) True True
    # CHECK: 14.0
    test_tile_dynamic_with_bounds()

    # CHECK: == test_tile_and_distribute
    # CHECK: --tile[ 0 ,  0 ]--
    # CHECK: ((2, 2):(3, 1)) True False
    # CHECK: 0.0 1.0
    # CHECK: 3.0 4.0
    # CHECK: ----thread[ 0 ]----
    # CHECK: 0.0
    # CHECK: ----thread[ 1 ]----
    # CHECK: 1.0
    # CHECK: ----thread[ 2 ]----
    # CHECK: 3.0
    # CHECK: ----thread[ 3 ]----
    # CHECK: 4.0
    # CHECK: --tile[ 0 ,  1 ]--
    # CHECK: ((2, 1):(3, 1)) True True
    # CHECK: 2.0
    # CHECK: 5.0
    # CHECK: ----thread[ 0 ]----
    # CHECK: 2.0
    # CHECK: ----thread[ 1 ]----
    # CHECK: ----thread[ 2 ]----
    # CHECK: 5.0
    # CHECK: ----thread[ 3 ]----

    # CHECK: --tile[ 1 ,  0 ]--
    # CHECK: ((2, 2):(3, 1)) True False
    # CHECK: 6.0 7.0
    # CHECK: 9.0 10.0
    # CHECK: ----thread[ 0 ]----
    # CHECK: 6.0
    # CHECK: ----thread[ 1 ]----
    # CHECK: 7.0
    # CHECK: ----thread[ 2 ]----
    # CHECK: 9.0
    # CHECK: ----thread[ 3 ]----
    # CHECK: 10.0
    # CHECK: --tile[ 1 ,  1 ]--
    # CHECK: ((2, 1):(3, 1)) True True
    # CHECK: 8.0
    # CHECK: 11.0
    # CHECK: ----thread[ 0 ]----
    # CHECK: 8.0
    # CHECK: ----thread[ 1 ]----
    # CHECK: ----thread[ 2 ]----
    # CHECK: 11.0
    # CHECK: ----thread[ 3 ]----

    # CHECK: --tile[ 2 ,  0 ]--
    # CHECK: ((1, 2):(3, 1)) True True
    # CHECK: 12.0 13.0
    # CHECK: ----thread[ 0 ]----
    # CHECK: 12.0
    # CHECK: ----thread[ 1 ]----
    # CHECK: 13.0
    # CHECK: ----thread[ 2 ]----
    # CHECK: ----thread[ 3 ]----

    # CHECK: --tile[ 2 ,  1 ]--
    # CHECK: ((1, 1):(3, 1)) True True
    # CHECK: 14.0
    # CHECK: ----thread[ 0 ]----
    # CHECK: 14.0
    # CHECK: ----thread[ 1 ]----
    # CHECK: ----thread[ 2 ]----
    # CHECK: ----thread[ 3 ]----
    test_tile_and_distribute()

    # CHECK: == test_tile_iterator_masked
    # CHECK: 0.0 1.0 2.0
    # CHECK: 3.0 4.0 5.0
    # CHECK: 6.0 7.0 8.0
    # CHECK: 9.0 10.0 11.0
    # CHECK: 12.0 13.0 14.0
    # CHECK: --tile[ 0 ,  0 ]--
    # CHECK: ((2, 2):(3, 1))
    # CHECK: 0.0 1.0
    # CHECK: 3.0 4.0
    # CHECK: --tile[ 1 ,  0 ]--
    # CHECK: ((2, 2):(3, 1))
    # CHECK: 6.0 7.0
    # CHECK: 9.0 10.0
    # CHECK: --tile[ 2 ,  0 ]--
    # CHECK: ((1, 2):(3, 1))
    # CHECK: 12.0 13.0
    # CHECK: --tile[ 0 ,  1 ]--
    # CHECK: ((2, 1):(3, 1))
    # CHECK: 2.0
    # CHECK: 5.0
    # CHECK: --tile[ 1 ,  1 ]--
    # CHECK: ((2, 1):(3, 1))
    # CHECK: 8.0
    # CHECK: 11.0
    # CHECK: --tile[ 2 ,  1 ]--
    # CHECK: ((1, 1):(3, 1))
    # CHECK: 14.0
    test_tile_iterator_masked()

    # CHECK: == test_tile_and_vectorize
    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0
    # CHECK: --tile[ 0 ,  0 ]--
    # CHECK: [0.0, 1.0]
    # CHECK: [4.0, 5.0]
    # CHECK: --tile[ 0 ,  1 ]--
    # CHECK: [2.0, 3.0]
    # CHECK: [6.0, 7.0]
    # CHECK: --tile[ 1 ,  0 ]--
    # CHECK: [8.0, 9.0]
    # CHECK: --tile[ 1 ,  1 ]--
    # CHECK: [10.0, 11.0]
    test_tile_and_vectorize()
