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

from std.math import sqrt

from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout._fillers import arange, random
from layout.layout_tensor import LayoutTensorIter

from std.utils import IndexList


#  CHECK-LABEL: test_fill_and_print
def test_fill_and_print() raises:
    print("== test_fill_and_print")

    comptime layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var dynamic_layout = RuntimeLayout[
        layout, element_type=.int32, linear_idx_type=.int32
    ](
        RuntimeTuple[layout.shape, element_type=.int32](4, 8),
        RuntimeTuple[layout.stride, element_type=.int32](8, 1),
    )

    var storage = List(length=dynamic_layout.size(), fill=Float32(0))

    var tensor = LayoutTensor[
        .float32,
        layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
    ](storage, dynamic_layout)
    arange(tensor)

    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    print(tensor)


#  CHECK-LABEL: test_set_and_get_items
def test_set_and_get_items() raises:
    print("== test_set_and_get_items")

    comptime layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var dynamic_layout = RuntimeLayout[
        layout, element_type=.int32, linear_idx_type=.int32
    ](
        RuntimeTuple[layout.shape, element_type=.int32](4, 4),
        RuntimeTuple[layout.stride, element_type=.int32](4, 1),
    )

    var storage = List(length=dynamic_layout.size(), fill=Float32(0))

    var tensor = LayoutTensor[
        .float32,
        layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
    ](storage, dynamic_layout)

    for i in range(4):
        for j in range(4):
            tensor[i, j] = Float32(i * 4 + j + 2)

    # CHECK: 2.0 3.0 4.0 5.0
    # CHECK: 6.0 7.0 8.0 9.0
    # CHECK: 10.0 11.0 12.0 13.0
    # CHECK: 14.0 15.0 16.0 17.0
    print(tensor)


#  CHECK-LABEL: test_tile
def test_tile() raises:
    print("== test_tile")

    comptime layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var dynamic_layout = RuntimeLayout[
        layout, element_type=.int32, linear_idx_type=.int32
    ](
        RuntimeTuple[layout.shape, element_type=.int32](4, 4),
        RuntimeTuple[layout.stride, element_type=.int32](4, 1),
    )

    var storage = List(length=dynamic_layout.size(), fill=Float32(0))

    var tensor = LayoutTensor[
        .float32,
        layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
    ](storage, dynamic_layout)
    arange(tensor)

    var tile = tensor.tile[2, 2](0, 0)
    # CHECK: ((2, 2):(-1, 1))
    print(materialize[tile.layout]())

    # CHECK: ((2, 2):(4, 1))
    print(tile.runtime_layout)

    # CHECK: ----tile-data[ 0 , 0 ]----
    # CHECK: 0.0 1.0
    # CHECK: 4.0 5.0
    # CHECK: ----tile-data[ 0 , 1 ]----
    # CHECK: 2.0 3.0
    # CHECK: 6.0 7.0
    # CHECK: ----tile-data[ 1 , 0 ]----
    # CHECK: 8.0 9.0
    # CHECK: 12.0 13.0
    # CHECK: ----tile-data[ 1 , 1 ]----
    # CHECK: 10.0 11.0
    # CHECK: 14.0 15.0
    for tile_i in range(2):
        for tile_j in range(2):
            print("----tile-data[", tile_i, ",", tile_j, "]----")
            var tile_2x2 = tensor.tile[2, 2](tile_i, tile_j)
            print(tile_2x2)


def test_tile_and_distribute():
    print("== test_tile_and_distribute")

    comptime layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    var dynamic_layout = RuntimeLayout[
        layout, element_type=.int64, linear_idx_type=.int64
    ](
        RuntimeTuple[layout.shape](8, 8).cast[.int64](),
        RuntimeTuple[layout.stride](8, 1).cast[.int64](),
    )

    var storage = List(length=dynamic_layout.size(), fill=Float32(0))

    var tensor = LayoutTensor[
        .float32,
        layout,
        layout_int_type=.int64,
        linear_idx_type=.int64,
    ](storage, dynamic_layout)
    arange(tensor)

    # ---tile-data[ 0 , 0 ]----
    # 0.0 1.0 2.0 3.0
    # 8.0 9.0 10.0 11.0
    # 16.0 17.0 18.0 19.0
    # 24.0 25.0 26.0 27.0
    # ----fragments-data[ 0 ]----
    # 0.0 2.0
    # 16.0 18.0
    # ----fragments-data[ 1 ]----
    # 1.0 3.0
    # 17.0 19.0
    # ----fragments-data[ 2 ]----
    # 8.0 10.0
    # 24.0 26.0
    # ----fragments-data[ 3 ]----
    # 9.0 11.0
    # 25.0 27.0
    # ----tile-data[ 0 , 1 ]----
    # 4.0 5.0 6.0 7.0
    # 12.0 13.0 14.0 15.0
    # 20.0 21.0 22.0 23.0
    # 28.0 29.0 30.0 31.0
    # ----fragments-data[ 0 ]----
    # 4.0 6.0
    # 20.0 22.0
    # ----fragments-data[ 1 ]----
    # 5.0 7.0
    # 21.0 23.0
    # ----fragments-data[ 2 ]----
    # 12.0 14.0
    # 28.0 30.0
    # ----fragments-data[ 3 ]----
    # 13.0 15.0
    # 29.0 31.0
    # ----tile-data[ 1 , 0 ]----
    # 32.0 33.0 34.0 35.0
    # 40.0 41.0 42.0 43.0
    # 48.0 49.0 50.0 51.0
    # 56.0 57.0 58.0 59.0
    # ----fragments-data[ 0 ]----
    # 32.0 34.0
    # 48.0 50.0
    # ----fragments-data[ 1 ]----
    # 33.0 35.0
    # 49.0 51.0
    # ----fragments-data[ 2 ]----
    # 40.0 42.0
    # 56.0 58.0
    # ----fragments-data[ 3 ]----
    # 41.0 43.0
    # 57.0 59.0
    # ----tile-data[ 1 , 1 ]----
    # 36.0 37.0 38.0 39.0
    # 44.0 45.0 46.0 47.0
    # 52.0 53.0 54.0 55.0
    # 60.0 61.0 62.0 63.0
    # ----fragments-data[ 0 ]----
    # 36.0 38.0
    # 52.0 54.0
    # ----fragments-data[ 1 ]----
    # 37.0 39.0
    # 53.0 55.0
    # ----fragments-data[ 2 ]----
    # 44.0 46.0
    # 60.0 62.0
    # ----fragments-data[ 3 ]----
    # 45.0 47.0
    # 61.0 63.0
    for tile_i in range(2):
        for tile_j in range(2):
            print("----tile-data[", tile_i, ",", tile_j, "]----")
            var tile_4x4 = tensor.tile[4, 4](tile_i, tile_j)
            print(tile_4x4)
            for th_i in range(4):
                var tile_2x2 = tile_4x4.distribute[Layout.row_major(2, 2)](th_i)
                print("----fragments-data[", th_i, "]----")
                print(tile_2x2)


# CHECK-LABEL: test_tile_and_vectorize
def test_tile_and_vectorize():
    print("== test_tile_and_vectorize")

    comptime layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var dynamic_layout = RuntimeLayout[
        layout, element_type=.int32, linear_idx_type=.int32
    ](
        RuntimeTuple[layout.shape, element_type=.int32](16, 16),
        RuntimeTuple[layout.stride, element_type=.int32](16, 1),
    )

    var storage = List(length=dynamic_layout.size(), fill=Float32(0))

    var tensor = LayoutTensor[
        .float32,
        layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
    ](storage, dynamic_layout)
    arange(tensor)

    # CHECK: ----tile-data[ 0 , 0 ]----
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0
    # CHECK: 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0
    # CHECK: 96.0 97.0 98.0 99.0 100.0 101.0 102.0 103.0
    # CHECK: 112.0 113.0 114.0 115.0 116.0 117.0 118.0 119.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [0.0, 1.0, 2.0, 3.0, 16.0, 17.0, 18.0, 19.0, 32.0, 33.0, 34.0, 35.0, 48.0, 49.0, 50.0, 51.0] [4.0, 5.0, 6.0, 7.0, 20.0, 21.0, 22.0, 23.0, 36.0, 37.0, 38.0, 39.0, 52.0, 53.0, 54.0, 55.0]
    # CHECK: [64.0, 65.0, 66.0, 67.0, 80.0, 81.0, 82.0, 83.0, 96.0, 97.0, 98.0, 99.0, 112.0, 113.0, 114.0, 115.0] [68.0, 69.0, 70.0, 71.0, 84.0, 85.0, 86.0, 87.0, 100.0, 101.0, 102.0, 103.0, 116.0, 117.0, 118.0, 119.0]
    # CHECK: ----tile-data[ 0 , 1 ]----
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0
    # CHECK: 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0
    # CHECK: 104.0 105.0 106.0 107.0 108.0 109.0 110.0 111.0
    # CHECK: 120.0 121.0 122.0 123.0 124.0 125.0 126.0 127.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [8.0, 9.0, 10.0, 11.0, 24.0, 25.0, 26.0, 27.0, 40.0, 41.0, 42.0, 43.0, 56.0, 57.0, 58.0, 59.0] [12.0, 13.0, 14.0, 15.0, 28.0, 29.0, 30.0, 31.0, 44.0, 45.0, 46.0, 47.0, 60.0, 61.0, 62.0, 63.0]
    # CHECK: [72.0, 73.0, 74.0, 75.0, 88.0, 89.0, 90.0, 91.0, 104.0, 105.0, 106.0, 107.0, 120.0, 121.0, 122.0, 123.0] [76.0, 77.0, 78.0, 79.0, 92.0, 93.0, 94.0, 95.0, 108.0, 109.0, 110.0, 111.0, 124.0, 125.0, 126.0, 127.0]
    # CHECK: ----tile-data[ 1 , 0 ]----
    # CHECK: 128.0 129.0 130.0 131.0 132.0 133.0 134.0 135.0
    # CHECK: 144.0 145.0 146.0 147.0 148.0 149.0 150.0 151.0
    # CHECK: 160.0 161.0 162.0 163.0 164.0 165.0 166.0 167.0
    # CHECK: 176.0 177.0 178.0 179.0 180.0 181.0 182.0 183.0
    # CHECK: 192.0 193.0 194.0 195.0 196.0 197.0 198.0 199.0
    # CHECK: 208.0 209.0 210.0 211.0 212.0 213.0 214.0 215.0
    # CHECK: 224.0 225.0 226.0 227.0 228.0 229.0 230.0 231.0
    # CHECK: 240.0 241.0 242.0 243.0 244.0 245.0 246.0 247.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [128.0, 129.0, 130.0, 131.0, 144.0, 145.0, 146.0, 147.0, 160.0, 161.0, 162.0, 163.0, 176.0, 177.0, 178.0, 179.0] [132.0, 133.0, 134.0, 135.0, 148.0, 149.0, 150.0, 151.0, 164.0, 165.0, 166.0, 167.0, 180.0, 181.0, 182.0, 183.0]
    # CHECK: [192.0, 193.0, 194.0, 195.0, 208.0, 209.0, 210.0, 211.0, 224.0, 225.0, 226.0, 227.0, 240.0, 241.0, 242.0, 243.0] [196.0, 197.0, 198.0, 199.0, 212.0, 213.0, 214.0, 215.0, 228.0, 229.0, 230.0, 231.0, 244.0, 245.0, 246.0, 247.0]
    # CHECK: ----tile-data[ 1 , 1 ]----
    # CHECK: 136.0 137.0 138.0 139.0 140.0 141.0 142.0 143.0
    # CHECK: 152.0 153.0 154.0 155.0 156.0 157.0 158.0 159.0
    # CHECK: 168.0 169.0 170.0 171.0 172.0 173.0 174.0 175.0
    # CHECK: 184.0 185.0 186.0 187.0 188.0 189.0 190.0 191.0
    # CHECK: 200.0 201.0 202.0 203.0 204.0 205.0 206.0 207.0
    # CHECK: 216.0 217.0 218.0 219.0 220.0 221.0 222.0 223.0
    # CHECK: 232.0 233.0 234.0 235.0 236.0 237.0 238.0 239.0
    # CHECK: 248.0 249.0 250.0 251.0 252.0 253.0 254.0 255.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [136.0, 137.0, 138.0, 139.0, 152.0, 153.0, 154.0, 155.0, 168.0, 169.0, 170.0, 171.0, 184.0, 185.0, 186.0, 187.0] [140.0, 141.0, 142.0, 143.0, 156.0, 157.0, 158.0, 159.0, 172.0, 173.0, 174.0, 175.0, 188.0, 189.0, 190.0, 191.0]
    # CHECK: [200.0, 201.0, 202.0, 203.0, 216.0, 217.0, 218.0, 219.0, 232.0, 233.0, 234.0, 235.0, 248.0, 249.0, 250.0, 251.0] [204.0, 205.0, 206.0, 207.0, 220.0, 221.0, 222.0, 223.0, 236.0, 237.0, 238.0, 239.0, 252.0, 253.0, 254.0, 255.0]

    for tile_i in range(2):
        for tile_j in range(2):
            print("----tile-data[", tile_i, ",", tile_j, "]----")
            var tensor_8x8 = tensor.tile[8, 8](tile_i, tile_j)
            print(tensor_8x8)
            var tensor_v_2x2 = tensor_8x8.vectorize[4, 4]()
            print("----vectorized-matrix----")
            print(tensor_v_2x2)

    # CHECK: ----tile-data[ 0 , 0 ]----
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0
    # CHECK: 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0
    # CHECK: 96.0 97.0 98.0 99.0 100.0 101.0 102.0 103.0
    # CHECK: 112.0 113.0 114.0 115.0 116.0 117.0 118.0 119.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [0.0, 16.0, 32.0, 48.0] [1.0, 17.0, 33.0, 49.0] [2.0, 18.0, 34.0, 50.0] [3.0, 19.0, 35.0, 51.0] [4.0, 20.0, 36.0, 52.0] [5.0, 21.0, 37.0, 53.0] [6.0, 22.0, 38.0, 54.0] [7.0, 23.0, 39.0, 55.0]
    # CHECK: [64.0, 80.0, 96.0, 112.0] [65.0, 81.0, 97.0, 113.0] [66.0, 82.0, 98.0, 114.0] [67.0, 83.0, 99.0, 115.0] [68.0, 84.0, 100.0, 116.0] [69.0, 85.0, 101.0, 117.0] [70.0, 86.0, 102.0, 118.0] [71.0, 87.0, 103.0, 119.0]
    # CHECK: ----tile-data[ 0 , 1 ]----
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0
    # CHECK: 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0
    # CHECK: 104.0 105.0 106.0 107.0 108.0 109.0 110.0 111.0
    # CHECK: 120.0 121.0 122.0 123.0 124.0 125.0 126.0 127.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [8.0, 24.0, 40.0, 56.0] [9.0, 25.0, 41.0, 57.0] [10.0, 26.0, 42.0, 58.0] [11.0, 27.0, 43.0, 59.0] [12.0, 28.0, 44.0, 60.0] [13.0, 29.0, 45.0, 61.0] [14.0, 30.0, 46.0, 62.0] [15.0, 31.0, 47.0, 63.0]
    # CHECK: [72.0, 88.0, 104.0, 120.0] [73.0, 89.0, 105.0, 121.0] [74.0, 90.0, 106.0, 122.0] [75.0, 91.0, 107.0, 123.0] [76.0, 92.0, 108.0, 124.0] [77.0, 93.0, 109.0, 125.0] [78.0, 94.0, 110.0, 126.0] [79.0, 95.0, 111.0, 127.0]
    # CHECK: ----tile-data[ 1 , 0 ]----
    # CHECK: 128.0 129.0 130.0 131.0 132.0 133.0 134.0 135.0
    # CHECK: 144.0 145.0 146.0 147.0 148.0 149.0 150.0 151.0
    # CHECK: 160.0 161.0 162.0 163.0 164.0 165.0 166.0 167.0
    # CHECK: 176.0 177.0 178.0 179.0 180.0 181.0 182.0 183.0
    # CHECK: 192.0 193.0 194.0 195.0 196.0 197.0 198.0 199.0
    # CHECK: 208.0 209.0 210.0 211.0 212.0 213.0 214.0 215.0
    # CHECK: 224.0 225.0 226.0 227.0 228.0 229.0 230.0 231.0
    # CHECK: 240.0 241.0 242.0 243.0 244.0 245.0 246.0 247.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [128.0, 144.0, 160.0, 176.0] [129.0, 145.0, 161.0, 177.0] [130.0, 146.0, 162.0, 178.0] [131.0, 147.0, 163.0, 179.0] [132.0, 148.0, 164.0, 180.0] [133.0, 149.0, 165.0, 181.0] [134.0, 150.0, 166.0, 182.0] [135.0, 151.0, 167.0, 183.0]
    # CHECK: [192.0, 208.0, 224.0, 240.0] [193.0, 209.0, 225.0, 241.0] [194.0, 210.0, 226.0, 242.0] [195.0, 211.0, 227.0, 243.0] [196.0, 212.0, 228.0, 244.0] [197.0, 213.0, 229.0, 245.0] [198.0, 214.0, 230.0, 246.0] [199.0, 215.0, 231.0, 247.0]
    # CHECK: ----tile-data[ 1 , 1 ]----
    # CHECK: 136.0 137.0 138.0 139.0 140.0 141.0 142.0 143.0
    # CHECK: 152.0 153.0 154.0 155.0 156.0 157.0 158.0 159.0
    # CHECK: 168.0 169.0 170.0 171.0 172.0 173.0 174.0 175.0
    # CHECK: 184.0 185.0 186.0 187.0 188.0 189.0 190.0 191.0
    # CHECK: 200.0 201.0 202.0 203.0 204.0 205.0 206.0 207.0
    # CHECK: 216.0 217.0 218.0 219.0 220.0 221.0 222.0 223.0
    # CHECK: 232.0 233.0 234.0 235.0 236.0 237.0 238.0 239.0
    # CHECK: 248.0 249.0 250.0 251.0 252.0 253.0 254.0 255.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [136.0, 152.0, 168.0, 184.0] [137.0, 153.0, 169.0, 185.0] [138.0, 154.0, 170.0, 186.0] [139.0, 155.0, 171.0, 187.0] [140.0, 156.0, 172.0, 188.0] [141.0, 157.0, 173.0, 189.0] [142.0, 158.0, 174.0, 190.0] [143.0, 159.0, 175.0, 191.0]
    # CHECK: [200.0, 216.0, 232.0, 248.0] [201.0, 217.0, 233.0, 249.0] [202.0, 218.0, 234.0, 250.0] [203.0, 219.0, 235.0, 251.0] [204.0, 220.0, 236.0, 252.0] [205.0, 221.0, 237.0, 253.0] [206.0, 222.0, 238.0, 254.0] [207.0, 223.0, 239.0, 255.0]
    for tile_i in range(2):
        for tile_j in range(2):
            print("----tile-data[", tile_i, ",", tile_j, "]----")
            var tensor_8x8 = tensor.tile[8, 8](tile_i, tile_j)
            print(tensor_8x8)
            var tensor_v_2x8 = tensor_8x8.vectorize[4, 1]()
            print("----vectorized-matrix----")
            print(tensor_v_2x8)

    # CHECK: ----tile-data[ 0 , 0 ]----
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0
    # CHECK: 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0
    # CHECK: 96.0 97.0 98.0 99.0 100.0 101.0 102.0 103.0
    # CHECK: 112.0 113.0 114.0 115.0 116.0 117.0 118.0 119.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [64.0, 65.0, 66.0, 67.0] [68.0, 69.0, 70.0, 71.0]
    # CHECK: [80.0, 81.0, 82.0, 83.0] [84.0, 85.0, 86.0, 87.0]
    # CHECK: [96.0, 97.0, 98.0, 99.0] [100.0, 101.0, 102.0, 103.0]
    # CHECK: [112.0, 113.0, 114.0, 115.0] [116.0, 117.0, 118.0, 119.0]
    # CHECK: ----tile-data[ 0 , 1 ]----
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0
    # CHECK: 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0
    # CHECK: 104.0 105.0 106.0 107.0 108.0 109.0 110.0 111.0
    # CHECK: 120.0 121.0 122.0 123.0 124.0 125.0 126.0 127.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    # CHECK: [72.0, 73.0, 74.0, 75.0] [76.0, 77.0, 78.0, 79.0]
    # CHECK: [88.0, 89.0, 90.0, 91.0] [92.0, 93.0, 94.0, 95.0]
    # CHECK: [104.0, 105.0, 106.0, 107.0] [108.0, 109.0, 110.0, 111.0]
    # CHECK: [120.0, 121.0, 122.0, 123.0] [124.0, 125.0, 126.0, 127.0]
    # CHECK: ----tile-data[ 1 , 0 ]----
    # CHECK: 128.0 129.0 130.0 131.0 132.0 133.0 134.0 135.0
    # CHECK: 144.0 145.0 146.0 147.0 148.0 149.0 150.0 151.0
    # CHECK: 160.0 161.0 162.0 163.0 164.0 165.0 166.0 167.0
    # CHECK: 176.0 177.0 178.0 179.0 180.0 181.0 182.0 183.0
    # CHECK: 192.0 193.0 194.0 195.0 196.0 197.0 198.0 199.0
    # CHECK: 208.0 209.0 210.0 211.0 212.0 213.0 214.0 215.0
    # CHECK: 224.0 225.0 226.0 227.0 228.0 229.0 230.0 231.0
    # CHECK: 240.0 241.0 242.0 243.0 244.0 245.0 246.0 247.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [128.0, 129.0, 130.0, 131.0] [132.0, 133.0, 134.0, 135.0]
    # CHECK: [144.0, 145.0, 146.0, 147.0] [148.0, 149.0, 150.0, 151.0]
    # CHECK: [160.0, 161.0, 162.0, 163.0] [164.0, 165.0, 166.0, 167.0]
    # CHECK: [176.0, 177.0, 178.0, 179.0] [180.0, 181.0, 182.0, 183.0]
    # CHECK: [192.0, 193.0, 194.0, 195.0] [196.0, 197.0, 198.0, 199.0]
    # CHECK: [208.0, 209.0, 210.0, 211.0] [212.0, 213.0, 214.0, 215.0]
    # CHECK: [224.0, 225.0, 226.0, 227.0] [228.0, 229.0, 230.0, 231.0]
    # CHECK: [240.0, 241.0, 242.0, 243.0] [244.0, 245.0, 246.0, 247.0]
    # CHECK: ----tile-data[ 1 , 1 ]----
    # CHECK: 136.0 137.0 138.0 139.0 140.0 141.0 142.0 143.0
    # CHECK: 152.0 153.0 154.0 155.0 156.0 157.0 158.0 159.0
    # CHECK: 168.0 169.0 170.0 171.0 172.0 173.0 174.0 175.0
    # CHECK: 184.0 185.0 186.0 187.0 188.0 189.0 190.0 191.0
    # CHECK: 200.0 201.0 202.0 203.0 204.0 205.0 206.0 207.0
    # CHECK: 216.0 217.0 218.0 219.0 220.0 221.0 222.0 223.0
    # CHECK: 232.0 233.0 234.0 235.0 236.0 237.0 238.0 239.0
    # CHECK: 248.0 249.0 250.0 251.0 252.0 253.0 254.0 255.0
    # CHECK: ----vectorized-matrix----
    # CHECK: [136.0, 137.0, 138.0, 139.0] [140.0, 141.0, 142.0, 143.0]
    # CHECK: [152.0, 153.0, 154.0, 155.0] [156.0, 157.0, 158.0, 159.0]
    # CHECK: [168.0, 169.0, 170.0, 171.0] [172.0, 173.0, 174.0, 175.0]
    # CHECK: [184.0, 185.0, 186.0, 187.0] [188.0, 189.0, 190.0, 191.0]
    # CHECK: [200.0, 201.0, 202.0, 203.0] [204.0, 205.0, 206.0, 207.0]
    # CHECK: [216.0, 217.0, 218.0, 219.0] [220.0, 221.0, 222.0, 223.0]
    # CHECK: [232.0, 233.0, 234.0, 235.0] [236.0, 237.0, 238.0, 239.0]
    # CHECK: [248.0, 249.0, 250.0, 251.0] [252.0, 253.0, 254.0, 255.0]
    for tile_i in range(2):
        for tile_j in range(2):
            print("----tile-data[", tile_i, ",", tile_j, "]----")
            var tensor_8x8 = tensor.tile[8, 8](tile_i, tile_j)
            print(tensor_8x8)
            var tensor_v_8x2 = tensor_8x8.vectorize[1, 4]()
            print("----vectorized-matrix----")
            print(tensor_v_8x2)


# CHECK-LABEL: test_copy_from
def test_copy_from():
    print("== test_copy_from")
    comptime layout = Layout(
        IntTuple(8, 8), IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE)
    )

    var dynamic_layout = RuntimeLayout[
        layout, element_type=.int32, linear_idx_type=.int32
    ](
        RuntimeTuple[layout.shape, element_type=.int32](8, 8),
        RuntimeTuple[layout.stride, element_type=.int32](8, 1),
    )
    var src_engine = List(length=dynamic_layout.size(), fill=Float32(0))
    var src_tensor = LayoutTensor[
        .float32,
        layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
    ](src_engine, dynamic_layout)
    arange(src_tensor)

    var dst_storage = List(length=dynamic_layout.size(), fill=Float32(0))
    var dst_tensor = LayoutTensor[
        .float32,
        layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
    ](dst_storage, dynamic_layout)
    print(dst_tensor)
    dst_tensor.copy_from(src_tensor)
    print(dst_tensor)


# CHECK-LABEL: test_linspace_fill
def test_linspace_fill():
    print("== test_linspace_fill")
    comptime layout = Layout(
        IntTuple(8, 8), IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE)
    )

    var dynamic_layout = RuntimeLayout[
        layout, element_type=.int32, linear_idx_type=.int32
    ](
        RuntimeTuple[layout.shape, element_type=.int32](8, 8),
        RuntimeTuple[layout.stride, element_type=.int32](8, 1),
    )
    var src_engine = List(length=dynamic_layout.size(), fill=Float32(0))
    var src_tensor = LayoutTensor[
        .float32,
        layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
    ](src_engine, dynamic_layout)
    arange(src_tensor)

    # CHECK: ----source-tensor----
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0

    print("----source-tensor----")
    print(src_tensor)

    # CHECK: ----source-tensor----
    # CHECK: 42.0 42.0 42.0 42.0 42.0 42.0 42.0 42.0
    # CHECK: 42.0 42.0 42.0 42.0 42.0 42.0 42.0 42.0
    # CHECK: 42.0 42.0 42.0 42.0 42.0 42.0 42.0 42.0
    # CHECK: 42.0 42.0 42.0 42.0 42.0 42.0 42.0 42.0
    # CHECK: 42.0 42.0 42.0 42.0 42.0 42.0 42.0 42.0
    # CHECK: 42.0 42.0 42.0 42.0 42.0 42.0 42.0 42.0
    # CHECK: 42.0 42.0 42.0 42.0 42.0 42.0 42.0 42.0
    # CHECK: 42.0 42.0 42.0 42.0 42.0 42.0 42.0 42.0

    var src_tensor_copy = src_tensor.fill(42.0)
    print("----source-tensor----")
    print(src_tensor)

    # CHECK: ----source-tensor-copy----
    # CHECK: 42.0 42.0 42.0 42.0 42.0 42.0 42.0 42.0
    # CHECK: 42.0 42.0 42.0 42.0 42.0 42.0 42.0 42.0
    # CHECK: 42.0 42.0 42.0 42.0 42.0 42.0 42.0 42.0
    # CHECK: 42.0 42.0 42.0 42.0 42.0 42.0 42.0 42.0
    # CHECK: 42.0 42.0 42.0 42.0 42.0 42.0 42.0 42.0
    # CHECK: 42.0 42.0 42.0 42.0 42.0 42.0 42.0 42.0
    # CHECK: 42.0 42.0 42.0 42.0 42.0 42.0 42.0 42.0
    # CHECK: 42.0 42.0 42.0 42.0 42.0 42.0 42.0 42.0
    # CHECK: True
    print("----source-tensor-copy----")
    print(src_tensor_copy)
    print(src_tensor.ptr == src_tensor_copy.ptr)


# CHECK-LABEL: test_random_fill
def test_random_fill():
    print("== test_random_fill")
    comptime layout = Layout(8 * 8 * 8 * 8)

    comptime RuntimeLayoutType = RuntimeLayout[
        layout, element_type=.int32, linear_idx_type=.int32
    ]

    var dynamic_layout = RuntimeLayoutType(
        RuntimeLayoutType.ShapeType(comptime (layout.size())),
        RuntimeLayoutType.StrideType(1),
    )
    var src_engine = List(length=dynamic_layout.size(), fill=Float32(0))
    var src_tensor = LayoutTensor[
        .float32,
        layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
    ](src_engine, dynamic_layout)
    random(src_tensor)
    var sum: Float32 = 0.0
    for i in range(src_tensor.runtime_layout.size()):
        sum += rebind[Float32](src_tensor[i])
    var mean = sum / Float32(src_tensor.runtime_layout.size())

    var variance: Float32 = 0.0
    for i in range(src_tensor.runtime_layout.size()):
        var diff = rebind[Float32](src_tensor[i]) - mean
        variance += diff * diff
    variance = sqrt(variance / Float32(src_tensor.runtime_layout.size()))
    # Check that the mean value is close to 0.5 and variance is more than 0.1
    # CHECK: ----mean-variance----
    # CHECK: True
    # CHECK: True
    print("----mean-variance----")
    print(abs(mean - 0.5) < 0.01)
    print(variance > 0.1)


# CHECK-LABEL: test_iterator
def test_iterator():
    print("== test_iterator")
    comptime layout = Layout(IntTuple(UNKNOWN_VALUE, 8), IntTuple(8, 1))

    var dynamic_layout = RuntimeLayout[
        layout, element_type=.int32, linear_idx_type=.int32
    ](
        RuntimeTuple[layout.shape, element_type=.int32](8, 8),
        RuntimeTuple[layout.stride, element_type=.int32](8, 1),
    )

    var ptr = List(length=dynamic_layout.size(), fill=Float32(0))
    var tensor = LayoutTensor[
        .float32,
        layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
    ](ptr, dynamic_layout)
    arange(tensor)

    # CHECK: ((4, 4):(8, 1))
    # CHECK: 64 32 32
    # CHECK: 36.0 37.0 38.0 39.0
    # CHECK: 44.0 45.0 46.0 47.0
    # CHECK: 52.0 53.0 54.0 55.0
    # CHECK: 60.0 61.0 62.0 63.0
    var iter4x4_axis0 = tensor.tiled_iterator[4, 4, axis=0](0, 1)
    iter4x4_axis0 += 1
    print(iter4x4_axis0.runtime_layout)
    print(iter4x4_axis0.bound, iter4x4_axis0.offset, iter4x4_axis0.stride)
    print(iter4x4_axis0[])

    # CHECK: ((2, 2):(8, 1))
    # CHECK: 8 2 2
    # CHECK: 34.0 35.0
    # CHECK: 42.0 43.0
    var iter2x2_axis1 = tensor.tiled_iterator[2, 2, axis=1](2, 0)
    iter2x2_axis1 += 1
    print(iter2x2_axis1.runtime_layout)
    print(iter2x2_axis1.bound, iter2x2_axis1.offset, iter2x2_axis1.stride)
    print(iter2x2_axis1[])

    # CHECK: ((4, 2):(2, 1))
    # CHECK: 64 8 8
    # CHECK: 8.0 9.0
    # CHECK: 10.0 11.0
    # CHECK: 12.0 13.0
    # CHECK: 14.0 15.0
    comptime layout1 = Layout(
        IntTuple(4, UNKNOWN_VALUE), IntTuple(2, UNKNOWN_VALUE)
    )
    var dynamic_layout1 = RuntimeLayout[
        layout1, element_type=.int32, linear_idx_type=.int32
    ](
        RuntimeTuple[layout1.shape, element_type=.int32](4, 2),
        RuntimeTuple[layout1.stride, element_type=.int32](2, 1),
    )
    var iter = LayoutTensorIter[
        DType.float32,
        layout1,
        layout_int_type=DType.int32,
        linear_idx_type=.int32,
        circular=True,
    ](ptr.unsafe_ptr(), 64, dynamic_layout1)
    iter += 9
    print(iter.runtime_layout)
    print(iter.bound, iter.offset, iter.stride)
    print(iter[])

    comptime M = 8
    comptime N = 8
    comptime BM = 2
    comptime BN = 2
    comptime type = DType.float32
    comptime unknown_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    # alias layout = Layout.row_major(M, N)
    var runtime_layout = RuntimeLayout[
        unknown_layout,
        element_type=.int32,
        linear_idx_type=.int32,
    ].row_major(IndexList[2, element_type=.int32](M, N))
    var ptr1 = List(length=M * N, fill=Scalar[type](0))

    var tensor1 = LayoutTensor[
        type,
        unknown_layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
    ](ptr1, runtime_layout)
    arange(tensor1)

    var tensor_slice1 = tensor1.tiled_iterator[BM, BN](0, 0)

    # CHECK: 0.0 1.0
    # CHECK: 8.0 9.0
    print(tensor_slice1[])

    var tensor_slice2 = tensor1.tiled_iterator[BM, BN](1, 0)

    # CHECK: 16.0 17.0
    # CHECK: 24.0 25.0
    print(tensor_slice2[])


# CHECK-LABEL: test_split
def test_split():
    print("== test_split")

    var ptr = List(length=16, fill=Float32(0))

    comptime layout_Ux4 = Layout(IntTuple(UNKNOWN_VALUE, 4), IntTuple(4, 1))
    var dynamic_layout_2x4 = RuntimeLayout[
        layout_Ux4, element_type=.int32, linear_idx_type=.int32
    ](
        RuntimeTuple[layout_Ux4.shape, element_type=.int32](2, 4),
        RuntimeTuple[layout_Ux4.stride, element_type=.int32](4, 1),
    )
    var tensor_Ux4 = LayoutTensor[
        .float32,
        layout_Ux4,
        layout_int_type=.int32,
        linear_idx_type=.int32,
    ](ptr, dynamic_layout_2x4)
    arange(tensor_Ux4)
    # CHECK: 0.0 1.0
    # CHECK: 4.0 5.0
    print(tensor_Ux4.split[axis=1](2, 0))
    # CHECK: 2.0 3.0
    # CHECK: 6.0 7.0
    print(tensor_Ux4.split[axis=1](2, 1))

    comptime layout_4x4 = Layout(IntTuple(4, 4), IntTuple(4, 1))
    var dynamic_layout_4x4 = RuntimeLayout[
        layout_4x4, element_type=.int32, linear_idx_type=.int32
    ](
        RuntimeTuple[layout_4x4.shape, element_type=.int32](4, 4),
        RuntimeTuple[layout_4x4.stride, element_type=.int32](4, 1),
    )
    var tensor_4x4 = LayoutTensor[
        .float32,
        layout_4x4,
        layout_int_type=.int32,
        linear_idx_type=.int32,
    ](ptr, dynamic_layout_4x4)
    arange(tensor_4x4)
    var tensor_4x4_split0 = tensor_4x4.split[axis=0](2, 0)
    var tensor_4x4_split1 = tensor_4x4.split[axis=0](2, 1)

    # CHECK: ((-1, 4):(4, 1))
    print(materialize[tensor_4x4_split0.layout]())
    # CHECK: ((2, 4):(4, 1))
    print(tensor_4x4_split0.runtime_layout)
    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 4.0 5.0 6.0 7.0
    print(tensor_4x4_split0)

    # CHECK: ((-1, 4):(4, 1))
    print(materialize[tensor_4x4_split1.layout]())
    # CHECK: ((2, 4):(4, 1))
    print(tensor_4x4_split1.runtime_layout)
    # CHECK: 8.0 9.0 10.0 11.0
    # CHECK: 12.0 13.0 14.0 15.0
    print(tensor_4x4_split1)

    comptime layout_Ux8 = Layout(IntTuple(UNKNOWN_VALUE, 8), IntTuple(8, 1))
    var dynamic_layout_Ux8 = RuntimeLayout[
        layout_Ux8, element_type=.int32, linear_idx_type=.int32
    ](
        RuntimeTuple[layout_Ux8.shape, element_type=.int32](2, 8),
        RuntimeTuple[layout_Ux8.stride, element_type=.int32](8, 1),
    )
    var tensor_Ux8 = LayoutTensor[
        .float32,
        layout_Ux8,
        layout_int_type=.int32,
        linear_idx_type=.int32,
    ](ptr, dynamic_layout_Ux8)
    var tensor_Ux8_split0 = tensor_Ux8.split[1, split_alignment=3](3, 0)
    var tensor_Ux8_split1 = tensor_Ux8.split[1, split_alignment=3](3, 1)
    var tensor_Ux8_split2 = tensor_Ux8.split[1, split_alignment=3](3, 2)

    # CHECK: ((-1, -1):(8, 1))
    print(materialize[tensor_Ux8_split0.layout]())
    # CHECK: ((2, 3):(8, 1))
    print(tensor_Ux8_split0.runtime_layout)
    # CHECK: 0.0 1.0 2.0
    # CHECK: 8.0 9.0 10.0
    print(tensor_Ux8_split0)

    # CHECK: ((-1, -1):(8, 1))
    print(materialize[tensor_Ux8_split1.layout]())
    # CHECK: ((2, 3):(8, 1))
    print(tensor_Ux8_split1.runtime_layout)
    # CHECK: 3.0 4.0 5.0
    # CHECK: 11.0 12.0 13.0
    print(tensor_Ux8_split1)

    # CHECK: ((-1, -1):(8, 1))
    print(materialize[tensor_Ux8_split2.layout]())
    # CHECK: ((2, 2):(8, 1))
    print(tensor_Ux8_split2.runtime_layout)
    # CHECK: 6.0 7.0
    # CHECK: 14.0 15.0
    print(tensor_Ux8_split2)

    comptime layout_8x2 = Layout(IntTuple(8, 2), IntTuple(2, 1))
    var tensor_8x2 = LayoutTensor[.float32, layout_8x2](ptr)
    var tensor_8x2_split1 = tensor_8x2.split[0, split_alignment=3](3, 1)
    var tensor_8x2_split2 = tensor_8x2.split[0, split_alignment=3](3, 2)

    # CHECK: ((-1, 2):(2, 1))
    print(materialize[tensor_8x2_split1.layout]())
    # CHECK: ((3, 2):(2, 1))
    print(tensor_8x2_split1.runtime_layout)
    # CHECK: 6.0 7.0
    # CHECK: 8.0 9.0
    # CHECK: 10.0 11.0
    print(tensor_8x2_split1)

    # CHECK: ((-1, 2):(2, 1))
    print(materialize[tensor_8x2_split2.layout]())
    # CHECK: ((2, 2):(2, 1))
    print(tensor_8x2_split2.runtime_layout)
    # CHECK: 12.0 13.0
    # CHECK: 14.0 15.0
    print(tensor_8x2_split2)


def main() raises:
    test_fill_and_print()
    test_set_and_get_items()
    test_tile()
    test_tile_and_distribute()
    test_tile_and_vectorize()
    test_copy_from()
    test_linspace_fill()
    test_random_fill()
    test_iterator()
    test_split()
