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

# DOC: max/layout/tensors.mdx

from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    print_layout,
    UNKNOWN_VALUE,
    RuntimeLayout,
    RuntimeTuple,
)
from std.math import ceildiv
from std.collections import Set, Array

from layout.layout_tensor import LayoutTensorIter, _compute_distribute_layout
from layout.layout import (
    tile_to_shape,
    blocked_product,
    crd2idx,
    idx2crd,
    coalesce,
)
from layout.int_tuple import flatten
from std.memory import Pointer, unsafe_memset
from std.testing import assert_equal
from std.utils import Index, IndexList


def accessing_tensor_elements_example() raises:
    comptime rows = 4
    comptime columns = 8
    comptime layout = Layout.row_major(rows, columns)
    var storage = Array[Float32, rows * columns](uninitialized=True)
    for i in range(rows * columns):
        storage[i] = Float32(i)
    var tensor = LayoutTensor[.float32, layout](storage)

    var row, col = 0, 1
    # start-access-example-1
    var element = tensor[row, col][
        0
    ]  # element is guaranteed to be a scalar value
    # end-access-example-1

    assert_equal(element, 1)
    row, col = 0, 0

    # start-access-example-2
    var elements = tensor.load[4](row, col)
    elements = elements * 2
    tensor.store(row, col, elements)
    # end-access-example-2

    element = tensor[0, 2][0]
    assert_equal(element, 4)


def accessing_nested_tensor_elements_example() raises:
    comptime rows = 4
    comptime columns = 6
    comptime tiler = Layout.row_major(2, 3)
    comptime layout = blocked_product(Layout.col_major(2, 2), tiler)
    var storage = Array[Float32, rows * columns](uninitialized=True)
    for i in range(rows * columns):
        storage[i] = Float32(i)
    var tensor = LayoutTensor[.float32, layout](storage)

    # start-access-nested-tensor-example
    var element = tensor[1, 0, 0, 1][0]
    # end-access-nested-tensor-example

    assert_equal(element, 5)


def layout_tensor_on_cpu_example() raises:
    # start-layout-tensor-on-cpu-example
    comptime rows = 8
    comptime columns = 16
    comptime layout = Layout.row_major(rows, columns)
    var storage = Array[Float32, rows * columns](uninitialized=True)
    var tensor = LayoutTensor[.float32, layout](storage)
    # end-layout-tensor-on-cpu-example
    assert_equal(tensor.size(), rows * columns)
    _ = tensor


def layout_tensor_from_pointer_example() raises:
    # start-layout-tensor-from-pointer-example
    comptime rows = 1024
    comptime columns = 1024
    comptime buf_size = rows * columns
    comptime layout = Layout.row_major(rows, columns)
    var ptr = alloc[Float32]({count = buf_size}).unsafe_leak()
    unsafe_memset(ptr, 0, buf_size)
    var tensor = LayoutTensor[.float32, layout](ptr)
    # end-layout-tensor-from-pointer-example
    assert_equal(tensor.size(), rows * columns)
    _ = tensor
    ptr.unsafe_free()


def layout_tensor_tile_example() raises:
    # start-layout-tensor-tile-example
    comptime rows = 2
    comptime columns = 4
    comptime tile_size = 32
    comptime tile_layout = Layout.row_major(tile_size, tile_size)
    comptime tiler_layout = Layout.row_major(rows, columns)
    comptime tiled_layout = blocked_product(tile_layout, tiler_layout)
    var storage = Array[Float32, tiled_layout.size()](uninitialized=True)
    for i in range(comptime (tiled_layout.size())):
        storage[i] = Float32(i)
    var tensor = LayoutTensor[.float32, tiled_layout](storage)
    var tile = tensor.tile[32, 32](0, 1)
    # end-layout-tensor-tile-example
    assert_equal(tile[0, 0][0], Float32(tile_size * tile_size))
    # start-layout-tensor-tile-example-2
    var my_tile: tensor.TileType[tile_size, tile_size]
    for i in range(rows):
        for j in range(columns):
            my_tile = tensor.tile[tile_size, tile_size](i, j)
            # ... do something with the tile ...
            # end-layout-tensor-tile-example-2
            _ = my_tile


# Iterates through a block of memory one tile at a time.
# This essentially treats the memory as a flat array of
# tiles (or a 2D row-major matrix of tiles).
def layout_tensor_iterator_example() raises:
    # start-layout-tensor-iterator-example-1
    comptime buf_size = 128
    var storage = Array[Int16, buf_size](uninitialized=True)
    for i in range(buf_size):
        storage[i] = Int16(i)
    comptime tile_layout = Layout.row_major(4, 4)
    var iter = LayoutTensorIter[.int16, tile_layout](
        storage.unsafe_ptr(), buf_size
    )

    for i in range(ceildiv(buf_size, comptime (tile_layout.size()))):
        var tile = iter[]
        # ... do something with tile
        iter += 1
        # end-layout-tensor-iterator-example-1
        comptime tile_size = tile_layout.size()
        assert_equal(tile[0, 0][0], Int16(i * tile_size))


def layout_tensor_iterator_example2() raises:
    # TODO: set up a tiled layout tensor as input and
    # validate output.
    comptime rows = 4
    comptime cols = 8
    comptime size = rows * cols
    comptime tile_size = 2
    var storage = Array[Int32, size](uninitialized=True)
    for i in range(size):
        storage[i] = Int32(i)

    comptime layout = Layout.row_major(rows, cols)
    var tensor = LayoutTensor[.int32, layout, masked=True](storage)
    # start-layout-tensor-iterator-example-2
    # given a tensor of size rows x cols
    comptime num_row_tiles = ceildiv(rows, tile_size)
    comptime num_col_tiles = ceildiv(cols, tile_size)

    for i in range(num_row_tiles):
        var iter = tensor.tiled_iterator[tile_size, tile_size, axis=1](i, 0)

        for _ in range(num_col_tiles):
            var tile = iter[]
            # ... do something with the tile
            iter += 1
            # end-layout-tensor-iterator-example-2
            _ = tile


def main() raises:
    accessing_tensor_elements_example()
    accessing_nested_tensor_elements_example()
    layout_tensor_on_cpu_example()
    layout_tensor_from_pointer_example()
    layout_tensor_tile_example()
    layout_tensor_iterator_example()
    layout_tensor_iterator_example2()
