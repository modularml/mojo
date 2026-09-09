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

# DOC: max/layout/layouts.mdx

from layout import IntTuple, Layout, print_layout
from layout.layout import (
    blocked_product,
    make_layout,
    make_ordered_layout,
    tile_to_shape,
    zipped_divide,
)


def use_layout_constructor():
    print("layout constructor")
    var tiled_layout = Layout(
        [[3, 2], [2, 5]],  # shape
        [[1, 6], [3, 12]],  # strides
    )
    print_layout(tiled_layout)
    print()


def use_tile_to_shape():
    print("tile to shape")
    var tts = tile_to_shape(Layout.col_major(3, 2), [6, 10])
    print_layout(tts)
    print()


def use_blocked_product():
    print("blocked product")
    # Define 2x3 tile
    var tile = Layout.col_major(3, 2)
    # Define a 2x5 tiler
    var tiler = Layout.col_major(2, 5)
    var blocked = blocked_product(tile.copy(), tiler.copy())

    print("Tile:")
    print_layout(tile)
    print("\nTiler:")
    print_layout(tiler)
    print("\nTiled layout:")
    print(blocked)
    print()


def use_make_ordered_layout():
    print("make ordered layout")
    var ordered = make_ordered_layout(
        [[3, 2], [2, 5]],  # shape
        [[0, 2], [1, 3]],  # order
    )
    print(ordered)


def use_make_layout():
    print("make layout")
    var layout1 = Layout([2, 3], [3, 1])
    var layout2 = Layout([4, 5], [5, 1])
    var combined = make_layout(layout1, layout2)
    print_layout(combined)


def use_zipped_divide():
    # from layout import Layout, IntTuple
    # from layout.layout import zipped_divide
    print("zipped divide")
    # Create layouts
    var base = Layout.row_major(6, 8)
    var pattern = Layout([2, 2])
    var result = zipped_divide(base, pattern)
    print_layout(result)


def main() raises:
    use_layout_constructor()
    use_tile_to_shape()
    use_blocked_product()
    use_make_ordered_layout()
    use_make_layout()
    use_zipped_divide()
