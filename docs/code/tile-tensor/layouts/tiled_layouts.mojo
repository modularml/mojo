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
# DOC: max/tile-tensor/layouts.mdx

from layout import Coord, coord, Idx, print_layout, col_major, row_major
from layout.tile_layout import (
    Layout,
    blocked_product,
    col_major_nested,
    row_major_nested,
    zipped_divide,
)


def use_layout_constructor():
    print("layout constructor")
    var tiled_layout = Layout(
        Coord(coord[3, 2], coord[2, 5]),  # shape
        Coord(coord[1, 6], coord[3, 12]),  # strides
    )
    print_layout(tiled_layout.to_layout())
    print()


# start-blocked-product-example
def use_blocked_product():
    print("blocked product")
    # Define 3x2 tile
    var tile = col_major[3, 2]()
    # Define a 2x5 tiler
    var tiler = col_major[2, 5]()
    var blocked = blocked_product(tile, tiler)

    print("Tile:")
    print_layout(tile.to_layout())
    print("\nTiler:")
    print_layout(tiler.to_layout())
    print("\nTiled layout:")
    print_layout(blocked.to_layout())
    print()


# end-blocked-product-example


def use_zipped_divide():
    print("zipped divide")
    # Create layouts
    # start-zipped-divide-example
    var base = row_major[6, 4]()
    var result = zipped_divide[coord[2, 2]](base)
    print_layout(base.to_layout())
    print_layout(result.to_layout())
    # end-zipped-divide-example
    print(result(coord[1, 1]))
    print(result(coord[1]))
    print(result(Coord(coord[1, 0], coord[1, 0])))


def minimal_repro():
    var base = row_major[6, 8]()
    var result = zipped_divide[coord[2, 2]](base)
    var linear_idx = Int(result(coord[1, 1]))
    var natural_coords = result.idx2crd(linear_idx)
    print(linear_idx, natural_coords)  # 24, ((1, 0), (1, 0))


def use_nested_layouts():
    print("nested layouts")
    # start-nested-layout-example
    # A re-nested row-major layout: a 2x2 outer grid of 3x4 inner
    # fragments. Shape ((2, 3), (2, 4)) flattens to (2, 3, 2, 4),
    # producing row-major strides (24, 8, 4, 1), re-nested as
    # ((24, 8), (4, 1)).
    var nested = row_major_nested(
        Coord(Coord(Idx[2], Idx[3]), Coord(Idx[2], Idx[4]))
    )
    print_layout(nested.to_layout())

    # The column-major variant re-nests col-major strides over the same
    # flattened shape: (1, 2, 6, 12), re-nested as ((1, 2), (6, 12)).
    var nested_col = col_major_nested(
        Coord(Coord(Idx[2], Idx[3]), Coord(Idx[2], Idx[4]))
    )
    print_layout(nested_col.to_layout())
    # end-nested-layout-example
    print()


def main() raises:
    use_layout_constructor()
    use_blocked_product()
    use_zipped_divide()
    use_nested_layouts()
