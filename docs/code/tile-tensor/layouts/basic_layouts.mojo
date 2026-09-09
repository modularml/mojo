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
from layout import Coord, coord, Idx, print_layout
from layout.tile_layout import Layout, col_major, row_major


def row_and_column_major():
    print("row major and column major")
    # start-row-major-and-column-major
    var row_major2x4 = row_major[2, 4]()
    var col_major6x6 = col_major[6, 6]()
    # end-row-major-and-column-major

    # start-nd-layout-example
    comptime row_major3d = row_major[4, 4, 4]()
    comptime col_major3d = col_major[4, 4, 4]()
    # end-nd-layout-example

    # start-coord-layout-example
    var from_coords = row_major(coord[6, 8])
    # end-coord-layout-example
    _, _, _ = from_coords, row_major2x4, col_major6x6


def coords_to_index():
    print("coordinates to index")
    # start-print-layout-example
    var row_major3x4 = row_major[3, 4]()
    print_layout(row_major3x4.to_layout())
    # end-print-layout-example

    # start-coords-to-index-example
    var coords = coord[1, 1]
    var idx = row_major3x4(coords)
    print("index at (1, 1): ", idx)
    print("coordinates at index 7:", row_major3x4.idx2crd(7))
    # end-coords-to-index-example
    print()


def nested_modes():
    print("nested modes")
    var layout_a = Layout(coord[4, 4], coord[4, 1])
    print_layout(layout_a.to_layout())
    print()
    var layout_b = Layout(
        Coord(coord[2, 2], coord[2, 2]),
        Coord(coord[1, 4], coord[2, 8]),
    )
    print_layout(layout_b.to_layout())
    print()


def specifying_coords():
    var a, b, c = 4, 6, 8
    # start-comptime-runtime-coords
    var comptime_coords = coord[4, 4]
    var runtime_coords = Coord(Int32(a), Int32(b), Int32(c))
    # end-comptime-runtime-coords
    _, _, _, _, _ = (a, b, c, comptime_coords, runtime_coords)


def dynamic_dimensions():
    print("dynamic dimensions")
    comptime columns = 8
    var rows = 4
    comptime comptime_int = Idx[columns]  # Compile-time int
    comptime comptime_int2 = Idx[4]  # Compile-time int from Int literal
    var runtime_int = rows  # Run-time int from dynamic value
    var mixed_shape = Coord((rows, Idx[columns]))  # Define a mixed Coord
    var mixed_layout = row_major((rows, Idx[columns]))
    print_layout(mixed_layout.to_layout())
    var shape = Coord(rows, Idx[columns])
    var stride = coord[8, 1]
    var layout2 = Layout(shape, stride)
    print_layout(layout2.to_layout())
    # start-runtime-layout-example
    # Layout with compile-time dimensions
    comptime row_major_comptime = row_major(coord[16, 8])

    # Layout with run-time dimensions
    var a, b = 4, 8
    var row_major_runtime = row_major(Coord(Int32(a), Int32(b)))

    # Mixed layout with one run-time dimension and one compile-time dimension
    var row_major_mixed = row_major((rows, Idx[columns]))
    # end-runtime-layout-example
    _, _, _, _, _ = (
        runtime_int,
        mixed_shape,
        row_major_comptime,
        row_major_runtime,
        row_major_mixed,
    )


def nested_coords():
    print("more coords")
    var rows, cols = 4, 8
    var runtime_coord = Coord(Int64(rows), Int64(cols))
    _ = runtime_coord

    # start-nested-coord-example
    var shape1 = Coord((Idx[6], Idx[8]))
    var shape2 = Coord((coord[2, 2], coord[3, 4]))
    var shape3 = Coord((shape1, shape2))
    print(shape3)
    # end-nested-coord-example


def main() raises:
    row_and_column_major()
    coords_to_index()
    nested_modes()
    specifying_coords()
    dynamic_dimensions()
    nested_coords()
