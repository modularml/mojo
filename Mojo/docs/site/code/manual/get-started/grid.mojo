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
from std.random import random_si64  # Add this import to grid.mojo


struct Grid[num_cols: Int, num_rows: Int]:
    var cells: List[Int]
    var next_cells: List[Int]
    var count: Int

    def __init__(out self):
        self.count = Self.num_cols * Self.num_rows
        self.cells = List[Int](length=self.count, fill=0)
        self.next_cells = List[Int](length=self.count, fill=0)

    comptime to_index = lambda (x: Int, y: Int) -> Int: (y * Self.num_cols + x)

    comptime to_coord = lambda (i: Int) -> Tuple[Int, Int]: (
        (i % Self.num_cols, i // Self.num_cols)
    )

    def print_grid(self):
        for index in range(len(self.cells)):
            print("X" if self.cells[index] else ".", end="")
            if index % Self.num_cols == (Self.num_cols - 1):
                print()

    def is_valid_coord(self, coord: Tuple[Int, Int]) -> Bool:
        return not (
            coord[0] < 0
            or coord[0] >= Self.num_cols
            or coord[1] < 0
            or coord[1] >= Self.num_rows
        )

    def __setitem__(mut self, coord: Tuple[Int, Int], value: Int):
        if not self.is_valid_coord(coord):
            return  # no op
        self.cells[Self.to_index(coord[0], coord[1])] = value

    def __getitem__(self, coord: Tuple[Int, Int]) -> Int:
        return self.cells[Self.to_index(coord[0], coord[1])]

    @staticmethod
    def random_grid(var clumps: Int = 2) -> Self:
        var grid = Self()
        for _ in range(clumps + 1):
            var idx = Int(random_si64(0, Int64(grid.count) - 1))
            var x, y = Self.to_coord(idx)
            grid[(x, y)] = 1  # Fill index cell

            for dx in range(-1, 2):  # -1, 0, or 1
                for dy in range(-1, 2):
                    if not grid.is_valid_coord((x + dx, y + dy)):
                        continue
                    if random_si64(0, 3) > 0:
                        continue  # 75% skip
                    grid[(x + dx, y + dy)] = 1
        return grid^

    def evolve(mut self, var i: Int):
        var is_live = Bool(self.cells[i])
        self.next_cells[i] = 0

        # Count the neighbors
        var ncount = -1 if is_live else 0
        var x, y = Self.to_coord(i)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                var nx, ny = x + dx, y + dy
                if not self.is_valid_coord((nx, ny)):
                    continue
                ncount += self.cells[Self.to_index(nx, ny)]

        # Live cell stays alive with two or three live neighbors
        if is_live and (ncount == 2 or ncount == 3):
            self.next_cells[i] = 1
        # Dead cell becomes alive with exactly three live neighbors
        elif not is_live and ncount == 3:
            self.next_cells[i] = 1

    def evolve(mut self):
        for i in range(self.count):
            var x, y = Self.to_coord(i)
            # Edges are excluded from evolution and will always go dead.
            if (
                x == 0
                or y == 0
                or x == Self.num_cols - 1
                or y == Self.num_rows - 1
            ):
                self.next_cells[i] = 0  # Edges go dead
                continue
            self.evolve(i)

        # Swap the current and next cell states
        self.cells = self.next_cells^  # Transfer
        self.next_cells = List[Int](length=self.count, fill=0)  # Fresh
