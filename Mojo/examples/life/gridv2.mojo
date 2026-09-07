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

from std import random
from std.collections import Optional

from std.memory import (
    ThinAllocation,
    alloc,
    dealloc,
    unsafe_memcpy,
    unsafe_memset_zero,
)


struct Grid[rows: Int, cols: Int](Copyable, Writable):
    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    comptime num_cells = Self.rows * Self.cols
    var data: ThinAllocation[Int8]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(out self):
        self.data = alloc[Int8]({count = self.num_cells}).into_thin()
        unsafe_memset_zero(self.data.unsafe_ptr(), self.num_cells)

    def __init__(out self, *, copy: Self):
        self.data = alloc[Int8]({count = self.num_cells}).into_thin()
        unsafe_memcpy(
            dest=self.data.unsafe_ptr(),
            src=copy.data.unsafe_ptr(),
            count=self.num_cells,
        )
        # The lifetime of `existing` continues unchanged

    def __deinit__(deinit self):
        dealloc(self.data^.unsafe_with_layout({count = Self.num_cells}))

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def random(seed: Optional[Int] = None) -> Self:
        if seed:
            random.seed(seed.value())
        else:
            random.seed()

        var grid = Self()
        random.randint(grid.data.unsafe_ptr(), grid.num_cells, 0, 1)

        return grid^

    # ===-------------------------------------------------------------------===#
    # Indexing
    # ===-------------------------------------------------------------------===#

    def __getitem__(self, row: Int, col: Int) -> Int8:
        return self.data.unsafe_ptr().unsafe_offset(row * Self.cols + col)[]

    def __setitem__(mut self, row: Int, col: Int, value: Int8) -> None:
        self.data.unsafe_ptr().unsafe_offset(row * Self.cols + col)[] = value

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def write_to(self, mut writer: Some[Writer]):
        for row in range(Self.rows):
            for col in range(Self.cols):
                if self[row, col] == 1:
                    writer.write_string("*")
                else:
                    writer.write_string(" ")
            if row != Self.rows - 1:
                writer.write_string("\n")

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def evolve(self) -> Self:
        var next_generation = Self()

        for row in range(Self.rows):
            # Calculate neighboring row indices, handling "wrap-around"
            var row_above = (row - 1) % Self.rows
            var row_below = (row + 1) % Self.rows

            for col in range(Self.cols):
                # Calculate neighboring column indices, handling "wrap-around"
                var col_left = (col - 1) % Self.cols
                var col_right = (col + 1) % Self.cols

                # Determine number of populated cells around the current cell
                var num_neighbors = (
                    self[row_above, col_left]
                    + self[row_above, col]
                    + self[row_above, col_right]
                    + self[row, col_left]
                    + self[row, col_right]
                    + self[row_below, col_left]
                    + self[row_below, col]
                    + self[row_below, col_right]
                )

                if num_neighbors | self[row, col] == 3:
                    next_generation[row, col] = 1

        return next_generation^
