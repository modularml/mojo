# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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

import random
from collections import Optional

from memory import UnsafePointer, memcpy, memset_zero


struct Grid[rows: Int, cols: Int](StringableRaising):
    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    alias num_cells = rows * cols
    var data: UnsafePointer[Int8]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(out self):
        self.data = UnsafePointer[Int8].alloc(self.num_cells)
        memset_zero(self.data, self.num_cells)

    fn __copyinit__(out self, existing: Self):
        self.data = UnsafePointer[Int8].alloc(self.num_cells)
        memcpy(dest=self.data, src=existing.data, count=self.num_cells)
        # The lifetime of `existing` continues unchanged

    fn __moveinit__(out self, owned existing: Self):
        self.data = existing.data
        # Then the lifetime of `existing` ends here, but
        # Mojo does NOT call its destructor

    fn __del__(owned self):
        for i in range(self.num_cells):
            (self.data + i).destroy_pointee()
        self.data.free()

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def random(seed: Optional[Int] = None) -> Self:
        if seed:
            random.seed(seed.value())
        else:
            random.seed()

        grid = Self()
        random.randint(grid.data, grid.num_cells, 0, 1)

        return grid

    # ===-------------------------------------------------------------------===#
    # Indexing
    # ===-------------------------------------------------------------------===#

    def __getitem__(self, row: Int, col: Int) -> Int8:
        return (self.data + row * cols + col)[]

    def __setitem__(mut self, row: Int, col: Int, value: Int8) -> None:
        (self.data + row * cols + col)[] = value

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __str__(self) -> String:
        str = String()
        for row in range(rows):
            for col in range(cols):
                if self[row, col] == 1:
                    str += "*"
                else:
                    str += " "
            if row != rows - 1:
                str += "\n"
        return str

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def evolve(self) -> Self:
        next_generation = Self()

        for row in range(rows):
            # Calculate neighboring row indices, handling "wrap-around"
            row_above = (row - 1) % rows
            row_below = (row + 1) % rows

            for col in range(cols):
                # Calculate neighboring column indices, handling "wrap-around"
                col_left = (col - 1) % cols
                col_right = (col + 1) % cols

                # Determine number of populated cells around the current cell
                num_neighbors = (
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

        return next_generation
