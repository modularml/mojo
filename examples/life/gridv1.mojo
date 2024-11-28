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

from collections import Optional
import random


@value
struct Grid(StringableRaising):
    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var rows: Int
    var cols: Int
    var data: List[List[Int]]

    # ===-------------------------------------------------------------------===#
    # Indexing
    # ===-------------------------------------------------------------------===#

    def __getitem__(self, row: Int, col: Int) -> Int:
        return self.data[row][col]

    def __setitem__(inout self, row: Int, col: Int, value: Int) -> None:
        self.data[row][col] = value

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __str__(self) -> String:
        str = String()
        for row in range(self.rows):
            for col in range(self.cols):
                if self[row, col] == 1:
                    str += "*"
                else:
                    str += " "
            if row != self.rows - 1:
                str += "\n"
        return str

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def glider() -> Self:
        var glider = List(
            List(0, 1, 0, 0, 0, 0, 0, 0),
            List(0, 0, 1, 0, 0, 0, 0, 0),
            List(1, 1, 1, 0, 0, 0, 0, 0),
            List(0, 0, 0, 0, 0, 0, 0, 0),
            List(0, 0, 0, 0, 0, 0, 0, 0),
            List(0, 0, 0, 0, 0, 0, 0, 0),
            List(0, 0, 0, 0, 0, 0, 0, 0),
            List(0, 0, 0, 0, 0, 0, 0, 0),
        )
        return Grid(8, 8, glider)

    @staticmethod
    def random(rows: Int, cols: Int, seed: Optional[Int] = None) -> Self:
        if seed:
            random.seed(seed.value())
        else:
            random.seed()

        data = List[List[Int]]()

        for row in range(rows):
            row_data = List[Int]()
            for col in range(cols):
                row_data.append(int(random.random_si64(0, 1)))
            data.append(row_data)

        return Self(rows, cols, data)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def evolve(self) -> Self:
        next_generation = List[List[Int]]()

        for row in range(self.rows):
            row_data = List[Int]()

            row_above = (row - 1) % self.rows
            row_below = (row + 1) % self.rows

            for col in range(self.cols):
                col_left = (col - 1) % self.cols
                col_right = (col + 1) % self.cols

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
                    row_data.append(1)
                else:
                    row_data.append(0)

            next_generation.append(row_data)

        return Self(self.rows, self.cols, next_generation)
