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

    def __setitem__(mut self, row: Int, col: Int, value: Int) -> None:
        self.data[row][col] = value

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __str__(self) -> String:
        # Create an empty String
        str = String()

        # Iterate through rows 0 through rows-1
        for row in range(self.rows):
            # Iterate through columns 0 through cols-1
            for col in range(self.cols):
                if self[row, col] == 1:
                    str += "*"  # If cell is populated, append an asterisk
                else:
                    str += " "  # If cell is not populated, append a space
            if row != self.rows - 1:
                str += "\n"  # Add a newline between rows, but not at the end
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
            # Seed the random number generator using the current time.
            random.seed()

        data = List[List[Int]]()

        for row in range(rows):
            row_data = List[Int]()
            for col in range(cols):
                # Generate a random 0 or 1 and append it to the row.
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

            # Calculate neighboring row indices, handling "wrap-around"
            row_above = (row - 1) % self.rows
            row_below = (row + 1) % self.rows

            for col in range(self.cols):
                # Calculate neighboring column indices, handling "wrap-around"
                col_left = (col - 1) % self.cols
                col_right = (col + 1) % self.cols

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

                # Determine the state of the current cell for the next generation
                new_state = 0
                if self[row, col] == 1 and (
                    num_neighbors == 2 or num_neighbors == 3
                ):
                    new_state = 1
                elif self[row, col] == 0 and num_neighbors == 3:
                    new_state = 1
                row_data.append(new_state)

            next_generation.append(row_data)

        return Self(self.rows, self.cols, next_generation)
