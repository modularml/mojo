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

from gridv1 import Grid
from testing import *

var data4x4 = List(
    List(0, 1, 1, 0),
    List(1, 1, 0, 0),
    List(0, 0, 1, 1),
    List(1, 0, 0, 1),
)
var str4x4 = " ** \n**  \n  **\n*  *"


def test_gridv1_init():
    grid = Grid(4, 4, data4x4)
    assert_equal(4, grid.rows)
    assert_equal(4, grid.cols)
    for row in range(4):
        assert_equal(data4x4[row], grid.data[row])


def test_gridv1_index():
    grid = Grid(4, 4, data4x4)
    for row in range(4):
        for col in range(4):
            assert_equal(data4x4[row][col], grid[row, col])
            grid[row, col] = 1
            assert_equal(1, grid[row, col])
            grid[row, col] = 0
            assert_equal(0, grid[row, col])


def test_gridv1_str():
    grid = Grid(4, 4, data4x4)
    grid_str = str(grid)
    assert_equal(str4x4, grid_str)


def test_gridv1_evolve():
    data_gen2 = List(
        List(0, 0, 1, 0),
        List(1, 0, 0, 0),
        List(0, 0, 1, 0),
        List(1, 0, 0, 0),
    )
    data_gen3 = List(
        List(0, 1, 0, 1),
        List(0, 1, 0, 1),
        List(0, 1, 0, 1),
        List(0, 1, 0, 1),
    )

    grid_gen1 = Grid(4, 4, data4x4)

    grid_gen2 = grid_gen1.evolve()
    for row in range(4):
        for col in range(4):
            assert_equal(data_gen2[row][col], grid_gen2[row, col])

    grid_gen3 = grid_gen2.evolve()
    for row in range(4):
        for col in range(4):
            assert_equal(data_gen3[row][col], grid_gen3[row, col])
