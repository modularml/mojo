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
# tests.mojo
# Tests for get-started.mdx code examples.
#
# `Grid` comes from `grid.mojo` in this directory, which is the finished
# struct the tutorial accretes across its later steps. The earlier steps
# have no module to import, so their snippets are reproduced here as
# written on the page.
#
# Not tested (no runnable behavior to assert from this file):
#   - `print_grid()` output itself (writes to stdout). The rendering rule
#     is pinned instead by `render()` below, which is the same loop
#     accumulating into a `String`.
#   - `from grid import Grid` as a source-layout step (the import at the
#     top of this file covers it)
#   - `seed()` producing distinct runs, and the specific cells
#     `random_grid()` fills (random); only its invariants are asserted
#   - the `main()` animation loop: ANSI escape codes, `sleep()`, and the
#     300-generation loop have no assertable result
from std.random import seed
from std.testing import assert_equal, assert_false, assert_true

from grid import Grid

# --- Helper: the `print_grid()` loop, accumulating instead of printing ---


def render(cells: List[Int], num_cols: Int) -> String:
    var out = String()
    for index in range(len(cells)):
        out += "X" if cells[index] else "."
        if index % num_cols == (num_cols - 1):
            out += "\n"
    return out


# The grid the tutorial prints in its first two steps.
comptime GLIDER_8X8 = String(
    "..X.....\n"
    "X.X.....\n"
    ".XX.....\n"
    "........\n"
    "........\n"
    "........\n"
    "........\n"
    "........\n"
)


# --- Game state: a list, a lambda, and tuple coordinates ---


def test_glider_grid() raises:
    var count, num_cols = 64, 8
    var glider_grid: List[Int] = List[Int](length=64, fill=0)

    # Convert (x, y) coordinates to a linear index in the grid
    var to_index = lambda (x: Int, y: Int) -> Int: y * num_cols + x

    for coord in [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
        glider_grid[to_index(coord[0], coord[1])] = 1

    assert_equal(count, 64)
    assert_equal(len(glider_grid), 64)
    # The five glider cells, and only those, are live.
    assert_equal(render(glider_grid, num_cols), GLIDER_8X8)
    var live = 0
    for index in range(count):
        live += glider_grid[index]
    assert_equal(live, 5)


def test_to_index_lambda() raises:
    var num_cols = 8
    var to_index = lambda (x: Int, y: Int) -> Int: y * num_cols + x

    # x is the column, y is the row.
    assert_equal(to_index(0, 0), 0)
    assert_equal(to_index(2, 0), 2)
    assert_equal(to_index(0, 1), 8)
    assert_equal(to_index(7, 7), 63)


def test_list_expression() raises:
    var values: List[Int] = [12, -7, 64]  # This is a list expression
    assert_equal(len(values), 3)
    assert_equal(values[0], 12)
    assert_equal(values[1], -7)
    assert_equal(values[2], 64)


def test_tuple_mixed_types() raises:
    # Tuples have a fixed number of elements and can contain different types.
    var pair = (0, 1)
    assert_equal(pair[0], 0)
    assert_equal(pair[1], 1)
    var mixed = (1, "two", 3.0)
    assert_equal(mixed[0], 1)
    assert_equal(mixed[1], "two")
    assert_equal(mixed[2], 3.0)


# --- Add reusable printing: the display loop moved into a function ---


def print_grid(grid: List[Int], num_cols: Int):
    for index in range(len(grid)):
        print("X" if grid[index] else ".", end="")
        if index % num_cols == (num_cols - 1):
            print()


def test_print_grid_runs() raises:
    # `print_grid()` writes to stdout, so this only checks that the
    # extracted function compiles and runs over the tutorial's grid.
    var glider_grid: List[Int] = List[Int](length=64, fill=0)
    for coord in [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
        glider_grid[coord[1] * 8 + coord[0]] = 1
    print_grid(glider_grid, 8)


# --- Add lookups: the constants and conversions hoisted to `comptime` ---

comptime count = 64
comptime num_cols = 8
comptime num_rows = count // num_cols

comptime to_index = lambda (x: Int, y: Int) -> Int: (y * num_cols + x)
comptime to_coord = lambda (i: Int) -> Tuple[Int, Int]: (
    (i % num_cols, i // num_cols)
)


def test_comptime_constants() raises:
    assert_equal(count, 64)
    assert_equal(num_cols, 8)
    assert_equal(num_rows, 8)


def test_comptime_conversions() raises:
    assert_equal(to_index(0, 1), 8)
    assert_equal(to_index(2, 2), 18)

    # The declared `Tuple[Int, Int]` return type converts `i // num_cols`
    # back to an `Int`, truncating toward negative infinity.
    var x, y = to_coord(18)
    assert_equal(x, 2)
    assert_equal(y, 2)
    var last_x, last_y = to_coord(63)
    assert_equal(last_x, 7)
    assert_equal(last_y, 7)

    # Every index round-trips through both directions.
    for i in range(count):
        var cx, cy = to_coord(i)
        assert_equal(to_index(cx, cy), i)


def test_comptime_grid_setup() raises:
    var glider_grid: List[Int] = List[Int](length=64, fill=0)
    for coord in [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
        glider_grid[to_index(coord[0], coord[1])] = 1
    assert_equal(render(glider_grid, num_cols), GLIDER_8X8)


# --- Define a `Grid` type: parameters size the instance ---


def test_grid_init() raises:
    var glider_grid = Grid[8, 8]()
    assert_equal(glider_grid.count, 64)
    assert_equal(len(glider_grid.cells), 64)
    # Every field is set before the initializer returns; cells start dead.
    for index in range(glider_grid.count):
        assert_equal(glider_grid.cells[index], 0)


def test_grid_parameters() raises:
    # `count` comes from the grid dimensions, so it tracks the parameters.
    assert_equal(Grid[4, 6]().count, 24)
    assert_equal(Grid[60, 20]().count, 1200)


# --- Add grid operations: indexing through `__getitem__`/`__setitem__` ---


def test_grid_indexing() raises:
    var glider_grid = Grid[8, 8]()
    glider_grid[(2, 1)] = 1
    assert_equal(glider_grid[(2, 1)], 1)
    # `Self.to_index()` puts (x, y) at y * num_cols + x.
    assert_equal(glider_grid.cells[10], 1)
    glider_grid[(2, 1)] = 0
    assert_equal(glider_grid[(2, 1)], 0)


def test_grid_static_conversions() raises:
    assert_equal(Grid[8, 8].to_index(2, 1), 10)
    var x, y = Grid[8, 8].to_coord(10)
    assert_equal(x, 2)
    assert_equal(y, 1)


def test_grid_glider_setup() raises:
    var glider_grid = Grid[8, 8]()
    for coord in [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
        glider_grid[coord] = 1  # Uses indexing
    assert_equal(render(glider_grid.cells, 8), GLIDER_8X8)


# --- Add some protection: coordinates are checked before indexing ---


def test_is_valid_coord() raises:
    var glider_grid = Grid[8, 8]()
    assert_true(glider_grid.is_valid_coord((0, 0)))
    assert_true(glider_grid.is_valid_coord((7, 7)))
    assert_false(glider_grid.is_valid_coord((-1, 0)))
    assert_false(glider_grid.is_valid_coord((0, -1)))
    assert_false(glider_grid.is_valid_coord((8, 0)))
    assert_false(glider_grid.is_valid_coord((0, 8)))


def test_setitem_out_of_bounds_is_a_no_op() raises:
    var glider_grid = Grid[8, 8]()
    glider_grid[(-1, 0)] = 1
    glider_grid[(0, -1)] = 1
    glider_grid[(8, 0)] = 1
    glider_grid[(0, 8)] = 1
    # Nothing was written, so no cell wrapped around into the grid.
    for index in range(glider_grid.count):
        assert_equal(glider_grid.cells[index], 0)


# --- Construct the random grid: only the invariants are deterministic ---


def test_random_grid_single_clump() raises:
    seed()
    # `clumps + 1` clumps are built, so 0 asks for one: a center cell plus
    # up to its eight neighbors.
    var grid = Grid[8, 8].random_grid(0)
    assert_equal(grid.count, 64)
    var live = 0
    for index in range(grid.count):
        live += grid.cells[index]
    assert_true(live >= 1)
    assert_true(live <= 9)


def test_random_grid_default_clumps() raises:
    seed()
    # The default is two, which builds three clumps.
    var grid = Grid[8, 8].random_grid()
    var live = 0
    for index in range(grid.count):
        live += grid.cells[index]
    assert_true(live >= 1)
    assert_true(live <= 27)


# --- Evolve: the three rules, applied to every interior cell ---


def test_evolve_still_life() raises:
    # A 2 x 2 block has each of its cells at exactly three neighbors,
    # so it survives unchanged.
    var grid = Grid[8, 8]()
    for coord in [(3, 3), (4, 3), (3, 4), (4, 4)]:
        grid[coord] = 1
    grid.evolve()
    assert_equal(grid[(3, 3)], 1)
    assert_equal(grid[(4, 3)], 1)
    assert_equal(grid[(3, 4)], 1)
    assert_equal(grid[(4, 4)], 1)


def test_evolve_blinker_oscillates() raises:
    # A vertical blinker turns horizontal, then back.
    var grid = Grid[8, 8]()
    for coord in [(3, 2), (3, 3), (3, 4)]:
        grid[coord] = 1

    grid.evolve()
    assert_equal(grid[(2, 3)], 1)
    assert_equal(grid[(3, 3)], 1)
    assert_equal(grid[(4, 3)], 1)
    assert_equal(grid[(3, 2)], 0)
    assert_equal(grid[(3, 4)], 0)

    grid.evolve()
    assert_equal(grid[(3, 2)], 1)
    assert_equal(grid[(3, 3)], 1)
    assert_equal(grid[(3, 4)], 1)
    assert_equal(grid[(2, 3)], 0)
    assert_equal(grid[(4, 3)], 0)


def test_evolve_underpopulation() raises:
    # A lone live cell has no neighbors, so it dies.
    var grid = Grid[8, 8]()
    grid[(3, 3)] = 1
    grid.evolve()
    assert_equal(grid[(3, 3)], 0)


def test_evolve_birth() raises:
    # A dead cell with exactly three live neighbors becomes alive.
    var grid = Grid[8, 8]()
    for coord in [(3, 2), (4, 2), (3, 3)]:
        grid[coord] = 1
    assert_equal(grid[(4, 3)], 0)
    grid.evolve()
    assert_equal(grid[(4, 3)], 1)


def test_evolve_edges_go_dead() raises:
    # Edges are excluded from evolution and always go dead.
    var grid = Grid[8, 8]()
    for coord in [(0, 0), (7, 0), (0, 7), (7, 7), (4, 0), (0, 4)]:
        grid[coord] = 1
    grid.evolve()
    for x in range(8):
        assert_equal(grid[(x, 0)], 0)
        assert_equal(grid[(x, 7)], 0)
    for y in range(8):
        assert_equal(grid[(0, y)], 0)
        assert_equal(grid[(7, y)], 0)


def main() raises:
    test_glider_grid()
    test_to_index_lambda()
    test_list_expression()
    test_tuple_mixed_types()
    test_print_grid_runs()
    test_comptime_constants()
    test_comptime_conversions()
    test_comptime_grid_setup()
    test_grid_init()
    test_grid_parameters()
    test_grid_indexing()
    test_grid_static_conversions()
    test_grid_glider_setup()
    test_is_valid_coord()
    test_setitem_out_of_bounds_is_a_no_op()
    test_random_grid_single_clump()
    test_random_grid_default_clumps()
    test_evolve_still_life()
    test_evolve_blinker_oscillates()
    test_evolve_underpopulation()
    test_evolve_birth()
    test_evolve_edges_go_dead()
