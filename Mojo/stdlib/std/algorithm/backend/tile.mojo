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
"""Implementations of tile functions."""


# ===-----------------------------------------------------------------------===#
# tile
# ===-----------------------------------------------------------------------===#

comptime Static1DTileUnitFunc = def[width: Int](Int) -> None
"""Signature of a 1D tiled function with static tile size.

The function takes a static tile size parameter and an offset argument,
i.e. `func[tile_size: Int](offset: Int)`.
"""

comptime Dynamic1DTileUnitFunc = def(Int, Int) -> None
"""Signature of a 1D tiled function with dynamic tile size.

The function takes a dynamic tile size and an offset argument,
i.e. `func(offset: Int, tile_size: Int)`.
"""


comptime BinaryTile1DTileUnitFunc = def[width: Int](Int, Int) -> None
"""
Signature of a tiled function that performs some work with a dynamic tile size
and a secondary static tile size.
"""


@always_inline
def tile[
    tile_size_list: List[Int]
](
    offset: Int,
    upperbound: Int,
    workgroup_function: Some[Static1DTileUnitFunc],
):
    """A generator that launches work groups in specified list of tile sizes.

    A workgroup function is a function that can process a configurable
    consecutive "tile" of workload. E.g.
      `work_on[3](5)`
    should launch computation on item 5,6,7, and should be semantically
    equivalent to
      `work_on[1](5)`, `work_on[1](6)`, `work_on[1](7)`.

    This generator will try to proceed with the given list of tile sizes on the
    listed order. E.g.
        `tile[func, (3,2,1)](offset, upperbound)`
    will try to call `func[3]` starting from offset until remaining work is less
    than 3 from upperbound and then try `func[2]`, and then `func[1]`, etc.

    Parameters:
        tile_size_list: List of tile sizes to launch work.

    Args:
        offset: The initial index to start the work from.
        upperbound: The runtime upperbound that the work function should not
          exceed.
        workgroup_function: Workgroup function that processes one tile of
            workload.
    """

    # Initialize where to start on the overall work load.
    var current_offset: Int = offset

    comptime for tile_size in tile_size_list:
        # Process work with the tile size until there's not enough remaining work
        #  to fit in a tile.
        while current_offset <= upperbound - tile_size:
            workgroup_function[tile_size](current_offset)
            current_offset += tile_size


@always_inline
def tile(
    offset: Int,
    upperbound: Int,
    *tile_size_list: Int,
    workgroup_function: Some[Dynamic1DTileUnitFunc],
):
    """A generator that launches work groups in specified list of tile sizes.

    This is the version of tile generator for the case where work_group function
    can take the tile size as a runtime value.

    Args:
        offset: The initial index to start the work from.
        upperbound: The runtime upperbound that the work function should not
          exceed.
        tile_size_list: List of tile sizes to launch work.
        workgroup_function: Workgroup function that processes one tile of
          workload.
    """
    # Initialize the work_idx with the starting offset.
    var work_idx = offset
    # Iterate on the list of given tile sizes.
    for tile_size in tile_size_list:
        # Launch workloads on the current tile sizes until cannot proceed.
        while work_idx <= upperbound - tile_size:
            workgroup_function(work_idx, tile_size)
            work_idx += tile_size
    # Clean up the remaining workload with a residue tile that exactly equals to
    #  the remaining workload size.
    # Note: This is the key difference from the static version of tile
    #  generator.
    if work_idx < upperbound:
        workgroup_function(work_idx, upperbound - work_idx)


@always_inline
def tile[
    secondary_tile_size_list: List[Int],
    secondary_cleanup_tile: Int,
](
    offset: Int,
    upperbound: Int,
    *primary_tile_size_list: Int,
    primary_cleanup_tile: Int,
    workgroup_function: Some[BinaryTile1DTileUnitFunc],
):
    """A generator that launches work groups in specified list of tile sizes
    until the sum of primary_tile_sizes has exceeded the upperbound.

    Parameters:
        secondary_tile_size_list: List of static tile sizes to launch work.
        secondary_cleanup_tile: Last static tile to use when primary tile sizes
          don't fit exactly within the upperbound.

    Args:
        offset: The initial index to start the work from.
        upperbound: The runtime upperbound that the work function should not
          exceed.
        primary_tile_size_list: List of dynamic tile sizes to launch work.
        primary_cleanup_tile: Last dynamic tile to use when primary tile sizes
          don't fit exactly within the upperbound.
        workgroup_function: Workgroup function that processes one tile of
          workload.
    """
    var work_idx = offset
    comptime num_tiles = len(secondary_tile_size_list)

    comptime for i in range(num_tiles):
        comptime secondary_tile_size = secondary_tile_size_list[i]
        var primary_tile_size = primary_tile_size_list[i]

        while work_idx <= upperbound - primary_tile_size:
            workgroup_function[secondary_tile_size](work_idx, primary_tile_size)
            work_idx += primary_tile_size

    # launch the last cleanup tile
    if work_idx < upperbound:
        workgroup_function[secondary_cleanup_tile](
            work_idx, primary_cleanup_tile
        )


# ===-----------------------------------------------------------------------===#
# tile2d
# ===-----------------------------------------------------------------------===#


comptime Static2DTileUnitFunc = def[tile_x: Int, tile_y: Int](Int, Int) -> None
"""Signature of a 2D tiled function with static tile size.

The function takes static tile size parameters and offset arguments, i.e.
`func[tile_size_x: Int, tile_size_y: Int](offset_x: Int, offset_y: Int)`.
"""


@always_inline
def tile[
    tile_sizes_x: List[Int],
    tile_sizes_y: List[Int],
](
    offset_x: Int,
    offset_y: Int,
    upperbound_x: Int,
    upperbound_y: Int,
    workgroup_function: Some[Static2DTileUnitFunc],
):
    """Launches workgroup_function using the largest tile sizes possible in each
    dimension, starting from the x and y offset, until the x and y upperbounds
    are reached.

    Parameters:
        tile_sizes_x: List of tile sizes to use for the first parameter of workgroup_function.
        tile_sizes_y: List of tile sizes to use for the second parameter of workgroup_function.

    Args:
        offset_x: Initial x offset passed to workgroup_function.
        offset_y: Initial y offset passed to workgroup_function.
        upperbound_x: Max offset in x dimension passed to workgroup function.
        upperbound_y: Max offset in y dimension passed to workgroup function.
        workgroup_function: Function that is invoked for each tile and offset.
    """
    # Initialize where to start on the overall work load.
    var current_offset_y: Int = offset_y

    comptime for tile_size_y in tile_sizes_y:
        while current_offset_y <= upperbound_y - tile_size_y:
            var current_offset_x = offset_x

            comptime for tile_size_x in tile_sizes_x:
                while current_offset_x <= upperbound_x - tile_size_x:
                    workgroup_function[tile_size_x, tile_size_y](
                        current_offset_x, current_offset_y
                    )
                    current_offset_x += tile_size_x

            current_offset_y += tile_size_y
