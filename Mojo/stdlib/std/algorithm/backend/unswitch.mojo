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
"""Implementations of unswitch and tile_and_unswitch functions."""

from .tile import Static1DTileUnitFunc, tile


# ===-----------------------------------------------------------------------===#
# Unswitch
# ===-----------------------------------------------------------------------===#

# Signature of a function that unswitch can take.
comptime SwitchedFunction = def[sw: Bool]() raises -> None
"""Signature of a function that unswitch can take."""

# Version of unswitch supporting 2 predicates.
comptime SwitchedFunction2 = def[sw0: Bool, sw1: Bool]() -> None
"""Signature for unswitch supporting 2 predicates."""


@always_inline
def unswitch(
    dynamic_switch: Bool, switched_func: Some[SwitchedFunction]
) raises:
    """Performs a functional unswitch transformation.

    Unswitch is a simple pattern that is similar idea to loop unswitching
    pass but extended to functional patterns. The pattern facilitates the
    following code transformation that reduces the number of branches in the
    generated code

    Before:

        for i in range(...)
            if i < xxx:
                ...

    After:

        if i < ...
            for i in range(...)
                ...
        else
            for i in range(...)
                if i < xxx:
                    ...

    This unswitch function generalizes that pattern with the help of meta
    parameters and can be used to perform both loop unswitching and other
    tile predicate lifting like in simd and amx.

    TODO: Generalize to support multiple predicates.
    TODO: Once nested lambdas compose well should make unswitch compose with
    tile in an easy way.

    Args:
        dynamic_switch: The dynamic condition that enables the unswitched code
          path.
        switched_func: The function containing the inner loop logic that can be
          unswitched.

    Raises:
        If the operation fails.
    """
    if dynamic_switch:
        switched_func[True]()
    else:
        switched_func[False]()


@always_inline
def unswitch(
    dynamic_switch: Bool, switched_func: Some[def[sw: Bool]() -> None]
):
    """Performs a functional unswitch transformation.

    Unswitch is a simple pattern that is similar idea to loop unswitching
    pass but extended to functional patterns. The pattern facilitates the
    following code transformation that reduces the number of branches in the
    generated code

    Before:

        for i in range(...)
            if i < xxx:
                ...

    After:

        if i < ...
            for i in range(...)
                ...
        else
            for i in range(...)
                if i < xxx:
                    ...

    This unswitch function generalizes that pattern with the help of meta
    parameters and can be used to perform both loop unswitching and other
    tile predicate lifting like in simd and amx.

    TODO: Generalize to support multiple predicates.
    TODO: Once nested lambdas compose well should make unswitch compose with
    tile in an easy way.

    Args:
        dynamic_switch: The dynamic condition that enables the unswitched code
          path.
        switched_func: The function containing the inner loop logic that can be
          unswitched.
    """

    if dynamic_switch:
        switched_func[True]()
    else:
        switched_func[False]()


@always_inline
def unswitch(
    dynamic_switch_a: Bool,
    dynamic_switch_b: Bool,
    switched_func: Some[SwitchedFunction2],
):
    """Performs a functional 2-predicates unswitch transformation.

    Args:
        dynamic_switch_a: The first dynamic condition that enables the outer
          unswitched code path.
        dynamic_switch_b: The second dynamic condition that enables the inner
          unswitched code path.
        switched_func: The function containing the inner loop logic that has 2
          predicates which can be unswitched.
    """
    # TODO: This could be a lot easier to write once parameter names can be
    #  removed.
    if dynamic_switch_a:

        @always_inline
        def switched_a_true[static_switch: Bool]() {imm}:
            switched_func[True, static_switch]()

        unswitch(dynamic_switch_b, switched_a_true)
    else:

        @always_inline
        def switched_a_false[static_switch: Bool]() {imm}:
            switched_func[False, static_switch]()

        unswitch(dynamic_switch_b, switched_a_false)


# ===-----------------------------------------------------------------------===#
# TileWithUnswitch
# ===-----------------------------------------------------------------------===#

comptime Static1DTileUnswitchUnitFunc = def[width: Int, sw: Bool](
    Int, Int
) -> None
"""Signature of a tiled function with static tile size and unswitch flag.

The function takes a static tile size parameter and offset arguments,
i.e. `func[tile_size: Int](offset: Int)`.
"""

comptime Static1DTileUnitFuncWithFlag = def[width: Int, flag: Bool](Int) -> None
"""Signature of a tiled function with a static tile size, offset, and flag."""


@always_inline("nodebug")
def tile_and_unswitch[
    tile_size_list: List[Int],
](
    offset: Int,
    upperbound: Int,
    workgroup_function: Some[Static1DTileUnswitchUnitFunc],
):
    """Performs time and unswitch functional transformation.

    A variant of static tile given a workgroup function that can be unswitched.
    This generator is a fused version of tile and unswitch, where the static
    unswitch is true throughout the "inner" portion of the workload and is
    false only on the residue tile.

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
    var current_offset = offset
    var remaining = upperbound - offset

    comptime for tile_size in tile_size_list:
        # Process work with the tile size until there's not enough remaining work
        #  to fit in a tile.
        while remaining >= tile_size:
            workgroup_function[tile_size, True](current_offset, upperbound)
            current_offset += tile_size
            remaining -= tile_size

    # Use the last tile size to process the residue.
    if remaining > 0:
        workgroup_function[tile_size_list[len(tile_size_list) - 1], False](
            current_offset, upperbound
        )


comptime Dynamic1DTileUnswitchUnitFunc = def[sw: Bool](Int, Int, Int) -> None
"""Signature of a dynamic tiled unswitch unit function."""


@always_inline
def tile_and_unswitch(
    offset: Int,
    upperbound: Int,
    *tile_size_list: Int,
    workgroup_function: Some[Dynamic1DTileUnswitchUnitFunc],
):
    """Performs time and unswitch functional transformation.

    A variant of dynamic tile given a workgroup function that can be
    unswitched. This generator is a fused version of tile and unswitch, where
    the static unswitch is true throughout the "inner" portion of the workload
    and is false only on the residue tile.

    Args:
        offset: The initial index to start the work from.
        upperbound: The runtime upperbound that the work function should not exceed.
        tile_size_list: List of tile sizes to launch work.
        workgroup_function: Workgroup function that processes one tile of
          workload.
    """

    # Initialize where to start on the overall work load.
    var current_offset: Int = offset

    for tile_size in tile_size_list:
        # Process work with the tile size until there's not enough remaining work
        #  to fit in a tile.
        while current_offset <= upperbound - tile_size:
            workgroup_function[True](current_offset, upperbound, tile_size)
            current_offset += tile_size

    # Use the last tile size to process the residue.
    if current_offset < upperbound:
        workgroup_function[False](
            current_offset,
            upperbound,
            tile_size_list[len(tile_size_list) - 1],
        )


@always_inline
def tile_middle_unswitch_boundaries[
    middle_tile_sizes: List[Int],
    left_tile_size: Int = 1,  # No tiling by default.
    right_tile_size: Int = 1,  # No tiling by default.
](
    left_boundary_start: Int,
    left_boundary_end: Int,
    right_boundary_start: Int,
    right_boundary_end: Int,
    work_fn: Some[Static1DTileUnitFuncWithFlag],
):
    """Divides 1d iteration space into three parts and tiles them with different
    steps.

    The 1d iteration space is divided into:
        1. [left_boundary_start, left_boundary_end), effected by left boundary.
        2. [left_boundary_end, right_boundary_start), not effected by any boundary.
        3. [right_boundary_start, right_boundary_end), effected by right boundary.

    work_fn's switch is true for the left and right boundaries, implying boundary
    conditions like padding in convolution. The middle part is tiled with static
    tile sizes with the switch as false.

    Parameters:
        middle_tile_sizes: List of tile sizes for the middle part.
        left_tile_size: Tile size for the left boundary region.
        right_tile_size: Tile size for the right boundary region.

    Args:
        left_boundary_start: Start index of the left boundary.
        left_boundary_end: End index of the left boundary.
        right_boundary_start: Start index of the right boundary.
        right_boundary_end: End index of the right boundary.
        work_fn: Work function that processes one tile of workload.

    `middle_tile_sizes` should be in descending order for optimal performance.
    (Larger tile size appeared later in the list fails the while-loop.)
    """

    var offset = left_boundary_start

    # Handle the edge case where filter window is so large that every input
    # point is effected by padding.
    var min_boundary_end = min(left_boundary_end, right_boundary_end)

    # Left boundary region.
    while offset < min_boundary_end:
        work_fn[left_tile_size, True](offset)
        offset += left_tile_size

    # Middle
    comptime for tile_size in middle_tile_sizes:
        while offset <= right_boundary_start - tile_size:
            work_fn[tile_size, False](offset)
            offset += tile_size

    # Right boundary region.
    while offset < right_boundary_end:
        work_fn[right_tile_size, True](offset)
        offset += right_tile_size


comptime Static1DTileUnitFuncWithFlags = def[
    width: Int, left_flag: Bool, right_flag: Bool
](Int) -> None
"""Signature of a tiled function with left and right boundary flags."""


@always_inline
def tile_middle_unswitch_boundaries[
    tile_size: Int,
    size: Int,
](work_fn: Some[Static1DTileUnitFuncWithFlags],):
    """Tile 1d iteration space with boundary conditions at both ends.

    This generator is primarily for convolution with static shapes. `work_fn`'s
    flags hints the function to handle padding at the boundary. The size is the
    static output row size, i.e., WO dimension.

    Parameters:
        tile_size: 1D Tile size.
        size: Iteration range is [0, size).

    Args:
        work_fn: Work function that updates one tile. It has two flags for
            left and right boundaries, respectively.
    """

    # Tile size covers the entire range, e.g., using 14x2 register tile for
    # 14x14 image. Both sides of the tile has boundary conditions.
    comptime if size <= tile_size:
        work_fn[size, True, True](0)
    else:
        # Set bounds of tile sizes on boundaries. E.g. for 7x7 image and
        # tile_size = 6, it's better to use tile_sizes 4 and 3 than using
        # 6 and 1 since the it's tricky to handle padding with very small
        # tile size.
        comptime tile_size_lbound = min(tile_size, size // 2)
        comptime tile_size_rbound = min(tile_size, size - size // 2)

        var offset = 0

        # left boundary
        work_fn[tile_size_lbound, True, False](offset)

        # middle
        @always_inline
        def update_middle[_tile_size: Int](_offset: Int) {imm}:
            work_fn[_tile_size, False, False](_offset)

        comptime num_middle_points = size - tile_size_lbound - tile_size_rbound
        comptime remainder = num_middle_points % tile_size
        # `tile` can't handle zero tile size.
        comptime tile_size_remainder = remainder if remainder > 0 else 1

        tile[[tile_size, tile_size_remainder]](
            tile_size_lbound, size - tile_size_rbound, update_middle
        )

        # right boundary
        work_fn[tile_size_rbound, False, True](size - tile_size_rbound)
