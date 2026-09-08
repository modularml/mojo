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
"""This module provides GPU thread and block indexing functionality.

It defines aliases and functions for accessing GPU grid, block, and thread
indices and dimensions.
"""

import std._gpu.primitives.id

comptime block_dim = std._gpu.primitives.id.block_dim
"""Contains the dimensions of the block as `x`, `y`, and `z` values.

For example: `block_dim.y`."""

comptime block_id_in_cluster = std._gpu.primitives.id.block_id_in_cluster
"""Contains the block id of the threadblock within a cluster, as `x`, `y`, and `z` values."""

comptime block_idx = std._gpu.primitives.id.block_idx
"""Contains the block index in the grid, as `x`, `y`, and `z` values."""

comptime cluster_dim = std._gpu.primitives.id.cluster_dim
"""Contains the dimensions of the cluster, as `x`, `y`, and `z` values."""

comptime cluster_idx = std._gpu.primitives.id.cluster_idx
"""Contains the cluster index in the grid, as `x`, `y`, and `z` values."""

comptime global_idx = std._gpu.primitives.id.global_idx
"""Contains the global offset of the kernel launch, as `x`, `y`, and `z`
values."""

comptime grid_dim = std._gpu.primitives.id.grid_dim
"""Provides accessors for getting the `x`, `y`, and `z`
dimensions of a grid."""

comptime thread_idx = std._gpu.primitives.id.thread_idx
"""Contains the thread index in the block, as `x`, `y`, and `z` values."""


@always_inline("nodebug")
def lane_id() -> Int:
    """Returns the lane ID of the current thread within its warp.

    The lane ID is a unique identifier for each thread within a warp, ranging from 0 to
    WARP_SIZE-1. This ID is commonly used for warp-level programming and thread
    synchronization within a warp.

    Returns:
        The lane ID (0 to WARP_SIZE-1) of the current thread.
    """
    return std._gpu.primitives.id.lane_id()


@always_inline("nodebug")
def sm_id() -> Int:
    """Returns the Streaming Multiprocessor (SM) ID of the current thread.

    The SM ID uniquely identifies which physical streaming multiprocessor the thread is
    executing on. This is useful for SM-level optimizations and understanding hardware
    utilization.

    If called on non-NVIDIA GPUs, this function aborts as this functionality
    is only supported on NVIDIA hardware.

    Returns:
        The SM ID of the current thread.
    """
    return std._gpu.primitives.id.sm_id()


@always_inline("nodebug")
def warp_id[*, broadcast: Bool = False]() -> Int:
    """Returns the warp ID of the current thread within its block.
    The warp ID is a unique identifier for each warp within a block, ranging
    from 0 to BLOCK_SIZE/WARP_SIZE-1. This ID is commonly used for warp-level
    programming and synchronization within a block.

    Parameters:
        broadcast: If true, broadcasts the warp ID to all threads in the warp,
                   ensuring that all threads in the same warp have the same
                   value. This can be useful for certain warp-level algorithms.

    Returns:
        The warp ID (0 to BLOCK_SIZE/WARP_SIZE-1) of the current thread.
    """
    return std._gpu.primitives.id.warp_id[broadcast=broadcast]()
