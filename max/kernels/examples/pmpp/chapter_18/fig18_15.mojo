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

"""Figure 18.15: Frontier-based BFS with privatization (shared memory).

Uses shared memory to create a private frontier per block,
reducing contention on global memory atomics.
"""

from max.gpu import block_idx, thread_idx, block_dim, grid_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation
from std.atomic import Atomic
from std.collections import List

from graph_utils import (
    UNVISITED,
    generate_random_graph,
    cpu_bfs,
    verify_levels,
)

comptime BLOCK_SIZE = 256
comptime PRIVATE_FRONTIER_CAPACITY = 256


def bfs_kernel(
    src_ptrs: UnsafePointer[UInt32, MutAnyOrigin],
    dst: UnsafePointer[UInt32, MutAnyOrigin],
    level: UnsafePointer[UInt32, MutAnyOrigin],
    prev_frontier: UnsafePointer[UInt32, MutAnyOrigin],
    curr_frontier: UnsafePointer[UInt32, MutAnyOrigin],
    num_prev_frontier_dev: Int32,
    num_curr_frontier: UnsafePointer[UInt32, MutAnyOrigin],
    curr_level: UInt32,
):
    """BFS kernel with private frontier in shared memory."""
    # `Int` is not device-passable; widen the fixed-width arg.
    var num_prev_frontier = Int(num_prev_frontier_dev)
    var curr_frontier_s = unsafe_stack_allocation[
        PRIVATE_FRONTIER_CAPACITY,
        UInt32,
        address_space=.SHARED,
    ]()
    var num_curr_frontier_s = unsafe_stack_allocation[
        1,
        UInt32,
        address_space=.SHARED,
    ]()

    if thread_idx.x == 0:
        num_curr_frontier_s[0] = 0

    barrier()

    var i = block_idx.x * block_dim.x + thread_idx.x

    if i < num_prev_frontier:
        var vertex = Int(prev_frontier[i])

        var edge_start = Int(src_ptrs[vertex])
        var edge_end = Int(src_ptrs[vertex + 1])

        for edge in range(edge_start, edge_end):
            var neighbor = Int(dst[edge])
            if level[neighbor] == UNVISITED:
                level[neighbor] = curr_level
                var idx_s = Int(
                    Atomic.fetch_add(num_curr_frontier_s, UInt32(1))
                )

                if idx_s < PRIVATE_FRONTIER_CAPACITY:
                    curr_frontier_s[idx_s] = UInt32(neighbor)
                else:
                    num_curr_frontier_s[0] = UInt32(PRIVATE_FRONTIER_CAPACITY)
                    var idx = Int(
                        Atomic.fetch_add(num_curr_frontier, UInt32(1))
                    )
                    curr_frontier[idx] = UInt32(neighbor)

    barrier()

    if thread_idx.x == 0:
        if num_curr_frontier_s[0] > PRIVATE_FRONTIER_CAPACITY:
            num_curr_frontier_s[0] = UInt32(PRIVATE_FRONTIER_CAPACITY)

    barrier()

    var start_idx_ptr = unsafe_stack_allocation[
        1,
        UInt32,
        address_space=.SHARED,
    ]()
    if thread_idx.x == 0:
        var local_count = Int(num_curr_frontier_s[0])
        start_idx_ptr[0] = Atomic.fetch_add(
            num_curr_frontier, UInt32(local_count)
        )

    barrier()

    var curr_frontier_start_idx = Int(start_idx_ptr[0])
    var local_frontier_size = Int(num_curr_frontier_s[0])

    var tid = thread_idx.x
    while tid < local_frontier_size:
        curr_frontier[curr_frontier_start_idx + tid] = curr_frontier_s[tid]
        tid += block_dim.x


def main() raises:
    print("Figure 18.15: Frontier-Based BFS with Privatization")
    var ctx = DeviceContext()

    comptime NUM_VERTICES = 1000

    print("Generating random graph with", NUM_VERTICES, "vertices...")
    var src_ptrs = List[UInt32]()
    var dst = List[UInt32]()
    generate_random_graph(NUM_VERTICES, 4, src_ptrs, dst)
    var num_edges = len(dst)
    print("Graph has", num_edges, "edges")

    var h_level = alloc[UInt32](NUM_VERTICES)
    for i in range(NUM_VERTICES):
        h_level[i] = UNVISITED

    var start_vertex = 0
    h_level[start_vertex] = 0

    var d_src_ptrs = ctx.enqueue_create_buffer[.uint32](NUM_VERTICES + 1)
    var d_dst = ctx.enqueue_create_buffer[.uint32](num_edges)
    var d_level = ctx.enqueue_create_buffer[.uint32](NUM_VERTICES)
    var d_prev_frontier = ctx.enqueue_create_buffer[.uint32](NUM_VERTICES * 2)
    var d_curr_frontier = ctx.enqueue_create_buffer[.uint32](NUM_VERTICES * 2)
    var d_num_curr_frontier = ctx.enqueue_create_buffer[.uint32](1)

    var h_src_ptrs = alloc[UInt32](NUM_VERTICES + 1)
    var h_dst = alloc[UInt32](num_edges)
    for i in range(NUM_VERTICES + 1):
        h_src_ptrs[i] = src_ptrs[i]
    for i in range(num_edges):
        h_dst[i] = dst[i]

    ctx.enqueue_copy(d_src_ptrs, h_src_ptrs)
    ctx.enqueue_copy(d_dst, h_dst)
    ctx.enqueue_copy(d_level, h_level)

    var h_frontier = alloc[UInt32](1)
    h_frontier[0] = UInt32(start_vertex)
    ctx.enqueue_copy(d_prev_frontier, h_frontier)
    h_frontier.free()

    var num_prev_frontier = 1
    var curr_level: UInt32 = 1

    print("Running BFS...")
    while num_prev_frontier > 0:
        var h_zero = alloc[UInt32](1)
        h_zero[0] = 0
        ctx.enqueue_copy(d_num_curr_frontier, h_zero)
        h_zero.free()

        var grid_size = (num_prev_frontier + BLOCK_SIZE - 1) // BLOCK_SIZE
        if grid_size == 0:
            grid_size = 1

        ctx.enqueue_function[bfs_kernel](
            d_src_ptrs,
            d_dst,
            d_level,
            d_prev_frontier,
            d_curr_frontier,
            Int32(num_prev_frontier),
            d_num_curr_frontier,
            curr_level,
            grid_dim=(grid_size, 1, 1),
            block_dim=(BLOCK_SIZE, 1, 1),
        )
        ctx.synchronize()

        var h_num = alloc[UInt32](1)
        ctx.enqueue_copy(h_num, d_num_curr_frontier)
        ctx.synchronize()
        num_prev_frontier = Int(h_num[0])
        h_num.free()

        var temp = d_prev_frontier
        d_prev_frontier = d_curr_frontier
        d_curr_frontier = temp

        curr_level += 1

    print("BFS completed at level", curr_level - 1)

    ctx.enqueue_copy(h_level, d_level)
    ctx.synchronize()

    print("Verifying results...")
    var expected_level = List[UInt32]()
    cpu_bfs(NUM_VERTICES, src_ptrs, dst, start_vertex, expected_level)

    if verify_levels(NUM_VERTICES, expected_level, h_level):
        print("Figure 18.15 BFS Passed!")
    else:
        print("Figure 18.15 BFS Failed!")

    h_level.free()
    h_src_ptrs.free()
    h_dst.free()
