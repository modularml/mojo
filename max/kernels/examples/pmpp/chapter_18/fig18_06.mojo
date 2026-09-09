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

"""Figure 18.6: Basic vertex-centric BFS with CSR graph (push-based).

This is the simplest BFS implementation where each thread checks one vertex.
If the vertex was visited in the previous level, it pushes updates to all
its unvisited neighbors.
"""

from max.gpu import block_idx, thread_idx, block_dim, grid_dim
from max.gpu.host import DeviceContext
from std.atomic import Atomic
from std.collections import List

from graph_utils import (
    UNVISITED,
    generate_random_graph,
    cpu_bfs,
    verify_levels,
)

comptime BLOCK_SIZE = 256


def bfs_kernel(
    src_ptrs: UnsafePointer[UInt32, MutAnyOrigin],
    dst: UnsafePointer[UInt32, MutAnyOrigin],
    num_vertices_dev: Int32,
    level: UnsafePointer[UInt32, MutAnyOrigin],
    new_vertex_visited: UnsafePointer[UInt32, MutAnyOrigin],
    curr_level: UInt32,
):
    """BFS kernel: vertex-centric push-based traversal.

    Each thread processes one vertex. If the vertex was visited at the previous
    level, it iterates through all neighbors and marks unvisited ones with the
    current level.

    Args:
        src_ptrs: CSR row pointers.
        dst: CSR column indices (neighbor vertices).
        num_vertices_dev: Total number of vertices.
        level: Level array (distance from source).
        new_vertex_visited: Flag indicating if any new vertex was visited.
        curr_level: Current BFS level being processed.
    """
    # `Int` is not device-passable; widen the fixed-width arg.
    var num_vertices = Int(num_vertices_dev)
    var vertex = block_idx.x * block_dim.x + thread_idx.x

    if vertex < num_vertices:
        if level[vertex] == curr_level - 1:
            # Iterate through all edges of this vertex
            var edge_start = Int(src_ptrs[vertex])
            var edge_end = Int(src_ptrs[vertex + 1])

            for edge in range(edge_start, edge_end):
                var neighbor = Int(dst[edge])
                if level[neighbor] == UNVISITED:
                    level[neighbor] = curr_level
                    _ = Atomic.fetch_add(new_vertex_visited, UInt32(1))


def main() raises:
    print("Figure 18.6: Basic Vertex-Centric BFS (Push-Based)")
    var ctx = DeviceContext()

    comptime NUM_VERTICES = 1000

    # Generate random graph
    print("Generating random graph with", NUM_VERTICES, "vertices...")
    var src_ptrs = List[UInt32]()
    var dst = List[UInt32]()
    generate_random_graph(NUM_VERTICES, 4, src_ptrs, dst)
    var num_edges = len(dst)
    print("Graph has", num_edges, "edges")

    # Allocate host memory
    var h_level = alloc[UInt32](NUM_VERTICES)
    for i in range(NUM_VERTICES):
        h_level[i] = UNVISITED

    var start_vertex = 0
    h_level[start_vertex] = 0

    # Allocate device memory
    var d_src_ptrs = ctx.enqueue_create_buffer[.uint32](NUM_VERTICES + 1)
    var d_dst = ctx.enqueue_create_buffer[.uint32](num_edges)
    var d_level = ctx.enqueue_create_buffer[.uint32](NUM_VERTICES)
    var d_new_vertex_visited = ctx.enqueue_create_buffer[.uint32](1)

    # Copy graph to device
    var h_src_ptrs = alloc[UInt32](NUM_VERTICES + 1)
    var h_dst = alloc[UInt32](num_edges)
    for i in range(NUM_VERTICES + 1):
        h_src_ptrs[i] = src_ptrs[i]
    for i in range(num_edges):
        h_dst[i] = dst[i]

    ctx.enqueue_copy(d_src_ptrs, h_src_ptrs)
    ctx.enqueue_copy(d_dst, h_dst)
    ctx.enqueue_copy(d_level, h_level)

    # BFS iteration
    var curr_level: UInt32 = 1
    var grid_size = (NUM_VERTICES + BLOCK_SIZE - 1) // BLOCK_SIZE

    print("Running BFS...")
    while True:
        # Reset new_vertex_visited flag
        var h_flag = alloc[UInt32](1)
        h_flag[0] = 0
        ctx.enqueue_copy(d_new_vertex_visited, h_flag)

        ctx.enqueue_function[bfs_kernel](
            d_src_ptrs,
            d_dst,
            Int32(NUM_VERTICES),
            d_level,
            d_new_vertex_visited,
            curr_level,
            grid_dim=(grid_size, 1, 1),
            block_dim=(BLOCK_SIZE, 1, 1),
        )
        ctx.synchronize()

        # Check if any new vertex was visited
        ctx.enqueue_copy(h_flag, d_new_vertex_visited)
        ctx.synchronize()

        if h_flag[0] == 0:
            h_flag.free()
            break

        h_flag.free()
        curr_level += 1

    print("BFS completed at level", curr_level - 1)

    # Copy results back
    ctx.enqueue_copy(h_level, d_level)
    ctx.synchronize()

    # Verify with CPU BFS
    print("Verifying results...")
    var expected_level = List[UInt32]()
    cpu_bfs(NUM_VERTICES, src_ptrs, dst, start_vertex, expected_level)

    if verify_levels(NUM_VERTICES, expected_level, h_level):
        print("Figure 18.6 BFS Passed!")
    else:
        print("Figure 18.6 BFS Failed!")

    # Cleanup
    h_level.free()
    h_src_ptrs.free()
    h_dst.free()
