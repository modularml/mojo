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

"""Figure 18.10: Edge-centric BFS with COO graph.

This is an edge-centric BFS where each thread processes one edge.
It checks if the source vertex of the edge was visited in the previous level
and if so, marks the destination vertex as visited at the current level.
"""

from max.gpu import block_idx, thread_idx, block_dim, grid_dim
from max.gpu.host import DeviceContext
from std.atomic import Atomic
from std.collections import List

from graph_utils import (
    UNVISITED,
    generate_random_graph,
    csr_to_coo,
    cpu_bfs,
    verify_levels,
)

comptime BLOCK_SIZE = 256


def bfs_kernel(
    coo_src: UnsafePointer[UInt32, MutAnyOrigin],
    coo_dst: UnsafePointer[UInt32, MutAnyOrigin],
    num_edges_dev: Int32,
    level: UnsafePointer[UInt32, MutAnyOrigin],
    new_vertex_visited: UnsafePointer[UInt32, MutAnyOrigin],
    curr_level: UInt32,
):
    """BFS kernel: edge-centric traversal using COO graph."""
    # `Int` is not device-passable; widen the fixed-width arg.
    var num_edges = Int(num_edges_dev)
    var edge = block_idx.x * block_dim.x + thread_idx.x

    if edge < num_edges:
        var vertex = Int(coo_src[edge])
        if level[vertex] == curr_level - 1:
            var neighbor = Int(coo_dst[edge])
            if level[neighbor] == UNVISITED:
                level[neighbor] = curr_level
                _ = Atomic.fetch_add(new_vertex_visited, UInt32(1))


def main() raises:
    print("Figure 18.10: Edge-Centric BFS (COO Graph)")
    var ctx = DeviceContext()

    comptime NUM_VERTICES = 1000

    print("Generating random graph with", NUM_VERTICES, "vertices...")
    var src_ptrs = List[UInt32]()
    var dst = List[UInt32]()
    generate_random_graph(NUM_VERTICES, 4, src_ptrs, dst)

    print("Converting to COO format...")
    var coo_src = List[UInt32]()
    var coo_dst = List[UInt32]()
    csr_to_coo(NUM_VERTICES, src_ptrs, dst, coo_src, coo_dst)

    var num_edges = len(coo_src)
    print("Graph has", num_edges, "edges")

    var h_level = alloc[UInt32](NUM_VERTICES)
    for i in range(NUM_VERTICES):
        h_level[i] = UNVISITED

    var start_vertex = 0
    h_level[start_vertex] = 0

    var d_coo_src = ctx.enqueue_create_buffer[.uint32](num_edges)
    var d_coo_dst = ctx.enqueue_create_buffer[.uint32](num_edges)
    var d_level = ctx.enqueue_create_buffer[.uint32](NUM_VERTICES)
    var d_new_vertex_visited = ctx.enqueue_create_buffer[.uint32](1)

    var h_coo_src = alloc[UInt32](num_edges)
    var h_coo_dst = alloc[UInt32](num_edges)
    for i in range(num_edges):
        h_coo_src[i] = coo_src[i]
        h_coo_dst[i] = coo_dst[i]

    ctx.enqueue_copy(d_coo_src, h_coo_src)
    ctx.enqueue_copy(d_coo_dst, h_coo_dst)
    ctx.enqueue_copy(d_level, h_level)

    var curr_level: UInt32 = 1
    var grid_size = (num_edges + BLOCK_SIZE - 1) // BLOCK_SIZE

    print("Running BFS...")
    while True:
        var h_flag = alloc[UInt32](1)
        h_flag[0] = 0
        ctx.enqueue_copy(d_new_vertex_visited, h_flag)

        ctx.enqueue_function[bfs_kernel](
            d_coo_src,
            d_coo_dst,
            Int32(num_edges),
            d_level,
            d_new_vertex_visited,
            curr_level,
            grid_dim=(grid_size, 1, 1),
            block_dim=(BLOCK_SIZE, 1, 1),
        )
        ctx.synchronize()

        ctx.enqueue_copy(h_flag, d_new_vertex_visited)
        ctx.synchronize()

        if h_flag[0] == 0:
            h_flag.free()
            break

        h_flag.free()
        curr_level += 1

    print("BFS completed at level", curr_level - 1)

    ctx.enqueue_copy(h_level, d_level)
    ctx.synchronize()

    print("Verifying results...")
    var expected_level = List[UInt32]()
    cpu_bfs(NUM_VERTICES, src_ptrs, dst, start_vertex, expected_level)

    if verify_levels(NUM_VERTICES, expected_level, h_level):
        print("Figure 18.10 BFS Passed!")
    else:
        print("Figure 18.10 BFS Failed!")

    h_level.free()
    h_coo_src.free()
    h_coo_dst.free()
