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

"""Figure 18.8: Vertex-centric BFS with CSC graph (pull-based).

This is a pull-based BFS where each unvisited vertex checks if any of its
incoming neighbors were visited in the previous level. If so, the vertex
marks itself as visited at the current level.
"""

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from std.atomic import Atomic
from std.collections import List

from graph_utils import (
    UNVISITED,
    generate_random_graph,
    csr_to_csc,
    cpu_bfs,
    verify_levels,
)

comptime BLOCK_SIZE = 256


def bfs_kernel(
    dst_ptrs: UnsafePointer[UInt32, MutAnyOrigin],
    src: UnsafePointer[UInt32, MutAnyOrigin],
    num_vertices_dev: Int32,
    level: UnsafePointer[UInt32, MutAnyOrigin],
    new_vertex_visited: UnsafePointer[UInt32, MutAnyOrigin],
    curr_level: UInt32,
):
    """BFS kernel: vertex-centric pull-based traversal using CSC graph."""
    # `Int` is not device-passable; widen the fixed-width arg.
    var num_vertices = Int(num_vertices_dev)
    var vertex = block_idx.x * block_dim.x + thread_idx.x

    if vertex < num_vertices:
        if level[vertex] == UNVISITED:
            var edge_start = Int(dst_ptrs[vertex])
            var edge_end = Int(dst_ptrs[vertex + 1])

            for edge in range(edge_start, edge_end):
                var neighbor = Int(src[edge])
                if level[neighbor] == curr_level - 1:
                    level[vertex] = curr_level
                    _ = Atomic.fetch_add(new_vertex_visited, UInt32(1))
                    break


def main() raises:
    print("Figure 18.8: Vertex-Centric BFS (Pull-Based with CSC)")
    var ctx = DeviceContext()

    comptime NUM_VERTICES = 1000

    print("Generating random graph with", NUM_VERTICES, "vertices...")
    var src_ptrs = List[UInt32]()
    var dst = List[UInt32]()
    generate_random_graph(NUM_VERTICES, 4, src_ptrs, dst)

    print("Converting to CSC format...")
    var dst_ptrs = List[UInt32]()
    var src = List[UInt32]()
    csr_to_csc(NUM_VERTICES, src_ptrs, dst, dst_ptrs, src)

    var num_edges = len(src)
    print("Graph has", num_edges, "edges")

    var h_level = alloc[UInt32](NUM_VERTICES)
    for i in range(NUM_VERTICES):
        h_level[i] = UNVISITED

    var start_vertex = 0
    h_level[start_vertex] = 0

    var d_dst_ptrs = ctx.enqueue_create_buffer[.uint32](NUM_VERTICES + 1)
    var d_src = ctx.enqueue_create_buffer[.uint32](num_edges)
    var d_level = ctx.enqueue_create_buffer[.uint32](NUM_VERTICES)
    var d_new_vertex_visited = ctx.enqueue_create_buffer[.uint32](1)

    var h_dst_ptrs = alloc[UInt32](NUM_VERTICES + 1)
    var h_src = alloc[UInt32](num_edges)
    for i in range(NUM_VERTICES + 1):
        h_dst_ptrs[i] = dst_ptrs[i]
    for i in range(num_edges):
        h_src[i] = src[i]

    ctx.enqueue_copy(d_dst_ptrs, h_dst_ptrs)
    ctx.enqueue_copy(d_src, h_src)
    ctx.enqueue_copy(d_level, h_level)

    var curr_level: UInt32 = 1
    var grid_size = (NUM_VERTICES + BLOCK_SIZE - 1) // BLOCK_SIZE

    print("Running BFS...")
    while True:
        var h_flag = alloc[UInt32](1)
        h_flag[0] = 0
        ctx.enqueue_copy(d_new_vertex_visited, h_flag)

        ctx.enqueue_function[bfs_kernel](
            d_dst_ptrs,
            d_src,
            Int32(NUM_VERTICES),
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
        print("Figure 18.8 BFS Passed!")
    else:
        print("Figure 18.8 BFS Failed!")

    h_level.free()
    h_dst_ptrs.free()
    h_src.free()
