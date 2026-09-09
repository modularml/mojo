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

"""Graph utilities for BFS algorithms.

This module provides graph data structures and utility functions for BFS
implementations, mirroring the CUDA graph_utils.h functionality.
"""

from std.random import random_ui64
from std.collections import List

comptime UNVISITED: UInt32 = 0xFFFFFFFF


struct CSRGraph(Copyable):
    """Compressed Sparse Row graph representation."""

    var num_vertices: Int
    var num_edges: Int
    var src_ptrs: UnsafePointer[UInt32, MutUntrackedOrigin]
    var dst: UnsafePointer[UInt32, MutUntrackedOrigin]

    def __init__(out self):
        self.num_vertices = 0
        self.num_edges = 0
        self.src_ptrs = UnsafePointer[
            UInt32, MutUntrackedOrigin
        ].unsafe_dangling()
        self.dst = UnsafePointer[UInt32, MutUntrackedOrigin].unsafe_dangling()

    def __init__(
        out self,
        nv: Int,
        ne: Int,
        sp: UnsafePointer[UInt32, MutUntrackedOrigin],
        d: UnsafePointer[UInt32, MutUntrackedOrigin],
    ):
        self.num_vertices = nv
        self.num_edges = ne
        self.src_ptrs = sp
        self.dst = d


struct CSCGraph(Copyable):
    """Compressed Sparse Column graph representation."""

    var num_vertices: Int
    var num_edges: Int
    var dst_ptrs: UnsafePointer[UInt32, MutUntrackedOrigin]
    var src: UnsafePointer[UInt32, MutUntrackedOrigin]

    def __init__(out self):
        self.num_vertices = 0
        self.num_edges = 0
        self.dst_ptrs = UnsafePointer[
            UInt32, MutUntrackedOrigin
        ].unsafe_dangling()
        self.src = UnsafePointer[UInt32, MutUntrackedOrigin].unsafe_dangling()

    def __init__(
        out self,
        nv: Int,
        ne: Int,
        dp: UnsafePointer[UInt32, MutUntrackedOrigin],
        s: UnsafePointer[UInt32, MutUntrackedOrigin],
    ):
        self.num_vertices = nv
        self.num_edges = ne
        self.dst_ptrs = dp
        self.src = s


struct COOGraph(Copyable):
    """Coordinate (COO) graph representation."""

    var num_vertices: Int
    var num_edges: Int
    var src: UnsafePointer[UInt32, MutUntrackedOrigin]
    var dst: UnsafePointer[UInt32, MutUntrackedOrigin]

    def __init__(out self):
        self.num_vertices = 0
        self.num_edges = 0
        self.src = UnsafePointer[UInt32, MutUntrackedOrigin].unsafe_dangling()
        self.dst = UnsafePointer[UInt32, MutUntrackedOrigin].unsafe_dangling()

    def __init__(
        out self,
        nv: Int,
        ne: Int,
        s: UnsafePointer[UInt32, MutUntrackedOrigin],
        d: UnsafePointer[UInt32, MutUntrackedOrigin],
    ):
        self.num_vertices = nv
        self.num_edges = ne
        self.src = s
        self.dst = d


def generate_random_graph(
    num_vertices: Int,
    num_extra_edges_per_vertex: Int,
    mut src_ptrs: List[UInt32],
    mut dst: List[UInt32],
):
    """Generate a random graph in CSR format.

    Creates a connected graph by adding sequential edges (i -> i+1) and
    additional random edges per vertex.

    Args:
        num_vertices: Number of vertices in the graph.
        num_extra_edges_per_vertex: Number of random edges to add per vertex.
        src_ptrs: Output CSR row pointers.
        dst: Output CSR column indices (destination vertices).
    """
    var adj = List[List[UInt32]](capacity=num_vertices)
    for _ in range(num_vertices):
        adj.append(List[UInt32]())

    for i in range(num_vertices):
        # Ensure connectivity: add edge to next vertex
        if i < num_vertices - 1:
            adj[i].append(UInt32(i + 1))

        # Add random edges
        for _ in range(num_extra_edges_per_vertex):
            var neighbor = Int(random_ui64(0, UInt64(num_vertices - 1)))
            if neighbor != i:
                adj[i].append(UInt32(neighbor))

    # Flatten to CSR format - clear and rebuild
    src_ptrs.clear()
    dst.clear()

    src_ptrs.append(0)
    for i in range(num_vertices):
        for j in range(len(adj[i])):
            dst.append(adj[i][j])
        src_ptrs.append(UInt32(len(dst)))


def csr_to_csc(
    num_vertices: Int,
    src_ptrs: List[UInt32],
    dst: List[UInt32],
    mut dst_ptrs: List[UInt32],
    mut src: List[UInt32],
):
    """Convert CSR graph to CSC format (transpose).

    Args:
        num_vertices: Number of vertices.
        src_ptrs: CSR row pointers.
        dst: CSR column indices.
        dst_ptrs: Output CSC column pointers.
        src: Output CSC row indices (source vertices).
    """
    var in_edges = List[List[UInt32]](capacity=num_vertices)
    for _ in range(num_vertices):
        in_edges.append(List[UInt32]())

    for u in range(num_vertices):
        var start = Int(src_ptrs[u])
        var end = Int(src_ptrs[u + 1])
        for i in range(start, end):
            var v = Int(dst[i])
            in_edges[v].append(UInt32(u))

    dst_ptrs.clear()
    src.clear()

    dst_ptrs.append(0)
    for v in range(num_vertices):
        for j in range(len(in_edges[v])):
            src.append(in_edges[v][j])
        dst_ptrs.append(UInt32(len(src)))


def csr_to_coo(
    num_vertices: Int,
    src_ptrs: List[UInt32],
    dst: List[UInt32],
    mut coo_src: List[UInt32],
    mut coo_dst: List[UInt32],
):
    """Convert CSR graph to COO format.

    Args:
        num_vertices: Number of vertices.
        src_ptrs: CSR row pointers.
        dst: CSR column indices.
        coo_src: Output COO source vertices.
        coo_dst: Output COO destination vertices.
    """
    coo_src.clear()
    coo_dst.clear()

    for i in range(num_vertices):
        var start = Int(src_ptrs[i])
        var end = Int(src_ptrs[i + 1])
        for j in range(start, end):
            coo_src.append(UInt32(i))
            coo_dst.append(dst[j])


def cpu_bfs(
    num_vertices: Int,
    src_ptrs: List[UInt32],
    dst: List[UInt32],
    start_vertex: Int,
    mut expected_level: List[UInt32],
):
    """Simple CPU BFS for verification.

    Args:
        num_vertices: Number of vertices.
        src_ptrs: CSR row pointers.
        dst: CSR column indices.
        start_vertex: Starting vertex for BFS.
        expected_level: Output level of each vertex from start (UNVISITED if unreachable).
    """
    expected_level.clear()
    for _ in range(num_vertices):
        expected_level.append(UNVISITED)

    # Simple queue-based BFS using a list
    var queue = List[Int]()
    queue.append(start_vertex)
    expected_level[start_vertex] = 0

    var front = 0
    while front < len(queue):
        var u = queue[front]
        front += 1

        var start = Int(src_ptrs[u])
        var end = Int(src_ptrs[u + 1])

        for i in range(start, end):
            var v = Int(dst[i])
            if expected_level[v] == UNVISITED:
                expected_level[v] = expected_level[u] + 1
                queue.append(v)


def verify_levels(
    num_vertices: Int,
    expected: List[UInt32],
    h_level: UnsafePointer[mut=False, UInt32, _],
) -> Bool:
    """Verify GPU BFS results against CPU reference.

    Args:
        num_vertices: Number of vertices.
        expected: Expected levels from CPU BFS.
        h_level: GPU computed levels (already copied to host).

    Returns:
        True if results match, False otherwise.
    """
    for i in range(num_vertices):
        if expected[i] != h_level[i]:
            print(
                "Mismatch at vertex",
                i,
                ": CPU",
                expected[i],
                ", GPU",
                h_level[i],
            )
            return False
    return True
