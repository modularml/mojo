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

# Figure 16.4: Floyd-Warshall bottom-up CUDA kernel translated to Mojo
# GPU-accelerated all-pairs shortest path algorithm

from std.math import ceildiv
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.itertools import product
from std.memory import unsafe_stack_allocation

comptime INF_DIST = Float32.MAX

# ========================== KERNEL CODE ==========================


def floyd_warshall_kernel(
    A: UnsafePointer[Float32, MutAnyOrigin],
    num_vertices_dev: Int32,
    curr_k_dev: Int32,
):
    """GPU kernel for Floyd-Warshall algorithm (Fig 16.4).

    Uses blockIdx.y for row i, blockIdx.x * blockDim.x + threadIdx.x for column j.

    Args:
        A: Distance matrix (num_vertices x num_vertices, flattened).
        num_vertices_dev: Number of vertices in the graph.
        curr_k_dev: Current intermediate vertex being considered.
    """
    # `Int` is not device-passable; widen the fixed-width args.
    var num_vertices = Int(num_vertices_dev)
    var curr_k = Int(curr_k_dev)
    # Allocate shared memory for row value - all threads in block share same i
    var k_row_shared = unsafe_stack_allocation[
        1,
        Float32,
        address_space=.SHARED,
    ]()

    # blockIdx.y gives row i, blockIdx.x and threadIdx.x give column j
    var i = block_idx.y
    var j = block_idx.x * block_dim.x + thread_idx.x

    if i >= num_vertices or j >= num_vertices:
        return

    # Thread 0 loads A[i][k] into shared memory once for the whole block
    if thread_idx.x == 0:
        k_row_shared[0] = Float32(A[i * num_vertices + curr_k])

    barrier()

    # Get k_row from shared memory
    var k_row = Float32(k_row_shared[0])

    # Early exit if no path through curr_k from row i
    if k_row == INF_DIST:
        return

    # Each thread loads its own A[k][j]
    var k_col = A[curr_k * num_vertices + j]

    # Early exit if no path through curr_k to column j
    if k_col == INF_DIST:
        return

    var curr_cost = A[i * num_vertices + j]
    var intermediate_cost = k_row + k_col

    if intermediate_cost < curr_cost:
        A[i * num_vertices + j] = intermediate_cost


# ========================== CPU REFERENCE ==========================


def cpu_floyd_warshall(dist: UnsafePointer[mut=True, Float32, _], V: Int):
    """CPU version of Floyd-Warshall algorithm for verification.

    Args:
        dist: Distance matrix (V x V, flattened).
        V: Number of vertices.
    """
    for k in range(V):
        for i, j in product(range(V), range(V)):
            var idx_ij = i * V + j
            var idx_ik = i * V + k
            var idx_kj = k * V + j

            var current_dist = dist[idx_ij]
            var new_dist = dist[idx_ik] + dist[idx_kj]

            if current_dist > new_dist:
                dist[idx_ij] = new_dist


def initialize_dist(dist: UnsafePointer[mut=True, Float32, _], V: Int):
    """Initialize distance matrix with diagonal = 0, others = INF.

    Args:
        dist: Distance matrix to initialize.
        V: Number of vertices.
    """
    for i, j in product(range(V), range(V)):
        if i == j:
            dist[i * V + j] = 0.0
        else:
            dist[i * V + j] = INF_DIST


def print_dist(dist: UnsafePointer[Float32, _], V: Int, title: String):
    """Print distance matrix (partial for large matrices).

    Args:
        dist: Distance matrix.
        V: Number of vertices.
        title: Title for the output.
    """
    print(title + ":")
    var max_print = min(V, 10)
    for i in range(max_print):
        var row_str = String("")
        for j in range(max_print):
            var d = dist[i * V + j]
            if d == INF_DIST:
                row_str += "  INF "
            else:
                row_str += String(d) + " "
        print(row_str)
    if V > 10:
        print(
            "... (showing first 10x10 of "
            + String(V)
            + "x"
            + String(V)
            + " matrix)"
        )
    print("")


# Simple LCG random number generator
struct SimpleRandom:
    var state: UInt32

    def __init__(out self, seed: UInt32):
        self.state = seed

    def next(mut self) -> UInt32:
        # LCG parameters from Numerical Recipes
        self.state = self.state * 1664525 + 1013904223
        return self.state


# ========================== MAIN ==========================


def main() raises:
    var num_vertices = 100  # Default size

    print("Floyd-Warshall Algorithm (Fig 16.4) - GPU Implementation")
    print("Matrix size:", num_vertices, "x", num_vertices)

    var matrix_size = num_vertices * num_vertices

    # Allocate host memory
    var h_dist = alloc[Float32](matrix_size)
    var h_ref = alloc[Float32](matrix_size)

    # Initialize distance matrix
    print("Initializing", num_vertices, "x", num_vertices, "distance matrix")
    initialize_dist(h_dist, num_vertices)

    # Add some random edges for testing
    var rng = SimpleRandom(42)  # Fixed seed for reproducibility
    var num_edges = num_vertices * 2  # 2 edges per vertex on average
    print("Adding", num_edges, "random edges")

    for _ in range(num_edges):
        var i = Int(rng.next() % UInt32(num_vertices))
        var j = Int(rng.next() % UInt32(num_vertices))
        if i != j:
            var weight = Float32((rng.next() % 20) + 1)  # Random weight 1-20
            h_dist[i * num_vertices + j] = weight

    # Copy for CPU reference
    for i in range(matrix_size):
        h_ref[i] = h_dist[i]

    # Print initial state (first 10x10)
    if num_vertices <= 20:
        print_dist(h_dist, num_vertices, "Initial distance matrix")

    # GPU computation
    with DeviceContext() as ctx:
        comptime dtype = DType.float32

        # Allocate device memory
        var d_dist = ctx.enqueue_create_buffer[dtype](matrix_size)

        # Copy to device
        ctx.enqueue_copy(d_dist, h_dist)

        # Launch configuration: blockIdx.y = row, multiple blocks per row for columns
        var threads_per_block = 256  # Fixed thread count per block
        var blocks_per_row = ceildiv(num_vertices, threads_per_block)

        print("\nLaunching Floyd-Warshall kernel (Fig 16.4)")
        print(
            "Config: (",
            blocks_per_row,
            ",",
            num_vertices,
            ") grid x",
            threads_per_block,
            "threads",
        )
        print(
            "Strategy: blockIdx.y = row, blockIdx.x * blockDim.x + threadIdx.x"
            " = column"
        )
        print("Number of iterations:", num_vertices)

        # Run Floyd-Warshall algorithm
        for k in range(num_vertices):
            ctx.enqueue_function[floyd_warshall_kernel](
                d_dist,
                Int32(num_vertices),
                Int32(k),
                grid_dim=(blocks_per_row, num_vertices, 1),
                block_dim=(threads_per_block, 1, 1),
            )

            if k % 20 == 0 and k > 0:
                print("Completed iteration", k, "/", num_vertices, "...")

        # Copy result back
        ctx.enqueue_copy(h_dist, d_dist)
        ctx.synchronize()

    print("GPU Floyd-Warshall completed")

    # Print GPU result (first 10x10)
    if num_vertices <= 20:
        print_dist(h_dist, num_vertices, "GPU Floyd-Warshall result")

    # Run CPU reference for comparison
    print("Running CPU Floyd-Warshall reference...")
    cpu_floyd_warshall(h_ref, num_vertices)
    print("CPU Floyd-Warshall completed")

    # Print CPU result (first 10x10)
    if num_vertices <= 20:
        print_dist(h_ref, num_vertices, "CPU Floyd-Warshall result")

    # Verify GPU result matches CPU result
    var errors = 0
    for i in range(matrix_size):
        if errors >= 10:
            break
        # Use small epsilon for floating point comparison
        var diff = abs(h_dist[i] - h_ref[i])

        if diff > 0.01 and not (h_dist[i] == INF_DIST and h_ref[i] == INF_DIST):
            var row, col = divmod(i, num_vertices)
            print(
                "Mismatch at [",
                row,
                ",",
                col,
                "]: GPU=",
                h_dist[i],
                ", CPU=",
                h_ref[i],
            )
            errors += 1

    if errors == 0:
        print("\n✓ SUCCESS: GPU and CPU results match!")
    else:
        print("\n✗ ERROR: Found", errors, "mismatches between GPU and CPU")

    # Cleanup
    h_dist.free()
    h_ref.free()
