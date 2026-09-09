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

# Figure 16.3: Bottom-up implementation of Floyd-Warshall algorithm
# All-pairs shortest path algorithm using dynamic programming

from std.itertools import product

comptime INF = Float32.MAX

# ========================== CPU IMPLEMENTATION ==========================


def floyd_warshall_cpu(dist: UnsafePointer[mut=True, Float32, _], V: Int):
    """CPU version of Floyd-Warshall algorithm (matches Fig 16.3 pseudo-code).

    Args:
        dist: Distance matrix (V x V, flattened).
        V: Number of vertices.
    """
    # Initialize nearest neighbors to actual distance, all others = infinity
    # (done by caller)

    # Order of visiting k values not important, must visit each value
    for k in range(V):
        for i, j in product(range(V), range(V)):
            # If vertex k is on the shortest path from i to j, then
            # update the value of dist[i][j]
            var idx_ij = i * V + j
            var idx_ik = i * V + k
            var idx_kj = k * V + j

            var current_dist = dist[idx_ij]
            var new_dist = dist[idx_ik] + dist[idx_kj]

            if current_dist > new_dist:
                dist[idx_ij] = new_dist


def initialize_dist(dist: UnsafePointer[mut=True, Float32, _], V: Int):
    """Initialize distance matrix to infinity.

    Args:
        dist: Distance matrix to initialize.
        V: Number of vertices.
    """
    for i in range(V * V):
        dist[i] = INF


def verify_results(
    result: UnsafePointer[Float32, _],
    expected: UnsafePointer[Float32, _],
    V: Int,
) -> Bool:
    """Verify Floyd-Warshall results.

    Args:
        result: Computed distance matrix.
        expected: Expected distance matrix.
        V: Number of vertices.

    Returns:
        True if results match, False otherwise.
    """
    var errors = 0
    for i in range(V * V):
        if result[i] != expected[i]:
            if errors < 10:
                print(
                    "Mismatch at index",
                    i,
                    ": got",
                    result[i],
                    "expected",
                    expected[i],
                )
            errors += 1

    if errors > 0:
        print("Total errors:", errors, "out of", V * V, "elements")

    return errors == 0


# ========================== TEST CODE ==========================


def test_floyd_warshall() raises:
    """Test Floyd-Warshall algorithm."""
    print("Testing Floyd-Warshall algorithm (CPU version)...")

    var V = 4  # Number of vertices
    var dist = alloc[Float32](V * V)

    # Initialize nearest neighbors to actual distance, all others = infinity
    initialize_dist(dist, V)

    # Example graph from typical Floyd-Warshall examples:
    # Set distances for edges
    dist[0 * V + 1] = 5.0  # 0 -> 1
    dist[0 * V + 3] = 10.0  # 0 -> 3
    dist[1 * V + 2] = 3.0  # 1 -> 2
    dist[2 * V + 3] = 1.0  # 2 -> 3

    # Distance from vertex to itself is 0
    for i in range(V):
        dist[i * V + i] = 0.0

    # Run Floyd-Warshall
    floyd_warshall_cpu(dist, V)

    # Print results
    print("\nShortest distances between all pairs:")
    for i in range(V):
        var row_str = String("")
        for j in range(V):
            var d = dist[i * V + j]
            if d == INF:
                row_str += "INF "
            else:
                row_str += String(d) + " "
        print(row_str)

    # Verify expected results
    var expected = alloc[Float32](V * V)
    # Expected shortest paths for the example graph
    expected[0 * V + 0] = 0.0  # 0 to 0
    expected[0 * V + 1] = 5.0  # 0 to 1 (direct)
    expected[0 * V + 2] = 8.0  # 0 to 2 (via 1)
    expected[0 * V + 3] = 9.0  # 0 to 3 (via 1->2->3)
    expected[1 * V + 0] = INF  # 1 to 0 (no path)
    expected[1 * V + 1] = 0.0  # 1 to 1
    expected[1 * V + 2] = 3.0  # 1 to 2 (direct)
    expected[1 * V + 3] = 4.0  # 1 to 3 (via 2)
    expected[2 * V + 0] = INF  # 2 to 0 (no path)
    expected[2 * V + 1] = INF  # 2 to 1 (no path)
    expected[2 * V + 2] = 0.0  # 2 to 2
    expected[2 * V + 3] = 1.0  # 2 to 3 (direct)
    expected[3 * V + 0] = INF  # 3 to 0 (no path)
    expected[3 * V + 1] = INF  # 3 to 1 (no path)
    expected[3 * V + 2] = INF  # 3 to 2 (no path)
    expected[3 * V + 3] = 0.0  # 3 to 3

    var passed = verify_results(dist, expected, V)

    if passed:
        print("\n✓ All tests passed!")
    else:
        raise Error("Tests failed!")

    dist.free()
    expected.free()


def main() raises:
    """Main entry point."""
    try:
        test_floyd_warshall()
    except e:
        print("Error:", e)
