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

# Figure 16.1: Top-down and bottom-up Mojo implementations of Fibonacci series


# ========================== FIBONACCI IMPLEMENTATIONS ==========================


def fibonacci_bottomup(n: Int) -> Int:
    """Bottom-up Fibonacci using dynamic programming (matches Fig 16.1).

    Args:
        n: The Fibonacci number to compute.

    Returns:
        The nth Fibonacci number.
    """
    if n != 0 and n != 1:
        # Allocate table
        var table = alloc[Int](n + 1)

        # Initialize table
        table[0] = 0
        table[1] = 1

        # Fill table using bottom-up approach
        for i in range(2, n + 1):
            table[i] = table[i - 1] + table[i - 2]

        var result = table[n]
        table.free()
        return result
    else:
        return n


def fibonacci_topdown_helper(
    n: Int, hash_table: UnsafePointer[mut=True, Int, _]
) -> Int:
    """Helper for top-down Fibonacci with memoization.

    Args:
        n: The Fibonacci number to compute.
        hash_table: Memoization table.

    Returns:
        The nth Fibonacci number.
    """
    # Check if already calculated (hash_table[n] != -1)
    if hash_table[n] != -1:
        return hash_table[n]

    var result: Int
    if n != 0 and n != 1:
        result = fibonacci_topdown_helper(
            n - 1, hash_table
        ) + fibonacci_topdown_helper(n - 2, hash_table)
    else:
        result = n

    # Store in hash_table
    hash_table[n] = result
    return result


def fibonacci_topdown(n: Int) -> Int:
    """Top-down Fibonacci with memoization (matches Fig 16.1).

    Args:
        n: The Fibonacci number to compute.

    Returns:
        The nth Fibonacci number.
    """
    # Allocate hash_table (memoization array), initialize to -1
    var hash_table = alloc[Int](n + 1)
    for i in range(n + 1):
        hash_table[i] = -1

    var result = fibonacci_topdown_helper(n, hash_table)
    hash_table.free()
    return result


# ========================== TEST CODE ==========================


def test_fibonacci() raises:
    """Test both Fibonacci implementations."""
    print("Testing Fibonacci implementations...")

    # Test cases
    var test_values = alloc[Int](9)
    test_values[0] = 0
    test_values[1] = 1
    test_values[2] = 2
    test_values[3] = 3
    test_values[4] = 4
    test_values[5] = 5
    test_values[6] = 10
    test_values[7] = 15
    test_values[8] = 20

    var expected = alloc[Int](9)
    expected[0] = 0
    expected[1] = 1
    expected[2] = 1
    expected[3] = 2
    expected[4] = 3
    expected[5] = 5
    expected[6] = 55
    expected[7] = 610
    expected[8] = 6765

    for i in range(9):
        var n = test_values[i]
        var expected_result = expected[i]

        # Test top-down
        var result_topdown = fibonacci_topdown(n)
        if result_topdown != expected_result:
            print("Top-down test failed for n =", n)
            print("Expected:", expected_result, "Got:", result_topdown)
            raise Error("Top-down test failed")

        # Test bottom-up
        var result_bottomup = fibonacci_bottomup(n)
        if result_bottomup != expected_result:
            print("Bottom-up test failed for n =", n)
            print("Expected:", expected_result, "Got:", result_bottomup)
            raise Error("Bottom-up test failed")

        print(
            "fib("
            + String(n)
            + ") = "
            + String(result_topdown)
            + " (top-down: ✓, bottom-up: ✓)"
        )

    print("\nAll tests passed! ✓")

    test_values.free()
    expected.free()


def main() raises:
    """Main entry point."""
    try:
        test_fibonacci()
    except e:
        print("Error:", e)
