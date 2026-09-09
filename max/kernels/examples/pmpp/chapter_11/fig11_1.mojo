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


# ========================== CPU CODE ==========================
def sequential_scan(
    input: UnsafePointer[mut=False, Float32, _],
    output: UnsafePointer[mut=True, Float32, _],
    N: UInt32,
):
    """Sequential scan (inclusive scan) on CPU.

    Args:
        input: Input array.
        output: Output array for scan result.
        N: Number of elements.
    """
    output[0] = input[0]
    for i in range(1, Int(N)):
        output[i] = output[i - 1] + input[i]


# ========================== TEST CODE ==========================
def main() raises:
    """Example usage of sequential scan."""
    comptime N = 16

    # Allocate host memory
    var h_input = alloc[Float32](N)
    var h_output = alloc[Float32](N)

    # Initialize input with simple values
    print("Input: ", end="")
    for i in range(N):
        h_input[i] = 1.0
        print(h_input[i], " ", end="")
    print()

    # Run sequential scan
    sequential_scan(h_input, h_output, N)

    # Print results
    print("Output (inclusive scan): ", end="")
    for i in range(N):
        print(h_output[i], " ", end="")
    print()

    # Verify correctness
    var passed = True
    for i in range(N):
        var expected = Float32(i + 1)
        if h_output[i] != expected:
            print(
                "Error at index",
                i,
                ": expected",
                expected,
                ", got",
                h_output[i],
            )
            passed = False

    if passed:
        print("SUCCESS: All values match!")
    else:
        print("FAILED: Found errors")

    # Cleanup
    h_input.free()
    h_output.free()
