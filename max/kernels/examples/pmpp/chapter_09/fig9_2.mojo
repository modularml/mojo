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
# ========================== KERNEL CODE ==========================
def histogram_sequential(
    input: UnsafePointer[mut=False, Float32, _],
    output: UnsafePointer[mut=True, Float32, _],
    w: Int,
    h: Int,
):
    """Sequential histogram calculation.

    Args:
        input: Pointer to input data.
        output: Pointer to output histogram bins.
        w: Width of the data.
        h: Height of the data.
    """
    for i in range(w * h):
        var bin = input[i]
        output[Int(bin)] += 1


# ========================== TEST CODE ==========================
def main() raises:
    """Example usage of histogram_sequential."""
    # Create simple test data: 16 values in range [0, 5)
    var size = 16
    var num_bins = 6

    # Allocate and initialize input
    var input = alloc[Float32](size)
    input[0] = 0.0
    input[1] = 1.0
    input[2] = 2.0
    input[3] = 3.0
    input[4] = 1.0
    input[5] = 2.0
    input[6] = 0.0
    input[7] = 4.0
    input[8] = 1.0
    input[9] = 2.0
    input[10] = 3.0
    input[11] = 0.0
    input[12] = 1.0
    input[13] = 2.0
    input[14] = 3.0
    input[15] = 1.0

    # Allocate and initialize output bins to zero
    var output = alloc[Float32](num_bins)
    for i in range(num_bins):
        output[i] = 0.0

    # Run histogram
    histogram_sequential(input, output, 4, 4)

    # Print results
    print("Histogram results:")
    for i in range(num_bins):
        print("  Bin", i, ":", Int(output[i]))

    # Verify expected results
    # Expected: bin 0=3, bin 1=5, bin 2=4, bin 3=3, bin 4=1
    var expected = List[Int]()
    expected.append(3)
    expected.append(5)
    expected.append(4)
    expected.append(3)
    expected.append(1)
    expected.append(0)
    var passed = True
    for i in range(num_bins):
        if Int(output[i]) != expected[i]:
            passed = False
            print(
                "FAIL: Expected bin",
                i,
                "to be",
                expected[i],
                "but got",
                Int(output[i]),
            )

    if passed:
        print("✓ Test PASSED")
    else:
        print("✗ Test FAILED")

    # Clean up
    input.free()
    output.free()
