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

"""Figure 15.4 - Clear function for initializing accumulator array."""

comptime tM = 8
comptime tN = 4


@always_inline
def clear(C_r: SIMD[.float32, tM * tN]) -> SIMD[.float32, tM * tN]:
    """Clear accumulator array (initialize to zero).

    Args:
        C_r: Register array to clear (tM x tN elements as SIMD vector).

    Returns:
        Cleared SIMD vector (all zeros).
    """
    return SIMD[.float32, tM * tN](0.0)


def main():
    print("Figure 15.4 - Clear function for accumulator arrays")
    print("This is a helper function used in matrix multiplication kernels.")

    # Test the function
    var test_array = SIMD[.float32, tM * tN](5.0)
    print("Before clear: sum =", test_array.reduce_add())

    test_array = clear(test_array)
    print("After clear: sum =", test_array.reduce_add())

    print("SUCCESS: Module loaded and tested correctly")
