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


# Docs example *only* includes the signature, to show the elements of a
# parameter list.
def my_sort[
    # infer-only parameters
    dtype: DType,
    width: SIMDLength,
    //,
    # positional-only parameter
    values: SIMD[dtype, width],
    /,
    # positional-or-keyword parameter
    compare: def(Scalar[dtype], Scalar[dtype]) thin -> Int,
    *,
    # keyword-only parameter
    reverse: Bool = False,
]() -> SIMD[dtype, width]:
    var sorted = SIMD[dtype, width](values)
    for output_position in range(width):
        var lowest = output_position
        for compare_position in range(output_position + 1, width):
            if compare(sorted[lowest], sorted[compare_position]) == 1:
                lowest = compare_position
        if lowest != output_position:
            sorted[output_position], sorted[lowest] = (
                sorted[lowest],
                sorted[output_position],
            )
    return sorted


def main():
    comptime dtype = DType.int32
    comptime input2 = SIMD[dtype, 8](9, 3, 3, 1, 11, 10, 5, 2)

    def compare(lhs: Scalar[dtype], rhs: Scalar[dtype]) -> Int:
        if lhs == rhs:
            return 0
        else:
            return -1 if lhs < rhs else 1

    comptime sorted = my_sort[input2, compare, reverse=False]()
    print(sorted)
