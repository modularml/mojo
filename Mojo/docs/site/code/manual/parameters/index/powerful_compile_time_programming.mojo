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


def slice[
    dtype: DType, size: SIMDLength, //
](x: SIMD[dtype, size], offset: Int) -> SIMD[dtype, size // 2]:
    comptime new_size = size // 2
    var result = SIMD[dtype, new_size]()
    for i in range(new_size):
        result[i] = Scalar[dtype](x[i + offset])
    return result


def reduce_add(x: SIMD) -> Int:
    comptime if x.length == 1:
        return Int(x[0])
    elif x.length == 2:
        return Int(x[0]) + Int(x[1])

    # Extract the top/bottom halves, add them, sum the elements.
    comptime half_size = x.length // 2
    var lhs = slice(x, 0)
    var rhs = slice(x, half_size)
    return reduce_add(lhs + rhs)


def main():
    var x = SIMD[.int, 4](1, 2, 3, 4)
    print(x)
    print("Elements sum:", reduce_add(x))
