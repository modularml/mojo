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


def main():
    var vector = SIMD[.int16, 4](1, 2, 3, 4)
    vector = vector * vector
    for i in range(4):
        print(vector[i], end=" ")
    print()

    # Example: "Using parameterized types and functions"

    # start-simd-usage-example
    # Make a vector of 4 floats.
    var small_vec = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)

    # Make a big vector containing 1.0 in float16 format.
    var big_vec = SIMD[.float16, 32](1.0)

    # Do some math and convert the elements to float32.
    var bigger_vec = (big_vec + big_vec).cast[.float32]()

    # You can write types out explicitly if you want of course.
    var bigger_vec2: SIMD[DType.float32, 32] = bigger_vec

    print("small_vec DType:", small_vec.dtype, "length:", Int(small_vec.length))
    print(
        "bigger_vec2 DType:",
        bigger_vec2.dtype,
        "length:",
        Int(bigger_vec2.length),
    )
    # end-simd-usage-example

    # second example

    from std.math import sqrt

    def rsqrt[
        dt: DType, width: SIMDLength
    ](x: SIMD[dt, width]) -> SIMD[dt, width]:
        return 1 / sqrt(x)

    var v = SIMD[.float16, 4](42)
    print(rsqrt(v))

    _ = small_vec
    _ = bigger_vec2
