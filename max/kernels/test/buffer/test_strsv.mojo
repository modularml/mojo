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

from std.testing import TestSuite

comptime simd_width = 8


def strsv[
    size: Int
](L_ptr_in: ImmPointer[Float32, _], x_ptr_in: MutPointer[Float32, _],):
    # assuming size is a multiple of simd_width
    var x_ptr = x_ptr_in
    var L_ptr = L_ptr_in
    var n: Int = size
    var x_solved_storage = Array[Float32, simd_width * simd_width](fill={})
    var x_solved_ptr = x_solved_storage.unsafe_ptr()
    var x_solved = x_solved_ptr

    while True:
        for j in range(simd_width):
            var x_j = x_ptr[j]
            for i in range(j + 1, simd_width):
                x_ptr[i] = x_j.fma(-L_ptr[i + j * size], x_ptr[i])

        n -= simd_width
        if n <= 0:
            return

        # Save the solution of the triangular tile in stack, while
        # packing them as simd vectors.
        var x_vec: SIMD[.float32, simd_width]
        for i in range(simd_width):
            # Broadcast one solution value to a simd vector.
            x_vec = x_ptr[i]
            x_solved.store(i * simd_width, x_vec)

        x_ptr += simd_width
        L_ptr += simd_width

        # Update the columns under the triangular tile
        # Move down tile by tile.
        for i in range(0, n, simd_width):
            x_vec = x_ptr.load[width=simd_width](i)
            # Move to right column by column within in a tile.
            for j in range(simd_width):
                var x_solved_vec = x_solved.load[width=simd_width](
                    j * simd_width
                )
                var L_col_vec = L_ptr.load[width=simd_width](i + j * size)
                x_vec = x_solved_vec.fma(-L_col_vec, x_vec)
            x_ptr.store(i, x_vec)

        L_ptr += size * simd_width


# Fill the lower triangle matrix.
def fill_L[size: Int](L: MutPointer[Float32, _]):
    for j in range(size):
        for i in range(size):
            if i == j:
                L[i + j * size] = 1.0
            else:
                L[i + j * size] = -2.5 / Float32(size * (size - 1))


# Fill the rhs, which is also used to save the solution vector.
def fill_x[size: Int](x: MutPointer[Float32, _]):
    for i in range(size):
        x[i] = 1.0


def naive_strsv[
    size: Int
](L: ImmPointer[Float32, _], x: MutPointer[Float32, _],):
    for j in range(size):
        var x_j = x[j]
        for i in range(j + 1, size):
            x[i] = x[i] - x_j * L[i + j * size]


# CHECK-LABEL: test_strsv
def test_strsv() raises:
    print("== test_strsv")

    comptime size: Int = 64
    var l_stack = Array[Float32, size * size](fill={})
    var x0_stack = Array[Float32, size](fill={})
    var x1_stack = Array[Float32, size](fill={})

    var L = l_stack.unsafe_ptr()
    var x0_ptr = x0_stack.unsafe_ptr()
    var x0 = x0_ptr
    var x1_ptr = x1_stack.unsafe_ptr()
    var x1 = x1_ptr

    fill_L[size](L)
    fill_x[size](x0)
    fill_x[size](x1)
    naive_strsv[size](L, x0)
    strsv[size](L, x1)

    var err: Float32 = 0.0
    for i in range(size):
        err += abs(x0[i] - x1[i])

    # CHECK: 0.0
    print(err)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
