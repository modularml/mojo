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
#
# This file only tests the Apple AMX matmul functionality which is defined as a
# A^T.B where A and B are 16x16 Float32 matrices.
#
# ===----------------------------------------------------------------------=== #

from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from linalg.arch.cpu.apple_amx_intrinsics import *
from std.testing import *


def fill_a(buf: TileTensor[mut=True, ...]):
    comptime assert buf.flat_rank == 2
    # Fills the A matrix with the following values row + 2*col
    for i in range(Int(buf.dim[0]())):
        for j in range(Int(buf.dim[1]())):
            buf[i, j] = Scalar[buf.dtype](i // (j + 1) + j)


def fill_b(buf: TileTensor[mut=True, ...]):
    comptime assert buf.flat_rank == 2
    # Fills the A matrix with the following values row/(col + 1) + col
    for i in range(Int(buf.dim[0]())):
        for j in range(Int(buf.dim[1]())):
            buf[i, j] = Scalar[buf.dtype](i // (j + 1) + j)


def clear_c(buf: TileTensor[mut=True, ...]):
    _ = buf.fill(0)


def test_dot_at_b[dtype: DType, M: Int, N: Int]() raises:
    # Create TileTensors with static shapes
    var a_matrix = stack_allocation[dtype=dtype](row_major[M, N]())
    var b_matrix = stack_allocation[dtype=dtype](row_major[M, N]())
    var c_matrix = stack_allocation[dtype=dtype](row_major[M, N]())

    fill_a(a_matrix)
    fill_b(b_matrix)
    clear_c(c_matrix)

    dot_at_b(c_matrix, a_matrix, b_matrix)

    for m in range(Int(c_matrix.dim[0]())):
        for n in range(Int(c_matrix.dim[1]())):
            var golden = Scalar[dtype](0)
            for k in range(Int(a_matrix.dim[1]())):
                golden += a_matrix[k, m][0] * b_matrix[k, n][0]
            assert_almost_equal(
                c_matrix[m, n],
                golden,
                msg=String(t"invalid value at m={m},n={n}"),
            )


def main() raises:
    test_dot_at_b[.float32, 16, 16]()
    test_dot_at_b[.float16, 32, 32]()
