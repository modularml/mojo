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

# Rank-1 -> rank-2 broadcasting in TileTensor's in-place binary ops. The
# rank-1 operand's dimension matches the tensor's first dimension and is
# applied per row. Regression test for the broadcast branch falling through
# into the same-rank loop, which double-applied the op to the first elements
# and read the rhs out of range.

# RUN: %mojo %s | FileCheck %s

from layout import TileTensor, row_major


def main():
    var a_data = Array[Float32, 8](fill=0)
    var b_data = Array[Float32, 2](fill=0)
    var a = TileTensor(a_data, row_major[2, 4]())
    var b = TileTensor(b_data, row_major[2]())

    for i in range(2):
        for j in range(4):
            a[i, j] = Float32(i * 4 + j)
    b[0] = 10.0
    b[1] = 20.0

    # CHECK{LITERAL}: [[10.0, 11.0, 12.0, 13.0], [24.0, 25.0, 26.0, 27.0]]
    a += b
    print(a)

    # CHECK{LITERAL}: [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
    a -= b
    print(a)

    # CHECK{LITERAL}: [[0.0, 10.0, 20.0, 30.0], [80.0, 100.0, 120.0, 140.0]]
    a *= b
    print(a)

    # CHECK{LITERAL}: [[10.0, 10.0, 20.0, 30.0], [80.0, 100.0, 120.0, 140.0]]
    a.max(b)
    print(a)
