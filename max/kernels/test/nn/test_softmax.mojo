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

from std.sys.info import simd_width_of

from layout._fillers import arange
from layout import (
    Idx,
    TileTensor,
    row_major,
)
from nn.softmax import logsoftmax_inline, softmax_2_pass


from std.utils import IndexList


# CHECK-LABEL: test_logsoftmax
def test_logsoftmax() raises:
    print("== test_logsoftmax")
    comptime type = DType.float32
    comptime simd_width = simd_width_of[type]()

    def logsoftmax_test_nd[rank: Int, shape: IndexList[rank]]() raises:
        comptime if rank == 1:
            var in_stack = Array[Scalar[type], shape[0]](fill={})
            var in_tt = TileTensor(in_stack, row_major[shape[0]]())
            var out_stack = Array[Scalar[type], shape[0]](fill=0)
            arange(in_tt)
            var out_tt = TileTensor(out_stack, row_major[shape[0]]())
            logsoftmax_inline[type, simd_width, rank](in_tt, out_tt, rank - 1)
            for i in range(out_tt.num_elements()):
                print(out_tt[i])
        else:
            comptime if rank == 2:
                comptime sz = shape[0] * shape[1]
                var in_stack = Array[Scalar[type], sz](fill={})
                var in_tt = TileTensor(
                    in_stack, row_major[shape[0], shape[1]]()
                )
                var out_stack = Array[Scalar[type], sz](fill=0)
                arange(in_tt)
                var out_tt = TileTensor(
                    out_stack, row_major[shape[0], shape[1]]()
                )
                logsoftmax_inline[type, simd_width, rank](
                    in_tt, out_tt, rank - 1
                )
                # Print as flat 1D
                var out_flat = TileTensor(out_tt._storage, row_major[sz]())
                for i in range(out_flat.num_elements()):
                    print(out_flat[i])

    logsoftmax_test_nd[1, IndexList[1](5)]()

    # CHECK: -4.45191{{[0-9]+}}
    # CHECK-NEXT: -3.451914{{[0-9]+}}
    # CHECK-NEXT: -2.451914{{[0-9]+}}
    # CHECK-NEXT: -1.451914{{[0-9]+}}
    # CHECK-NEXT: -0.451914{{[0-9]+}}

    logsoftmax_test_nd[2, IndexList[2](3, 4)]()

    # CHECK: -3.440189{{[0-9]+}}
    # CHECK-NEXT: -2.440189{{[0-9]+}}
    # CHECK-NEXT: -1.440189{{[0-9]+}}
    # CHECK-NEXT: -0.440189{{[0-9]+}}
    # CHECK-NEXT: -3.440189{{[0-9]+}}
    # CHECK-NEXT: -2.440189{{[0-9]+}}
    # CHECK-NEXT: -1.440189{{[0-9]+}}
    # CHECK-NEXT: -0.440189{{[0-9]+}}
    # CHECK-NEXT: -3.440189{{[0-9]+}}
    # CHECK-NEXT: -2.440189{{[0-9]+}}
    # CHECK-NEXT: -1.440189{{[0-9]+}}
    # CHECK-NEXT: -0.440189{{[0-9]+}}


# CHECK-LABEL: test_softmax_2pass
def test_softmax_2pass():
    print("== test_softmax_2pass")
    comptime type = DType.float32
    comptime simd_width = simd_width_of[type]()
    comptime sz = 5

    var in_stack = Array[Scalar[type], sz](
        fill_with=lambda (i: Int) -> Scalar[type]: Scalar[type](i)
    )
    var in_tt = TileTensor(in_stack, row_major[sz]())
    var out_stack = Array[Scalar[type], sz](fill=0)
    var out_tt = TileTensor(out_stack, row_major[sz]())

    softmax_2_pass[simd_width, type](out_tt, in_tt)

    for i in range(sz):
        print(out_tt[i])

    # CHECK: 0.01165{{[0-9]+}}
    # CHECK-NEXT: 0.03168{{[0-9]+}}
    # CHECK-NEXT: 0.08612{{[0-9]+}}
    # CHECK-NEXT: 0.23412{{[0-9]+}}
    # CHECK-NEXT: 0.63640{{[0-9]+}}


def main() raises:
    test_logsoftmax()
    test_softmax_2pass()
