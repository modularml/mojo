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


from max.algorithm import elementwise
from max.gpu.host import DeviceContext
from layout import (
    TileTensor,
    Coord,
    Idx,
    row_major,
    coord_to_index_list,
)
from nn.arange import arange, arange_shape

from std.utils.index import IndexList


def print_elements(tensor: TileTensor) raises:
    print("New shape:", tensor.layout.shape_coord())
    print("New strides:", tensor.layout.stride_coord())

    @always_inline
    def print_elements_lambda[
        simd_width: Int, alignment: Int = 1
    ](idx: Coord) {var}:
        print(tensor[idx])

    elementwise[1](
        print_elements_lambda,
        tensor.layout.shape_coord(),
        DeviceContext(api="cpu"),
    )


# slice_dim
def test_arange[
    dtype: DType,
](start: Scalar[dtype], stop: Scalar[dtype], step: Scalar[dtype]) raises:
    var outshape: IndexList[1]
    try:
        outshape = arange_shape(start, stop, step)
    except e:
        outshape = IndexList[1]()
        print(e)
    print("Expected output shape: ")
    print(outshape)

    comptime max_output_size = 64

    if max_output_size < outshape[0]:
        print("Memory is larger than static limit, test failed")
        return

    var memory4 = Array[Scalar[dtype], max_output_size](fill={})
    var out_tensor = TileTensor(memory4, row_major(Coord(outshape)))

    @always_inline
    def arange_lambda[simd_width: Int, alignment: Int = 1](idx: Coord) {var}:
        var index = IndexList[1](Int(idx[0].value()))
        var range_val = arange[dtype, simd_width](start, stop, step, index)
        # Extract first element only: idx may have rank > 1 from elementwise,
        # but out_tensor is 1D so we need a single-element coordinate.
        out_tensor.store[width=simd_width](Coord(idx[0]), range_val)

    elementwise[1](
        arange_lambda,
        out_tensor.layout.shape_coord(),
        DeviceContext(api="cpu"),
    )

    print_elements(out_tensor)


# CHECK-LABEL: == test_arrange_basic
def test_arrange_basic() raises:
    print("== test_arrange_basic")

    # CHECK-NEXT: Expected output shape:
    # CHECK-NEXT: (6,)
    # CHECK-NEXT: New shape: (6)
    # CHECK-NEXT: New strides: (1)
    # CHECK-NEXT: 0
    # CHECK-NEXT: 1
    # CHECK-NEXT: 2
    # CHECK-NEXT: 3
    # CHECK-NEXT: 4
    # CHECK-NEXT: 5

    # print(np.arange(0, 6, 1))
    test_arange[.int32](0, 6, 1)

    # CHECK-NEXT: Expected output shape:
    # CHECK-NEXT: (3,)
    # CHECK-NEXT: New shape: (3)
    # CHECK-NEXT: New strides: (1)
    # CHECK-NEXT: 38
    # CHECK-NEXT: 15
    # CHECK-NEXT: -8

    # print(np.arange(38, -13, -23))
    test_arange[.int32](38, -13, -23)


def main() raises:
    test_arrange_basic()
