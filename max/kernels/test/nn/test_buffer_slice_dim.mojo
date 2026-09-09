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
from layout import Coord, TileTensor, coord_to_index_list, row_major
from nn.slice import slice_dim_as_view

from std.utils.index import IndexList


def print_elements[dtype: DType](tensor: TileTensor[dtype, ...]) raises:
    var shape = tensor.layout.shape_coord()
    var stride = coord_to_index_list(tensor.layout.stride_coord())
    print("New shape:", coord_to_index_list(shape))
    print("New strides:", stride)

    @always_inline
    def print_elements_lambda[
        simd_width: Int, alignment: Int = 1
    ](coords: Coord) {var}:
        var idx = tensor.layout(coords)
        print(tensor.raw_load(idx))

    elementwise[1](print_elements_lambda, shape, DeviceContext(api="cpu"))


# slice_dim
def test_slice_dim[
    dtype: DType, numelems: Int, outer_rank: Int, dim: Int
](dims: IndexList[outer_rank], start: Int, stop: Int, step: Int) raises:
    var memory1 = Array[Scalar[dtype], numelems](
        fill_with=lambda (i: Int) -> Scalar[dtype]: Scalar[dtype](i)
    )
    var in_tensor = TileTensor(
        memory1,
        row_major(Coord(dims)),
    )

    var shape = coord_to_index_list(in_tensor.layout.shape_coord())
    var stride = coord_to_index_list(in_tensor.layout.stride_coord())
    print("In shape:", shape)
    print("In strides:", stride)

    # Perform the slice even if we are testing the copy so we get the target size.
    var sliced = slice_dim_as_view[dtype, dim](
        in_tensor,
        start,
        stop,
        step,
    )

    print_elements(sliced)


# CHECK-LABEL: == test_slice_dim_basic
def test_slice_dim_basic() raises:
    print("== test_slice_dim_basic")

    # CHECK-NEXT: In shape: (4, 4)
    # CHECK-NEXT: In strides: (4, 1)
    # CHECK-NEXT: New shape: (2, 4)
    # CHECK-NEXT: New strides: (4, 1)
    # CHECK-NEXT: 8.0
    # CHECK-NEXT: 9.0
    # CHECK-NEXT: 10.0
    # CHECK-NEXT: 11.0
    # CHECK-NEXT: 12.0
    # CHECK-NEXT: 13.0
    # CHECK-NEXT: 14.0
    # CHECK-NEXT: 15.0

    # print(torch.arange(0, 16).reshape(4, 4)[2:4:1, :].flatten())
    test_slice_dim[.float32, 16, 2, 0](IndexList[2](4, 4), 2, 4, 1)

    # CHECK-NEXT: In shape: (4, 4)
    # CHECK-NEXT: In strides: (4, 1)
    # CHECK-NEXT: New shape: (4, 2)
    # CHECK-NEXT: New strides: (4, 1)
    # CHECK-NEXT: 2.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 6.0
    # CHECK-NEXT: 7.0
    # CHECK-NEXT: 10.0
    # CHECK-NEXT: 11.0
    # CHECK-NEXT: 14.0
    # CHECK-NEXT: 15.0

    # print(torch.arange(0, 16).reshape(4, 4)[:, 2:4:1].flatten())
    test_slice_dim[.float32, 16, 2, 1](IndexList[2](4, 4), 2, 4, 1)


def main() raises:
    test_slice_dim_basic()
