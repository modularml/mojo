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
from nn.slice import slice_as_copy, slice_as_view

from std.utils.index import Index, IndexList


def print_elements[dtype: DType](tensor: TileTensor[dtype, ...]) raises:
    var shape = tensor.layout.shape_coord()
    var stride = coord_to_index_list(tensor.layout.stride_coord())
    print("New shape:", coord_to_index_list(shape))
    print("New strides:", stride)

    @always_inline
    def print_elements_lambda[
        simd_width: Int, alignment: Int = 1
    ](coords: Coord) {var}:
        print(tensor.load(coords))

    elementwise[1](print_elements_lambda, shape, DeviceContext(api="cpu"))


# slice_dim
def test_slice[
    dtype: DType,
    numelems: Int,
    outer_rank: Int,
    static_shape: IndexList[outer_rank],
](
    dims: IndexList[outer_rank],
    starts: IndexList[outer_rank],
    stops: IndexList[outer_rank],
    steps: IndexList[outer_rank],
    use_copy: Bool,
) raises:
    # Isn't always used but is used for the output buffer if we copy.
    var output_mem = Array[Scalar[dtype], numelems](fill={})

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

    var start_tensor_mem = Array[Int, outer_rank](
        fill_with=lambda (dim: Int) -> Int: Int(starts[dim])
    )
    var start_tensor = TileTensor(
        start_tensor_mem,
        row_major(Coord(IndexList[1](outer_rank))),
    )

    var end_tensor_mem = Array[Int, outer_rank](
        fill_with=lambda (dim: Int) -> Int: Int(stops[dim])
    )
    var end_tensor = TileTensor(
        end_tensor_mem,
        row_major(Coord(IndexList[1](outer_rank))),
    )

    var step_tensor_mem = Array[Int, outer_rank](
        fill_with=lambda (dim: Int) -> Int: Int(steps[dim])
    )
    var step_tensor = TileTensor(
        step_tensor_mem,
        row_major(Coord(IndexList[1](outer_rank))),
    )

    # Perform the slice even if we are testing the copy so we get the target size.
    var sliced = slice_as_view(
        in_tensor,
        start_tensor,
        end_tensor,
        step_tensor,
    )

    if not use_copy:
        print_elements(sliced)
    else:
        print("As copy")

        var sliced_shape = coord_to_index_list(sliced.layout.shape_coord())
        var output_buffer = TileTensor(
            output_mem,
            row_major(Coord(sliced_shape)),
        )

        slice_as_copy(
            output_buffer,
            in_tensor,
            start_tensor,
            end_tensor,
            step_tensor,
            DeviceContext(api="cpu"),
        )

        print_elements(output_buffer)


# CHECK-LABEL: == test_slice_basic
def test_slice_basic() raises:
    print("== test_slice_basic")

    # CHECK-NEXT: In shape: (4, 4, 4)
    # CHECK-NEXT: In strides: (16, 4, 1)
    # CHECK-NEXT: New shape: (2, 2, 2)
    # CHECK-NEXT: New strides: (16, 4, 1)
    # CHECK-NEXT: 42.0
    # CHECK-NEXT: 43.0
    # CHECK-NEXT: 46.0
    # CHECK-NEXT: 47.0
    # CHECK-NEXT: 58.0
    # CHECK-NEXT: 59.0
    # CHECK-NEXT: 62.0
    # CHECK-NEXT: 63.0

    # print(torch.arange(0, 64).reshape(4, 4, 4)[2:4:1, 2:4:1, 2:4:1].flatten())
    test_slice[.float32, 64, 3, IndexList[3](4, 4, 4)](
        IndexList[3](4, 4, 4),
        Index(2, 2, 2),
        Index(4, 4, 4),
        Index(1, 1, 1),
        False,
    )

    # CHECK-NEXT: In shape: (4, 4, 4)
    # CHECK-NEXT: In strides: (16, 4, 1)
    # CHECK-NEXT: New shape: (2, 2, 2)
    # CHECK-NEXT: New strides: (16, 4, 1)
    # CHECK-NEXT: 42
    # CHECK-NEXT: 43
    # CHECK-NEXT: 46
    # CHECK-NEXT: 47
    # CHECK-NEXT: 58
    # CHECK-NEXT: 59
    # CHECK-NEXT: 62
    # CHECK-NEXT: 63
    test_slice[.uint8, 64, 3, IndexList[3](4, 4, 4)](
        IndexList[3](4, 4, 4),
        Index(2, 2, 2),
        Index(4, 4, 4),
        Index(1, 1, 1),
        False,
    )


# CHECK-LABEL: == test_slice_identity
def test_slice_identity() raises:
    print("== test_slice_identity")

    # CHECK-NEXT: In shape: (2, 2, 4)
    # CHECK-NEXT: In strides: (8, 4, 1)
    # CHECK-NEXT: New shape: (2, 2, 4)
    # CHECK-NEXT: New strides: (8, 4, 1)
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 2.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 4.0
    # CHECK-NEXT: 5.0
    # CHECK-NEXT: 6.0
    # CHECK-NEXT: 7.0
    # CHECK-NEXT: 8.0
    # CHECK-NEXT: 9.0
    # CHECK-NEXT: 10.0
    # CHECK-NEXT: 11.0
    # CHECK-NEXT: 12.0
    # CHECK-NEXT: 13.0
    # CHECK-NEXT: 14.0
    # CHECK-NEXT: 15.0

    # print(torch.arange(0, 16).reshape(2, 2, 4)[0:2:1, 0:2:1, 0:4:1].flatten())

    # Check slicing along all dimensions returns the original tensor.
    test_slice[.float32, 16, 3, IndexList[3](2, 2, 4)](
        IndexList[3](2, 2, 4),
        Index(0, 0, 0),
        Index(2, 2, 4),
        Index(1, 1, 1),
        False,
    )


# CHECK-LABEL: == test_slice_steps
def test_slice_steps() raises:
    print("== test_slice_steps")

    # CHECK-NEXT: In shape: (2, 4, 8)
    # CHECK-NEXT: In strides: (32, 8, 1)
    # CHECK-NEXT: New shape: (1, 2, 4)
    # CHECK-NEXT: New strides: (64, 16, 2)
    # CHECK-NEXT: 0.0
    # CHECK-NEXT: 2.0
    # CHECK-NEXT: 4.0
    # CHECK-NEXT: 6.0
    # CHECK-NEXT: 16.0
    # CHECK-NEXT: 18.0
    # CHECK-NEXT: 20.0
    # CHECK-NEXT: 22.0

    # print(torch.arange(0, 64).reshape(2, 4, 8)[0:2:2, 0:4:2, 0:8:2].flatten())
    test_slice[.float32, 64, 3, IndexList[3](2, 4, 8)](
        IndexList[3](2, 4, 8),
        Index(0, 0, 0),
        Index(2, 4, 8),
        Index(2, 2, 2),
        False,
    )


# CHECK-LABEL: == test_slice_1D
def test_slice_1D() raises:
    print("== test_slice_1D")

    # CHECK-NEXT: In shape: (64,)
    # CHECK-NEXT: In strides: (1,)
    # CHECK-NEXT: New shape: (4,)
    # CHECK-NEXT: New strides: (4,)
    # CHECK-NEXT: 16.0
    # CHECK-NEXT: 20.0
    # CHECK-NEXT: 24.0
    # CHECK-NEXT: 28.0

    # print(torch.arange(0, 64)[16:30:4].flatten())
    test_slice[.float32, 64, 1, IndexList[1](64)](
        IndexList[1](64), Index(16), Index(30), Index(4), False
    )


# CHECK-LABEL: == test_slice_empty
def test_slice_empty() raises:
    print("== test_slice_empty")

    # CHECK-NEXT: In shape: (64,)
    # CHECK-NEXT: In strides: (1,)
    # CHECK-NEXT: New shape: (0,)
    # CHECK-NEXT: New strides: (1,)

    # print(torch.arange(0, 64)[8:8:1].flatten())
    test_slice[.float32, 64, 1, IndexList[1](64)](
        IndexList[1](64), Index(8), Index(8), Index(1), False
    )


# CHECK-LABEL: == test_slice_4D
def test_slice_4D() raises:
    print("== test_slice_4D")

    # CHECK-NEXT: In shape: (2, 4, 4, 2)
    # CHECK-NEXT: In strides: (32, 8, 2, 1)
    # CHECK-NEXT: New shape: (1, 1, 4, 1)
    # CHECK-NEXT: New strides: (32, 16, 2, 1)
    # CHECK-NEXT: 49.0
    # CHECK-NEXT: 51.0
    # CHECK-NEXT: 53.0
    # CHECK-NEXT: 55.0

    # print(torch.arange(0, 64).reshape(2, 4, 4, 2)[1:2:1, 2:4:2, 0:4:1, 1:2:1].flatten())
    test_slice[.float32, 64, 4, IndexList[4](2, 4, 4, 2)](
        IndexList[4](2, 4, 4, 2),
        Index(1, 2, 0, 1),
        Index(2, 4, 4, 2),
        Index(1, 2, 1, 1),
        False,
    )


# CHECK-LABEL: == test_slice_copy
def test_slice_copy() raises:
    print("== test_slice_copy")

    # CHECK-NEXT: In shape: (2, 4, 4, 2)
    # CHECK-NEXT: In strides: (32, 8, 2, 1)
    # CHECK-NEXT: As copy
    # CHECK-NEXT: New shape: (1, 1, 4, 1)

    # Strides should be contiguous in the copy.
    # CHECK-NEXT: New strides: (4, 4, 1, 1)
    # CHECK-NEXT: 49.0
    # CHECK-NEXT: 51.0
    # CHECK-NEXT: 53.0
    # CHECK-NEXT: 55.0

    # print(torch.arange(0, 64).reshape(2, 4, 4, 2)[1:2:1, 2:4:2, 0:4:1, 1:2:1].flatten())
    test_slice[.float32, 64, 4, IndexList[4](2, 4, 4, 2)](
        IndexList[4](2, 4, 4, 2),
        Index(1, 2, 0, 1),
        Index(2, 4, 4, 2),
        Index(1, 2, 1, 1),
        True,
    )


# CHECK-LABEL: == test_slice_negative
def test_slice_negative() raises:
    print("== test_slice_negative")

    # CHECK-NEXT: In shape: (2, 4, 4, 2)
    # CHECK-NEXT: In strides: (32, 8, 2, 1)
    # CHECK-NEXT: New shape: (1, 2, 4, 1)
    # CHECK-NEXT: New strides: (32, 16, 2, 1)

    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 5.0
    # CHECK-NEXT: 7.0

    # CHECK-NEXT: 17.0
    # CHECK-NEXT: 19.0
    # CHECK-NEXT: 21.0
    # CHECK-NEXT: 23.0

    # print(torch.arange(0, 64).reshape(2, 4, 4, 2)[-2:-1:1, -4:-1:2, -4:4:1, -1:2:1].flatten())
    test_slice[.float32, 64, 4, IndexList[4](2, 4, 4, 2)](
        IndexList[4](2, 4, 4, 2),
        Index(-2, -4, -4, -1),
        Index(-1, -1, 4, 2),
        Index(1, 2, 1, 1),
        False,
    )


# CHECK-LABEL: == test_slice_negative_step_1D
def test_slice_negative_step_1D() raises:
    print("== test_slice_negative_step_1D")

    # CHECK: In shape: (15,)
    # CHECK-NEXT: In strides: (1,)
    # CHECK-NEXT: New shape: (6,)
    # CHECK-NEXT: New strides: (-1,)

    # CHECK-NEXT: 14.0
    # CHECK-NEXT: 13.0
    # CHECK-NEXT: 12.0
    # CHECK-NEXT: 11.0
    # CHECK-NEXT: 10.0
    # CHECK-NEXT: 9.0

    # print(np.arange(0, 15)[14:8:-1])
    test_slice[.float32, 15, 1, IndexList[1](15)](
        IndexList[1](15),
        Index(
            14,
        ),
        Index(
            8,
        ),
        Index(
            -1,
        ),
        False,
    )


# CHECK-LABEL: == test_slice_negative_step_2D
def test_slice_negative_step_2D() raises:
    print("== test_slice_negative_step_2D")

    # CHECK: In shape: (16, 4)
    # CHECK-NEXT: In strides: (4, 1)
    # CHECK-NEXT: New shape: (4, 2)
    # CHECK-NEXT: New strides: (-8, -1)

    # CHECK-NEXT: 59.0
    # CHECK-NEXT: 58.0
    # CHECK-NEXT: 51.0
    # CHECK-NEXT: 50.0
    # CHECK-NEXT: 43.0
    # CHECK-NEXT: 42.0
    # CHECK-NEXT: 35.0
    # CHECK-NEXT: 34.0

    # print(np.arange(0, 64).reshape(16, 4)[14:6:-2, -1:1:-1])
    test_slice[.float32, 64, 2, IndexList[2](16, 4)](
        IndexList[2](16, 4),
        Index(14, -1),
        Index(6, 1),
        Index(-2, -1),
        False,
    )


# CHECK-LABEL: == test_slice_negative_step_3D
def test_slice_negative_step_3D() raises:
    print("== test_slice_negative_step_3D")

    # CHECK: In shape: (8, 2, 4)
    # CHECK-NEXT: In strides: (8, 4, 1)
    # CHECK-NEXT: New shape: (2, 2, 2)
    # CHECK-NEXT: New strides: (-16, 4, -2)

    # CHECK-NEXT: 59.0
    # CHECK-NEXT: 57.0
    # CHECK-NEXT: 63.0
    # CHECK-NEXT: 61.0
    # CHECK-NEXT: 43.0
    # CHECK-NEXT: 41.0
    # CHECK-NEXT: 47.0
    # CHECK-NEXT: 45.0

    # print(np.arange(0, 64).reshape(8, 2, 4)[-1:4:-2, :, 4:0:-2])
    test_slice[.float32, 64, 3, IndexList[3](8, 2, 4)](
        IndexList[3](8, 2, 4),
        Index(-1, 0, -1),
        Index(4, 2, 0),
        Index(-2, 1, -2),
        False,
    )


# CHECK-LABEL: == test_slice_negative_step_4D
def test_slice_negative_step_4D() raises:
    print("== test_slice_negative_step_4D")

    # CHECK: In shape: (2, 4, 2, 4)
    # CHECK-NEXT: In strides: (32, 8, 4, 1)
    # CHECK-NEXT: New shape: (1, 2, 1, 3)
    # CHECK-NEXT: New strides: (-32, -16, -4, -1)

    # CHECK-NEXT: 63.0
    # CHECK-NEXT: 62.0
    # CHECK-NEXT: 61.0
    # CHECK-NEXT: 47.0
    # CHECK-NEXT: 46.0
    # CHECK-NEXT: 45.0

    # print(np.arange(0, 64).reshape(2, 4, 2, 4)[-1:0:-1, -1:0:-2, -1:0:-1, -1:0:-1].stride)
    test_slice[.float32, 64, 4, IndexList[4](2, 4, 2, 4)](
        IndexList[4](2, 4, 2, 4),
        Index(-1, -1, -1, -1),
        Index(0, 0, 0, 0),
        Index(-1, -2, -1, -1),
        False,
    )


# CHECK-LABEL: == test_slice_negative_step_2
def test_slice_negative_step_2() raises:
    print("== test_slice_negative_step_2")

    # CHECK: In shape: (3, 3)
    # CHECK-NEXT: In strides: (3, 1)
    # CHECK-NEXT: New shape: (1, 1)
    # CHECK-NEXT: New strides: (-3, -1)

    # CHECK-NEXT: 8.0

    # print(np.arange(0, 9).reshape(3,3)[3:-2:-1, 3:-2:-1])
    test_slice[.float32, 9, 2, IndexList[2](3, 3)](
        IndexList[2](3, 3),
        Index(3, 3),
        Index(-2, -2),
        Index(-1, -1),
        False,
    )


# CHECK-LABEL: == test_slice_negative_step_3
def test_slice_negative_step_3() raises:
    print("== test_slice_negative_step_3")

    # CHECK: In shape: (3, 3)
    # CHECK-NEXT: In strides: (3, 1)
    # CHECK-NEXT: New shape: (2, 2)
    # CHECK-NEXT: New strides: (-3, -1)

    # CHECK-NEXT: 8.0
    # CHECK-NEXT: 7.0
    # CHECK-NEXT: 5.0
    # CHECK-NEXT: 4.0

    # print(np.arange(0, 9).reshape(3,3)[3:-3:-1, 3:-3:-1])
    test_slice[.float32, 9, 2, IndexList[2](3, 3)](
        IndexList[2](3, 3),
        Index(3, 3),
        Index(-3, -3),
        Index(-1, -1),
        False,
    )


# CHECK-LABEL: == test_slice_negative_step_4
def test_slice_negative_step_4() raises:
    print("== test_slice_negative_step_4")

    # CHECK: In shape: (3, 3)
    # CHECK-NEXT: In strides: (3, 1)
    # CHECK-NEXT: New shape: (3, 3)
    # CHECK-NEXT: New strides: (-3, -1)

    # CHECK-NEXT: 8.0
    # CHECK-NEXT: 7.0
    # CHECK-NEXT: 6.0
    # CHECK-NEXT: 5.0
    # CHECK-NEXT: 4.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 2.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 0.0

    # print(np.arange(0, 9).reshape(3,3)[3:-3:-1, 3:-3:-1])
    test_slice[.float32, 9, 2, IndexList[2](3, 3)](
        IndexList[2](3, 3),
        Index(3, 3),
        Index(-4, -4),
        Index(-1, -1),
        False,
    )


# CHECK-LABEL: == test_truncated_last_dim
def test_truncated_last_dim() raises:
    print("== test_truncated_last_dim")

    # CHECK: In shape: (3, 3)
    # CHECK-NEXT: In strides: (3, 1)
    # CHECK-NEXT: New shape: (2, 2)
    # CHECK-NEXT: New strides: (3, 2)

    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 5.0
    # CHECK-NEXT: 6.0
    # CHECK-NEXT: 8.0

    # print(torch.arange(0, 9).reshape(3,3)[1:56:1, 0:234567:2])
    test_slice[.float32, 9, 2, IndexList[2](3, 3)](
        IndexList[2](3, 3),
        Index(1, 0),
        Index(56, 234567),
        Index(1, 2),
        False,
    )


# CHECK-LABEL: == test_truncated_first_and_last_dim
def test_truncated_first_and_last_dim() raises:
    print("== test_truncated_first_and_last_dim")

    # CHECK: In shape: (3, 3)
    # CHECK-NEXT: In strides: (3, 1)
    # CHECK-NEXT: New shape: (0, 0)
    # CHECK-NEXT: New strides: (3, 2)

    # print(torch.arange(0, 9).reshape(3,3)[3:56:1, 60:234567:2])
    test_slice[.float32, 9, 2, IndexList[2](3, 3)](
        IndexList[2](3, 3),
        Index(3, 60),
        Index(56, 234567),
        Index(1, 2),
        False,
    )


# CHECK-LABEL: == test_truncated_last_dim_reverse
def test_truncated_last_dim_reverse() raises:
    print("== test_truncated_last_dim_reverse")

    # CHECK: In shape: (3, 3)
    # CHECK-NEXT: In strides: (3, 1)
    # CHECK-NEXT: New shape: (3, 2)
    # CHECK-NEXT: New strides: (-3, -2)

    # CHECK-NEXT: 8.0
    # CHECK-NEXT: 6.0
    # CHECK-NEXT: 5.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 2.0
    # CHECK-NEXT: 0.0

    # print(np.arange(0, 9).reshape(3,3)[323534:-242:-1, 435432:-3242:-2])
    test_slice[.float32, 9, 2, IndexList[2](3, 3)](
        IndexList[2](3, 3),
        Index(323534, 435432),
        Index(-242, -3242),
        Index(-1, -2),
        False,
    )


# CHECK-LABEL: == test_truncated_first_and_last_dim_reverse
def test_truncated_first_and_last_dim_reverse() raises:
    print("== test_truncated_first_and_last_dim_reverse")

    # CHECK: In shape: (3, 3)
    # CHECK-NEXT: In strides: (3, 1)
    # CHECK-NEXT: New shape: (0, 0)
    # CHECK-NEXT: New strides: (-3, -2)

    # print(np.arange(0, 9).reshape(3,3)[-30:-242:-1, -40:-3242:-2])
    test_slice[.float32, 9, 2, IndexList[2](3, 3)](
        IndexList[2](3, 3),
        Index(-30, -40),
        Index(-242, -3242),
        Index(-1, -2),
        False,
    )


# CHECK-LABEL: == test_last_dim_edge
def test_last_dim_edge() raises:
    print("== test_last_dim_edge")

    # CHECK: In shape: (3, 3)
    # CHECK-NEXT: In strides: (3, 1)
    # CHECK-NEXT: New shape: (2, 2)
    # CHECK-NEXT: New strides: (-3, -1)

    # CHECK-NEXT: 8.0
    # CHECK-NEXT: 7.0
    # CHECK-NEXT: 5.0
    # CHECK-NEXT: 4.0

    # print(np.arange(0, 9).reshape(3,3)[2:0:-1, 2:0:-1]
    test_slice[.float32, 9, 2, IndexList[2](3, 3)](
        IndexList[2](3, 3),
        Index(2, 2),
        Index(0, 0),
        Index(-1, -1),
        False,
    )


# CHECK-LABEL: == test_last_dim_edge_2
def test_last_dim_edge_2() raises:
    print("== test_last_dim_edge_2")

    # CHECK: In shape: (3, 3)
    # CHECK-NEXT: In strides: (3, 1)
    # CHECK-NEXT: New shape: (2, 2)
    # CHECK-NEXT: New strides: (-3, -1)

    # CHECK-NEXT: 8.0
    # CHECK-NEXT: 7.0
    # CHECK-NEXT: 5.0
    # CHECK-NEXT: 4.0

    # print(np.arange(0, 9).reshape(3,3)[3:0:-1, 3:0:-1])
    test_slice[.float32, 9, 2, IndexList[2](3, 3)](
        IndexList[2](3, 3),
        Index(3, 3),
        Index(0, 0),
        Index(-1, -1),
        False,
    )


# CHECK-LABEL: == test_last_dim_edge_3
def test_last_dim_edge_3() raises:
    print("== test_last_dim_edge_3")

    # CHECK: In shape: (3, 3)
    # CHECK-NEXT: In strides: (3, 1)
    # CHECK-NEXT: New shape: (2, 2)
    # CHECK-NEXT: New strides: (-3, -1)

    # CHECK-NEXT: 8.0
    # CHECK-NEXT: 7.0
    # CHECK-NEXT: 5.0
    # CHECK-NEXT: 4.0

    # print(np.arange(0, 9).reshape(3,3)[4:0:-1, 4:0:-1])
    test_slice[.float32, 9, 2, IndexList[2](3, 3)](
        IndexList[2](3, 3),
        Index(4, 4),
        Index(0, 0),
        Index(-1, -1),
        False,
    )


# CHECK-LABEL: == test_last_dim_edge_4
def test_last_dim_edge_4() raises:
    print("== test_last_dim_edge_4")

    # CHECK: In shape: (3, 3)
    # CHECK-NEXT: In strides: (3, 1)
    # CHECK-NEXT: New shape: (3, 3)
    # CHECK-NEXT: New strides: (-3, -1)

    # CHECK-NEXT: 8.0
    # CHECK-NEXT: 7.0
    # CHECK-NEXT: 6.0
    # CHECK-NEXT: 5.0
    # CHECK-NEXT: 4.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 2.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 0.0

    # print(np.arange(0, 9).reshape(3,3)[4:-4:-1, 4:-4:-1])
    test_slice[.float32, 9, 2, IndexList[2](3, 3)](
        IndexList[2](3, 3),
        Index(4, 4),
        Index(-4, -4),
        Index(-1, -1),
        False,
    )


# CHECK-LABEL: == test_out_of_bounds
def test_out_of_bounds() raises:
    print("== test_out_of_bounds")

    # CHECK: In shape: (3, 3)
    # CHECK-NEXT: In strides: (3, 1)
    # CHECK-NEXT: New shape: (2, 2)
    # CHECK-NEXT: New strides: (3, 1)

    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 4.0
    # CHECK-NEXT: 6.0
    # CHECK-NEXT: 7.0

    # print(np.arange(0, 9).reshape(3, 3)[1:5:1, -5:-1:1])
    test_slice[.float32, 9, 2, IndexList[2](3, 3)](
        IndexList[2](3, 3),
        Index(1, -5),
        Index(5, -1),
        Index(1, 1),
        False,
    )


def main() raises:
    test_slice_basic()
    test_slice_identity()
    test_slice_steps()
    test_slice_1D()
    test_slice_empty()
    test_slice_4D()
    test_slice_copy()
    test_slice_negative()

    test_slice_negative_step_1D()
    test_slice_negative_step_2D()
    test_slice_negative_step_3D()
    test_slice_negative_step_4D()

    test_slice_negative_step_2()
    test_slice_negative_step_3()
    test_slice_negative_step_4()

    test_truncated_last_dim()
    test_truncated_first_and_last_dim()
    test_truncated_last_dim_reverse()
    test_truncated_first_and_last_dim_reverse()
    test_last_dim_edge()
    test_last_dim_edge_2()
    test_last_dim_edge_3()
    test_last_dim_edge_4()

    test_out_of_bounds()
