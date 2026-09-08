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

from layout import Coord, TileTensor, row_major
from nn.conv.conv_transpose import conv_transpose_naive

from std.utils.index import Index, IndexList


# CHECK-LABEL: test_convtranspose_pads
# CHECK: 1.0 ,1.0 ,3.0 ,
# CHECK: 1.0 ,1.0 ,3.0 ,
# CHECK: 7.0 ,4.0 ,9.0 ,
# CHECK: 7.0 ,4.0 ,9.0 ,
# CHECK: 7.0 ,4.0 ,9.0 ,
# CHECK: 13.0 ,7.0 ,15.0 ,
# CHECK: 13.0 ,7.0 ,15.0 ,
# CHECK: 1.0 ,1.0 ,3.0 ,
# CHECK: 1.0 ,1.0 ,3.0 ,
# CHECK: 7.0 ,4.0 ,9.0 ,
# CHECK: 7.0 ,4.0 ,9.0 ,
# CHECK: 7.0 ,4.0 ,9.0 ,
# CHECK: 13.0 ,7.0 ,15.0 ,
# CHECK: 13.0 ,7.0 ,15.0 ,
def test_convtranspose_pads():
    print("== test_convtranspose_pads")
    comptime type = DType.float32

    comptime input_layout = row_major[1, 1, 3, 3, 1]()
    var input_stack = Array[Scalar[type], input_layout.product()](
        fill_with=lambda (i: Int) -> Scalar[type]: Float32(i)
    )
    var input_stack_ptr: MutPointer[
        Scalar[type], origin_of(input_stack)
    ] = input_stack.unsafe_ptr()
    var input = TileTensor(input_stack_ptr, input_layout)

    comptime filter_layout = row_major[1, 3, 3, 2, 1]()
    var filter_stack = Array[Scalar[type], filter_layout.product()](fill=1.0)
    var filter_stack_ptr: MutPointer[
        Scalar[type], origin_of(filter_stack)
    ] = filter_stack.unsafe_ptr()
    var filter = TileTensor(filter_stack_ptr, filter_layout)

    comptime output_layout = row_major[1, 1, 7, 3, 2]()
    var output_stack = Array[Scalar[type], output_layout.product()](fill={})
    var output_stack_ptr: MutPointer[
        Scalar[type], origin_of(output_stack)
    ] = output_stack.unsafe_ptr()
    var output = TileTensor(output_stack_ptr, output_layout)

    var stride = Index(1, 3, 2)
    var dilation = Index(1, 1, 1)
    var pad_d = Index(0, 0)
    var pad_h = Index(1, 1)
    var pad_w = Index(2, 2)

    conv_transpose_naive[type](
        TileTensor(
            output._storage,
            row_major(Coord(IndexList[5](1, 1, 7, 3, 2))),
        ),
        TileTensor(
            input._storage,
            row_major(Coord(IndexList[5](1, 1, 3, 3, 1))),
        ),
        TileTensor(
            filter._storage,
            row_major(Coord(IndexList[5](1, 3, 3, 2, 1))),
        ),
        stride,
        dilation,
        pad_d,
        pad_h,
        pad_w,
    )

    print()
    for k in range(2):
        for i in range(7):
            for j in range(3):
                print(output[0, 0, i, j, k], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_convtranspose
# CHECK: 0.0 ,1.0 ,3.0 ,3.0 ,2.0 ,
# CHECK: 3.0 ,8.0 ,15.0 ,12.0 ,7.0 ,
# CHECK: 9.0 ,21.0 ,36.0 ,27.0 ,15.0 ,
# CHECK: 9.0 ,20.0 ,33.0 ,24.0 ,13.0 ,
# CHECK: 6.0 ,13.0 ,21.0 ,15.0 ,8.0 ,
# CHECK: 0.0 ,1.0 ,3.0 ,3.0 ,2.0 ,
# CHECK: 3.0 ,8.0 ,15.0 ,12.0 ,7.0 ,
# CHECK: 9.0 ,21.0 ,36.0 ,27.0 ,15.0 ,
# CHECK: 9.0 ,20.0 ,33.0 ,24.0 ,13.0 ,
# CHECK: 6.0 ,13.0 ,21.0 ,15.0 ,8.0 ,
def test_convtranspose():
    print("== test_convtranspose")
    comptime type = DType.float32

    comptime input_layout = row_major[1, 1, 3, 3, 1]()
    var input_stack = Array[Scalar[type], input_layout.product()](
        fill_with=lambda (i: Int) -> Scalar[type]: Float32(i)
    )
    var input_stack_ptr: MutPointer[
        Scalar[type], origin_of(input_stack)
    ] = input_stack.unsafe_ptr()
    var input = TileTensor(input_stack_ptr, input_layout)

    comptime filter_layout = row_major[1, 3, 3, 2, 1]()
    var filter_stack = Array[Scalar[type], filter_layout.product()](fill=1.0)
    var filter_stack_ptr: MutPointer[
        Scalar[type], origin_of(filter_stack)
    ] = filter_stack.unsafe_ptr()
    var filter = TileTensor(filter_stack_ptr, filter_layout)

    comptime output_layout = row_major[1, 1, 5, 5, 2]()
    var output_stack = Array[Scalar[type], output_layout.product()](fill={})
    var output_stack_ptr: MutPointer[
        Scalar[type], origin_of(output_stack)
    ] = output_stack.unsafe_ptr()
    var output = TileTensor(output_stack_ptr, output_layout)

    var stride = Index(1, 1, 1)
    var dilation = Index(1, 1, 1)
    var pad_d = Index(0, 0)
    var pad_h = Index(0, 0)
    var pad_w = Index(0, 0)

    conv_transpose_naive[type](
        TileTensor(
            output._storage,
            row_major(Coord(IndexList[5](1, 1, 5, 5, 2))),
        ),
        TileTensor(
            input._storage,
            row_major(Coord(IndexList[5](1, 1, 3, 3, 1))),
        ),
        TileTensor(
            filter._storage,
            row_major(Coord(IndexList[5](1, 3, 3, 2, 1))),
        ),
        stride,
        dilation,
        pad_d,
        pad_h,
        pad_w,
    )

    print()
    for l in range(2):
        for j in range(5):
            for k in range(5):
                print(output[0, 0, j, k, l], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_convtranspose_dilation
# CHECK: 21.0 ,56.0 ,13.0 ,16.0 ,2.0 ,
# CHECK: 63.0 ,35.0 ,67.0 ,10.0 ,14.0 ,
# CHECK: 24.0 ,22.0 ,76.0 ,76.0 ,21.0 ,
# CHECK: 9.0 ,5.0 ,88.0 ,45.0 ,63.0 ,
# CHECK: 3.0 ,2.0 ,33.0 ,18.0 ,54.0 ,
def test_convtranspose_dilation():
    print("== test_convtranspose_dilation")
    comptime type = DType.float32

    comptime input_layout = row_major[1, 1, 3, 3, 1]()
    var input_stack = Array[Scalar[type], input_layout.product()](fill={})
    var input_stack_ptr: MutPointer[
        Scalar[type], origin_of(input_stack)
    ] = input_stack.unsafe_ptr()
    var input = TileTensor(input_stack_ptr, input_layout)
    input.raw_store(0, 3)
    input.raw_store(1, 8)
    input.raw_store(2, 1)
    input.raw_store(3, 9)
    input.raw_store(4, 5)
    input.raw_store(5, 7)
    input.raw_store(6, 3)
    input.raw_store(7, 2)
    input.raw_store(8, 6)

    comptime filter_layout = row_major[1, 2, 2, 1, 1]()
    var filter_stack = Array[Scalar[type], filter_layout.product()](fill={})
    var filter_stack_ptr: MutPointer[
        Scalar[type], origin_of(filter_stack)
    ] = filter_stack.unsafe_ptr()
    var filter = TileTensor(filter_stack_ptr, filter_layout)
    filter.raw_store(0, 7)
    filter.raw_store(1, 2)
    filter.raw_store(2, 1)
    filter.raw_store(3, 9)

    comptime output_layout = row_major[1, 1, 5, 5, 1]()
    var output_stack = Array[Scalar[type], output_layout.product()](fill={})
    var output_stack_ptr: MutPointer[
        Scalar[type], origin_of(output_stack)
    ] = output_stack.unsafe_ptr()
    var output = TileTensor(output_stack_ptr, output_layout)
    var stride = Index(1, 1, 1)
    var dilation = Index(1, 2, 2)
    var pad_d = Index(0, 0)
    var pad_h = Index(0, 0)
    var pad_w = Index(0, 0)

    conv_transpose_naive[type](
        TileTensor(
            output._storage,
            row_major(Coord(IndexList[5](1, 1, 5, 5, 1))),
        ),
        TileTensor(
            input._storage,
            row_major(Coord(IndexList[5](1, 1, 3, 3, 1))),
        ),
        TileTensor(
            filter._storage,
            row_major(Coord(IndexList[5](1, 2, 2, 1, 1))),
        ),
        stride,
        dilation,
        pad_d,
        pad_h,
        pad_w,
    )

    print()
    for l in range(1):
        for j in range(5):
            for k in range(5):
                print(output[0, 0, j, k, l], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_convtranspose_attributes
# CHECK: 0.0 ,0.0 ,1.0 ,1.0 ,3.0 ,2.0 ,2.0 ,0.0 ,
# CHECK: 0.0 ,0.0 ,1.0 ,1.0 ,3.0 ,2.0 ,2.0 ,0.0 ,
# CHECK: 0.0 ,0.0 ,1.0 ,1.0 ,3.0 ,2.0 ,2.0 ,0.0 ,
# CHECK: 3.0 ,3.0 ,7.0 ,4.0 ,9.0 ,5.0 ,5.0 ,0.0 ,
# CHECK: 3.0 ,3.0 ,7.0 ,4.0 ,9.0 ,5.0 ,5.0 ,0.0 ,
# CHECK: 3.0 ,3.0 ,7.0 ,4.0 ,9.0 ,5.0 ,5.0 ,0.0 ,
# CHECK: 6.0 ,6.0 ,13.0 ,7.0 ,15.0 ,8.0 ,8.0 ,0.0 ,
# CHECK: 6.0 ,6.0 ,13.0 ,7.0 ,15.0 ,8.0 ,8.0 ,0.0 ,
# CHECK: 6.0 ,6.0 ,13.0 ,7.0 ,15.0 ,8.0 ,8.0 ,0.0 ,
# CHECK: 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,
# CHECK: 0.0 ,0.0 ,1.0 ,1.0 ,3.0 ,2.0 ,2.0 ,0.0 ,
# CHECK: 0.0 ,0.0 ,1.0 ,1.0 ,3.0 ,2.0 ,2.0 ,0.0 ,
# CHECK: 0.0 ,0.0 ,1.0 ,1.0 ,3.0 ,2.0 ,2.0 ,0.0 ,
# CHECK: 3.0 ,3.0 ,7.0 ,4.0 ,9.0 ,5.0 ,5.0 ,0.0 ,
# CHECK: 3.0 ,3.0 ,7.0 ,4.0 ,9.0 ,5.0 ,5.0 ,0.0 ,
# CHECK: 3.0 ,3.0 ,7.0 ,4.0 ,9.0 ,5.0 ,5.0 ,0.0 ,
# CHECK: 6.0 ,6.0 ,13.0 ,7.0 ,15.0 ,8.0 ,8.0 ,0.0 ,
# CHECK: 6.0 ,6.0 ,13.0 ,7.0 ,15.0 ,8.0 ,8.0 ,0.0 ,
# CHECK: 6.0 ,6.0 ,13.0 ,7.0 ,15.0 ,8.0 ,8.0 ,0.0 ,
# CHECK: 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,
def test_convtranspose_attributes():
    print("== test_convtranspose_attributes")
    comptime type = DType.float32

    comptime input_layout = row_major[1, 1, 3, 3, 1]()
    var input_stack = Array[Scalar[type], input_layout.product()](
        fill_with=lambda (i: Int) -> Scalar[type]: Float32(i)
    )
    var input_stack_ptr: MutPointer[
        Scalar[type], origin_of(input_stack)
    ] = input_stack.unsafe_ptr()
    var input = TileTensor(input_stack_ptr, input_layout)

    comptime filter_layout = row_major[1, 3, 3, 2, 1]()
    var filter_stack = Array[Scalar[type], filter_layout.product()](fill=1.0)
    var filter_stack_ptr: MutPointer[
        Scalar[type], origin_of(filter_stack)
    ] = filter_stack.unsafe_ptr()
    var filter = TileTensor(filter_stack_ptr, filter_layout)

    comptime output_layout = row_major[1, 1, 10, 8, 2]()
    var output_stack = Array[Scalar[type], output_layout.product()](fill={})
    var output_stack_ptr: MutPointer[
        Scalar[type], origin_of(output_stack)
    ] = output_stack.unsafe_ptr()
    var output = TileTensor(output_stack_ptr, output_layout)

    var stride = Index(1, 3, 2)
    var dilation = Index(1, 1, 1)
    var pad_d = Index(0, 0)
    var pad_h = Index(0, 0)
    var pad_w = Index(0, 0)

    conv_transpose_naive[type](
        TileTensor(
            output._storage,
            row_major(Coord(IndexList[5](1, 1, 10, 8, 2))),
        ),
        TileTensor(
            input._storage,
            row_major(Coord(IndexList[5](1, 1, 3, 3, 1))),
        ),
        TileTensor(
            filter._storage,
            row_major(Coord(IndexList[5](1, 3, 3, 2, 1))),
        ),
        stride,
        dilation,
        pad_d,
        pad_h,
        pad_w,
    )

    print()
    for l in range(2):
        for j in range(10):
            for k in range(8):
                print(output[0, 0, j, k, l], ",", end="")
            print()
        print()
    print()


def main():
    test_convtranspose_pads()
    test_convtranspose()
    test_convtranspose_dilation()
    test_convtranspose_attributes()
