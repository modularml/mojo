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

from std.sys import align_of

from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.element import Element

from std.utils import IndexList


# CHECK-LABEL: test_element_load
def test_element_load():
    print("== test_element_load")
    var tensor_8x8 = LayoutTensor[
        .float32, Layout.row_major(8, 8), MutAnyOrigin
    ].stack_allocation[stack_alignment=align_of[SIMD[.float32, 4]]()]()
    arange(tensor_8x8)

    # CHECK: vector_1x4
    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    print("vector_1x4")
    for i in range(8):
        for j in range(2):
            var tensor_8x8_v_1_4 = tensor_8x8.as_imm().vectorize[1, 4]()
            var offset = materialize[tensor_8x8_v_1_4.layout]()(IntTuple(i, j))
            var elem = Element[
                tensor_8x8_v_1_4.dtype, tensor_8x8_v_1_4.element_layout
            ].load(tensor_8x8_v_1_4.ptr + offset)
            print(elem, end=" ")
        print("")

    # CHECK: vector_4x1
    # CHECK: [0.0, 8.0, 16.0, 24.0] [1.0, 9.0, 17.0, 25.0] [2.0, 10.0, 18.0, 26.0] [3.0, 11.0, 19.0, 27.0] [4.0, 12.0, 20.0, 28.0] [5.0, 13.0, 21.0, 29.0] [6.0, 14.0, 22.0, 30.0] [7.0, 15.0, 23.0, 31.0]
    # CHECK: [32.0, 40.0, 48.0, 56.0] [33.0, 41.0, 49.0, 57.0] [34.0, 42.0, 50.0, 58.0] [35.0, 43.0, 51.0, 59.0] [36.0, 44.0, 52.0, 60.0] [37.0, 45.0, 53.0, 61.0] [38.0, 46.0, 54.0, 62.0] [39.0, 47.0, 55.0, 63.0]
    print("vector_4x1")
    for i in range(2):
        for j in range(8):
            var tensor_8x8_v_4_1 = tensor_8x8.as_imm().vectorize[4, 1]()
            var offset = materialize[tensor_8x8_v_4_1.layout]()(IntTuple(i, j))
            var elem = Element[
                tensor_8x8_v_4_1.dtype, tensor_8x8_v_4_1.element_layout
            ].load(tensor_8x8_v_4_1.ptr + offset)
            print(elem, end=" ")
        print("")

    # CHECK: vector_4x4
    # CHECK: [0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0, 16.0, 17.0, 18.0, 19.0, 24.0, 25.0, 26.0, 27.0] [4.0, 5.0, 6.0, 7.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0, 22.0, 23.0, 28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0, 40.0, 41.0, 42.0, 43.0, 48.0, 49.0, 50.0, 51.0, 56.0, 57.0, 58.0, 59.0] [36.0, 37.0, 38.0, 39.0, 44.0, 45.0, 46.0, 47.0, 52.0, 53.0, 54.0, 55.0, 60.0, 61.0, 62.0, 63.0]
    print("vector_4x4")
    for i in range(2):
        for j in range(2):
            var tensor_8x8_v_4_4 = tensor_8x8.as_imm().vectorize[4, 4]()
            var offset = materialize[tensor_8x8_v_4_4.layout]()(IntTuple(i, j))
            var elem = Element[
                tensor_8x8_v_4_4.dtype, tensor_8x8_v_4_4.element_layout
            ].load(tensor_8x8_v_4_4.ptr + offset)
            print(elem, end=" ")
        print("")


# CHECK-LABEL: test_element_store
def test_element_store():
    print("== test_element_store")
    var tensor_8x8 = LayoutTensor[
        .float32, Layout.row_major(8, 8), MutAnyOrigin
    ].stack_allocation[stack_alignment=align_of[SIMD[.float32, 4]]()]()
    arange(tensor_8x8)

    # CHECK: vector_1x4
    # CHECK: 0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0
    # CHECK: 80.0 90.0 100.0 110.0 120.0 130.0 140.0 150.0
    # CHECK: 160.0 170.0 180.0 190.0 200.0 210.0 220.0 230.0
    # CHECK: 240.0 250.0 260.0 270.0 280.0 290.0 300.0 310.0
    # CHECK: 320.0 330.0 340.0 350.0 360.0 370.0 380.0 390.0
    # CHECK: 400.0 410.0 420.0 430.0 440.0 450.0 460.0 470.0
    # CHECK: 480.0 490.0 500.0 510.0 520.0 530.0 540.0 550.0
    # CHECK: 560.0 570.0 580.0 590.0 600.0 610.0 620.0 630.0
    print("vector_1x4")
    for i in range(8):
        for j in range(2):
            var tensor_8x8_v_1_4 = tensor_8x8.vectorize[1, 4]()
            var offset = materialize[tensor_8x8_v_1_4.layout]()(IntTuple(i, j))
            var elem = Element[
                tensor_8x8_v_1_4.dtype, tensor_8x8_v_1_4.element_layout
            ].load(tensor_8x8_v_1_4.ptr + offset)
            elem.element_data *= 10
            elem.store(tensor_8x8_v_1_4.ptr + offset)
    print(tensor_8x8)

    # CHECK: vector_4x1
    # CHECK: 0.0 100.0 200.0 300.0 400.0 500.0 600.0 700.0
    # CHECK: 800.0 900.0 1000.0 1100.0 1200.0 1300.0 1400.0 1500.0
    # CHECK: 1600.0 1700.0 1800.0 1900.0 2000.0 2100.0 2200.0 2300.0
    # CHECK: 2400.0 2500.0 2600.0 2700.0 2800.0 2900.0 3000.0 3100.0
    # CHECK: 3200.0 3300.0 3400.0 3500.0 3600.0 3700.0 3800.0 3900.0
    # CHECK: 4000.0 4100.0 4200.0 4300.0 4400.0 4500.0 4600.0 4700.0
    # CHECK: 4800.0 4900.0 5000.0 5100.0 5200.0 5300.0 5400.0 5500.0
    # CHECK: 5600.0 5700.0 5800.0 5900.0 6000.0 6100.0 6200.0 6300.0
    print("vector_4x1")
    for i in range(2):
        for j in range(8):
            var tensor_8x8_v_4_1 = tensor_8x8.vectorize[4, 1]()
            var offset = materialize[tensor_8x8_v_4_1.layout]()(IntTuple(i, j))
            var elem = Element[
                tensor_8x8_v_4_1.dtype, tensor_8x8_v_4_1.element_layout
            ].load(tensor_8x8_v_4_1.ptr + offset)
            elem.element_data *= 10
            elem.store(tensor_8x8_v_4_1.ptr + offset)
    print(tensor_8x8)

    # CHECK: vector_4x4
    # CHECK: 0.0 1000.0 2000.0 3000.0 4000.0 5000.0 6000.0 7000.0
    # CHECK: 8000.0 9000.0 10000.0 11000.0 12000.0 13000.0 14000.0 15000.0
    # CHECK: 16000.0 17000.0 18000.0 19000.0 20000.0 21000.0 22000.0 23000.0
    # CHECK: 24000.0 25000.0 26000.0 27000.0 28000.0 29000.0 30000.0 31000.0
    # CHECK: 32000.0 33000.0 34000.0 35000.0 36000.0 37000.0 38000.0 39000.0
    # CHECK: 40000.0 41000.0 42000.0 43000.0 44000.0 45000.0 46000.0 47000.0
    # CHECK: 48000.0 49000.0 50000.0 51000.0 52000.0 53000.0 54000.0 55000.0
    # CHECK: 56000.0 57000.0 58000.0 59000.0 60000.0 61000.0 62000.0 63000.0
    print("vector_4x4")
    for i in range(2):
        for j in range(2):
            var tensor_8x8_v_4_4 = tensor_8x8.vectorize[4, 4]()
            var offset = materialize[tensor_8x8_v_4_4.layout]()(IntTuple(i, j))
            var elem = Element[
                tensor_8x8_v_4_4.dtype, tensor_8x8_v_4_4.element_layout
            ].load(tensor_8x8_v_4_4.ptr + offset)
            elem.element_data *= 10
            elem.store(tensor_8x8_v_4_4.ptr + offset)

    print(tensor_8x8)


def test_element_dynamic_layout() raises:
    print("== test_element_dynamic_layout")

    comptime layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var dynamic_layout = RuntimeLayout[
        layout, element_type=.int32, linear_idx_type=.int32
    ](
        RuntimeTuple[layout.shape, element_type=.int32](8, 8),
        RuntimeTuple[layout.stride, element_type=.int32](8, 1),
    )

    var storage = List(length=dynamic_layout.size(), fill=Float32(0))

    var tensor_8x8 = LayoutTensor[
        .float32,
        layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
    ](storage, dynamic_layout)

    arange(tensor_8x8)

    for tile_i in range(2):
        for tile_j in range(2):
            var tensor_8x8_v_4_4 = tensor_8x8.vectorize[4, 4]()
            var offset = tensor_8x8_v_4_4.runtime_layout(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE)](
                    tile_i, tile_j
                )
            )
            var elem = Element[
                tensor_8x8_v_4_4.dtype,
                tensor_8x8_v_4_4.element_layout,
                index_type=tensor_8x8_v_4_4.linear_idx_type,
            ].load(
                tensor_8x8_v_4_4.ptr + offset,
                tensor_8x8_v_4_4.runtime_element_layout,
            )
            elem.element_data *= 10
            elem.store(tensor_8x8_v_4_4.ptr + offset)

    # CHECK: 0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0
    # CHECK: 80.0 90.0 100.0 110.0 120.0 130.0 140.0 150.0
    # CHECK: 160.0 170.0 180.0 190.0 200.0 210.0 220.0 230.0
    # CHECK: 240.0 250.0 260.0 270.0 280.0 290.0 300.0 310.0
    # CHECK: 320.0 330.0 340.0 350.0 360.0 370.0 380.0 390.0
    # CHECK: 400.0 410.0 420.0 430.0 440.0 450.0 460.0 470.0
    # CHECK: 480.0 490.0 500.0 510.0 520.0 530.0 540.0 550.0
    # CHECK: 560.0 570.0 580.0 590.0 600.0 610.0 620.0 630.0
    print(tensor_8x8)

    comptime layoutUx8 = Layout.row_major(UNKNOWN_VALUE, 8)
    comptime tensor_Ux8_type = ManagedLayoutTensor[.float32, layoutUx8]
    var runtime_layoutUx8 = RuntimeLayout[
        layoutUx8,
        element_type=tensor_Ux8_type.element_type,
        linear_idx_type=tensor_Ux8_type.index_type,
    ].row_major(IndexList[2, element_type=tensor_Ux8_type.element_type](8, 8))

    var tensor_Ux8 = tensor_Ux8_type(runtime_layoutUx8)
    arange(tensor_Ux8.tensor(), 0, 0.5)
    # CHECK: 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5
    # CHECK: 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5
    # CHECK: 8.0 8.5 9.0 9.5 10.0 10.5 11.0 11.5
    # CHECK: 12.0 12.5 13.0 13.5 14.0 14.5 15.0 15.5
    # CHECK: 16.0 16.5 17.0 17.5 18.0 18.5 19.0 19.5
    # CHECK: 20.0 20.5 21.0 21.5 22.0 22.5 23.0 23.5
    # CHECK: 24.0 24.5 25.0 25.5 26.0 26.5 27.0 27.5
    # CHECK: 28.0 28.5 29.0 29.5 30.0 30.5 31.0 31.5
    print(tensor_Ux8.tensor())

    var tensor_Ux8_vec4_d1 = tensor_Ux8.tensor().vectorize[1, 4]()

    # CHECK: ((1, 4):(0, 1))
    # CHECK: [0.0, 0.5, 1.0, 1.5] [2.0, 2.5, 3.0, 3.5]
    # CHECK: [4.0, 4.5, 5.0, 5.5] [6.0, 6.5, 7.0, 7.5]
    # CHECK: [8.0, 8.5, 9.0, 9.5] [10.0, 10.5, 11.0, 11.5]
    # CHECK: [12.0, 12.5, 13.0, 13.5] [14.0, 14.5, 15.0, 15.5]
    # CHECK: [16.0, 16.5, 17.0, 17.5] [18.0, 18.5, 19.0, 19.5]
    # CHECK: [20.0, 20.5, 21.0, 21.5] [22.0, 22.5, 23.0, 23.5]
    # CHECK: [24.0, 24.5, 25.0, 25.5] [26.0, 26.5, 27.0, 27.5]
    # CHECK: [28.0, 28.5, 29.0, 29.5] [30.0, 30.5, 31.0, 31.5]
    print(materialize[tensor_Ux8_vec4_d1.element_layout]())
    print(tensor_Ux8_vec4_d1)

    comptime layout8xU = Layout.row_major(8, UNKNOWN_VALUE)
    comptime tensor_8xU_type = ManagedLayoutTensor[.float32, layout8xU]
    var runtime_layout8xU = RuntimeLayout[
        layout8xU,
        element_type=tensor_8xU_type.element_type,
        linear_idx_type=tensor_8xU_type.index_type,
    ].row_major(IndexList[2, element_type=tensor_8xU_type.element_type](8, 2))

    var tensor_8xU = tensor_8xU_type(runtime_layout8xU)
    arange(tensor_8xU.tensor(), 0, 0.5)
    # CHECK: 0.0 0.5
    # CHECK: 1.0 1.5
    # CHECK: 2.0 2.5
    # CHECK: 3.0 3.5
    # CHECK: 4.0 4.5
    # CHECK: 5.0 5.5
    # CHECK: 6.0 6.5
    # CHECK: 7.0 7.5
    print(tensor_8xU.tensor())

    var tensor_Ux8_vec4_d0 = tensor_8xU.tensor().vectorize[4, 1]()
    # CHECK: ((4, 1):(-1, 0))
    # CHECK: [0.0, 1.0, 2.0, 3.0] [0.5, 1.5, 2.5, 3.5]
    # CHECK: [4.0, 5.0, 6.0, 7.0] [4.5, 5.5, 6.5, 7.5]
    print(materialize[tensor_Ux8_vec4_d0.element_layout]())
    print(tensor_Ux8_vec4_d0)

    _ = tensor_Ux8^
    _ = tensor_8xU^


# CHECK-LABEL: test_element_masked_load
def test_element_masked_load():
    print("== test_element_masked_load")
    var tensor_4x4_stack = Array[Float32, 4 * 4](fill={})
    var tensor_4x4 = LayoutTensor[.float32, Layout.row_major(4, 4)](
        tensor_4x4_stack
    )
    arange(tensor_4x4)
    var tensor_1x3 = LayoutTensor[.float32, Layout.row_major(1, 3)](
        tensor_4x4.ptr
    )

    var tensor_1x3_v4 = tensor_1x3.as_imm().vectorize[1, 4]()
    # CHECK: [0.0, 1.0, 2.0, 0.0]
    print(
        Element[
            tensor_1x3_v4.dtype,
            tensor_1x3_v4.element_layout,
            index_type=tensor_1x3_v4.linear_idx_type,
        ].masked_load(
            tensor_1x3_v4.ptr,
            type_of(tensor_1x3_v4.runtime_element_layout).row_major(
                IndexList[2, element_type=.int32](1, 3)
            ),
        )
    )

    # CHECK: [0.0, 4.0, 8.0, 0.0]
    var tensor_3x4 = LayoutTensor[.float32, Layout.row_major(3, 4)](
        tensor_4x4.ptr
    )

    var tensor_3x1_v4 = tensor_3x4.as_imm().vectorize[4, 1]()

    print(
        Element[index_type=tensor_3x1_v4.linear_idx_type].masked_load(
            tensor_3x1_v4.ptr,
            type_of(tensor_3x1_v4.runtime_element_layout).row_major(
                IndexList[2, element_type=.int32](3, 1)
            ),
        )
    )

    var tensor_3x4_v4x4 = tensor_3x4.as_imm().vectorize[4, 4]()

    # CHECK: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 0.0, 0.0, 0.0, 0.0]
    print(
        Element[index_type=tensor_3x4_v4x4.linear_idx_type].masked_load(
            tensor_3x4_v4x4.ptr,
            type_of(tensor_3x4_v4x4.runtime_element_layout).row_major(
                IndexList[2, element_type=.int32](3, 4)
            ),
        )
    )


# CHECK-LABEL: test_element_masked_store
def test_element_masked_store():
    print("== test_element_masked_store")
    var tensor_4x4_stack = Array[Float32, 4 * 4](fill={})
    var tensor_4x4 = LayoutTensor[.float32, Layout.row_major(4, 4)](
        tensor_4x4_stack
    ).fill(-1)

    var tensor_4x4_vec_1_4 = tensor_4x4.vectorize[1, 4]()
    var element_v_1_4 = Element[index_type=tensor_4x4_vec_1_4.linear_idx_type](
        SIMD[
            tensor_4x4_vec_1_4.dtype, tensor_4x4_vec_1_4.element_layout.size()
        ](1),
        type_of(tensor_4x4_vec_1_4.runtime_element_layout).row_major(
            IndexList[2, element_type=.int32](1, 3)
        ),
    )
    element_v_1_4.masked_store(tensor_4x4_vec_1_4.ptr)
    # CHECK: vec_1x4:mask_1x3
    # CHECK: 1.0 1.0 1.0 -1.0
    # CHECK: -1.0 -1.0 -1.0 -1.0
    # CHECK: -1.0 -1.0 -1.0 -1.0
    # CHECK: -1.0 -1.0 -1.0 -1.0
    print("vec_1x4:mask_1x3")
    print(tensor_4x4)
    _ = tensor_4x4.fill(-1)

    var tensor_4x4_vec_4_1 = tensor_4x4.vectorize[4, 1]()
    var element_v_4_1 = Element[index_type=tensor_4x4_vec_4_1.linear_idx_type](
        SIMD[
            tensor_4x4_vec_4_1.dtype, tensor_4x4_vec_4_1.element_layout.size()
        ](1),
        type_of(tensor_4x4_vec_4_1.runtime_element_layout).row_major(
            IndexList[2, element_type=.int32](2, 1)
        ),
    )
    element_v_4_1.masked_store(tensor_4x4_vec_4_1.ptr)
    print("vec_4x1:mask_1x2")
    # CHECK: vec_4x1:mask_1x2
    # CHECK: 1.0 -1.0 -1.0 -1.0
    # CHECK: 1.0 -1.0 -1.0 -1.0
    # CHECK: -1.0 -1.0 -1.0 -1.0
    # CHECK: -1.0 -1.0 -1.0 -1.0
    print(tensor_4x4)
    _ = tensor_4x4.fill(-1)

    var tensor_4x4_vec_4_4 = tensor_4x4.vectorize[4, 4]()
    var element_v_4_4 = Element[index_type=tensor_4x4.linear_idx_type](
        SIMD[
            tensor_4x4_vec_4_4.dtype, tensor_4x4_vec_4_4.element_layout.size()
        ](1),
        type_of(tensor_4x4_vec_4_4.runtime_element_layout).row_major(
            IndexList[2, element_type=.int32](3, 2)
        ),
    )
    element_v_4_4.masked_store(tensor_4x4_vec_4_4.ptr)
    print("vec_4x4:mask_3x2")
    # CHECK: vec_4x4:mask_3x2
    # CHECK: 1.0 1.0 -1.0 -1.0
    # CHECK: 1.0 1.0 -1.0 -1.0
    # CHECK: 1.0 1.0 -1.0 -1.0
    # CHECK: -1.0 -1.0 -1.0 -1.0
    print(tensor_4x4)


def main() raises:
    test_element_load()
    test_element_store()
    test_element_dynamic_layout()
    test_element_masked_load()
    test_element_masked_store()
