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


from layout import Coord, Idx, TensorLayout, TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.concat import _concat_parallel, _concat_serial, concat

from std.utils import IndexList, StaticTuple


def _tuple_to_list[
    LayoutType: TensorLayout,
    //,
    dtype: DType,
](
    elems: StaticTuple[
        TileTensor[dtype, LayoutType, ImmutAnyOrigin],
        ...,
    ]
) -> List[TileTensor[dtype, LayoutType, ImmutAnyOrigin]]:
    var output = List[TileTensor[dtype, LayoutType, ImmutAnyOrigin]](
        capacity=len(elems)
    )
    for i in range(len(elems)):
        output.append(elems[i].as_immut())
    return output^


def test_concat() raises:
    print("== test_concat")

    comptime dtype = DType.float32
    comptime rank = 4
    comptime concat_axis = 2

    comptime l1 = row_major[2, 2, 1, 2]()
    comptime l2 = row_major[2, 2, 2, 2]()
    comptime l3 = row_major[2, 2, 3, 2]()
    var x1_stack = Array[Scalar[dtype], l1.product()](fill=0)
    var x2_stack = Array[Scalar[dtype], l2.product()](fill=1)
    var x3_stack = Array[Scalar[dtype], l3.product()](fill=2)
    var x1 = TileTensor(x1_stack, l1)
    var x2 = TileTensor(x2_stack, l2)
    var x3 = TileTensor(x3_stack, l3)

    comptime out_layout = row_major[2, 2, 6, 2]()
    var out_stack = Array[Scalar[dtype], out_layout.product()](fill=-1)
    var output = TileTensor(out_stack, out_layout)
    var x1_dyn = x1.make_dynamic[.int64]()

    var input_tuple = StaticTuple[
        TileTensor[dtype, x1_dyn.LayoutType, ImmutAnyOrigin],
        3,
    ](
        x1_dyn.as_unsafe_any_origin().as_immut(),
        x2.make_dynamic[.int64]().as_unsafe_any_origin().as_immut(),
        x3.make_dynamic[.int64]().as_unsafe_any_origin().as_immut(),
    )

    @__parameter
    @always_inline
    def epilogue_plus_one[
        c_type: DType, _rank: Int, width: SIMDLength, *, alignment: Int
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        var coord = Coord(indices)
        comptime assert output.flat_rank >= coord.flat_rank
        output.store[width=width](
            coord,
            rebind[SIMD[dtype, width]](val + 1),
        )

    concat[dtype, epilogue_fn=epilogue_plus_one](
        output.make_dynamic[.int64](),
        concat_axis,
        input_tuple,
        DeviceContext(api="cpu"),
    )

    # CHECK: == test_concat
    # CHECK-COUNT-2: 1.0
    # CHECK-COUNT-4: 2.0
    # CHECK-COUNT-6: 3.0
    # CHECK-COUNT-2: 1.0
    # CHECK-COUNT-4: 2.0
    # CHECK-COUNT-6: 3.0
    # CHECK-COUNT-2: 1.0
    # CHECK-COUNT-4: 2.0
    # CHECK-COUNT-6: 3.0
    var output_flat = TileTensor(
        output._storage,
        row_major(Coord(output.num_elements())),
    )
    for i in range(output.layout.product()):
        print(output_flat.load[1]((i,)))


def test_concat_parallel() raises:
    print("== test_concat_parallel")

    comptime dtype = DType.float32
    comptime rank = 4
    comptime concat_axis = 2

    comptime l1 = row_major[2, 2, 1, 2]()
    comptime l2 = row_major[2, 2, 2, 2]()
    comptime l3 = row_major[2, 2, 3, 2]()
    var x1_stack = Array[Scalar[dtype], l1.product()](fill=0)
    var x2_stack = Array[Scalar[dtype], l2.product()](fill=1)
    var x3_stack = Array[Scalar[dtype], l3.product()](fill=2)
    var x1 = TileTensor(x1_stack, l1)
    var x2 = TileTensor(x2_stack, l2)
    var x3 = TileTensor(x3_stack, l3)

    var x1_dyn = x1.make_dynamic[.int64]()
    var x2_dyn = x2.make_dynamic[.int64]()
    var x3_dyn = x3.make_dynamic[.int64]()

    comptime out_layout = row_major[2, 2, 6, 2]()
    var out_stack = Array[Scalar[dtype], out_layout.product()](fill=-1)
    var output = TileTensor(out_stack, out_layout)

    var input_tuple = StaticTuple[
        TileTensor[dtype, x1_dyn.LayoutType, ImmutAnyOrigin],
        3,
    ](
        x1_dyn.as_unsafe_any_origin().as_immut(),
        x2_dyn.as_unsafe_any_origin().as_immut(),
        x3_dyn.as_unsafe_any_origin().as_immut(),
    )

    @__parameter
    @always_inline
    def epilogue_plus_one[
        c_type: DType, _rank: Int, width: SIMDLength, *, alignment: Int
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        var coord = Coord(indices)
        comptime assert output.flat_rank >= coord.flat_rank
        output.store[width=width](
            coord,
            rebind[SIMD[dtype, width]](val + 1),
        )

    var input_vec = _tuple_to_list(input_tuple)
    _concat_parallel[dtype, epilogue_plus_one](
        output.make_dynamic[.int64](), concat_axis, input_vec
    )

    # CHECK: == test_concat_parallel
    # CHECK-COUNT-2: 1.0
    # CHECK-COUNT-4: 2.0
    # CHECK-COUNT-6: 3.0
    # CHECK-COUNT-2: 1.0
    # CHECK-COUNT-4: 2.0
    # CHECK-COUNT-6: 3.0
    # CHECK-COUNT-2: 1.0
    # CHECK-COUNT-4: 2.0
    # CHECK-COUNT-6: 3.0
    var output_flat = TileTensor(
        output._storage,
        row_major(Coord(output.num_elements())),
    )
    for i in range(output.layout.product()):
        print(output_flat.load[1]((i,)))


# CHECK-LABEL: test_concat_inner
def test_concat_inner() raises:
    print("== test_concat_inner")

    comptime dtype = DType.float32
    comptime rank = 5
    comptime concat_axis = 2

    comptime l1 = row_major[1, 1, 1, 2, 2]()
    comptime l2 = row_major[1, 1, 2, 2, 2]()
    comptime l3 = row_major[1, 1, 3, 2, 2]()
    var x1_stack = Array[Scalar[dtype], l1.product()](fill=0)
    var x2_stack = Array[Scalar[dtype], l2.product()](fill=1)
    var x3_stack = Array[Scalar[dtype], l3.product()](fill=2)
    var x1 = TileTensor(x1_stack, l1)
    var x2 = TileTensor(x2_stack, l2)
    var x3 = TileTensor(x3_stack, l3)

    var x1_dyn = x1.make_dynamic[.int64]()
    var x2_dyn = x2.make_dynamic[.int64]()
    var x3_dyn = x3.make_dynamic[.int64]()

    comptime out_layout = row_major[1, 1, 6, 2, 2]()
    var out_stack = Array[Scalar[dtype], out_layout.product()](fill=-1)
    var output = TileTensor(out_stack, out_layout)

    var input_tuple = StaticTuple[
        TileTensor[dtype, x1_dyn.LayoutType, ImmutAnyOrigin],
        3,
    ](
        x1_dyn.as_unsafe_any_origin().as_immut(),
        x2_dyn.as_unsafe_any_origin().as_immut(),
        x3_dyn.as_unsafe_any_origin().as_immut(),
    )

    var input_vec = _tuple_to_list(input_tuple)

    @__parameter
    @always_inline
    def epilogue_plus_one[
        c_type: DType, _rank: Int, width: SIMDLength, *, alignment: Int
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        var coord = Coord(indices)
        comptime assert output.flat_rank >= coord.flat_rank
        output.store[width=width](
            coord,
            rebind[SIMD[dtype, width]](val + 1),
        )

    _concat_serial[dtype, epilogue_plus_one](
        output.make_dynamic[.int64](), concat_axis, input_vec
    )

    # CHECK-COUNT-4: 1.0
    # CHECK-COUNT-8: 2.0
    # CHECK-COUNT-12: 3.0
    var output_flat = TileTensor(
        output._storage,
        row_major(Coord(output.num_elements())),
    )
    for i in range(output.layout.product()):
        print(output_flat.load[1]((i,)))


def main() raises:
    test_concat()
    test_concat_parallel()
    test_concat_inner()
