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


from std.math import iota
from std.random import rand, seed

from layout import (
    Coord,
    Idx,
    DefaultEngine,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from layout.coord import DynamicCoord
from layout.tile_layout import Layout

from nn.topk import _top_k_cpu, _top_k_sampling

from std.utils.index import IndexList, product

comptime FillFnType = def[rank: Int, dtype: DType](
    TileTensor[mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]]
) -> None


struct TestTensor[rank: Int, dtype: DType](Movable):
    var storage: List[Scalar[Self.dtype]]
    var shape: IndexList[Self.rank]

    def __init__(out self, shape: IndexList[Self.rank]):
        self.storage = List[Scalar[Self.dtype]](
            length=shape.flattened_length(), fill=0
        )
        self.shape = shape

    def to_tile_tensor(
        ref self,
    ) -> TileTensor[
        Self.dtype,
        Layout[
            shape_types=DynamicCoord[.int64, Self.rank].element_types,
            stride_types=DynamicCoord[.int64, Self.rank].element_types,
        ],
        origin_of(self.storage),
    ]:
        return rebind[
            TileTensor[
                Self.dtype,
                Layout[
                    shape_types=DynamicCoord[.int64, Self.rank].element_types,
                    stride_types=DynamicCoord[.int64, Self.rank].element_types,
                ],
                origin_of(self.storage),
            ]
        ](
            TileTensor(
                Span[Scalar[Self.dtype]](self.storage),
                row_major(Coord(self.shape)),
            ).make_dynamic[.int64]()
        )


def test_case_sampling[
    rank: Int,
    dtype: DType,
    FillFn: ImplicitlyCopyable & FillFnType,
](
    fill_fn: FillFn,
    K: Int,
    axis: Int,
    input_shape: IndexList[rank],
    temperature: Scalar[dtype] = 1,
) raises:
    var input_ptr = List(length=product(input_shape), fill=Scalar[dtype](0))
    var input = TileTensor(input_ptr, row_major(Coord(input_shape)))

    var output_shape: IndexList[rank]
    var output_idxs_shape: IndexList[rank]

    comptime if rank == 1:
        output_shape = IndexList[rank](K)
        output_idxs_shape = IndexList[rank](1)
    elif rank == 2:
        output_shape = IndexList[rank](input_shape[0], K)
        output_idxs_shape = IndexList[rank](input_shape[0], 1)
    else:
        output_shape = IndexList[rank](input_shape[0], input_shape[1], K)
        output_idxs_shape = IndexList[rank](input_shape[0], input_shape[1], 1)

    var output_vals_ptr = List(
        length=product(output_shape), fill=Scalar[dtype](0)
    )
    var output_idxs_ptr = List(length=product(output_idxs_shape), fill=Int64(0))
    var out_vals = TileTensor(output_vals_ptr, row_major(Coord(output_shape)))
    var out_idxs = TileTensor(
        output_idxs_ptr, row_major(Coord(output_idxs_shape))
    )

    fill_fn[rank, dtype](input)

    var max_k = K

    var batch_size: Int
    comptime if rank == 1:
        batch_size = 1
    elif rank == 2:
        batch_size = input_shape[0]
    else:
        batch_size = input_shape[0] * input_shape[1]
    var temperature_ptr = List(length=batch_size, fill=Float32(0))
    for i in range(batch_size):
        temperature_ptr[i] = temperature.cast[.float32]()

    var temperature_buf = Optional(
        TileTensor(temperature_ptr, row_major(Int64(batch_size)))
        .as_unsafe_any_origin()
        .as_immut()
    )

    var seed_ptr = List(length=batch_size, fill=UInt64(12))
    var seed_buf = Optional(
        TileTensor(seed_ptr, row_major(Int64(batch_size)))
        .as_unsafe_any_origin()
        .as_immut()
    )

    _top_k_sampling(
        max_k,
        input,
        out_vals,
        out_idxs,
        temperature=temperature_buf,
        seed=seed_buf,
    )

    var _xxx_no_lifetimes = input  # intentionally bad name
    var _xx_no_lifetimes = out_vals
    var _x_no_lifetimes = out_idxs

    for i in range(out_idxs.num_elements()):
        print(out_idxs.raw_load(i), end="")
        print(",", end="")
    print("")


def test_case[
    rank: Int,
    dtype: DType,
    FillFn: ImplicitlyCopyable & FillFnType,
    largest: Bool = True,
](
    fill_fn: FillFn,
    K: Int,
    axis: Int,
    input_shape: IndexList[rank],
    sorted: Bool = True,
):
    var input = TestTensor[rank, dtype](input_shape)

    var output_shape = input_shape
    output_shape[axis] = K
    var out_vals = TestTensor[rank, dtype](output_shape)
    var out_idxs = TestTensor[rank, .int64](output_shape)

    var input_buf = input.to_tile_tensor()
    fill_fn[rank, dtype](input_buf)

    _top_k_cpu[largest=largest](
        input.to_tile_tensor(),
        K,
        axis,
        out_vals.to_tile_tensor(),
        out_idxs.to_tile_tensor(),
        1,  # force multithreading for small test cases,
        sorted,
    )

    var _xxx_no_origins = input^  # intentionally bad name

    for i in range(len(out_vals.storage)):
        print(out_vals.storage[i], end="")
        print(",", end="")
    print("")
    for i in range(len(out_idxs.storage)):
        print(out_idxs.storage[i], end="")
        print(",", end="")
    print("")


def main() raises:
    seed(1)

    def fill_iota[
        rank: Int, dtype: DType
    ](
        buf: TileTensor[
            mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
        ]
    ):
        iota(
            buf._storage,
            coord_to_index_list(buf.layout.shape_coord()).flattened_length(),
        )

    def fill_rand[
        rank: Int, dtype: DType
    ](
        buf: TileTensor[
            mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
        ]
    ):
        rand(
            buf._storage,
            coord_to_index_list(buf.layout.shape_coord()).flattened_length(),
        )

    def test_1d_sorted():
        print("== test_1d_sorted")
        test_case[1, .float32](fill_iota, 5, 0, IndexList[1](10), sorted=True)

    # CHECK-LABEL: test_1d_sorted
    # CHECK: 9.0,8.0,7.0,6.0,5.0,
    # CHECK: 9,8,7,6,5,
    test_1d_sorted()

    def test_1d_notsorted():
        print("== test_1d_notsorted")
        test_case[1, .float32](fill_iota, 5, 0, IndexList[1](10), sorted=False)

    # CHECK-LABEL: test_1d_notsorted
    # CHECK: 8.0,7.0,6.0,9.0,5.0,
    # CHECK: 8,7,6,9,5,
    test_1d_notsorted()

    def test_axis_1():
        print("== test_axis_1")
        test_case[2, .float32](fill_iota, 2, 1, IndexList[2](4, 4), sorted=True)

    # CHECK-LABEL: test_axis_1
    # CHECK: 3.0,2.0,7.0,6.0,11.0,10.0,15.0,14.0,
    # CHECK-NEXT: 3,2,3,2,3,2,3,2,
    test_axis_1()

    def test_axis_1_notsorted():
        print("== test_axis_1_notsorted")
        test_case[2, .float32](
            fill_iota, 2, 1, IndexList[2](4, 4), sorted=False
        )

    # CHECK-LABEL: test_axis_1_notsorted
    # CHECK: 3.0,2.0,7.0,6.0,11.0,10.0,15.0,14.0,
    # CHECK-NEXT: 3,2,3,2,3,2,3,2,
    test_axis_1_notsorted()

    def test_smallest():
        print("== test_smallest")
        test_case[2, .float32, largest=False](
            fill_iota, 2, 1, IndexList[2](4, 4), False
        )

    # CHECK-LABEL: test_smallest
    # CHECK: 0.0,1.0,4.0,5.0,8.0,9.0,12.0,13.0,
    # CHECK-NEXT: 0,1,0,1,0,1,0,1,
    test_smallest()

    def test_axis_0():
        print("== test_axis_0")
        test_case[2, .float32](fill_iota, 2, 0, IndexList[2](4, 4))

    # CHECK-LABEL: test_axis_0
    # CHECK: 12.0,13.0,14.0,15.0,8.0,9.0,10.0,11.0,
    # CHECK-NEXT: 3,3,3,3,2,2,2,2,
    test_axis_0()

    def fill_identical[
        rank: Int, dtype: DType
    ](
        buf: TileTensor[
            mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
        ]
    ):
        _ = buf.fill(1)

    def test_identical():
        print("== test_identical")
        test_case[2, .float32](fill_identical, 3, 0, IndexList[2](4, 4))

    # CHECK-LABEL: test_identical
    # CHECK: 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
    # CHECK-NEXT: 0,0,0,0,1,1,1,1,2,2,2,2,
    test_identical()

    def test_identical_large():
        print("== test_identical_large")
        test_case[2, .float32](fill_identical, 3, 0, IndexList[2](33, 33))

    # CHECK-LABEL: test_identical_large
    # CHECK: 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
    # CHECK: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    test_identical_large()

    def test_max_k():
        print("== test_max_k")
        test_case[2, .float32](fill_iota, 3, 0, IndexList[2](3, 4))

    # CHECK-LABEL: test_max_k
    # CHECK: 8.0,9.0,10.0,11.0,4.0,5.0,6.0,7.0,0.0,1.0,2.0,3.0,
    # CHECK-NEXT: 2,2,2,2,1,1,1,1,0,0,0,0,
    test_max_k()

    def fill_custom[
        rank: Int, dtype: DType
    ](
        buf: TileTensor[
            mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
        ]
    ):
        var flat_buf = TileTensor(
            buf._storage,
            row_major(buf.num_elements()),
        )

        for i in range(flat_buf.num_elements()):
            var idx = flat_buf.layout(Coord(i))
            flat_buf.raw_store(
                idx, Scalar[dtype](flat_buf.num_elements() - i - 1)
            )
        flat_buf[0] = -1

    def test_5d():
        print("== test_5d")
        test_case[5, .float32](fill_custom, 1, 1, IndexList[5](1, 4, 3, 2, 1))

    # CHECK-LABEL: == test_5d
    # CHECK: 17.0,22.0,21.0,20.0,19.0,18.0,
    # CHECK-NEXT: 1,0,0,0,0,0,
    test_5d()

    def test_1d_sorted_sampling() raises:
        print("== test_1d_sorted_sampling")
        comptime rank = 1
        test_case_sampling[1, .float32](
            fill_iota,
            5,
            0,
            IndexList[1](10),
            temperature=0,
        )

    # CHECK-LABEL: test_1d_sorted_sampling
    # CHECK: 9,
    test_1d_sorted_sampling()

    def test_2d_sorted_sampling() raises:
        print("== test_2d_sorted_sampling")
        test_case_sampling[2, .float32](
            fill_rand,
            5,
            1,
            IndexList[2](5, 10),
            temperature=0,
        )

    # CHECK-LABEL: test_2d_sorted_sampling
    # CHECK: 0,7,8,1,7,
    test_2d_sorted_sampling()

    def test_3d_sorted_sampling() raises:
        print("== test_3d_sorted_sampling")
        test_case_sampling[3, .float32](
            fill_rand,
            5,
            2,
            IndexList[3](3, 5, 10),
            temperature=0,
        )

    # CHECK-LABEL: test_3d_sorted_sampling
    # 6,9,5,2,3,1,7,9,5,1,9,0,2,3,4,
    test_3d_sorted_sampling()

    def ones[
        rank: Int, dtype: DType
    ](
        buf: TileTensor[
            mut=True, dtype, ..., Engine=DefaultEngine[element_width=1]
        ]
    ):
        for i in range(
            coord_to_index_list(buf.layout.shape_coord()).flattened_length()
        ):
            buf.raw_store(i, 1)

    def test_1d_sorted_sampling_temp() raises:
        print("== test_1d_sorted_sampling_temp")
        comptime rank = 1
        test_case_sampling[1, .float32](
            fill_rand, 5, 0, IndexList[1](10), temperature=0.7
        )

    # CHECK-LABEL: test_1d_sorted_sampling_temp
    # CHECK: 4,
    test_1d_sorted_sampling_temp()

    def test_2d_sorted_sampling_temp() raises:
        print("== test_2d_sorted_sampling_temp")
        test_case_sampling[2, .float32](
            fill_rand,
            5,
            1,
            IndexList[2](50, 10),
            temperature=0.7,
        )

    # CHECK-LABEL: test_2d_sorted_sampling_temp
    # CHECK: 2,3,9,2,6,7,4,8,0,5,5,7,5,4,3,3,2,4,3,8,1,2,2,3,5,5,5,2,6,3,9,1,2,0,8,7,1,6,2,2,8,3,2,1,4,8,0,9,2,8,
    test_2d_sorted_sampling_temp()

    def test_2d_sorted_sampling_temp_zero() raises:
        print("== test_2d_sorted_sampling_temp_zero")
        test_case_sampling[2, .float32](
            fill_rand,
            5,
            1,
            IndexList[2](50, 10),
            temperature=0.0,
        )

    # CHECK-LABEL: test_2d_sorted_sampling_temp_zero
    # CHECK: 2,6,3,2,0,8,0,1,7,8,1,6,2,1,6,3,6,9,6,9,1,3,4,6,0,1,2,6,1,5,5,7,1,7,0,8,6,0,3,5,6,9,0,7,0,8,1,2,4,8,
    test_2d_sorted_sampling_temp_zero()

    def test_deterministic_sampling() raises:
        print("== test_deterministic_sampling")
        test_case_sampling[2, .float32](
            ones,
            5,
            1,
            IndexList[2](50, 10),
        )

    # CHECK-LABEL: test_deterministic_sampling
    # CHECK: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    test_deterministic_sampling()
