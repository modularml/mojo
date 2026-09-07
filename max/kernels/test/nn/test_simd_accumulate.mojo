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


from layout import TileTensor, row_major
from linalg.accumulate import _Accumulator, _simd_load_maybe_partial
from std.testing import *


# TODO: rewrite c-layout comments according to the new struct.
def test_maybe_partial_load() raises:
    comptime simd_size = 4
    comptime size = simd_size + 1

    var a = Array[Float32, size](fill=1.0)

    var vec = _simd_load_maybe_partial[simd_size, False](a.unsafe_ptr(), 0)
    assert_equal(vec, SIMD[.float32, simd_size](1.0))

    vec = _simd_load_maybe_partial[simd_size, True](
        a.unsafe_ptr(), simd_size, 1
    )
    assert_equal(vec, SIMD[.float32, simd_size](1.0, 0.0, 0.0, 0.0))


def test_accumulate[
    simd_size: Int = 4, num_rows: Int = 2, num_cols: Int = 2, length: Int = 2
]() raises:
    comptime type = DType.float32

    # A: [[ 0.0, 0.0 ],
    #     [ 1.0, 1.0 ],
    #     [ 2.0, 2.0 ],
    #     [ 3.0, 3.0 ]]
    var a = Array[Scalar[type], 2 * num_rows * length](fill={})
    var a_base_ptr: MutPointer[Scalar[type], origin_of(a)] = a.unsafe_ptr()
    for i in range(2 * num_rows):
        var a_ptr = a_base_ptr + i * length
        a_ptr[0] = Scalar[type](i)
        a_ptr[1] = Scalar[type](i)

    # 4 x 0.0 denotes 0.0, 0.0, 0.0, 0.0
    # B: [[4 x 0.0, 4 x 0.0, 4 x 1.0, 4 x 1.0],
    #     [4 x 2.0, 4 x 2.0, 4 x 3.0, 4 x 3.0]]
    comptime b_size = 2 * num_cols * simd_size * length
    comptime kernel_width = num_cols * simd_size
    var b = Array[Scalar[type], b_size](fill={})
    var b_base_ptr: MutPointer[Scalar[type], origin_of(b)] = b.unsafe_ptr()

    for i in range(2 * length):
        var b_ptr = b_base_ptr + i * num_cols * simd_size

        comptime for j in range(num_cols):
            (b_ptr + j * simd_size).store(SIMD[type, simd_size](i))

    var acc = _Accumulator[type, num_rows, num_cols, simd_size]()
    acc.init(0)
    acc.accumulate(length, a.unsafe_ptr(), length, b.unsafe_ptr(), kernel_width)

    # C results:
    # C[0,0]:[0.0, 0.0, 0.0, 0.0]  C[0,1]:[0.0, 0.0, 0.0, 0.0]
    # C[1,0]:[1.0, 1.0, 1.0, 1.0]  C[1,1]:[1.0, 1.0, 1.0, 1.0]
    assert_equal(acc[0, 0], SIMD[type, simd_size](0.0))
    assert_equal(acc[0, 1], SIMD[type, simd_size](0.0))
    assert_equal(
        acc[1, 0],
        SIMD[type, simd_size](1.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[type, simd_size](1.0),
    )

    acc.accumulate(
        length,
        a.unsafe_ptr(),
        2 * length,
        b_base_ptr + kernel_width,
        kernel_width,
    )

    # C results:
    # C[0,0]:[0.0, 0.0, 0.0, 0.0]  C[0,1]:[0.0, 0.0, 0.0, 0.0]
    # C[1,0]:[7.0, 7.0, 7.0, 7.0]  C[1,1]:[7.0, 7.0, 7.0, 7.0]
    assert_equal(acc[0, 0], SIMD[type, simd_size](0.0))
    assert_equal(acc[0, 1], SIMD[type, simd_size](0.0))
    assert_equal(
        acc[1, 0],
        SIMD[type, simd_size](7.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[type, simd_size](7.0),
    )

    acc.accumulate(
        length,
        a_base_ptr + length,
        2 * length,
        b_base_ptr + kernel_width,
        2 * kernel_width,
    )

    # C results:
    # C[0,0]:[4.0, 4.0, 4.0, 4.0]     C[0,1]:[4.0, 4.0, 4.0, 4.0]
    # C[1,0]:[19.0, 19.0, 19.0, 19.0] C[1,1]:[19.0, 19.0, 19.0, 19.0]
    assert_equal(acc[0, 0], SIMD[type, simd_size](4.0))
    assert_equal(acc[0, 1], SIMD[type, simd_size](4.0))
    assert_equal(
        acc[1, 0],
        SIMD[type, simd_size](19.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[type, simd_size](19.0),
    )


def test_accumulate_with_offsets[
    simd_size: Int = 4, num_rows: Int = 2, num_cols: Int = 2, length: Int = 2
]() raises:
    comptime type = DType.float32

    # A: [[ 0.0, 0.0 ],
    #     [ 1.0, 1.0 ],
    #     [ 2.0, 2.0 ],
    #     [ 3.0, 3.0 ]]
    var a = Array[Scalar[type], 2 * num_rows * length](fill={})
    var a_base_ptr: MutPointer[Scalar[type], origin_of(a)] = a.unsafe_ptr()
    for i in range(2 * num_rows):
        var a_ptr = a_base_ptr + i * length
        a_ptr[0] = Scalar[type](i)
        a_ptr[1] = Scalar[type](i)

    # 4 x 0.0 denotes 0.0, 0.0, 0.0, 0.0
    # B: [[4 x 0.0, 4 x 0.0, 4 x 1.0, 4 x 1.0],
    #     [4 x 2.0, 4 x 2.0, 4 x 3.0, 4 x 3.0]]
    comptime b_size = 2 * num_cols * simd_size * length
    comptime kernel_width = num_cols * simd_size
    var b = Array[Scalar[type], b_size](fill={})
    var b_base_ptr: MutPointer[Scalar[type], origin_of(b)] = b.unsafe_ptr()

    for i in range(2 * length):
        var b_ptr = b_base_ptr + i * num_cols * simd_size

        comptime for j in range(num_cols):
            (b_ptr + j * simd_size).store(SIMD[type, simd_size](i))

    var a_base_stack = Array[Int32, num_rows](
        fill_with=lambda (i: Int) -> Int32: Int32(i * length)
    )
    var a_base_offsets = TileTensor(a_base_stack, row_major[num_rows]())

    var acc = _Accumulator[type, num_rows, num_cols, simd_size]()
    acc.init(0)
    acc.accumulate(
        length, a.unsafe_ptr(), a_base_offsets, 0, b.unsafe_ptr(), kernel_width
    )

    # C results:
    # [0.0, 0.0, 0.0, 0.0]
    # [0.0, 0.0, 0.0, 0.0]
    # [1.0, 1.0, 1.0, 1.0]
    # [1.0, 1.0, 1.0, 1.0]
    assert_equal(acc[0, 0], SIMD[type, simd_size](0.0))
    assert_equal(acc[0, 1], SIMD[type, simd_size](0.0))
    assert_equal(
        acc[1, 0],
        SIMD[type, simd_size](1.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[type, simd_size](1.0),
    )

    a_base_offsets[0] = 0
    a_base_offsets[1] = Int32(2 * length)
    acc.accumulate(
        length,
        a.unsafe_ptr(),
        a_base_offsets,
        0,
        b_base_ptr + kernel_width,
        kernel_width,
    )

    # C results:
    # [0.0, 0.0, 0.0, 0.0]
    # [0.0, 0.0, 0.0, 0.0]
    # [7.0, 7.0, 7.0, 7.0]
    # [7.0, 7.0, 7.0, 7.0]
    assert_equal(acc[0, 0], SIMD[type, simd_size](0.0))
    assert_equal(acc[0, 1], SIMD[type, simd_size](0.0))
    assert_equal(
        acc[1, 0],
        SIMD[type, simd_size](7.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[type, simd_size](7.0),
    )

    a_base_offsets[0] = Int32(length)
    a_base_offsets[1] = Int32(3 * length)

    acc.accumulate(
        length,
        a.unsafe_ptr(),
        a_base_offsets,
        0,
        b_base_ptr + kernel_width,
        2 * kernel_width,
    )

    # C results:
    # [4.0, 4.0, 4.0, 4.0]
    # [4.0, 4.0, 4.0, 4.0]
    # [19.0, 19.0, 19.0, 19.0]
    # [19.0, 19.0, 19.0, 19.0]
    assert_equal(acc[0, 0], SIMD[type, simd_size](4.0))
    assert_equal(acc[0, 1], SIMD[type, simd_size](4.0))
    assert_equal(
        acc[1, 0],
        SIMD[type, simd_size](19.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[type, simd_size](19.0),
    )


def test_load_store[
    simd_size: Int = 4, num_rows: Int = 2, num_cols: Int = 2, length: Int = 2
]() raises:
    comptime type = DType.float32
    comptime size = simd_size + 1
    comptime residual = 1
    comptime row_size = num_cols * simd_size + residual
    comptime one_vec = SIMD[type, simd_size](1.0)
    comptime residual_vec = SIMD[type, simd_size](-1.0, 0.0, 0.0, 0.0)

    var a = Array[Scalar[type], num_rows * row_size](fill={})
    var a_ptr: MutPointer[Scalar[type], origin_of(a)] = a.unsafe_ptr()

    # A: [[ 4x0.0, 4x1.0, -1.0],
    #     [ 4x1.0, 4x2.0, -1.0]]
    comptime for i in range(num_rows):
        comptime for j in range(num_cols):
            a_ptr.store(
                i * row_size + j * simd_size,
                SIMD[type, simd_size](i + j),
            )

        a_ptr.store(
            i * row_size + num_cols * simd_size,
            SIMD[type, residual](-1.0),
        )

    var tile0 = _Accumulator[type, num_rows, num_cols, simd_size]()
    tile0.load(a.unsafe_ptr(), row_size)

    assert_equal(
        tile0[0, 0],
        SIMD[type, simd_size](0.0),
    )
    assert_equal(
        tile0[0, 1],
        SIMD[type, simd_size](1.0),
    )
    assert_equal(
        tile0[1, 0],
        SIMD[type, simd_size](1.0),
    )
    assert_equal(
        tile0[1, 1],
        SIMD[type, simd_size](2.0),
    )

    # Update A: [[ 4x1.0, 4x1.0, -1.0],
    #            [ 4x1.0, 4x1.0, -1.0]]
    tile0[0, 0] = one_vec
    tile0[1, 1] = one_vec
    tile0.store(a.unsafe_ptr(), row_size)

    var tile1 = _Accumulator[type, num_rows, num_cols + 1, simd_size]()

    tile1.load[partial_load=True](a.unsafe_ptr(), row_size, residual)

    assert_equal(tile1[0, 0], one_vec)
    assert_equal(tile1[0, 1], one_vec)
    assert_equal(tile1[0, 2], residual_vec)
    assert_equal(tile1[1, 0], one_vec)
    assert_equal(tile1[1, 1], one_vec)
    assert_equal(tile1[1, 2], residual_vec)

    var residual_vec1 = SIMD[type, residual](-2.0)

    # TODO: replace the following with simd.mojo:insert (after resolving its issue).
    @always_inline
    def simd_insert(mut x: SIMD[type, _], y: SIMD[type, _]):
        comptime assert x.length >= y.length

        comptime for i in range(y.length):
            x[i] = y[i]

    simd_insert(tile1[0, 2], residual_vec1)
    simd_insert(tile1[1, 2], residual_vec1)

    # Update A: [[ 4x1.0, 4x1.0, -2.0],
    #            [ 4x1.0, 4x1.0, -2.0]]
    tile1.store[partial_store=True](a.unsafe_ptr(), row_size, residual)

    assert_equal(a_ptr.load[width=residual](row_size - residual), residual_vec1)
    assert_equal(
        a_ptr.load[width=residual](2 * row_size - residual),
        residual_vec1,
    )


def main() raises:
    test_maybe_partial_load()
    test_accumulate()
    test_accumulate_with_offsets()
    test_load_store()
