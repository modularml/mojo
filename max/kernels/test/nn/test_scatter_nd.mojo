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
from max.gpu.host import DeviceContext
from nn.gather_scatter import scatter_nd_generator
from std.testing import assert_equal


@always_inline
def use_update[
    dtype: DType, width: SIMDLength, //
](input_val: SIMD[dtype, width], update_val: SIMD[dtype, width]) -> SIMD[
    dtype, width
]:
    return update_val


def main() raises:
    def test_scatternd() raises:
        print("== test_scatternd")
        # data: 4x4x4 = 64 elements
        var data_ptr = List(length=64, fill=Float32(0))
        var data_vals: Array[Float32, 64] = [
            Float32(1),
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]
        for i in range(64):
            data_ptr[i] = data_vals[i]

        var data = TileTensor(data_ptr, row_major[4, 4, 4]())

        # indices: 2x1 = 2 elements
        var indices_ptr = List(length=2, fill=Int64(0))
        indices_ptr[1] = 2

        var indices = TileTensor(indices_ptr, row_major[2, 1]())

        # updates: 2x4x4 = 32 elements
        var updates_ptr = List(length=32, fill=Float32(0))
        var updates_vals: Array[Float32, 32] = [
            Float32(5),
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
        ]
        for i in range(32):
            updates_ptr[i] = updates_vals[i]

        var updates = TileTensor(updates_ptr, row_major[2, 4, 4]())

        # output: 4x4x4 = 64 elements
        var output_ptr = List(length=64, fill=Float32(0))
        var output = TileTensor(output_ptr, row_major[4, 4, 4]())

        # expected output
        var expected: Array[Float32, 64] = [
            Float32(5),
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]

        scatter_nd_generator[reduce_fn=use_update](
            data, indices, updates, output, DeviceContext(api="cpu")
        )

        for i in range(64):
            assert_equal(output_ptr[i], expected[i])
        _ = output_ptr^
        _ = updates_ptr^
        _ = indices_ptr^
        _ = data_ptr^

    test_scatternd()

    def test_scatternd_add() raises:
        print("== test_scatternd_add")
        # data: 4x4x4 = 64 elements
        var data_ptr = List(length=64, fill=Float32(0))
        var data_vals: Array[Float32, 64] = [
            Float32(1),
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]
        for i in range(64):
            data_ptr[i] = data_vals[i]

        var data = TileTensor(data_ptr, row_major[4, 4, 4]())

        # indices: 2x1 = 2 elements (both pointing to index 0)
        var indices_ptr = List(length=2, fill=Int64(0))

        var indices = TileTensor(indices_ptr, row_major[2, 1]())

        # updates: 2x4x4 = 32 elements
        var updates_ptr = List(length=32, fill=Float32(0))
        var updates_vals: Array[Float32, 32] = [
            Float32(5),
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
        ]
        for i in range(32):
            updates_ptr[i] = updates_vals[i]

        var updates = TileTensor(updates_ptr, row_major[2, 4, 4]())

        # output: 4x4x4 = 64 elements
        var output_ptr = List(length=64, fill=Float32(0))
        var output = TileTensor(output_ptr, row_major[4, 4, 4]())

        # expected output (add reduction)
        var expected: Array[Float32, 64] = [
            Float32(7),
            8,
            9,
            10,
            13,
            14,
            15,
            16,
            18,
            17,
            16,
            15,
            16,
            15,
            14,
            13,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]

        @always_inline
        def _add[
            ty: DType, width: SIMDLength
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return v1 + v2

        scatter_nd_generator[reduce_fn=_add](
            data, indices, updates, output, DeviceContext(api="cpu")
        )

        for i in range(64):
            assert_equal(output_ptr[i], expected[i])
        _ = output_ptr^
        _ = updates_ptr^
        _ = indices_ptr^
        _ = data_ptr^

    test_scatternd_add()

    def test_scatternd_max() raises:
        print("== test_scatternd_max")
        # data: 4x4x4 = 64 elements
        var data_ptr = List(length=64, fill=Float32(0))
        var data_vals: Array[Float32, 64] = [
            Float32(1),
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]
        for i in range(64):
            data_ptr[i] = data_vals[i]

        var data = TileTensor(data_ptr, row_major[4, 4, 4]())

        # indices: 2x1 = 2 elements (both pointing to index 0)
        var indices_ptr = List(length=2, fill=Int64(0))

        var indices = TileTensor(indices_ptr, row_major[2, 1]())

        # updates: 2x4x4 = 32 elements
        var updates_ptr = List(length=32, fill=Float32(0))
        var updates_vals: Array[Float32, 32] = [
            Float32(5),
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
        ]
        for i in range(32):
            updates_ptr[i] = updates_vals[i]

        var updates = TileTensor(updates_ptr, row_major[2, 4, 4]())

        # output: 4x4x4 = 64 elements
        var output_ptr = List(length=64, fill=Float32(0))
        var output = TileTensor(output_ptr, row_major[4, 4, 4]())

        # expected output (max reduction)
        var expected: Array[Float32, 64] = [
            Float32(5),
            5,
            5,
            5,
            6,
            6,
            7,
            8,
            8,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]

        @always_inline
        def _max[
            ty: DType, width: SIMDLength
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return max(v1, v2)

        scatter_nd_generator[reduce_fn=_max](
            data, indices, updates, output, DeviceContext(api="cpu")
        )

        for i in range(64):
            assert_equal(output_ptr[i], expected[i])
        _ = output_ptr^
        _ = updates_ptr^
        _ = indices_ptr^
        _ = data_ptr^

    test_scatternd_max()

    def test_scatternd_min() raises:
        print("== test_scatternd_min")
        # data: 4x4x4 = 64 elements
        var data_ptr = List(length=64, fill=Float32(0))
        var data_vals: Array[Float32, 64] = [
            Float32(1),
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]
        for i in range(64):
            data_ptr[i] = data_vals[i]

        var data = TileTensor(data_ptr, row_major[4, 4, 4]())

        # indices: 2x1 = 2 elements (both pointing to index 0)
        var indices_ptr = List(length=2, fill=Int64(0))

        var indices = TileTensor(indices_ptr, row_major[2, 1]())

        # updates: 2x4x4 = 32 elements
        var updates_ptr = List(length=32, fill=Float32(0))
        var updates_vals: Array[Float32, 32] = [
            Float32(5),
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
        ]
        for i in range(32):
            updates_ptr[i] = updates_vals[i]

        var updates = TileTensor(updates_ptr, row_major[2, 4, 4]())

        # output: 4x4x4 = 64 elements
        var output_ptr = List(length=64, fill=Float32(0))
        var output = TileTensor(output_ptr, row_major[4, 4, 4]())

        # expected output (min reduction)
        var expected: Array[Float32, 64] = [
            Float32(1),
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]

        @always_inline
        def _min[
            ty: DType, width: SIMDLength
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return min(v1, v2)

        scatter_nd_generator[reduce_fn=_min](
            data, indices, updates, output, DeviceContext(api="cpu")
        )

        for i in range(64):
            assert_equal(output_ptr[i], expected[i])
        _ = output_ptr^
        _ = updates_ptr^
        _ = indices_ptr^
        _ = data_ptr^

    test_scatternd_min()

    def test_scatternd_multiply() raises:
        print("== test_scatternd_multiply")
        # data: 4x4x4 = 64 elements
        var data_ptr = List(length=64, fill=Float32(0))
        var data_vals: Array[Float32, 64] = [
            Float32(1),
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]
        for i in range(64):
            data_ptr[i] = data_vals[i]

        var data = TileTensor(data_ptr, row_major[4, 4, 4]())

        # indices: 2x1 = 2 elements (both pointing to index 0)
        var indices_ptr = List(length=2, fill=Int64(0))

        var indices = TileTensor(indices_ptr, row_major[2, 1]())

        # updates: 2x4x4 = 32 elements
        var updates_ptr = List(length=32, fill=Float32(0))
        var updates_vals: Array[Float32, 32] = [
            Float32(5),
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
        ]
        for i in range(32):
            updates_ptr[i] = updates_vals[i]

        var updates = TileTensor(updates_ptr, row_major[2, 4, 4]())

        # output: 4x4x4 = 64 elements
        var output_ptr = List(length=64, fill=Float32(0))
        var output = TileTensor(output_ptr, row_major[4, 4, 4]())

        # expected output (multiply reduction)
        var expected: Array[Float32, 64] = [
            Float32(5),
            10,
            15,
            20,
            60,
            72,
            84,
            96,
            168,
            147,
            126,
            105,
            128,
            96,
            64,
            32,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]

        @always_inline
        def _mul[
            ty: DType, width: SIMDLength
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return v1 * v2

        scatter_nd_generator[reduce_fn=_mul](
            data, indices, updates, output, DeviceContext(api="cpu")
        )

        for i in range(64):
            assert_equal(output_ptr[i], expected[i])
        _ = output_ptr^
        _ = updates_ptr^
        _ = indices_ptr^
        _ = data_ptr^

    test_scatternd_multiply()

    def test_scatternd_add_parallel_duplicates() raises:
        print("== test_scatternd_add_parallel_duplicates")
        # More index rows than the CPU elementwise grain size (32768), so
        # the reduce runs on several workers concurrently. Duplicate index
        # vectors must still accumulate atomically — with 100k rows
        # colliding on 8 target rows, a non-atomic reduce drops updates.
        comptime rows = 64
        comptime cols = 4
        comptime n_idx = 100_000
        comptime n_targets = 8

        var data_ptr = List(length=rows * cols, fill=Float32(0))
        for i in range(rows * cols):
            data_ptr[i] = Float32(i % 7)
        var data = TileTensor(data_ptr, row_major[rows, cols]())

        var indices_ptr = List(length=n_idx, fill=Int64(0))
        for k in range(n_idx):
            indices_ptr[k] = Int64((k % n_targets) * 7 + 1)
        var indices = TileTensor(indices_ptr, row_major[n_idx, 1]())

        var updates_ptr = List(length=n_idx * cols, fill=Float32(1))
        var updates = TileTensor(updates_ptr, row_major[n_idx, cols]())

        var output_ptr = List(length=rows * cols, fill=Float32(0))
        var output = TileTensor(output_ptr, row_major[rows, cols]())

        @always_inline
        def _add[
            ty: DType, width: SIMDLength
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return v1 + v2

        scatter_nd_generator[reduce_fn=_add](
            data, indices, updates, output, DeviceContext(api="cpu")
        )

        comptime dups_per_target = n_idx // n_targets
        for i in range(rows * cols):
            var expected = data_ptr[i]
            var row = i // cols
            if row % 7 == 1 and row < n_targets * 7:
                expected += Float32(dups_per_target)
            assert_equal(output_ptr[i], expected, String("i=", i))

        _ = output_ptr^
        _ = updates_ptr^
        _ = indices_ptr^
        _ = data_ptr^

    test_scatternd_add_parallel_duplicates()
