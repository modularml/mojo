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

from max.gpu.host import DeviceContext
from nn.gather_scatter import scatter_elements
from extensibility import DynamicTensor
from std.testing import assert_equal

from std.utils import IndexList


def main() raises:
    var ctx = DeviceContext(api="cpu")

    def test_scatter_ax0(ctx: DeviceContext) raises:
        print("== test_scatter_ax0")

        var data_ptr = List(length=9, fill=Float32(0))
        var data = DynamicTensor[.float32, 2](
            data_ptr.unsafe_ptr(), IndexList[2](3, 3)
        )

        var indices_ptr = List(length=6, fill=Int32(0))
        indices_ptr[0] = 1
        indices_ptr[2] = 2
        indices_ptr[4] = 2
        indices_ptr[5] = 1
        var indices = DynamicTensor[.int32, 2](
            indices_ptr.unsafe_ptr(), IndexList[2](2, 3)
        )

        var updates_ptr = List(length=6, fill=Float32(0))
        updates_ptr[0] = 1.0
        updates_ptr[1] = 1.1
        updates_ptr[2] = 1.2
        updates_ptr[3] = 2.0
        updates_ptr[4] = 2.1
        updates_ptr[5] = 2.2
        var updates = DynamicTensor[.float32, 2](
            updates_ptr.unsafe_ptr(), IndexList[2](2, 3)
        )

        var output_ptr = List(length=9, fill=Float32(0))
        var output = DynamicTensor[.float32, 2](
            output_ptr.unsafe_ptr(), IndexList[2](3, 3)
        )

        var expected: Array[Float32, 9] = [
            Float32(2.0),
            1.1,
            0.0,
            1.0,
            0.0,
            2.2,
            0.0,
            2.1,
            1.2,
        ]

        scatter_elements(data, indices, updates, 0, output, ctx)

        for i in range(9):
            assert_equal(output_ptr[i], expected[i])
        _ = output_ptr^
        _ = updates_ptr^
        _ = indices_ptr^
        _ = data_ptr^

    # CHECK-LABEL: test_scatter_ax0
    # CHECK-NOT: FAIL
    test_scatter_ax0(ctx)

    def test_scatter_ax1(ctx: DeviceContext) raises:
        print("== test_scatter_ax1")

        var data_ptr = List(length=5, fill=Float32(0))
        for i in range(5):
            data_ptr[i] = Float32(i + 1)
        var data = DynamicTensor[.float32, 2](
            data_ptr.unsafe_ptr(), IndexList[2](1, 5)
        )

        var indices_ptr = List(length=2, fill=Int32(0))
        indices_ptr[0] = 1
        indices_ptr[1] = 3
        var indices = DynamicTensor[.int32, 2](
            indices_ptr.unsafe_ptr(), IndexList[2](1, 2)
        )

        var updates_ptr = List(length=2, fill=Float32(0))
        updates_ptr[0] = 1.1
        updates_ptr[1] = 2.1
        var updates = DynamicTensor[.float32, 2](
            updates_ptr.unsafe_ptr(), IndexList[2](1, 2)
        )

        var output_ptr = List(length=5, fill=Float32(0))
        var output = DynamicTensor[.float32, 2](
            output_ptr.unsafe_ptr(), IndexList[2](1, 5)
        )

        var expected: Array[Float32, 5] = [
            Float32(1.0),
            1.1,
            3.0,
            2.1,
            5.0,
        ]

        scatter_elements(data, indices, updates, 1, output, ctx)

        for i in range(5):
            assert_equal(output_ptr[i], expected[i])
        _ = output_ptr^
        _ = updates_ptr^
        _ = indices_ptr^
        _ = data_ptr^

    # CHECK-LABEL: test_scatter_ax1
    # CHECK-NOT: FAIL
    test_scatter_ax1(ctx)

    def test_scatter_neg_indices(ctx: DeviceContext) raises:
        print("== test_scatter_neg_indices")

        var data_ptr = List(length=5, fill=Float32(0))
        for i in range(5):
            data_ptr[i] = Float32(i + 1)
        var data = DynamicTensor[.float32, 2](
            data_ptr.unsafe_ptr(), IndexList[2](1, 5)
        )

        var indices_ptr = List(length=2, fill=Int32(0))
        indices_ptr[0] = 1
        indices_ptr[1] = -3
        var indices = DynamicTensor[.int32, 2](
            indices_ptr.unsafe_ptr(), IndexList[2](1, 2)
        )

        var updates_ptr = List(length=2, fill=Float32(0))
        updates_ptr[0] = 1.1
        updates_ptr[1] = 2.1
        var updates = DynamicTensor[.float32, 2](
            updates_ptr.unsafe_ptr(), IndexList[2](1, 2)
        )

        var output_ptr = List(length=5, fill=Float32(0))
        var output = DynamicTensor[.float32, 2](
            output_ptr.unsafe_ptr(), IndexList[2](1, 5)
        )

        var expected: Array[Float32, 5] = [
            Float32(1.0),
            1.1,
            2.1,
            4.0,
            5.0,
        ]

        scatter_elements(data, indices, updates, 1, output, ctx)

        for i in range(5):
            assert_equal(output_ptr[i], expected[i])
        _ = output_ptr^
        _ = updates_ptr^
        _ = indices_ptr^
        _ = data_ptr^

    # CHECK-LABEL: test_scatter_neg_indices
    # CHECK-NOT: FAIL
    test_scatter_neg_indices(ctx)

    def test_scatter_reduce_max(ctx: DeviceContext) raises:
        print("== test_scatter_reduce_max")

        var data_ptr = List(length=5, fill=Float32(0))
        for i in range(5):
            data_ptr[i] = Float32(i + 1)
        var data = DynamicTensor[.float32, 2](
            data_ptr.unsafe_ptr(), IndexList[2](1, 5)
        )

        var indices_ptr = List(length=2, fill=Int32(0))
        indices_ptr[0] = 1
        indices_ptr[1] = 1
        var indices = DynamicTensor[.int32, 2](
            indices_ptr.unsafe_ptr(), IndexList[2](1, 2)
        )

        var updates_ptr = List(length=2, fill=Float32(0))
        updates_ptr[0] = 1.1
        updates_ptr[1] = 2.1
        var updates = DynamicTensor[.float32, 2](
            updates_ptr.unsafe_ptr(), IndexList[2](1, 2)
        )

        var output_ptr = List(length=5, fill=Float32(0))
        var output = DynamicTensor[.float32, 2](
            output_ptr.unsafe_ptr(), IndexList[2](1, 5)
        )

        var expected: Array[Float32, 5] = [
            Float32(1.0),
            2.1,
            3.0,
            4.0,
            5.0,
        ]

        @always_inline
        def _max[
            ty: DType, width: SIMDLength
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return max(v1, v2)

        scatter_elements[reduce_fn=_max](data, indices, updates, 1, output, ctx)

        for i in range(5):
            assert_equal(output_ptr[i], expected[i])
        _ = output_ptr^
        _ = updates_ptr^
        _ = indices_ptr^
        _ = data_ptr^

    # CHECK-LABEL: test_scatter_reduce_max
    # CHECK-NOT: FAIL
    test_scatter_reduce_max(ctx)

    def test_scatter_reduce_add_parallel_duplicates(
        ctx: DeviceContext,
    ) raises:
        print("== test_scatter_reduce_add_parallel_duplicates")
        # More index rows than the CPU elementwise grain size (32768), so
        # updates run on several workers concurrently (elementwise
        # parallelizes over the outer dimension). Duplicate indices must
        # still accumulate atomically — with 100k rows colliding on 8
        # target rows, a non-atomic reduce drops updates.
        comptime rows = 64
        comptime n_idx = 100_000
        comptime n_targets = 8

        var data_ptr = List(length=rows, fill=Float32(0))
        for i in range(rows):
            data_ptr[i] = Float32(i % 7)
        var data = DynamicTensor[.float32, 2](
            data_ptr.unsafe_ptr(), IndexList[2](rows, 1)
        )

        var indices_ptr = List(length=n_idx, fill=Int32(0))
        for k in range(n_idx):
            indices_ptr[k] = Int32((k % n_targets) * 7 + 1)
        var indices = DynamicTensor[.int32, 2](
            indices_ptr.unsafe_ptr(), IndexList[2](n_idx, 1)
        )

        var updates_ptr = List(length=n_idx, fill=Float32(1))
        var updates = DynamicTensor[.float32, 2](
            updates_ptr.unsafe_ptr(), IndexList[2](n_idx, 1)
        )

        var output_ptr = List(length=rows, fill=Float32(0))
        var output = DynamicTensor[.float32, 2](
            output_ptr.unsafe_ptr(), IndexList[2](rows, 1)
        )

        @always_inline
        def _add[
            ty: DType, width: SIMDLength
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return v1 + v2

        scatter_elements[reduce_fn=_add](data, indices, updates, 0, output, ctx)

        comptime dups_per_target = n_idx // n_targets
        for i in range(rows):
            var expected = data_ptr[i]
            if i % 7 == 1 and i < n_targets * 7:
                expected += Float32(dups_per_target)
            assert_equal(output_ptr[i], expected, String("i=", i))

        _ = output_ptr^
        _ = updates_ptr^
        _ = indices_ptr^
        _ = data_ptr^

    # CHECK-LABEL: test_scatter_reduce_add_parallel_duplicates
    # CHECK-NOT: FAIL
    test_scatter_reduce_add_parallel_duplicates(ctx)
