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

from max.algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    elementwise,
)
from max.gpu.host import DeviceContext
from std.testing import assert_equal, assert_true
from std.testing import TestSuite

from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import IndexList, Index


def _linear_index[
    rank: Int
](coords: IndexList[rank], shape: IndexList[rank]) -> Int:
    """Convert multi-dimensional coordinates to linear index (row-major)."""
    var linear_idx = 0
    var stride = 1

    comptime for i in reversed(range(rank)):
        linear_idx += coords[i] * stride
        stride *= shape[i]
    return linear_idx


def test_elementwise() raises:
    var ctx = DeviceContext(api="cpu")

    def run_elementwise[
        numelems: Int,
        outer_rank: Int,
        shape: IndexList[outer_rank],
    ](ctx: DeviceContext) raises:
        var memory1 = Array[Float32, numelems](fill={})
        var buffer1 = Span(memory1)

        var memory2 = Array[Float32, numelems](fill={})
        var buffer2 = Span(memory2)

        var memory3 = Array[Float32, numelems](fill={})
        var out_buffer = Span(memory3)

        var x: Float32 = 1.0
        for i in range(numelems):
            buffer1.unsafe_ptr()[unsafe_offset=i] = 2.0
            buffer2.unsafe_ptr()[unsafe_offset=i] = x
            out_buffer.unsafe_ptr()[unsafe_offset=i] = 0.0
            x += 1.0

        @always_inline
        @__copy_capture(buffer1, buffer2, out_buffer, shape)
        @__parameter
        def func[simd_width: Int, alignment: Int = 1](idx: Coord):
            var index = rebind[IndexList[outer_rank]](coord_to_index_list(idx))
            var linear_idx = _linear_index(index, shape)
            var in1 = buffer1.unsafe_ptr().unsafe_load[width=simd_width](
                linear_idx
            )
            var in2 = buffer2.unsafe_ptr().unsafe_load[width=simd_width](
                linear_idx
            )
            out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
                linear_idx, in1 * in2
            )

        elementwise[func, simd_width=1](
            Coord(shape),
            ctx,
        )

        for i2 in range(min(numelems, 64)):
            assert_equal(
                out_buffer.unsafe_ptr().unsafe_offset(i2).unsafe_load(),
                Float32(2 * (i2 + 1)),
            )

    run_elementwise[16, 1, Index(16)](ctx)
    run_elementwise[16, 2, Index(4, 4)](ctx)
    run_elementwise[16, 3, Index(4, 2, 2)](ctx)
    run_elementwise[32, 4, Index(4, 2, 2, 2)](ctx)
    run_elementwise[32, 5, Index(4, 2, 1, 2, 2)](ctx)
    run_elementwise[131072, 2, Index(1024, 128)](ctx)


def test_elementwise_implicit_runtime() raises:
    var ctx = DeviceContext(api="cpu")
    var vector_stack = Array[Int, 20](fill={})
    var vector = Span(vector_stack)

    for i in range(len(vector)):
        vector.unsafe_ptr()[unsafe_offset=i] = Int(i)

    @always_inline
    @__copy_capture(vector)
    @__parameter
    def func[simd_width: Int, alignment: Int = 1](idx: Coord):
        vector.unsafe_ptr()[unsafe_offset=idx[0].value()] = 42

    elementwise[func, simd_width=1](20, ctx)

    for i in range(len(vector)):
        assert_equal(vector.unsafe_ptr()[unsafe_offset=i], 42)


def test_indices_conversion() raises:
    var shape = IndexList[4](3, 4, 5, 6)
    assert_equal(
        _get_start_indices_of_nth_subvolume[0](10, shape),
        IndexList[4](0, 0, 1, 4),
    )
    assert_equal(
        _get_start_indices_of_nth_subvolume[1](10, shape),
        IndexList[4](0, 2, 0, 0),
    )
    assert_equal(
        _get_start_indices_of_nth_subvolume[2](10, shape),
        IndexList[4](2, 2, 0, 0),
    )
    assert_equal(
        _get_start_indices_of_nth_subvolume[3](2, shape),
        IndexList[4](2, 0, 0, 0),
    )
    assert_equal(
        _get_start_indices_of_nth_subvolume[4](0, shape),
        IndexList[4](0, 0, 0, 0),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
