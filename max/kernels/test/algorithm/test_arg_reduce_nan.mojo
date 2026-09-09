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
"""NaN behaviour of the `ArgMax`/`ArgMin` reduction monoids.

These monoids back the `mo.reduce.arg_max` / `mo.reduce.arg_min` graph ops
(`algorithm.reductions.reduce_argmax` emits `acc_indices` straight into the
output tensor), so an out-of-range index here reaches consumers as real tensor
data. A downstream `gather` on such an index is an out-of-bounds device access
that kills the process, so the index must always land in `[0, axis_size)`.

The contract mirrors the one `nn/argmaxmin_gpu` already documents and tests:
a NaN never compares greater, so it is skipped, and a row with no valid
candidate reports index 0.

The `test_arg*_splitk_*` cases cover the same index contract through
`algorithm.cpu.rowwise`'s split-axis tier, where a large row fans out
across multiple workers and their partials are merged.
"""

from algorithm.cpu.rowwise import _CPU_SPLITK_MIN_AXIS
from algorithm.reduce_op import ArgMax, ArgMin
from algorithm.reductions import reduce_argmax, reduce_argmin
from max.algorithm.backend.cpu.parallelize import _get_num_workers
from std.testing import TestSuite, assert_equal, assert_true
from std.utils.coord import Coord
from std.utils.index import Index
from std.utils.numerics import nan


def _argmax_index(values: List[Float32]) -> Int:
    var acc = ArgMax[.float32, 1]()
    for i in range(len(values)):
        acc.accumulate[.float32, 1](Float32(values[i]), Int64(i))
    return Int(acc.reduce().acc_indices[0])


def _argmin_index(values: List[Float32]) -> Int:
    var acc = ArgMin[.float32, 1]()
    for i in range(len(values)):
        acc.accumulate[.float32, 1](Float32(values[i]), Int64(i))
    return Int(acc.reduce().acc_indices[0])


comptime _SPLITK_AXIS_SIZE = 1 << 21
"""Clears `_CPU_SPLITK_MIN_AXIS`, so a single row this long fans out across
multiple CPU workers (`algorithm.cpu.rowwise`'s split-K tier) whenever the
host has at least 4 logical cores."""

comptime _SPLITK_ROW_SHAPE = Index(1, _SPLITK_AXIS_SIZE)

comptime _SPLITK_WINNER_IDX = 1_000_000
"""Split-K requires >= 4 workers to fire at all (`num_outputs * 4 <=
num_workers` at `num_outputs == 1`), which caps the per-worker stripe at
`ceildiv(_SPLITK_AXIS_SIZE, 4) == 524_288` elements even on the smallest
qualifying worker pool. This index always lands past the first stripe,
regardless of how many workers the host actually has."""


def _assert_reaches_splitk_tier() raises:
    """Fails loudly instead of silently exercising the cooperative tier
    when the host's worker pool is too small to reach split-K."""
    comptime assert (
        _SPLITK_AXIS_SIZE >= _CPU_SPLITK_MIN_AXIS
    ), "_SPLITK_AXIS_SIZE must clear the split-K axis threshold"
    assert_true(
        _get_num_workers(_SPLITK_AXIS_SIZE) >= 4,
        (
            "host worker pool has fewer than 4 threads, too small to reach the"
            " CPU split-K tier"
        ),
    )


def _run_splitk_arg[
    is_max: Bool, dtype: DType = DType.float32
](mut row: List[Scalar[dtype]]) raises -> Int:
    """Drives `reduce_argmax` (`is_max`) or `reduce_argmin` over `row` as a
    single `[1, len(row)]` row. `input_fn`/`output_fn` read/write straight
    off `List`'s own buffer: the row is contiguous and the output is one
    `Int64`, so no tensor view is needed."""
    var input_ptr = row.unsafe_ptr()
    var output_row = List(length=1, fill=Int64(0))
    var output_ptr = output_row.unsafe_ptr()

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {input_ptr} -> SIMD[dtype, width]:
        return input_ptr.unsafe_load[width=width](Int(coords[1].value()))

    @always_inline
    def output_fn[
        width: SIMDLength
    ](coords: Coord, val: SIMD[.int64, width]) {output_ptr}:
        output_ptr.unsafe_store[width=width](val)

    comptime if is_max:
        reduce_argmax[dtype, target="cpu", reduce_dim=1](
            input_fn, output_fn, Coord(_SPLITK_ROW_SHAPE)
        )
    else:
        reduce_argmin[dtype, target="cpu", reduce_dim=1](
            input_fn, output_fn, Coord(_SPLITK_ROW_SHAPE)
        )
    return Int(output_row[0])


def test_argmax_plain() raises:
    assert_equal(_argmax_index([1.0, 3.0, 2.0]), 1)


def test_argmax_ties_keep_lowest_index() raises:
    assert_equal(_argmax_index([2.0, 2.0, 1.0]), 0)


def test_argmax_skips_trailing_nan() raises:
    # The NaN arrives after the winner, so it must not displace it. Before the
    # fix the `le` compare took the NaN, and the `eq` in `reduce` then matched
    # no lane, emitting the `Int64.MAX` identity.
    var nan_f32 = nan[.float32]()
    assert_equal(_argmax_index([1.0, 2.0, nan_f32]), 1)


def test_argmax_all_nan_reports_zero() raises:
    var nan_f32 = nan[.float32]()
    assert_equal(_argmax_index([nan_f32, nan_f32, nan_f32]), 0)


def test_argmin_plain() raises:
    assert_equal(_argmin_index([3.0, 1.0, 2.0]), 1)


def test_argmin_skips_trailing_nan() raises:
    var nan_f32 = nan[.float32]()
    assert_equal(_argmin_index([2.0, 1.0, nan_f32]), 1)


def test_argmin_all_nan_reports_zero() raises:
    var nan_f32 = nan[.float32]()
    assert_equal(_argmin_index([nan_f32, nan_f32, nan_f32]), 0)


def test_argmax_splitk_winner_in_non_first_split() raises:
    """Places the winner outside the first split, so a cross-worker merge
    that finalizes partials too early or corrupts a partial slot reports
    a per-split index instead of the true one."""
    _assert_reaches_splitk_tier()
    var row = List(length=_SPLITK_AXIS_SIZE, fill=Float32(0))
    row[_SPLITK_WINNER_IDX] = 1.0
    assert_equal(_run_splitk_arg[is_max=True](row), _SPLITK_WINNER_IDX)


def test_argmin_splitk_winner_in_non_first_split() raises:
    """See `test_argmax_splitk_winner_in_non_first_split`."""
    _assert_reaches_splitk_tier()
    var row = List(length=_SPLITK_AXIS_SIZE, fill=Float32(0))
    row[_SPLITK_WINNER_IDX] = -1.0
    assert_equal(_run_splitk_arg[is_max=False](row), _SPLITK_WINNER_IDX)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
