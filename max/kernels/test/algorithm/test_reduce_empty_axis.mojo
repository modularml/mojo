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
"""CPU half of the empty-reduce-axis regression.

`algorithm.gpu.rowwise` and `algorithm.cpu.rowwise`'s `launch` used to guard
on `total_size == 0 or axis_size == 0` and return before writing anything.
That guard is wrong for a reduce-shaped body (one output per row): an
empty *axis* still leaves `product(shape \\ axis)` outputs that must be
written with the monoid identity, and `total_size` (the flattened
product, axis included) is `0` right along with `axis_size` even when
the row count itself is not.

On CPU the symptom is an unwritten (uninitialized) output; the GPU
sibling (`gpu/algorithm/test_reduce_empty_axis.mojo`) covers the nastier
half — a *stale* value surviving from a prior launch into the recycled
output buffer. Both outputs here are pre-poisoned with a value no
monoid identity can produce, so a skipped write is caught the same way
an uninitialized one would be.

Covers both an inner-axis (`reduce_dim == rank - 1`) and a non-inner-axis
empty reduction: `cpu.rowwise.launch` sizes its work differently in each
branch (`num_outputs` directly on the inner path, `num_outputs //
inner_dim` for the non-inner slice count), so exercising only one would
leave the other unverified. Rank 1 gets its own case: it is the only
shape whose output count comes entirely from the empty product in
`_num_outputs_excluding_axis` (no dims left to multiply once `axis` is
excluded), and it is what `ops.sum` on a whole empty tensor lowers to.
"""

from algorithm.reductions import (
    reduce_argmax,
    reduce_max,
    reduce_mean,
    reduce_sum,
)
from std.math import isnan
from std.testing import TestSuite, assert_equal, assert_true
from std.utils.coord import Coord
from std.utils.index import Index
from std.utils.numerics import min_finite

comptime _POISON = Float32(111)
"""Never produced by any monoid identity here (`sum`'s `0`, `max`'s
`min_finite`), so it can only survive in the output if `launch` skipped
the write."""


def _run_reduce_sum_inner[
    dtype: DType = DType.float32
](num_rows: Int, axis_size: Int) raises -> List[Scalar[dtype]]:
    """`reduce_sum` over a `[num_rows, axis_size]` input, axis=1 (inner)."""
    var input_buf = List(length=num_rows * axis_size, fill=Scalar[dtype](2))
    var input_ptr = input_buf.unsafe_ptr()
    var output_buf = List(length=num_rows, fill=Scalar[dtype](_POISON))
    var output_ptr = output_buf.unsafe_ptr()

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var input_ptr, var axis_size} -> SIMD[dtype, width]:
        return input_ptr.unsafe_load[width=width](
            Int(coords[0].value()) * axis_size + Int(coords[1].value())
        )

    @always_inline
    def output_fn[
        width: SIMDLength
    ](coords: Coord, val: SIMD[dtype, width]) {var output_ptr}:
        output_ptr.unsafe_store[width=width](Int(coords[0].value()), val)

    reduce_sum[dtype, target="cpu", reduce_dim=1](
        input_fn, output_fn, Coord(Index(num_rows, axis_size))
    )
    return output_buf^


def _run_reduce_max_non_inner[
    dtype: DType = DType.float32
](axis_size: Int, num_rows: Int) raises -> List[Scalar[dtype]]:
    """`reduce_max` over an `[axis_size, num_rows]` input, axis=0
    (non-inner — the tiled/cooperative tiers, not the warp tier)."""
    var input_buf = List(length=axis_size * num_rows, fill=Scalar[dtype](2))
    var input_ptr = input_buf.unsafe_ptr()
    var output_buf = List(length=num_rows, fill=Scalar[dtype](_POISON))
    var output_ptr = output_buf.unsafe_ptr()

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var input_ptr, var num_rows} -> SIMD[dtype, width]:
        return input_ptr.unsafe_load[width=width](
            Int(coords[0].value()) * num_rows + Int(coords[1].value())
        )

    @always_inline
    def output_fn[
        width: SIMDLength
    ](coords: Coord, val: SIMD[dtype, width]) {var output_ptr}:
        output_ptr.unsafe_store[width=width](Int(coords[1].value()), val)

    reduce_max[dtype, target="cpu", reduce_dim=0](
        input_fn, output_fn, Coord(Index(axis_size, num_rows))
    )
    return output_buf^


def _run_reduce_argmax_inner[
    dtype: DType = DType.float32
](num_rows: Int, axis_size: Int) raises -> List[Int64]:
    """`reduce_argmax` over a `[num_rows, axis_size]` input, axis=1."""
    var input_buf = List(length=num_rows * axis_size, fill=Scalar[dtype](2))
    var input_ptr = input_buf.unsafe_ptr()
    var output_buf = List(length=num_rows, fill=Int64(-1))
    var output_ptr = output_buf.unsafe_ptr()

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var input_ptr, var axis_size} -> SIMD[dtype, width]:
        return input_ptr.unsafe_load[width=width](
            Int(coords[0].value()) * axis_size + Int(coords[1].value())
        )

    @always_inline
    def output_fn[
        width: SIMDLength
    ](coords: Coord, val: SIMD[.int64, width]) {var output_ptr}:
        output_ptr.unsafe_store[width=width](Int(coords[0].value()), val)

    reduce_argmax[dtype, target="cpu", reduce_dim=1](
        input_fn, output_fn, Coord(Index(num_rows, axis_size))
    )
    return output_buf^


def _run_reduce_mean_inner_f32[
    dtype: DType = DType.float32
](num_rows: Int, axis_size: Int) raises -> List[Scalar[dtype]]:
    """`reduce_mean` (float) over a `[num_rows, axis_size]` input, axis=1."""
    var input_buf = List(length=num_rows * axis_size, fill=Scalar[dtype](2))
    var input_ptr = input_buf.unsafe_ptr()
    var output_buf = List(length=num_rows, fill=Scalar[dtype](_POISON))
    var output_ptr = output_buf.unsafe_ptr()

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var input_ptr, var axis_size} -> SIMD[dtype, width]:
        return input_ptr.unsafe_load[width=width](
            Int(coords[0].value()) * axis_size + Int(coords[1].value())
        )

    @always_inline
    def output_fn[
        width: SIMDLength
    ](coords: Coord, val: SIMD[dtype, width]) {var output_ptr}:
        output_ptr.unsafe_store[width=width](Int(coords[0].value()), val)

    reduce_mean[dtype, target="cpu", reduce_dim=1](
        input_fn, output_fn, Coord(Index(num_rows, axis_size))
    )
    return output_buf^


def _run_reduce_mean_inner_i32[
    dtype: DType = DType.int32
](num_rows: Int, axis_size: Int) raises -> List[Scalar[dtype]]:
    """`reduce_mean` (integer) over a `[num_rows, axis_size]` input,
    axis=1 — the dtype with no NaN to signal "no data" with."""
    var input_buf = List(length=num_rows * axis_size, fill=Scalar[dtype](2))
    var input_ptr = input_buf.unsafe_ptr()
    var output_buf = List(length=num_rows, fill=Scalar[dtype](111))
    var output_ptr = output_buf.unsafe_ptr()

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var input_ptr, var axis_size} -> SIMD[dtype, width]:
        return input_ptr.unsafe_load[width=width](
            Int(coords[0].value()) * axis_size + Int(coords[1].value())
        )

    @always_inline
    def output_fn[
        width: SIMDLength
    ](coords: Coord, val: SIMD[dtype, width]) {var output_ptr}:
        output_ptr.unsafe_store[width=width](Int(coords[0].value()), val)

    reduce_mean[dtype, target="cpu", reduce_dim=1](
        input_fn, output_fn, Coord(Index(num_rows, axis_size))
    )
    return output_buf^


def _run_reduce_sum_rank1[
    dtype: DType = DType.float32
](axis_size: Int) raises -> List[Scalar[dtype]]:
    """`reduce_sum` over a rank-1 `[axis_size]` input, axis=0. The output
    is a single value: excluding `axis` leaves no dims to multiply, so the
    row count is the empty product, `1`."""
    var input_buf = List(length=axis_size, fill=Scalar[dtype](2))
    var input_ptr = input_buf.unsafe_ptr()
    var output_buf = List(length=1, fill=Scalar[dtype](_POISON))
    var output_ptr = output_buf.unsafe_ptr()

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var input_ptr} -> SIMD[dtype, width]:
        return input_ptr.unsafe_load[width=width](Int(coords[0].value()))

    @always_inline
    def output_fn[
        width: SIMDLength
    ](coords: Coord, val: SIMD[dtype, width]) {var output_ptr}:
        output_ptr.unsafe_store[width=width](0, val)

    reduce_sum[dtype, target="cpu", reduce_dim=0](
        input_fn, output_fn, Coord(Index(axis_size))
    )
    return output_buf^


def test_reduce_sum_empty_axis_inner() raises:
    """`sum` over an empty inner axis: identity `0`, one per (nonzero)
    row. `(2, 0)` reduced on axis -1 is the shape the original report hit."""
    var out = _run_reduce_sum_inner(num_rows=2, axis_size=0)
    assert_equal(len(out), 2)
    assert_equal(out[0], Float32(0))
    assert_equal(out[1], Float32(0))


def test_reduce_sum_empty_axis_inner_many_rows() raises:
    """Enough rows that the cooperative tier chunks them across more than
    one worker, still at `axis_size == 0`."""
    var out = _run_reduce_sum_inner(num_rows=257, axis_size=0)
    for i in range(257):
        assert_equal(out[i], Float32(0), String("row ", i))


def test_reduce_max_empty_axis_non_inner() raises:
    """`max` over an empty *non-inner* axis: the ambiguous monoid (numpy
    raises; this codebase already defines an identity, `ReduceMax`'s
    `min_finite` — see the GPU sibling's module docstring for why every
    monoid keeps its identity here). Non-inner exercises
    `cpu.rowwise.launch`'s `slice_size` division, untouched by the
    inner-axis test above.
    """
    var out = _run_reduce_max_non_inner(axis_size=0, num_rows=3)
    assert_equal(len(out), 3)
    comptime identity = min_finite[.float32]()
    for i in range(3):
        assert_equal(out[i], identity, String("row ", i))


def test_reduce_argmax_empty_axis_inner() raises:
    """`argmax` over an empty axis: no candidate ever wins, so the
    `ArgMax` monoid's `Int64.MAX` identity index gets clamped to `0` by
    `_in_range_index` — the same contract `test_arg_reduce_nan.mojo`
    already establishes for an all-NaN row (`_in_range_index`'s own
    docstring names "an empty row" as one of the cases it exists for)."""
    var out = _run_reduce_argmax_inner(num_rows=2, axis_size=0)
    assert_equal(len(out), 2)
    assert_equal(out[0], Int64(0))
    assert_equal(out[1], Int64(0))


def test_reduce_mean_empty_axis_float_is_nan() raises:
    """`mean` is `sum / axis_size` — `0 / 0` for an empty axis. For a
    floating `dtype` that's the IEEE-754 `0 * inf = NaN` `numpy.mean`
    itself reports (with a warning) for an empty reduction; no dedicated
    fix needed here, just the `launch` guard reaching this `write` at
    all."""
    var out = _run_reduce_mean_inner_f32(num_rows=2, axis_size=0)
    assert_equal(len(out), 2)
    assert_true(isnan(out[0]), "row 0")
    assert_true(isnan(out[1]), "row 1")


def test_reduce_mean_empty_axis_int_is_zero() raises:
    """Integer `mean` has no NaN to signal "no data" with, and dividing by
    the empty axis size is undefined behavior — `reduce_mean`'s integer
    branch substitutes a nonzero divisor when `axis_size == 0`, which is
    exact (not a guess) because the numerator is already `0` (the
    `ReduceSum` identity). That lands on `0`, the identity `sum` reports
    for the same shape. The GPU sibling covers the device divide, where
    the substitution is what keeps this off a `pop.div` by zero."""
    var out = _run_reduce_mean_inner_i32(num_rows=2, axis_size=0)
    assert_equal(len(out), 2)
    assert_equal(out[0], Int32(0))
    assert_equal(out[1], Int32(0))


def test_reduce_sum_rank1_empty() raises:
    """Rank 1, `(0,)`: exactly one identity, from the empty product. The
    old guard returned on `total_size == 0` and left the single output
    unwritten — the shape `ops.sum` over a whole empty tensor
    produces."""
    var out = _run_reduce_sum_rank1(axis_size=0)
    assert_equal(len(out), 1)
    assert_equal(out[0], Float32(0))


def test_reduce_sum_rank1_nonempty_unchanged() raises:
    """The same rank-1 path at a real axis size, so the empty case above
    is pinned against a working baseline rather than a vacuous one."""
    var out = _run_reduce_sum_rank1(axis_size=4)
    assert_equal(len(out), 1)
    assert_equal(out[0], Float32(8))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
