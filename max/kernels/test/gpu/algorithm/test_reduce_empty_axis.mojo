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
"""GPU half of the empty-reduce-axis regression — the nastier failure mode.

`algorithm.gpu.rowwise.launch` used to guard on `num_rows == 0 or
row_size == 0` and return before writing anything. That guard is wrong
for a reduce-shaped body (one output per row): an empty *axis* still
leaves `product(shape \\ axis)` outputs that must be written with the
monoid identity.

Nothing faulted along the way, which is why this stayed hidden. `num_rows`
came from `flattened_length() // row_size`, and Mojo's `//` inserts a
zero-guard, so the `0 // 0` yielded `0` rather than trapping — the same
`0` a shape with genuinely no rows produces. The guard could not tell the
two apart and took the silent branch.

On GPU, skipping the write is worse than on CPU (`test_reduce_empty_axis.mojo`
in `test/algorithm/`, which only needs to show an unwritten output):
output buffers are recycled across launches, so the value left behind is
whatever the *previous* launch on that buffer wrote — a plausible,
non-obviously-wrong number that also changes run to run. Each test here
reproduces that exact shape: launch once with a real (nonzero) axis so
the output buffer holds a real, distinguishable result, then relaunch on
the *same* output buffer with an empty axis and require the stale value
to have been replaced by the identity, not merely absent from a fresh
buffer.

Ambiguous-monoid note (`max`/`argmax`): `numpy.amax` raises on an empty
axis; this codebase does not special-case it. Every `ReduceOp` monoid's
`__init__` is a required, unconditional identity (`ReduceMax`:
`min_finite`; `ArgMax`: `Int64.MAX`, clamped to index `0` by
`_in_range_index` — the same clamp `test_arg_reduce_nan.mojo` already
exercises for an all-NaN row, since "no candidate ever won" covers both
cases identically), and `rowwise.launch` runs one generic code path for
every monoid. Special-casing max/min/argmax/argmin to raise instead would
mean a second code path (a host-side shape check ahead of dispatch) that
`sum`/`product`/`mean` don't need — more surface for exactly the kind of
skip-the-write bug this file guards against, for a case (an
empty-batch-dependent shape reaching a raise mid-request) that is worse
in a serving context than a defined sentinel value. So: same identity
rule for every monoid, on both backends.

Tier coverage: at `row_size == 0` an inner-axis reduce always lands on the
warp tier (`row_size <= warp_max` holds trivially), so the inner-axis
cases here exercise `_WarpKernel` and nothing else. The non-inner cases
exist to reach the other two kernels that recompute the row count on
device — the cooperative `_BlockKernel` at few output columns, the tiled
tier at many.
"""

from std.sys import align_of

from algorithm.reductions import (
    reduce_argmax,
    reduce_max,
    reduce_mean,
    reduce_sum,
)
from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext
from std.math import isnan
from std.testing import TestSuite, assert_equal, assert_true
from std.utils.index import Index
from std.utils.numerics import min_finite


def _test_reduce_sum_stale_buffer_gpu[dtype: DType = DType.float32]() raises:
    """The originally reported pattern for `sum`: real axis, then empty
    axis, same output buffer."""
    with DeviceContext() as ctx:
        comptime num_rows = 2
        comptime real_axis_size = 4

        var in_h = ctx.enqueue_create_host_buffer[dtype](
            num_rows * real_axis_size
        )
        var out_h = ctx.enqueue_create_host_buffer[dtype](num_rows)
        ctx.synchronize()
        for i in range(num_rows * real_axis_size):
            in_h[i] = 2

        var in_d = ctx.enqueue_create_buffer[dtype](num_rows * real_axis_size)
        var out_d = ctx.enqueue_create_buffer[dtype](num_rows)
        ctx.enqueue_copy(in_d, in_h)

        var in_buf = TileTensor(
            in_d, row_major(Coord(Index(num_rows, real_axis_size)))
        )
        var out_ptr = out_d.unsafe_ptr()

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var in_buf} -> SIMD[dtype, width]:
            var idx = in_buf.layout(coords)
            return in_buf.raw_load[
                width=width, alignment=alignment * align_of[dtype]()
            ](idx)

        @always_inline
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[dtype, width]) {var out_ptr}:
            # `out_ptr` is 1-D (`num_rows` elements) while `coords` carries
            # the full input rank (`Row.emit` zeroes the axis position
            # rather than dropping it) — index by the row component
            # directly rather than routing through a rank-2 `TileTensor`
            # over a rank-1 buffer.
            #
            # `{var out_ptr}` and not `{out_ptr}`: the bare form captures by
            # reference, which hands the kernel the address of a host stack
            # slot. NVIDIA tolerates that; AMD faults on it ("Memory access
            # fault ... on address 0x7ff...", a host address), and every
            # closure in this file needs the copy.
            out_ptr.unsafe_store[width=width](Int(coords[0].value()), val)

        # Run 1: real data (all 2s) over a real axis. Sum = 2 * 4 = 8/row.
        reduce_sum[dtype, target="gpu", reduce_dim=1](
            input_fn, output_fn, Coord(Index(num_rows, real_axis_size)), ctx
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        assert_equal(out_h[0], Scalar[dtype](8))
        assert_equal(out_h[1], Scalar[dtype](8))

        # Run 2: SAME `out_d`, empty axis. Before the fix, `launch`'s
        # `flattened_length() // row_size` returned `0` (the zero-guard, not
        # a trap), the `num_rows == 0` arm read that as "no rows", and it
        # returned before writing — `out_d` would still read back `8`.
        reduce_sum[dtype, target="gpu", reduce_dim=1](
            input_fn, output_fn, Coord(Index(num_rows, 0)), ctx
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        assert_equal(out_h[0], Scalar[dtype](0))
        assert_equal(out_h[1], Scalar[dtype](0))

        _ = in_buf
        _ = in_d^
        _ = out_d^


def _test_reduce_max_stale_buffer_gpu[dtype: DType = DType.float32]() raises:
    """Same pattern for `max` — the ambiguous monoid. Identity is
    `ReduceMax`'s `min_finite`, not a raise; see the module docstring."""
    with DeviceContext() as ctx:
        comptime num_rows = 2
        comptime real_axis_size = 4

        var in_h = ctx.enqueue_create_host_buffer[dtype](
            num_rows * real_axis_size
        )
        var out_h = ctx.enqueue_create_host_buffer[dtype](num_rows)
        ctx.synchronize()
        for r in range(num_rows):
            for c in range(real_axis_size):
                in_h[r * real_axis_size + c] = Scalar[dtype](c + 1)

        var in_d = ctx.enqueue_create_buffer[dtype](num_rows * real_axis_size)
        var out_d = ctx.enqueue_create_buffer[dtype](num_rows)
        ctx.enqueue_copy(in_d, in_h)

        var in_buf = TileTensor(
            in_d, row_major(Coord(Index(num_rows, real_axis_size)))
        )
        var out_ptr = out_d.unsafe_ptr()

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var in_buf} -> SIMD[dtype, width]:
            var idx = in_buf.layout(coords)
            return in_buf.raw_load[
                width=width, alignment=alignment * align_of[dtype]()
            ](idx)

        @always_inline
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[dtype, width]) {var out_ptr}:
            # See `test_reduce_sum_stale_buffer_gpu`'s `output_fn` for why
            # this indexes `out_ptr` directly rather than through a
            # rank-mismatched `TileTensor`.
            out_ptr.unsafe_store[width=width](Int(coords[0].value()), val)

        # Run 1: real max = real_axis_size (4).
        reduce_max[dtype, target="gpu", reduce_dim=1](
            input_fn, output_fn, Coord(Index(num_rows, real_axis_size)), ctx
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        assert_equal(out_h[0], Scalar[dtype](real_axis_size))
        assert_equal(out_h[1], Scalar[dtype](real_axis_size))

        # Run 2: SAME `out_d`, empty axis. Must become the identity, not
        # the stale `4` from run 1.
        reduce_max[dtype, target="gpu", reduce_dim=1](
            input_fn, output_fn, Coord(Index(num_rows, 0)), ctx
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        comptime identity = min_finite[dtype]()
        assert_equal(out_h[0], identity)
        assert_equal(out_h[1], identity)

        _ = in_buf
        _ = in_d^
        _ = out_d^


def _test_reduce_argmax_stale_buffer_gpu[dtype: DType = DType.float32]() raises:
    """Same pattern for `argmax`. A real run picks a real winning index
    (not `0`), so a stale-buffer bug can't hide behind the identity index
    also being `0`."""
    with DeviceContext() as ctx:
        comptime out_dtype = DType.int64
        comptime num_rows = 2
        comptime real_axis_size = 4
        # Deliberately not `0`, so `out_h` reading back `0` after run 1
        # would itself be a (different) bug, not a false pass for run 2.
        comptime winner_idx = 2

        var in_h = ctx.enqueue_create_host_buffer[dtype](
            num_rows * real_axis_size
        )
        var out_h = ctx.enqueue_create_host_buffer[out_dtype](num_rows)
        ctx.synchronize()
        for i in range(num_rows * real_axis_size):
            in_h[i] = 1
        for r in range(num_rows):
            in_h[r * real_axis_size + winner_idx] = 5

        var in_d = ctx.enqueue_create_buffer[dtype](num_rows * real_axis_size)
        var out_d = ctx.enqueue_create_buffer[out_dtype](num_rows)
        ctx.enqueue_copy(in_d, in_h)

        var in_buf = TileTensor(
            in_d, row_major(Coord(Index(num_rows, real_axis_size)))
        )
        var out_ptr = out_d.unsafe_ptr()

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var in_buf} -> SIMD[dtype, width]:
            var idx = in_buf.layout(coords)
            return in_buf.raw_load[
                width=width, alignment=alignment * align_of[dtype]()
            ](idx)

        @always_inline
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[out_dtype, width]) {var out_ptr}:
            # See `test_reduce_sum_stale_buffer_gpu`'s `output_fn` for why
            # this indexes `out_ptr` directly rather than through a
            # rank-mismatched `TileTensor`.
            out_ptr.unsafe_store[width=width](Int(coords[0].value()), val)

        # Run 1: real winner at `winner_idx`.
        reduce_argmax[dtype, target="gpu", reduce_dim=1](
            input_fn, output_fn, Coord(Index(num_rows, real_axis_size)), ctx
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        assert_equal(out_h[0], Int64(winner_idx))
        assert_equal(out_h[1], Int64(winner_idx))

        # Run 2: SAME `out_d`, empty axis. Must become `0` (the
        # `_in_range_index`-clamped identity), not the stale `winner_idx`.
        reduce_argmax[dtype, target="gpu", reduce_dim=1](
            input_fn, output_fn, Coord(Index(num_rows, 0)), ctx
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        assert_equal(out_h[0], Int64(0))
        assert_equal(out_h[1], Int64(0))

        _ = in_buf
        _ = in_d^
        _ = out_d^


def _max_non_inner_real_then_empty[
    dtype: DType = DType.float32
](num_cols: Int) raises:
    """`reduce_max` over `[axis, num_cols]` on axis 0 — real axis first,
    then empty, same output buffer. `num_cols` selects the tier: a few
    columns cannot fill the tiled tier's occupancy gate and fall to
    `_BlockKernel`, many take `_TiledKernel`.

    The input is synthesized from the closure rather than a device buffer:
    the empty run reads no elements at all, and the real run only needs a
    value distinguishable from the identity.
    """
    with DeviceContext() as ctx:
        comptime real_axis_size = 4
        comptime real_value = Scalar[dtype](7)

        var out_h = ctx.enqueue_create_host_buffer[dtype](num_cols)
        var out_d = ctx.enqueue_create_buffer[dtype](num_cols)
        ctx.synchronize()
        var out_ptr = out_d.unsafe_ptr()

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {} -> SIMD[dtype, width]:
            return SIMD[dtype, width](real_value)

        @always_inline
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[dtype, width]) {var out_ptr}:
            # Non-inner axis, so the output column is the *inner* coord.
            out_ptr.unsafe_store[width=width](Int(coords[1].value()), val)

        reduce_max[dtype, target="gpu", reduce_dim=0](
            input_fn, output_fn, Coord(Index(real_axis_size, num_cols)), ctx
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        for i in range(num_cols):
            assert_equal(out_h[i], real_value, String("real col ", i))

        reduce_max[dtype, target="gpu", reduce_dim=0](
            input_fn, output_fn, Coord(Index(0, num_cols)), ctx
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        comptime identity = min_finite[dtype]()
        for i in range(num_cols):
            assert_equal(out_h[i], identity, String("empty col ", i))

        _ = out_d^


def test_reduce_max_non_inner_stale_buffer_gpu() raises:
    """Non-inner axis, few columns: the cooperative `_BlockKernel` tier."""
    _max_non_inner_real_then_empty(num_cols=3)


def test_reduce_max_non_inner_tiled_stale_buffer_gpu() raises:
    """Non-inner axis, enough columns to clear the tiled tier's occupancy
    gate, so the row count is the one `_TiledKernel` computes."""
    _max_non_inner_real_then_empty(num_cols=4096)


def _test_reduce_sum_rank1_stale_buffer_gpu[
    dtype: DType = DType.float32
]() raises:
    """Rank 1, `(0,)`: excluding `axis` leaves no dims to multiply, so the
    single output comes entirely from the empty product. This is what
    `ops.sum` over a whole empty tensor lowers to, and the old
    `num_rows == 0` guard swallowed it.
    """
    with DeviceContext() as ctx:
        comptime real_axis_size = 4
        comptime real_value = Scalar[dtype](2)

        var out_h = ctx.enqueue_create_host_buffer[dtype](1)
        var out_d = ctx.enqueue_create_buffer[dtype](1)
        ctx.synchronize()
        var out_ptr = out_d.unsafe_ptr()

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {} -> SIMD[dtype, width]:
            return SIMD[dtype, width](real_value)

        @always_inline
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[dtype, width]) {var out_ptr}:
            out_ptr.unsafe_store[width=width](0, val)

        reduce_sum[dtype, target="gpu", reduce_dim=0](
            input_fn, output_fn, Coord(Index(real_axis_size)), ctx
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        assert_equal(out_h[0], real_value * real_axis_size)

        reduce_sum[dtype, target="gpu", reduce_dim=0](
            input_fn, output_fn, Coord(Index(0)), ctx
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        assert_equal(out_h[0], Scalar[dtype](0))

        _ = out_d^


def _test_reduce_mean_int_empty_axis_gpu[dtype: DType = DType.int32]() raises:
    """Integer `mean` on device — the case `reduce_mean`'s divisor
    substitution exists for. `SIMD.__truediv__` lowers to a raw `pop.div`
    with no zero-guard, so an unsubstituted empty axis is a hardware fault
    here, not a wrong number: the CPU half of this regression cannot
    reach it, and every other case in this file goes through `sum`/`max`,
    which never divide.
    """
    with DeviceContext() as ctx:
        comptime num_rows = 2
        comptime real_axis_size = 4
        comptime real_value = Scalar[dtype](2)

        var out_h = ctx.enqueue_create_host_buffer[dtype](num_rows)
        var out_d = ctx.enqueue_create_buffer[dtype](num_rows)
        ctx.synchronize()
        var out_ptr = out_d.unsafe_ptr()

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {} -> SIMD[dtype, width]:
            return SIMD[dtype, width](real_value)

        @always_inline
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[dtype, width]) {var out_ptr}:
            out_ptr.unsafe_store[width=width](Int(coords[0].value()), val)

        # Run 1: mean of four 2s is 2, so the stale value is not the
        # identity the empty run must produce.
        reduce_mean[dtype, target="gpu", reduce_dim=1](
            input_fn, output_fn, Coord(Index(num_rows, real_axis_size)), ctx
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        assert_equal(out_h[0], real_value)
        assert_equal(out_h[1], real_value)

        reduce_mean[dtype, target="gpu", reduce_dim=1](
            input_fn, output_fn, Coord(Index(num_rows, 0)), ctx
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        assert_equal(out_h[0], Scalar[dtype](0))
        assert_equal(out_h[1], Scalar[dtype](0))

        _ = out_d^


def _test_reduce_mean_float_empty_axis_gpu[
    dtype: DType = DType.float32
]() raises:
    """Float `mean` on device relies on IEEE-754 `0 * inf = NaN` surviving
    codegen. Worth pinning separately from the CPU case: a device compiler
    that assumed no-NaN would fold this to something finite, and the
    integer branch above cannot catch that.
    """
    with DeviceContext() as ctx:
        comptime num_rows = 2
        comptime real_axis_size = 4
        comptime real_value = Scalar[dtype](2)

        var out_h = ctx.enqueue_create_host_buffer[dtype](num_rows)
        var out_d = ctx.enqueue_create_buffer[dtype](num_rows)
        ctx.synchronize()
        var out_ptr = out_d.unsafe_ptr()

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {} -> SIMD[dtype, width]:
            return SIMD[dtype, width](real_value)

        @always_inline
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[dtype, width]) {var out_ptr}:
            out_ptr.unsafe_store[width=width](Int(coords[0].value()), val)

        reduce_mean[dtype, target="gpu", reduce_dim=1](
            input_fn, output_fn, Coord(Index(num_rows, real_axis_size)), ctx
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        assert_equal(out_h[0], real_value)
        assert_equal(out_h[1], real_value)

        reduce_mean[dtype, target="gpu", reduce_dim=1](
            input_fn, output_fn, Coord(Index(num_rows, 0)), ctx
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        assert_true(isnan(out_h[0]), "row 0")
        assert_true(isnan(out_h[1]), "row 1")

        _ = out_d^


def test_reduce_sum_stale_buffer_gpu() raises:
    _test_reduce_sum_stale_buffer_gpu()


def test_reduce_max_stale_buffer_gpu() raises:
    _test_reduce_max_stale_buffer_gpu()


def test_reduce_argmax_stale_buffer_gpu() raises:
    _test_reduce_argmax_stale_buffer_gpu()


def test_reduce_sum_rank1_stale_buffer_gpu() raises:
    _test_reduce_sum_rank1_stale_buffer_gpu()


def test_reduce_mean_int_empty_axis_gpu() raises:
    _test_reduce_mean_int_empty_axis_gpu()


def test_reduce_mean_float_empty_axis_gpu() raises:
    _test_reduce_mean_float_empty_axis_gpu()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
