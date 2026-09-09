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

"""Tests for `DevicePointer` arithmetic, bounds checking, and origin casts.

These tests exercise the host-side offset math and origin/mutability casts on
`DevicePointer` without touching the device. They run against any
`DeviceContext` backend, including `api="cpu"`, so no GPU is required.

A `DevicePointer` keeps its `DeviceBuffer` alive for as long as the pointer is
live, so tests can use a pointer freely after the buffer's last *textual* use
without manually extending the buffer's lifetime;
`test_pointer_outlives_buffer_last_use` is the regression guard for that
property.
"""

from asyncrt_test_utils import create_test_device_context
from max.gpu.host import DevicePointer
from std.testing import (
    TestSuite,
    assert_equal,
    assert_raises,
    assert_true,
    assert_false,
)


comptime _LENGTH = 64


# ===-----------------------------------------------------------------------===#
# Acquisition
# ===-----------------------------------------------------------------------===#


def test_device_ptr_starts_at_offset_zero() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    assert_equal(p.offset(), 0)


def test_device_ptr_accepts_length_minus_1() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = DevicePointer(buf, _LENGTH - 1)
    assert_equal(p.offset(), _LENGTH - 1)


def test_device_ptr_preserves_buffer_size() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    assert_equal(len(p.buffer()), _LENGTH)


def test_init_zero_size_buffer_raises() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](0)
    with assert_raises(contains="size of DeviceBuffer must not be 0"):
        _ = DevicePointer(buf)


def test_init_zero_size_buffer_with_offset_raises() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](0)
    with assert_raises(contains="invalid DeviceBuffer of size '0'"):
        _ = DevicePointer(buf, 0)


def test_init_negative_offset_raises() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    with assert_raises(contains="invalid offset '-1'"):
        _ = DevicePointer(buf, -1)


def test_init_offset_equal_to_size_raises() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    with assert_raises(contains="invalid offset"):
        _ = DevicePointer(buf, _LENGTH)


def test_init_offset_past_end_raises() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    with assert_raises(contains="invalid offset"):
        _ = DevicePointer(buf, _LENGTH + 1)


# ===-----------------------------------------------------------------------===#
# Lifetime
# ===-----------------------------------------------------------------------===#


def test_pointer_outlives_buffer_last_use() raises:
    """Regression guard: a `DevicePointer` keeps its `DeviceBuffer` alive.

    `buf`'s last *textual* use is the `device_ptr()` call, yet the pointer is
    dereferenced afterwards (`p + 8` reads the buffer size, `unsafe_ptr()` reads
    its address). The borrow checker must extend `buf`'s lifetime past these
    uses; otherwise ASAP destruction would free `buf` while the pointer is
    still in use, producing a heap-use-after-free (caught by ASAN).
    """
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    var q = p + 8
    assert_equal(q.offset(), 8)
    assert_equal(len(p.buffer()), _LENGTH)
    assert_true(p.unsafe_ptr() == buf.unsafe_ptr().as_unsafe_any_origin())


# ===-----------------------------------------------------------------------===#
# Forward arithmetic: __add__
# ===-----------------------------------------------------------------------===#


def test_add_advances_offset() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    var q = p + 8
    assert_equal(q.offset(), 8)
    # Original is unchanged.
    assert_equal(p.offset(), 0)


def test_add_zero_is_noop() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    var q = p + 0
    assert_equal(q.offset(), 0)


def test_add_chained() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    var q = (p + 4) + 4
    assert_equal(q.offset(), 8)


def test_add_negative_offsets_backward() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    # Move forward, then add a negative.
    var q = p + 10
    var r = q + -3
    assert_equal(r.offset(), 7)


def test_add_below_zero_raises() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    with assert_raises(contains="invalid offset"):
        _ = p + -1


def test_add_past_end_raises() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    with assert_raises(contains="invalid offset"):
        _ = p + _LENGTH


# ===-----------------------------------------------------------------------===#
# Backward arithmetic: __sub__
# ===-----------------------------------------------------------------------===#


def test_sub_decreases_offset() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    var q = p + 16
    var r = q - 4
    assert_equal(r.offset(), 12)


def test_sub_zero_is_noop() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    var q = (p + 8) - 0
    assert_equal(q.offset(), 8)


def test_sub_round_trip() raises:
    """`(p + n) - n` returns to the original offset."""
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    var q = (p + 16) - 16
    assert_equal(q.offset(), p.offset())


def test_sub_negative_advances_forward() raises:
    """`p - (-n)` is equivalent to `p + n`."""
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    var q = p - -8
    assert_equal(q.offset(), 8)


def test_sub_below_zero_raises() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    with assert_raises(contains="invalid offset"):
        _ = p - 1


def test_sub_negative_past_end_raises() raises:
    """`p - (-n)` is equivalent to `p + n`."""
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    with assert_raises(contains="invalid offset"):
        var q = p - -_LENGTH


# ===-----------------------------------------------------------------------===#
# In-place arithmetic: __iadd__ / __isub__
# ===-----------------------------------------------------------------------===#


def test_iadd_mutates_offset() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    p += 12
    assert_equal(p.offset(), 12)


def test_isub_mutates_offset() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    p += 20
    p -= 5
    assert_equal(p.offset(), 15)


def test_iadd_past_end_raises() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    with assert_raises(contains="invalid offset"):
        p += _LENGTH + 1
    # After a failed in-place op, offset should be unchanged.
    assert_equal(p.offset(), 0)


def test_isub_below_zero_raises() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    p += 4
    with assert_raises(contains="invalid offset"):
        p -= 5
    # After a failed in-place op, offset should be unchanged.
    assert_equal(p.offset(), 4)


# ===-----------------------------------------------------------------------===#
# Comparison ordering
# ===-----------------------------------------------------------------------===#


def test_equality_same_buffer_same_offset() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    var q = buf.device_ptr()
    assert_true(p == q)
    assert_false(p != q)


def test_equality_same_buffer_different_offset() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    var q = buf.device_ptr() + 4
    assert_false(p == q)
    assert_true(p != q)


def test_equality_different_buffer() raises:
    var ctx = create_test_device_context()
    var abuf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var a = abuf.device_ptr()
    var bbuf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var b = bbuf.device_ptr()
    assert_false(a == b)
    assert_true(a != b)


def test_equality_buffer_copies_same_allocation() raises:
    """Pointers from two `DeviceBuffer` copies of one allocation compare equal.

    `DeviceBuffer` is refcounted: a copy names the *same* device allocation but
    is a distinct host object at a different address. Equality and ordering are
    defined by the underlying C++ handle, so pointers borrowed from the two
    copies compare equal and order without raising. This guards against
    regressing to host-object-address identity.
    """
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var buf_copy = buf  # Same handle, distinct host object.
    var p = buf.device_ptr()
    var q = buf_copy.device_ptr()
    assert_true(p == q)
    assert_false(p != q)
    # Ordering across the copies shares the allocation, so it must not raise.
    assert_true(p <= q)
    assert_true(p >= q)
    assert_false(p < q)
    assert_false(p > q)


def test_ordering_same_buffer() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    var q = buf.device_ptr() + 4
    assert_true(p < q)
    assert_true(p <= q)
    assert_true(q > p)
    assert_true(q >= p)


def test_lt_cross_buffer_raises() raises:
    var ctx = create_test_device_context()
    var buf_a = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var buf_b = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf_a.device_ptr()
    var q = buf_b.device_ptr()
    with assert_raises(contains="DeviceBuffer does not match"):
        _ = p < q
    with assert_raises(contains="DeviceBuffer does not match"):
        _ = p <= q
    with assert_raises(contains="DeviceBuffer does not match"):
        _ = q > p
    with assert_raises(contains="DeviceBuffer does not match"):
        _ = q >= p


# ===-----------------------------------------------------------------------===#
# Raw pointer access (on backends that expose it)
# ===-----------------------------------------------------------------------===#


def test_unsafe_ptr_matches_buffer_at_offset_zero() raises:
    """At offset 0, `unsafe_ptr()` should resolve to the same address as
    `DeviceBuffer.unsafe_ptr()`. The CPU backend exposes raw pointers."""
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    assert_equal(p.unsafe_ptr(), buf.unsafe_ptr().as_unsafe_any_origin())


def test_unsafe_ptr_advances_by_offset() raises:
    """`(p + n).unsafe_ptr()` must point `n` elements past `p.unsafe_ptr()`."""
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr()
    var q = p + 8
    assert_true(q.unsafe_ptr() == Pointer(p.unsafe_ptr()).unsafe_offset(8))


# ===-----------------------------------------------------------------------===#
# Origin and mutability casts
# ===-----------------------------------------------------------------------===#
#
# The casts are compile-time reinterprets, so the type system is what checks
# them: every `_offset_of_imm` / `_offset_of_mut` call and every annotated
# result type below fails to compile if its cast did not take effect.


def _offset_of_imm(p: DevicePointer[mut=False, DType.float32, _]) -> Int:
    return p.offset()


def _offset_of_mut(p: DevicePointer[mut=True, DType.float32, _]) -> Int:
    return p.offset()


def test_as_imm_yields_immutable_borrow() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr() + 4
    var q = p.as_imm()
    assert_equal(_offset_of_imm(q), 4)
    assert_equal(len(q.buffer()), _LENGTH)
    assert_true(q == p)


def test_unsafe_mut_cast_restores_mutability() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr().as_imm() + 16
    assert_equal(_offset_of_mut(p.unsafe_mut_cast[True]()), 16)


def test_unsafe_origin_cast_retargets_origin() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr() + 4
    var q: DevicePointer[
        DType.float32, MutUntrackedOrigin
    ] = p.unsafe_origin_cast[MutUntrackedOrigin]()
    assert_equal(q.offset(), 4)
    assert_true(q == p)


def test_as_unsafe_any_origin_discards_origin() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = buf.device_ptr() + 8
    var q: DevicePointer[
        DType.float32, MutUnsafeAnyOrigin
    ] = p.as_unsafe_any_origin()
    assert_equal(q.offset(), 8)
    assert_true(q == p)


# ===-----------------------------------------------------------------------===#
# Writable
# ===-----------------------------------------------------------------------===#


def test_write_to() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var p = DevicePointer(buf, 5)
    var expected = String(
        t"DevicePointer[{DType.float32}](buffer=DeviceBuffer(size={len(buf)}),"
        t" offset=5)"
    )
    assert_equal(String(p), expected)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
