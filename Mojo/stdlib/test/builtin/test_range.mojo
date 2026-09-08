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

from std.testing import assert_equal, assert_false, assert_true, TestSuite
from std.testing.prop import PropTest, PropTestConfig
from std.testing.prop.strategy import SIMD


comptime DTYPES = [
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
]


# Regression test for cyclic dependency bug in MSTDL-2217
# This helper must be declared before any test function that calls it.
# The bug was triggered when a function using range(Int, Int) was declared
# before main/test functions, causing a cyclic dependency during overload
# resolution: range -> Int -> Equatable.__eq__ -> range.
def _range_with_int_params_helper(start: Int, end: Int) -> Int:
    var sum = 0
    for i in range(start, end):
        sum += i
    return sum


def test_range_with_int_params_declaration_order() raises:
    assert_equal(_range_with_int_params_helper(0, 5), 10)  # 0+1+2+3+4
    assert_equal(_range_with_int_params_helper(1, 4), 6)  # 1+2+3
    assert_equal(_range_with_int_params_helper(5, 5), 0)  # empty range


def _test_range_iter_bounds[
    I: Iterator
](var range_iter: I, len: Int) raises where conforms_to(I.Element, Deinitable):
    var iter = range_iter^

    for i in range(len):
        var lower, upper = iter.bounds()
        assert_equal(len - i, lower)
        assert_equal(len - i, upper.value())
        _ = iter.__next__()

    var lower, upper = iter.bounds()
    assert_equal(0, lower)
    assert_equal(0, upper.value())


def test_range_int_bounds() raises:
    _test_range_iter_bounds(range(0), 0)
    _test_range_iter_bounds(range(10), 10)
    _test_range_iter_bounds(range(0, 10), 10)
    _test_range_iter_bounds(range(5, 10), 5)
    _test_range_iter_bounds(range(10, 0, -1), 10)
    _test_range_iter_bounds(range(0, 10, 2), 5)
    _test_range_iter_bounds(range(0, 11, 2), 6)
    _test_range_iter_bounds(range(38, -13, -23), 3)


def test_range_uint_bounds() raises:
    _test_range_iter_bounds(range(UInt(0)), 0)
    _test_range_iter_bounds(range(UInt(10)), 10)
    _test_range_iter_bounds(range(UInt(0), UInt(10)), 10)
    _test_range_iter_bounds(range(UInt(5), UInt(10)), 5)
    _test_range_iter_bounds(range(UInt(0), UInt(10), UInt(2)), 5)
    _test_range_iter_bounds(range(UInt(0), UInt(11), UInt(2)), 6)


def _test_range_scalar_bounds[dtype: DType]() raises:
    comptime scalar = Scalar[dtype]

    _test_range_iter_bounds(range(scalar(0)), 0)
    _test_range_iter_bounds(range(scalar(10)), 10)
    _test_range_iter_bounds(range(scalar(0), scalar(10)), 10)
    _test_range_iter_bounds(range(scalar(5), scalar(10)), 5)
    _test_range_iter_bounds(range(scalar(0), scalar(10), scalar(2)), 5)
    _test_range_iter_bounds(range(scalar(0), scalar(11), scalar(2)), 6)

    comptime if dtype.is_signed():
        _test_range_iter_bounds(range(scalar(10), scalar(0), scalar(-1)), 10)
        _test_range_iter_bounds(range(scalar(38), scalar(-13), scalar(-23)), 3)


def test_range_scalar_bounds() raises:
    comptime for dtype in DTYPES:
        _test_range_scalar_bounds[dtype]()


def test_larger_than_int_max_bounds() raises:
    def test[I: Iterator](iter: I) raises:
        var lower, upper = iter.bounds()
        assert_equal(lower, Int.MAX)
        assert_false(upper)

    # UInt
    test(range(UInt.MAX))
    test(range(UInt(1), UInt.MAX))
    test(range(UInt(1), UInt.MAX, UInt(1)))

    # UInt64
    test(range(UInt64.MAX))
    test(range(UInt64(1), UInt64.MAX))
    test(range(UInt64(1), UInt64.MAX, UInt64(1)))

    # Reversing preserves the element count, so it has to clamp the same way.
    test(reversed(range(UInt(1), UInt.MAX, UInt(1))))
    test(reversed(range(UInt64(1), UInt64.MAX, UInt64(1))))


def test_range_len() raises:
    # Usual cases
    assert_equal(range(10).__len__(), 10, "len(range(10))")
    assert_equal(range(0, 10).__len__(), 10, "len(range(0, 10))")
    assert_equal(range(5, 10).__len__(), 5, "len(range(5, 10))")
    assert_equal(range(10, 0, -1).__len__(), 10, "len(range(10, 0, -1))")
    assert_equal(range(0, 10, 2).__len__(), 5, "len(range(0, 10, 2))")
    assert_equal(range(38, -13, -23).__len__(), 3, "len(range(38, -13, -23))")

    # Edge cases
    assert_equal(range(0).__len__(), 0, "len(range(0))")
    assert_equal(range(-10).__len__(), 0, "len(range(-10))")
    assert_equal(range(0, 0).__len__(), 0, "len(range(0, 0))")
    assert_equal(range(10, 0).__len__(), 0, "len(range(10, 0))")
    assert_equal(range(0, 0, 1).__len__(), 0, "len(range(0, 0, 1))")

    assert_equal(range(5, 10, -1).__len__(), 0, "len(range(5, 10, -1))")
    assert_equal(range(10, 5, 1).__len__(), 0, "len(range(10, 5, 1))")
    assert_equal(range(5, 10, -10).__len__(), 0, "len(range(5, 10, -10))")
    assert_equal(range(10, 5, 10).__len__(), 0, "len(range(10, 5, 10))")
    assert_equal(range(5, 10, 20).__len__(), 1, "len(range(5, 10, 20))")
    assert_equal(range(10, 5, -20).__len__(), 1, "len(range(10, 5, -20))")


def test_range_len_uint_maxuint() raises:
    # `__len__()` returns `Int`. A count that exceeds `Int.MAX` asserts rather
    # than clamping, so it can't be exercised here under `-D ASSERT=all`; the
    # clamped size hint for that case is covered by
    # `test_larger_than_int_max_bounds`. An empty range still reports 0 without
    # overflowing.
    assert_equal(
        range(UInt.MAX, UInt(0), UInt(1)).__len__(),
        0,
        "len(range(UInt.MAX, 0, 1))",
    )


def test_range_len_uint_empty() raises:
    assert_equal(
        range(UInt(0), UInt(0), UInt(1)).__len__(), 0, "len(range(0, 0, 1))"
    )
    assert_equal(
        range(UInt(10), UInt(10), UInt(1)).__len__(), 0, "len(range(10, 10, 1))"
    )


def test_range_len_uint() raises:
    assert_equal(range(UInt(10)).__len__(), 10, "len(range(10))")

    # start < end
    assert_equal(range(UInt(0), UInt(10)).__len__(), 10, "len(range(0, 10))")
    assert_equal(range(UInt(5), UInt(10)).__len__(), 5, "len(range(5, 10))")
    assert_equal(
        range(UInt(0), UInt(10), UInt(2)).__len__(), 5, "len(range(0, 10, 2))"
    )
    # start > end
    assert_equal(
        range(UInt(10), UInt(0), UInt(1)).__len__(), 0, "len(range(10, 0, 1))"
    )


def _test_range_len_scalar[dtype: DType]() raises:
    comptime scalar = Scalar[dtype]

    # empty
    assert_equal(range(scalar(0), scalar(0), scalar(1)).__len__(), 0)
    assert_equal(range(scalar(10), scalar(10), scalar(1)).__len__(), 0)

    # start = 0
    assert_equal(range(scalar(10)).__len__(), 10)

    # start < end
    assert_equal(range(scalar(0), scalar(10)).__len__(), 10)
    assert_equal(range(scalar(5), scalar(10)).__len__(), 5)
    assert_equal(range(scalar(0), scalar(10), scalar(2)).__len__(), 5)

    # start > end
    assert_equal(range(scalar(10), scalar(0), scalar(1)).__len__(), 0)


def test_range_len_scalar() raises:
    comptime for dtype in DTYPES:
        _test_range_len_scalar[dtype]()


def test_range_getitem() raises:
    # Usual cases
    assert_equal(range(10)[3], 3, "range(10)[3]")
    assert_equal(range(0, 10)[3], 3, "range(0, 10)[3]")
    assert_equal(range(5, 10)[3], 8, "range(5, 10)[3]")
    assert_equal(range(5, 10)[4], 9, "range(5, 10)[4]")
    assert_equal(range(10, 0, -1)[2], 8, "range(10, 0, -1)[2]")
    assert_equal(range(0, 10, 2)[4], 8, "range(0, 10, 2)[4]")
    assert_equal(range(38, -13, -23)[1], 15, "range(38, -13, -23)[1]")


def test_range_getitem_uint() raises:
    assert_equal(range(UInt(10))[3], 3, "range(10)[3]")

    assert_equal(range(UInt(0), UInt(10))[3], 3, "range(0, 10)[3]")
    assert_equal(range(UInt(5), UInt(10))[3], 8, "range(5, 10)[3]")
    assert_equal(range(UInt(5), UInt(10))[4], 9, "range(5, 10)[4]")

    # Specify the step size > 1
    assert_equal(range(UInt(0), UInt(10), UInt(2))[4], 8, "range(0, 10, 2)[4]")

    # start > end
    var bad_strided_uint_range = range(UInt(10), UInt(5), UInt(1))
    var bad_strided_uint_range_iter = bad_strided_uint_range.__iter__()
    assert_equal(UInt(0), UInt(bad_strided_uint_range_iter.__len__()))


def test_range_reversed() raises:
    # `reversed()` produces exactly the forward elements in reverse order.
    def assert_reversed_matches(start: Int, end: Int, step: Int) raises:
        var forward = List[Int]()
        for x in range(start, end, step):
            forward.append(x)
        var backward = List[Int]()
        for x in reversed(range(start, end, step)):
            backward.append(x)
        assert_equal(len(backward), len(forward))
        for i in range(len(forward)):
            assert_equal(backward[i], forward[len(forward) - 1 - i])

    # The one- and two-argument forms reverse through their own
    # `__reversed__`, so spell them out rather than routing through the
    # three-argument one.
    var zero_starting = List[Int]()
    for x in reversed(range(10)):
        zero_starting.append(x)
    assert_equal(zero_starting, [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    var sequential = List[Int]()
    for x in reversed(range(5, 10)):
        sequential.append(x)
    assert_equal(sequential, [9, 8, 7, 6, 5])

    var empty = List[Int]()
    for x in reversed(range(0)):
        empty.append(x)
    for x in reversed(range(5, 5)):
        empty.append(x)
    for x in reversed(range(10, 5)):
        empty.append(x)
    assert_equal(len(empty), 0)

    # Strided, both directions
    assert_reversed_matches(0, 10, 2)
    assert_reversed_matches(38, -13, -23)
    # Empty, both directions
    assert_reversed_matches(5, 5, 1)
    assert_reversed_matches(10, 5, 1)
    assert_reversed_matches(5, 10, -1)

    # A reversed range's sum and length match the original's.
    def test_sum_reversed(start: Int, end: Int, step: Int) raises:
        var forward = range(start, end, step)
        var ibackward = forward.__reversed__()
        assert_equal(
            forward.__len__(),
            ibackward.__len__(),
            "len(forward), len(backward)",
        )
        var forward_sum = 0
        var backward_sum = 0
        for x in forward:
            forward_sum += x
            backward_sum += ibackward.__next__()
        assert_equal(forward_sum, backward_sum, "forward_sum, backward_sum")

    # Test using loops and reversed
    for end in range(10, 13):
        test_sum_reversed(1, end, 3)

    for end in reversed(range(10, 13)):
        test_sum_reversed(20, end, -3)


def _assert_scalar_reversed_matches[
    dtype: DType, //
](start: Scalar[dtype], end: Scalar[dtype], step: Scalar[dtype]) raises:
    """Asserts `reversed()` yields the forward elements in reverse order."""
    var forward = List[Scalar[dtype]]()
    for x in range(start, end, step):
        forward.append(x)
    var backward = List[Scalar[dtype]]()
    for x in reversed(range(start, end, step)):
        backward.append(x)
    assert_equal(len(backward), len(forward))
    assert_equal(len(backward), range(start, end, step).__len__())
    for i in range(len(forward)):
        assert_equal(backward[i], forward[len(forward) - 1 - i])


def _assert_forward_terminates[
    dtype: DType, //
](start: Scalar[dtype], end: Scalar[dtype], step: Scalar[dtype]) raises:
    """Asserts forward iteration yields exactly the `len()` grid elements."""
    var expected = range(start, end, step).__len__()
    var seen = List[Scalar[dtype]]()
    for x in range(start, end, step):
        # Bail out rather than hang, and let the count assert below report it.
        if len(seen) > expected:
            break
        seen.append(x)
    assert_equal(len(seen), expected)
    for i in range(len(seen)):
        assert_equal(seen[i], start + Scalar[dtype](i) * step)


def test_range_forward_step_past_dtype_limit() raises:
    # On the last element, `start + step` can fall outside the dtype. The
    # wrapped cursor re-entered the range and iterated forever, disagreeing
    # with `len()` (MSTDL-2975).
    _assert_forward_terminates(UInt8(250), UInt8(255), UInt8(2))
    _assert_forward_terminates(UInt8(0), UInt8(255), UInt8(200))
    _assert_forward_terminates(UInt64.MAX - 1, UInt64.MAX, UInt64(5))
    # Signed overflows the same way, in both step directions.
    _assert_forward_terminates(Int8(120), Int8(127), Int8(5))
    _assert_forward_terminates(Int8(126), Int8.MAX, Int8(5))
    _assert_forward_terminates(Int8(-120), Int8(-128), Int8(-5))
    _assert_forward_terminates(Int8.MIN + 1, Int8.MIN, Int8(-5))
    _assert_forward_terminates(Int64.MAX - 1, Int64.MAX, Int64(5))
    _assert_forward_terminates(Int64.MIN + 1, Int64.MIN, Int64(-5))
    # The grid landing exactly on `end` never wrapped, and still must not.
    _assert_forward_terminates(UInt8(250), UInt8.MAX, UInt8(1))
    _assert_forward_terminates(Int8(120), Int8(125), Int8(5))
    # Reversing such a range agrees with forward now that it terminates.
    _assert_scalar_reversed_matches(UInt8(250), UInt8(255), UInt8(2))
    _assert_scalar_reversed_matches(Int8(120), Int8(127), Int8(5))


def _prop_forward_terminates[dtype: DType](var draw: SIMD[dtype, 4]) raises:
    comptime T = Scalar[dtype]
    var step = (draw[1] & 7) + 1
    # A raw offset, not a whole number of steps, so `end` usually sits off the
    # range's grid. That is what lets the cursor step over `end` instead of
    # onto it, which is the only way it can leave the dtype. Under 16 elements
    # either way, so a wrapping cursor is caught by the count rather than
    # iterating the whole dtype first.
    var span = draw[2] & 15

    # `start` roams the whole dtype, so `start + span` reaches past the limit.
    _assert_forward_terminates(draw[0], draw[0] + span, step)

    # The wrap needs the last step to cross the limit, which a uniform draw
    # reaches on `Int8` and never on `Int64`, so pin the remaining triples
    # against both bounds (MSTDL-3095).
    _assert_forward_terminates(T.MAX - span, T.MAX, step)
    _assert_forward_terminates(T.MIN, T.MIN + span, step)
    comptime if not dtype.is_unsigned():
        _assert_forward_terminates(T.MIN + span, T.MIN, -step)
        _assert_forward_terminates(T.MAX, T.MAX - span, -step)


def test_range_forward_scalar_properties() raises:
    # The seed is pinned: a property test that picks its own seed reports a
    # failure no one else can reproduce.
    comptime for dtype in DTYPES:
        PropTest(config=PropTestConfig(runs=100, seed=0)).test[
            _prop_forward_terminates[dtype]
        ](SIMD[dtype, 4].strategy())


def test_range_forward_step_past_int_limit() raises:
    # The `Indexer` overloads reach the same cursor through `DType.int`.
    var seen = List[Int]()
    for x in range(Int.MAX - 1, Int.MAX, 5):
        if len(seen) > 1:
            break
        seen.append(x)
    assert_equal(len(seen), 1)
    assert_equal(seen[0], Int.MAX - 1)

    var descending = List[Int]()
    for x in range(Int.MIN + 1, Int.MIN, -5):
        if len(descending) > 1:
            break
        descending.append(x)
    assert_equal(len(descending), 1)
    assert_equal(descending[0], Int.MIN + 1)


def test_range_forward_bounds_past_dtype_limit() raises:
    # The step off the last element wraps out of the dtype and lands the cursor
    # back inside `[start, end)`. The iterator is spent at that point, and its
    # size hint has to say so rather than count from where the cursor landed.
    _test_range_iter_bounds(range(UInt8(250), UInt8(255), UInt8(2)), 3)
    _test_range_iter_bounds(range(UInt8(0), UInt8(255), UInt8(200)), 2)
    _test_range_iter_bounds(range(Int8(120), Int8(127), Int8(5)), 2)
    _test_range_iter_bounds(range(Int8(-120), Int8(-128), Int8(-5)), 2)
    _test_range_iter_bounds(range(Int.MAX - 1, Int.MAX, 5), 1)
    _test_range_iter_bounds(range(Int.MIN + 1, Int.MIN, -5), 1)


def test_range_forward_step_past_dtype_limit_comptime() raises:
    # Unrolling the same ranges must terminate too: a cursor that wrapped back
    # into the range would hang the compiler rather than the program.
    var unsigned = List[UInt8]()
    comptime for i in range(UInt8(250), UInt8(255), UInt8(2)):
        unsigned.append(i)
    assert_equal(unsigned, [UInt8(250), UInt8(252), UInt8(254)])

    var signed = List[Int8]()
    comptime for i in range(Int8(120), Int8(127), Int8(5)):
        signed.append(i)
    assert_equal(signed, [Int8(120), Int8(125)])

    var descending = List[Int8]()
    comptime for i in range(Int8(-120), Int8(-128), Int8(-5)):
        descending.append(i)
    assert_equal(descending, [Int8(-120), Int8(-125)])


def test_range_reversed_scalar_at_dtype_bounds() raises:
    # A reversed range must not depend on `start - step` being representable:
    # these all sit within `step` of the dtype's limit (MSTDL-2973).
    _assert_scalar_reversed_matches(Int8.MIN, Int8.MIN + 8, Int8(1))
    _assert_scalar_reversed_matches(Int8.MIN, Int8.MIN + 12, Int8(3))
    _assert_scalar_reversed_matches(Int8.MAX, Int8.MAX - 8, Int8(-1))
    _assert_scalar_reversed_matches(Int16.MIN, Int16.MIN + 8, Int16(1))
    _assert_scalar_reversed_matches(Int32.MIN, Int32.MIN + 8, Int32(1))
    _assert_scalar_reversed_matches(Int64.MIN, Int64.MIN + 8, Int64(1))
    _assert_scalar_reversed_matches(Int64.MAX, Int64.MAX - 8, Int64(-1))
    _assert_scalar_reversed_matches(UInt8(0), UInt8(8), UInt8(1))
    # Empty at the limit: the last element the reversed iterator would start
    # from is itself outside the dtype, so it must never be materialized.
    _assert_scalar_reversed_matches(Int8.MIN, Int8.MIN, Int8(1))
    _assert_scalar_reversed_matches(Int8.MAX, Int8.MAX, Int8(-1))
    _assert_scalar_reversed_matches(UInt8(0), UInt8(0), UInt8(1))


def test_range_reversed_scalar_wide_span() raises:
    # `end - start` overflows the dtype, so deriving the last element by
    # snapping `end` with `%` would land the cursor off the range's grid and
    # the reverse walk would miss `start` — wandering, or never terminating.
    _assert_scalar_reversed_matches(Int8(-128), Int8(100), Int8(3))
    _assert_scalar_reversed_matches(Int8(-128), Int8(120), Int8(5))
    _assert_scalar_reversed_matches(Int8(-128), Int8(100), Int8(1))
    _assert_scalar_reversed_matches(Int8(127), Int8(-120), Int8(-3))
    _assert_scalar_reversed_matches(Int16.MIN, Int16(30000), Int16(3))


def test_range_reversed_scalar_unsigned() raises:
    # An unsigned strided range is reversible; its step keeps pointing forward.
    _assert_scalar_reversed_matches(UInt8(0), UInt8(8), UInt8(2))
    _assert_scalar_reversed_matches(UInt8(0), UInt8(7), UInt8(2))
    _assert_scalar_reversed_matches(UInt8(3), UInt8(10), UInt8(1))
    _assert_scalar_reversed_matches(UInt32(0), UInt32(10), UInt32(3))
    # Empty, in both bound orders.
    _assert_scalar_reversed_matches(UInt8(5), UInt8(5), UInt8(1))
    _assert_scalar_reversed_matches(UInt8(10), UInt8(5), UInt8(1))
    # High values, but no step that would carry the forward cursor past
    # `UInt8.MAX` (see MSTDL-2975).
    _assert_scalar_reversed_matches(UInt8(250), UInt8.MAX, UInt8(1))
    _assert_scalar_reversed_matches(UInt8(248), UInt8(254), UInt8(2))


def _prop_reversed_matches[dtype: DType](var draw: SIMD[dtype, 4]) raises:
    comptime T = Scalar[dtype]
    var step = (draw[1] & 7) + 1
    # A raw offset now that the forward cursor stops on an `end` it steps over,
    # so reversal is checked against ends that sit off the range's grid too.
    # Under 16 elements: an unbounded span would collect a `List` the size of
    # the dtype.
    var span = draw[2] & 15

    # `start` roams the whole dtype, so `start + span` reaches past the limit.
    _assert_scalar_reversed_matches(draw[0], draw[0] + span, step)

    # A uniform draw lands within a step of the limit about 3% of the time on
    # `Int8` and never on `Int64`, so pin the remaining triples there until a
    # strategy can bias toward boundary values on its own (MSTDL-3095).
    _assert_scalar_reversed_matches(T.MIN, T.MIN + span, step)
    _assert_scalar_reversed_matches(T.MAX - span, T.MAX, step)
    comptime if not dtype.is_unsigned():
        _assert_scalar_reversed_matches(T.MAX, T.MAX - span, -step)
        _assert_scalar_reversed_matches(T.MIN + span, T.MIN, -step)


def test_range_reversed_scalar_properties() raises:
    # The seed is pinned: a property test that picks its own seed reports a
    # failure no one else can reproduce.
    comptime for dtype in DTYPES:
        PropTest(config=PropTestConfig(runs=100, seed=0)).test[
            _prop_reversed_matches[dtype]
        ](SIMD[dtype, 4].strategy())


def test_range_reversed_scalar_sequential() raises:
    # The one- and two-argument scalar forms reverse to the same inclusive
    # bound, so they hold at the dtype's limit too.
    var descending = List[Int8]()
    for x in reversed(range(Int8.MIN, Int8.MIN + 4)):
        descending.append(x)
    assert_equal(len(descending), 4)
    assert_equal(descending[0], Int8.MIN + 3)
    assert_equal(descending[3], Int8.MIN)

    var empty = 0
    for _ in reversed(range(Int8.MIN, Int8.MIN)):
        empty += 1
    assert_equal(empty, 0)

    var zero_starting = List[Int8]()
    for x in reversed(range(Int8(4))):
        zero_starting.append(x)
    assert_equal(len(zero_starting), 4)
    assert_equal(zero_starting[0], Int8(3))
    assert_equal(zero_starting[3], Int8(0))

    var zero_starting_empty = 0
    for _ in reversed(range(Int8(0))):
        zero_starting_empty += 1
    assert_equal(zero_starting_empty, 0)


def test_range_reversed_bounds() raises:
    # A reversed iterator's size hint has to shrink as it is consumed, the
    # same as a forward one's, so collections built from it preallocate right.
    _test_range_iter_bounds(reversed(range(10)), 10)
    _test_range_iter_bounds(reversed(range(5, 10)), 5)
    _test_range_iter_bounds(reversed(range(0, 10, 3)), 4)
    _test_range_iter_bounds(reversed(range(10, 0, -3)), 4)
    _test_range_iter_bounds(reversed(range(Int8.MIN, Int8.MIN + 8, Int8(1))), 8)
    _test_range_iter_bounds(reversed(range(UInt8(0), UInt8(8), UInt8(2))), 4)
    _test_range_iter_bounds(reversed(range(5, 5)), 0)


def test_range_reversed_getitem() raises:
    # A reversed range indexes down from its first element, so `r[i]` has to
    # agree with the `i`th value the walk yields.
    var ascending = reversed(range(0, 10, 3))
    assert_equal(len(ascending), 4)
    assert_equal(ascending[0], 9)
    assert_equal(ascending[1], 6)
    assert_equal(ascending[3], 0)

    var descending = reversed(range(10, 0, -3))
    assert_equal(len(descending), 4)
    assert_equal(descending[0], 1)
    assert_equal(descending[3], 10)

    # At the dtype's limit, where the forward range's `start - step` is itself
    # unrepresentable.
    var at_limit = reversed(range(Int8.MIN, Int8.MIN + 8, Int8(2)))
    assert_equal(len(at_limit), 4)
    assert_equal(at_limit[0], Int8.MIN + 6)
    assert_equal(at_limit[3], Int8.MIN)

    var unsigned = reversed(range(UInt8(0), UInt8(8), UInt8(2)))
    assert_equal(len(unsigned), 4)
    assert_equal(unsigned[0], UInt8(6))
    assert_equal(unsigned[3], UInt8(0))


def test_range_reversed_scalar_strided() raises:
    # Both step directions, on and off the grid.
    _assert_scalar_reversed_matches(Int32(1), Int32(10), Int32(2))
    _assert_scalar_reversed_matches(Int32(0), Int32(20), Int32(2))
    _assert_scalar_reversed_matches(Int32(38), Int32(-13), Int32(-23))
    _assert_scalar_reversed_matches(Int32(10), Int32(0), Int32(-1))
    _assert_scalar_reversed_matches(Int32(10), Int32(0), Int32(-3))
    # Empty in either direction.
    _assert_scalar_reversed_matches(Int32(5), Int32(10), Int32(-1))
    _assert_scalar_reversed_matches(Int32(10), Int32(5), Int32(1))
    _assert_scalar_reversed_matches(Int32(5), Int32(5), Int32(1))
    # A zero step is empty, and reversing it must not divide by it.
    _assert_scalar_reversed_matches(Int32(5), Int32(10), Int32(0))
    _assert_scalar_reversed_matches(UInt8(5), UInt8(10), UInt8(0))


def test_range_reversed_float() raises:
    # `reversed()` must equal forward in reverse order, element-for-element,
    # exact even for steps not representable in binary.
    def assert_reversed_matches(
        start: Float64, end: Float64, step: Float64
    ) raises:
        var forward = List[Float64]()
        for x in range(start, end, step):
            forward.append(x)
        var backward = List[Float64]()
        for x in reversed(range(start, end, step)):
            backward.append(x)
        assert_equal(len(backward), len(forward))
        for i in range(len(forward)):
            assert_equal(backward[i], forward[len(forward) - 1 - i])

    # Exact-binary steps.
    assert_reversed_matches(5.0, 0.0, -0.5)  # fractional negative step
    assert_reversed_matches(0.0, 5.0, 0.5)  # ascending fractional step
    assert_reversed_matches(5.0, 0.6, -0.5)  # end not aligned to the grid
    assert_reversed_matches(0.0, 10.0, 3.0)  # step magnitude greater than one
    assert_reversed_matches(0.0, 5.0, -0.5)  # empty: step points the wrong way
    # Steps not exactly representable in binary.
    assert_reversed_matches(0.0, 1.0, 0.1)
    assert_reversed_matches(1.0, 0.0, -0.1)
    assert_reversed_matches(2.0, -1.0, -0.3)

    # Accumulation hazards: hundreds to thousands of steps.
    assert_reversed_matches(0.0, 100.0, 0.1)  # ~1000 steps
    assert_reversed_matches(-5.0, 5.0, 0.1)  # crosses zero, ~100 steps
    assert_reversed_matches(0.0, 10.0, 0.3)  # repr-nasty step
    assert_reversed_matches(10.0, 0.0, -0.7)  # descending repr-nasty step

    # Spot-check the exact reversed values for the originally reported case.
    var expected: List[Float64] = [
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
    ]
    var actual = List[Float64]()
    for x in reversed(range(5.0, 0.0, -0.5)):
        actual.append(x)
    assert_equal(actual, expected)


def test_range_float_forward_count() raises:
    # Non-representable step used to drift to 11 elements for [0, 1) by 0.1.
    var values = List[Float64]()
    for x in range(0.0, 1.0, 0.1):
        values.append(x)
    assert_equal(len(values), 10)
    assert_equal(values[0], 0.0)
    assert_equal(values[1], 0.1)


def test_range_float_zero_step() raises:
    # Zero step is empty both directions, not an infinite loop.
    var count = 0
    for _ in range(5.0, 0.0, 0.0):
        count += 1
    assert_equal(count, 0)
    var reverse_count = 0
    for _ in reversed(range(5.0, 0.0, 0.0)):
        reverse_count += 1
    assert_equal(reverse_count, 0)


def _test_range_int_zero_step[dtype: DType]() raises:
    comptime scalar = Scalar[dtype]

    # A zero step has no direction, so the range is empty whichever way the
    # bounds point. `start > end` used to step in place forever for signed
    # dtypes, and `start < end` for unsigned ones.
    var forward = 0
    for _ in range(scalar(0), scalar(3), scalar(0)):
        forward += 1
    assert_equal(forward, 0)

    var backward = 0
    for _ in range(scalar(3), scalar(0), scalar(0)):
        backward += 1
    assert_equal(backward, 0)

    # `len()` and `bounds()` used to divide by the zero step.
    assert_equal(range(scalar(3), scalar(0), scalar(0)).__len__(), 0)
    assert_equal(range(scalar(0), scalar(3), scalar(0)).__len__(), 0)
    var lower, upper = range(scalar(3), scalar(0), scalar(0)).bounds()
    assert_equal(lower, 0)
    assert_equal(upper.value(), 0)


def test_range_int_zero_step() raises:
    comptime for dtype in DTYPES:
        _test_range_int_zero_step[dtype]()

    var count = 0
    for _ in range(3, 0, 0):
        count += 1
    assert_equal(count, 0)

    # `__reversed__` used to take the modulus of the zero step.
    var reverse_count = 0
    for _ in reversed(range(3, 0, 0)):
        reverse_count += 1
    assert_equal(reverse_count, 0)


def test_range_int_zero_step_comptime() raises:
    # Unrolling a zero-step range used to hang the compiler forever.
    var count = 0
    comptime for _ in range(3, 0, 0):
        count += 1
    assert_equal(count, 0)

    comptime for _ in range(0, 3, 0):
        count += 1
    assert_equal(count, 0)


def test_range_float_grid() raises:
    # On-grid `end` is excluded, no `// + 1` overcount (1.0 = 4 * 0.25).
    var v = List[Float64]()
    for x in range(0.0, 1.0, 0.25):
        v.append(x)
    assert_equal(v, [0.0, 0.25, 0.5, 0.75])


def test_range_float_empty() raises:
    # Wrong-direction ranges are empty for either step sign.
    var forward = 0
    for _ in range(5.0, 0.0, 0.5):  # positive step, end < start
        forward += 1
    assert_equal(forward, 0)
    var backward = 0
    for _ in range(0.0, 5.0, -0.5):  # negative step, end > start
        backward += 1
    assert_equal(backward, 0)


def test_indexing() raises:
    var r = range(10)
    assert_equal(r[Int(4)], 4)
    assert_equal(r[3], 3)


def test_range_bounds() raises:
    var start = 0
    var end = 10

    # verify loop iteration
    var r = range(start, end)
    var last_seen = -1
    for x in r:
        last_seen = x
    assert_equal(last_seen, end - 1)

    # verify index lookup
    var ln = r.__len__()
    assert_equal(r[ln - 1], last_seen)


def test_scalar_range() raises:
    var r = range(UInt8(2), 16, 4)
    assert_equal(r.start, 2)
    assert_equal(r.end, 16)
    assert_equal(r.step, 4)

    def append_many[T: Copyable, //](mut list: List[T], *values: T):
        for value in values:
            list.append(value.copy())

    var expected_elements = List[UInt8]()
    append_many(expected_elements, 2, 6, 10, 14)
    var actual_elements = List[UInt8]()
    for e in r:
        actual_elements.append(UInt8(e))
    assert_equal(actual_elements, expected_elements)


def test_range_compile_time() raises:
    """Tests that verify compile-time parameter loops work correctly with
    various scalar types.
    """

    comptime for i in range(10):
        assert_true(i >= 0)

    comptime for i in reversed(range(10)):
        assert_true(i >= 0)

    comptime for i in range(UInt8(10)):
        assert_true(i >= 0)

    comptime for i in range(Int32(10)):
        assert_true(i >= 0)

    comptime for i in range(UInt16(1), 10, 2):
        assert_true(i >= 0)

    comptime for i in range(Int16(1), 10, 2):
        assert_true(i >= 0)

    comptime for i in reversed(range(Int16(1), 10, 2)):
        assert_true(i >= 0)

    # The reversed walk's inclusive bound has to hold at comptime too.
    comptime for i in reversed(range(Int8.MIN, Int8.MIN + 4, Int8(1))):
        assert_true(i <= Int8.MIN + 3)

    comptime for i in range(Int64(10), 1, -2):
        assert_true(i > 0)
        assert_true(i <= 10)


def test_range_iterable() raises:
    var ai = 0
    var bi = UInt8(0)
    var ci = 0
    for a, b, c in zip(range(0, 10), range(UInt8(10)), range(0, 20, 2)):
        assert_equal(a, ai)
        assert_equal(b, bi)
        assert_equal(c, ci)
        ai += 1
        bi += 1
        ci += 2


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
