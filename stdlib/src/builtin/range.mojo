# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements a 'range' call.

These are Mojo built-ins, so you don't need to import them.
"""

from math import abs as _abs

from python.object import PythonObject


@always_inline
fn _div_ceil_positive(numerator: Int, denominator: Int) -> Int:
    """Divides an integer by another integer, and round up to the nearest
    integer.

    Constraints:
      Will raise an exception if denominator is zero.
      Assumes that both inputs are positive.

    Args:
      numerator: The numerator.
      denominator: The denominator.

    Returns:
      The ceiling of numerator divided by denominator.
    """
    debug_assert(denominator != 0, "divide by zero")
    return (numerator + denominator - 1)._positive_div(denominator)


@register_passable("trivial")
struct _ZeroStartingRange(Sized):
    var curr: Int
    var end: Int

    @always_inline("nodebug")
    fn __init__(end: Int) -> Self:
        return Self {curr: end, end: end}

    @always_inline("nodebug")
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(inout self) -> Int:
        let curr = self.curr
        self.curr -= 1
        return self.end - curr

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return self.curr

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Int:
        return idx


@value
@register_passable("trivial")
struct _SequentialRange(Sized):
    var start: Int
    var end: Int

    @always_inline("nodebug")
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(inout self) -> Int:
        let start = self.start
        self.start += 1
        return start

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return self.end - self.start if self.start < self.end else 0

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Int:
        return self.start + idx


@value
@register_passable("trivial")
struct _StridedRangeIterator(Sized):
    var start: Int
    var end: Int
    var step: Int

    @always_inline
    fn __len__(self) -> Int:
        if self.step > 0 and self.start < self.end:
            return self.end - self.start
        elif self.step < 0 and self.start > self.end:
            return self.start - self.end
        else:
            return 0

    @always_inline
    fn __next__(inout self) -> Int:
        let result = self.start
        self.start += self.step
        return result


@value
@register_passable("trivial")
struct _StridedRange(Sized):
    var start: Int
    var end: Int
    var step: Int

    @always_inline("nodebug")
    fn __init__(end: Int) -> Self:
        return Self {start: 0, end: end, step: 1}

    @always_inline("nodebug")
    fn __init__(start: Int, end: Int) -> Self:
        return Self {start: start, end: end, step: 1}

    @always_inline("nodebug")
    fn __iter__(self) -> _StridedRangeIterator:
        return _StridedRangeIterator(self.start, self.end, self.step)

    @always_inline
    fn __next__(inout self) -> Int:
        let result = self.start
        self.start += self.step
        return result

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return _div_ceil_positive(_abs(self.start - self.end), _abs(self.step))

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Int:
        return self.start + idx * self.step


@always_inline("nodebug")
fn range[type: Intable](end: type) -> _ZeroStartingRange:
    """Constructs a [0; end) Range.

    Parameters:
        type: The type of the end value.

    Args:
        end: The end of the range.

    Returns:
        The constructed range.
    """
    return _ZeroStartingRange(int(end))


@always_inline
fn range[type: IntableRaising](end: type) raises -> _ZeroStartingRange:
    """Constructs a [0; end) Range.

    Parameters:
        type: The type of the end value.

    Args:
        end: The end of the range.

    Returns:
        The constructed range.
    """
    return _ZeroStartingRange(int(end))


@always_inline("nodebug")
fn range[t0: Intable, t1: Intable](start: t0, end: t1) -> _SequentialRange:
    """Constructs a [start; end) Range.

    Parameters:
        t0: The type of the start value.
        t1: The type of the end value.

    Args:
        start: The start of the range.
        end: The end of the range.

    Returns:
        The constructed range.
    """
    let s = int(start)
    let e = int(end)
    return _SequentialRange(s, e)


@always_inline("nodebug")
fn range[
    t0: IntableRaising, t1: IntableRaising
](start: t0, end: t1) raises -> _SequentialRange:
    """Constructs a [start; end) Range.

    Parameters:
        t0: The type of the start value.
        t1: The type of the end value.

    Args:
        start: The start of the range.
        end: The end of the range.

    Returns:
        The constructed range.
    """
    let s = int(start)
    let e = int(end)
    return _SequentialRange(s, e)


@always_inline
fn range[
    t0: Intable, t1: Intable, t2: Intable
](start: t0, end: t1, step: t2) -> _StridedRange:
    """Constructs a [start; end) Range with a given step.

    Parameters:
        t0: The type of the start value.
        t1: The type of the end value.
        t2: The type of the step value.

    Args:
        start: The start of the range.
        end: The end of the range.
        step: The step for the range.

    Returns:
        The constructed range.
    """
    return _StridedRange(int(start), int(end), int(step))


@always_inline
fn range[
    t0: IntableRaising, t1: IntableRaising, t2: IntableRaising
](start: t0, end: t1, step: t2) raises -> _StridedRange:
    """Constructs a [start; end) Range with a given step.

    Parameters:
        t0: The type of the start value.
        t1: The type of the end value.
        t2: The type of the step value.

    Args:
        start: The start of the range.
        end: The end of the range.
        step: The step for the range.

    Returns:
        The constructed range.
    """
    return _StridedRange(int(start), int(end), int(step))
