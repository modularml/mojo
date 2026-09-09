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
"""Defines Mojo's built-in `range()` function.

In Mojo, ranges are values, not loop constructs, generators, or lists. Every
range is a half-open interval, [start, end).

The stand-alone `range()` function constructs zero-based, sequential, and
strided ranges.

`range()` is built in. You don't need to import it.
"""

from std.math import ceil, ceildiv, fma
from std.sys.info import size_of
from std.sys.intrinsics import unlikely

from std.python import PythonObject

from std.utils._select import _select_register_value as select

# ===----------------------------------------------------------------------=== #
# Range
# ===----------------------------------------------------------------------=== #


struct _ZeroStartingRange[dtype: DType = .int](
    ImplicitlyCopyable,
    Iterable,
    Iterator,
    ReversibleRange,
    Sized,
    TrivialRegisterPassable,
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Scalar[Self.dtype]
    comptime ReversedType = _StridedRange[Self.dtype, forward=False]
    var curr: Scalar[Self.dtype]
    var end: Scalar[Self.dtype]

    @always_inline
    def __init__(out self, end: Scalar[Self.dtype]):
        self.curr = 0
        self.end = max(end, 0)

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    def __next__(mut self) raises StopIteration -> Scalar[Self.dtype]:
        var curr = self.curr
        if curr == self.end:
            raise StopIteration()
        self.curr = curr + 1
        return curr

    @always_inline
    def __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    def __len__(self) -> Int:
        return _len_as_int(self.end - self.curr)

    @always_inline
    def __getitem__[I: Indexer](self, idx: I) -> Scalar[Self.dtype]:
        var i = index(idx)
        assert i < self.__len__(), "index out of range"
        return Scalar[Self.dtype](i)

    @always_inline
    def __reversed__(self) -> Self.ReversedType:
        comptime assert (
            not Self.dtype.is_unsigned()
        ), "cannot reverse an unsigned range"
        # The reversed walk stops at an *inclusive* `0`, and `idx` starts it
        # exhausted for an empty range, so the wrapped `end - 1` is never read.
        return Self.ReversedType(
            self.end - 1,
            Scalar[Self.dtype](0),
            Scalar[Self.dtype](1),
            1 if self.end == 0 else 0,
        )

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return _scalar_range_bounds(self.end - self.curr)


struct _SequentialRange[dtype: DType = .int](
    ImplicitlyCopyable,
    Iterable,
    Iterator,
    ReversibleRange,
    Sized,
    TrivialRegisterPassable,
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Scalar[Self.dtype]
    comptime ReversedType = _StridedRange[Self.dtype, forward=False]
    var start: Scalar[Self.dtype]
    var end: Scalar[Self.dtype]

    @always_inline
    def __init__(out self, start: Scalar[Self.dtype], end: Scalar[Self.dtype]):
        self.start = start
        self.end = max(start, end)

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    def __next__(mut self) raises StopIteration -> Scalar[Self.dtype]:
        var start = self.start
        if start == self.end:
            raise StopIteration()
        self.start = start + 1
        return start

    @always_inline
    def __len__(self) -> Int:
        return _len_as_int(self.end - self.start)

    @always_inline
    def __getitem__[I: Indexer](self, idx: I) -> Scalar[Self.dtype]:
        var i = index(idx)
        assert i < self.__len__(), "index out of range"
        return self.start + Scalar[Self.dtype](i)

    @always_inline
    def __reversed__(self) -> Self.ReversedType:
        comptime assert (
            not Self.dtype.is_unsigned()
        ), "cannot reverse an unsigned range"
        # `start` is the reversed walk's *inclusive* bound, so an empty range
        # is flagged through `idx` rather than through a `start - 1` sentinel
        # that is unrepresentable at the dtype's lower limit.
        return Self.ReversedType(
            self.end - 1,
            self.start,
            Scalar[Self.dtype](1),
            1 if self.start == self.end else 0,
        )

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return _scalar_range_bounds(self.end - self.start)


@always_inline
def _fp_range_count[
    dtype: DType, //
](start: Scalar[dtype], end: Scalar[dtype], step: Scalar[dtype]) -> Int:
    # A zero step is empty.
    if step == 0:
        return 0
    # This calculation avoids `// + 1`, which overcounts by one when `end`
    # lands on the grid. `ceil` and `/` are correct for forward and backward
    # ranges.
    var raw = ceil((end - start) / step)
    return Int(raw) if raw > 0 else 0


# Integer ranges iterate by value: `start` is the next element, and `end` is
# the bound it is compared against. Reverse iteration walks the same grid
# downward, subtracting `step`, but holds `end` as the *inclusive* first
# element of the forward range and flags exhaustion through `idx`. The
# exclusive `start - step` sentinel it replaces is unrepresentable whenever
# `start` sits within `step` of the dtype's limit, which silently emptied the
# reversed iterator.
#
# Floating-point ranges iterate by index (`fma(k, step, start)`), avoiding
# drift, with `idx` as the element cursor. Reverse iteration mirrors forward
# bit for bit, counting `idx` down from -1 (which `__next__` maps to
# `count - 1`).
struct _StridedRange[dtype: DType = .int, forward: Bool = True](
    ImplicitlyCopyable,
    Iterable,
    Iterator,
    ReversibleRange,
    Sized,
    TrivialRegisterPassable,
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Scalar[Self.dtype]
    comptime ReversedType = _StridedRange[Self.dtype, forward=not Self.forward]
    var start: Scalar[Self.dtype]
    var end: Scalar[Self.dtype]
    var step: Scalar[Self.dtype]
    var idx: Int
    """Floating-point: the element cursor. Integer forward: nonzero once a step
    has wrapped out of the dtype — not a general exhausted flag, since a range
    that ends normally leaves it zero. Integer reverse: nonzero once exhausted.

    Only a `__reversed__` ever passes this non-zero at construction: an integer
    one to hand back an iterator that is already exhausted, a floating-point one
    to seat the cursor."""

    @always_inline
    def __init__(
        out self,
        start: Scalar[Self.dtype],
        end: Scalar[Self.dtype],
        step: Scalar[Self.dtype],
        idx: Int = 0,
    ):
        comptime if Self.dtype.is_integral() and Self.forward:
            # A zero step has no direction, so the range is empty. Collapsing it
            # to a canonical empty range here keeps `__next__` from stepping in
            # place forever, and keeps `__len__`, `bounds()`, and `__reversed__`
            # from dividing by the step. Doing it once at construction also
            # keeps the check out of the iteration loop.
            var degenerate = step == 0
            self.start = select(degenerate, Scalar[Self.dtype](0), start)
            self.end = select(degenerate, Scalar[Self.dtype](0), end)
            self.step = select(degenerate, Scalar[Self.dtype](1), step)
        else:
            # A zero float step is already empty by `_fp_range_count`, and the
            # `fma` cursor needs the values as given. A reversed integer range
            # needs them intact too: its `end` is an inclusive bound, so the
            # collapse above would not describe the same empty range. It never
            # sees a zero step regardless — the forward range it is built from
            # has already swapped one out.
            self.start = start
            self.end = end
            self.step = step
        self.idx = idx

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    def __next__(mut self) raises StopIteration -> Scalar[Self.dtype]:
        comptime if Self.dtype.is_floating_point():
            var count = _fp_range_count(self.start, self.end, self.step)
            comptime if Self.forward:
                if self.idx >= count:
                    raise StopIteration()
                var result = fma(
                    Scalar[Self.dtype](self.idx), self.step, self.start
                )
                self.idx += 1
                return result
            else:
                var i = count + self.idx
                if i < 0:
                    raise StopIteration()
                var result = fma(Scalar[Self.dtype](i), self.step, self.start)
                self.idx -= 1
                return result
        elif Self.forward:
            # `|` and not `or`: short-circuiting would put a second branch in
            # the loop, which stops it from rotating.
            var exhausted = self.idx != 0

            # If the type is unsigned, then 'step' cannot be negative.
            comptime if Self.dtype.is_unsigned():
                if exhausted | (self.start >= self.end):
                    raise StopIteration()
            else:
                if self.step > 0:
                    if exhausted | (self.start >= self.end):
                        raise StopIteration()
                elif exhausted | (self.end >= self.start):
                    raise StopIteration()

            var result = self.start
            # `start + step` wraps when the element after this one falls
            # outside the dtype. The wrapped cursor lands on the far side of
            # `end`, which the bound test above reads as "keep going", so the
            # loop would restart near the opposite limit and never finish.
            # Wrapping is the only way a step can move the cursor against its
            # own direction, so detect it that way and record it, which stops
            # the next call for either step direction.
            #
            # Recording the wrap beside the cursor rather than correcting the
            # cursor or the bound is what keeps this affordable. `start` stays
            # a plain `start + k * step` recurrence and `end` stays
            # loop-invariant, so derived addresses still strength-reduce to
            # pointer bumps and a bound the caller wrote as a literal still
            # proves the wrap away entirely. Correcting either one is a
            # `select`, which is neither, and every strided loop in every GPU
            # kernel pays for it.
            var next = self.start + self.step
            var wrapped = next < self.start
            comptime if not Self.dtype.is_unsigned():
                # A negative step moves the cursor down without wrapping.
                wrapped = wrapped != (self.step < 0)
            self.idx = Int(wrapped)
            self.start = next
            return result
        else:
            if self.idx != 0:
                raise StopIteration()
            var result = self.start
            # `end` is the inclusive first element of the forward range and
            # `__reversed__` puts `start` exactly on the grid, so this fires
            # precisely once that element has been produced. Deliberately an
            # inequality rather than `==`: iteration terminates even if `start`
            # were ever off-grid, and stepping is skipped at the bound so the
            # cursor can never leave the dtype's range.
            var at_end = select(
                self.step > 0, self.start <= self.end, self.start >= self.end
            )
            if at_end:
                self.idx = 1
            else:
                self.start -= self.step
            return result

    @always_inline
    def _unsigned_count(self) -> Scalar[Self.dtype]:
        """Returns the number of elements left, in the range's own dtype.

        An unsigned range's element count can exceed `Int`, so each caller
        decides how to report that.
        """
        comptime assert Self.dtype.is_unsigned(), "dtype must be unsigned"

        comptime if Self.forward:
            # A wrapped cursor sits back inside `[start, end)` and would count
            # as elements still to come, so `idx` settles it first.
            return select(
                (self.idx == 0) & (self.start < self.end),
                ceildiv(self.end - self.start, self.step),
                0,
            )
        else:
            if self.idx != 0:
                return 0
            # `end` is inclusive, hence the `+ 1`. An unsigned step is never
            # negative, so the reversed walk always has `start >= end`.
            return (self.start - self.end) // self.step + 1

    @always_inline
    def _signed_count(self) -> Int:
        """Returns the number of elements left."""
        # Compute the length using `Int` so small signed dtypes whose element
        # count exceeds the dtype's range don't overflow.
        #
        # TODO(MSTDL-3087): `int64` and `int` have nothing wider to widen to,
        # so a span past `Int.MAX` wraps here and the count comes out wrong
        # instead of asserting the way the unsigned side does.
        comptime assert Self.dtype.is_signed(), "dtype must be signed"

        var start = Int(self.start)
        var end = Int(self.end)
        var step = Int(self.step)
        comptime if Self.forward:
            # If the step is positive we want to check that the start is
            # smaller than the end, if the step is negative we want to check
            # the reverse. We break this into selects to avoid branches.
            var c1 = (step > 0) & (start > end)
            var c2 = (step < 0) & (start < end)
            # A wrapped cursor sits back inside the range and would count as
            # elements still to come, so `idx` joins the emptiness test.
            var cnd = c1 | c2 | (self.idx != 0)
            var numerator = abs(start - end)
            var denominator = abs(step)
            return ceildiv(
                select(cnd, 0, numerator), select(cnd, 1, denominator)
            )
        else:
            if self.idx != 0:
                return 0
            # `end` is inclusive, hence the `+ 1`.
            return abs(start - end) // abs(step) + 1

    @always_inline
    def __len__(self) -> Int:
        comptime if Self.dtype.is_unsigned():
            # `bounds()` clamps an unsigned count > `Int.MAX` for the size
            # hint; `len()` must not hide that, so assert the exact count fits.
            return _len_as_int(self._unsigned_count())
        else:
            # Signed integer: the exact count fits in `Int` for every dtype
            # narrower than it, and MSTDL-3087 tracks the wider ones. (A float
            # `dtype` makes `_signed_count()` a compile-time error, as
            # intended.)
            return self._signed_count()

    @always_inline
    def __getitem__[I: Indexer](self, idx: I) -> Scalar[Self.dtype]:
        var i = index(idx)
        assert i < self.__len__(), "index out of range"
        comptime if Self.forward:
            return self.start + Scalar[Self.dtype](i) * self.step
        else:
            return self.start - Scalar[Self.dtype](i) * self.step

    @always_inline
    def __reversed__(self) -> Self.ReversedType:
        # Reversing back would have to rebuild the forward range's exclusive
        # `end`, one step past the last element, which is exactly the value
        # that can fall outside the dtype. So the round trip is rejected.
        comptime assert Self.forward, "a reversed range cannot be reversed"

        comptime if Self.dtype.is_floating_point():
            # Reverse starts the cursor at -1; `__next__` maps it to
            # count - 1.
            return Self.ReversedType(self.start, self.end, self.step, -1)
        # The reversed walk keeps the forward step and subtracts it, ending on
        # `start` held as an *inclusive* bound. Its first element comes from
        # the element count rather than from snapping `end` with `%`, because
        # `end - start` can overflow the dtype and leave the cursor off the
        # range's grid — which would make the walk miss `start` entirely.
        # `count - 1` wraps when the range is empty; `idx` starts the iterator
        # exhausted in that case, so the cursor is never read.
        elif Self.dtype.is_unsigned():
            var count = self._unsigned_count()
            var last = self.start + (count - 1) * self.step
            return Self.ReversedType(
                select(count == 0, self.start, last),
                self.start,
                self.step,
                1 if count == 0 else 0,
            )
        else:
            # As above, but the count and the `start + (count - 1) * step`
            # product are computed in `Int` so neither can overflow the dtype
            # they are measured in.
            var count = self._signed_count()
            var last = Int(self.start) + (count - 1) * Int(self.step)
            # Select before narrowing: an empty range's `last` sits one step
            # outside the dtype, which is exactly what this all avoids.
            return Self.ReversedType(
                Scalar[Self.dtype](select(count == 0, Int(self.start), last)),
                self.start,
                self.step,
                1 if count == 0 else 0,
            )

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        comptime assert Self.dtype.is_integral(), "dtype must be integral"

        comptime if Self.dtype.is_unsigned():
            # An unsigned range's element count can exceed `Int.MAX`;
            # `_scalar_range_bounds` clamps the lower bound and reports an
            # unknown (`None`) upper bound in that case.
            return _scalar_range_bounds(self._unsigned_count())
        else:  # is_signed
            var length = self._signed_count()
            return (length, {length})


@always_inline
def range[T: Indexer, //](end: T) -> _ZeroStartingRange[.int]:
    """Returns the integer sequence `[0, end)`.

    Integer ranges are values. They support `len()`, O(1) indexing, and
    `reversed()` without allocating. `reversed(range(n))` iterates from
    `n - 1` down to `0`.

    Parameters:
        T: The type of the end value. Constrained to `Indexer`.

    Args:
        end: The exclusive upper bound. Negative values produce an empty range.

    Returns:
        A zero-based integer range over `[0, end)`.

    Performance:
        O(1) construction. O(1) indexing. No list allocation.

    Examples:

    ```mojo
    for i in range(5):
        print(i)  # 0, 1, 2, 3, 4

    var steps = range(1000)
    print(steps[499])  # 499
    print(len(steps))  # 1000

    for i in reversed(range(5)):
        print(i)  # 4, 3, 2, 1, 0
    ```
    """
    return _ZeroStartingRange(index(end))


@always_inline
def range[T: Indexer, //](start: T, end: T) -> _SequentialRange[.int]:
    """Returns the integer sequence `[start, end)`.

    **The two-argument form never counts down.** `range(7, 3)` is empty,
    not `[7, 6, 5, 4]`. Use the three-argument form with a negative step to
    count downward. The range supports `len()`, O(1) indexing, and
    `reversed()`.

    Parameters:
        T: The type of the start and end values. Constrained to `Indexer`.

    Args:
        start: The inclusive lower bound.
        end: The exclusive upper bound. When `end <= start`, the range is empty.

    Returns:
        A sequential integer range over `[start, end)`.

    Performance:
        O(1) construction. O(1) indexing. No list allocation.

    Examples:

    ```mojo
    for i in range(3, 7):
        print(i)  # 3, 4, 5, 6

    for i in range(-3, 4):
        print(i)  # -3, -2, -1, 0, 1, 2, 3

    print(len(range(7, 3)))  # 0
    ```
    """
    return _SequentialRange(index(start), index(end))


@always_inline
def range[T: Indexer, //](start: T, end: T, step: T) -> _StridedRange[.int]:
    """Returns the integer sequence `[start, end)` with a given step.

    When you don't know which bound is larger, choose the direction with an
    inline conditional:

    ```mojo
    var step = 1 if end > start else -1
    for i in range(start, end, step):
        ...
    ```

    Parameters:
        T: The type of the start, end, and step values. Constrained to
            `Indexer`.

    Args:
        start: The inclusive lower bound when stepping forward, or the
            inclusive upper bound when stepping backward.
        end: The exclusive bound in the direction of the step.
        step: The increment per iteration. A positive step counts up, and a
            negative step counts down. A zero step produces an empty range.

    Returns:
        A strided integer range over `[start, end)` by `step`.

    Performance:
        O(1) construction. O(1) indexing. No list allocation.

    Examples:

    ```mojo
    for i in range(0, 10, 2):
        print(i)  # 0, 2, 4, 6, 8

    for i in range(7, 3, -1):
        print(i)  # 7, 6, 5, 4

    var evens = range(0, 2_000_000, 2)
    print(evens[999_999])  # 1_999_998
    print(len(evens))      # 1_000_000
    ```
    """
    return _StridedRange(index(start), index(end), index(step))


# ===----------------------------------------------------------------------=== #
# Range Scalar
# ===----------------------------------------------------------------------=== #


def _len_as_int[dtype: DType](n: Scalar[dtype]) -> Int:
    """Converts a non-negative range length to `Int`.

    `len()` returns `Int`, but an unsigned range's element count can exceed
    `Int.MAX`. Silently clamping there would hide the true length and let
    indexed access read the wrong element, so asking for `len()` in that case
    is a bug: assert instead. Use `bounds()`, whose upper bound is `None`, for
    the size hint of such a range.
    """
    comptime if size_of[Scalar[dtype]]() >= size_of[Int]():
        assert UInt(n) <= UInt(Int.MAX), "range length exceeds Int.MAX"

    return Int(n)


def _scalar_range_bounds[
    dtype: DType
](len: Scalar[dtype]) -> Tuple[Int, Optional[Int]]:
    comptime if size_of[Scalar[dtype]]() >= size_of[Int]():
        if unlikely(UInt(len) > UInt(Int.MAX)):
            return (Int.MAX, None)

    return (Int(len), {Int(len)})


@always_inline
def range[dtype: DType, //](end: Scalar[dtype]) -> _ZeroStartingRange[dtype]:
    """Returns the scalar sequence `[0, end)` with elements of type `dtype`.

    Use this overload when you need typed scalar elements.

    The `dtype` is inferred from the argument, so `Int32(8)` produces
    `Int32` elements. This form requires an integer `dtype`; floating-point
    ranges require an explicit step. These integer ranges support `len()`
    and O(1) indexing. Signed-integer ranges can be reversed; reversing an
    unsigned range is a compile-time error.

    Parameters:
        dtype: The `DType` of the sequence elements. Inferred from `end`.

    Args:
        end: The exclusive upper bound. Negative values produce an empty range.

    Returns:
        A zero-based scalar range over `[0, end)`.

    Performance:
        O(1). No list allocation.

    Examples:

    ```mojo
    for i in range(UInt8(4)):
        print(i)  # 0, 1, 2, 3 (each value is UInt8)
    ```
    """
    comptime assert dtype.is_numeric(), "range requires a numeric dtype"
    comptime assert dtype.is_integral(), (
        "a floating-point range requires an explicit step; use range(start,"
        " end, step)"
    )
    return _ZeroStartingRange[dtype](end)


@always_inline
def range[
    dtype: DType, //
](start: Scalar[dtype], end: Scalar[dtype]) -> _SequentialRange[dtype]:
    """Returns the scalar sequence `[start, end)` with elements of type `dtype`.

    **The two-argument form never counts down.** The range is empty
    when `end <= start`. Use the three-argument form with a negative
    step to count downward. This form requires an integer `dtype`;
    floating-point ranges require an explicit step. These integer ranges
    support `len()` and O(1) indexing. Signed-integer ranges can be
    reversed; reversing an unsigned range is a compile-time error.

    Parameters:
        dtype: The `DType` of the sequence elements. Inferred from the arguments.

    Args:
        start: The inclusive lower bound.
        end: The exclusive upper bound. When `end <= start`, the range is empty.

    Returns:
        A sequential scalar range over `[start, end)`.

    Performance:
        O(1). No list allocation.

    Examples:

    ```mojo
    for i in range(Int32(3), Int32(7)):
        print(i)  # 3, 4, 5, 6  — each value is Int32
    ```
    """
    comptime assert dtype.is_numeric(), "range requires a numeric dtype"
    comptime assert dtype.is_integral(), (
        "a floating-point range requires an explicit step; use range(start,"
        " end, step)"
    )
    return _SequentialRange[dtype](start, end)


@always_inline
def range[
    dtype: DType, //
](
    start: Scalar[dtype], end: Scalar[dtype], step: Scalar[dtype]
) -> _StridedRange[dtype]:
    """Returns the scalar sequence `[start, end)` with a given step.

    Integer scalar ranges support `len()`, O(1) indexing, and `reversed()`,
    including unsigned ranges. Float ranges are iteration-only. Each element
    is computed as `fma(i, step, start)`, and `reversed()` is a bit-for-bit
    mirror of forward iteration.

    **Float endpoints are exclusive.** To include a specific endpoint, push
    `end` past it by a fraction of the step:

    ```mojo
    # [0.0, 0.25, 0.5, 0.75] — 1.0 not included
    for t in range(Float64(0.0), Float64(1.0), Float64(0.25)):
        print(t)

    # [0.0, 0.25, 0.5, 0.75, 1.0] — endpoint included
    var tolerance = Float64(0.1)
    for t in range(Float64(0.0), Float64(1) + tolerance, Float64(0.25)):
        print(t)
    ```

    **A zero step yields an empty range.** A step of zero has no direction, so
    the range has no elements regardless of the bounds.

    Parameters:
        dtype: The `DType` of the sequence elements. Inferred from the arguments.

    Args:
        start: The inclusive lower bound when stepping forward, or the
            inclusive upper bound when stepping backward.
        end: The exclusive bound in the direction of the step.
        step: The increment per iteration. A positive step counts up, and a
            negative step counts down. A zero step yields an empty range.

    Returns:
        A strided scalar range over `[start, end)` by `step`.

    Performance:
        O(1). No list allocation.

    Examples:

    ```mojo
    # Walk t over [0, 1)
    for t in range(Float64(0.0), Float64(1.0), Float64(0.25)):
        print(t)  # 0.0, 0.25, 0.5, 0.75

    # Integer scalar range — len() and indexed access work
    var r = range(Int32(0), Int32(20), Int32(2))
    print(len(r))       # 10
    print(r[Int32(3)])  # 6
    ```
    """
    comptime assert dtype.is_numeric(), "range requires a numeric dtype"
    return _StridedRange[dtype](start, end, step)
