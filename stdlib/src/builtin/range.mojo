# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Implements a 'range' call.

These are Mojo built-ins, so you don't need to import them.
"""


# FIXME(MOCO-658): Explicit conformance to these traits shouldn't be needed.
from builtin._stubs import _IntIterable, _StridedIterable, _UIntStridedIterable
from python import (
    PythonObject,
)  # TODO: remove this and fixup downstream imports
from math import ceildiv
from utils._select import _select_register_value as select
from utils.string_slice import StringSlice, _StringSliceIter, Stringlike
from collections.list import _ListIter
from collections.dict import Dict, _DictKeyIter, _DictValueIter, _DictEntryIter

# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@always_inline
fn _sign(x: Int) -> Int:
    var result = 0
    result = select(x > 0, 1, result)
    result = select(x < 0, -1, result)
    return result


# ===----------------------------------------------------------------------=== #
# Range
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct _ZeroStartingRange(Sized, ReversibleRange, _IntIterable):
    var curr: Int
    var end: Int

    @always_inline
    fn __init__(inout self, end: Int):
        self.curr = max(0, end)
        self.end = self.curr

    @always_inline
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(inout self) -> Int:
        var curr = self.curr
        self.curr -= 1
        return self.end - curr

    @always_inline
    fn __has_more__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        return self.curr

    @always_inline
    fn __getitem__(self, idx: Int) -> Int:
        debug_assert(idx < self.__len__(), "index out of range")
        return index(idx)

    @always_inline
    fn __reversed__(self) -> _StridedRange:
        return range(self.end - 1, -1, -1)


@value
@register_passable("trivial")
struct _SequentialRange(Sized, ReversibleRange, _IntIterable):
    var start: Int
    var end: Int

    @always_inline
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(inout self) -> Int:
        var start = self.start
        self.start += 1
        return start

    @always_inline
    fn __has_more__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        return max(0, self.end - self.start)

    @always_inline
    fn __getitem__(self, idx: Int) -> Int:
        debug_assert(idx < self.__len__(), "index out of range")
        return self.start + index(idx)

    @always_inline
    fn __reversed__(self) -> _StridedRange:
        return range(self.end - 1, self.start - 1, -1)


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
        var result = self.start
        self.start += self.step
        return result

    @always_inline
    fn __has_more__(self) -> Bool:
        return self.__len__() > 0


@value
@register_passable("trivial")
struct _StridedRange(Sized, ReversibleRange, _StridedIterable):
    var start: Int
    var end: Int
    var step: Int

    @always_inline
    fn __init__(inout self, start: Int, end: Int):
        self.start = start
        self.end = end
        self.step = 1

    @always_inline
    fn __iter__(self) -> _StridedRangeIterator:
        return _StridedRangeIterator(self.start, self.end, self.step)

    @always_inline
    fn __next__(inout self) -> Int:
        var result = self.start
        self.start += self.step
        return result

    @always_inline
    fn __has_more__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        # If the step is positive we want to check that the start is smaller
        # than the end, if the step is negative we want to check the reverse.
        # We break this into selects to avoid generating branches.
        var c1 = (self.step > 0) & (self.start > self.end)
        var c2 = (self.step < 0) & (self.start < self.end)
        var cnd = c1 | c2

        var numerator = abs(self.start - self.end)
        var denominator = abs(self.step)

        # If the start is after the end and step is positive then we
        # are generating an empty range. In this case divide 0/1 to
        # return 0 without a branch.
        return ceildiv(select(cnd, 0, numerator), select(cnd, 1, denominator))

    @always_inline
    fn __getitem__(self, idx: Int) -> Int:
        debug_assert(idx < self.__len__(), "index out of range")
        return self.start + index(idx) * self.step

    @always_inline
    fn __reversed__(self) -> _StridedRange:
        var shifted_end = self.end - _sign(self.step)
        var start = shifted_end - ((shifted_end - self.start) % self.step)
        var end = self.start - self.step
        var step = -self.step
        return range(start, end, step)


@always_inline
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


@always_inline
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
    return _SequentialRange(int(start), int(end))


@always_inline
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
    return _SequentialRange(int(start), int(end))


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


# ===----------------------------------------------------------------------=== #
# Range UInt
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct _UIntZeroStartingRange(UIntSized):
    var curr: UInt
    var end: UInt

    @always_inline
    fn __init__(inout self, end: UInt):
        self.curr = max(0, end)
        self.end = self.curr

    @always_inline
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(inout self) -> UInt:
        var curr = self.curr
        self.curr -= 1
        return self.end - curr

    @always_inline
    fn __has_more__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> UInt:
        return self.curr

    @always_inline
    fn __getitem__(self, idx: UInt) -> UInt:
        debug_assert(idx < self.__len__(), "index out of range")
        return idx


@value
@register_passable("trivial")
struct _UIntStridedRangeIterator(UIntSized):
    var start: UInt
    var end: UInt
    var step: UInt

    @always_inline
    fn __len__(self) -> UInt:
        return select(self.start < self.end, self.end - self.start, 0)

    @always_inline
    fn __next__(inout self) -> UInt:
        var result = self.start
        self.start += self.step
        return result

    @always_inline
    fn __has_more__(self) -> Bool:
        return self.__len__() > 0


@value
@register_passable("trivial")
struct _UIntStridedRange(UIntSized, _UIntStridedIterable):
    var start: UInt
    var end: UInt
    var step: UInt

    @always_inline
    fn __init__(inout self, start: UInt, end: UInt, step: UInt):
        self.start = start
        self.end = end
        debug_assert(
            step != 0, "range() arg 3 (the step size) must not be zero"
        )
        debug_assert(
            step != UInt(Int(-1)),
            (
                "range() arg 3 (the step size) cannot be -1.  Reverse range is"
                " not supported yet for UInt ranges."
            ),
        )
        self.step = step

    @always_inline
    fn __iter__(self) -> _UIntStridedRangeIterator:
        return _UIntStridedRangeIterator(self.start, self.end, self.step)

    @always_inline
    fn __next__(inout self) -> UInt:
        if self.start >= self.end:
            return self.end
        var result = self.start
        self.start += self.step
        return result

    @always_inline
    fn __has_more__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> UInt:
        if self.start >= self.end:
            return 0
        return ceildiv(self.end - self.start, self.step)

    @always_inline
    fn __getitem__(self, idx: UInt) -> UInt:
        debug_assert(idx < self.__len__(), "index out of range")
        return self.start + idx * self.step


@always_inline
fn range(end: UInt) -> _UIntZeroStartingRange:
    """Constructs a [0; end) Range.

    Args:
        end: The end of the range.

    Returns:
        The constructed range.
    """
    return _UIntZeroStartingRange(end)


@always_inline
fn range(start: UInt, end: UInt, step: UInt = 1) -> _UIntStridedRange:
    """Constructs a [start; end) Range with a given step.

    Args:
        start: The start of the range.
        end: The end of the range.
        step: The step for the range.  Defaults to 1.

    Returns:
        The constructed range.
    """
    return _UIntStridedRange(start, end, step)


# ===----------------------------------------------------------------------=== #
# Range Scalar
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct _ZeroStartingScalarRange[type: DType]:
    var curr: Scalar[type]
    var end: Scalar[type]

    @always_inline
    fn __init__(inout self, end: Scalar[type]):
        self.curr = max(0, end)
        self.end = self.curr

    @always_inline
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(inout self) -> Scalar[type]:
        var curr = self.curr
        self.curr -= 1
        return self.end - curr

    @always_inline
    fn __has_more__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Scalar[type]:
        return self.curr

    @always_inline
    fn __getitem__(self, idx: Scalar[type]) -> Scalar[type]:
        debug_assert(idx < self.__len__(), "index out of range")
        return idx

    @always_inline
    fn __reversed__(self) -> _StridedScalarRange[type]:
        constrained[
            not type.is_unsigned(), "cannot reverse an unsigned range"
        ]()
        return range(self.end - 1, Scalar[type](-1), Scalar[type](-1))

    @always_inline
    fn __bool__(self) -> Bool:
        return self.__has_more__()


@value
@register_passable("trivial")
struct _SequentialScalarRange[type: DType]:
    var start: Scalar[type]
    var end: Scalar[type]

    @always_inline
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(inout self) -> Scalar[type]:
        var start = self.start
        self.start += 1
        return start

    @always_inline
    fn __has_more__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Scalar[type]:
        return max(0, self.end - self.start)

    @always_inline
    fn __getitem__(self, idx: Scalar[type]) -> Scalar[type]:
        debug_assert(idx < self.__len__(), "index out of range")
        return self.start + idx

    @always_inline
    fn __reversed__(self) -> _StridedRange:
        return range(self.end - 1, self.start - 1, -1)

    @always_inline
    fn __bool__(self) -> Bool:
        return self.__has_more__()


@value
@register_passable("trivial")
struct _StridedScalarRangeIterator[type: DType]:
    var start: Scalar[type]
    var end: Scalar[type]
    var step: Scalar[type]

    @always_inline
    fn __has_more__(self) -> Bool:
        # If the type is unsigned, then 'step' cannot be negative.
        @parameter
        if type.is_unsigned():
            return self.start < self.end
        else:
            if self.step > 0:
                return self.start < self.end
            return self.end < self.start

    @always_inline
    fn __next__(inout self) -> Scalar[type]:
        var result = self.start
        self.start += self.step
        return result

    @always_inline
    fn __bool__(self) -> Bool:
        return self.__has_more__()


@value
@register_passable("trivial")
struct _StridedScalarRange[type: DType]:
    var start: Scalar[type]
    var end: Scalar[type]
    var step: Scalar[type]

    @always_inline
    fn __iter__(self) -> _StridedScalarRangeIterator[type]:
        return _StridedScalarRangeIterator(self.start, self.end, self.step)


@always_inline
fn range[type: DType, //](end: Scalar[type]) -> _ZeroStartingScalarRange[type]:
    """Constructs a [start; end) Range with a given step.

    Parameters:
        type: The range type.

    Args:
        end: The end of the range.

    Returns:
        The constructed range.
    """
    return _ZeroStartingScalarRange(end)


@always_inline
fn range[
    type: DType, //
](start: Scalar[type], end: Scalar[type]) -> _SequentialScalarRange[type]:
    """Constructs a [start; end) Range with a given step.

    Parameters:
        type: The range type.

    Args:
        start: The start of the range.
        end: The end of the range.

    Returns:
        The constructed range.
    """
    return _SequentialScalarRange(start, end)


@always_inline
fn range[
    type: DType, //
](
    start: Scalar[type], end: Scalar[type], step: Scalar[type]
) -> _StridedScalarRange[type]:
    """Constructs a [start; end) Range with a given step.

    Parameters:
        type: The range type.

    Args:
        start: The start of the range.
        end: The end of the range.
        step: The step for the range.  Defaults to 1.

    Returns:
        The constructed range.
    """
    return _StridedScalarRange(start, end, step)


# ===----------------------------------------------------------------------=== #
# Utils
# ===----------------------------------------------------------------------=== #


trait _Iterator:
    fn __iter__(self) -> Self:
        ...


# TODO: this is the goal
# trait _Iterator[T: AnyType]:
#     fn __iter__(self) -> Self:
#         ...

#     fn __next__(inout self) -> T:
#         ...

#     fn __has_more__(self) -> Bool:
#         ...


# trait _Iterable[T: AnyType]:
#     fn __iter__(ref [_]self) -> _Iterator[T]:
#         ...


# fn iter[T: AnyType, I: _Iterable[T]](value: T) -> _Iterator[T]:
#     ...

# fn bool[T: AnyType, I: _Iterator[T]](value: I) -> Bool:
#     return value.__has_more__()


fn iter[T: _Iterator](value: T) -> T:
    """Get an iterator from the iterator.

    Parameters:
        T: The type conforming to _Iterator.

    Args:
        value: The value to get the iterator of.

    Returns:
        The iterator of the value.
    """
    return value.__iter__()


fn iter[
    T: DType
](value: _StridedScalarRange[T]) -> _StridedScalarRangeIterator[T]:
    """Return an iterator.

    Parameters:
        T: The type that the iterator yields.

    Args:
        value: The iterable value.

    Returns:
        The type's Iterator.
    """
    return value.__iter__()


@always_inline
fn iter[
    T: CollectionElement
](
    ref [_]value: List[T, *_],
) -> _ListIter[
    T, __type_of(value).hint_trivial_type, __origin_of(value)
]:
    """Return an iterator.

    Parameters:
        T: The type that the iterator yields.

    Args:
        value: The iterable value.

    Returns:
        The type's Iterator.
    """
    return value.__iter__()


fn iter[
    K: KeyElement,
    V: CollectionElement,
](ref [_]value: Dict[K, V],) -> _DictKeyIter[K, V, __origin_of(value), False]:
    """Get an iterator of the input dict.

    **Note**: iterators are currently non-raising.

    Parameters:
        K: The type of the keys in the dict.
        V: The type of the values in the dict.

    Args:
        value: The dict to get the iterator of.

    Returns:
        The iterator of the dict keys.
    """
    return value.__reversed__()


fn iter[
    K: KeyElement,
    V: CollectionElement,
    dict_mutability: Bool,
    dict_origin: Origin[dict_mutability].type,
](ref [_]value: _DictValueIter[K, V, dict_origin]) -> _DictValueIter[
    K, V, dict_origin
]:
    """Get an iterator of the input dict values.

    **Note**: iterators are currently non-raising.

    Parameters:
        K: The type of the keys in the dict.
        V: The type of the values in the dict.
        dict_mutability: Whether the reference to the dict values is mutable.
        dict_origin: The origin of the dict values.

    Args:
        value: The dict values to get the iterator of.

    Returns:
        The iterator of the dict values.
    """
    return value.__iter__()


fn iter[
    K: KeyElement,
    V: CollectionElement,
    dict_mutability: Bool,
    dict_origin: Origin[dict_mutability].type,
](ref [_]value: _DictEntryIter[K, V, dict_origin]) -> _DictEntryIter[
    K, V, dict_origin
]:
    """Get an iterator of the input dict items.

    **Note**: iterators are currently non-raising.

    Parameters:
        K: The type of the keys in the dict.
        V: The type of the values in the dict.
        dict_mutability: Whether the reference to the dict items is mutable.
        dict_origin: The origin of the dict items.

    Args:
        value: The dict items to get the iterator of.

    Returns:
        The iterator of the dict items.
    """
    var src = value.src
    return _DictEntryIter[K, V, dict_origin](src[]._reserved() - 1, 0, src)


@always_inline
fn iter[T: Stringlike](ref [_]value: T) -> _StringSliceIter[__origin_of(value)]:
    """Return an iterator.

    Parameters:
        T: The type that the iterator yields.

    Args:
        value: The iterable value.

    Returns:
        The type's Iterator.
    """
    return value.__iter__()
