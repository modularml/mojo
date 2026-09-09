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
"""Provides iterator utilities for common iteration patterns.

This module includes functions for creating specialized iterators:

- `count()` - Creates an infinite counter with customizable start and step values
- `cycle()` - Cycles through an iterable indefinitely
- `drop_while()` - Drops elements while predicate is true, then yields the rest
- `product()` - Computes the Cartesian product of two, three, or four iterables
- `repeat()` - Repeats an element a specified number of times
- `drop()` - Drops the first n elements and yields the rest
- `take()` - Yields the first n elements
- `take_while()` - Yields elements while predicate is true

These utilities enable functional-style iteration patterns and composable iterator
operations.
"""

from std.memory import forget_deinit

# ===-----------------------------------------------------------------------===#
# count
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _CountIterator(
    Iterable, IterableOwned, Iterator, TrivialRegisterPassable
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self
    comptime Element = Int
    var start: Int
    var step: Int

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self

    @always_inline
    def __next__(mut self) raises StopIteration -> Int:
        var result = self.start
        self.start += self.step
        return result


@always_inline
def count(start: Int = 0, step: Int = 1) -> _CountIterator:
    """Constructs an iterator that starts at the value `start` with a stride of
    `step`.

    Args:
        start: The start of the iterator.
        step: The stride of the iterator.

    Returns:
        The constructed iterator.
    """
    return {start, step}


# ===-----------------------------------------------------------------------===#
# product
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _Product2[IteratorTypeA: Iterator, IteratorTypeB: Copyable & Iterator](
    Copyable where conforms_to(IteratorTypeA, Copyable),
    Iterable where conforms_to(IteratorTypeA, Copyable),
    IterableOwned,
    Iterator,
) where conforms_to(IteratorTypeA.Element, Deinitable):
    comptime Element = Tuple[
        Self.IteratorTypeA.Element, Self.IteratorTypeB.Element
    ]
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self

    var _inner_a: Self.IteratorTypeA
    var _inner_b: Self.IteratorTypeB
    var _inner_a_elem: Optional[Self.IteratorTypeA.Element]
    var _initial_inner_b: Self.IteratorTypeB

    def __init__(
        out self,
        var inner_a: Self.IteratorTypeA,
        var inner_b: Self.IteratorTypeB,
    ):
        self._inner_a = inner_a^
        self._inner_b = inner_b.copy()
        self._inner_a_elem = None
        self._initial_inner_b = inner_b^

    def __init__(
        out self, *, copy: Self
    ) where conforms_to(Self.IteratorTypeA, Copyable):
        comptime assert conforms_to(Self.IteratorTypeA.Element, Copyable)

        self._inner_a = copy._inner_a.copy()
        self._inner_b = copy._inner_b.copy()
        self._inner_a_elem = copy._inner_a_elem.copy()
        self._initial_inner_b = copy._initial_inner_b.copy()

    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)] where conforms_to(
        Self.IteratorTypeA, Copyable
    ):
        return self.copy()

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __next__(mut self) raises StopIteration -> Self.Element:
        comptime assert conforms_to(Self.IteratorTypeA.Element, Copyable)

        # Take the first element from 'a' if we haven't got it yet.
        if not self._inner_a_elem:
            self._inner_a_elem = next(self._inner_a)

        try:
            # Get the next element from 'b' if it exists.
            var b_val = next(self._inner_b)

            var elem = self._inner_a_elem.unsafe_value().copy()
            return elem^, b_val^
        except:
            # reset if we reach the end of the B iterator and grab the next
            # item from the A iterator.
            self._inner_b = self._initial_inner_b.copy()
            self._inner_a_elem = next(self._inner_a)
            var b_val = next(self._inner_b)
            # If a and b iterators had more elements, return this one.
            var elem = self._inner_a_elem.unsafe_value().copy()
            return elem^, b_val^

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        # compute a * initial_b + b for lower and upper

        var a_bounds = self._inner_a.bounds()
        var b_bounds = self._inner_b.bounds() if self._inner_a_elem else (
            0,
            Optional[Int](0),
        )
        var initial_b_bounds = self._initial_inner_b.bounds()

        var lower_bound = a_bounds[0] * initial_b_bounds[0] + b_bounds[0]
        if not a_bounds[1] or not initial_b_bounds[1]:
            return (lower_bound, None)

        var upper_bound = a_bounds[1].unsafe_value() * initial_b_bounds[
            1
        ].unsafe_value() + b_bounds[1].or_else(0)
        return (lower_bound, upper_bound)


@always_inline
def product[
    IterableTypeA: Iterable, IterableTypeB: Iterable
](ref iterable_a: IterableTypeA, ref iterable_b: IterableTypeB) -> _Product2[
    IterableTypeA.IteratorType[origin_of(iterable_a)],
    IterableTypeB.IteratorType[origin_of(iterable_b)],
] where conforms_to(
    IterableTypeA.IteratorType[origin_of(iterable_a)].Element,
    Deinitable,
) and conforms_to(
    IterableTypeB.IteratorType[origin_of(iterable_b)], Copyable & Iterator
):
    """Returns an iterator that yields tuples of the elements of the outer
    product of the iterables.

    Parameters:
        IterableTypeA: The type of the first iterable.
        IterableTypeB: The type of the second iterable.

    Args:
        iterable_a: The first iterable.
        iterable_b: The second iterable.

    Returns:
        A product iterator that yields outer product tuples of elements from both
        iterables.

    Examples:

    ```mojo
    from std.itertools import product

    var l = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]
    for a, b in product(l, l2):
        print(a, b)
    ```
    """
    return {
        iter(iterable_a),
        iter(iterable_b),
    }


def _flatten[
    T: Movable, //, *InnerTs: Movable
](
    var arg: Tuple[T, Tuple[*InnerTs]],
    # TODO(MOCO-4359): Use `out result: Tuple[T, *InnerTs]` once trailing
    # unpacking is supported.
    out result: Tuple[
        *TypeList._concat[
            TypeList.of[Trait=Movable, T]().values,
            TypeList[Trait=Movable, InnerTs.values]().values,
        ]()
    ],
):
    """Right-flattens `(a, (b, c, ...))` into `(a, b, c, ...)`.

    The nested `product` iterators build their tuples one pair at a time, so
    each `__next__` yields a right-nested tuple that has to be flattened. The
    element types are only known to be `Movable` and `Tuple` is only
    conditionally `Deinitable`, so `arg` can't be dropped implicitly.
    Each element is moved into `result` instead, then `arg` is discarded
    without running a destructor.
    """
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(result))

    Pointer(to=result[0]).unsafe_write_move_from(
        rebind[Pointer[type_of(result[0]), origin_of(arg)]](Pointer(to=arg[0]))
    )

    comptime for j in range(type_of(arg[1]).__len__()):
        Pointer(to=result[j + 1]).unsafe_write_move_from(
            rebind[Pointer[type_of(result[j + 1]), origin_of(arg)]](
                Pointer(to=arg[1][j])
            )
        )

    # Every element has been moved out and the `!kgen.struct` destructor is
    # trivial, so discard `arg` without running a destructor.
    forget_deinit(arg^)


# ===-----------------------------------------------------------------------===#
# product (3 iterables)
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _Product3[
    IteratorTypeA: Iterator,
    IteratorTypeB: Copyable & Iterator,
    IteratorTypeC: Copyable & Iterator,
](
    Copyable where conforms_to(IteratorTypeA, Copyable),
    Iterable where conforms_to(IteratorTypeA, Copyable),
    IterableOwned,
    Iterator,
) where conforms_to(
    IteratorTypeA.Element, Deinitable
) and conforms_to(
    IteratorTypeB.Element, Deinitable
):
    comptime Element = Tuple[
        Self.IteratorTypeA.Element,
        Self.IteratorTypeB.Element,
        Self.IteratorTypeC.Element,
    ]
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self

    comptime _Product2Type = _Product2[Self.IteratorTypeB, Self.IteratorTypeC]
    comptime _OuterProduct2Type = _Product2[
        Self.IteratorTypeA, Self._Product2Type
    ]

    var _inner: Self._OuterProduct2Type

    def __init__(
        out self,
        var inner_a: Self.IteratorTypeA,
        var inner_b: Self.IteratorTypeB,
        var inner_c: Self.IteratorTypeC,
    ):
        var product2 = Self._Product2Type(inner_b^, inner_c^)
        self._inner = Self._OuterProduct2Type(inner_a^, product2^)

    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)] where conforms_to(
        Self.IteratorTypeA, Copyable
    ):
        return self.copy()

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __next__(mut self) raises StopIteration -> Self.Element:
        # `next` yields the right-nested `(a, (b, c))`; flatten to `(a, b, c)`.
        return rebind_var[Self.Element](_flatten(next(self._inner)))

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return self._inner.bounds()


@always_inline
def product[
    IterableTypeA: Iterable, IterableTypeB: Iterable, IterableTypeC: Iterable
](
    ref iterable_a: IterableTypeA,
    ref iterable_b: IterableTypeB,
    ref iterable_c: IterableTypeC,
) -> _Product3[
    IterableTypeA.IteratorType[origin_of(iterable_a)],
    IterableTypeB.IteratorType[origin_of(iterable_b)],
    IterableTypeC.IteratorType[origin_of(iterable_c)],
] where (
    conforms_to(
        IterableTypeB.IteratorType[origin_of(iterable_b)], Copyable & Iterator
    )
    and conforms_to(
        IterableTypeC.IteratorType[origin_of(iterable_c)], Copyable & Iterator
    )
    and conforms_to(
        IterableTypeA.IteratorType[origin_of(iterable_a)].Element,
        Deinitable,
    )
    and conforms_to(
        IterableTypeB.IteratorType[origin_of(iterable_b)].Element,
        Deinitable,
    )
):
    """Returns an iterator that yields tuples of the elements of the outer
    product of three iterables.

    Parameters:
        IterableTypeA: The type of the first iterable.
        IterableTypeB: The type of the second iterable.
        IterableTypeC: The type of the third iterable.

    Args:
        iterable_a: The first iterable.
        iterable_b: The second iterable.
        iterable_c: The third iterable.

    Returns:
        A product iterator that yields outer product tuples of elements from all
        three iterables.

    Examples:

    ```mojo
    from std.itertools import product

    var l1 = [1, 2]
    var l2 = [3, 4]
    var l3 = [5, 6]
    for a, b, c in product(l1, l2, l3):
        print(a, b, c)
    ```
    """
    return {
        iter(iterable_a),
        iter(iterable_b),
        iter(iterable_c),
    }


# ===-----------------------------------------------------------------------===#
# product (4 iterables)
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _Product4[
    IteratorTypeA: Iterator,
    IteratorTypeB: Copyable & Iterator,
    IteratorTypeC: Copyable & Iterator,
    IteratorTypeD: Copyable & Iterator,
](
    Copyable where conforms_to(IteratorTypeA, Copyable),
    Iterable where conforms_to(IteratorTypeA, Copyable),
    IterableOwned,
    Iterator,
) where (
    conforms_to(IteratorTypeA.Element, Deinitable)
    and conforms_to(IteratorTypeB.Element, Deinitable)
    and conforms_to(IteratorTypeC.Element, Deinitable)
):
    comptime Element = Tuple[
        Self.IteratorTypeA.Element,
        Self.IteratorTypeB.Element,
        Self.IteratorTypeC.Element,
        Self.IteratorTypeD.Element,
    ]
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self

    comptime _Product3Type = _Product3[
        Self.IteratorTypeB, Self.IteratorTypeC, Self.IteratorTypeD
    ]
    comptime _Product2Type = _Product2[Self.IteratorTypeA, Self._Product3Type]

    var _inner: Self._Product2Type

    def __init__(
        out self,
        var inner_a: Self.IteratorTypeA,
        var inner_b: Self.IteratorTypeB,
        var inner_c: Self.IteratorTypeC,
        var inner_d: Self.IteratorTypeD,
    ):
        var product3 = Self._Product3Type(inner_b^, inner_c^, inner_d^)
        self._inner = Self._Product2Type(inner_a^, product3^)

    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)] where conforms_to(
        Self.IteratorTypeA, Copyable
    ):
        return self.copy()

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __next__(mut self) raises StopIteration -> Self.Element:
        # `next` yields the right-nested `(a, (b, c, d))`; flatten it.
        return rebind_var[Self.Element](_flatten(next(self._inner)))

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return self._inner.bounds()


@always_inline
def product[
    IterableTypeA: Iterable,
    IterableTypeB: Iterable,
    IterableTypeC: Iterable,
    IterableTypeD: Iterable,
](
    ref iterable_a: IterableTypeA,
    ref iterable_b: IterableTypeB,
    ref iterable_c: IterableTypeC,
    ref iterable_d: IterableTypeD,
) -> _Product4[
    IterableTypeA.IteratorType[origin_of(iterable_a)],
    IterableTypeB.IteratorType[origin_of(iterable_b)],
    IterableTypeC.IteratorType[origin_of(iterable_c)],
    IterableTypeD.IteratorType[origin_of(iterable_d)],
] where (
    conforms_to(
        IterableTypeB.IteratorType[origin_of(iterable_b)], Copyable & Iterator
    )
    and conforms_to(
        IterableTypeC.IteratorType[origin_of(iterable_c)], Copyable & Iterator
    )
    and conforms_to(
        IterableTypeD.IteratorType[origin_of(iterable_d)], Copyable & Iterator
    )
    and conforms_to(
        IterableTypeA.IteratorType[origin_of(iterable_a)].Element,
        Deinitable,
    )
    and conforms_to(
        IterableTypeB.IteratorType[origin_of(iterable_b)].Element,
        Deinitable,
    )
    and conforms_to(
        IterableTypeC.IteratorType[origin_of(iterable_c)].Element,
        Deinitable,
    )
):
    """Returns an iterator that yields tuples of the elements of the outer
    product of four iterables.

    Parameters:
        IterableTypeA: The type of the first iterable.
        IterableTypeB: The type of the second iterable.
        IterableTypeC: The type of the third iterable.
        IterableTypeD: The type of the fourth iterable.

    Args:
        iterable_a: The first iterable.
        iterable_b: The second iterable.
        iterable_c: The third iterable.
        iterable_d: The fourth iterable.

    Returns:
        A product iterator that yields outer product tuples of elements from all
        four iterables.

    Examples:

    ```mojo
    from std.itertools import product

    var l1 = [1, 2]
    var l2 = [3, 4]
    var l3 = [5, 6]
    var l4 = [7, 8]
    for a, b, c, d in product(l1, l2, l3, l4):
        print(a, b, c, d)
    ```
    """
    return {
        iter(iterable_a),
        iter(iterable_b),
        iter(iterable_c),
        iter(iterable_d),
    }


@always_inline
def product(
    var iterable_a: Some[IterableOwned],
    var iterable_b: Some[IterableOwned],
) -> _Product2[
    type_of(iterable_a).IteratorOwnedType,
    type_of(iterable_b).IteratorOwnedType,
] where conforms_to(
    type_of(iterable_b).IteratorOwnedType, Copyable & Iterator
) and conforms_to(
    type_of(iterable_a).IteratorOwnedType.Element, Deinitable
):
    """Returns an iterator that yields tuples of the elements of the outer
    product of the iterables, consuming both iterables.

    Args:
        iterable_a: The first iterable to consume.
        iterable_b: The second iterable to consume.

    Returns:
        A product iterator that yields outer product tuples of elements from both
        iterables.
    """
    return {
        iter(iterable_a^),
        iter(iterable_b^),
    }


@always_inline
def product(
    var iterable_a: Some[IterableOwned],
    var iterable_b: Some[IterableOwned],
    var iterable_c: Some[IterableOwned],
) -> _Product3[
    type_of(iterable_a).IteratorOwnedType,
    type_of(iterable_b).IteratorOwnedType,
    type_of(iterable_c).IteratorOwnedType,
] where (
    conforms_to(type_of(iterable_b).IteratorOwnedType, Copyable & Iterator)
    and conforms_to(type_of(iterable_c).IteratorOwnedType, Copyable & Iterator)
    and conforms_to(
        type_of(iterable_a).IteratorOwnedType.Element,
        Deinitable,
    )
    and conforms_to(
        type_of(iterable_b).IteratorOwnedType.Element,
        Deinitable,
    )
):
    """Returns an iterator that yields tuples of the elements of the outer
    product of three iterables, consuming all three iterables.

    Args:
        iterable_a: The first iterable to consume.
        iterable_b: The second iterable to consume.
        iterable_c: The third iterable to consume.

    Returns:
        A product iterator that yields outer product tuples of elements from all
        three iterables.
    """
    return {
        iter(iterable_a^),
        iter(iterable_b^),
        iter(iterable_c^),
    }


@always_inline
def product(
    var iterable_a: Some[IterableOwned],
    var iterable_b: Some[IterableOwned],
    var iterable_c: Some[IterableOwned],
    var iterable_d: Some[IterableOwned],
) -> _Product4[
    type_of(iterable_a).IteratorOwnedType,
    type_of(iterable_b).IteratorOwnedType,
    type_of(iterable_c).IteratorOwnedType,
    type_of(iterable_d).IteratorOwnedType,
] where (
    conforms_to(type_of(iterable_b).IteratorOwnedType, Copyable & Iterator)
    and conforms_to(type_of(iterable_c).IteratorOwnedType, Copyable & Iterator)
    and conforms_to(type_of(iterable_d).IteratorOwnedType, Copyable & Iterator)
    and conforms_to(
        type_of(iterable_a).IteratorOwnedType.Element,
        Deinitable,
    )
    and conforms_to(
        type_of(iterable_b).IteratorOwnedType.Element,
        Deinitable,
    )
    and conforms_to(
        type_of(iterable_c).IteratorOwnedType.Element,
        Deinitable,
    )
):
    """Returns an iterator that yields tuples of the elements of the outer
    product of four iterables, consuming all four iterables.

    Args:
        iterable_a: The first iterable to consume.
        iterable_b: The second iterable to consume.
        iterable_c: The third iterable to consume.
        iterable_d: The fourth iterable to consume.

    Returns:
        A product iterator that yields outer product tuples of elements from all
        four iterables.
    """
    return {
        iter(iterable_a^),
        iter(iterable_b^),
        iter(iterable_c^),
        iter(iterable_d^),
    }


# ===-----------------------------------------------------------------------===#
# cycle
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _CycleIterator[InnerIteratorType: Iterator & Copyable](
    Copyable, Iterable, IterableOwned, Iterator
):
    """Iterator that cycles through an iterable indefinitely.

    This iterator keeps a copy of the original iterator and resets to it
    when the current iterator is exhausted. This is a lazy implementation
    that does no work at construction time.

    Parameters:
        InnerIteratorType: The type of the inner iterator.
    """

    comptime Element = Self.InnerIteratorType.Element
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self

    var _orig: Self.InnerIteratorType
    var _iter: Self.InnerIteratorType

    def __init__(out self, var iterator: Self.InnerIteratorType):
        """Creates a cycle iterator from an iterator.

        Args:
            iterator: The iterator to cycle through.
        """
        self._orig = iterator.copy()
        self._iter = iterator^

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    @always_inline
    def __next__(mut self) raises StopIteration -> Self.Element:
        try:
            return next(self._iter)
        except StopIteration:
            # Reset to original and try again
            self._iter = self._orig.copy()
            # If this also raises StopIteration, the original was empty
            return next(self._iter)


@always_inline
def cycle[
    IterableType: Iterable
](ref iterable: IterableType) -> _CycleIterator[
    type_of(iter(iterable))
] where conforms_to(type_of(iter(iterable)), Copyable & Iterator):
    """Creates an iterator that cycles through an iterable indefinitely.

    This function returns an iterator that yields elements from the input
    iterable repeatedly in an infinite loop. The elements are yielded in
    the same order as they appear in the original iterable.

    This is a lazy implementation - no work is done at construction time.
    The iterator keeps a copy of the original iterator and resets to it
    when exhausted.

    Parameters:
        IterableType: The type of the iterable.

    Args:
        iterable: The iterable to cycle through.

    Returns:
        An iterator that yields elements from the iterable forever.

    Examples:

    ```mojo
    from std.itertools import cycle

    # Cycle through a list
    var colors = ["red", "green", "blue"]
    var color_cycle = cycle(colors)

    # Get 6 elements (cycles twice)
    var count = 0
    for color in color_cycle:
        print(color)
        count += 1
        if count >= 6:
            break
    # Output: red, green, blue, red, green, blue
    ```
    """
    return _CycleIterator(iter(iterable))


@always_inline
def cycle(
    var iterable: Some[IterableOwned],
) -> _CycleIterator[type_of(iterable).IteratorOwnedType] where conforms_to(
    type_of(iterable).IteratorOwnedType, Copyable & Iterator
):
    """Creates an iterator that cycles through an iterable indefinitely,
    consuming the iterable.

    Args:
        iterable: The iterable to consume and cycle through.

    Returns:
        An iterator that yields elements from the iterable forever.
    """
    return _CycleIterator(iter(iterable^))


# ===-----------------------------------------------------------------------===#
# take_while
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _TakeWhileIterator[
    InnerIteratorType: Iterator,
    //,
    predicate: def(InnerIteratorType.Element) thin -> Bool,
](
    Copyable where conforms_to(InnerIteratorType, Copyable),
    Iterable where conforms_to(InnerIteratorType, Copyable),
    IterableOwned,
    Iterator,
):
    """Iterator that yields elements while predicate returns True.

    Parameters:
        InnerIteratorType: The type of the inner iterator.
        predicate: A function that takes an element and returns True if the
            element should be yielded.
    """

    comptime Element = Self.InnerIteratorType.Element
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self

    var _inner: Self.InnerIteratorType
    var _exhausted: Bool

    def __init__(out self, var inner: Self.InnerIteratorType):
        """Creates a take_while iterator from an inner iterator.

        Args:
            inner: The inner iterator to wrap.
        """
        self._inner = inner^
        self._exhausted = False

    def __init__(
        out self, *, copy: Self
    ) where conforms_to(Self.InnerIteratorType, Copyable):
        self._inner = copy._inner.copy()
        self._exhausted = copy._exhausted

    @always_inline
    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)] where conforms_to(
        Self.InnerIteratorType, Copyable
    ):
        return self.copy()

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    @always_inline
    def __next__(mut self) raises StopIteration -> Self.Element:
        comptime assert conforms_to(Self.Element, Deinitable)

        if self._exhausted:
            raise StopIteration()
        var elem = next(self._inner)
        if not Self.predicate(elem):
            self._exhausted = True
            # Discard the element that failed the predicate
            _ = elem^
            raise StopIteration()
        return elem^


@always_inline
def take_while[
    origin: ImmOrigin,
    IterableType: Iterable,
    //,
    predicate: def(IterableType.IteratorType[origin].Element) thin -> Bool,
](ref[origin] iterable: IterableType) -> _TakeWhileIterator[
    InnerIteratorType=IterableType.IteratorType[origin],
    predicate=predicate,
] where conforms_to(IterableType.IteratorType[origin].Element, Deinitable,):
    """Creates an iterator that yields elements while predicate returns True.

    This function returns an iterator that yields elements from the input
    iterable as long as the predicate function returns True for each element.
    Once the predicate returns False, the iterator stops immediately and does
    not yield any more elements.

    Parameters:
        origin: The origin of the iterable.
        IterableType: The type of the iterable.
        predicate: A function that takes an element and returns True if the
            element should be yielded.

    Args:
        iterable: The iterable to take elements from.

    Returns:
        An iterator that yields elements while predicate returns True.

    Examples:

    ```mojo
    from std.itertools import take_while

    var nums = [1, 2, 3, 4, 5, 6, 7]
    for num in take_while[lambda (x: Int) -> Bool: x < 5](nums):
        print(num)  # Prints: 1, 2, 3, 4
    ```
    """
    # FIXME(MOCO-3238): This rebind shouldn't ve needed, something isn't getting
    # substituted through associated types right.
    return _TakeWhileIterator[predicate=predicate](
        rebind_var[IterableType.IteratorType[origin]](iter(iterable))
    )


@always_inline
def take_while[
    IterableType: IterableOwned,
    //,
    predicate: def(IterableType.IteratorOwnedType.Element) thin -> Bool,
](var iterable: IterableType) -> _TakeWhileIterator[
    InnerIteratorType=IterableType.IteratorOwnedType,
    predicate=predicate,
] where conforms_to(IterableType.IteratorOwnedType.Element, Deinitable,):
    """Creates an iterator that yields elements while predicate returns True,
    consuming the iterable.

    Parameters:
        IterableType: The type of the iterable.
        predicate: A function that takes an element and returns True if the
            element should be yielded.

    Args:
        iterable: The iterable to consume and take elements from.

    Returns:
        An iterator that yields elements while predicate returns True.
    """
    return _TakeWhileIterator[predicate=predicate](iter(iterable^))


# ===-----------------------------------------------------------------------===#
# drop_while
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _DropWhileIterator[
    InnerIteratorType: Iterator,
    //,
    predicate: def(InnerIteratorType.Element) thin -> Bool,
](
    Copyable where conforms_to(InnerIteratorType, Copyable),
    Iterable where conforms_to(InnerIteratorType, Copyable),
    IterableOwned,
    Iterator,
):
    """Iterator that drops elements while predicate returns True, then yields rest.

    Parameters:
        InnerIteratorType: The type of the inner iterator.
        predicate: A function that takes an element and returns True if the
            element should be dropped.
    """

    comptime Element = Self.InnerIteratorType.Element
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self

    var _inner: Self.InnerIteratorType
    var _dropping: Bool

    def __init__(out self, var inner: Self.InnerIteratorType):
        """Creates a drop_while iterator from an inner iterator.

        Args:
            inner: The inner iterator to wrap.
        """
        self._inner = inner^
        self._dropping = True

    def __init__(
        out self, *, copy: Self
    ) where conforms_to(Self.InnerIteratorType, Copyable):
        self._inner = copy._inner.copy()
        self._dropping = copy._dropping

    @always_inline
    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)] where conforms_to(
        Self.InnerIteratorType, Copyable
    ):
        return self.copy()

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    @always_inline
    def __next__(mut self) raises StopIteration -> Self.Element:
        comptime assert conforms_to(Self.Element, Deinitable)

        if self._dropping:
            while True:
                var elem = next(self._inner)
                if Self.predicate(elem):
                    # Discard the element that matched the predicate
                    _ = elem^
                    continue
                self._dropping = False
                return elem^
        return next(self._inner)


@always_inline
def drop_while[
    origin: ImmOrigin,
    IterableType: Iterable,
    //,
    predicate: def(IterableType.IteratorType[origin].Element) thin -> Bool,
](ref[origin] iterable: IterableType) -> _DropWhileIterator[
    InnerIteratorType=IterableType.IteratorType[origin],
    predicate=predicate,
] where conforms_to(IterableType.IteratorType[origin].Element, Deinitable,):
    """Creates an iterator that drops elements while predicate returns True.

    This function returns an iterator that drops elements from the input
    iterable as long as the predicate function returns True for each element.
    Once the predicate returns False, the iterator starts yielding all
    remaining elements unconditionally.

    Parameters:
        origin: The origin of the iterable.
        IterableType: The type of the iterable.
        predicate: A function that takes an element and returns True if the
            element should be dropped.

    Args:
        iterable: The iterable to drop elements from.

    Returns:
        An iterator that drops elements while predicate returns True, then
        yields all remaining elements.

    Examples:

    ```mojo
    from std.itertools import drop_while

    var nums = [1, 2, 3, 4, 5, 6, 1, 2]
    for num in drop_while[lambda (x: Int) -> Bool: x < 5](nums):
        print(num)  # Prints: 5, 6, 1, 2
    ```
    """
    # FIXME(MOCO-3238): This rebind shouldn't ve needed, something isn't getting
    # substituted through associated types right.
    return _DropWhileIterator[predicate=predicate](
        rebind_var[IterableType.IteratorType[origin]](iter(iterable))
    )


@always_inline
def drop_while[
    IterableType: IterableOwned,
    //,
    predicate: def(IterableType.IteratorOwnedType.Element) thin -> Bool,
](var iterable: IterableType) -> _DropWhileIterator[
    InnerIteratorType=IterableType.IteratorOwnedType,
    predicate=predicate,
] where conforms_to(IterableType.IteratorOwnedType.Element, Deinitable,):
    """Creates an iterator that drops elements while predicate returns True,
    consuming the iterable.

    Parameters:
        IterableType: The type of the iterable.
        predicate: A function that takes an element and returns True if the
            element should be dropped.

    Args:
        iterable: The iterable to consume and drop elements from.

    Returns:
        An iterator that drops elements while predicate returns True, then
        yields all remaining elements.
    """
    return _DropWhileIterator[predicate=predicate](iter(iterable^))


# ===-----------------------------------------------------------------------===#
# repeat
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _RepeatIterator[ElementType: Copyable & Deinitable](
    Copyable, Iterable, IterableOwned, Iterator
):
    """Iterator that repeats an element a specified number of times.

    Parameters:
        ElementType: The type of the element to repeat.
    """

    comptime Element = Self.ElementType
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self

    var element: Self.ElementType
    var remaining: Int

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    @always_inline
    def __next__(mut self) raises StopIteration -> Self.ElementType:
        if self.remaining <= 0:
            raise StopIteration()
        self.remaining -= 1
        return self.element.copy()


@always_inline
def repeat[
    ElementType: Copyable & Deinitable
](element: ElementType, *, times: Int) -> _RepeatIterator[ElementType]:
    """Constructs an iterator that repeats the given element a specified number of times.

    This function creates an iterator that returns the same element over and over
    for the specified number of times.

    Parameters:
        ElementType: The type of the element to repeat.

    Args:
        element: The element to repeat.
        times: The number of times to repeat the element.

    Returns:
        An iterator that repeats the element the specified number of times.

    Examples:

    ```mojo
    from std.itertools import repeat

    # Repeat a value 3 times
    var it = repeat(42, times=3)
    for val in it:
        print(val)  # Prints: 42, 42, 42

    # Repeat a string 5 times
    var str_it = repeat("hello", times=5)
    for s in str_it:
        print(s)  # Prints: hello, hello, hello, hello, hello
    ```
    """
    assert times >= 0, "The `times` argument must be non-negative"
    return {element.copy(), times}


# ===-----------------------------------------------------------------------===#
# take
# ===-----------------------------------------------------------------------===#


struct _TakeIterator[InnerIteratorType: Iterator](
    Copyable where conforms_to(InnerIteratorType, Copyable),
    Iterable where conforms_to(InnerIteratorType, Copyable),
    IterableOwned,
    Iterator,
):
    """Iterator that yields the first `n` elements from an inner iterator.

    Parameters:
        InnerIteratorType: The type of the inner iterator.
    """

    comptime Element = Self.InnerIteratorType.Element
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self

    var _inner: Self.InnerIteratorType
    var _remaining: Int

    def __init__(out self, var inner: Self.InnerIteratorType, *, count: Int):
        """Creates a take iterator.

        Args:
            inner: The inner iterator to wrap.
            count: The maximum number of elements to yield.
        """
        self._inner = inner^
        self._remaining = count

    def __init__(
        out self, *, copy: Self
    ) where conforms_to(Self.InnerIteratorType, Copyable):
        self._inner = copy._inner.copy()
        self._remaining = copy._remaining

    @always_inline
    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)] where conforms_to(
        Self.InnerIteratorType, Copyable
    ):
        return self.copy()

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    @always_inline
    def __next__(mut self) raises StopIteration -> Self.Element:
        if self._remaining <= 0:
            raise StopIteration()
        var value = next(self._inner)
        self._remaining -= 1
        return value^

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var remaining = max(0, self._remaining)
        var lower, upper = self._inner.bounds()
        lower = min(lower, remaining)
        if upper:
            return (lower, min(upper.value(), remaining))
        return (lower, remaining)


@always_inline
def take[
    origin: ImmOrigin,
    IterableType: Iterable,
    //,
](ref[origin] iterable: IterableType, count: Int) -> _TakeIterator[
    IterableType.IteratorType[origin]
]:
    """Creates an iterator that yields the first `count` elements.

    This function returns an iterator that yields at most `count` elements
    from the input iterable, then stops.

    Parameters:
        origin: The origin of the iterable.
        IterableType: The type of the iterable.

    Args:
        iterable: The iterable to take elements from.
        count: The maximum number of elements to yield.

    Returns:
        An iterator that yields at most `count` elements.

    Examples:

    ```mojo
    from std.itertools import take

    var nums = [1, 2, 3, 4, 5]
    for num in take(nums, 3):
        print(num)  # Prints: 1, 2, 3
    ```
    """
    assert count >= 0, "The `count` argument must be non-negative"
    # Unlike `drop` and `take_while`, `take` has no `Deinitable`
    # constraint on the element type: it never discards an element, it just
    # stops yielding once `count` is reached.
    # FIXME(MOCO-3238): This rebind shouldn't be needed, something isn't getting
    # substituted through associated types right.
    return _TakeIterator(
        rebind_var[IterableType.IteratorType[origin]](iter(iterable)),
        count=count,
    )


@always_inline
def take[
    IterableType: IterableOwned, //
](var iterable: IterableType, count: Int) -> _TakeIterator[
    IterableType.IteratorOwnedType
]:
    """Creates an iterator that yields the first `count` elements, consuming
    the iterable.

    Parameters:
        IterableType: The type of the iterable.

    Args:
        iterable: The iterable to consume and take elements from.
        count: The maximum number of elements to yield.

    Returns:
        An iterator that yields at most `count` elements.
    """
    assert count >= 0, "The `count` argument must be non-negative"
    return _TakeIterator(iter(iterable^), count=count)


# ===-----------------------------------------------------------------------===#
# drop
# ===-----------------------------------------------------------------------===#


struct _DropIterator[InnerIteratorType: Iterator](
    Copyable where conforms_to(InnerIteratorType, Copyable),
    Iterable where conforms_to(InnerIteratorType, Copyable),
    IterableOwned,
    Iterator,
):
    """Iterator that drops the first `n` elements and yields the rest.

    Parameters:
        InnerIteratorType: The type of the inner iterator.
    """

    comptime Element = Self.InnerIteratorType.Element
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self

    var _inner: Self.InnerIteratorType
    var _to_drop: Int

    def __init__(out self, var inner: Self.InnerIteratorType, *, count: Int):
        """Creates a drop iterator.

        Args:
            inner: The inner iterator to wrap.
            count: The number of elements to drop.
        """
        self._inner = inner^
        self._to_drop = count

    def __init__(
        out self, *, copy: Self
    ) where conforms_to(Self.InnerIteratorType, Copyable):
        self._inner = copy._inner.copy()
        self._to_drop = copy._to_drop

    @always_inline
    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)] where conforms_to(
        Self.InnerIteratorType, Copyable
    ):
        return self.copy()

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    @always_inline
    def __next__(mut self) raises StopIteration -> Self.Element:
        comptime assert conforms_to(Self.Element, Deinitable)

        while self._to_drop > 0:
            # Discard dropped elements. If `next` raises, `_to_drop` is not
            # decremented, leaving the iterator in a consistent exhausted
            # state.
            var elem = next(self._inner)
            _ = elem^
            self._to_drop -= 1
        return next(self._inner)

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var to_drop = max(0, self._to_drop)
        var lower, upper = self._inner.bounds()
        lower = max(0, lower - to_drop)
        if upper:
            return (lower, max(0, upper.value() - to_drop))
        return (lower, None)


@always_inline
def drop[
    origin: ImmOrigin,
    IterableType: Iterable,
    //,
](ref[origin] iterable: IterableType, count: Int) -> _DropIterator[
    IterableType.IteratorType[origin]
] where conforms_to(IterableType.IteratorType[origin].Element, Deinitable,):
    """Creates an iterator that drops the first `count` elements.

    This function returns an iterator that drops the first `count` elements
    from the input iterable, then yields all remaining elements.

    Parameters:
        origin: The origin of the iterable.
        IterableType: The type of the iterable.

    Args:
        iterable: The iterable to drop elements from.
        count: The number of elements to drop.

    Returns:
        An iterator that drops the first `count` elements.

    Examples:

    ```mojo
    from std.itertools import drop

    var nums = [1, 2, 3, 4, 5]
    for num in drop(nums, 2):
        print(num)  # Prints: 3, 4, 5
    ```
    """
    assert count >= 0, "The `count` argument must be non-negative"
    # FIXME(MOCO-3238): This rebind shouldn't be needed, something isn't getting
    # substituted through associated types right.
    return _DropIterator(
        rebind_var[IterableType.IteratorType[origin]](iter(iterable)),
        count=count,
    )


@always_inline
def drop[
    IterableType: IterableOwned, //
](var iterable: IterableType, count: Int) -> _DropIterator[
    IterableType.IteratorOwnedType
] where conforms_to(IterableType.IteratorOwnedType.Element, Deinitable,):
    """Creates an iterator that drops the first `count` elements, consuming
    the iterable.

    Parameters:
        IterableType: The type of the iterable.

    Args:
        iterable: The iterable to consume and drop elements from.
        count: The number of elements to drop.

    Returns:
        An iterator that drops the first `count` elements.
    """
    assert count >= 0, "The `count` argument must be non-negative"
    return _DropIterator(
        iter(iterable^),
        count=count,
    )
