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
"""Iteration traits and utilities: Iterable, IterableOwned, Iterator,
enumerate, zip, map.

This package defines the core iteration protocol for Mojo through the
`Iterable`, `IterableOwned`, and `Iterator` traits. Types that conform to
these traits can be used with `for` loops and iteration utilities like
`enumerate()`, `zip()`, and `map()`.

The iteration protocol consists of three key traits:

- `Iterable`: Types that can produce an iterator by borrowing (`ref self`).
  The iterator borrows the collection and yields references or copies of
  elements without consuming the source.
- `IterableOwned`: Types that can produce an iterator by taking ownership
  (`var self`). The iterator consumes the collection, taking ownership of
  its elements.
- `Iterator`: Types that can produce a sequence of values one at a time.

Examples:

```mojo
from std.iter import enumerate, zip, map

# Enumerate with index
var items = ["a", "b", "c"]
for index, value in enumerate(items):
    print(index, value)

# Zip multiple iterables
var numbers = [1, 2, 3]
var letters = ["x", "y", "z"]
for num, letter in zip(numbers, letters):
    print(num, letter)

# Map a function over an iterable
var values = [1, 2, 3, 4]
for squared in map[lambda (x: Int) -> Int: x * x](values):
    print(squared)
```
"""

import std.memory
from std.builtin.rebind import downcast


from std.builtin.variadics import TypeList


# ===-----------------------------------------------------------------------===#
# Iterable
# ===-----------------------------------------------------------------------===#


trait Iterable:
    """Describes a type that can produce an iterator by borrowing.

    Conforming types implement `__iter__(ref self)`, which borrows the
    collection (immutably or mutably, depending on the call-site origin) and
    returns an iterator whose elements may reference the source data. The
    collection remains usable after iteration.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator
    """The iterator type returned when borrowing this collection.

    Parameterized on the mutability and origin so the iterator
    can yield references tied to the source collection's origin.
    """

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Borrows the collection and returns an iterator over its elements.

        Returns:
            An iterator over the elements.
        """
        ...


trait IterableOwned:
    """Describes a type that can produce an iterator by giving up ownership.

    Conforming types implement `__iter__(var self)`, which takes the collection
    by value and returns an iterator that owns the underlying data.

    Use `IterableOwned` when the caller no longer needs the collection after
    iteration, or when yielding owned elements is more natural (e.g. draining
    a temporary result).
    """

    comptime IteratorOwnedType: Iterator
    """The iterator type returned when the collection is consumed."""

    def __iter__(var self) -> Self.IteratorOwnedType:
        """Consumes the collection and returns an iterator over its elements.

        Returns:
            An iterator that owns the collection's elements.
        """
        ...


# ===-----------------------------------------------------------------------===#
# Iterator
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct StopIteration(TrivialRegisterPassable, Writable):
    """A custom error type for Iterator's that run out of elements."""

    def write_to(self, mut writer: Some[Writer]):
        """This always writes "StopIteration".

        Args:
            writer: The writer to write to.
        """
        writer.write("StopIteration")


trait Iterator(Deinitable, Movable):
    """The `Iterator` trait describes a type that can be used as an
    iterator, e.g. in a `for` loop.
    """

    comptime Element: Movable

    def __next__(mut self) raises StopIteration -> Self.Element:
        """Returns the next element from the iterator.

        Raises:
            StopIteration if there are no more elements.

        Returns:
            The next element.
        """
        ...

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        """Returns bounds `[lower, upper]` for the remaining iterator length.

        This helps collections pre-allocate memory when constructed from iterators.
        The default implementation returns `(0, None)`.

        Returns:
            A tuple where the first element is the lower bound and the second
            is an optional upper bound (`None` means unknown or `upper > Int.MAX`).

        Safety:

        If the upper bound is not None, implementations must ensure that `lower <= upper`.
        The bounds are hints only - iterators may not comply with them. Never omit safety
        checks when using `bounds` to build collections.

        Examples:

        ```mojo
        from std.iter import Iterator

        def preallocate[I: Iterator](mut iter: I) -> List[Int]:
            var lower, _upper = iter.bounds()
            # Pre-allocate based on estimated iterator length
            return List[Int](capacity=lower)
        ```
        """
        return (0, None)

    def nth(var self, n: Int) -> Optional[Self.Element]:
        """Advances the iterator by `n` elements (destroying them) and returns
        the next element, or `None` if the iterator is exhausted first.

        Args:
            n: The 0-indexed position of the element to return. Must be
                non-negative.

        Returns:
            The element at index `n`, or `None` if the iterator has fewer than
            `n + 1` remaining elements.

        Constraints:
            `Self.Element` must conform to `Deinitable` so the
            intermediate elements can be discarded.

        Examples:

        ```mojo
        var l = [10, 20, 30, 40]
        print(iter(l).nth(0).value())   # 10
        print(iter(l).nth(3).value())   # 40
        var missing = iter(l).nth(10)   # None
        ```
        """
        # `Self.Element` is only declared `Movable` on the trait, so a
        # bare `_ = self.__next__()` won't type-check without this assertion.
        # Drop this workaround once MOCO-3947 lets us put the bound in a
        # `where` clause on the method.
        comptime assert conforms_to(Self.Element, Deinitable)

        debug_assert[assert_mode="safe"](n.ge(0), "nth: n must be non-negative")
        try:
            for _ in range(n):
                _ = self.__next__()
            return self.__next__()
        except StopIteration:
            return None


@always_inline
def iter(
    var iterable: Some[IterableOwned],
) -> type_of(iterable).IteratorOwnedType:
    """Constructs an owned iterator from an iterable.

    Args:
        iterable: The iterable to construct the iterator from.

    Returns:
        An owned iterator for the given iterable.
    """
    return iterable^.__iter__()


@always_inline
def iter[
    IterableType: Iterable
](ref iterable: IterableType) -> IterableType.IteratorType[origin_of(iterable)]:
    """Constructs a borrowed iterator from an iterable.

    Parameters:
        IterableType: The type of the iterable.

    Args:
        iterable: The iterable to construct the iterator from.

    Returns:
        A borrowed iterator for the given iterable.
    """
    return iterable.__iter__()


@always_inline
def next[
    IteratorType: Iterator
](mut iterator: IteratorType) raises StopIteration -> IteratorType.Element:
    """Advances the iterator and returns the next element.

    Parameters:
        IteratorType: The type of the iterator.

    Args:
        iterator: The iterator to advance.

    Returns:
        The next element from the iterator.

    Raises:
        StopIteration: If the iterator is exhausted.
    """
    return iterator.__next__()


# ===-----------------------------------------------------------------------===#
# empty
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _Empty[T: Movable](
    ImplicitlyCopyable,
    Iterable,
    IterableOwned,
    Iterator,
):
    """Iterator that yields nothing."""

    comptime Element = Self.T

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    comptime IteratorOwnedType: Iterator = Self

    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    def __next__(mut self) raises StopIteration -> Self.Element:
        raise StopIteration()

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return Tuple(0, Optional(0))


@always_inline
def empty[T: Movable]() -> _Empty[T]:
    """Creates an iterator that yields nothing.

    Parameters:
        T: Type of the iterator's notional elements.

    Returns:
        An iterator that yields nothing.
    """
    return _Empty[T]()


# ===-----------------------------------------------------------------------===#
# once
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _Once[T: Movable & Deinitable](
    Copyable where conforms_to(T, Copyable),
    Iterable where conforms_to(T, Copyable),
    IterableOwned,
    Iterator,
    Movable,
):
    """An iterator that yields an element exactly once."""

    comptime Element = Self.T

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    comptime IteratorOwnedType: Iterator = Self

    var _inner: Optional[Self.T]

    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)] where conforms_to(
        Self.Element, Copyable
    ):
        return self.copy()

    def __next__(mut self) raises StopIteration -> Self.Element:
        if not self._inner:
            raise StopIteration()
        return self._inner.unsafe_take()

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return self._inner.bounds()


@always_inline
def once[T: Movable & Deinitable, //](var element: T, /) -> _Once[T]:
    """Creates an iterator that yields an element exactly once.

    Parameters:
        T: The type of the element to be yielded exactly once.

    Args:
        element: The element to be yielded exactly once.

    Returns:
        An iterator that yields the specified element exactly once.
    """
    return _Once(Optional(element^))


# ===-----------------------------------------------------------------------===#
# enumerate
# ===-----------------------------------------------------------------------===#


struct _Enumerate[InnerIteratorType: Iterator](
    Copyable where conforms_to(InnerIteratorType, Copyable),
    Iterable where conforms_to(InnerIteratorType, Copyable),
    IterableOwned,
    Iterator,
):
    """An iterator that yields tuples of the index and the element of the
    original iterator.
    """

    comptime Element = Tuple[Int, Self.InnerIteratorType.Element]
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self
    var _inner: Self.InnerIteratorType
    var _count: Int

    def __init__(
        out self, var iterator: Self.InnerIteratorType, *, start: Int = 0
    ):
        self._inner = iterator^
        self._count = start

    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)] where conforms_to(
        Self.InnerIteratorType, Copyable
    ):
        return self.copy()

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __next__(mut self) raises StopIteration -> Self.Element:
        # This raises on error.
        var elt = next(self._inner)
        var count = self._count
        self._count += 1
        return count, elt^

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return self._inner.bounds()


@always_inline
def enumerate[
    IterableType: Iterable
](ref iterable: IterableType, *, start: Int = 0) -> _Enumerate[
    IterableType.IteratorType[origin_of(iterable)]
]:
    """Returns an iterator that yields tuples of the index and the element of
    the original iterator.

    Parameters:
        IterableType: The type of the iterable.

    Args:
        iterable: An iterable object (e.g., list, string, etc.).
        start: The starting index for enumeration (default is 0).

    Returns:
        An enumerate iterator that yields tuples of `(index, element)`.

    Examples:

    ```mojo
    var l = ["hey", "hi", "hello"]
    for i, elem in enumerate(l):
        print(i, elem)
    ```
    """
    return _Enumerate(iter(iterable), start=start)


@always_inline
def enumerate(
    var iterable: Some[IterableOwned], *, start: Int = 0
) -> _Enumerate[type_of(iterable).IteratorOwnedType]:
    """Returns an iterator that yields tuples of the index and the element of
    the original iterator, consuming the iterable.

    Args:
        iterable: An iterable object to consume and enumerate.
        start: The starting index for enumeration (default is 0).

    Returns:
        An enumerate iterator that yields tuples of `(index, element)`.
    """
    return _Enumerate(iter(iterable^), start=start)


# ===-----------------------------------------------------------------------===#
# zip
# ===-----------------------------------------------------------------------===#


struct _ZipIterator[origin: Origin, *Ts: Iterator](
    Copyable where conforms_to(Tuple[*Ts], Copyable),
    Iterable where conforms_to(Tuple[*Ts], Copyable),
    IterableOwned,
    Iterator,
):
    """Yields tuples of elements drawn in lockstep from its inner iterators.

    Iteration stops as soon as any inner iterator raises `StopIteration`.
    When that happens mid-tuple, any elements already produced for the
    current tuple are destroyed before propagating the exception, which is
    why each element type in `Ts` must be `Deinitable`.

    Parameters:
        origin: The origin from which the inner iterators were produced.
            Used by the `zip()` factory overloads to thread lifetime info
            into the borrowed iterator types in `Ts`, and set to
            `MutUntrackedOrigin` by the owning overload since its iterators
            own their data.
        Ts: The inner iterator types being zipped. Each must conform to
            `Iterator` and its `Element` must be `Deinitable`.
    """

    comptime _InjectedValues = Tuple[*Self.Ts]
    var _values: Self._InjectedValues

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self
    comptime _mapper[T: Iterator] = T.Element
    comptime Element = Tuple[
        *TypeList[Trait=Iterator, Self.Ts.values]().map[Self._mapper]()
    ]

    @always_inline
    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)] where conforms_to(
        Tuple[*Self.Ts], Copyable
    ):
        return self.copy()

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __next__(mut self) raises StopIteration -> Self.Element:
        var initialized = 0
        var res: Self.Element
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(res))
        try:
            comptime for i in range(Self._InjectedValues.__len__()):
                Pointer(to=res[i]).unsafe_write(
                    rebind_var[type_of(res[i])](next(self._values[i]))
                )
                initialized += 1
            return res^
        except StopIteration:
            comptime for i in range(Self._InjectedValues.__len__()):
                comptime assert conforms_to(type_of(res[i]), Deinitable)
                if i < initialized:
                    Pointer(to=res[i]).unsafe_deinit_pointee()

            std.memory.forget_deinit(res^)
            raise StopIteration

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var res_lower = Int.MAX
        var res_upper = Optional[Int](None)
        comptime for i in range(Self._InjectedValues.__len__()):
            var lower, upper = self._values[i].bounds()
            res_lower = min(res_lower, lower)
            if upper:
                res_upper = min(res_upper.or_else(Int.MAX), upper.value())

        return (res_lower, res_upper)


def zip[
    *Ts: Iterable
](
    *iterables: *Ts,
    out res: _ZipIterator[
        iterables.origin, *_iterable_to_iterator[iterables.origin, *Ts]
    ],
) where res.Ts.all_conforms_to[Deinitable]():
    """Returns an iterator that yields tuples of the elements of the original
    iterables.

    Parameters:
        Ts: The type of the iterables.

    Args:
        iterables: The iterables.

    Returns:
        A zip iterator that yields tuples of elements from all iterables.

    Examples:

    ```mojo
    var l = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]
    for a, b in zip(l, l2):
        print(a, b)
    ```
    """
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(res))

    comptime for i in range(res._InjectedValues.__len__()):
        Pointer(to=res._values[i]).unsafe_write(
            rebind_var[type_of(res._values[i])](iter(iterables[i]))
        )


def zip[
    *Ts: IterableOwned
](
    var *iterables: *Ts,
    out res: _ZipIterator[
        MutUntrackedOrigin, *_iterable_owned_to_iterator[*Ts]
    ],
) where res.Ts.all_conforms_to[Deinitable]():
    """Returns an iterator that yields tuples of the elements of the original
    iterables.

    Parameters:
        Ts: The type of the iterables.

    Args:
        iterables: The iterables.

    Returns:
        A zip iterator that yields tuples of elements from all iterables.

    Examples:

    ```mojo
    for a, b in zip(["hey", "hi", "hello"], [10, 20, 30]):
        print(a, b)
    ```
    """
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(res))

    @__parameter
    def init_elt[idx: Int](var elt: iterables.Ts[idx]):
        Pointer(to=res._values[idx]).unsafe_write(
            rebind_var[type_of(res._values[idx])](iter(elt^))
        )

    iterables^.consume_elements[init_elt]()


# ===-----------------------------------------------------------------------===#
# map
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _MapIterator[
    OutputType: Copyable,
    InnerIteratorType: Iterator,
    //,
    function: def(var InnerIteratorType.Element) thin -> OutputType,
](
    Copyable where conforms_to(InnerIteratorType, Copyable),
    Iterable where conforms_to(InnerIteratorType, Copyable),
    IterableOwned,
    Iterator,
):
    comptime Element = Self.OutputType
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self

    var _inner: Self.InnerIteratorType

    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)] where conforms_to(
        Self.InnerIteratorType, Copyable
    ):
        return self.copy()

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __next__(mut self) raises StopIteration -> Self.Element:
        return Self.function(next(self._inner))

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return self._inner.bounds()


@always_inline
def map[
    origin: ImmOrigin,
    IterableType: Iterable,
    ResultType: Copyable,
    //,
    function: def(
        var IterableType.IteratorType[origin].Element
    ) thin -> ResultType,
](ref[origin] iterable: IterableType) -> _MapIterator[function]:
    """Returns an iterator that applies `function` to each element of the input
    iterable.

    Parameters:
        origin: The origin of the iterable.
        IterableType: The type of the iterable.
        ResultType: The return type of the function.
        function: The function to apply to each element.

    Args:
        iterable: The iterable to map over.

    Returns:
        A map iterator that yields the results of applying `function` to each
        element.

    Examples:

    ```mojo
    var l = [1, 2, 3]
    var m = map[lambda (x: Int) -> Int: x + 1](l)

    # outputs:
    # 2
    # 3
    # 4
    for elem in m:
        print(elem)
    ```
    """
    # FIXME(MOCO-3238): This rebind shouldn't ve needed, something isn't getting
    # substituted through associated types right.
    return {
        rebind_var[_MapIterator[function].InnerIteratorType](iter(iterable))
    }


@always_inline
def map[
    IterableType: IterableOwned,
    ResultType: Copyable,
    //,
    function: def(
        var IterableType.IteratorOwnedType.Element
    ) thin -> ResultType,
](var iterable: IterableType) -> _MapIterator[function]:
    """Returns an iterator that applies `function` to each element of the input
    iterable, consuming the iterable.

    Parameters:
        IterableType: The type of the iterable.
        ResultType: The return type of the function.
        function: The function to apply to each element.

    Args:
        iterable: The iterable to consume and map over.

    Returns:
        A map iterator that yields the results of applying `function` to each
        element.
    """
    # FIXME(MOCO-3238): This rebind shouldn't be needed, something isn't getting
    # substituted through associated types right.
    return {
        rebind_var[_MapIterator[function].InnerIteratorType](iter(iterable^))
    }


# ===-----------------------------------------------------------------------===#
# peekable
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _PeekableIterator[InnerIterator: Iterator](
    Copyable where conforms_to(InnerIterator, Copyable) and conforms_to(
        InnerIterator.Element, Copyable
    ),
    Iterable where conforms_to(InnerIterator, Copyable) and conforms_to(
        InnerIterator.Element, Copyable
    ),
    IterableOwned,
    Iterator,
) where conforms_to(InnerIterator.Element, Deinitable):
    comptime Element = Self.InnerIterator.Element
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self

    var _inner: Self.InnerIterator
    var _next: Optional[Self.Element]

    def __init__(out self, var inner: Self.InnerIterator):
        self._inner = inner^
        self._next = None

    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)] where conforms_to(
        Self.InnerIterator, Copyable
    ) and conforms_to(Self.InnerIterator.Element, Copyable):
        return self.copy()

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __next__(mut self) raises StopIteration -> Self.Element:
        if self._next:
            return self._next.unsafe_take()
        return next(self._inner)

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var peek_len = 1 if self._next else 0
        var lower, upper = self._inner.bounds()
        if upper:
            return (lower + peek_len, upper.value() + peek_len)
        else:
            return (lower + peek_len, None)

    def peek(
        mut self,
    ) -> Optional[Pointer[Self.Element, ImmOrigin(origin_of(self._next[]))]]:
        if not self._next:
            try:
                self._next = next(self._inner)
            except:
                return None
        return ImmPointer[Self.Element](to=self._next.unsafe_value())


def peekable(
    ref iterable: Some[Iterable],
) -> _PeekableIterator[
    type_of(iterable).IteratorType[origin_of(iterable)]
] where conforms_to(
    type_of(iterable).IteratorType[origin_of(iterable)].Element,
    Deinitable,
):
    """Returns a peekable iterator that can use the `peek` method to look ahead
    at the next element without advancing the iterator.

    Args:
        iterable: The iterable to create a peekable iterator from.

    Returns:
        A peekable iterator.
    """
    return {iter(iterable)}


def peekable(
    var iterable: Some[IterableOwned],
) -> _PeekableIterator[type_of(iterable).IteratorOwnedType] where conforms_to(
    type_of(iterable).IteratorOwnedType.Element,
    Deinitable,
):
    """Returns a peekable iterator that can use the `peek` method to look ahead
    at the next element without advancing the iterator, consuming the iterable.

    Args:
        iterable: The iterable to consume and create a peekable iterator from.

    Returns:
        A peekable iterator.
    """
    return {iter(iterable^)}


# ===-----------------------------------------------------------------------===#
# chain
# ===-----------------------------------------------------------------------===#

comptime _all_yield_same_ref_condition[
    T0: Movable, origin: Origin, T: Iterable
] = T.IteratorType[origin].Element == T0

comptime _all_yield_same_ref[origin: Origin, *Ts: Iterable]: Bool = Ts.all[
    _all_yield_same_ref_condition[Ts[0].IteratorType[origin].Element, origin, _]
]()

comptime _all_yield_same_owned_condition[
    T0: Movable, T: IterableOwned
] = T.IteratorOwnedType.Element == T0

comptime _all_yield_same_owned[*Ts: IterableOwned]: Bool = Ts.all[
    _all_yield_same_owned_condition[Ts[0].IteratorOwnedType.Element, _]
]()


struct _ChainedIterator[*Ts: Iterator](
    Copyable where conforms_to(Tuple[*Ts], Copyable),
    Iterable where conforms_to(Tuple[*Ts], Copyable),
    IterableOwned,
    Iterator,
):
    comptime _Iterators = Tuple[*Self.Ts]
    var _idx: Int
    var _iterators: Self._Iterators

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self
    comptime Element = Self.Ts[0].Element

    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)] where conforms_to(
        Tuple[*Self.Ts], Copyable
    ):
        return self.copy()

    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    @always_inline
    def __next__(mut self) raises StopIteration -> Self.Element:
        comptime for i in range(Self._Iterators.__len__()):
            if self._idx <= i:
                try:
                    return rebind_var[Self.Element](next(self._iterators[i]))
                except:
                    self._idx += 1
        raise StopIteration()

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var final_lb = 0
        var final_ub = Optional(0)
        comptime for i in range(Self._Iterators.__len__()):
            if self._idx > i:
                continue

            var lb, ub = self._iterators[i].bounds()
            final_lb = final_lb + min(lb, Int.MAX - final_lb)
            if final_ub and ub:
                final_ub = final_ub.unsafe_value() + min(
                    ub.unsafe_value(), Int.MAX - final_ub.unsafe_value()
                )
            else:
                final_ub = None
        return final_lb, final_ub


def chain[
    *Ts: Iterable
](
    *iterables: *Ts,
    out res: _ChainedIterator[*_iterable_to_iterator[iterables.origin, *Ts]],
) where _all_yield_same_ref[iterables.origin, *Ts]:
    """Chain multiple iterables that return the same type.

    Parameters:
        Ts: The iterator types.

    Args:
        iterables: The iterables.

    Returns:
        The chained iterator.
    """
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(res))
    res._idx = 0

    comptime for i in range(res._Iterators.__len__()):
        Pointer(to=res._iterators[i]).unsafe_write(
            rebind_var[type_of(res._iterators[i])](iter(iterables[i]))
        )


def chain[
    *Ts: IterableOwned
](
    var *iterables: *Ts,
    out res: _ChainedIterator[*_iterable_owned_to_iterator[*Ts]],
) where _all_yield_same_owned[*Ts]:
    """Chain multiple iterables that return the same type.

    Parameters:
        Ts: The iterator types.

    Args:
        iterables: The iterables.

    Returns:
        The chained iterator.
    """
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(res))
    res._idx = 0

    @__parameter
    def init_elt[idx: Int](var elt: iterables.Ts[idx]):
        Pointer(to=res._iterators[idx]).unsafe_write(
            rebind_var[type_of(res._iterators[idx])](iter(elt^))
        )

    iterables^.consume_elements[init_elt]()


# ===-----------------------------------------------------------------------===#
# utilities
# ===-----------------------------------------------------------------------===#


comptime _map_iterable_iterator[origin: Origin, T: Iterable] = T.IteratorType[
    origin
]
comptime _iterable_to_iterator[origin: Origin, *Ts: Iterable] = TypeList[
    Trait=Iterable, Ts.values
]().map[_map_iterable_iterator[origin, ...]]()

comptime _map_iterable_owned_iterator[T: IterableOwned] = T.IteratorOwnedType
comptime _iterable_owned_to_iterator[*Ts: IterableOwned] = TypeList[
    Trait=IterableOwned, Ts.values
]().map[_map_iterable_owned_iterator]()
