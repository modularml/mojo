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
"""Implements a doubly-linked list data structure.

This module provides the `LinkedList` type, a doubly-linked list where each
element points to both the next and previous elements. This structure enables
efficient insertion and deletion at any position, though random access requires
traversal. The implementation includes iterator support for forward and reverse
traversal.
"""

from std.builtin.rebind import downcast, rebind_var
from std.collections import check_bounds
from std.reflection import call_location
import std.format._utils as fmt
from std.hashlib.hasher import Hasher
from std.memory.alloc import alloc, dealloc, ThinAllocation, Layout
from std.os import abort

from std.sys import align_of, size_of


@explicit_destroy(
    "A `Node` with a non-`Deinitable` element must be consumed with"
    " `_into_value()`; its owning `LinkedList` manages this."
)
struct Node[
    T: Movable,
](
    Deinitable where conforms_to(T, Deinitable),
    Movable,
):
    """A node in a linked list data structure.

    Parameters:
        T: The type of element stored in the node.
    """

    comptime _PointerType = Optional[Pointer[Self, MutUntrackedOrigin]]

    var value: Self.T
    """The value stored in this node."""
    var _prev: Self._PointerType
    """The previous node in the list."""
    var _next: Self._PointerType
    """The next node in the list."""

    @doc_hidden
    def prev(
        ref self,
    ) -> ref[self._prev] Optional[Pointer[Self, MutUntrackedOrigin]]:
        return self._prev

    @doc_hidden
    def next(
        ref self,
    ) -> ref[self._next] Optional[Pointer[Self, MutUntrackedOrigin]]:
        return self._next

    def __init__(
        out self,
        var value: Self.T,
        prev: Optional[Self._PointerType],
        next: Optional[Self._PointerType],
    ):
        """Initialize a new Node with the given value and optional prev/next
        pointers.

        Args:
            value: The value to store in this node.
            prev: Optional pointer to the previous node.
            next: Optional pointer to the next node.
        """
        self.value = value^
        self._prev = prev.value() if prev else Self._PointerType()
        self._next = next.value() if next else Self._PointerType()

    def _into_value(deinit self) -> Self.T:
        return self.value^

    @no_inline
    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Write this node's value to the given writer.

        Args:
            writer: The writer to write the value to.
        """
        writer.write(self.value)


def _make_node[
    T: Movable
](
    out node: Node[T],
    var value: T,
    prev: Optional[Pointer[Node[T], MutUntrackedOrigin]],
    next: Optional[Pointer[Node[T], MutUntrackedOrigin]],
):
    """Initialize a new Node with the given value and optional prev/next
    pointers.

    Args:
        value: The value to store in this node.
        prev: Optional pointer to the previous node.
        next: Optional pointer to the next node.
    """
    node = Node(value^, prev, next)


@fieldwise_init
struct _LinkedListIter[
    mut: Bool,
    //,
    ElementType: Copyable,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    var src: Pointer[LinkedList[Self.ElementType], Self.origin]
    var curr: Optional[Pointer[Node[Self.ElementType], MutUntrackedOrigin]]

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    comptime Element = Self.ElementType  # FIXME(MOCO-2068): shouldn't be needed.

    def __init__(out self, src: Pointer[LinkedList[Self.Element], Self.origin]):
        self.src = src

        comptime if Self.forward:
            self.curr = self.src[]._head
        else:
            self.curr = self.src[]._tail

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    def __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
        if not self.curr:
            raise StopIteration()
        var old = self.curr

        comptime if Self.forward:
            self.curr = self.curr.value()[].next()
        else:
            self.curr = self.curr.value()[].prev()

        return old.value()[].value


@fieldwise_init
struct _LinkedListIterOwned[T: Movable & Deinitable](
    IterableOwned, Iterator, Movable
):
    """An owning iterator for LinkedList.

    Parameters:
        T: The type of the elements in the linked list.
    """

    comptime Element = Self.T
    comptime IteratorOwnedType = Self

    var _list: LinkedList[Self.T]

    @always_inline
    def __deinit__(deinit self):
        # LinkedList.__deinit__ handles destroying remaining nodes.
        pass

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __next__(mut self) raises StopIteration -> Self.Element:
        if self._list._size <= 0:
            raise StopIteration()

        # Pop from front: take the head node, relink, and return the value.
        # (Mirrors LinkedList.pop() but operates on the head instead of tail.)
        var nn = self._list._head.value()
        var node = nn.unsafe_take_pointee()
        self._list._head = node.next()
        self._list._size -= 1
        if self._list._size == 0:
            self._list._tail = LinkedList[Self.T]._NodePointer()
        else:
            self._list._head.value()[].prev() = LinkedList[
                Self.T
            ]._NodePointer()
        dealloc(
            ThinAllocation(unsafe_owned_ptr=nn).unsafe_with_layout({count = 1})
        )
        return node^._into_value()

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var sz = self._list._size
        return (sz, {sz})


@explicit_destroy(
    "Use `deinit_with()` to explicitly destroy a `LinkedList` with"
    " non-`Deinitable` elements"
)
struct LinkedList[ElementType: Movable](
    Boolable,
    Copyable where conforms_to(ElementType, Copyable),
    Defaultable,
    Deinitable where conforms_to(ElementType, Deinitable),
    Equatable where conforms_to(ElementType, Equatable),
    Hashable where conforms_to(ElementType, Hashable),
    Iterable,
    IterableOwned where conforms_to(ElementType, Deinitable),
    Movable,
    Sized,
    Writable where conforms_to(ElementType, Writable),
):
    """A doubly-linked list implementation.

    Parameters:
        ElementType: The type of elements stored in the list. Must implement the
            `Movable` trait.

    A doubly-linked list is a data structure where each element points to both
    the next and previous elements, allowing for efficient insertion and deletion
    at any position.
    """

    comptime _NodePointer = Optional[
        Pointer[Node[Self.ElementType], MutUntrackedOrigin]
    ]

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _LinkedListIter[
        downcast[Self.ElementType, Copyable],
        iterable_origin,
    ]
    """The iterator type for this linked list.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    comptime IteratorOwnedType: Iterator where conforms_to(
        Self.ElementType, Deinitable
    ) = _LinkedListIterOwned[Self.ElementType]

    """The owned iterator type for this linked list."""

    var _head: Self._NodePointer
    """The first node in the list."""
    var _tail: Self._NodePointer
    """The last node in the list."""
    var _size: Int
    """The number of elements in the list."""

    def __init__(out self):
        """Initialize an empty linked list.

        Notes:
            Time Complexity: O(1).
        """
        self._head = Self._NodePointer()
        self._tail = Self._NodePointer()
        self._size = 0

    def __init__(
        out self,
        var *elements: Self.ElementType,
        __list_literal__: NoneType = None,
    ):
        """Initialize a linked list with the given elements.

        Args:
            elements: Variable number of elements to initialize the list with.
            __list_literal__: Tell Mojo to use this method for list literals.

        Notes:
            Time Complexity: O(n) in len(elements).
        """
        self = Self()

        # Transfer all of the elements into the list.
        def init_elt(idx: Int, var elt: Self.ElementType) {ref}:
            self.append(elt^)

        elements^.consume_elements(init_elt)

    def __init__(
        out self, *, copy: Self
    ) where conforms_to(Self.ElementType, Copyable):
        """Initialize this list as a copy of another list.

        Args:
            copy: The list to copy from.

        Notes:
            Time Complexity: O(n) in len(elements).
        """
        self = Self()
        var curr = copy._head
        while curr:
            self.append(curr.value()[].value.copy())
            curr = curr.value()[].next()

    def _delete_list_elements(
        mut self, destroy_func: Some[def(var Self.ElementType)]
    ):
        """Hand each element to `destroy_func`, then free every node.

        Shared teardown for `clear`, `__deinit__`, and `deinit_with`.

        Leaves the list "spent": `_head`/`_tail` dangle at freed memory, so the
        caller must either reset them (see `clear`) or be about to drop `self`
        (see `__deinit__`/`deinit_with`).

        Args:
            destroy_func: A closure called once per element to consume it.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        var curr = self._head
        while curr:
            var nn = curr.value()
            var next = nn[].next()
            destroy_func(nn.unsafe_take_pointee()._into_value())
            dealloc(
                ThinAllocation(unsafe_owned_ptr=nn).unsafe_with_layout(
                    {count = 1}
                )
            )
            curr = next

    def __deinit__(
        deinit self,
    ) where conforms_to(Self.ElementType, Deinitable):
        """Clean up the list by freeing all nodes.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        self.clear()

    def deinit_with(
        deinit self, deinit_func: Some[def(var Self.ElementType)], /
    ):
        """Consume the list, deinitializing each element with a closure.

        Use this to tear down a `LinkedList` whose elements are not
        `Deinitable`.

        Args:
            deinit_func: A closure called once per element to deinitialize it.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        self._delete_list_elements(deinit_func)

    def append(mut self, var value: Self.ElementType):
        """Add an element to the end of the list.

        Args:
            value: The value to append.

        Notes:
            Time Complexity: O(1).
        """
        var addr: Pointer[Node[Self.ElementType], MutUntrackedOrigin] = alloc(
            Layout[Node[Self.ElementType]].single()
        ).unsafe_leak()
        var value_ptr = Pointer(to=addr[].value)
        value_ptr.unsafe_write(value^)
        addr[].prev() = self._tail
        addr[].next() = Self._NodePointer()
        if self._tail:
            self._tail.value()[].next() = addr
        else:
            self._head = addr
        self._tail = addr
        self._size += 1

    def prepend(mut self, var value: Self.ElementType):
        """Add an element to the beginning of the list.

        Args:
            value: The value to prepend.

        Notes:
            Time Complexity: O(1).
        """
        var node = _make_node[Self.ElementType](value^, None, self._head)
        var addr: Pointer[Node[Self.ElementType], MutUntrackedOrigin] = alloc(
            Layout[Node[Self.ElementType]].single()
        ).unsafe_leak()
        addr.unsafe_write(node^)
        if self:
            self._head.value()[].prev() = addr
        else:
            self._tail = addr
        self._head = addr
        self._size += 1

    def reverse(mut self):
        """Reverse the order of elements in the list.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        var prev = Self._NodePointer()
        var curr = self._head
        while curr:
            var nn = curr.value()
            var next = nn[].next()
            nn[].next() = prev
            nn[].prev() = next
            prev = curr
            curr = next
        self._tail = self._head
        self._head = prev

    def pop(mut self) raises -> Self.ElementType:
        """Remove and return the last element of the list.

        Returns:
            The last element in the list.

        Notes:
            Time Complexity: O(1).

        Raises:
            If the operation fails.
        """
        if not self._tail:
            raise "Pop on empty list."

        var nn = self._tail.value()
        var node = nn.unsafe_take_pointee()
        self._tail = node.prev()
        self._size -= 1
        if self._size == 0:
            self._head = Self._NodePointer()
        else:
            self._tail.value()[].next() = Self._NodePointer()
        dealloc(
            ThinAllocation(unsafe_owned_ptr=nn).unsafe_with_layout({count = 1})
        )
        return node^._into_value()

    @always_inline
    def pop[
        I: Indexer & Deinitable, //
    ](mut self, var i: I) raises -> Self.ElementType:
        """Remove the ith element of the list, counting from the tail if
        given a negative index.

        Parameters:
            I: The type of index to use.

        Args:
            i: The index of the element to get.

        Returns:
            Ownership of the indicated element.

        Notes:
            Time Complexity: O(n) in len(self).

        Raises:
            If the operation fails.
        """
        var idx = index(i)
        var current = self._get_node_ptr(idx)

        if current:
            var nn = current.value()
            var node = nn.unsafe_take_pointee()
            if node.prev():
                node.prev().value()[].next() = node.next()
            else:
                self._head = node.next()
            if node.next():
                node.next().value()[].prev() = node.prev()
            else:
                self._tail = node.prev()

            dealloc(
                ThinAllocation(unsafe_owned_ptr=nn).unsafe_with_layout(
                    {count = 1}
                )
            )
            self._size -= 1
            return node^._into_value()

        raise Error("Invalid index for pop: ", idx)

    def maybe_pop(mut self) -> Optional[Self.ElementType]:
        """Removes the tail of the list and returns it, if it exists.

        Returns:
            The tail of the list, if it was present.

        Notes:
            Time Complexity: O(1).
        """
        if not self._tail:
            return Optional[Self.ElementType]()
        var nn = self._tail.value()
        var node = nn.unsafe_take_pointee()
        self._tail = node.prev()
        self._size -= 1
        if self._size == 0:
            self._head = Self._NodePointer()
        else:
            self._tail.value()[].next() = Self._NodePointer()
        dealloc(
            ThinAllocation(unsafe_owned_ptr=nn).unsafe_with_layout({count = 1})
        )
        return node^._into_value()

    @always_inline
    def maybe_pop[
        I: Indexer & Deinitable, //
    ](mut self, var i: I) -> Optional[Self.ElementType]:
        """Remove the ith element of the list, counting from the tail if
        given a negative index.

        Parameters:
            I: The type of index to use.

        Args:
            i: The index of the element to get.

        Returns:
            The element, if it was found.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        var current = self._get_node_ptr(index(i))

        if not current:
            return Optional[Self.ElementType]()
        else:
            var nn = current.value()
            var node = nn.unsafe_take_pointee()
            if node.prev():
                node.prev().value()[].next() = node.next()
            else:
                self._head = node.next()
            if node.next():
                node.next().value()[].prev() = node.prev()
            else:
                self._tail = node.prev()

            dealloc(
                ThinAllocation(unsafe_owned_ptr=nn).unsafe_with_layout(
                    {count = 1}
                )
            )
            self._size -= 1
            return Optional[Self.ElementType](node^._into_value())

    def clear(
        mut self,
    ) where conforms_to(Self.ElementType, Deinitable):
        """Removes all elements from the list.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        var current = self._head
        while current:
            var nn = current.value()
            current = nn[].next()
            nn.unsafe_deinit_pointee()
            dealloc(
                ThinAllocation(unsafe_owned_ptr=nn).unsafe_with_layout(
                    {count = 1}
                )
            )

        self._head = Self._NodePointer()
        self._tail = Self._NodePointer()
        self._size = 0

    @always_inline
    def insert[I: Indexer](mut self, idx: I, var elem: Self.ElementType):
        """Insert an element `elem` into the list at index `idx`.

        Parameters:
            I: The type of index to use.

        Args:
            idx: The index to insert `elem` at. Must be in the range
                `[0, len(self)]`.
            elem: The item to insert into the list.

        Notes:
            Time Complexity: O(n) in len(self).
        """

        # Valid range is `[0, len(self)]` (an index of `len(self)` appends).
        var i = index(idx)
        check_bounds(i, len(self) + 1)

        if i == 0:
            var node: Pointer[
                Node[Self.ElementType], MutUntrackedOrigin
            ] = alloc(Layout[Node[Self.ElementType]].single()).unsafe_leak()
            node.unsafe_write(
                _make_node[Self.ElementType](
                    elem^, Self._NodePointer(), Self._NodePointer()
                )
            )

            if self._head:
                node[].next() = self._head
                self._head.value()[].prev() = node

            self._head = node

            if not self._tail:
                self._tail = node

            self._size += 1
            return

        i -= 1

        # `i <= len(self)` (asserted above) guarantees a valid predecessor node.
        var current = self._get_node_ptr(i)
        var curr_nn = current.value()
        var next = curr_nn[].next()
        var node: Pointer[Node[Self.ElementType], MutUntrackedOrigin] = alloc(
            Layout[Node[Self.ElementType]].single()
        ).unsafe_leak()
        var data = Pointer(to=node[].value)
        data.unsafe_write(elem^)
        node[].next() = next
        node[].prev() = current
        if next:
            next.value()[].prev() = node
        curr_nn[].next() = node
        if node[].next() == Self._NodePointer():
            self._tail = node
        if node[].prev() == Self._NodePointer():
            self._head = node
        self._size += 1

    def extend(mut self, deinit other: Self):
        """Extends the list with another.

        Args:
            other: The list to append to this one.

        Notes:
            Time Complexity: O(1).
        """
        if self._tail:
            self._tail.value()[].next() = other._head
            if other._head:
                other._head.value()[].prev() = self._tail
            if other._tail:
                self._tail = other._tail

            self._size += other._size
        else:
            self._head = other._head
            self._tail = other._tail
            self._size = other._size

    def count(
        self, imm elem: Self.ElementType
    ) -> Int where conforms_to(Self.ElementType, Equatable):
        """Count the occurrences of `elem` in the list.

        Args:
            elem: The element to search for.

        Returns:
            The number of occurrences of `elem` in the list.

        Notes:
            Time Complexity: O(n) in len(self) compares.
        """
        var current = self._head
        var count = 0
        while current:
            if current.value()[].value == elem:
                count += 1

            current = current.value()[].next()

        return count

    def index(
        self,
        value: Self.ElementType,
    ) raises -> Int where conforms_to(Self.ElementType, Equatable):
        """Returns the index of the first occurrence of a value in the list.

        Args:
            value: The value to search for.

        Returns:
            The index of the first occurrence of the value in the list.

        Raises:
            ValueError: If the value is not found in the list.

        Notes:
            Unlike Python's `list.index()`, this method does not accept
            `start`/`stop` parameters.

            Time Complexity: O(n) in len(self).
        """
        var current = self._head
        var i = 0
        while current:
            if current.unsafe_value()[].value == value:
                return i
            current = current.unsafe_value()[].next()
            i += 1

        raise "ValueError: Given element is not in linked list"

    def __contains__(
        self, value: Self.ElementType
    ) -> Bool where conforms_to(Self.ElementType, Equatable):
        """Checks if the list contains `value`.

        Args:
            value: The value to search for in the list.

        Returns:
            Whether the list contains `value`.

        Notes:
            Time Complexity: O(n) in len(self) compares.
        """
        var current = self._head
        while current:
            if current.value()[].value == value:
                return True
            current = current.value()[].next()

        return False

    def __eq__(
        imm self,
        imm other: Self,
    ) -> Bool where conforms_to(Self.ElementType, Equatable):
        """Checks if the two lists are equal.

        Args:
            other: The list to compare to.

        Returns:
            Whether the lists are equal.

        Notes:
            Time Complexity: O(n) in min(len(self), len(other)) compares.
        """
        if self._size != other._size:
            return False

        var self_curr = self._head
        var other_curr = other._head

        while self_curr:
            if self_curr.value()[].value != other_curr.value()[].value:
                return False

            self_curr = self_curr.value()[].next()
            other_curr = other_curr.value()[].next()

        return True

    def __hash__(
        self, mut hasher: Some[Hasher]
    ) where conforms_to(Self.ElementType, Hashable):
        """Hash the elements of this list.

        Args:
            hasher: The hasher instance.
        """
        var curr = self._head
        while curr:
            curr.value()[].value.__hash__(hasher)
            curr = curr.value()[].next()

    @__unsafe_nested_origins_read_only
    @always_inline
    def get_nth[
        I: Indexer
    ](ref self, idx: I) -> ref[
        origin_of(self)._get_owned_interior["element"]
    ] Self.ElementType:
        """Get the element at the specified index.

        Parameters:
            I: The type of index to use.

        Args:
            idx: The index of the element to get.

        Returns:
            A reference to the element at the specified index.

        Notes:
            Time Complexity: O(n/2) in len(self).
        """
        return Pointer(
            to=self._get_node_ptr(idx).value()[].value
        )._get_ref_with_unsafe_interior_origin["element", origin_of(self)]()

    @always_inline
    def _get_node_ptr[I: Indexer, //](ref self, idx: I) -> Self._NodePointer:
        """Get an optional pointer to the node at the specified index.

        Parameters:
            I: The type of index to use.

        Args:
            idx: The index of the node to get.

        Returns:
            An optional pointer to the node at the specified index.

        Notes:
            This method optimizes traversal by starting from either the head or
            tail depending on which is closer to the target index.

            Time Complexity: O(n/2) in len(self).
        """
        var i = index(idx)
        var l = len(self)
        check_bounds(i, l, call_location[inline_count=2]())
        var mid = l // 2
        if i <= mid:
            var curr = self._head
            for _ in range(i):
                curr = curr.value()[].next()
            return curr
        else:
            var curr = self._tail
            for _ in range(l - i - 1):
                curr = curr.value()[].prev()
            return curr

    def __len__(self) -> Int:
        """Get the number of elements in the list.

        Returns:
            The number of elements in the list.

        Notes:
            Time Complexity: O(1).
        """
        return self._size

    def __iter__(
        var self,
    ) -> Self.IteratorOwnedType where conforms_to(Self.ElementType, Deinitable):
        """Consume the linked list and return an iterator over its elements.

        Returns:
            An iterator that owns the linked list's elements.

        Notes:
            Time Complexity:
            - O(1) for iterator construction.
            - O(n) in len(self) for a complete iteration of the list.
        """
        return Self.IteratorOwnedType(self^)

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over elements of the list, returning immutable references.

        Returns:
            An iterator of immutable references to the list elements.

        Notes:
            Time Complexity:
            - O(1) for iterator construction.
            - O(n) in len(self) for a complete iteration of the list.
        """
        # TODO(MSTDL-2390): Remove `Copyable` constraint once we have better iter traits.
        comptime assert conforms_to(
            Self.ElementType, Copyable
        ), "LinkedList iteration requires the element to be `Copyable`."
        return _LinkedListIter(Pointer(to=self))

    # TODO(MSTDL-2390): Remove `Copyable` constraint once we have better iter traits.
    def __reversed__(
        ref self,
    ) -> _LinkedListIter[
        Self.ElementType,
        origin_of(self),
        forward=False,
    ] where conforms_to(Self.ElementType, Copyable):
        """Iterate backwards over the list, returning immutable references.

        Returns:
            A reversed iterator of immutable references to the list elements.

        Notes:
            Time Complexity:
            - O(1) for iterator construction.
            - O(n) in len(self) for a complete iteration of the list.
        """
        return {Pointer(to=self)}

    def __bool__(self) -> Bool:
        """Check if the list is non-empty.

        Returns:
            True if the list has elements, False otherwise.

        Notes:
            Time Complexity: O(1).
        """
        return len(self) != 0

    def _write_self_to[
        f: def(Self.ElementType, mut Some[Writer]) thin
    ](self, mut writer: Some[Writer]) where conforms_to(
        Self.ElementType, Writable
    ):
        var iterator = self.__iter__()

        def iterate(mut w: Some[Writer]) raises StopIteration {mut iterator}:
            f(iterator.__next__(), w)

        fmt.write_sequence_to(writer, iterate)
        _ = iterator^

    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.ElementType, Writable):
        """Write the list to the given writer.

        Args:
            writer: The writer to write the list to.
        """
        self._write_self_to[f=fmt.write_to[Self.ElementType]](writer)

    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.ElementType, Writable):
        """Write the repr representation of this LinkedList to a Writer.

        Args:
            writer: The writer to write to.
        """

        var self_ptr = Pointer(to=self)

        def write_fields(mut w: Some[Writer]) {self_ptr}:
            self_ptr[]._write_self_to[f=fmt.write_repr_to[Self.ElementType]](w)

        fmt.FormatStruct(writer, "LinkedList").params(
            fmt.TypeNames[Self.ElementType](),
        ).fields(write_fields)
