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

"""Implements the Span type.

You can import these APIs from the `utils.span` module. For example:

```mojo
from utils import Span
```
"""

from collections import InlineArray
from memory import Pointer, UnsafePointer
from builtin.builtin_list import _lit_mut_cast


@value
struct _SpanIter[
    is_mutable: Bool, //,
    T: CollectionElement,
    lifetime: Lifetime[is_mutable].type,
    forward: Bool = True,
]:
    """Iterator for Span.

    Parameters:
        is_mutable: Whether the reference to the span is mutable.
        T: The type of the elements in the span.
        lifetime: The lifetime of the Span.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var src: Span[T, lifetime]

    @always_inline
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(
        inout self,
    ) -> Pointer[T, lifetime]:
        @parameter
        if forward:
            self.index += 1
            return Pointer.address_of(self.src[self.index - 1])
        else:
            self.index -= 1
            return Pointer.address_of(self.src[self.index])

    @always_inline
    fn __hasmore__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        @parameter
        if forward:
            return len(self.src) - self.index
        else:
            return self.index


@value
struct Span[
    is_mutable: Bool, //,
    T: CollectionElement,
    lifetime: Lifetime[is_mutable].type,
](CollectionElementNew):
    """A non owning view of contiguous data.

    Parameters:
        is_mutable: Whether the span is mutable.
        T: The type of the elements in the span.
        lifetime: The lifetime of the Span.
    """

    # Field
    var _data: UnsafePointer[T]
    var _len: Int

    # ===------------------------------------------------------------------===#
    # Life cycle methods
    # ===------------------------------------------------------------------===#

    @always_inline
    fn __init__(inout self, *, unsafe_ptr: UnsafePointer[T], len: Int):
        """Unsafe construction from a pointer and length.

        Args:
            unsafe_ptr: The underlying pointer of the span.
            len: The length of the view.
        """
        self._data = unsafe_ptr
        self._len = len

    @always_inline
    fn __init__(inout self, *, other: Self):
        """Explicitly construct a deep copy of the provided Span.

        Args:
            other: The Span to copy.
        """
        self._data = other._data
        self._len = other._len

    @always_inline
    fn __init__(inout self, ref [lifetime]list: List[T, *_]):
        """Construct a Span from a List.

        Args:
            list: The list to which the span refers.
        """
        self._data = list.data
        self._len = len(list)

    @always_inline
    fn __init__[
        size: Int, //
    ](inout self, ref [lifetime]array: InlineArray[T, size]):
        """Construct a Span from an InlineArray.

        Parameters:
            size: The size of the InlineArray.

        Args:
            array: The array to which the span refers.
        """

        self._data = UnsafePointer.address_of(array).bitcast[T]()
        self._len = size

    # ===------------------------------------------------------------------===#
    # Operator dunders
    # ===------------------------------------------------------------------===#

    @always_inline
    fn __getitem__(self, idx: Int) -> ref [lifetime] T:
        """Get a reference to an element in the span.

        Args:
            idx: The index of the value to return.

        Returns:
            An element reference.
        """
        # TODO: Simplify this with a UInt type.
        debug_assert(
            -self._len <= int(idx) < self._len, "index must be within bounds"
        )

        var offset = idx
        if offset < 0:
            offset += len(self)
        return self._data[offset]

    @always_inline
    fn __getitem__(self, slc: Slice) -> Self:
        """Get a new span from a slice of the current span.

        Args:
            slc: The slice specifying the range of the new subslice.

        Returns:
            A new span that points to the same data as the current span.
        """
        var start: Int
        var end: Int
        var step: Int
        start, end, step = slc.indices(len(self))

        if step < 0:
            step = -step
            var new_len = (start - end + step - 1) // step
            var buff = UnsafePointer[T].alloc(new_len)
            i = 0
            while start > end:
                buff[i] = self._data[start]
                start -= step
                i += 1
            return Span[T, lifetime](unsafe_ptr=buff, len=new_len)

        var res = Self(
            unsafe_ptr=(self._data + start),
            len=len(range(start, end, step)),
        )

        return res

    @always_inline
    fn __iter__(self) -> _SpanIter[T, lifetime]:
        """Get an iterator over the elements of the span.

        Returns:
            An iterator over the elements of the span.
        """
        return _SpanIter(0, self)

    # ===------------------------------------------------------------------===#
    # Trait implementations
    # ===------------------------------------------------------------------===#

    @always_inline
    fn __len__(self) -> Int:
        """Returns the length of the span. This is a known constant value.

        Returns:
            The size of the span.
        """
        return self._len

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    fn unsafe_ptr(self) -> UnsafePointer[T]:
        """
        Gets a pointer to the first element of this slice.

        Returns:
            A pointer pointing at the first element of this slice.
        """

        return self._data

    fn as_ref(self) -> Pointer[T, lifetime]:
        """
        Gets a Pointer to the first element of this slice.

        Returns:
            A Pointer pointing at the first element of this slice.
        """

        return Pointer[T, lifetime].address_of(self._data[0])

    @always_inline
    fn copy_from[
        lifetime: MutableLifetime, //
    ](self: Span[T, lifetime], other: Span[T, _]):
        """
        Performs an element wise copy from all elements of `other` into all elements of `self`.

        Parameters:
            lifetime: The inferred mutable lifetime of the data within the Span.

        Args:
            other: The Span to copy all elements from.
        """
        debug_assert(len(self) == len(other), "Spans must be of equal length")
        for i in range(len(self)):
            self[i] = other[i]

    fn __bool__(self) -> Bool:
        """Check if a span is non-empty.

        Returns:
           True if a span is non-empty, False otherwise.
        """
        return len(self) > 0

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the lifetime.
    @__unsafe_disable_nested_lifetime_exclusivity
    fn __eq__[
        T: EqualityComparableCollectionElement, //
    ](self: Span[T, lifetime], rhs: Span[T]) -> Bool:
        """Verify if span is equal to another span.

        Parameters:
            T: The type of the elements in the span. Must implement the
              traits `EqualityComparable` and `CollectionElement`.

        Args:
            rhs: The span to compare against.

        Returns:
            True if the spans are equal in length and contain the same elements, False otherwise.
        """
        # both empty
        if not self and not rhs:
            return True
        if len(self) != len(rhs):
            return False
        # same pointer and length, so equal
        if self.unsafe_ptr() == rhs.unsafe_ptr():
            return True
        for i in range(len(self)):
            if self[i] != rhs[i]:
                return False
        return True

    @always_inline
    fn __ne__[
        T: EqualityComparableCollectionElement, //
    ](self: Span[T, lifetime], rhs: Span[T]) -> Bool:
        """Verify if span is not equal to another span.

        Parameters:
            T: The type of the elements in the span. Must implement the
              traits `EqualityComparable` and `CollectionElement`.

        Args:
            rhs: The span to compare against.

        Returns:
            True if the spans are not equal in length or contents, False otherwise.
        """
        return not self == rhs

    fn fill[lifetime: MutableLifetime, //](self: Span[T, lifetime], value: T):
        """
        Fill the memory that a span references with a given value.

        Parameters:
            lifetime: The inferred mutable lifetime of the data within the Span.

        Args:
            value: The value to assign to each element.
        """
        for element in self:
            element[] = value

    fn get_immutable(self) -> Span[T, _lit_mut_cast[lifetime, False].result]:
        """
        Return an immutable version of this span.

        Returns:
            A span covering the same elements, but without mutability.
        """
        return Span[T, _lit_mut_cast[lifetime, False].result](
            unsafe_ptr=self._data, len=self._len
        )
