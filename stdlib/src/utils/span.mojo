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
from memory import Reference, UnsafePointer, bitcast, memcmp
from sys.intrinsics import _type_is_eq
from builtin.builtin_list import _lit_mut_cast
from sys import simdwidthof
from bit import count_trailing_zeros
from builtin.dtype import _uint_type_of_width


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
    ) -> Reference[T, lifetime]:
        @parameter
        if forward:
            self.index += 1
            return self.src[self.index - 1]
        else:
            self.index -= 1
            return self.src[self.index]

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
        T2: CollectionElement, size: Int, //
    ](inout self, ref [lifetime]array: InlineArray[T2, size]):
        """Construct a Span from an InlineArray.

        Parameters:
            T2: The type of the elements in the span.
            size: The size of the InlineArray.

        Args:
            array: The array to which the span refers.
        """

        constrained[_type_is_eq[T, T2](), "array element is not Span.T"]()

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
        debug_assert(
            step == 1, "Slice must be within bounds and step must be 1"
        )
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

    fn as_ref(self) -> Reference[T, lifetime]:
        """
        Gets a Reference to the first element of this slice.

        Returns:
            A Reference pointing at the first element of this slice.
        """

        return self._data[0]

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

    fn find[
        D: DType, //, from_left: Bool = True
    ](self: Span[Scalar[D]], subseq: Span[Scalar[D]], start: Int = 0) -> Int:
        """Finds the offset of the first occurrence of `subseq` starting at
        `start`. If not found, returns -1.

        Parameters:
            D: The `DType` of the Scalar.
            from_left: Whether to search the first occurrence from the left.

        Args:
            subseq: The sub sequence to find.
            start: The offset from which to find.

        Returns:
            The offset of `subseq` relative to the beginning of the `Span`.
        """
        var _len = len(self)

        if not subseq:

            @parameter
            if from_left:
                return 0
            else:
                return _len

        if _len < len(subseq) + start:
            return -1

        var start_norm = max(_len + start, 0) if start < 0 else min(_len, start)
        var haystack = __type_of(self)(
            unsafe_ptr=self.unsafe_ptr() + start_norm, len=_len - start_norm
        )
        var loc: UnsafePointer[Scalar[D]]

        @parameter
        if from_left:
            loc = _memmem(haystack, subseq)
        else:
            loc = _memrmem(haystack, subseq)

        return int(loc) - int(self.unsafe_ptr()) if loc else -1

    @always_inline
    fn rfind[
        D: DType, //
    ](self: Span[Scalar[D]], subseq: Span[Scalar[D]], start: Int = 0) -> Int:
        """Finds the offset of the last occurrence of `subseq` starting at
        `start`. If not found, returns -1.

        Parameters:
            D: The `DType` of the Scalar.

        Args:
            subseq: The sub sequence to find.
            start: The offset from which to find.

        Returns:
            The offset of `subseq` relative to the beginning of the `Span`.
        """
        return self.find[from_left=False](subseq, start)


# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _align_down(value: Int, alignment: Int) -> Int:
    return value._positive_div(alignment) * alignment


@always_inline
fn _memchr[
    type: DType
](
    source: UnsafePointer[Scalar[type]], char: Scalar[type], len: Int
) -> UnsafePointer[Scalar[type]]:
    if not len:
        return UnsafePointer[Scalar[type]]()
    alias bool_mask_width = simdwidthof[DType.bool]()
    var first_needle = SIMD[type, bool_mask_width](char)
    var vectorized_end = _align_down(len, bool_mask_width)

    for i in range(0, vectorized_end, bool_mask_width):
        var bool_mask = source.load[width=bool_mask_width](i) == first_needle
        var mask = bitcast[_uint_type_of_width[bool_mask_width]()](bool_mask)
        if mask:
            return source + int(i + count_trailing_zeros(mask))

    for i in range(vectorized_end, len):
        if source[i] == char:
            return source + i
    return UnsafePointer[Scalar[type]]()


@always_inline
fn _memmem[
    type: DType
](
    haystack_span: Span[Scalar[type]], needle_span: Span[Scalar[type]]
) -> UnsafePointer[Scalar[type]]:
    var haystack = haystack_span.unsafe_ptr()
    var haystack_len = len(haystack_span)
    var needle = needle_span.unsafe_ptr()
    var needle_len = len(needle_span)
    if not needle_len:
        return haystack
    if needle_len > haystack_len:
        return UnsafePointer[Scalar[type]]()
    if needle_len == 1:
        return _memchr[type](haystack, needle[0], haystack_len)

    alias bool_mask_width = simdwidthof[DType.bool]()
    var vectorized_end = _align_down(
        haystack_len - needle_len + 1, bool_mask_width
    )

    var first_needle = SIMD[type, bool_mask_width](needle[0])
    var last_needle = SIMD[type, bool_mask_width](needle[needle_len - 1])

    for i in range(0, vectorized_end, bool_mask_width):
        var first_block = haystack.load[width=bool_mask_width](i)
        var last_block = haystack.load[width=bool_mask_width](
            i + needle_len - 1
        )

        var eq_first = first_needle == first_block
        var eq_last = last_needle == last_block

        var bool_mask = eq_first & eq_last
        var mask = bitcast[_uint_type_of_width[bool_mask_width]()](bool_mask)

        while mask:
            var offset = int(i + count_trailing_zeros(mask))
            if memcmp(haystack + offset + 1, needle + 1, needle_len - 1) == 0:
                return haystack + offset
            mask = mask & (mask - 1)

    # remaining partial block compare using byte-by-byte
    #
    for i in range(vectorized_end, haystack_len - needle_len + 1):
        if haystack[i] != needle[0]:
            continue

        if memcmp(haystack + i + 1, needle + 1, needle_len - 1) == 0:
            return haystack + i
    _ = haystack_span, needle_span
    return UnsafePointer[Scalar[type]]()


@always_inline
fn _memrchr[
    type: DType
](
    source: UnsafePointer[Scalar[type]], char: Scalar[type], len: Int
) -> UnsafePointer[Scalar[type]]:
    if not len:
        return UnsafePointer[Scalar[type]]()
    for i in reversed(range(len)):
        if source[i] == char:
            return source + i
    return UnsafePointer[Scalar[type]]()


@always_inline
fn _memrmem[
    type: DType
](
    haystack_span: Span[Scalar[type]], needle_span: Span[Scalar[type]]
) -> UnsafePointer[Scalar[type]]:
    var haystack = haystack_span.unsafe_ptr()
    var haystack_len = len(haystack_span)
    var needle = needle_span.unsafe_ptr()
    var needle_len = len(needle_span)
    if not needle_len:
        return haystack
    if needle_len > haystack_len:
        return UnsafePointer[Scalar[type]]()
    if needle_len == 1:
        return _memrchr[type](haystack, needle[0], haystack_len)
    for i in reversed(range(haystack_len - needle_len + 1)):
        if haystack[i] != needle[0]:
            continue
        if memcmp(haystack + i + 1, needle + 1, needle_len - 1) == 0:
            return haystack + i
    _ = haystack_span, needle_span
    return UnsafePointer[Scalar[type]]()
