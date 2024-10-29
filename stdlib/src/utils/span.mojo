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
from collections import normalize_index
from memory import Pointer, UnsafePointer, bitcast, memcmp
from builtin.builtin_list import _lit_mut_cast
from sys import simdwidthof
from bit import count_trailing_zeros
from builtin.dtype import _uint_type_of_width


trait AsBytes:
    """
    The `AsBytes` trait denotes a type that can be returned as a immutable byte
    span.
    """

    fn as_bytes(ref [_]self) -> Span[Byte, __origin_of(self)]:
        """Returns a contiguous slice of the bytes owned by this string.

        Returns:
            A contiguous slice pointing to the bytes owned by this string.

        Notes:
            This does not include the trailing null terminator.
        """
        ...


@value
struct _SpanIter[
    is_mutable: Bool, //,
    T: CollectionElement,
    origin: Origin[is_mutable].type,
    forward: Bool = True,
]:
    """Iterator for Span.

    Parameters:
        is_mutable: Whether the reference to the span is mutable.
        T: The type of the elements in the span.
        origin: The origin of the Span.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var src: Span[T, origin]

    @always_inline
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(
        inout self,
    ) -> Pointer[T, origin]:
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
    origin: Origin[is_mutable].type,
](CollectionElementNew):
    """A non owning view of contiguous data.

    Parameters:
        is_mutable: Whether the span is mutable.
        T: The type of the elements in the span.
        origin: The origin of the Span.
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
    fn __init__(inout self, ref [origin]list: List[T, *_]):
        """Construct a Span from a List.

        Args:
            list: The list to which the span refers.
        """
        self._data = list.data
        self._len = len(list)

    @always_inline
    fn __init__[
        size: Int, //
    ](inout self, ref [origin]array: InlineArray[T, size]):
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
    fn __getitem__(self, idx: Int) -> ref [origin] T:
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
    fn __iter__(self) -> _SpanIter[T, origin]:
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

    fn as_ref(self) -> Pointer[T, origin]:
        """
        Gets a Pointer to the first element of this slice.

        Returns:
            A Pointer pointing at the first element of this slice.
        """

        return Pointer[T, origin].address_of(self._data[0])

    @always_inline
    fn copy_from[
        origin: MutableOrigin, //
    ](self: Span[T, origin], other: Span[T, _]):
        """
        Performs an element wise copy from all elements of `other` into all elements of `self`.

        Parameters:
            origin: The inferred mutable origin of the data within the Span.

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
    # accesses to the origin.
    @__unsafe_disable_nested_origin_exclusivity
    fn __eq__[
        T: EqualityComparableCollectionElement, //
    ](self: Span[T, origin], rhs: Span[T]) -> Bool:
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
    ](self: Span[T, origin], rhs: Span[T]) -> Bool:
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

    fn fill[origin: MutableOrigin, //](self: Span[T, origin], value: T):
        """
        Fill the memory that a span references with a given value.

        Parameters:
            origin: The inferred mutable origin of the data within the Span.

        Args:
            value: The value to assign to each element.
        """
        for element in self:
            element[] = value

    fn get_immutable(self) -> Span[T, _lit_mut_cast[origin, False].result]:
        """
        Return an immutable version of this span.

        Returns:
            A span covering the same elements, but without mutability.
        """
        return Span[T, _lit_mut_cast[origin, False].result](
            unsafe_ptr=self._data, len=self._len
        )

    fn find[
        O1: ImmutableOrigin,
        O2: ImmutableOrigin,
        D: DType, //,
        from_left: Bool = True,
        single_value: Bool = False,
        unsafe_dont_normalize: Bool = False,
    ](
        self: Span[Scalar[D], O1], subseq: Span[Scalar[D], O2], start: Int = 0
    ) -> Int:
        """Finds the offset of the first occurrence of `subseq` starting at
        `start`. If not found, returns `-1`.

        Parameters:
            O1: The immutable origin of `self`.
            O2: The immutable origin of `subseq`.
            D: The `DType` of the Scalar.
            from_left: Whether to search the first occurrence from the left.
            single_value: Whether to search with the `subseq`s first value.
            unsafe_dont_normalize: Whether to not normalize the index (no
                negative indexing, no bounds checks).

        Args:
            subseq: The sub sequence to find.
            start: The offset from which to find.

        Returns:
            The offset of `subseq` relative to the beginning of the `Span`.
        """
        _len = len(self)

        if not subseq:

            @parameter
            if from_left:
                return 0
            else:
                return _len

        var n_s: Int

        @parameter
        if unsafe_dont_normalize:
            debug_assert(0 <= start < _len, "unsafe index access")
            n_s = start
        else:
            n_s = normalize_index["Span", ignore_zero_length=True](start, self)
        s_ptr = self.unsafe_ptr()
        haystack = __type_of(self)(unsafe_ptr=s_ptr + n_s, len=_len - n_s)
        var loc: UnsafePointer[Scalar[D]]

        @parameter
        if from_left and not single_value:
            loc = _memmem(haystack, subseq)
        elif from_left:
            loc = _memchr(haystack, subseq.unsafe_ptr()[0])
        elif not single_value:
            loc = _memrmem(haystack, subseq)
        else:
            loc = _memrchr(haystack, subseq.unsafe_ptr()[0])

        return int(loc) - int(s_ptr) if loc else -1

    @always_inline
    fn rfind[
        O1: ImmutableOrigin,
        O2: ImmutableOrigin,
        D: DType, //,
        single_value: Bool = False,
    ](
        self: Span[Scalar[D], O1], subseq: Span[Scalar[D], O2], start: Int = 0
    ) -> Int:
        """Finds the offset of the last occurrence of `subseq` starting at
        `start`. If not found, returns `-1`.

        Parameters:
            O1: The immutable origin of `self`.
            O2: The immutable origin of `subseq`.
            D: The `DType` of the Scalar.
            single_value: Whether to search with the `subseq`s first value.

        Args:
            subseq: The sub sequence to find.
            start: The offset from which to find.

        Returns:
            The offset of `subseq` relative to the beginning of the `Span`.
        """
        return self.find[from_left=False, single_value=single_value](
            subseq, start
        )


# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _align_down(value: Int, alignment: Int) -> Int:
    return value._positive_div(alignment) * alignment


@always_inline
fn _memchr[
    O: ImmutableOrigin, //, type: DType
](span: Span[Scalar[type], O], char: Scalar[type]) -> UnsafePointer[
    Scalar[type]
] as output:
    haystack = span.unsafe_ptr()
    length = len(span)
    alias bool_mask_width = simdwidthof[DType.bool]()
    first_needle = SIMD[type, bool_mask_width](char)
    vectorized_end = _align_down(length, bool_mask_width)

    for i in range(0, vectorized_end, bool_mask_width):
        bool_mask = haystack.load[width=bool_mask_width](i) == first_needle
        mask = bitcast[_uint_type_of_width[bool_mask_width]()](bool_mask)
        if mask:
            output = haystack + int(i + count_trailing_zeros(mask))
            return

    for i in range(vectorized_end, length):
        if haystack[i] == char:
            output = haystack + i
            return

    output = UnsafePointer[Scalar[type]]()


@always_inline
fn _memmem[
    O: ImmutableOrigin, //, type: DType
](
    haystack_span: Span[Scalar[type], O], needle_span: Span[Scalar[type]]
) -> UnsafePointer[Scalar[type]] as output:
    haystack = haystack_span.unsafe_ptr()
    haystack_len = len(haystack_span)
    needle = needle_span.unsafe_ptr()
    needle_len = len(needle_span)
    debug_assert(needle_len > 0, "needle_len must be > 0")
    if needle_len == 1:
        output = _memchr[type](haystack_span, needle[0])
        return
    elif needle_len > haystack_len:
        output = UnsafePointer[Scalar[type]]()
        return

    alias bool_mask_width = simdwidthof[DType.bool]()
    vectorized_end = _align_down(haystack_len - needle_len + 1, bool_mask_width)

    first_needle = SIMD[type, bool_mask_width](needle[0])
    last_needle = SIMD[type, bool_mask_width](needle[needle_len - 1])

    for i in range(0, vectorized_end, bool_mask_width):
        first_block = haystack.load[width=bool_mask_width](i)
        last_block = haystack.load[width=bool_mask_width](i + needle_len - 1)

        bool_mask = (first_needle == first_block) & (last_needle == last_block)
        mask = bitcast[_uint_type_of_width[bool_mask_width]()](bool_mask)

        while mask:
            offset = int(i + count_trailing_zeros(mask))
            if memcmp(haystack + offset + 1, needle + 1, needle_len - 1) == 0:
                output = haystack + offset
                return
            mask = mask & (mask - 1)

    # remaining partial block compare using byte-by-byte
    for i in range(vectorized_end, haystack_len - needle_len + 1):
        if haystack[i] != needle[0]:
            continue

        if memcmp(haystack + i + 1, needle + 1, needle_len - 1) == 0:
            output = haystack + i
            return
    output = UnsafePointer[Scalar[type]]()


@always_inline
fn _memrchr[
    O: ImmutableOrigin, //, type: DType
](span: Span[Scalar[type], O], char: Scalar[type]) -> UnsafePointer[
    Scalar[type]
] as output:
    haystack = span.unsafe_ptr()
    length = len(span)
    alias bool_mask_width = simdwidthof[DType.bool]()
    first_needle = SIMD[type, bool_mask_width](char)
    vectorized_end = _align_down(length, bool_mask_width)

    for i in reversed(range(vectorized_end, length)):
        if haystack[i] == char:
            output = haystack + i
            return

    for i in reversed(range(0, vectorized_end, bool_mask_width)):
        bool_mask = (
            haystack.load[width=bool_mask_width](i).reversed() == first_needle
        )
        mask = bitcast[_uint_type_of_width[bool_mask_width]()](bool_mask)
        if mask:
            output = haystack + int(i + count_trailing_zeros(mask))
            return

    output = UnsafePointer[Scalar[type]]()


@always_inline
fn _memrmem[
    O: ImmutableOrigin, //, type: DType
](
    haystack_span: Span[Scalar[type], O], needle_span: Span[Scalar[type]]
) -> UnsafePointer[Scalar[type]] as output:
    haystack = haystack_span.unsafe_ptr()
    haystack_len = len(haystack_span)
    needle = needle_span.unsafe_ptr()
    needle_len = len(needle_span)
    debug_assert(needle_len > 0, "needle_len must be > 0")

    if needle_len == 1:
        output = _memrchr[type](haystack_span, needle[0])
        return
    elif needle_len > haystack_len:
        output = UnsafePointer[Scalar[type]]()
        return

    alias bool_mask_width = simdwidthof[DType.bool]()
    vectorized_end = _align_down(haystack_len - needle_len + 1, bool_mask_width)

    for i in reversed(range(vectorized_end, haystack_len - needle_len + 1)):
        if haystack[i] != needle[0]:
            continue

        if memcmp(haystack + i + 1, needle + 1, needle_len - 1) == 0:
            output = haystack + i
            return

    first_needle = SIMD[type, bool_mask_width](needle[0])
    last_needle = SIMD[type, bool_mask_width](needle[needle_len - 1])

    for i in reversed(range(0, vectorized_end, bool_mask_width)):
        first_block = haystack.load[width=bool_mask_width](i).reversed()
        last_block = haystack.load[width=bool_mask_width](
            i + needle_len - 1
        ).reversed()

        bool_mask = (first_needle == first_block) & (last_needle == last_block)
        mask = bitcast[_uint_type_of_width[bool_mask_width]()](bool_mask)

        while mask:
            offset = int(i + count_trailing_zeros(mask))
            if memcmp(haystack + offset + 1, needle + 1, needle_len - 1) == 0:
                output = haystack + offset
                return
            mask = mask & (mask - 1)

    output = UnsafePointer[Scalar[type]]()
