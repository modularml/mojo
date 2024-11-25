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
"""Defines the DBuffer type."""


from builtin.builtin_list import _lit_mut_cast
from collections import InlineArray, List
from memory import UnsafePointer
from sys.info import bitwidthof
from utils import Span


@value
struct _DBufferIter[
    is_mutable: Bool, //,
    T: CollectionElement,
    origin: Origin[is_mutable].type,
    forward: Bool = True,
]:
    """Iterator for DBuffer.

    Parameters:
        is_mutable: Whether the reference to the DBuffer is mutable.
        T: The type of the elements in the DBuffer.
        origin: The origin of the DBuffer.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var src: DBuffer[T, origin]

    @always_inline
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(inout self) -> ref [origin] T:
        @parameter
        if forward:
            self.index += 1
            return self.src[self.index - 1]
        else:
            self.index -= 1
            return self.src[self.index]

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        @parameter
        if forward:
            return len(self.src) - self.index
        else:
            return self.index


# TODO: decide whether DBuffer will be the entrypoint API for Python-like bytes
# alias Bytes = DBuffer[Byte, _]
# """A buffer of bytes."""


struct DBuffer[
    is_mutable: Bool, //,
    T: CollectionElement,
    origin: Origin[is_mutable].type,
](CollectionElementNew):
    """A potentially owning view of contiguous data.

    Parameters:
        is_mutable: Whether the DBuffer is mutable.
        T: The type of the elements in the DBuffer.
        origin: The origin of the DBuffer.

    Examples:

    ```mojo
    %# from collections.dbuffer import DBuffer
    fn parse(
        owned buf: DBuffer[Byte], encoding: String = "utf-8"
    ) raises -> String:
        if encoding == "utf-16":
            ...
        elif encoding == "utf-8":
            debug_assert(
                len(buf) > 0 and buf[-1] == 0,
                "parser expects null terminated data"
            )
            return String(ptr=buf.steal_data(), length=len(buf))
        else:
            raise Error("Unsupported encoding")

    fn main() raises:
        l1 = List[Byte](ord("h"), ord("i"), 0)
        # l1 gets implicitly built into a DBuffer that doesn't own the data.
        # Since the DBuffer doesn't own the data, the method steal_data()
        # makes a copy of the data to pass as owned to the String constructor
        print(parse(l1)) # hi
        # Passing an owned DBuffer makes the .steal_data() inside the parse
        # method not allocate
        print(parse(DBuffer[origin=MutableAnyOrigin].own(l1^))) # hi
        # the compiler won't let you use l1 beyond this point
    ```
    .
    """

    alias _intwidth = bitwidthof[Int]()
    alias _shift = Self._intwidth - 1
    alias _len_mask = 0x7F_FF if Self._intwidth == 32 else 0x7F_FF_FF_FF
    alias _max_length = 2**Self._shift
    var _data: UnsafePointer[T]
    var _len: UInt

    # ===------------------------------------------------------------------===#
    # Life cycle methods
    # ===------------------------------------------------------------------===#

    @always_inline
    fn __init__(
        out self,
        *,
        ptr: UnsafePointer[T],
        length: UInt,
        self_is_owner: Bool = False,
        is_stack_alloc: Bool = False,
    ):
        """Unsafe construction from a pointer and length.

        Args:
            ptr: The underlying pointer of the DBuffer.
            length: The length of the view.
            self_is_owner: Whether the DBuffer instance is the owner of the
                data.
            is_stack_alloc: Whether the pointer is a stack allocation.

        Notes:
            If `is_stack_alloc` is True, then the `is_owner` bit is set to
            False.
        """

        debug_assert(
            length <= Self._max_length, "length must be <= ", Self._max_length
        )
        self._data = ptr
        self._len = length | (
            UInt(self_is_owner and not is_stack_alloc) << Self._shift
        )

    @always_inline
    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided DBuffer.

        Args:
            other: The DBuffer to copy.
        """

        var o_len = len(other)
        var buf = UnsafePointer[T].alloc(o_len)
        for i in range(o_len):
            buf[i] = other._data[i]
        self = Self(ptr=buf, length=o_len, self_is_owner=True)

    @always_inline
    @implicit
    fn __init__(out self, ref [origin]list: List[T, *_]):
        """Construct a DBuffer from a List.

        Args:
            list: The list to which the DBuffer refers.
        """
        self = Self(ptr=list.unsafe_ptr(), length=len(list))

    # TODO: this needs some sort of "SelfOrigin" which binds it to the
    # variable that holds it
    # TODO: this can potentially be abstracted over a `Stealable` trait
    @always_inline
    @staticmethod
    fn own(owned list: List[T, *_]) -> Self:
        """Construct a DBuffer from an owned List.

        Args:
            list: The list to which the DBuffer refers.

        Returns:
            The owned DBuffer with the data.

        Examples:

        ```mojo
        %# from collections.dbuffer import DBuffer
        l1 = List[Int](1, 2, 3, 4, 5, 6, 7)
        s1 = DBuffer[origin=MutableAnyOrigin].own(l1^)
        ```
        .
        """
        var l_len = len(list)  # to avoid steal_data() which sets it to 0
        return Self(ptr=list.steal_data(), length=l_len, self_is_owner=True)

    @always_inline
    @implicit
    fn __init__[
        size: Int, //
    ](inout self, ref [origin]array: InlineArray[T, size]):
        """Construct a DBuffer from an InlineArray.

        Parameters:
            size: The size of the InlineArray.

        Args:
            array: The array to which the DBuffer refers.
        """
        self = Self(
            ptr=UnsafePointer.address_of(array).bitcast[T](), length=UInt(size)
        )

    @always_inline
    @implicit
    fn __init__(out self, span: Span[T, origin]):
        """Construct a DBuffer from a Span.

        Args:
            span: The span from which to construct a DBuffer.
        """
        self = Self(ptr=span.unsafe_ptr(), length=len(span))

    fn __moveinit__(out self, owned existing: Self):
        """Move data of an existing DBuffer into a new one.

        Args:
            existing: The existing DBuffer.
        """
        self._data = existing._data
        self._len = existing._len

    fn __copyinit__(out self, existing: Self):
        """Creates a shallow non-owning copy of the given DBuffer.

        Args:
            existing: The DBuffer to copy.
        """
        self = Self(ptr=existing.unsafe_ptr(), length=len(existing))

    fn __del__(owned self):
        """If self.is_owner(), destroy all elements in the DBuffer and free its
        memory."""

        if self.is_owner():
            for i in range(len(self)):
                (self._data + i).destroy_pointee()
            self._data.free()

    # ===------------------------------------------------------------------===#
    # Operator dunders
    # ===------------------------------------------------------------------===#

    @always_inline
    fn __getitem__(self, idx: Int) -> ref [origin] T:
        """Get a reference to an element in the DBuffer.

        Args:
            idx: The index of the value to return.

        Returns:
            An element reference.
        """

        # TODO: use normalize_index
        debug_assert(
            -len(self) <= int(idx) < len(self), "index must be within bounds"
        )
        var offset = idx
        if offset < 0:
            offset += len(self)
        return self._data[offset]

    @always_inline
    fn __getitem__(self, slc: Slice) -> Self:
        """Get a new DBuffer from a slice of the current DBuffer.

        Args:
            slc: The slice specifying the range of the new subslice.

        Returns:
            A new DBuffer that points to the same data as the current DBuffer.

        Notes:
            If the step is not 1, this allocates a new buffer which owns the
            data.
        """

        start, end, step = slc.indices(len(self))

        if step == 1:
            return Self(ptr=self._data + start, length=end - start)

        var new_len = len(range(start, end, step))
        var buf = UnsafePointer[T].alloc(new_len)
        var i = 0
        # TODO: DType branch using memcpy

        if step < 0:
            while start > end:
                buf[i] = self._data[start]
                start += step
                i += 1
            return Self(ptr=buf, length=new_len, self_is_owner=True)

        while start < end:
            buf[i] = self._data[start]
            start += step
            i += 1
        return Self(ptr=buf, length=new_len, self_is_owner=True)

    @always_inline
    fn __iter__(self) -> _DBufferIter[T, origin]:
        """Get an iterator over the elements of the DBuffer.

        Returns:
            An iterator over the elements of the DBuffer.
        """
        return _DBufferIter(0, self)

    # ===------------------------------------------------------------------===#
    # Trait implementations
    # ===------------------------------------------------------------------===#

    @always_inline
    fn __len__(self) -> Int:
        """Returns the length of the DBuffer. This is a known constant value.

        Returns:
            The size of the DBuffer.
        """
        return int(self._len) & Self._len_mask

    fn __bool__(self) -> Bool:
        """Check if a DBuffer is non-empty.

        Returns:
            True if a DBuffer is non-empty, False otherwise.
        """
        return len(self) > 0

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the origin.
    @__unsafe_disable_nested_origin_exclusivity
    fn __eq__[
        T: EqualityComparableCollectionElement, //
    ](self: DBuffer[T, origin], rhs: DBuffer[T]) -> Bool:
        """Verify if DBuffer is equal to another DBuffer.

        Parameters:
            T: The type of the elements in the DBuffer. Must implement the
                traits `EqualityComparable` and `CollectionElement`.

        Args:
            rhs: The DBuffer to compare against.

        Returns:
            True if the DBuffers are equal in length and contain the same
            elements, False otherwise.
        """
        # both empty
        if not self and not rhs:
            return True
        if len(self) != len(rhs):
            return False
        # same pointer and length, so equal
        if self.unsafe_ptr() == rhs.unsafe_ptr():
            return True

        # TODO: DType branch using memcmp
        for i in range(len(self)):
            if self[i] != rhs[i]:
                return False
        return True

    @always_inline
    fn __ne__[
        T: EqualityComparableCollectionElement, //
    ](self: DBuffer[T, origin], rhs: DBuffer[T]) -> Bool:
        """Verify if DBuffer is not equal to another DBuffer.

        Parameters:
            T: The type of the elements in the DBuffer. Must implement the
              traits `EqualityComparable` and `CollectionElement`.

        Args:
            rhs: The DBuffer to compare against.

        Returns:
            True if the DBuffers are not equal in length or contents, False
            otherwise.
        """
        return not self == rhs

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn unsafe_ptr(self) -> UnsafePointer[T]:
        """Gets a pointer to the first element of this DBuffer.

        Returns:
            A pointer pointing at the first element of this DBuffer.
        """
        return self._data

    @always_inline("nodebug")
    fn is_owner(self) -> Bool:
        """Whether the DBuffer is the owner of the data.

        Returns:
            Whether the DBuffer is the owner of the data.

        Notes:
            If the pointer is stack allocated, this will always return False.
        """
        return bool(self._len >> Self._shift)

    fn steal_data(inout self) -> UnsafePointer[T]:
        """Take ownership of the underlying pointer from the DBuffer if
        `self.is_owner()`, otherwise create a deep copy.

        Returns:
            The underlying data if `self.is_owner()`, otherwise a deep copy.

        Notes:
            The DBuffer will still work as a non owning reference to the data.
            In the case that the DBuffer's pointer was stack allocated, this
            will copy the data to a new pointer in the heap.
        """

        if self.is_owner():
            self._len &= Self._len_mask
            return self.unsafe_ptr()

        var s_len = len(self)
        var buf = UnsafePointer[T].alloc(s_len)
        # TODO: DType branch using memcpy
        for i in range(s_len):
            (buf + i).init_pointee_copy(self[i])
        return buf

    fn get_immutable(self) -> DBuffer[T, _lit_mut_cast[origin, False].result]:
        """Return an immutable version of this DBuffer.

        Returns:
            A DBuffer covering the same elements, but without mutability.
        """
        return DBuffer[T, _lit_mut_cast[origin, False].result](
            ptr=self.unsafe_ptr(), length=len(self)
        )

    fn fill[origin: MutableOrigin, //](self: DBuffer[T, origin], value: T):
        """Fill the memory that a DBuffer references with a given value.

        Parameters:
            origin: The inferred mutable origin of the data within the DBuffer.

        Args:
            value: The value to assign to each element.
        """

        var ptr = self.unsafe_ptr()
        # TODO: DType branch using memset
        for i in range(len(self)):
            ptr[i] = value
