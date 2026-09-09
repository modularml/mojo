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

"""Implements the `Span` type.

You can import these APIs from the `memory` module. For example:

```mojo
from std.collections import Span
```
"""
from std.builtin.builtin_slice import ContiguousSlice
from std.reflection import call_location, reflect
from std.bit.mask import splat
from std.bit import pop_count
from std.memory import (
    pack_bits,
    unsafe_uninit_copy_n,
)
from std.collections import check_bounds, check_slice_bounds
from std.builtin.rebind import downcast
from std.sys import align_of
from std.sys.info import simd_width_of
from std.traits import IsTriviallyCopyable, IsTriviallyDeinitable

from std.algorithm import vectorize
from std.hashlib import Hasher
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
import std.format._utils as fmt


# ===-----------------------------------------------------------------------===#
# Span aliases
# ===-----------------------------------------------------------------------===#


comptime MutSpan[
    T: AnyType,
    origin: MutOrigin,
    *,
    address_space: AddressSpace = .GENERIC,
] = Span[T, origin, address_space=address_space]
"""A span providing mutable access to its elements.

Parameters:
    T: The type of the elements in the span.
    origin: The origin of the span.
    address_space: The address space of the span.
"""

comptime ImmSpan[
    T: AnyType,
    origin: ImmOrigin,
    *,
    address_space: AddressSpace = .GENERIC,
] = Span[T, origin, address_space=address_space]
"""A span providing read-only access to its elements.

Parameters:
    T: The type of the elements in the span.
    origin: The origin of the span.
    address_space: The address space of the span.
"""


@fieldwise_init
struct _SpanIter[
    mut: Bool,
    //,
    T: Copyable,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, IterableOwned, Iterator, Sized):
    """Iterator for Span.

    Parameters:
        mut: Whether the reference to the span is mutable.
        T: The type of the elements in the span.
        origin: The origin of the `Span`.
        forward: The iteration direction. False is backwards.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    """The iterator type for this span iterator.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """
    comptime IteratorOwnedType: Iterator = Self
    """The owned iterator type for this span iterator."""

    comptime Element = Self.T
    """The element type yielded by iteration."""

    var index: Int
    var src: Span[Self.T, Self.origin]

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    @always_inline
    def __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
        comptime if Self.forward:
            if self.index >= len(self.src):
                raise StopIteration()

            var curr = self.index
            self.index += 1
            return self.src._data[unsafe_offset=curr]
        else:
            if self.index <= 0:
                raise StopIteration()
            self.index -= 1
            return self.src._data[unsafe_offset=self.index]

    @always_inline
    def __len__(self) -> Int:
        """Returns the number of elements remaining in this iterator.

        Returns:
            The number of elements remaining in this iterator.
        """
        comptime if Self.forward:
            return len(self.src) - self.index
        else:
            return self.index

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        """Returns bounds `[lower, upper]` for the remaining iterator length.

        Returns:
            The lower and upper bound of this iterator. For `_SpanIter` both
            bounds are exact and equal to `len(self)`.
        """
        var n = len(self)
        return (n, {n})


@stable(since="1.0")
struct Span[
    mut: Bool,
    //,
    T: AnyType,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = .GENERIC,
](
    Boolable,
    Defaultable,
    DevicePassable where address_space == .GENERIC,
    Hashable where conforms_to(T, Hashable) and address_space == .GENERIC,
    ImplicitlyCopyable,
    Iterable where address_space == .GENERIC,
    IterableOwned where conforms_to(T, Copyable) and address_space == .GENERIC,
    Sized,
    TrivialRegisterPassable,
    Writable where conforms_to(T, Writable) and address_space == .GENERIC,
):
    """A non-owning view of contiguous data.

    Parameters:
        mut: Whether the span is mutable.
        T: The type of the elements in the span.
        origin: The origin of the Span.
        address_space: The address space of the data the span views.
    """

    # Aliases
    comptime Immutable = Span[
        Self.T, ImmOrigin(Self.origin), address_space=Self.address_space
    ]
    """The immutable version of the `Span`."""
    comptime _PointerType = Pointer[
        Self.T, Self.origin, address_space=Self.address_space
    ]
    """The (safe) pointer type for this `Span`."""
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _SpanIter[downcast[Self.T, Copyable], Self.origin]
    """The iterator type for this `Span`.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """
    comptime IteratorOwnedType: Iterator = _SpanIter[
        downcast[Self.T, Copyable], Self.origin
    ]
    """The owned iterator type for this `Span`."""

    comptime _is_generic_as = Self.address_space == .GENERIC
    """Whether this `Span` views the default (generic) address space.

    Most element-copying operations are only available on
    generic-address-space spans: the compiler rejects copying a value that is
    not register passable into or out of a non-default address space. `fill()`
    is the exception, accepting any address space when the element type
    conforms to `TrivialRegisterPassable`.
    """

    # Fields
    var _data: Self._PointerType
    var _len: Int

    comptime device_type: AnyType = Self
    """The device-side type for this `Span`."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ) where Self._is_generic_as:
        """Device type mapping is the identity function."""
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """
        Gets this type's name, for use in error messages when handing arguments
        to kernels.

        Returns:
            This type's name.
        """
        return String(
            "Span[",
            reflect[Self.T].name(),
            "]",
        )

    # ===------------------------------------------------------------------===#
    # Life cycle methods
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    @stable(since="1.0")
    def __init__(out self):
        """Create an empty / zero-length span."""
        self._data = Self._PointerType.unsafe_dangling()
        self._len = 0

    @doc_hidden
    @implicit
    @always_inline("nodebug")
    @stable(since="1.0")
    def __init__(
        other: Span,
        out self: ImmSpan[
            other.T, other.origin, address_space=other.address_space
        ],
    ):
        """Implicitly cast the mutable origin of self to an immutable one.

        Args:
            other: The Span to cast.
        """
        self = rebind[type_of(self)](other)

    @always_inline("builtin")
    def __init__(
        out self,
        *,
        unsafe_ptr: Pointer[
            Self.T, Self.origin, address_space=Self.address_space
        ],
        length: Int,
    ):
        """Unsafe construction from a pointer and length.

        Args:
            unsafe_ptr: The underlying pointer of the span.
            length: The length of the view.
        """
        self._data = unsafe_ptr
        self._len = length

    @always_inline
    @implicit
    def __init__(
        out self,
        ref[Self.origin] list: List[Self.T],
    ) where conforms_to(Self.T, Movable):
        """Construct a `Span` from a `List`.

        Args:
            list: The list to which the span refers.
        """
        self._data = rebind[type_of(self._data)](list.unsafe_ptr())
        self._len = len(list)

    @always_inline
    @implicit
    def __init__(
        out self,
        ref[Self.origin, Self.address_space] array: Array[Self.T, _],
    ):
        """Construct a `Span` from an `Array`.

        Args:
            array: The array to which the span refers.
        """
        self._data = array.unsafe_ptr()
        self._len = array.length

    # ===------------------------------------------------------------------===#
    # Operator dunders
    # ===------------------------------------------------------------------===#

    @stable(since="1.0")
    @always_inline
    def __getitem__(
        self, idx: Int, /
    ) -> ref[Self.origin, Self.address_space] Self.T:
        """Gets the span element at the given index.

        Args:
            idx: The index of the element.

        Returns:
            A reference to the element at the given index.
        """
        check_bounds(idx, len(self))
        return self._data[unsafe_offset=idx]

    @always_inline
    def __getitem__(
        self, idx: Some[Indexer]
    ) -> ref[Self.origin, Self.address_space] Self.T:
        """Gets the span element at the given index.

        Args:
            idx: The index of the element.

        Returns:
            A reference to the element at the given index.
        """
        check_bounds(idx, len(self))
        return self._data[unsafe_offset=idx]

    @always_inline
    def __getitem__(
        self, idx: IntLiteral
    ) -> ref[Self.origin, Self.address_space] Self.T:
        """Gets the span element at the given index.

        Args:
            idx: The index of the element.

        Returns:
            A reference to the element at the given index.
        """
        comptime assert IntLiteral[idx.value]() >= 0, (
            "negative indexing is not supported, use e.g. `x[len(x) - 1]`"
            " instead"
        )
        check_bounds(idx, len(self))
        return self._data[unsafe_offset=idx]

    @always_inline
    def __getitem__(self, slc: ContiguousSlice) -> Self:
        """Get a new span from a slice of the current span.

        Aborts if `slc`'s start or end index is out of bounds (valid range is
        `0` to `len(self)`, inclusive), or if start is greater than end.
        Negative indices are not supported and always abort.

        Args:
            slc: The slice specifying the range of the new subslice.

        Returns:
            A new span that points to the same data as the current span.
        """
        var start, end = check_slice_bounds(slc, len(self))
        return Self(
            unsafe_ptr=self._data.unsafe_offset(start), length=end - start
        )

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType where Self._is_generic_as:
        """Consume the span and return an iterator over its elements.

        Returns:
            An iterator over the elements of the span.
        """
        comptime assert conforms_to(
            Self.T, Copyable
        ), "Span iteration requires the element to be `Copyable`"
        return _SpanIter(
            0, rebind[Span[downcast[Self.T, Copyable], Self.origin]](self)
        )

    @always_inline
    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)] where Self._is_generic_as:
        """Get an iterator over the elements of the `Span`.

        Returns:
            An iterator over the elements of the `Span`.
        """
        comptime assert conforms_to(
            Self.T, Copyable
        ), "Span iteration requires the element to be `Copyable`"
        return _SpanIter(
            0, rebind[Span[downcast[Self.T, Copyable], Self.origin]](self)
        )

    @always_inline
    def __reversed__(
        self,
    ) -> _SpanIter[
        downcast[Self.T, Copyable], Self.origin, forward=False
    ] where Self._is_generic_as:
        """Iterate backwards over the `Span`.

        Returns:
            A reversed iterator of the `Span` elements.
        """
        comptime assert conforms_to(
            Self.T, Copyable
        ), "Span iteration requires the element to be `Copyable`"
        return _SpanIter[forward=False](
            len(self),
            rebind[Span[downcast[Self.T, Copyable], Self.origin]](self),
        )

    # ===------------------------------------------------------------------===#
    # Trait implementations
    # ===------------------------------------------------------------------===#

    @always_inline("builtin")
    def __len__(self) -> Int:
        """Returns the length of the span. This is a known constant value.

        Returns:
            The size of the span.
        """
        return self._len

    @__allow_legacy_custom_self_type
    @stable(since="1.0")
    def __contains__[
        dtype: DType, //
    ](
        self: Span[Scalar[dtype], _, address_space=_], value: Scalar[dtype]
    ) -> Bool:
        """Verify if a given value is present in the Span.

        Parameters:
            dtype: The DType of the scalars stored in the Span.

        Args:
            value: The value to find.

        Returns:
            True if the value is contained in the list, False otherwise.
        """

        comptime widths: Array[Int, 6] = [256, 128, 64, 32, 16, 8]
        var ptr = self.unsafe_ptr()
        var length = len(self)
        var processed = 0

        comptime for i in range(len(widths)):
            comptime width = widths[i]

            comptime if simd_width_of[dtype]() >= width:
                for _ in range((length - processed) // width):
                    if (
                        value
                        in ptr.unsafe_offset(processed).unsafe_load[
                            width=width
                        ]()
                    ):
                        return True
                    processed += width

        for i in range(length - processed):
            if ptr[unsafe_offset=processed + i] == value:
                return True
        return False

    @stable(since="1.0")
    @__allow_legacy_custom_self_type
    def __contains__(
        self: Span[Self.T, _], value: Self.T
    ) -> Bool where conforms_to(Self.T, Equatable):
        """Verify if a given value is present in the span.

        Performs a linear scan over all elements comparing with `==`.

        Args:
            value: The value to find.

        Returns:
            True if the value is contained in the span, False
            otherwise.
        """
        for i in range(len(self)):
            if self[i] == value:
                return True
        return False

    def _write_self_to[
        f: def(Self.T, mut Some[Writer]) thin
    ](self, mut writer: Some[Writer]) where Self._is_generic_as:
        var iterator = self.__iter__()

        def iterate(mut w: Some[Writer]) raises StopIteration {mut iterator}:
            f(iterator.__next__(), w)

        fmt.write_sequence_to(writer, iterate)
        _ = iterator^

    @no_inline
    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable) and Self._is_generic_as:
        """Write this span to a `Writer`.

        Args:
            writer: The object to write to.
        """
        self._write_self_to[f=fmt.write_to[Self.T]](writer)

    @no_inline
    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable) and Self._is_generic_as:
        """Write this span to a `Writer`.

        Args:
            writer: The object to write to.
        """

        var self_ptr = Pointer(to=self)

        def write_fields(mut w: Some[Writer]) {self_ptr}:
            self_ptr[]._write_self_to[f=fmt.write_repr_to[Self.T]](w)

        fmt.FormatStruct(writer, "Span").params(
            fmt.Named("mut", Self.mut),
            fmt.TypeNames[Self.T](),
        ).fields(write_fields)

    def __hash__[
        H: Hasher
    ](self, mut hasher: H) where (
        conforms_to(Self.T, Hashable) and Self._is_generic_as
    ):
        """Updates hasher with the hash of each element in the span.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        # The `where` clause guarantees a generic address space, but doesn't
        # narrow `Self`, so rebind before borrowing elements out of the span.
        var this = rebind[Span[Self.T, Self.origin]](self)
        hasher._update_with_simd(Int64(len(this)))
        for i in range(len(this)):
            this[i].__hash__(hasher)

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @always_inline
    def as_imm(self) -> Self.Immutable:
        """Return an immutable version of this `Span`.

        Returns:
            An immutable version of the same `Span`.
        """
        return rebind[Self.Immutable](self)

    @always_inline
    def unsafe_get(
        self, idx: Some[Indexer]
    ) -> ref[Self.origin, Self.address_space] Self.T:
        """Get a reference to the element at `index` without bounds checking.

        Args:
            idx: The index of the element to get.

        Returns:
            A reference to the element at the specified index.

        Safety:
            - This function does not do bounds checking and assumes the provided
            index is in: [0, len(self)). Not upholding this contract will result
            in undefined behavior.
            - This function does not support wraparound for negative indices.
        """
        check_bounds[cpu_default=False](idx, len(self))
        return self._data[unsafe_offset=idx]

    @always_inline("builtin")
    def unsafe_ptr(
        self,
    ) -> Pointer[Self.T, Self.origin, address_space=Self.address_space]:
        """Retrieves a pointer to the underlying memory, or a dangling
        pointer if the span doesn't point to anything.

        You should use the `len` of this `Span` to determine if the pointer
        is valid for reads and writes.

        Returns:
            The pointer to the underlying memory.
        """
        return self._data

    @always_inline
    def as_ref(
        self,
    ) -> Pointer[Self.T, Self.origin, address_space=Self.address_space]:
        """
        Gets a `Pointer` to the first element of this span.

        Returns:
            A `Pointer` pointing at the first element of this span.
        """

        return Self._PointerType(to=self._data[unsafe_offset=0])

    @always_inline
    @__allow_legacy_custom_self_type
    def copy_from(
        self: MutSpan[Self.T, _], other: Span[Self.T, _]
    ) where conforms_to(Self.T, Copyable & Deinitable):
        """
        Performs an element wise copy from all elements of `other` into all elements of `self`.

        Args:
            other: The `Span` to copy all elements from.
        """
        assert len(self) == len(other), "Spans must be of equal length"
        # For trivial types, unsafe_uninit_copy_n is a single memcpy (no destroy
        # needed). For non-trivial types, we keep the single-pass assignment
        # loop rather than unsafe_destroy_n + unsafe_uninit_copy_n, which would
        # be two passes over memory with worse cache locality.
        comptime if IsTriviallyCopyable[Self.T] and IsTriviallyDeinitable[
            Self.T
        ]:
            unsafe_uninit_copy_n[overlapping=False](
                # TODO(MOCO-4220) once fixed remove the unsafe_mut_cast
                dest=self.unsafe_ptr().unsafe_mut_cast[True](),
                src=other.unsafe_ptr(),
                count=len(self),
            )
        else:
            for i in range(len(self)):
                # TODO(MOCO-4220) once fixed this is self[i] = other[i].copy()
                self.unsafe_ptr().unsafe_mut_cast[True]()[
                    unsafe_offset=i
                ] = other[i].copy()

    def __bool__(self) -> Bool:
        """Check if a span is non-empty.

        Returns:
           True if a span is non-empty, False otherwise.
        """
        return len(self) > 0

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the origin.
    @__unsafe_nested_origins_read_only
    @__allow_legacy_custom_self_type
    def __eq__(
        self: Span[Self.T, _], rhs: Span[Self.T, _]
    ) -> Bool where conforms_to(Self.T, Equatable):
        """Verify if span is equal to another span.

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
    @__allow_legacy_custom_self_type
    def __ne__(
        self: Span[Self.T, _], rhs: Span[Self.T, _]
    ) -> Bool where conforms_to(Self.T, Equatable):
        """Verify if span is not equal to another span.

        Args:
            rhs: The span to compare against.

        Returns:
            True if the spans are not equal in length or contents, False otherwise.
        """
        return not self == rhs

    def fill(
        self, value: Self.T
    ) where Self.mut and (
        (Self._is_generic_as and conforms_to(Self.T, Copyable & Deinitable))
        or conforms_to(Self.T, TrivialRegisterPassable)
    ):
        """
        Fill the memory that a span references with a given value.

        Args:
            value: The value to assign to each element.

        Constraints:
            The span must be mutable. A register-passable element type works in
            any address space. All other element types require the span
            to view the default (generic) address space.
        """
        comptime if conforms_to(Self.T, TrivialRegisterPassable):
            for i in range(len(self)):
                # TODO(MOCO-4220) once fixed update body to:
                # self._data[unsafe_offset=i]= value
                var p = self._data.unsafe_offset(i).unsafe_mut_cast[True]()
                p[] = value
        else:
            comptime assert Self._is_generic_as
            comptime assert conforms_to(Self.T, Copyable & Deinitable)
            for ref element in self:
                # TODO(MOCO-4220) once fixed update body to:
                # element = value.copy()
                var p = Pointer(to=element).unsafe_mut_cast[True]()
                p[] = value.copy()

    @always_inline
    @__allow_legacy_custom_self_type
    def unsafe_swap_elements(
        self: MutSpan[Self.T, _], a: Int, b: Int
    ) where conforms_to(Self.T, Movable):
        """Swap the values at indices `a` and `b` without performing bounds checking.

        Args:
            a: The first element's index.
            b: The second element's index.

        Safety:
            - Both `a` and `b` must be in: [0, len(self)).
        """
        debug_assert(
            0 <= a < len(self),
            "Index `a` out of bounds: ",
            a,
        )
        debug_assert(
            0 <= b < len(self),
            "Index `b` out of bounds: ",
            b,
        )
        # TODO(MOCO-4220) once fixed remove unsafe_mut_cast
        var ptr = self.unsafe_ptr().unsafe_mut_cast[True]()

        # `a` and `b` may be equal, so we cannot use `swap` directly.  The
        # unsafe_origin_cast silence the (correct) exclusivity error.
        ptr.unsafe_offset(a).unsafe_origin_cast[MutAnyOrigin]().swap_pointees(
            ptr.unsafe_offset(b)
        )

    @__allow_legacy_custom_self_type
    def swap_elements(
        self: MutSpan[Self.T, _], a: Int, b: Int
    ) raises where conforms_to(Self.T, Movable):
        """
        Swap the values at indices `a` and `b`.

        Args:
            a: The first argument index.
            b: The second argument index.

        Raises:
            If a or b are larger than the length of the span.
        """
        var length = len(self)
        if a > length or b > length:
            raise Error(
                "index out of bounds (length: ",
                length,
                ", a: ",
                a,
                ", b: ",
                b,
                ")",
            )

        self.unsafe_swap_elements(a, b)

    @always_inline("nodebug")
    def __merge_with__[
        other_type: type_of(Span[Self.T, _, address_space=Self.address_space]),
    ](
        self,
        out result: Span[
            Self.T,
            origin_of(Self.origin, other_type.origin),
            address_space=Self.address_space,
        ],
    ):
        """Returns a pointer merged with the specified `other_type`.

        Parameters:
            other_type: The type of the pointer to merge with.

        Returns:
            A pointer merged with the specified `other_type`.
        """
        return {
            unsafe_ptr = self._data.unsafe_mut_cast[
                result.mut
            ]().unsafe_origin_cast[result.origin](),
            length = self._len,
        }

    @__allow_legacy_custom_self_type
    def reverse[dtype: DType, //](self: MutSpan[Scalar[dtype], _]):
        """Reverse the elements of the `Span` inplace.

        Parameters:
            dtype: The DType of the scalars the `Span` stores.
        """

        comptime widths = (256, 128, 64, 32, 16, 8, 4, 2)
        var ptr = self.unsafe_ptr()
        var length = len(self)
        var middle = length // 2
        var is_odd = length % 2 != 0
        var processed = 0

        comptime for i in range(len(widths)):
            comptime w = rebind[Int](widths[i])

            comptime if simd_width_of[dtype]() >= w:
                for _ in range((middle - processed) // w):
                    var lhs_ptr = ptr.unsafe_offset(processed)
                    var rhs_ptr = ptr.unsafe_offset(length - (processed + w))
                    var lhs_v = lhs_ptr.unsafe_load[width=w]().reversed()
                    var rhs_v = rhs_ptr.unsafe_load[width=w]().reversed()
                    lhs_ptr.unsafe_store(rhs_v)
                    rhs_ptr.unsafe_store(lhs_v)
                    processed += w

        if is_odd:
            var value = ptr[unsafe_offset=middle + 1]
            # Use an unsafe origin cast to silence the (correct) exclusivity error.
            var middle_prev = ptr.unsafe_offset(middle - 1).unsafe_origin_cast[
                MutAnyOrigin
            ]()
            ptr.unsafe_offset(middle + 1).unsafe_write_move_from(middle_prev)
            middle_prev.unsafe_write(value)

    @__allow_legacy_custom_self_type
    def apply[
        dtype: DType,
        //,
    ](
        self: MutSpan[Scalar[dtype], _],
        func: Some[def[w: SIMDLength](SIMD[dtype, w]) -> SIMD[dtype, w]],
    ):
        """Apply the function to the `Span` inplace.

        Parameters:
            dtype: The DType.

        Args:
            func: The function to evaluate.
        """

        comptime widths = (256, 128, 64, 32, 16, 8, 4)
        var ptr = self.unsafe_ptr()
        var length = len(self)
        var processed = 0

        comptime for i in range(len(widths)):
            comptime w = rebind[Int](widths[i])

            comptime if simd_width_of[dtype]() >= w:
                for _ in range((length - processed) // w):
                    var p_curr = ptr.unsafe_offset(processed)
                    p_curr.unsafe_store(func(p_curr.unsafe_load[width=w]()))
                    processed += w

        for i in range(length - processed):
            ptr.unsafe_offset(processed + i).unsafe_write(
                func(ptr[unsafe_offset=processed + i])
            )

    @__allow_legacy_custom_self_type
    def apply[
        dtype: DType,
        //,
    ](
        self: MutSpan[Scalar[dtype], _],
        func: Some[def[w: SIMDLength](SIMD[dtype, w]) -> SIMD[dtype, w]],
        *,
        cond: Some[def[w: SIMDLength](SIMD[dtype, w]) -> SIMD[.bool, w]],
    ):
        """Apply the function to the `Span` inplace where the condition is
        `True`.

        Parameters:
            dtype: The DType.

        Args:
            func: The function to evaluate.
            cond: The condition to apply the function.
        """

        comptime widths = (256, 128, 64, 32, 16, 8, 4)
        var ptr = self.unsafe_ptr()
        var length = len(self)
        var processed = 0

        comptime for i in range(len(widths)):
            comptime w = rebind[Int](widths[i])

            comptime if simd_width_of[dtype]() >= w:
                for _ in range((length - processed) // w):
                    var p_curr = ptr.unsafe_offset(processed)
                    var vec = p_curr.unsafe_load[width=w]()
                    p_curr.unsafe_store(cond(vec).select(func(vec), vec))
                    processed += w

        for i in range(length - processed):
            var vec = ptr[unsafe_offset=processed + i]
            if cond(vec):
                ptr.unsafe_offset(processed + i).unsafe_write(func(vec))

    @__allow_legacy_custom_self_type
    def count[
        dtype: DType,
        //,
        F: def[w: SIMDLength](v: SIMD[dtype, w]) -> SIMD[.bool, w],
    ](self: Span[Scalar[dtype], _, address_space=_], func: F) -> Int:
        """Count the amount of times the function returns `True`.

        Parameters:
            dtype: The DType.
            F: The function type to evaluate.

        Args:
            func: The function value to evaluate.

        Returns:
            The amount of times the function returns `True`.
        """

        comptime simdwidth = simd_width_of[dtype]()
        var ptr = self.unsafe_ptr().as_imm()
        var length = len(self)
        var count = 0

        def do_count[width: Int](idx: Int) {mut count, imm ptr, imm func}:
            var mask = func[width](ptr.unsafe_load[width=width](idx))
            count += mask.reduce_bit_count()

        vectorize[simdwidth](length, do_count)
        return count

    @always_inline
    def unsafe_subspan(self, *, offset: Int, length: Int) -> Self:
        """Returns a subspan of the current span.

        Args:
            offset: The starting offset of the subspan (self._data + offset).
            length: The length of the new subspan.

        Returns:
            A new span representing the specified subspan.

        Safety:
            This function does not do bounds checking and assumes the current
            span contains the specified subspan.
        """
        assert (
            0 <= offset and 0 <= length and offset + length <= len(self)
        ), "subspan out of bounds."
        return Self(unsafe_ptr=self._data.unsafe_offset(offset), length=length)

    @__allow_legacy_custom_self_type
    def _binary_search_index[
        dtype: DType,
        //,
    ](
        self: Span[Scalar[dtype], _, address_space=_], needle: Scalar[dtype]
    ) -> Optional[Int]:
        """Finds the index of `needle` with binary search.
        Args:
            needle: The value to binary search for.
        Returns:
            Returns None if `needle` is not present.
        Notes:
            This function will return an unspecified index if `self` is not
            sorted in ascending order.
        """

        var cursor = UInt(0)
        var length = UInt(len(self))
        var value = needle - Scalar[dtype](1)  # just to make it different
        while length > 0:
            var half = length >> 1 if length > 1 else length
            length -= half
            value = self.unsafe_get(cursor + half - 1)
            cursor += UInt(splat(value < needle)) & half

        return Optional(Int(cursor)) if value == needle else None

    @__allow_legacy_custom_self_type
    def binary_search_by[
        func: def(Self.T) thin -> Int,
    ](self: Span[Self.T, _]) -> Optional[Int]:
        """Finds an element using binary search with a custom comparison function.

        The comparison function should return:
        - A negative value if the element is less than the target
        - Zero if the element matches the target
        - A positive value if the element is greater than the target

        Parameters:
            func: A function that takes an element and returns an Int representing
                  the comparison result.

        Returns:
            Returns the index of the matching element if found, None otherwise.

        Notes:
            This function assumes that `self` is sorted according to the ordering
            defined by `func`. If not sorted, the result is unspecified.

        Example:
            ```mojo
            var data: List[String] = ["a", "bb", "ccc"]
            var span = Span(data)

            # Search for "bb"
            def cmp(elem: String) -> Int:
                if elem < "bb":
                    return -1
                elif elem > "bb":
                    return 1
                else:
                    return 0

            var index = span.binary_search_by[cmp]()
            if index:
                print("Found at index: ", index.value())
            else:
                print("Not found")
            ```
        """

        def _cmp(value: Self.T) {var} -> Int:
            return func(value)

        return self.binary_search_by(_cmp)

    @__allow_legacy_custom_self_type
    def binary_search_by[
        FuncType: def(Self.T) -> Int,
    ](self: Span[Self.T, _], func: FuncType) -> Optional[Int]:
        """Finds an element using binary search with a custom comparison function.

        The comparison function should return:
        - A negative value if the element is less than the target
        - Zero if the element matches the target
        - A positive value if the element is greater than the target

        Parameters:
            FuncType: The type of the supplied function.

        Args:
            func: A function that takes an element and returns an Int representing
                    the comparison result.

        Returns:
            Returns the index of the matching element if found, None otherwise.

        Notes:
            This function assumes that `self` is sorted according to the ordering
            defined by `func`. If not sorted, the result is unspecified.

        Example:
            ```mojo
            def main():
                var data: List[String] = ["a", "bb", "ccc"]
                var span = Span(data)

                # Search for "bb"
                def cmp(elem: String) {} -> Int:
                    if elem < "bb":
                        return -1
                    elif elem > "bb":
                        return 1
                    else:
                        return 0

                var index = span.binary_search_by(cmp)
                if index:
                    print("Found at index: ", index.value())
                else:
                    print("Not found")
            ```
        """

        var cursor = 0
        var length = len(self)
        var cmp_result = 1  # Initialize to non-zero value

        while length > 0:
            var half = length >> Int(length > 1)
            length -= half
            var mid_idx = cursor + half - 1
            cmp_result = func(self.unsafe_get(mid_idx))

            # If cmp_result < 0, search in the right half
            cursor += splat(cmp_result < 0) & half

        return Optional(cursor) if cmp_result == 0 else None
