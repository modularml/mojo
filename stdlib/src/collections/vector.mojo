# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Defines several vector-like classes.

You can import these APIs from the `collections` package. For example:

```mojo
from collections.vector import InlinedFixedVector
```
"""

from math import max

from memory.anypointer import AnyPointer
from memory.unsafe import Pointer, Reference, _LITRef, bitcast

from utils.static_tuple import StaticTuple

# ===----------------------------------------------------------------------===#
# _VecIter
# ===----------------------------------------------------------------------===#


@value
struct _VecIter[
    type: AnyRegType,
    vec_type: AnyRegType,
    deref: fn (Pointer[vec_type], Int) -> type,
](Sized):
    """Iterator for any random-access container"""

    var i: Int
    var size: Int
    var vec: Pointer[vec_type]

    fn __next__(inout self) -> type:
        self.i += 1
        return deref(self.vec, self.i - 1)

    fn __len__(self) -> Int:
        return self.size - self.i


# ===----------------------------------------------------------------------===#
# InlinedFixedVector
# ===----------------------------------------------------------------------===#


@always_inline
fn _calculate_fixed_vector_default_size[type: AnyRegType]() -> Int:
    alias prefered_bytecount = 64
    alias sizeof_type = sizeof[type]()

    @parameter
    if sizeof_type >= 256:
        return 0

    alias prefered_inline_bytes = prefered_bytecount - sizeof[
        InlinedFixedVector[type, 0]
    ]()
    alias num_elements = prefered_inline_bytes // sizeof_type
    return num_elements or 1


struct InlinedFixedVector[
    type: AnyRegType, size: Int = _calculate_fixed_vector_default_size[type]()
](Sized):
    """A dynamically-allocated vector with small-vector optimization and a fixed
    maximum capacity.

    The `InlinedFixedVector` does not resize or implement bounds checks. It is
    initialized with both a small-vector size (specified at compile time) and a
    maximum capacity (specified at runtime).

    The first `size` elements are stored in the statically-allocated small
    vector storage. Any remaining elements are stored in dynamically-allocated
    storage.

    When it is deallocated, it frees its memory.

    TODO: It should call its element destructors once we have traits.

    This data structure is useful for applications where the number of required
    elements is not known at compile time, but once known at runtime, is
    guaranteed to be equal to or less than a certain capacity.

    Parameters:
        type: The type of the elements.
        size: The statically-known small-vector size.
    """

    alias static_size: Int = size
    alias static_data_type = StaticTuple[size, type]
    var static_data: Self.static_data_type
    """The underlying static storage, used for small vectors."""
    var dynamic_data: Pointer[type]
    """The underlying dynamic storage, used to grow large vectors."""
    var current_size: Int
    """The number of elements in the vector."""
    var capacity: Int
    """The maximum number of elements that can fit in the vector."""

    @always_inline
    fn __init__(inout self, capacity: Int):
        """Constructs `InlinedFixedVector` with the given capacity.

        The dynamically allocated portion is `capacity - size`.

        Args:
            capacity: The requested maximum capacity of the vector.
        """
        self.static_data = Self.static_data_type()  # Undef initialization
        self.dynamic_data = Pointer[type]()
        if capacity > Self.static_size:
            self.dynamic_data = Pointer[type].alloc(capacity - size)
        self.current_size = 0
        self.capacity = capacity

    # TODO: Probably don't want this to be implicitly no-op copyable when we
    # have ownership.
    @always_inline
    fn __copyinit__(inout self, existing: Self):
        """Creates a shallow copy (doesn't copy the underlying elements).

        Args:
            existing: The `InlinedFixedVector` to copy.
        """
        self.static_data = existing.static_data
        self.dynamic_data = existing.dynamic_data
        self.current_size = existing.current_size
        self.capacity = existing.capacity

    @always_inline
    fn _del_old(self):
        """Destroys the object."""
        if self.capacity > Self.static_size:
            self.dynamic_data.free()

    @always_inline
    fn deepcopy(self) -> Self:
        """Creates a deep copy of this vector.

        Returns:
            The created copy of this vector.
        """
        var res = Self(self.capacity)
        for i in range(len(self)):
            res.append(self[i])
        return res

    @always_inline
    fn append(inout self, value: type):
        """Appends a value to this vector.

        Args:
            value: The value to append.
        """
        debug_assert(
            self.current_size < self.capacity,
            "index must be less than capacity",
        )
        if self.current_size < Self.static_size:
            self.static_data[self.current_size] = value
        else:
            self.dynamic_data[self.current_size - Self.static_size] = value
        self.current_size += 1

    @always_inline
    fn __len__(self) -> Int:
        """Gets the number of elements in the vector.

        Returns:
            The number of elements in the vector.
        """
        return self.current_size

    @always_inline
    fn __getitem__(self, i: Int) -> type:
        """Gets a vector element at the given index.

        Args:
            i: The index of the element.

        Returns:
            The element at the given index.
        """
        debug_assert(
            -self.current_size < i < self.current_size,
            "index must be within bounds",
        )
        var normalized_idx = i
        if i < 0:
            normalized_idx += len(self)

        if normalized_idx < Self.static_size:
            return self.static_data[normalized_idx]

        return self.dynamic_data[normalized_idx - Self.static_size]

    @always_inline
    fn __setitem__(inout self, i: Int, value: type):
        """Sets a vector element at the given index.

        Args:
            i: The index of the element.
            value: The value to assign.
        """
        debug_assert(i < self.current_size, "index must be within bounds")
        if i < Self.static_size:
            self.static_data[i] = value
        else:
            self.dynamic_data[i - Self.static_size] = value

    fn clear(inout self):
        """Clears the elements in the vector."""
        self.current_size = 0

    @staticmethod
    fn _deref_iter_impl(self: Pointer[Self], i: Int) -> type:
        return __get_address_as_lvalue(self.address)[i]

    alias _iterator = _VecIter[type, Self, Self._deref_iter_impl]

    fn __iter__(inout self) -> Self._iterator:
        """Iterate over the vector.

        Returns:
            An iterator to the start of the vector.
        """
        return Self._iterator(
            0, self.current_size, __get_lvalue_as_address(self)
        )


# ===----------------------------------------------------------------------===#
# DynamicVector
# ===----------------------------------------------------------------------===#


@value
struct _DynamicVectorIter[
    T: CollectionElement,
    vector_mutability: __mlir_type.`i1`,
    vector_lifetime: AnyLifetime[vector_mutability].type,
]:
    """Iterator for DynamicVector.

    Parameters:
        T: The type of the elements in the list.
        vector_mutability: Whether the reference to the vector is mutable.
        vector_lifetime: The lifetime of the DynamicVector
    """

    alias vector_type = DynamicVector[T]

    var index: Int
    var src: Reference[Self.vector_type, vector_mutability, vector_lifetime]

    fn __next__(
        inout self,
    ) -> Reference[T, vector_mutability, vector_lifetime]:
        self.index += 1
        return self.src[].__get_ref[vector_mutability, vector_lifetime](
            self.index - 1
        )

    fn __len__(self) -> Int:
        return len(self.src[]) - self.index


struct DynamicVector[T: CollectionElement](CollectionElement, Sized):
    """The `DynamicVector` type is a dynamically-allocated vector.

    It supports pushing and popping from the back resizing the underlying
    storage as needed.  When it is deallocated, it frees its memory.

    Parameters:
        T: The type of the elements.
    """

    var data: AnyPointer[T]
    """The underlying storage for the vector."""
    var size: Int
    """The number of elements in the vector."""
    var capacity: Int
    """The amount of elements that can fit in the vector without resizing it."""

    fn __init__(inout self):
        """Constructs an empty vector."""
        self.data = AnyPointer[T]()
        self.size = 0
        self.capacity = 0

    fn __init__(inout self, *, capacity: Int):
        """Constructs a vector with the given capacity.

        Args:
            capacity: The requested capacity of the vector.
        """
        self.data = AnyPointer[T].alloc(capacity)
        self.size = 0
        self.capacity = capacity

    fn __moveinit__(inout self, owned existing: Self):
        """Move data of an existing vector into a new one.

        Args:
            existing: The existing vector.
        """
        self.data = existing.data
        self.size = existing.size
        self.capacity = existing.capacity

    fn __copyinit__(inout self, existing: Self):
        """Creates a deepcopy of the given vector.

        Args:
            existing: The vector to copy.
        """
        self = Self(capacity=existing.capacity)
        for i in range(len(existing)):
            self.append(existing[i])

    fn __del__(owned self):
        """Destroy all elements in the vector and free its memory."""
        for i in range(self.size):
            _ = (self.data + i).take_value()
        if self.data:
            self.data.free()

    fn __len__(self) -> Int:
        """Gets the number of elements in the vector.

        Returns:
            The number of elements in the vector.
        """
        return self.size

    fn _realloc(inout self, new_capacity: Int):
        var new_data = AnyPointer[T].alloc(new_capacity)

        for i in range(self.size):
            (new_data + i).emplace_value((self.data + i).take_value())

        if self.data:
            self.data.free()
        self.data = new_data
        self.capacity = new_capacity

    fn append(inout self, owned value: T):
        """Appends a value to this vector.

        Args:
            value: The value to append.
        """
        if self.size >= self.capacity:
            self._realloc(max(1, self.capacity * 2))
        (self.data + self.size).emplace_value(value ^)
        self.size += 1

    fn extend(inout self, owned other: DynamicVector[T]):
        """Extends this vector by consuming the elements of `other`.

        Args:
            other: Vector whose elements will be added in order at the end of this vector.
        """

        var final_size = len(self) + len(other)
        var other_original_size = len(other)

        self.reserve(final_size)

        # Defensively mark `other` as logically being empty, as we will be doing
        # consuming moves out of `other`, and so we want to avoid leaving `other`
        # in a partially valid state where some elements have been consumed
        # but are still part of the valid `size` of the vector.
        #
        # That invalid intermediate state of `other` could potentially be
        # visible outside this function if a `__moveinit__()` constructor were
        # to throw (not currently possible AFAIK though) part way through the
        # logic below.
        other.size = 0

        var dest_ptr = self.data + len(self)

        for i in range(other_original_size):
            var src_ptr = other.data + i

            # This (TODO: optimistically) moves an element directly from the
            # `other` vector into this vector using a single `T.__moveinit()__`
            # call, without moving into an intermediate temporary value
            # (avoiding an extra redundant move constructor call).
            src_ptr.move_into(dest_ptr)

            dest_ptr = dest_ptr + 1

        # Update the size now that all new elements have been moved into this
        # vector.
        self.size = final_size

    fn push_back(inout self, owned value: T):
        """Appends a value to this vector.

        Args:
            value: The value to append.
        """
        self.append(value ^)

    fn pop_back(inout self) -> T:
        """Pops a value from the back of this vector.

        Returns:
            The popped value.
        """
        var ret_val = (self.data + (self.size - 1)).take_value()
        self.size -= 1
        if self.size * 4 < self.capacity:
            if self.capacity > 1:
                self._realloc(self.capacity // 2)
        return ret_val ^

    fn reserve(inout self, new_capacity: Int):
        """Reserves the requested capacity.

        If the current capacity is greater or equal, this is a no-op.
        Otherwise, the storage is reallocated and the date is moved.

        Args:
            new_capacity: The new capacity.
        """
        if self.capacity >= new_capacity:
            return
        self._realloc(new_capacity)

    fn resize(inout self, new_size: Int, value: T):
        """Resizes the vector to the given new size.

        If the new size is smaller than the current one, elements at the end
        are discarded. If the new size is larger than the current one, the
        vector is appended with new values elements up to the requested size.

        Args:
            new_size: The new size.
            value: The value to use to populate new elements.
        """
        self.reserve(new_size)
        for i in range(new_size, self.size):
            _ = (self.data + i).take_value()
        for i in range(self.size, new_size):
            (self.data + i).emplace_value(value)
        self.size = new_size

    fn reverse(inout self):
        """Reverses the elements of the vector."""

        self._reverse()

    # This method is private to avoid exposing the non-Pythonic `start` argument.
    fn _reverse(inout self, start: Int = 0):
        """Reverses the elements of the vector at positions after `start`.

        Args:
            start: A non-negative integer indicating the position after which to reverse elements.
        """

        # TODO(polish): Support a negative slice-like start position here that
        #               counts from the end.
        debug_assert(
            start >= 0,
            "DynamicVector reverse start position must be non-negative",
        )

        var earlier_idx = start
        var later_idx = len(self) - 1

        var effective_len = len(self) - start
        var half_len = effective_len // 2

        for _ in range(half_len):
            var earlier_ptr = self.data + earlier_idx
            var later_ptr = self.data + later_idx

            var tmp = earlier_ptr.take_value()
            later_ptr.move_into(earlier_ptr)
            later_ptr.emplace_value(tmp ^)

            earlier_idx += 1
            later_idx -= 1

    fn clear(inout self):
        """Clears the elements in the vector."""
        for i in range(self.size):
            _ = (self.data + i).take_value()
        self.size = 0

    fn steal_data(inout self) -> AnyPointer[T]:
        """Take ownership of the underlying pointer from the vector.

        Returns:
            The underlying data.
        """
        var ptr = self.data
        self.data = AnyPointer[T]()
        self.size = 0
        self.capacity = 0
        return ptr

    fn __setitem__(inout self, i: Int, owned value: T):
        """Sets a vector element at the given index.

        Args:
            i: The index of the element.
            value: The value to assign.
        """
        _ = (self.data + i).take_value()
        (self.data + i).emplace_value(value ^)

    fn __getitem__(self, i: Int) -> T:
        """Gets a copy of the vector element at the given index.

        FIXME(lifetimes): This should return a reference, not a copy!

        Args:
            i: The index of the element.

        Returns:
            A copy of the element at the given index.
        """
        if i < 0:
            return self[len(self) + i]
        return __get_address_as_lvalue((self.data + i).value)

    # TODO(30737): Replace __getitem__ with this as __refitem__, but lots of places use it
    fn __get_ref[
        mutability: __mlir_type.`i1`, self_life: AnyLifetime[mutability].type
    ](
        self: Reference[Self, mutability, self_life].mlir_ref_type,
        i: Int,
    ) -> Reference[T, mutability, self_life]:
        """Gets a reference to the vector element at the given index.

        Args:
            i: The index of the element.

        Returns:
            An immutable reference to the element at the given index.
        """
        # Mutability gets set to the local mutability of this
        # pointer value, ie. because we defined it with `let` it's now an
        # "immutable" reference regardless of the mutability of `self`.
        # This means we can't just use `AnyPointer.__refitem__` here
        # because the mutability won't match.
        var base_ptr = Reference(self)[].data
        return __mlir_op.`lit.ref.from_pointer`[
            _type = Reference[T, mutability, self_life].mlir_ref_type
        ]((base_ptr + i).value)

    fn __iter__[
        mutability: __mlir_type.`i1`, self_life: AnyLifetime[mutability].type
    ](
        self: Reference[Self, mutability, self_life].mlir_ref_type,
    ) -> _DynamicVectorIter[T, mutability, self_life]:
        """Iterate over elements of the vector, returning immutable references.

        Returns:
            An iterator of immutable references to the vector elements.
        """
        return _DynamicVectorIter[T, mutability, self_life](0, Reference(self))
