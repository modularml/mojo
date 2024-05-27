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
"""Defines the `Array` type.

You can import these APIs from the `collections` package. For example:

```mojo
from collections import Array
```
"""


# ===----------------------------------------------------------------------===#
# Array
# ===----------------------------------------------------------------------===#


@value
struct _ArrayIter[
    T: DType,
    capacity: Int,
    forward: Bool = True,
](Sized):
    """Iterator for Array.

    Parameters:
        T: The type of the elements in the Array.
        capacity: The maximum number of elements that the Array can hold.
        forward: The iteration direction. `False` is backwards.
    """

    alias type = Array[T, capacity]

    var index: Int
    var src: Self.type

    fn __iter__(self) -> Self:
        return self

    fn __next__(
        inout self,
    ) -> Scalar[T]:
        @parameter
        if forward:
            self.index += 1
            return self.src[self.index - 1]
        else:
            self.index -= 1
            return self.src[self.index]

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return len(self.src) - self.index
        else:
            return self.index


fn _closest_upper_pow_2(val: Int) -> Int:
    var v = val
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v += 1
    return v


@register_passable("trivial")
struct Array[T: DType = DType.int16, capacity: Int = 256 // T.bitwidth()](
    CollectionElement, Sized, Boolable
):
    """An Array allocated on the stack with a capacity and
    max_capacity known at compile time.

    It is backed by a `SIMD` vector. This struct has the same API
    as a regular `Array`.

    This is typically faster than Python's `Array` as it is stack-allocated
    and does not require any dynamic memory allocation.

    Notes:
        Setting Array items directly doesn't update self.capacity_left,
            methods like append(), extend(), concat() do.

    Parameters:
        T: The type of the elements in the Array.
        capacity: The number of elements that the Array can hold.
            Should be a power of two, otherwise space on the SIMD vector
            is wasted.
    """

    alias _vec_type = SIMD[T, _closest_upper_pow_2(capacity)]
    var vec: Self._vec_type
    """The underlying SIMD vector."""
    alias _scalar_type = Scalar[T]
    var capacity_left: UInt8
    """The current capacity left until expansion."""

    @always_inline
    fn __init__(inout self):
        """This constructor creates an empty Array."""
        self.vec = Self._vec_type()
        self.capacity_left = capacity

    @always_inline
    fn __init__(inout self, *, fill: Self._scalar_type):
        """Constructs a Array by filling it with the
        given value. Sets the capacity_left var to 0.

        Args:
            fill: The value to populate the Array with.
        """
        self.vec = Self._vec_type(fill)
        self.capacity_left = 0

    # TODO: Avoid copying elements in once owned varargs
    # allow transfers.
    fn __init__(inout self, *values: Self._scalar_type):
        """Constructs a Array from the given values.

        Args:
            values: The values to populate the Array with.
        """
        # FIXME: capacity should be statically determined from
        # this constructor
        self = Self()
        for value in values:
            self.append(value)

    fn __init__[cap: Int](inout self, values: SIMD[T, cap]):
        """Constructs a Array from the given values.

        Parameters:
            cap: The capacity of the SIMD vector.

        Args:
            values: The values to populate the Array with.
        """

        @parameter
        if cap == capacity:
            self.vec = values
            self.capacity_left = 0
        else:
            alias size = min(cap, capacity)
            self.capacity_left = capacity - size
            self.vec = Self._vec_type()

            @parameter
            for i in range(size):
                self.vec[i] = values[i]

    fn __init__[cap: Int](inout self, owned existing: Array[T, cap]):
        """Constructs a Array from an existing Array.

        Parameters:
            cap: The number of elements that the Array can hold.

        Args:
            existing: The existing Array.
        """
        self = Self()
        for i in range(capacity):
            self[i] = existing[i]
        self.capacity_left = existing.capacity_left
        # TODO enlargement if necessary to fit existing

    # FIXME
    # fn __init__(
    #     inout self: Self,
    #     *,
    #     unsafe_pointer: UnsafePointer[Self._scalar_type],
    #     size: Int,
    # ):
    #     """Constructs an Array from a pointer and its size.

    #     Args:
    #         unsafe_pointer: The pointer to the data.
    #         size: The number of elements pointed to.
    #     """
    #     var s = min(capacity, size)
    #     self.vec = Self._vec_type()
    #     # FIXME: will this even work? is there no faster way?
    #     for i in range(s):
    #         self.vec[i] = unsafe_pointer[i]
    #     self.capacity_left = capacity - s

    # FIXME
    # fn __init__[
    #     size: Int
    # ](inout self: Self, *, unsafe_pointer: UnsafePointer[Self._scalar_type]):
    #     """Constructs an Array from a pointer and its size.

    #     Parameter:
    #         size: The number of elements pointed to.

    #     Args:
    #         unsafe_pointer: The pointer to the data.
    #     """
    #     alias s = min(capacity, size)
    #     self.vec = Self._vec_type()

    #     @parameter
    #     for i in range(s):
    #         self.vec[i] = unsafe_pointer[i]
    #     self.capacity_left = capacity - s

    # FIXME
    # fn __init__[
    #     size: Int
    # ](inout self: Self, *, unsafe_pointer: UnsafePointer[T.type]):
    #     """Constructs an Array from a pointer and its size.

    #     Parameter:
    #         size: The number of elements pointed to.

    #     Args:
    #         unsafe_pointer: The pointer to the data.
    #     """
    #     alias s = min(capacity, size)
    #     self.vec = Self._vec_type()

    #     @parameter
    #     for i in range(s):
    #         self.vec[i] = rebind[SIMD[T, 0]](unsafe_pointer[i])
    #     self.capacity_left = capacity - s

    # FIXME
    # fn __init__[
    #     capacity: Int
    # ](inout self, owned existing: InlineArray[Self._scalar_type, capacity]):
    #     """Constructs a Array from an existing InlineArray.

    #     Args:
    #         existing: The existing InlineArray.
    #     """
    #     Self.__init__[capacity](self, unsafe_pointer=existing.unsafe_ptr())

    fn __init__[size: Int](inout self, owned existing: List[Self._scalar_type]):
        """Constructs a Array from an existing List.

        Parameters:
            size: The size of the List.

        Args:
            existing: The existing List.
        """
        self.vec = Self._vec_type()
        alias amnt = min(capacity, size)
        self.capacity_left = capacity - amnt

        @parameter
        for i in range(amnt):
            self.unsafe_set(i, existing[i])

    @always_inline
    fn __len__(self) -> Int:
        """Returns the length of the Array."""
        return int(capacity - self.capacity_left)

    @always_inline
    fn append(inout self, owned value: Self._scalar_type):
        """Appends a value to the Array. If full, sets
        the last element to the given value.

        Args:
            value: The value to append.
        """
        if self.capacity_left == 0:
            self.unsafe_set(capacity - 1, value)
            return
        self.unsafe_set(len(self), value)
        self.capacity_left -= 1

    @always_inline
    fn append[cap: Int](inout self, other: Array[T, cap]):
        """Appends the values of another Array up to Self.capacity.

        Parameters:
            cap: The capacity of the other Array.

        Args:
            other: The Array to append.
        """
        for i in range(self.capacity_left):
            self.append(other[i])

    fn __iter__(
        self,
    ) -> _ArrayIter[T, capacity]:
        """Iterate over elements of the Array, returning immutable references.

        Returns:
            An iterator of immutable references to the Array elements.
        """
        return _ArrayIter(0, self)

    fn __reversed__(
        self,
    ) -> _ArrayIter[T, capacity, False]:
        """Iterate backwards over the list, returning immutable references.

        Returns:
            A reversed iterator of immutable references to the list elements.
        """
        return _ArrayIter[forward=False](len(self), self)

    @always_inline
    fn __contains__(self, value: Self._scalar_type) -> Bool:
        """Verify if a given value is present in the Array.

        ```mojo
        var x = Array(1,2,3)
        if 3 in x: print("x contains 3")
        ```

        Args:
            value: The value to find.

        Returns:
            True if the value is contained in the Array, False otherwise.
        """
        return ~(self.vec ^ value).cast[DType.bool]().reduce_or()

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks whether the Array has any elements or not.

        Returns:
            `False` if the Array is empty, `True` if there is at least one element.
        """
        return len(self) > 0

    # fn __moveinit__(inout self, owned existing: Self):
    #     """Move data of an existing Array into a new one.

    #     Args:
    #         existing: The existing Array.
    #     """
    #     self.vec = existing.vec
    #     self.capacity_left = existing.capacity_left

    # fn __copyinit__(inout self, existing: Self):
    #     """Creates a deepcopy of the given Array.

    #     Args:
    #         existing: The Array to copy.
    #     """
    #     self = Self()
    #     for i in range(len(existing)):
    #         self.unsafe_set(i, existing[i])

    @always_inline("nodebug")
    fn concat[
        cap: Int
    ](self, owned other: Array[T, cap]) -> Array[
        T, Self.capacity + other.capacity
    ]:
        """Concatenates self with other and returns the result as a new Array.

        Parameters:
            cap: The capacity of the other Array.

        Args:
            other: Array whose elements will be combined with the elements of self.

        Returns:
            The newly created Array.
        """
        var arr = Array[T, Self.capacity + other.capacity](self)
        arr.extend(other)
        return arr

    fn __str__(self) -> String:
        """Returns a string representation of an `Array`.

        Note that since we can't condition methods on a trait yet,
        the way to call this method is a bit special. Here is an example below:

        ```mojo
        var my_array = Array(1, 2, 3)
        print(str(my_array))
        ```

        When the compiler supports conditional methods, then a simple `str(my_array)` will
        be enough.

        The elements' type must implement the `__str__()` for this to work.

        Returns:
            A string representation of the array.
        """
        # we do a rough estimation of the number of chars that we'll see
        # in the final string, we assume that str(x) will be at least one char.
        var minimum_capacity = (
            2  # '[' and ']'
            + len(self) * 3  # str(x) and ", "
            - 2  # remove the last ", "
        )
        var string_buffer = List[UInt8](capacity=minimum_capacity)
        string_buffer.append(0)  # Null terminator
        var result = String(string_buffer^)
        result += "["
        for i in range(len(self)):
            result += str(self[i])
            if i < len(self) - 1:
                result += ", "
        result += "]"
        return result

    fn __repr__(self) -> String:
        """Returns a string representation of an `Array`.
        Note that since we can't condition methods on a trait yet,
        the way to call this method is a bit special. Here is an example below:

        ```mojo
        var my_array = Array(1, 2, 3)
        print(repr(my_array))
        ```

        When the compiler supports conditional methods, then a simple `repr(my_array)` will
        be enough.

        The elements' type must implement the `__repr__()` for this to work.

        Returns:
            A string representation of the array.
        """
        return str(self)

    @always_inline
    fn insert(inout self, i: Int, owned value: Self._scalar_type):
        """Inserts a value to the list at the given index.
        `a.insert(len(a), value)` is equivalent to `a.append(value)`.

        Args:
            i: The index for the value.
            value: The value to insert.
        """
        debug_assert(
            abs(i) < capacity or i == -1 * capacity, "insert index out of range"
        )
        var norm_idx = min(i, capacity - 1) if i > -1 else max(0, capacity + i)

        var previous = value
        for i in range(norm_idx, capacity):
            var tmp = self.vec[i]
            self.vec[i] = previous
            previous = tmp
        if self.capacity_left > 0:
            self.capacity_left = min(
                self.capacity_left - 1, capacity - (norm_idx + 1)
            )

    @always_inline
    fn extend[cap: Int = capacity](inout self, owned other: Array[T, cap]):
        """Extends this list by consuming the elements of `other` up
        to Self.capacity.

        Parameters:
            cap: The capacity of the other Array.

        Args:
            other: Array whose elements will be added in order at the end of this Array.
        """
        var size_s = len(self)
        var size_o = len(other)
        if size_s + size_o > capacity:
            return
        for i in range(size_o):
            self.unsafe_set(size_s + i, other.unsafe_get(i))
            self.capacity_left -= 1

    fn pop(inout self, i: Int = -1) -> Self._scalar_type:
        """Pops a value from the list at the given index.

        Args:
            i: The index of the value to pop.

        Returns:
            The popped value.
        """
        debug_assert(
            abs(i) < len(self) or i == -1 * len(self), "pop index out of range"
        )
        var norm_idx = min(i, len(self) - 1) if i > -1 else max(
            0, len(self) + i
        )
        self.capacity_left += 1
        var val = self.unsafe_get(norm_idx)
        for i in range(len(self) - norm_idx):
            var offset = norm_idx + i
            if offset == len(self):
                self.unsafe_set(norm_idx, 0)
                break
            self.unsafe_set(offset, self.unsafe_get(offset + 1))
        return val

    fn index(
        self,
        value: Self._scalar_type,
        start: Int = 0,
        stop: Int = -1,
    ) -> OptionalReg[Int]:
        """
        Returns the index of the first occurrence of a value in an Array
        restricted by the range given the start and stop bounds.

        ```mojo
        var my_array = Array(1, 2, 3)
        print(my_array.index(2)) # prints `1`
        ```

        Args:
            value: The value to search for.
            start: The starting index of the search, treated as a slice index
                (defaults to 0).
            stop: The ending index of the search, treated as a slice index
                (defaults to the end of the Array).

        Returns:
            The Optional index of the first occurrence of the value in the Array.
        """

        var size = len(self)
        debug_assert(
            abs(start) < size or start == -1 * size,
            "start index must be within bounds",
        )
        var start_norm = min(start, size - 1) if start > -1 else max(
            0, size + start
        )

        debug_assert(
            abs(stop) < size or stop == -1 * size,
            "stop index must be within bounds",
        )
        var stop_norm: Int = min(stop, size - 1) if stop > -1 else max(
            0, size + stop + 1
        )
        if start == stop_norm:  # FIXME
            return None
        var s = ~(self.vec ^ Self._vec_type(value)).cast[DType.bool]()
        for i in range(start_norm, stop_norm):
            if s[i]:
                return i
        return None

    @always_inline
    fn _adjust_span(self, span: Slice) -> Slice:
        """Adjusts the span based on the Array length."""
        var new_span = span

        if new_span.start < 0:
            new_span.start = max(0, len(self) + new_span.start)

        if not new_span._has_end():
            new_span.end = len(self)
        elif new_span.end < 0:
            new_span.end = max(0, len(self) + new_span.end)

        if span.step < 0:
            var tmp = new_span.end
            new_span.end = new_span.start - 1
            new_span.start = tmp - 1
        return new_span

    @always_inline
    fn __getitem__(self, span: Slice) -> Self:
        """Gets the sequence of elements at the specified positions.

        Args:
            span: A slice that specifies positions of the new Array.

        Returns:
            A new Array containing the Array at the specified span.
        """

        var adjusted_span = self._adjust_span(span)
        var adjusted_span_len = len(adjusted_span)

        if not adjusted_span_len:
            return Self()

        var res = Self()
        for i in range(len(adjusted_span)):  # FIXME using memcpy?
            res[i] = self[adjusted_span[i]]
        return res

    fn __setitem__(inout self, idx: Int, owned value: Self._scalar_type):
        """Sets a Array element at the given index. This will
        not update self.capacity_left.

        Args:
            idx: The index of the element.
            value: The value to assign.
        """
        debug_assert(
            abs(idx) < len(self) or idx == -1 * len(self),
            "index must be within bounds",
        )
        var norm_idx = min(idx, len(self) - 1) if idx > -1 else max(
            0, len(self) + idx
        )
        self.vec[norm_idx] = value

    @always_inline
    fn __getitem__(self, idx: Int) -> Self._scalar_type:
        """Gets a copy of the element at the given index.

        Args:
            idx: The index of the element.

        Returns:
            A copy of the element at the given index.
        """
        debug_assert(
            abs(idx) < len(self) or idx == -1 * len(self),
            "index must be within bounds",
        )
        var norm_idx = min(idx, len(self) - 1) if idx > -1 else max(
            0, len(self) + idx
        )
        return self.vec[norm_idx]

    fn count(self, value: Self._scalar_type) -> Int:
        """Counts the number of occurrences of a value in the Array.
        Note that since we can't condition methods on a trait yet,
        the way to call this method is a bit special. Here is an example below.

        ```mojo
        var my_array = Array(1, 2, 3)
        print(my_array.count(1))
        ```

        Args:
            value: The value to count.

        Returns:
            The number of occurrences of the value in the list.
        """
        return int((value & self.vec).reduce_add())

    # FIXME: is this possible?
    # @always_inline
    # fn unsafe_ptr(self) -> UnsafePointer[Self._vec_type]:
    #     """Retrieves a pointer to the SIMD vector.

    #     Returns:
    #         The UnsafePointer to the SIMD vector.
    #     """
    #     return UnsafePointer.address_of(self.vec)

    @always_inline
    fn unsafe_get(self, idx: Int) -> Self._scalar_type:
        """Get a reference to an element of self without checking index bounds.
        Users should consider using `__getitem__` instead of this method as it is unsafe.
        If an index is out of bounds, this method will not abort, it will be considered
        undefined behavior.

        Note that there is no wraparound for negative indices, caution is advised.
        Using negative indices is considered undefined behavior.
        Never use `my_list.unsafe_get(-1)` to get the last element of the list. It will
        not work. Instead, do `my_list.unsafe_get(len(my_list) - 1)`.

        Args:
            idx: The index of the element to get.

        Returns:
            A reference to the element at the given index.
        """
        debug_assert(abs(idx) > len(self), "index must be within bounds")
        return self.vec[idx]

    @always_inline
    fn unsafe_set(inout self, idx: Int, value: Self._scalar_type):
        """Set a reference to an element of self without checking index bounds.
        Users should consider using `__setitem__` instead of this method as it is unsafe.
        If an index is out of bounds, this method will not abort, it will be considered
        undefined behavior.

        Note that there is no wraparound for negative indices, caution is advised.
        Using negative indices is considered undefined behavior.
        Never use `my_list.unsafe_set(-1)` to set the last element of the list. It will
        not work. Instead, do `my_list.unsafe_set(len(my_list) - 1)`.

        Args:
            idx: The index to set the element.
            value: The element.
        """
        debug_assert(
            abs(idx) > len(self) or idx == -1 * len(self),
            "index must be within bounds",
        )
        self.vec[idx] = value

    @always_inline("nodebug")
    fn sum(self) -> Self._scalar_type:
        """Calculates the sum of all elements.

        Returns:
            The result.
        """
        return self.vec.reduce_add()

    @always_inline("nodebug")
    fn avg(self) -> Self._scalar_type:
        """Calculates the average of all elements.

        Returns:
            The result.
        """
        return self.vec.reduce_add() / len(self)

    @always_inline("nodebug")
    fn min(self) -> Self._scalar_type:
        """Calculates the minimum of all elements.

        Returns:
            The result.
        """
        return self.vec.reduce_min()

    @always_inline("nodebug")
    fn max(self) -> Self._scalar_type:
        """Calculates the maximum of all elements.

        Returns:
            The result.
        """
        return self.vec.reduce_max()

    @always_inline("nodebug")
    fn min[
        cap: Int = capacity
    ](self, other: Array[T, cap]) -> Self._scalar_type:
        """Computes the elementwise minimum between the two vectors.

        Parameters:
            cap: The capacity of the other Array.

        Args:
            other: The other SIMD vector.

        Returns:
            A new SIMD vector where each element at position
                i is min(self[i], other[i]).
        """
        alias delta = Self.capacity - other.capacity

        @parameter
        if delta == 0:
            return self.vec.min(rebind[Self._vec_type](other.vec))
        elif delta > 0:
            var s = Self(other)
            return self.vec.min(s.vec)
        else:
            var s = Array[T, cap](self)
            return other.vec.min(s.vec)

    @always_inline("nodebug")
    fn max[
        cap: Int = capacity
    ](self, other: Array[T, cap]) -> Self._scalar_type:
        """Computes the elementwise maximum between the two Arrays.

        Parameters:
            cap: The capacity of the other Array.

        Args:
            other: The other SIMD vector.

        Returns:
            A new SIMD vector where each element at position
                i is max(self[i], other[i]).
        """
        alias delta = _closest_upper_pow_2(Self.capacity - other.capacity)

        @parameter
        if delta == 0:
            return self.vec.max(rebind[Self._vec_type](other.vec))
        elif delta > 0:
            var s = Self(other)
            return self.vec.max(s.vec)
        else:
            var s = Array[T, cap](self)
            return other.vec.max(s.vec)

    @always_inline("nodebug")
    fn dot(self, other: Self) -> Self._scalar_type:
        """Calculates the dot product between two Arrays.

        Args:
            other: The other Array.

        Returns:
            The result.
        """
        return (self.vec * other.vec).reduce_add()

    @always_inline("nodebug")
    @staticmethod
    fn _mask_vec(inout vec: Self._vec_type):
        @parameter
        for i in range(Self._vec_type.size - capacity):
            vec[capacity + i] = 0

    @always_inline("nodebug")
    @staticmethod
    fn _build_vec(value: Self._scalar_type) -> Self._vec_type:
        var vec = Self._vec_type(value)
        Self._mask_vec(vec)
        return vec

    @always_inline("nodebug")
    fn __mul__(self, other: Self) -> Self._scalar_type:
        """Calculates the dot product between two Arrays.

        Args:
            other: The other Array.

        Returns:
            The result.
        """
        return self.dot(other)

    @always_inline("nodebug")
    fn __mul__(self, value: Self._scalar_type) -> Self:
        """Calculates the elementwise multiplication
        of the given value.

        Args:
            value: The value.

        Returns:
            A new Array with the values.
        """
        return self.vec * Self._build_vec(value)

    @always_inline("nodebug")
    fn __imul__(inout self, owned value: Self._scalar_type):
        """Calculates the elementwise multiplication
        of the given value inplace.

        Args:
            value: The value.
        """
        self.vec *= Self._build_vec(value)

    @always_inline("nodebug")
    fn __truediv__(self, value: Self._scalar_type) -> Self:
        """Calculates the elementwise division
        of the given value.

        Args:
            value: The value.

        Returns:
            A new Array with the values.
        """
        return self.vec / Self._build_vec(value)

    @always_inline("nodebug")
    fn __itruediv__(inout self, owned value: Self._scalar_type):
        """Calculates the elementwise division
        of the given value inplace.

        Args:
            value: The value.
        """
        self.vec /= Self._build_vec(value)

    @always_inline("nodebug")
    fn __floordiv__(self, value: Self._scalar_type) -> Self:
        """Calculates the elementwise floordiv
        of the given value.

        Args:
            value: The value.

        Returns:
            A new Array with the values.
        """
        return self.vec // Self._build_vec(value)

    @always_inline("nodebug")
    fn __ifloordiv__(inout self, owned value: Self._scalar_type):
        """Calculates the elementwise floordiv
        of the given value inplace.

        Args:
            value: The value.
        """
        self.vec //= Self._build_vec(value)

    @always_inline("nodebug")
    fn __mod__(self, value: Self._scalar_type) -> Self:
        """Calculates the elementwise mod
        of the given value.

        Args:
            value: The value.

        Returns:
            A new Array with the values.
        """
        return self.vec % Self._build_vec(value)

    @always_inline("nodebug")
    fn __imod__(inout self, owned value: Self._scalar_type):
        """Calculates the elementwise mod
        of the given value inplace.

        Args:
            value: The value.
        """
        self.vec %= Self._build_vec(value)

    @always_inline("nodebug")
    fn __pow__(self, value: Self._scalar_type) -> Self:
        """Calculates the elementwise pow
        of the given value.

        Args:
            value: The value.

        Returns:
            A new Array with the values.
        """
        return Self(self.vec**value)

    @always_inline("nodebug")
    fn __ipow__(inout self, owned value: Self._scalar_type):
        """Calculates the elementwise pow
        of the given value inplace.

        Args:
            value: The value.
        """
        self.vec **= int(value)  # FIXME will we support float exp?

    @always_inline("nodebug")
    fn __abs__(self) -> Self._scalar_type:
        """Calculates the magnitude of the Array.

        Returns:
            The result.
        """
        return (self.vec**2).reduce_add()

    @always_inline
    fn __add__[
        cap: Int = capacity
    ](self, other: Array[T, cap]) -> Array[
        T, _closest_upper_pow_2(max(capacity, cap))
    ]:
        """Computes the elementwise addition between the two Arrays.

        Parameters:
            cap: The capacity of the other Array.

        Args:
            other: The other SIMD vector.

        Returns:
            A new SIMD vector where each element at position
                i is self[i] + other[i].
        """
        alias size = _closest_upper_pow_2(max(capacity, cap))
        alias new_simd = SIMD[T, size]
        alias new_arr = Array[T, size]
        alias delta = Self.capacity - other.capacity

        # FIXME no idea why but this doesn't currently accept using the alias
        @parameter
        if delta == 0:
            return rebind[SIMD[T, size]](self.vec) + rebind[new_simd](other.vec)
        elif delta > 0:
            var s = new_arr(other)
            return rebind[SIMD[T, size]](self.vec) + rebind[new_simd](s.vec)
        else:
            var s = new_arr(self)
            return rebind[SIMD[T, size]](other.vec) + rebind[new_simd](s.vec)

    @always_inline("nodebug")
    fn __add__(self, owned value: Self._scalar_type) -> Self:
        """Computes the elementwise addition of the value.

        Args:
            value: The value to broadcast.

        Returns:
            A new Array containing the result.
        """
        return Self(self.vec + value)

    @always_inline
    fn __sub__[
        cap: Int = capacity
    ](self, other: Array[T, cap]) -> Array[
        T, _closest_upper_pow_2(max(capacity, cap))
    ]:
        """Computes the elementwise subtraction between the two Arrays.

        Parameters:
            cap: The capacity of the other Array.

        Args:
            other: The other SIMD vector.

        Returns:
            A new SIMD vector where each element at position
                i is self[i] - other[i].
        """
        alias size = _closest_upper_pow_2(max(capacity, cap))
        alias new_simd = SIMD[T, size]
        alias new_arr = Array[T, size]
        alias delta = Self.capacity - other.capacity

        # FIXME no idea why but this currently doesn't accept using the alias
        @parameter
        if delta == 0:
            return rebind[SIMD[T, size]](self.vec) - rebind[new_simd](other.vec)
        elif delta > 0:
            var s = new_arr(other)
            return rebind[SIMD[T, size]](self.vec) - rebind[new_simd](s.vec)
        else:
            var s = new_arr(self)
            return rebind[SIMD[T, size]](other.vec) - rebind[new_simd](s.vec)

    @always_inline("nodebug")
    fn __sub__(self, owned value: Self._scalar_type) -> Self:
        """Computes the elementwise subtraction of the value.

        Args:
            value: The value to broadcast.

        Returns:
            A new Array containing the result.
        """
        return Self(self.vec - value)

    @always_inline("nodebug")
    fn __iadd__(inout self, owned other: Self):
        """Computes the elementwise addition between the two Arrays
        inplace.

        Args:
            other: The other Array.
        """
        self.vec += other.vec

    @always_inline("nodebug")
    fn __iadd__(inout self, owned value: Self._scalar_type):
        """Computes the elementwise addition of the value.

        Args:
            value: The value to broadcast.
        """
        self.vec += Self._build_vec(value)

    @always_inline("nodebug")
    fn __isub__(inout self, owned other: Self):
        """Computes the elementwise subtraction between the two Arrays
        inplace.

        Args:
            other: The other Array.
        """
        self.vec -= other.vec

    @always_inline("nodebug")
    fn __isub__(inout self, owned value: Self._scalar_type):
        """Computes the elementwise subtraction of the value.

        Args:
            value: The value to broadcast.
        """
        self.vec -= Self._build_vec(value)

    fn clear(inout self):
        """Zeroes the Array."""
        self.vec = self.vec.splat(0)
        self.capacity_left = capacity

    @always_inline("nodebug")
    fn cos(self, other: Self) -> Float64:
        """Calculates the cosine of the angle between two Arrays.

        Args:
            other: The other Array.

        Returns:
            The result.
        """
        return (self * other) / (abs(self.vec) * abs(other.vec))

    # TODO: need math funcs
    # @always_inline("nodebug")
    # fn theta(self, other: Self) -> Float64:
    #     """Calculates the angle between two Arrays.

    #     Args:
    #         other: The other Array.

    #     Returns:
    #         The result.
    #     """
    #     return acos(self.cos(other))

    # TODO: need math funcs
    # @always_inline("nodebug")
    # fn cross(self, other: Self) -> Self:
    #     """Calculates the cross product between two Arrays.
    #
    #     Returns:
    #         The result.
    #     """
    #     # TODO using matmul for big vectors
    #     # TODO quaternions/fma for 3d vectors
    #     var magns = abs(self.vec) * abs(other.vec)
    #     return magns * sin((self * other) / magns)

    # TODO
    # fn sqrt(self):
    #     ...
