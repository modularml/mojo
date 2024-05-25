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

from bit import countr_zero

# ===----------------------------------------------------------------------===#
# Array
# ===----------------------------------------------------------------------===#


@value
struct _ArrayIter[
    T: DType,
    current_capacity: Int,
    capacity_jump: Int,
    max_stack_size: Int,
    forward: Bool = True,
](Sized):
    """Iterator for Array.

    Parameters:
        T: The type of the elements in the Array.
        current_capacity: The maximum number of elements that the Array can hold.
        capacity_jump: The amount of items to expand in each stack enlargment.
        max_stack_size: The maximum size in the stack.
        forward: The iteration direction. `False` is backwards.
    """

    alias type = Array[T, current_capacity, capacity_jump, max_stack_size]

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


struct Array[
    T: DType = DType.index,
    current_capacity: Int = T.bitwidth() // 8,
    capacity_jump: Int = T.bitwidth() // 8,
    max_stack_size: Int = 32 * (T.bitwidth() // 8),
](CollectionElement, Sized, Boolable):
    """A Array allocated on the stack with a current_capacity and
    max_stack_size known at compile time.

    It is backed by a `SIMD` vector. This struct has the same API
    as a regular `Array`.

    This is typically faster than Python's `Array` as it is stack-allocated
    and does not require any dynamic memory allocation.

    Parameters:
        T: The type of the elements in the Array.
        current_capacity: The number of elements that the Array can hold.
        capacity_jump: The amount of capacity to expand in each stack enlargment.
        max_stack_size: The maximum size in the stack.
    """

    alias _vec_type = SIMD[T, current_capacity]
    var vec: Self._vec_type
    alias _scalar_type = Scalar[T]
    var stack_left: UInt8
    """The capacity left in the Stack."""

    @always_inline
    fn __init__(inout self):
        """This constructor creates an empty Array."""
        self.vec = Self._vec_type()
        self.stack_left = current_capacity

    @always_inline
    fn __init__(inout self, *, fill: Self._scalar_type):
        """Constructs a Array by filling it with the
        given value. Sets the stack_left var to 0.

        Args:
            fill: The value to populate the Array with.
        """
        self.vec = Self._vec_type(fill)
        self.stack_left = 0

    # TODO: Avoid copying elements in once owned varargs
    # allow transfers.
    fn __init__(inout self, *values: Self._scalar_type):
        """Constructs a Array from the given values.

        Args:
            values: The values to populate the Array with.
        """
        self = Self()
        var delta = current_capacity - len(values)
        if delta > -1:
            self.stack_left = delta
        else:
            self.stack_left = 0
        for value in values:
            self.append(value)

    fn __init__[cap: Int](inout self, values: SIMD[T, cap]):
        """Constructs a Array from the given values.

        Args:
            values: The values to populate the Array with.
        """
        self = Self()
        var delta = max(0, current_capacity - len(values))
        for value in range(current_capacity - delta):  # FIXME
            self.append(value)

    fn __init__[
        cap: Int, cap_j: Int, max_stack: Int
    ](inout self, owned existing: Array[T, cap, cap_j, max_stack]):
        """Constructs a Array from an existing Array.

        Parameters:
            cap: The number of elements that the Array can hold.
            cap_j: The amount of capacity to expand in each stack enlargment.
            max_stack: The maximum size in the stack.

        Args:
            existing: The existing Array.
        """
        self = Self()
        for i in range(current_capacity):
            self[i] = existing[i]
        # TODO enlargement if necessary to fit existing

    # fn __init__(inout self, owned existing: List[T.type]):
    #     """Constructs a Array from an existing List.

    #     Args:
    #         existing: The existing Array.
    #     """
    #     var size = min(current_capacity, existing.size)
    #     # TODO: need a SIMD constructor from DTypePointer/UnsafePointer
    #     self.vec = Self._vec_type(existing.unsafe_ptr(), size)
    #     self.stack_left =  current_capacity - size

    fn __init__(
        inout self: Self,
        *,
        unsafe_pointer: UnsafePointer[T.type],
        size: Int,
    ):
        """Constructs an Array from a pointer and its size.

        Args:
            unsafe_pointer: The pointer to the data.
            size: The number of elements in the Array.
        """
        var s = min(current_capacity, size)
        self.vec = Self._vec_type()
        for i in range(s):  # FIXME: will this even work?
            self.vec[i] = rebind[SIMD[T, 0]](unsafe_pointer[i])
        self.stack_left = current_capacity - s

    @always_inline
    fn __len__(self) -> Int:
        """Returns the length of the Array."""
        return int(current_capacity - self.stack_left)

    @always_inline
    fn append(inout self, owned value: Self._scalar_type):
        """Appends a value to the Array.

        Args:
            value: The value to append.
        """
        self[len(self) - 1] = value
        self.stack_left = max(0, self.stack_left - 1)

    @always_inline
    fn append[cap: Int](inout self, owned other: Array[T, cap]):
        """Appends another Array to the Array. Can only append up to
        current_capacity.

        Args:
            other: The Array to append.
        """
        var r = min(Self.current_capacity, other.current_capacity)
        for i in range(r):
            self[len(self) - 1 + i] = other[i]
        self.stack_left = max(0, self.stack_left - r)

    fn __iter__(
        self,
    ) -> _ArrayIter[T, current_capacity, capacity_jump, max_stack_size]:
        """Iterate over elements of the Array, returning immutable references.

        Returns:
            An iterator of immutable references to the Array elements.
        """
        return _ArrayIter(0, self)

    fn __reversed__(
        self,
    ) -> _ArrayIter[T, current_capacity, capacity_jump, max_stack_size, False]:
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
        return (value & self.vec).reduce_or()

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks whether the Array has any elements or not.

        Returns:
            `False` if the Array is empty, `True` if there is at least one element.
        """
        return len(self) > 0

    fn __moveinit__(inout self, owned existing: Self):
        """Move data of an existing Array into a new one.

        Args:
            existing: The existing Array.
        """
        self.vec = existing.vec
        self.stack_left = existing.stack_left

    fn __copyinit__(inout self, existing: Self):
        """Creates a deepcopy of the given Array.

        Args:
            existing: The Array to copy.
        """
        self = Self()
        for i in range(len(existing)):
            self.unsafe_set(i, existing[i])

    fn __setitem__(inout self, idx: Int, owned value: Self._scalar_type):
        """Sets a Array element at the given index.

        Args:
            idx: The index of the element.
            value: The value to assign.
        """
        debug_assert(abs(idx) > len(self), "index must be within bounds")
        var norm_idx = idx if idx > 0 else min(0, len(self) + idx)
        self.vec[norm_idx] = value

    @always_inline("nodebug")
    fn concat(
        self, owned other: Self
    ) -> Array[
        T,
        min(
            Self.max_stack_size, Self.current_capacity + other.current_capacity
        ),
    ]:
        """Concatenates self with other and returns the result as a new Array.

        Args:
            other: Array whose elements will be combined with the elements of self.

        Returns:
            The newly created Array.
        """
        alias size = min(
            Self.current_capacity + other.current_capacity, max_stack_size
        )
        var arr = Array[T, size, capacity_jump, max_stack_size]()
        arr.extend(self.vec)
        arr.extend(other.vec)
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
        debug_assert(abs(i) > len(self), "insert index out of range")

        var norm_idx = i if i > 0 else min(0, len(self) + i)

        var previous = value
        for i in range(norm_idx, len(self)):
            var tmp = self.vec[i]
            self.vec[i] = previous
            previous = tmp
        self.append(previous)

    @always_inline
    fn extend[
        cap: Int = current_capacity
    ](inout self, owned other: Array[T, cap]):
        """Extends this list by consuming the elements of `other`.

        Args:
            other: Array whose elements will be added in order at the end of this Array.
        """
        alias cap_sum = current_capacity + other.current_capacity
        if self.stack_left - len(other) < current_capacity:
            self.append(other)
            return
        elif cap_sum < max_stack_size:
            alias new_arr = Array[T, cap_sum, capacity_jump, max_stack_size]
            var s = rebind[SIMD[T, cap_sum]](self.vec)
            self = new_arr(s)
            self.append(other)
            return
        alias new_arr = Array[T, max_stack_size, capacity_jump, max_stack_size]
        var s = rebind[SIMD[T, max_stack_size]](self.vec)
        self = new_arr(s)
        self.append(other)

    @always_inline
    fn pop(inout self, i: Int = -1) -> Self._scalar_type:
        """Pops a value from the list at the given index.

        Args:
            i: The index of the value to pop.

        Returns:
            The popped value.
        """
        debug_assert(abs(i) > len(self), "pop index out of range")
        var norm_idx = i if i > 0 else len(self) + i
        self.stack_left += 1
        return self.vec[norm_idx]

    # TODO: Remove explicit self type when issue 1876 is resolved.
    fn index(
        self,
        value: Self._scalar_type,
        start: Int = 0,
        stop: Optional[Int] = None,
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
                (defaults to None, which means the end of the Array).

        Returns:
            The Optional index of the first occurrence of the value in the Array.
        """

        var size = len(self)
        debug_assert(abs(start) > size, "start index must be within bounds")
        var start_norm = start if start > 0 else min(0, size + start)

        var stop_norm: Int = stop.value()[] if stop else size
        debug_assert(abs(stop_norm) > size, "stop index must be within bounds")
        if stop_norm < 0:
            stop_norm = min(0, size + stop_norm)

        start_norm = max(start_norm, min(0, size))
        stop_norm = max(stop_norm, min(0, size))
        # FIXME ? maybe building a range 0..stop_norm and multiplying it
        # to the bitwise & comparison's result and doing reduce_min()?
        var s = (self[start_norm:stop_norm].vec & value).cast[DType.bool]()
        for i in range(stop_norm - start_norm):
            if s[i]:
                return i + start_norm
        return None

    fn steal_data(inout self) -> Self._vec_type:
        """Take ownership of the underlying pointer from the Array.

        Returns:
            The underlying data.
        """
        # TODO: is this even possible?
        return self.vec

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
        return res^

    @always_inline
    fn __getitem__(self, idx: Int) -> Self._scalar_type:
        """Gets a copy of the element at the given index.

        Args:
            idx: The index of the element.

        Returns:
            A copy of the element at the given index.
        """
        return self.vec[idx]

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

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[Self._vec_type]:
        """Retrieves a pointer to the SIMD vector.

        Returns:
            The UnsafePointer to the SIMD vector.
        """
        return UnsafePointer.address_of(self.vec)

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
        debug_assert(abs(idx) > len(self), "index must be within bounds")
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
        cap: Int = current_capacity
    ](self, other: Array[T, cap]) -> Self._scalar_type:
        """Computes the elementwise minimum between the two vectors.

        Args:
            other: The other SIMD vector.

        Returns:
            A new SIMD vector where each element at position
                i is min(self[i], other[i]).
        """
        alias delta = Self.current_capacity - other.current_capacity

        @parameter
        if delta == 0:
            return self.vec.min(rebind[Self._vec_type](other.vec))
        elif delta > 0:
            var s = Self()
            s.extend(other.vec)
            return self.vec.min(s.vec)
        else:
            var s = Array[T, cap]()
            s.extend(self.vec)
            return other.vec.min(s.vec)

    @always_inline("nodebug")
    fn max[
        cap: Int = current_capacity
    ](self, other: Array[T, cap]) -> Self._scalar_type:
        """Computes the elementwise maximum between the two Arrays.

        Args:
            other: The other SIMD vector.

        Returns:
            A new SIMD vector where each element at position
                i is max(self[i], other[i]).
        """
        alias delta = Self.current_capacity - other.current_capacity

        @parameter
        if delta == 0:
            return self.vec.max(rebind[Self._vec_type](other.vec))
        elif delta > 0:
            var s = Self()
            s.extend(other.vec)
            return self.vec.max(s.vec)
        else:
            var s = Array[T, cap]()
            s.extend(self.vec)
            return other.vec.max(s.vec)

    @always_inline("nodebug")
    fn dot(self, other: Self) -> Self._scalar_type:
        """Calculates the dot product between two Arrays.

        Returns:
            The result.
        """
        return (self.vec * other.vec).reduce_add()

    @always_inline("nodebug")
    fn __mul__(self, other: Self) -> Self._scalar_type:
        """Calculates the dot product between two Arrays.

        Returns:
            The result.
        """
        return self.dot(other)

    @always_inline("nodebug")
    fn __abs__(self) -> Self._scalar_type:
        """Calculates the magnitude of the Array.

        Returns:
            The result.
        """
        return abs(self.vec)

    @always_inline
    fn __add__[
        cap: Int = current_capacity
    ](self, other: Array[T, cap]) -> Array[T, max(current_capacity, cap)]:
        """Computes the elementwise addition between the two Arrays.

        Args:
            other: The other SIMD vector.

        Returns:
            A new SIMD vector where each element at position
                i is self[i] + other[i].
        """
        alias new_simd = SIMD[T, max(current_capacity, cap)]
        alias new_arr = Array[T, max(current_capacity, cap)]
        alias delta = Self.current_capacity - other.current_capacity

        @parameter
        if delta == 0:
            # FIXME no idea why but this doesn't currently accept using the alias
            return rebind[SIMD[T, max(current_capacity, cap)]](
                self.vec
            ) + rebind[new_simd](other.vec)
        elif delta > 0:
            var s = new_arr()
            s.extend(other.vec)
            return rebind[new_simd](self.vec) + s.vec
        else:
            var s = new_arr()
            s.extend(self.vec)
            return rebind[new_simd](other.vec) + s.vec

    @always_inline
    fn __sub__[
        cap: Int = current_capacity
    ](self, other: Array[T, cap]) -> Array[T, max(current_capacity, cap)]:
        """Computes the elementwise subtraction between the two Arrays.

        Args:
            other: The other SIMD vector.

        Returns:
            A new SIMD vector where each element at position
                i is self[i] - other[i].
        """
        alias new_simd = SIMD[T, max(current_capacity, cap)]
        alias new_arr = Array[T, max(current_capacity, cap)]
        alias delta = Self.current_capacity - other.current_capacity

        @parameter
        if delta == 0:
            # FIXME no idea why but this currently doesn't accept using the alias
            return rebind[SIMD[T, max(current_capacity, cap)]](
                self.vec
            ) - rebind[new_simd](other.vec)
        elif delta > 0:
            var s = new_arr()
            s.extend(other.vec)
            return rebind[new_simd](self.vec) - s.vec
        else:
            var s = new_arr()
            s.extend(self.vec)
            return rebind[new_simd](other.vec) - s.vec

    @always_inline("nodebug")
    fn __iadd__[
        cap: Int = current_capacity
    ](inout self, owned other: Array[T, cap]):
        """Computes the elementwise addition between the two Arrays
        inplace.

        Args:
            other: The other Array.
        """
        self = self + other

    @always_inline("nodebug")
    fn __isub__[
        cap: Int = current_capacity
    ](inout self, owned other: Array[T, cap]):
        """Computes the elementwise subtraction between the two Arrays
        inplace.

        Args:
            other: The other Array.
        """
        self = self - other

    fn clear(inout self):
        """Zeroes the Array."""
        self.vec = self.vec.splat(0)
        self.stack_left = current_capacity

    # @always_inline
    # fn theta(self, other: Self) -> Float64:
    #     """Calculates the angle between two Arrays.

    #     Returns:
    #         The result.
    #     """ # TODO: need math funcs
    #     return acos((self * other) / (abs(self.vec) * abs(other.vec)))

    # fn cross(self, other: Self) -> Self:
    #     """Calculates the cross product between two Arrays.

    #     Returns:
    #         The result.
    #     """
    #     # TODO using matmul for big vectors
    #     # TODO quaternions/fma for 3d vectors
    #     var magns = abs(self.vec) * abs(other.vec)
    #     return magns * sin((self * other) / magns) # TODO: need math funcs
