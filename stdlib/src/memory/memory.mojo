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
"""Defines functions for memory manipulations.

You can import these APIs from the `memory` package. For example:

```mojo
from memory import memcmp
```
"""


from sys import llvm_intrinsic
from sys.info import sizeof, triple_is_nvidia_cuda
from builtin.dtype import _integral_type_of

from .unsafe import AddressSpace, DTypePointer, Pointer, _GPUAddressSpace

# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@always_inline
fn _align_down(value: Int, alignment: Int) -> Int:
    return value._positive_div(alignment) * alignment


# ===----------------------------------------------------------------------===#
# memcmp
# ===----------------------------------------------------------------------===#


@always_inline
fn _memcmp_impl(s1: DTypePointer, s2: __type_of(s1), count: Int) -> Int:
    constrained[s1.type.is_integral(), "the input dtype must be integral"]()
    alias simd_width = simdwidthof[s1.type]()
    if count < simd_width:
        for i in range(count):
            var s1i = s1[i]
            var s2i = s2[i]
            var diff = s1i - s2i
            if diff:
                return 1 if diff > 0 else -1
        return 0

    var vector_end_simd = _align_down(count, simd_width)
    for i in range(0, vector_end_simd, simd_width):
        var s1i = s1.load[width=simd_width](i)
        var s2i = s2.load[width=simd_width](i)
        var diff = s1i - s2i
        if (diff != 0).reduce_or():
            for j in range(simd_width):
                return 1 if diff[j] > 0 else -1

    var last = count - simd_width
    if last <= 0:
        return 0

    var s1i = s1.load[width=simd_width](last)
    var s2i = s2.load[width=simd_width](last)
    var diff = s1i - s2i
    if (diff != 0).reduce_or():
        for j in range(simd_width):
            return 1 if diff[j] > 0 else -1
    return 0


@always_inline
fn memcmp(s1: DTypePointer, s2: __type_of(s1), count: Int) -> Int:
    """Compares two buffers. Both strings are assumed to be of the same length.

    Args:
        s1: The first buffer address.
        s2: The second buffer address.
        count: The number of elements in the buffers.

    Returns:
        Returns 0 if the bytes buffers are identical, 1 if s1 > s2, and -1 if
        s1 < s2. The comparison is performed by the first different byte in the
        buffer.
    """

    @parameter
    if s1.type.is_floating_point():
        alias integral_type = _integral_type_of[s1.type]()
        return _memcmp_impl(
            s1.bitcast[integral_type](), s2.bitcast[integral_type](), count
        )

    var byte_count = count * sizeof[s1.type]()

    @parameter
    if sizeof[s1.type]() >= sizeof[DType.int32]():
        return _memcmp_impl(
            s1.bitcast[DType.int32](),
            s2.bitcast[DType.int32](),
            byte_count // sizeof[DType.int32](),
        )

    return _memcmp_impl(s1, s2, count)


@always_inline
fn memcmp[
    type: AnyRegType, address_space: AddressSpace
](
    s1: Pointer[type, address_space],
    s2: Pointer[type, address_space],
    count: Int,
) -> Int:
    """Compares two buffers. Both strings are assumed to be of the same length.

    Parameters:
        type: The element type.
        address_space: The address space of the pointer.

    Args:
        s1: The first buffer address.
        s2: The second buffer address.
        count: The number of elements in the buffers.

    Returns:
        Returns 0 if the bytes strings are identical, 1 if s1 > s2, and -1 if
        s1 < s2. The comparison is performed by the first different byte in the
        byte strings.
    """
    var byte_count = count * sizeof[type]()

    @parameter
    if sizeof[type]() >= sizeof[DType.int32]():
        var ds1 = DTypePointer[DType.int32, address_space](s1.bitcast[Int32]())
        var ds2 = DTypePointer[DType.int32, address_space](s2.bitcast[Int32]())
        return _memcmp_impl(ds1, ds2, byte_count // sizeof[DType.int32]())

    var ds1 = DTypePointer[DType.int8, address_space](s1.bitcast[Int8]())
    var ds2 = DTypePointer[DType.int8, address_space](s2.bitcast[Int8]())
    return _memcmp_impl(ds1, ds2, byte_count)


# ===----------------------------------------------------------------------===#
# memcpy
# ===----------------------------------------------------------------------===#


@always_inline
fn memcpy[count: Int](dest: Pointer, src: __type_of(dest)):
    """Copies a memory area.

    Parameters:
        count: The number of elements to copy (not bytes!).

    Args:
        dest: The destination pointer.
        src: The source pointer.
    """
    alias n = count * sizeof[dest.type]()

    var dest_data = dest.bitcast[Int8]()
    var src_data = src.bitcast[Int8]()

    @parameter
    if n < 5:

        @unroll
        for i in range(n):
            dest_data[i] = src_data[i]
        return

    @parameter
    if n <= 16:

        @parameter
        if n >= 8:
            var ui64_size = sizeof[Int64]()
            dest_data.bitcast[Int64]().store(src_data.bitcast[Int64]()[0])
            dest_data.offset(n - ui64_size).bitcast[Int64]().store(
                src_data.offset(n - ui64_size).bitcast[Int64]()[0]
            )
            return

        var ui32_size = sizeof[Int32]()
        dest_data.bitcast[Int32]().store(src_data.bitcast[Int32]()[0])
        dest_data.offset(n - ui32_size).bitcast[Int32]().store(
            src_data.offset(n - ui32_size).bitcast[Int32]()[0]
        )
        return

    var dest_dtype_ptr = DTypePointer[DType.int8, dest.address_space](dest_data)
    var src_dtype_ptr = DTypePointer[DType.int8, src.address_space](src_data)

    # Copy in 32-byte chunks.
    alias chunk_size = 32
    alias vector_end = _align_down(n, chunk_size)
    for i in range(0, vector_end, chunk_size):
        dest_dtype_ptr.store(i, src_dtype_ptr.load[width=chunk_size](i))
    for i in range(vector_end, n):
        dest_dtype_ptr.store(i, src_dtype_ptr.load[width=1](i))


@always_inline
fn memcpy[count: Int](dest: DTypePointer, src: __type_of(dest)):
    """Copies a memory area.

    Parameters:
        count: The number of elements to copy (not bytes!).

    Args:
        dest: The destination pointer.
        src: The source pointer.
    """
    memcpy[count](dest.address, src.address)


@always_inline
fn memcpy(dest: Pointer, src: __type_of(dest), count: Int):
    """Copies a memory area.

    Args:
        dest: The destination pointer.
        src: The source pointer.
        count: The number of elements to copy.
    """
    var n = count * sizeof[dest.type]()

    var dest_data = dest.bitcast[Int8]()
    var src_data = src.bitcast[Int8]()

    if n < 5:
        if n == 0:
            return
        dest_data[0] = src_data[0]
        dest_data[n - 1] = src_data[n - 1]
        if n <= 2:
            return
        dest_data[1] = src_data[1]
        dest_data[n - 2] = src_data[n - 2]
        return

    if n <= 16:
        if n >= 8:
            var ui64_size = sizeof[Int64]()
            dest_data.bitcast[Int64]().store(src_data.bitcast[Int64]()[0])
            dest_data.offset(n - ui64_size).bitcast[Int64]().store(
                src_data.offset(n - ui64_size).bitcast[Int64]()[0]
            )
            return
        var ui32_size = sizeof[Int32]()
        dest_data.bitcast[Int32]().store(src_data.bitcast[Int32]()[0])
        dest_data.offset(n - ui32_size).bitcast[Int32]().store(
            src_data.offset(n - ui32_size).bitcast[Int32]()[0]
        )
        return

    # TODO (#10566): This branch appears to cause a 12% regression in BERT by
    # slowing down broadcast ops
    # if n <= 32:
    #    alias simd_16xui8_size = 16 * sizeof[Int8]()
    #    dest_data.store[width=16](src_data.load[width=16]())
    #    # note that some of these bytes may have already been written by the
    #    # previous simd_store
    #    dest_data.store[width=16](
    #        n - simd_16xui8_size, src_data.load[width=16](n - simd_16xui8_size)
    #    )
    #    return

    var dest_dtype_ptr = DTypePointer[DType.int8, dest.address_space](dest_data)
    var src_dtype_ptr = DTypePointer[DType.int8, src.address_space](src_data)

    # Copy in 32-byte chunks.
    alias chunk_size = 32
    var vector_end = _align_down(n, chunk_size)
    for i in range(0, vector_end, chunk_size):
        dest_dtype_ptr.store(i, src_dtype_ptr.load[width=chunk_size](i))
    for i in range(vector_end, n):
        dest_dtype_ptr.store(i, src_dtype_ptr.load[width=1](i))


@always_inline
fn memcpy(dest: DTypePointer, src: __type_of(dest), count: Int):
    """Copies a memory area.

    Args:
        dest: The destination pointer.
        src: The source pointer.
        count: The number of elements to copy (not bytes!).
    """
    memcpy(dest.address, src.address, count)


# ===----------------------------------------------------------------------===#
# memset
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _memset_llvm[
    address_space: AddressSpace
](ptr: Pointer[UInt8, address_space], value: UInt8, count: Int):
    llvm_intrinsic["llvm.memset", NoneType](
        ptr.address, value, count.value, False
    )


@always_inline
fn memset[
    type: DType, address_space: AddressSpace
](ptr: DTypePointer[type, address_space], value: UInt8, count: Int):
    """Fills memory with the given value.

    Parameters:
        type: The element dtype.
        address_space: The address space of the pointer.

    Args:
        ptr: Pointer to the beginning of the memory block to fill.
        value: The value to fill with.
        count: Number of elements to fill (in elements, not bytes).
    """
    memset(ptr.address, value, count)


@always_inline
fn memset[
    type: AnyRegType, address_space: AddressSpace
](ptr: Pointer[type, address_space], value: UInt8, count: Int):
    """Fills memory with the given value.

    Parameters:
        type: The element dtype.
        address_space: The address space of the pointer.

    Args:
        ptr: Pointer to the beginning of the memory block to fill.
        value: The value to fill with.
        count: Number of elements to fill (in elements, not bytes).
    """
    _memset_llvm(ptr.bitcast[UInt8](), value, count * sizeof[type]())


# ===----------------------------------------------------------------------===#
# memset_zero
# ===----------------------------------------------------------------------===#


@always_inline
fn memset_zero[
    type: DType, address_space: AddressSpace
](ptr: DTypePointer[type, address_space], count: Int):
    """Fills memory with zeros.

    Parameters:
        type: The element dtype.
        address_space: The address space of the pointer.

    Args:
        ptr: Pointer to the beginning of the memory block to fill.
        count: Number of elements to set (in elements, not bytes).
    """
    memset(ptr, 0, count)


@always_inline
fn memset_zero[
    type: AnyRegType, address_space: AddressSpace
](ptr: Pointer[type, address_space], count: Int):
    """Fills memory with zeros.

    Parameters:
        type: The element type.
        address_space: The address space of the pointer.

    Args:
        ptr: Pointer to the beginning of the memory block to fill.
        count: Number of elements to fill (in elements, not bytes).
    """
    memset(ptr, 0, count)


# ===----------------------------------------------------------------------===#
# stack_allocation
# ===----------------------------------------------------------------------===#


@always_inline
fn stack_allocation[
    count: Int,
    type: DType,
    /,
    alignment: Int = 1,
    address_space: AddressSpace = AddressSpace.GENERIC,
]() -> DTypePointer[type, address_space]:
    """Allocates data buffer space on the stack given a data type and number of
    elements.

    Parameters:
        count: Number of elements to allocate memory for.
        type: The data type of each element.
        alignment: Address alignment of the allocated data.
        address_space: The address space of the pointer.

    Returns:
        A data pointer of the given type pointing to the allocated space.
    """

    return stack_allocation[
        count, Scalar[type], alignment=alignment, address_space=address_space
    ]()


@always_inline
fn stack_allocation[
    count: Int,
    type: AnyRegType,
    /,
    alignment: Int = 1,
    address_space: AddressSpace = AddressSpace.GENERIC,
]() -> Pointer[type, address_space]:
    """Allocates data buffer space on the stack given a data type and number of
    elements.

    Parameters:
        count: Number of elements to allocate memory for.
        type: The data type of each element.
        alignment: Address alignment of the allocated data.
        address_space: The address space of the pointer.

    Returns:
        A data pointer of the given type pointing to the allocated space.
    """

    @parameter
    if triple_is_nvidia_cuda() and address_space == _GPUAddressSpace.SHARED:
        return __mlir_op.`pop.global_alloc`[
            count = count.value,
            _type = Pointer[type, address_space].pointer_type,
            alignment = alignment.value,
            address_space = address_space._value.value,
        ]()
    else:
        return __mlir_op.`pop.stack_allocation`[
            count = count.value,
            _type = Pointer[type, address_space].pointer_type,
            alignment = alignment.value,
            address_space = address_space._value.value,
        ]()


# ===----------------------------------------------------------------------===#
# malloc
# ===----------------------------------------------------------------------===#


@always_inline
fn _malloc[
    type: AnyRegType,
    /,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
](size: Int, /, *, alignment: Int = -1) -> Pointer[type, address_space]:
    @parameter
    if triple_is_nvidia_cuda():
        return external_call["malloc", Pointer[NoneType, address_space]](
            size
        ).bitcast[type]()
    else:
        return __mlir_op.`pop.aligned_alloc`[
            _type = Pointer[type, address_space].pointer_type
        ](alignment.value, size.value)


# ===----------------------------------------------------------------------===#
# aligned_free
# ===----------------------------------------------------------------------===#


@always_inline
fn _free(ptr: Pointer):
    @parameter
    if triple_is_nvidia_cuda():
        external_call["free", NoneType](ptr.bitcast[NoneType]())
    else:
        __mlir_op.`pop.aligned_free`(ptr.address)


@always_inline
fn _free(ptr: DTypePointer):
    _free(ptr.address)
