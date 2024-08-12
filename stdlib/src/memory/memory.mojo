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


from sys import alignof, llvm_intrinsic, sizeof, triple_is_nvidia_cuda

from builtin.dtype import _integral_type_of
from memory.reference import AddressSpace, _GPUAddressSpace

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
fn _memcmp_impl_unconstrained[
    type: DType
](s1: UnsafePointer[Scalar[type], *_], s2: __type_of(s1), count: Int) -> Int:
    alias simd_width = simdwidthof[type]()
    if count < simd_width:
        for i in range(count):
            var s1i = s1[i]
            var s2i = s2[i]
            if s1i != s2i:
                return 1 if s1i > s2i else -1
        return 0

    var iota = llvm_intrinsic[
        "llvm.experimental.stepvector",
        SIMD[DType.uint8, simd_width],
        has_side_effect=False,
    ]()

    var last = count - simd_width

    for i in range(0, last, simd_width):
        var s1i = s1.load[width=simd_width](i)
        var s2i = s2.load[width=simd_width](i)
        var diff = s1i != s2i
        if any(diff):
            var index = int(
                diff.select(
                    iota, SIMD[DType.uint8, simd_width](255)
                ).reduce_min()
            )
            return -1 if s1i[index] < s2i[index] else 1

    var s1i = s1.load[width=simd_width](last)
    var s2i = s2.load[width=simd_width](last)
    var diff = s1i != s2i
    if any(diff):
        var index = int(
            diff.select(iota, SIMD[DType.uint8, simd_width](255)).reduce_min()
        )
        return -1 if s1i[index] < s2i[index] else 1
    return 0


@always_inline
fn _memcmp_impl[
    type: DType
](s1: UnsafePointer[Scalar[type], *_], s2: __type_of(s1), count: Int) -> Int:
    constrained[type.is_integral(), "the input dtype must be integral"]()
    return _memcmp_impl_unconstrained(s1, s2, count)


@always_inline
fn memcmp[
    type: AnyType, address_space: AddressSpace
](
    s1: UnsafePointer[type, address_space],
    s2: UnsafePointer[type, address_space],
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
        return _memcmp_impl(
            s1.bitcast[Int32](),
            s2.bitcast[Int32](),
            byte_count // sizeof[DType.int32](),
        )

    return _memcmp_impl(s1.bitcast[Int8](), s2.bitcast[Int8](), byte_count)


# ===----------------------------------------------------------------------===#
# memcpy
# ===----------------------------------------------------------------------===#


@always_inline
fn memcpy[count: Int](dest: UnsafePointer, src: __type_of(dest)):
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

        @parameter
        for i in range(n):
            dest_data[i] = src_data[i]
        return

    @parameter
    if n <= 16:

        @parameter
        if n >= 8:
            var ui64_size = sizeof[Int64]()
            dest_data.bitcast[Int64]()[] = src_data.bitcast[Int64]()[0]
            dest_data.offset(n - ui64_size).bitcast[
                Int64
            ]()[] = src_data.offset(n - ui64_size).bitcast[Int64]()[0]
            return

        var ui32_size = sizeof[Int32]()
        dest_data.bitcast[Int32]()[] = src_data.bitcast[Int32]()[0]
        dest_data.offset(n - ui32_size).bitcast[Int32]()[] = src_data.offset(
            n - ui32_size
        ).bitcast[Int32]()[0]
        return

    var dest_ptr = dest_data.bitcast[Int8]()
    var src_ptr = src_data.bitcast[Int8]()

    # Copy in 32-byte chunks.
    alias chunk_size = 32
    alias vector_end = _align_down(n, chunk_size)
    for i in range(0, vector_end, chunk_size):
        dest_ptr.store(i, src_ptr.load[width=chunk_size](i))
    for i in range(vector_end, n):
        dest_ptr.store(i, src_ptr.load(i))


@always_inline
fn memcpy(
    dest_data: UnsafePointer[Int8, *_], src_data: __type_of(dest_data), n: Int
):
    """Copies a memory area.

    Args:
        dest_data: The destination pointer.
        src_data: The source pointer.
        n: The number of bytes to copy.
    """
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
            dest_data.bitcast[Int64]()[] = src_data.bitcast[Int64]()[0]
            dest_data.offset(n - ui64_size).bitcast[
                Int64
            ]()[] = src_data.offset(n - ui64_size).bitcast[Int64]()[0]
            return
        var ui32_size = sizeof[Int32]()
        dest_data.bitcast[Int32]()[] = src_data.bitcast[Int32]()[0]
        dest_data.offset(n - ui32_size).bitcast[Int32]()[] = src_data.offset(
            n - ui32_size
        ).bitcast[Int32]()[0]
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

    var dest_ptr = dest_data.bitcast[Int8]()
    var src_ptr = src_data.bitcast[Int8]()

    # Copy in 32-byte chunks.
    alias chunk_size = 32
    var vector_end = _align_down(n, chunk_size)
    for i in range(0, vector_end, chunk_size):
        dest_ptr.store(i, src_ptr.load[width=chunk_size](i))
    for i in range(vector_end, n):
        dest_ptr.store(i, src_ptr.load(i))


@always_inline
fn memcpy(dest: UnsafePointer, src: __type_of(dest), count: Int):
    """Copies a memory area.

    Args:
        dest: The destination pointer.
        src: The source pointer.
        count: The number of elements to copy.
    """
    var n = count * sizeof[dest.type]()
    memcpy(dest.bitcast[Int8](), src.bitcast[Int8](), n)


# ===----------------------------------------------------------------------===#
# memset
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _memset_llvm[
    address_space: AddressSpace
](ptr: UnsafePointer[UInt8, address_space], value: UInt8, count: Int):
    llvm_intrinsic["llvm.memset", NoneType](
        ptr.address, value, count.value, False
    )


@always_inline
fn memset[
    type: AnyType, address_space: AddressSpace
](ptr: UnsafePointer[type, address_space], value: UInt8, count: Int):
    """Fills memory with the given value.

    Parameters:
        type: The element dtype.
        address_space: The address space of the pointer.

    Args:
        ptr: UnsafePointer to the beginning of the memory block to fill.
        value: The value to fill with.
        count: Number of elements to fill (in elements, not bytes).
    """
    _memset_llvm(ptr.bitcast[UInt8](), value, count * sizeof[type]())


# ===----------------------------------------------------------------------===#
# memset_zero
# ===----------------------------------------------------------------------===#


@always_inline
fn memset_zero[
    type: AnyType, address_space: AddressSpace
](ptr: UnsafePointer[type, address_space], count: Int):
    """Fills memory with zeros.

    Parameters:
        type: The element type.
        address_space: The address space of the pointer.

    Args:
        ptr: UnsafePointer to the beginning of the memory block to fill.
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
    alignment: Int = alignof[type]() if triple_is_nvidia_cuda() else 1,
    address_space: AddressSpace = AddressSpace.GENERIC,
]() -> UnsafePointer[Scalar[type], address_space]:
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
    type: AnyType,
    /,
    alignment: Int = alignof[type]() if triple_is_nvidia_cuda() else 1,
    address_space: AddressSpace = AddressSpace.GENERIC,
]() -> UnsafePointer[type, address_space]:
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
    if triple_is_nvidia_cuda() and address_space in (
        _GPUAddressSpace.SHARED,
        _GPUAddressSpace.PARAM,
    ):
        return __mlir_op.`pop.global_alloc`[
            count = count.value,
            _type = UnsafePointer[type, address_space]._mlir_type,
            alignment = alignment.value,
            address_space = address_space._value.value,
        ]()
    else:
        return __mlir_op.`pop.stack_allocation`[
            count = count.value,
            _type = UnsafePointer[type, address_space]._mlir_type,
            alignment = alignment.value,
            address_space = address_space._value.value,
        ]()


# ===----------------------------------------------------------------------===#
# malloc
# ===----------------------------------------------------------------------===#


@always_inline
fn _malloc[
    type: AnyType,
    /,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
](size: Int, /, *, alignment: Int = -1) -> UnsafePointer[type, address_space]:
    @parameter
    if triple_is_nvidia_cuda():
        constrained[
            address_space is AddressSpace.GENERIC,
            "address space must be generic",
        ]()
        return external_call["malloc", UnsafePointer[NoneType, address_space]](
            size
        ).bitcast[type]()
    else:
        return __mlir_op.`pop.aligned_alloc`[
            _type = UnsafePointer[type, address_space]._mlir_type
        ](alignment.value, size.value)


# ===----------------------------------------------------------------------===#
# aligned_free
# ===----------------------------------------------------------------------===#


@always_inline
fn _free(ptr: UnsafePointer):
    @parameter
    if triple_is_nvidia_cuda():
        constrained[
            ptr.address_space is AddressSpace.GENERIC,
            "address space must be generic",
        ]()
        external_call["free", NoneType](ptr.bitcast[NoneType]())
    else:
        __mlir_op.`pop.aligned_free`(ptr.address)
