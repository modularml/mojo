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
"""Defines intrinsics.

You can import these APIs from the `sys` package. For example:

```mojo
from sys import PrefetchLocality
```
"""

import math

from memory import AddressSpace, UnsafePointer

from ._assembly import inlined_assembly
from .info import is_nvidia_gpu, sizeof

# ===-----------------------------------------------------------------------===#
# llvm_intrinsic
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn llvm_intrinsic[
    intrin: StringLiteral,
    type: AnyTrivialRegType,
    *types: AnyType,
    has_side_effect: Bool = True,
](*arguments: *types) -> type:
    """Calls an LLVM intrinsic with no arguments.

    Calls an LLVM intrinsic with the name intrin and return type type.

    Parameters:
      intrin: The name of the llvm intrinsic.
      type: The return type of the intrinsic.
      types: The argument types for the function.
      has_side_effect: If `True` the intrinsic will have side effects, otherwise its pure.

    Args:
      arguments: The arguments to the function.

    Returns:
      The result of calling the llvm intrinsic with no arguments.
    """

    var loaded_pack = arguments.get_loaded_kgen_pack()

    @parameter
    if _mlirtype_is_eq[type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value,
                _type=None,
            ](loaded_pack)
            return rebind[type](None)
        else:
            __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value,
                _type=None,
                hasSideEffects = __mlir_attr.false,
            ](loaded_pack)
            return rebind[type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value,
                _type=type,
            ](loaded_pack)
        else:
            return __mlir_op.`pop.call_llvm_intrinsic`[
                intrin = intrin.value,
                _type=type,
                hasSideEffects = __mlir_attr.false,
            ](loaded_pack)


# ===-----------------------------------------------------------------------===#
# _gather
# ===-----------------------------------------------------------------------===#


# NOTE: Converting from a scalar to a pointer is unsafe! The resulting pointer
# is assumed not to alias any Mojo-derived pointer. DO NOT proliferate usage of
# this function!
fn _unsafe_aliasing_address_to_pointer[
    type: DType
](owned addr: Scalar[DType.index]) -> UnsafePointer[Scalar[type]]:
    return UnsafePointer.address_of(addr).bitcast[
        UnsafePointer[Scalar[type]]
    ]()[]


@always_inline("nodebug")
fn gather[
    type: DType, size: Int, //
](
    owned base: SIMD[DType.index, size],
    mask: SIMD[DType.bool, size],
    passthrough: SIMD[type, size],
    alignment: Int = 0,
) -> SIMD[type, size]:
    """Reads scalar values from a SIMD vector, and gathers them into one vector.

    The gather function reads scalar values from a SIMD vector of memory
    locations and gathers them into one vector. The memory locations are
    provided in the vector of pointers `base` as addresses. The memory is
    accessed according to the provided mask. The mask holds a bit for each
    vector lane, and is used to prevent memory accesses to the masked-off
    lanes. The masked-off lanes in the result vector are taken from the
    corresponding lanes of the `passthrough` operand.

    In general, for some vector of pointers `base`, mask `mask`, and passthrough
    `pass` a call of the form:

    ```python
    gather(base, mask, pass)
    ```

    is equivalent to the following sequence of scalar loads in C++:

    ```cpp
    for (int i = 0; i < N; i++)
      result[i] = mask[i] ? *base[i] : passthrough[i];
    ```

    Parameters:
      type: DType of the return SIMD buffer.
      size: Size of the return SIMD buffer.

    Args:
      base: The vector containing memory addresses that gather will access.
      mask: A binary vector which prevents memory access to certain lanes of
        the base vector.
      passthrough: In the result vector, the masked-off lanes are replaced
        with the passthrough vector.
      alignment: The alignment of the source addresses. Must be 0 or a power
        of two constant integer value.

    Returns:
      A SIMD[type, size] containing the result of the gather operation.
    """

    @parameter
    if size == 1:
        return _unsafe_aliasing_address_to_pointer[type](
            base[0]
        ).load() if mask else passthrough[0]
    var result = llvm_intrinsic[
        "llvm.masked.gather",
        __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`],
    ](
        UnsafePointer.address_of(base).bitcast[
            __mlir_type[`!pop.simd<`, size.value, `, address>`],
        ]()[],
        Int32(alignment),
        mask,
        passthrough,
    )
    _ = base
    return result


# ===-----------------------------------------------------------------------===#
# _scatter
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn scatter[
    type: DType, size: Int, //
](
    value: SIMD[type, size],
    owned base: SIMD[DType.index, size],
    mask: SIMD[DType.bool, size],
    alignment: Int = 0,
):
    """Takes scalar values from a SIMD vector and `scatters` them into a
    vector of pointers.

    The scatter operation stores scalar values from a SIMD vector of memory
    locations and scatters them into a vector of pointers. The memory locations
    are provided in the vector of pointers `base` as addresses. The memory is
    stored according to the provided mask. The mask holds a bit for each vector
    lane, and is used to prevent memory accesses to the masked-off lanes.

    The `value` operand is a vector value to be written to memory. The `base`
    operand is a vector of pointers, pointing to where the value elements
    should be stored. It has the same underlying type as the value operand. The
    `mask` operand, mask, is a vector of boolean values. The types of the
    `mask` and the `value` operand must have the same number of vector
    elements.

    Scatter with overlapping addresses is guaranteed to be ordered from
    least-significant to most-significant element.

    In general, for some vector %value, vector of pointers %base, and mask
    %mask instructions of the form:

    ```mlir
    %0 = pop.simd.scatter %value, %base[%mask] : !pop.simd<N, type>
    ```

    is equivalent to the following sequence of scalar loads in C++:

    ```cpp
    for (int i = 0; i < N; i++)
      if (mask[i])
        base[i] = value[i];
    ```

    Parameters:
      type: DType of `value`, the result SIMD buffer.
      size: Size of `value`, the result SIMD buffer.

    Args:
      value: The vector that will contain the result of the scatter operation.
      base: The vector containing memory addresses that scatter will access.
      mask: A binary vector which prevents memory access to certain lanes of
        the base vector.
      alignment: The alignment of the source addresses. Must be 0 or a power
        of two constant integer value.
    """

    @parameter
    if size == 1:
        if mask:
            var ptr = _unsafe_aliasing_address_to_pointer[type](base[0])
            ptr.store(value[0])
        return
    llvm_intrinsic["llvm.masked.scatter", NoneType](
        value,
        UnsafePointer.address_of(base).bitcast[
            __mlir_type[`!pop.simd<`, size.value, `, address>`],
        ]()[],
        Int32(alignment),
        mask,
    )
    _ = base


# ===-----------------------------------------------------------------------===#
# prefetch
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct PrefetchLocality:
    """The prefetch locality.

    The locality, rw, and cache type correspond to LLVM prefetch intrinsic's
    inputs (see
    [LLVM prefetch locality](https://llvm.org/docs/LangRef.html#llvm-prefetch-intrinsic))
    """

    var value: Int32
    """The prefetch locality to use. It should be a value in [0, 3]."""
    alias NONE = PrefetchLocality(0)
    """No locality."""
    alias LOW = PrefetchLocality(1)
    """Low locality."""
    alias MEDIUM = PrefetchLocality(2)
    """Medium locality."""
    alias HIGH = PrefetchLocality(3)
    """Extremely local locality (keep in cache)."""

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: Int):
        """Constructs a prefetch locality option.

        Args:
            value: An integer value representing the locality. Should be a value
                   in the range `[0, 3]`.
        """
        self.value = value


@register_passable("trivial")
struct PrefetchRW:
    """Prefetch read or write."""

    var value: Int32
    """The read-write prefetch. It should be in [0, 1]."""
    alias READ = PrefetchRW(0)
    """Read prefetch."""
    alias WRITE = PrefetchRW(1)
    """Write prefetch."""

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: Int):
        """Constructs a prefetch read-write option.

        Args:
            value: An integer value representing the prefetch read-write option
                   to be used. Should be a value in the range `[0, 1]`.
        """
        self.value = value


# LLVM prefetch cache type
@register_passable("trivial")
struct PrefetchCache:
    """Prefetch cache type."""

    var value: Int32
    """The cache prefetch. It should be in [0, 1]."""
    alias INSTRUCTION = PrefetchCache(0)
    """The instruction prefetching option."""
    alias DATA = PrefetchCache(1)
    """The data prefetching option."""

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: Int):
        """Constructs a prefetch option.

        Args:
            value: An integer value representing the prefetch cache option to be
                   used. Should be a value in the range `[0, 1]`.
        """
        self.value = value


@register_passable("trivial")
struct PrefetchOptions:
    """Collection of configuration parameters for a prefetch intrinsic call.

    The op configuration follows similar interface as LLVM intrinsic prefetch
    op, with a "locality" attribute that specifies the level of temporal locality
    in the application, that is, how soon would the same data be visited again.
    Possible locality values are: `NONE`, `LOW`, `MEDIUM`, and `HIGH`.

    The op also takes a "cache tag" attribute giving hints on how the
    prefetched data will be used. Possible tags are: `ReadICache`, `ReadDCache`
    and `WriteDCache`.

    Note: the actual behavior of the prefetch op and concrete interpretation of
    these attributes are target-dependent.
    """

    var rw: PrefetchRW
    """Indicates prefetching for read or write."""
    var locality: PrefetchLocality
    """Indicates locality level."""
    var cache: PrefetchCache
    """Indicates i-cache or d-cache prefetching."""

    @always_inline("nodebug")
    fn __init__(out self):
        """Constructs an instance of PrefetchOptions with default params."""
        self.rw = PrefetchRW.READ
        self.locality = PrefetchLocality.HIGH
        self.cache = PrefetchCache.DATA

    @always_inline("nodebug")
    fn for_read(self) -> Self:
        """
        Sets the prefetch purpose to read.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.rw = PrefetchRW.READ
        return updated

    @always_inline("nodebug")
    fn for_write(self) -> Self:
        """
        Sets the prefetch purpose to write.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.rw = PrefetchRW.WRITE
        return updated

    @always_inline("nodebug")
    fn no_locality(self) -> Self:
        """
        Sets the prefetch locality to none.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.locality = PrefetchLocality.NONE
        return updated

    @always_inline("nodebug")
    fn low_locality(self) -> Self:
        """
        Sets the prefetch locality to low.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.locality = PrefetchLocality.LOW
        return updated

    @always_inline("nodebug")
    fn medium_locality(self) -> Self:
        """
        Sets the prefetch locality to medium.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.locality = PrefetchLocality.MEDIUM
        return updated

    @always_inline("nodebug")
    fn high_locality(self) -> Self:
        """
        Sets the prefetch locality to high.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.locality = PrefetchLocality.HIGH
        return updated

    @always_inline("nodebug")
    fn to_data_cache(self) -> Self:
        """
        Sets the prefetch target to data cache.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.cache = PrefetchCache.DATA
        return updated

    @always_inline("nodebug")
    fn to_instruction_cache(self) -> Self:
        """
        Sets the prefetch target to instruction cache.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.cache = PrefetchCache.INSTRUCTION
        return updated


@always_inline("nodebug")
fn prefetch[
    type: DType, //, params: PrefetchOptions = PrefetchOptions()
](addr: UnsafePointer[Scalar[type], **_]):
    """Prefetches an instruction or data into cache before it is used.

    The prefetch function provides prefetching hints for the target
    to prefetch instruction or data into cache before they are used.

    Parameters:
      type: The DType of value stored in addr.
      params: Configuration options for the prefect intrinsic.

    Args:
      addr: The data pointer to prefetch.
    """

    @parameter
    if is_nvidia_gpu():
        inlined_assembly[
            "prefetch.global.L2 [$0];",
            NoneType,
            constraints="l,~{memory}",
            has_side_effect=True,
        ](addr.bitcast[NoneType]())
    else:
        llvm_intrinsic["llvm.prefetch", NoneType](
            addr.bitcast[NoneType](),
            params.rw,
            params.locality,
            params.cache,
        )


# ===-----------------------------------------------------------------------===#
# masked load
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn masked_load[
    type: DType, //, size: Int
](
    addr: UnsafePointer[Scalar[type], **_],
    mask: SIMD[DType.bool, size],
    passthrough: SIMD[type, size],
    alignment: Int = 1,
) -> SIMD[type, size]:
    """Loads data from memory and return it, replacing masked lanes with values
    from the passthrough vector.

    Parameters:
      type: DType of the return SIMD buffer.
      size: Size of the return SIMD buffer.

    Args:
      addr: The base pointer for the load.
      mask: A binary vector which prevents memory access to certain lanes of
        the memory stored at addr.
      passthrough: In the result vector, the masked-off lanes are replaced
        with the passthrough vector.
      alignment: The alignment of the source addresses. Must be 0 or a power
        of two constant integer value. Default is 1.

    Returns:
      The loaded memory stored in a vector of type SIMD[type, size].
    """

    @parameter
    if size == 1:
        return addr.load() if mask else passthrough[0]

    return llvm_intrinsic["llvm.masked.load", SIMD[type, size]](
        addr.bitcast[NoneType]().address,
        Int32(alignment),
        mask,
        passthrough,
    )


# ===-----------------------------------------------------------------------===#
# masked store
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn masked_store[
    size: Int
](
    value: SIMD,
    addr: UnsafePointer[Scalar[value.type], **_],
    mask: SIMD[DType.bool, size],
    alignment: Int = 1,
):
    """Stores a value at a memory location, skipping masked lanes.

    Parameters:
      size: Size of `value`, the data to store.

    Args:
      value: The vector containing data to store.
      addr: A vector of memory location to store data at.
      mask: A binary vector which prevents memory access to certain lanes of
        `value`.
      alignment: The alignment of the destination locations. Must be 0 or a
        power of two constant integer value.
    """

    @parameter
    if size == 1:
        if mask:
            addr.store(value[0])
        return

    llvm_intrinsic["llvm.masked.store", NoneType](
        value,
        addr.bitcast[NoneType]().address,
        Int32(alignment),
        mask,
    )


# ===-----------------------------------------------------------------------===#
# compressed store
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn compressed_store[
    type: DType, size: Int
](
    value: SIMD[type, size],
    addr: UnsafePointer[Scalar[type], **_],
    mask: SIMD[DType.bool, size],
):
    """Compresses the lanes of `value`, skipping `mask` lanes, and stores
    at `addr`.

    Parameters:
      type: DType of `value`, the value to store.
      size: Size of `value`, the value to store.

    Args:
      value: The vector containing data to store.
      addr: The memory location to store the compressed data.
      mask: A binary vector which prevents memory access to certain lanes of
        `value`.
    """

    @parameter
    if size == 1:
        if mask:
            addr.store(value[0])
        return

    llvm_intrinsic["llvm.masked.compressstore", NoneType](
        value,
        addr.bitcast[NoneType]().address,
        mask,
    )


# ===-----------------------------------------------------------------------===#
# strided load
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn strided_load[
    type: DType, //, simd_width: Int
](
    addr: UnsafePointer[Scalar[type], **_],
    stride: Int,
    mask: SIMD[DType.bool, simd_width] = True,
) -> SIMD[type, simd_width]:
    """Loads values from addr according to a specific stride.

    Parameters:
      type: DType of `value`, the value to store.
      simd_width: The width of the SIMD vectors.

    Args:
      addr: The memory location to load data from.
      stride: How many lanes to skip before loading again.
      mask: A binary vector which prevents memory access to certain lanes of
        `value`.

    Returns:
      A vector containing the loaded data.
    """

    @parameter
    if simd_width == 1:
        return addr.load() if mask else Scalar[type]()

    var offset = int(addr) + stride * sizeof[type]() * math.iota[
        DType.index, simd_width
    ]()
    var passthrough = SIMD[type, simd_width]()
    return gather(offset, mask, passthrough)


# ===-----------------------------------------------------------------------===#
# strided store
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn strided_store[
    type: DType, //, simd_width: Int
](
    value: SIMD[type, simd_width],
    addr: UnsafePointer[Scalar[type], **_],
    stride: Int,
    mask: SIMD[DType.bool, simd_width] = True,
):
    """Loads values from addr according to a specific stride.

    Parameters:
      type: DType of `value`, the value to store.
      simd_width: The width of the SIMD vectors.

    Args:
      value: The values to store.
      addr: The location to store values at.
      stride: How many lanes to skip before storing again.
      mask: A binary vector which prevents memory access to certain lanes of
        `value`.
    """

    @parameter
    if simd_width == 1:
        if mask:
            addr.store(value[0])
        return

    var offset = int(addr) + stride * sizeof[type]() * math.iota[
        DType.index, simd_width
    ]()
    scatter(value, offset, mask)


# ===-------------------------------------------------------------------===#
# _mlirtype_is_eq
# ===-------------------------------------------------------------------===#


fn _mlirtype_is_eq[t1: AnyTrivialRegType, t2: AnyTrivialRegType]() -> Bool:
    """Compares the two type for equality.

    Parameters:
        t1: The LHS of the type comparison.
        t2: The RHS of the type comparison.

    Returns:
        Returns True if t1 and t2 are the same type and False otherwise.
    """
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        `#kgen.type<`,
        t1,
        `> : !kgen.type`,
        `,`,
        `#kgen.type<`,
        t2,
        `> : !kgen.type`,
        `> : i1`,
    ]


fn _type_is_eq[t1: AnyType, t2: AnyType]() -> Bool:
    """Compares the two type for equality.

    Parameters:
        t1: The LHS of the type comparison.
        t2: The RHS of the type comparison.

    Returns:
        Returns True if t1 and t2 are the same type and False otherwise.
    """
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        `#kgen.type<`,
        +t1,
        `> : !kgen.type`,
        `,`,
        `#kgen.type<`,
        +t2,
        `> : !kgen.type`,
        `> : i1`,
    ]


# ===----------------------------------------------------------------------=== #
# Transitional type used for llvm_intrinsic
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct _RegisterPackType[*a: AnyTrivialRegType]:
    var storage: __mlir_type[`!kgen.pack<`, a, `>`]

    @always_inline("nodebug")
    fn __getitem__[i: Int](self) -> a[i.value]:
        """Get the element.

        Parameters:
            i: The element index.

        Returns:
            The tuple element at the requested index.
        """
        return __mlir_op.`kgen.pack.extract`[index = i.value](self.storage)


# ===----------------------------------------------------------------------=== #
# expect
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn expect[T: AnyTrivialRegType, //, expected_val: T](val: T) -> T:
    """Provides information about expected (the most probable) value of `val`,
    which can be used by optimizers.

    Constraints:
        Only work with integer types.

    Parameters:
        T: The type of the input value.
        expected_val: The expected value of `val`.

    Args:
        val: The input value.

    Returns:
        The input value.
    """
    return llvm_intrinsic["llvm.expect", T, has_side_effect=False](
        val, expected_val
    )


# ===----------------------------------------------------------------------=== #
# likely
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn likely(val: Bool) -> Bool:
    """Provides information that the most probable value of `val` is going to be
    `True`. This information can be used by optimizers.

    Args:
        val: The input value which is likely to be `True` most of the time.

    Returns:
        The input value.
    """
    return expect[True](val)


# ===----------------------------------------------------------------------=== #
# unlikely
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn unlikely(val: Bool) -> Bool:
    """Provides information that the most probable value of `val` is going to be
    `False`. This information can be used by optimizers.

    Args:
        val: The input value which is likely to be `False` most of the time.

    Returns:
        The input value.
    """
    return expect[False](val)


# ===----------------------------------------------------------------------=== #
# assume
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn assume(val: Bool):
    """Signals to the optimizer that the condition is always true. This allows
    the optimizer to optimize the code.

    Args:
      val: The input value which is assumed to be `True`.
    """
    llvm_intrinsic["llvm.assume", NoneType](val)
