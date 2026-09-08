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
"""Defines intrinsics.

You can import these APIs from the `sys` package. For example:

```mojo
from std.sys import PrefetchLocality
```
"""

import std.math
from std.collections.string.string_span import _get_kgen_string
from std.memory._poison import _check_not_poison_masked
from std.sys.info import _is_sm_9x_or_newer, is_gpu


from ._assembly import inlined_assembly
from .info import is_amd_gpu, is_apple_gpu, is_nvidia_gpu, size_of


# ===-----------------------------------------------------------------------===#
# llvm_intrinsic
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def llvm_intrinsic[
    intrin: StaticString,
    type: TrivialRegisterPassable,
    *types: AnyType,
    has_side_effect: Bool = True,
](*args: *types) -> type:
    """Calls an LLVM intrinsic with the name `intrin` and return type `type`.

    Parameters:
        intrin: The name of the llvm intrinsic.
        type: The return type of the intrinsic.
        types: The argument types for the function.
        has_side_effect: If `True` the intrinsic will have side effects,
            otherwise its pure.

    Args:
        args: The arguments to the function.

    Returns:
        The result of calling the llvm intrinsic with no arguments.
    """

    var loaded_pack = args.get_loaded_kgen_pack()

    comptime intrin_kgen_string = _get_kgen_string[intrin]()

    comptime if type == NoneType:
        __mlir_op.`pop.call_llvm_intrinsic`[
            intrin=intrin_kgen_string,
            _type=None,
            hasSideEffects=has_side_effect.__mlir_i1__(),
        ](loaded_pack)
        return rebind[type](None)

    else:
        return __mlir_op.`pop.call_llvm_intrinsic`[
            intrin=intrin_kgen_string,
            _type=type,
            hasSideEffects=has_side_effect.__mlir_i1__(),
        ](loaded_pack)


# ===-----------------------------------------------------------------------===#
# _gather
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def gather[
    dtype: DType,
    size: SIMDLength,
    //,
    *,
    invariant: Bool = False,
    alignment: Int = 0,
](
    var base: SIMD[.int, size],
    mask: SIMD[.bool, size],
    passthrough: SIMD[dtype, size],
) -> SIMD[dtype, size]:
    """Reads scalar values from a SIMD vector, and gathers them into one vector.

    The gather function reads scalar values from a SIMD vector of memory
    locations and gathers them into one vector. The memory locations are
    provided in the vector of pointers `base` as addresses. The memory is
    accessed according to the provided mask. The mask holds a bit for each
    vector lane, and is used to prevent memory accesses to the masked-off
    lanes. The masked-off lanes in the result vector are taken from the
    corresponding lanes of the `passthrough` operand.

    In general, for some vector of pointers `base`, mask `mask`, and passthrough
    `passthrough` a call of the form:

    ```text
    from std.sys.intrinsics import gather

    result = gather(base, mask, passthrough)
    ```

    is equivalent to the following sequence of scalar loads in C++:

    ```cpp
    for (int i = 0; i < N; i++)
      result[i] = mask[i] ? *base[i] : passthrough[i];
    ```

    Parameters:
      dtype: DType of the return SIMD buffer.
      size: Size of the return SIMD buffer.
      invariant: Whether the memory is load invariant.
      alignment: The alignment of the source addresses. Must be 0 or a power
        of two constant integer value.

    Args:
      base: The vector containing memory addresses that gather will access.
      mask: A binary vector which prevents memory access to certain lanes of
        the base vector.
      passthrough: In the result vector, the masked-off lanes are replaced
        with the passthrough vector.

    Returns:
      A SIMD[dtype, size] containing the result of the gather operation.
    """

    comptime if size == 1:
        return Pointer[Scalar[dtype], MutUntrackedOrigin](
            unsafe_from_address=Int(base[0])
        ).unsafe_load[invariant=invariant]() if mask else SIMD[dtype, size](
            passthrough[0]
        )

    comptime if is_gpu() and invariant:
        var result = SIMD[dtype, size]()

        comptime for i in range(size):
            result[i] = Pointer[Scalar[dtype], MutUntrackedOrigin](
                unsafe_from_address=Int(base[i])
            ).unsafe_load[invariant=invariant]() if mask[i] else Scalar[dtype](
                passthrough[i]
            )
        return result

    var result = llvm_intrinsic[
        "llvm.masked.gather",
        SIMD[dtype, size]._mlir_type,
    ](
        Pointer(to=base).unsafe_bitcast[
            __mlir_type[`!kgen.simd<`, size._mlir_value, `, address>`],
        ]()[],
        Int32(alignment),
        mask,
        passthrough,
    )
    _ = base
    var loaded = SIMD[dtype, size](mlir_value=result)
    comptime if dtype.is_floating_point():
        _check_not_poison_masked[dtype, size](loaded, mask)
    return loaded


# ===-----------------------------------------------------------------------===#
# _scatter
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def scatter[
    dtype: DType,
    size: SIMDLength,
    //,
    alignment: Int = 0,
](
    value: SIMD[dtype, size],
    var base: SIMD[.int, size],
    mask: SIMD[.bool, size],
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

    In general, for some vector `value`, vector of pointers `base`, and mask
    `mask` a call of the form:

    ```text
    from std.sys.intrinsics import scatter

    scatter(value, base, mask)
    ```

    is equivalent to the following sequence of scalar stores in C++:

    ```cpp
    for (int i = 0; i < N; i++)
      if (mask[i])
        base[i] = value[i];
    ```

    Parameters:
      dtype: DType of `value`, the result SIMD buffer.
      size: Size of `value`, the result SIMD buffer.
      alignment: The alignment of the source addresses. Must be 0 or a power
        of two constant integer value.

    Args:
      value: The vector that will contain the result of the scatter operation.
      base: The vector containing memory addresses that scatter will access.
      mask: A binary vector which prevents memory access to certain lanes of
        the base vector.
    """

    comptime if size == 1:
        if mask:
            var ptr = Pointer[Scalar[dtype], MutUntrackedOrigin](
                unsafe_from_address=Int(base[0])
            )
            ptr.unsafe_store(value[0])
        return
    llvm_intrinsic["llvm.masked.scatter", NoneType](
        value,
        Pointer(to=base).unsafe_bitcast[
            __mlir_type[`!kgen.simd<`, size._mlir_value, `, address>`],
        ]()[],
        Int32(alignment),
        mask,
    )
    _ = base


# ===-----------------------------------------------------------------------===#
# prefetch
# ===-----------------------------------------------------------------------===#


struct PrefetchLocality(TrivialRegisterPassable):
    """The prefetch locality.

    The locality, rw, and cache type correspond to LLVM prefetch intrinsic's
    inputs (see
    [LLVM prefetch locality](https://llvm.org/docs/LangRef.html#llvm-prefetch-intrinsic))
    """

    var value: Int32
    """The prefetch locality to use. It should be a value in [0, 3]."""
    comptime NONE = PrefetchLocality(0)
    """No locality."""
    comptime LOW = PrefetchLocality(1)
    """Low locality."""
    comptime MEDIUM = PrefetchLocality(2)
    """Medium locality."""
    comptime HIGH = PrefetchLocality(3)
    """Extremely local locality (keep in cache)."""

    @always_inline("nodebug")
    def __init__(out self, value: Int):
        """Constructs a prefetch locality option.

        Args:
            value: An integer value representing the locality. Should be a value
                   in the range `[0, 3]`.
        """
        self.value = Int32(value)


struct PrefetchRW(TrivialRegisterPassable):
    """Prefetch read or write."""

    var value: Int32
    """The read-write prefetch. It should be in [0, 1]."""
    comptime READ = PrefetchRW(0)
    """Read prefetch."""
    comptime WRITE = PrefetchRW(1)
    """Write prefetch."""

    @always_inline("nodebug")
    def __init__(out self, value: Int):
        """Constructs a prefetch read-write option.

        Args:
            value: An integer value representing the prefetch read-write option
                   to be used. Should be a value in the range `[0, 1]`.
        """
        self.value = Int32(value)

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        """Checks if two prefetch read-write options are equal.

        Args:
            other: The option to compare with.

        Returns:
            True if the two prefetch read-write options are equal and False otherwise.
        """
        return self.value == other.value


# LLVM prefetch cache type
struct PrefetchCache(TrivialRegisterPassable):
    """Prefetch cache type."""

    var value: Int32
    """The cache prefetch. It should be in [0, 1]."""
    comptime INSTRUCTION = PrefetchCache(0)
    """The instruction prefetching option."""
    comptime DATA = PrefetchCache(1)
    """The data prefetching option."""

    @always_inline("nodebug")
    def __init__(out self, value: Int):
        """Constructs a prefetch option.

        Args:
            value: An integer value representing the prefetch cache option to be
                   used. Should be a value in the range `[0, 1]`.
        """
        self.value = Int32(value)


struct PrefetchOptions(Defaultable, TrivialRegisterPassable):
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
    def __init__(out self):
        """Constructs an instance of PrefetchOptions with default params."""
        self.rw = PrefetchRW.READ
        self.locality = PrefetchLocality.HIGH
        self.cache = PrefetchCache.DATA

    @always_inline("nodebug")
    def for_read(self) -> Self:
        """
        Sets the prefetch purpose to read.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.rw = PrefetchRW.READ
        return updated

    @always_inline("nodebug")
    def for_write(self) -> Self:
        """
        Sets the prefetch purpose to write.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.rw = PrefetchRW.WRITE
        return updated

    @always_inline("nodebug")
    def no_locality(self) -> Self:
        """
        Sets the prefetch locality to none.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.locality = PrefetchLocality.NONE
        return updated

    @always_inline("nodebug")
    def low_locality(self) -> Self:
        """
        Sets the prefetch locality to low.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.locality = PrefetchLocality.LOW
        return updated

    @always_inline("nodebug")
    def medium_locality(self) -> Self:
        """
        Sets the prefetch locality to medium.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.locality = PrefetchLocality.MEDIUM
        return updated

    @always_inline("nodebug")
    def high_locality(self) -> Self:
        """
        Sets the prefetch locality to high.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.locality = PrefetchLocality.HIGH
        return updated

    @always_inline("nodebug")
    def to_data_cache(self) -> Self:
        """
        Sets the prefetch target to data cache.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.cache = PrefetchCache.DATA
        return updated

    @always_inline("nodebug")
    def to_instruction_cache(self) -> Self:
        """
        Sets the prefetch target to instruction cache.

        Returns:
            The updated prefetch parameter.
        """
        var updated = self
        updated.cache = PrefetchCache.INSTRUCTION
        return updated


@always_inline("nodebug")
def prefetch[
    dtype: DType, //, params: PrefetchOptions = PrefetchOptions()
](addr: Pointer[Scalar[dtype], ...]):
    """Prefetches an instruction or data into cache before it is used.

    The prefetch function provides prefetching hints for the target
    to prefetch instruction or data into cache before they are used.

    Parameters:
      dtype: The DType of value stored in addr.
      params: Configuration options for the prefect intrinsic.

    Args:
      addr: The data pointer to prefetch.
    """

    comptime assert (
        params.rw == PrefetchRW.READ or type_of(addr).mut == True
    ), "prefetch pointer mutability must match the prefetch read-write option"

    comptime if is_nvidia_gpu():
        inlined_assembly[
            "prefetch.global.L2 [$0];",
            NoneType,
            constraints="l,~{memory}",
            has_side_effect=True,
        ](addr.unsafe_bitcast[NoneType]())
    elif is_apple_gpu():
        # Apple GPU officially does not support prefetch intrinsic
        pass
    else:
        llvm_intrinsic["llvm.prefetch", NoneType](
            addr.unsafe_bitcast[NoneType](),
            params.rw,
            params.locality,
            params.cache,
        )


# ===-----------------------------------------------------------------------===#
# masked load
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def masked_load[
    dtype: DType,
    //,
    size: SIMDLength,
    alignment: Int = 1,
](
    addr: ImmPointer[Scalar[dtype], ...],
    mask: SIMD[.bool, size],
    passthrough: SIMD[dtype, size],
) -> SIMD[dtype, size]:
    """Loads data from memory and return it, replacing masked lanes with values
    from the passthrough vector.

    Parameters:
      dtype: DType of the return SIMD buffer.
      size: Size of the return SIMD buffer.
      alignment: The alignment of the destination locations. Must be 0 or a
        power of two constant integer value. Default is 1.

    Args:
      addr: The base pointer for the load.
      mask: A binary vector which prevents memory access to certain lanes of
        the memory stored at addr.
      passthrough: In the result vector, the masked-off lanes are replaced
        with the passthrough vector.

    Returns:
      The loaded memory stored in a vector of type SIMD[dtype, size].
    """
    comptime if size == 1:
        return addr.unsafe_load() if mask else SIMD[dtype, size](passthrough[0])

    var result = llvm_intrinsic["llvm.masked.load", SIMD[dtype, size]](
        addr.unsafe_bitcast[NoneType](),
        Int32(alignment),
        mask,
        passthrough,
    )
    comptime if dtype.is_floating_point():
        _check_not_poison_masked[dtype, size](result, mask)
    return result


# ===-----------------------------------------------------------------------===#
# masked store
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def masked_store[
    size: SIMDLength,
    alignment: Int = 1,
](
    value: SIMD,
    addr: MutPointer[Scalar[value.dtype], ...],
    mask: SIMD[.bool, size],
):
    """Stores a value at a memory location, skipping masked lanes.

    Parameters:
      size: Size of `value`, the data to store.
      alignment: The alignment of the destination locations. Must be 0 or a
        power of two constant integer value. Default is 1.

    Args:
      value: The vector containing data to store.
      addr: A vector of memory location to store data at.
      mask: A binary vector which prevents memory access to certain lanes of
        `value`.
    """
    comptime if size == 1:
        if mask:
            addr.unsafe_store(value[0])
        return

    llvm_intrinsic["llvm.masked.store", NoneType](
        value,
        addr.unsafe_bitcast[NoneType](),
        Int32(alignment),
        mask,
    )


# ===-----------------------------------------------------------------------===#
# compressed store
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def compressed_store[
    dtype: DType, size: SIMDLength
](
    value: SIMD[dtype, size],
    addr: MutPointer[Scalar[dtype], ...],
    mask: SIMD[.bool, size],
):
    """Compresses the lanes of `value`, skipping `mask` lanes, and stores
    at `addr`.

    Parameters:
      dtype: DType of `value`, the value to store.
      size: Size of `value`, the value to store.

    Args:
      value: The vector containing data to store.
      addr: The memory location to store the compressed data.
      mask: A binary vector which prevents memory access to certain lanes of
        `value`.
    """
    comptime if size == 1:
        if mask:
            addr.unsafe_store(value[0])
        return

    llvm_intrinsic["llvm.masked.compressstore", NoneType](
        value,
        addr.unsafe_bitcast[NoneType](),
        mask,
    )


# ===-----------------------------------------------------------------------===#
# strided load
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def strided_load[
    dtype: DType, //, simd_width: SIMDLength, *, invariant: Bool = False
](
    addr: ImmPointer[Scalar[dtype], ...],
    stride: Int,
    mask: SIMD[.bool, simd_width] = SIMD[.bool, simd_width](fill=True),
) -> SIMD[dtype, simd_width]:
    """Loads values from addr according to a specific stride.

    Parameters:
      dtype: DType of `value`, the value to store.
      simd_width: The width of the SIMD vectors.
      invariant: Whether the memory is load invariant.

    Args:
      addr: The memory location to load data from.
      stride: How many lanes to skip before loading again.
      mask: A binary vector which prevents memory access to certain lanes of
        `value`.

    Returns:
      A vector containing the loaded data.
    """
    comptime if simd_width == 1:
        return addr.unsafe_load[invariant=invariant]() if mask else Scalar[
            dtype
        ]()

    comptime if is_apple_gpu():
        # The `gather` path below would erase address space via
        # `Int(addr)`; on Apple AIR the resulting GENERIC load silently
        # reads zero (MOCO-3762).
        var result = SIMD[dtype, simd_width]()
        comptime for i in range(simd_width):
            if mask[i]:
                result[i] = addr.unsafe_offset(i * stride).unsafe_load[
                    invariant=invariant
                ]()
        return result

    var offset = (
        SIMD[.int, simd_width](Int(addr))
        + SIMD[.int, simd_width](stride * size_of[dtype]())
        * std.math.iota[.int, simd_width]()
    )
    var passthrough = SIMD[dtype, simd_width]()
    return gather[invariant=invariant](offset, mask, passthrough)


# ===-----------------------------------------------------------------------===#
# strided store
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def strided_store[
    dtype: DType, //, simd_width: SIMDLength
](
    value: SIMD[dtype, simd_width],
    addr: MutPointer[Scalar[dtype], ...],
    stride: Int,
    mask: SIMD[.bool, simd_width] = SIMD[.bool, simd_width](fill=True),
):
    """Loads values from addr according to a specific stride.

    Parameters:
      dtype: DType of `value`, the value to store.
      simd_width: The width of the SIMD vectors.

    Args:
      value: The values to store.
      addr: The location to store values at.
      stride: How many lanes to skip before storing again.
      mask: A binary vector which prevents memory access to certain lanes of
        `value`.
    """
    comptime if simd_width == 1:
        if mask:
            addr.unsafe_store(value[0])
        return

    var offset = (
        SIMD[.int, simd_width](Int(addr))
        + SIMD[.int, simd_width](stride * size_of[dtype]())
        * std.math.iota[.int, simd_width]()
    )
    scatter(value, offset, mask)


# ===----------------------------------------------------------------------=== #
# Transitional type used for llvm_intrinsic
# ===----------------------------------------------------------------------=== #


struct _RegisterPackType[*Ts: TrivialRegisterPassable](TrivialRegisterPassable):
    comptime _mlir_type = __mlir_type[
        `!kgen.struct<`, ~Self.Ts.values, ` isParamPack>`
    ]

    var _mlir_value: Self._mlir_type

    @always_inline("nodebug")
    def __getitem_param__[i: Int](self) -> Self.Ts[i]:
        """Get the element.

        Parameters:
            i: The element index.

        Returns:
            The tuple element at the requested index.
        """
        return __mlir_op.`kgen.struct.extract`[index=i.__mlir_index__()](
            self._mlir_value
        )


# ===----------------------------------------------------------------------=== #
# expect
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
def expect[T: TrivialRegisterPassable, //, expected_val: T](val: T) -> T:
    """Provides information about expected (the most probable) value of `val`,
    which can be used by optimizers.

    Parameters:
        T: The type of the input value.
        expected_val: The expected value of `val`.

    Args:
        val: The input value.

    Returns:
        The input value.

    Notes:
        Only works with integer/boolean types.
    """
    if __is_run_in_comptime_interpreter:
        return val
    return llvm_intrinsic["llvm.expect", T, has_side_effect=False](
        val, expected_val
    )


# ===----------------------------------------------------------------------=== #
# likely
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
def likely(val: Bool) -> Bool:
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
def unlikely(val: Bool) -> Bool:
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
def assume(val: Bool):
    """Signals to the optimizer that the condition is always true. This allows
    the optimizer to optimize the code.

    Args:
      val: The input value which is assumed to be `True`.
    """
    if __is_run_in_comptime_interpreter:
        return
    llvm_intrinsic["llvm.assume", NoneType](val)


# ===-----------------------------------------------------------------------===#
# implicitarg_ptr
# ===-----------------------------------------------------------------------===#


@always_inline
def implicitarg_ptr(
    out result: Pointer[UInt8, MutUntrackedOrigin, address_space=.CONSTANT]
):
    """
    Get a pointer to AMD's implicit arguments table.

    Returns:
        A pointer to LLVM's implicit arguments table.
    """
    comptime assert is_amd_gpu(), "This intrinsic is only defined for AMD GPUs"
    result = llvm_intrinsic[
        "llvm.amdgcn.implicitarg.ptr",
        type_of(result),
    ]()


# ===-----------------------------------------------------------------------===#
# readfirstlane
# ===-----------------------------------------------------------------------===#


@always_inline
def readfirstlane[T: AnyType, //](value: Pointer[T, ...]) -> type_of(value):
    """
    Get the value in the lowest active lane of the input operand.

    Parameters:
        T: The type the pointer points to.

    Args:
        value: The input pointer.

    Returns:
        The value in the lowest active lane of the input operand.
    """
    comptime assert is_amd_gpu(), "This intrinsic is only defined for AMD GPUs"
    return llvm_intrinsic[
        "llvm.amdgcn.readfirstlane", type_of(value), type_of(value)
    ](value)


@always_inline
def readfirstlane[dtype: DType](value: Scalar[dtype]) -> Scalar[dtype]:
    """Gets the value in the lowest active lane of the input operand.

    Constraints:
        The scalar type must be 2, 4, or 8 bytes wide.

    Parameters:
        dtype: The element type.

    Args:
        value: The input scalar value.

    Returns:
        The value in the lowest active lane of the input operand.
    """
    comptime assert is_amd_gpu(), "This intrinsic is only defined for AMD GPUs"
    comptime assert (
        size_of[Scalar[dtype]]() >= 2
    ), "readfirstlane requires a scalar type of at least 16 bits"
    return llvm_intrinsic[
        "llvm.amdgcn.readfirstlane", Scalar[dtype], Scalar[dtype]
    ](value)


# ===-----------------------------------------------------------------------===#
# sendmsg
# ===-----------------------------------------------------------------------===#


@always_inline
def sendmsg(opcode: Int32, msg: Int32):
    """
    Send a message to fixed function hardware.
    Refer to the specific ISA manual for the ops and messages.

    Args:
        opcode: The operation to perform.
        msg: The message to send.
    """
    comptime assert is_amd_gpu(), "This intrinsic is only defined for AMD GPUs"
    _ = llvm_intrinsic["llvm.amdgcn.s.sendmsg", NoneType, Int32, Int32](
        opcode, msg
    )


# ===-----------------------------------------------------------------------===#
# ballot
# ===-----------------------------------------------------------------------===#


@always_inline
def ballot[dtype: DType](value: Bool) -> Scalar[dtype]:
    """
    Returns a bitfield(Int32 or Int64) containing the result
    of its Bool argument in all active lanes, and zero in all inactive lanes.
    For example, ballot(True) returns EXEC mask.

    Parameters:
        dtype: The DType of the return type.

    Args:
        value: The value to place across the mask.

    Returns:
        A bitfield(Int32 or Int64) containing the result of its Bool argument in all active lanes.
    """
    comptime assert is_amd_gpu(), "This intrinsic is only defined for AMD GPUs"
    comptime assert (
        dtype == .int32 or dtype == .int64
    ), "This intrinsic is only defined for i32 or i64"
    return llvm_intrinsic["llvm.amdgcn.ballot", Scalar[dtype]](value)
