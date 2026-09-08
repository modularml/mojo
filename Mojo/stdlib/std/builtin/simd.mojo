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
"""Implements SIMD primitives and abstractions.

Provides high-performance SIMD primitives and abstractions for
vectorized computation in Mojo. It enables efficient data-parallel operations
by leveraging hardware vector processing units across different architectures.

Key Features:
1. Architecture-agnostic SIMD abstractions with automatic hardware detection
2. Optimized vector operations for common numerical computations
3. Explicit control over vectorization strategies and memory layouts
4. Zero-cost abstractions that compile to efficient machine code
5. Support for different vector widths and element types

Primary Components:
- Vector types: Strongly-typed vector containers with element-wise operations
- SIMD intrinsics: Low-level access to hardware SIMD instructions
- Vectorized algorithms: Common algorithms optimized for SIMD execution
- Memory utilities: Aligned memory allocation and vector load/store operations

Performance Considerations:
- Vector width selection should match target hardware capabilities
- Memory alignment affects load/store performance
- Data layout transformations may be necessary for optimal vectorization

Integration:
This module is designed to work seamlessly with other Mojo numerical computing
components, including tensor operations, linear algebra routines, and
domain-specific libraries for machine learning and scientific computing.
"""

import std.math
from std.collections import Array
from std.collections.interval import IntervalElement
from std.collections.string.string import (
    _calc_initial_buffer_size_int32,
    _calc_initial_buffer_size_int64,
    _calc_initial_buffer_size,
)
from std.hashlib.hasher import Hasher
from std.math import Ceilable, CeilDivable, Floorable, Truncable
from std.math.math import _call_ptx_intrinsic, trunc
from std.sys import (
    CompilationTarget,
    _RegisterPackType,
    align_of,
    bit_width_of,
    is_amd_gpu,
    is_apple_gpu,
    is_big_endian,
    is_gpu,
    is_nvidia_gpu,
    llvm_intrinsic,
    simd_width_of,
    size_of,
)
from std.sys._assembly import inlined_assembly
from std.sys.info import (
    _cdna_4_or_newer,
    _is_amd_mi300x,
    _is_sm_9x_or_newer,
    _is_sm_100x_or_newer,
    _is_sm_120x_or_newer,
    is_32bit,
)

from std.bit import bit_width, byte_swap, pop_count
from std.builtin._format_float import _write_float
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.builtin.format_int import _write_int
from std.builtin.simd_length import SIMDLength
from std.builtin.int import _FromInt
from std.math import DivModable, Powable
from std.memory import bitcast, unsafe_memcpy, pack_bits
from std.python import (
    ConvertibleToPython,
    ConvertibleFromPython,
    Python,
    PythonObject,
)

from std.utils import IndexList, StaticTuple
from std.utils._visualizers import lldb_formatter_wrapping_type
from std.utils.coord import CoordLike, Coord
from std.utils.numerics import FPUtils
from std.utils.numerics import inf as _inf
from std.utils.numerics import isinf as _isinf
from std.utils.numerics import isnan as _isnan
from std.utils.numerics import max_finite as _max_finite
from std.utils.numerics import max_or_inf as _max_or_inf
from std.utils.numerics import min_finite as _min_finite
from std.utils.numerics import min_or_neg_inf as _min_or_neg_inf
from std.utils.numerics import nan as _nan

from .dtype import (
    _integral_type_of,
    _uint_type_of_width,
    _unsigned_integral_type_of,
)

# ===----------------------------------------------------------------------=== #
# Type Aliases
# ===----------------------------------------------------------------------=== #
comptime Scalar = SIMD[
    _, length=__mlir_attr[`#lit.struct<{_mlir_value = 1}> : `, SIMDLength]
]
"""Represents a scalar dtype."""

comptime Int = Scalar[DType.int]
"""Represents a signed integer suitable for indexing."""
comptime Int8 = Scalar[DType.int8]
"""Represents an 8-bit signed scalar integer."""
comptime UInt8 = Scalar[DType.uint8]
"""Represents an 8-bit unsigned scalar integer."""
comptime Int16 = Scalar[DType.int16]
"""Represents a 16-bit signed scalar integer."""
comptime UInt16 = Scalar[DType.uint16]
"""Represents a 16-bit unsigned scalar integer."""
comptime Int32 = Scalar[DType.int32]
"""Represents a 32-bit signed scalar integer."""
comptime UInt32 = Scalar[DType.uint32]
"""Represents a 32-bit unsigned scalar integer."""
comptime Int64 = Scalar[DType.int64]
"""Represents a 64-bit signed scalar integer."""
comptime UInt64 = Scalar[DType.uint64]
"""Represents a 64-bit unsigned scalar integer."""
comptime Int128 = Scalar[DType.int128]
"""Represents a 128-bit signed scalar integer."""
comptime UInt128 = Scalar[DType.uint128]
"""Represents a 128-bit unsigned scalar integer."""
comptime Int256 = Scalar[DType.int256]
"""Represents a 256-bit signed scalar integer."""
comptime UInt256 = Scalar[DType.uint256]
"""Represents a 256-bit unsigned scalar integer."""

comptime Float4_e2m1fn = Scalar[DType.float4_e2m1fn]
"""Represents a 4-bit `e2m1` floating point format.

This type is encoded as `s.ee.m` and defined by the
[Open Compute MX Format Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf):

- (s)ign: 1 bit
- (e)xponent: 2 bits
- (m)antissa: 1 bits
- exponent_bias: 1
"""
comptime Float8_e5m2 = Scalar[DType.float8_e5m2]
"""Represents the 8-bit E5M2 floating point format.

This type is from the [OFP8
standard](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1),
encoded as `seeeeemm`:
- (s)ign: 1 bit
- (e)xponent: 5 bits
- (m)antissa: 2 bits
- exponent bias: 15
- nan: {0,1}11111{01,10,11}
- inf: 01111100
- -inf: 11111100
- -0: 10000000
"""
comptime Float8_e5m2fnuz = Scalar[DType.float8_e5m2fnuz]
"""Represents an 8-bit floating point format.

This type is encoded as `seeeeemm`:
- (s)ign: 1 bit
- (e)xponent: 5 bits
- (m)antissa: 2 bits
- exponent bias: 16
- nan: 10000000
- fn: finite (no inf or -inf encodings)
- uz: unsigned zero (no -0 encoding)
"""
comptime Float8_e4m3fn = Scalar[DType.float8_e4m3fn]
"""Represents the E4M3 floating point format defined in the [OFP8
standard](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1).

This type is named differently across libraries and vendors, for example:
- Mojo, PyTorch, JAX, and LLVM refer to it as `e4m3fn`.
- OCP, NVIDIA CUDA, and AMD ROCm refer to it as `e4m3`.

In these contexts, they are all referring to the same finite type specified
in the OFP8 standard above, encoded as `seeeemmm`:
- (s)ign: 1 bit
- (e)xponent: 4 bits
- (m)antissa: 3 bits
- exponent bias: 7
- nan: 01111111, 11111111
- -0: 10000000
- fn: finite (no inf or -inf encodings)
"""
comptime Float8_e4m3fnuz = Scalar[DType.float8_e4m3fnuz]
"""Represents an 8-bit e4m3fnuz floating point format.

This type is encoded as `seeeemmm`:
- (s)ign: 1 bit
- (e)xponent: 4 bits
- (m)antissa: 3 bits
- exponent bias: 8
- nan: 10000000
- fn: finite (no inf or -inf encodings)
- uz: unsigned zero (no -0 encoding)
"""
comptime Float8_e8m0fnu = Scalar[DType.float8_e8m0fnu]
"""Represents the 8-bit E8M0FNU floating point format.

This type is defined in the [OCP MX
spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf),
encoded as `eeeeeeee`:
- (e)xponent: 8 bits
- (m)antissa: 0 bits
- exponent bias: 127
- nan: 11111111
- fn: finite (no inf or -inf encodings)
- u: unsigned (no sign bit or zero value)
"""
comptime BFloat16 = Scalar[DType.bfloat16]
"""Represents a 16-bit brain floating point value."""
comptime Float16 = Scalar[DType.float16]
"""Represents a 16-bit floating point value."""
comptime Float32 = Scalar[DType.float32]
"""Represents a 32-bit floating point value."""
comptime Float64 = Scalar[DType.float64]
"""Represents a 64-bit floating point value."""

comptime Byte = UInt8
"""Represents a byte (backed by an 8-bit unsigned integer)."""

comptime UInt = Scalar[DType.uint]
"""Represents an unsigned integer of platform-dependent bit-width."""

# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@always_inline("builtin")
def _simd_dtype_checks[dtype: DType]() -> Bool:
    """Checks that the SIMD element dtype can be held in a value.

    The sub-byte float formats are storage-only: no target registers a
    `pop.cast` for them, so such a value can neither be produced from nor read
    back into another float type.

    Parameters:
      dtype: The data type of SIMD vector elements.

    Returns:
        Whether a SIMD value may be constructed with this element dtype.
    """
    return (
        dtype != DType.float4_e2m1fn
        and dtype != DType.float6_e2m3fn
        and dtype != DType.float6_e3m2fn
    )


@always_inline("builtin")
def _simd_construction_checks[dtype: DType, size: SIMDLength]():
    """Checks if the SIMD dtype and size are valid.

    The SIMD size is valid if it is a power of two and is positive.

    Parameters:
      dtype: The data type of SIMD vector elements.
      size: The number of elements in the SIMD vector. The size must not be greater than 2**15.
    """
    comptime assert _simd_dtype_checks[dtype](), (
        "cannot construct a SIMD value with a sub-byte float dtype; these are"
        " storage-only formats with no conversion support on any target."
    )
    comptime assert size.is_power_of_two(), "simd width must be power of 2"
    # MOCO-1388: Until LLVM's issue #122571 is fixed, LLVM's SelectionDAG has
    # a limit of 2^15 for the number of operands of the instruction.
    # NOTE: Even after the limit increases in LLVM, compile time might be 3x
    # slower than with GCC, therefore until we have a real use case for large
    # SIMD, we better to keep limit at 2^15.
    # NOTE: Might need to revisit the limit for targets that use GlobalISel
    # as it does have smaller limit now.
    # comptime assert (
    #     size <= 2**15
    # ), "simd size is too large and must be less than 2^15"


@always_inline("nodebug")
def _has_native_bf16_support() -> Bool:
    return is_gpu()


@always_inline("nodebug")
def _has_native_f8_support() -> Bool:
    return _is_sm_9x_or_newer() or is_nvidia_gpu["sm_89"]() or is_amd_gpu()


# Apple's Metal AIR backend has no `llvm.vector.splice` lowering;
# `SIMD.{rotate,shift}_{left,right}` use `shufflevector` masks instead.


@always_inline("nodebug")
def _apple_rotate_mask[size: Int, shift: Int]() -> IndexList[size]:
    """Mask for `SIMD.rotate_left[shift]()` on Apple GPU; any sign of `shift`.
    """
    var res = IndexList[size]()
    comptime for i in range(size):
        res[i] = (i + shift + size) % size
    return res


@always_inline("nodebug")
def _apple_shift_mask[size: Int, shift: Int]() -> IndexList[size]:
    """Mask for `SIMD.shift_{left,right}[shift]()` on Apple GPU.

    Sign of `shift` selects direction (positive=left, negative=right);
    out-of-range lanes index `other`, which callers pass as zero.
    """
    var res = IndexList[size]()
    comptime for i in range(size):
        var src = i + shift
        if 0 <= src < size:
            res[i] = src
        else:
            res[i] = size
    return res


# ===----------------------------------------------------------------------=== #
# FastMathFlag
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct FastMathFlag(Equatable, ImplicitlyCopyable, RegisterPassable):
    """Flags for controlling fast-math optimizations in floating-point operations.

    FastMathFlag provides compile-time controls for various floating-point math
    optimization modes that trade strict IEEE 754 compliance for performance.

    Available flags:
    - `NONE`: No fast-math optimizations.
    - `NNAN`: Assume operands and results are not NaN.
    - `NINF`: Assume operands and results are not +/- infinity.
    - `NSZ`: Treat the sign of a zero as insignificant.
    - `ARCP`: Allow reciprocal of values.
    - `CONTRACT`: Allow floating-point contraction (e.g., fused multiply-add).
    - `AFN`: Allow algebraic function approximations.
    - `REASSOC`: Allow reassociation of floating-point operations.
    - `FAST`: Enable all fast-math optimizations.

    Examples:
        ```mojo
        from std.builtin.simd import FastMathFlag
        var value = Float32(2.0)
        var multiplier = Float32(3.0)
        var accumulator = Float32(1.0)

        # Use contract flag for fused multiply-add
        var result = value.fma[FastMathFlag.CONTRACT](multiplier, accumulator)

        # Use fast flag for maximum optimization
        var fast_result = value.fma[FastMathFlag.FAST](multiplier, accumulator)
        ```
    """

    var _value: UInt8

    comptime NONE = Self(0)
    """No fast-math optimizations enabled."""

    comptime NNAN = Self(1)
    """Assume no NaN values."""

    comptime NINF = Self(2)
    """Assume no infinite values."""

    comptime NSZ = Self(3)
    """Treat the sign of zero as insignificant."""

    comptime ARCP = Self(4)
    """Allow reciprocal approximations."""

    comptime CONTRACT = Self(5)
    """Allow floating-point contraction."""

    comptime AFN = Self(6)
    """Allow approximate function implementations."""

    comptime REASSOC = Self(7)
    """Allow reassociation of operations."""

    comptime FAST = Self(8)
    """Enable all fast-math optimizations."""

    def __eq__(self, other: Self) -> Bool:
        """Compares two FastMathFlag values for identity.

        Args:
            other: The FastMathFlag to compare against.

        Returns:
            True if both flags have the same value, False otherwise.
        """
        return self._value == other._value

    def _mlir_attr(self) -> __mlir_type.`!kgen.deferred`:
        if self == FastMathFlag.NONE:
            return __mlir_attr.`#pop.fmf<none>`
        if self == FastMathFlag.NNAN:
            return __mlir_attr.`#pop.fmf<nnan>`
        if self == FastMathFlag.NINF:
            return __mlir_attr.`#pop.fmf<ninf>`
        if self == FastMathFlag.NSZ:
            return __mlir_attr.`#pop.fmf<nsz>`
        if self == FastMathFlag.ARCP:
            return __mlir_attr.`#pop.fmf<arcp>`
        if self == FastMathFlag.CONTRACT:
            return __mlir_attr.`#pop.fmf<contract>`
        if self == FastMathFlag.AFN:
            return __mlir_attr.`#pop.fmf<afn>`
        if self == FastMathFlag.REASSOC:
            return __mlir_attr.`#pop.fmf<reassoc>`
        if self == FastMathFlag.FAST:
            return __mlir_attr.`#pop.fmf<fast>`

        return __mlir_attr.`#pop.fmf<none>`


# ===----------------------------------------------------------------------=== #
# SIMD
# ===----------------------------------------------------------------------=== #


@lldb_formatter_wrapping_type
struct SIMD[dtype: DType, length: SIMDLength](
    Absable,
    Boolable,
    CeilDivable,
    Ceilable,
    Comparable,
    ConvertibleFromPython,
    CoordLike,
    Defaultable,
    DevicePassable,
    DivModable,
    Equatable,
    Floorable,
    Hashable,
    Indexer,
    Intable,
    IntervalElement,
    Powable,
    Roundable,
    Sized,
    TrivialRegisterPassable,
    Truncable,
    Writable,
    _FromInt,
):
    """Represents a vector type that leverages hardware acceleration to process
    multiple data elements with a single operation.

    SIMD (Single Instruction, Multiple Data) is a fundamental parallel
    computing paradigm where a single CPU instruction operates on multiple data
    elements at once. Modern CPUs can perform 4, 8, 16, or even 32 operations
    in parallel using SIMD, delivering substantial performance improvements
    over scalar operations. Instead of processing one value at a time, SIMD
    processes entire vectors of values with each instruction.

    For example, when adding two vectors of four values, a scalar operation
    adds each value in the vector one by one, while a SIMD operation adds all
    four values at once using vector registers:

    ```text
    Scalar operation:                SIMD operation:
    ┌─────────────────────────┐      ┌───────────────────────────┐
    │ 4 instructions          │      │ 1 instruction             │
    │ 4 clock cycles          │      │ 1 clock cycle             │
    │                         │      │                           │
    │ ADD  a[0], b[0] → c[0]  │      │ Vector register A         │
    │ ADD  a[1], b[1] → c[1]  │      │ ┌─────┬─────┬─────┬─────┐ │
    │ ADD  a[2], b[2] → c[2]  │      │ │a[0] │a[1] │a[2] │a[3] │ │
    │ ADD  a[3], b[3] → c[3]  │      │ └─────┴─────┴─────┴─────┘ │
    └─────────────────────────┘      │           +               │
                                     │ Vector register B         │
                                     │ ┌─────┬─────┬─────┬─────┐ │
                                     │ │b[0] │b[1] │b[2] │b[3] │ │
                                     │ └─────┴─────┴─────┴─────┘ │
                                     │           ↓               │
                                     │        SIMD_ADD           │
                                     │           ↓               │
                                     │ Vector register C         │
                                     │ ┌─────┬─────┬─────┬─────┐ │
                                     │ │c[0] │c[1] │c[2] │c[3] │ │
                                     │ └─────┴─────┴─────┴─────┘ │
                                     └───────────────────────────┘
    ```

    The `SIMD` type maps directly to hardware vector registers and
    instructions. Mojo automatically generates optimal SIMD code that leverages
    CPU-specific instruction sets (such as AVX and NEON) without requiring
    manual intrinsics or assembly programming.

    This type is the foundation of high-performance CPU computing in Mojo,
    enabling you to write code that automatically leverages modern CPU vector
    capabilities while maintaining code clarity and portability.

    **Caution:** If you declare a SIMD vector size larger than the vector
    registers of the target hardware, the compiler will break up the SIMD into
    multiple vector registers for compatibility. However, you should avoid
    using a vector that's more than 2x the hardware's vector register size
    because the resulting code will perform poorly.

    Key properties:

    - **Hardware-mapped**: Directly maps to CPU vector registers
    - **Type-safe**: Data types and vector sizes are checked at compile time
    - **Zero-cost**: No runtime overhead compared to hand-optimized intrinsics
    - **Portable**: Same code works across different CPU architectures
      (x86, ARM, etc.)
    - **Composable**: Seamlessly integrates with Mojo's parallelization features

    Key APIs:

    - Construction:
      - Broadcast single value to all elements: `SIMD[dtype, length](value)`
      - Initialize with specific values: `SIMD[dtype, length](v1, v2, ...)`
      - Zero-initialized vector: `SIMD[dtype, length]()`

    - Element operations:
      - Arithmetic: `+`, `-`, `*`, `/`, `%`, `//`
      - Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
      - Math functions: `sqrt()`, `sin()`, `cos()`, `fma()`, etc.
      - Bit operations: `&`, `|`, `^`, `~`, `<<`, `>>`

    - Vector operations:
      - Horizontal reductions: `reduce_add()`, `reduce_mul()`, `reduce_min()`, `reduce_max()`
      - Element-wise conditional selection: `select(true_case, false_case)`
      - Vector manipulation: `shuffle()`, `slice()`, `join()`, `split()`
      - Type conversion: `cast[target_dtype]()`

    Examples:

    Vectorized math operations:

    ```mojo
    # Process 8 floating-point numbers simultaneously
    var a = SIMD[.float32, 8](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    var b = SIMD[.float32, 8](2.0)  # Broadcast 2.0 to all elements
    var result = a * b + 1.0
    print(result)  # => [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]
    ```

    Conditional operations with masking:

    ```mojo
    # Double the positive values and negate the negative values
    var values = SIMD[.int32, 4](1, -2, 3, -4)
    var is_positive = values.gt(0)  # greater-than: gets SIMD of booleans
    var result = is_positive.select(values * 2, values * -1)
    print(result)  # => [2, 2, 6, 4]
    ```

    Horizontal reductions:

    ```mojo
    # Sum all elements in a vector
    var data = SIMD[.float64, 4](10.5, 20.3, 30.1, 40.7)
    var total = data.reduce_add()
    var maximum = data.reduce_max()
    print(total, maximum)  # => 101.6 40.7
    ```

    Constraints:
        The length of the SIMD vector must be positive and a power of 2.

    Parameters:
        dtype: The data type of SIMD vector elements.
        length: The length of the SIMD vector (number of elements).
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    comptime _mlir_type = __mlir_type[
        `!kgen.simd<`,
        Self.length._mlir_value,
        `, `,
        Self.dtype._mlir_value,
        `>`,
    ]

    var _mlir_value: Self._mlir_type
    """The underlying storage for the vector."""

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime MAX = Self(_max_or_inf[Self.dtype]())
    """Gets the maximum value for the SIMD value, potentially +inf."""

    comptime MIN = Self(_min_or_neg_inf[Self.dtype]())
    """Gets the minimum value for the SIMD value, potentially -inf."""

    comptime MAX_FINITE = Self(_max_finite[Self.dtype]())
    """Returns the maximum finite value of SIMD value."""

    comptime MIN_FINITE = Self(_min_finite[Self.dtype]())
    """Returns the minimum (lowest) finite value of SIMD value."""

    comptime _Mask = SIMD[.bool, Self.length]

    comptime device_type: AnyType = Self
    """SIMD types are remapped to the same type when passed to accelerator devices."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Device type mapping is the identity function."""
        # `where` clause on the conformance would be cleaner but triggers a
        # KGEN parameter-evaluator crash when instantiated in certain scopes.
        comptime assert Self.dtype != DType.int and Self.dtype != DType.uint, (
            "Int and UInt do not conform to DevicePassable; use a "
            "fixed-width type such as Int32 or Int64 instead"
        )
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """
        Gets this type's name, for use in error messages when handing arguments
        to kernels.
        TODO: This will go away soon, when we get better error messages for
        kernel calls.

        Returns:
            This type's name.
        """
        return String(t"SIMD[{repr(Self.dtype)}, {repr(Int(Self.length))}]")

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    def __init__(out self):
        """Default initializer of the SIMD vector.

        By default the SIMD vectors are initialized to all zeros.
        """
        # make sure this constructor is called at compile time
        comptime res = SIMD[Self.dtype, Self.length](0)
        self = res

    # The target dtype is a defaulted parameter rather than `Self.dtype` so that
    # this overload stays out of the way of the `Floatable` constructor below:
    # spelled with `Self.dtype`, `Float64(x)` for an `Intable` and `Floatable`
    # `x` becomes ambiguous.
    @always_inline("nodebug")
    def __init__[
        T: Intable, target_dtype: DType = DType.int
    ](out self: Scalar[target_dtype], value: T):
        """Initialize an integer scalar from an intable value.

        Parameters:
            T: The Intable type.
            target_dtype: The dtype of the scalar to construct.

        Args:
            value: The value to initialize from.

        Constraints:
            The target dtype must be integral.

        Example:

        ```mojo
        var x = 42
        var p = Pointer(to=x)
        print(UInt(p))  # the address of `x`
        ```
        """
        comptime assert (
            target_dtype.is_integral()
        ), "constructing from an `Intable` value requires an integral dtype"
        self = Scalar[target_dtype](value.__int__())

    @always_inline("nodebug")
    def __init__[T: IntableRaising](out self: Int, value: T) raises:
        """Initialize from a raising intable value.

        Parameters:
            T: The IntableRaising type.

        Args:
            value: The value to initialize from.

        Raises:
            Any errors from the conversion to Int.
        """
        self = value.__int__()

    @always_inline("nodebug")
    def __init__[
        other_dtype: DType, //
    ](out self, value: SIMD[other_dtype, Self.length], /):
        """Initialize from another SIMD of the same size. If the value
        passed is a scalar, you can initialize a SIMD vector with more elements.

        Parameters:
            other_dtype: The type of the value that is being cast from.

        Args:
            value: The value to cast from.

        Example:

        ```mojo
        print(UInt64(UInt8(42))) # 42
        print(SIMD[.uint64, 4](UInt8(42))) # [42, 42, 42, 42]
        ```

        Casting behavior:

        ```mojo
        # Basic casting preserves value within range
        Int8(UInt8(127)) == Int8(127)

        # Numbers above signed max wrap to negative using two's complement
        Int8(UInt8(128)) == Int8(-128)
        Int8(UInt8(129)) == Int8(-127)
        Int8(UInt8(256)) == Int8(0)

        # Negative signed cast to unsigned using two's complement
        UInt8(Int8(-128)) == UInt8(128)
        UInt8(Int8(-127)) == UInt8(129)
        UInt8(Int8(-1)) == UInt8(255)

        # Truncate precision after downcast and upcast
        Float64(Float32(Float64(123456789.123456789))) == Float64(123456792.0)

        # Rightmost bits of significand become 0's on upcast
        Float64(Float32(0.3)) == Float64(0.30000001192092896)

        # Numbers equal after truncation of float literal and cast truncation
        Float32(Float64(123456789.123456789)) == Float32(123456789.123456789)

        # Float to int/uint floors
        Int64(Float64(42.2)) == Int64(42)
        ```
        """
        self = value.cast[Self.dtype]()

    @always_inline("builtin")
    @implicit
    def __init__(out self, value: SIMDLength, /):
        """Initializes the SIMD vector with a signed integer.

        The signed integer value is splatted across all the elements of the SIMD
        vector.

        Args:
            value: The input value.
        """
        _simd_construction_checks[Self.dtype, Self.length]()

        self._mlir_value = __mlir_op.`pop.simd.splat`[_type=Self._mlir_type](
            __mlir_op.`pop.cast`[_type=Scalar[Self.dtype]._mlir_type](
                __mlir_op.`pop.cast_from_builtin`[
                    _type=__mlir_type.`!kgen.scalar<index>`
                ](value._mlir_value)
            )
        )

    @doc_hidden
    @always_inline("builtin")
    def __init__(out self, *, from_int: Int):
        _simd_construction_checks[Self.dtype, Self.length]()
        self = Self(from_int)

    @always_inline
    def __init__[T: Floatable, //](out self: Float64, value: T, /):
        """Initialize a Float64 from a type conforming to Floatable.

        Parameters:
            T: The Floatable type.

        Args:
            value: The object to get the float point representation of.
        """
        self = value.__float__()

    @always_inline
    def __init__[
        T: FloatableRaising, //
    ](out self: Float64, value: T, /) raises:
        """Initialize a Float64 from a type conforming to FloatableRaising.

        Parameters:
            T: The FloatableRaising type.

        Args:
            value: The object to get the float point representation of.

        Raises:
            If the type does not have a float point representation.
        """
        self = value.__float__()

    @always_inline("builtin")
    @implicit
    def __init__(out self, value: IntLiteral, /):
        """Initializes the SIMD vector with an integer.

        The integer value is splatted across all the elements of the SIMD
        vector.

        Args:
            value: The input value.
        """
        _simd_construction_checks[Self.dtype, Self.length]()
        self._mlir_value = __mlir_attr[
            `#pop.int_literal_convert<`,
            value.value,
            `> : `,
            Self._mlir_type,
        ]

    @always_inline("nodebug")
    @implicit
    def __init__(out self: SIMD[.bool, Self.length], value: Bool, /):
        """Initializes a Scalar with a bool value.

        Since this constructor does not splat, it can be implicit.

        Args:
            value: The bool value to initialize the Scalar with.
        """

        # NOTE: due to some issues with the out Self parameter not always being
        # respected (i.e. through implicit conversion paths), it's better to do
        # this check instead of constraining the signature, because otherwise
        # the error would point to a type mismatch. All this should be fixed by
        # using a requires clause when it becomes available.
        comptime assert Self.length == 1, (
            "must be a scalar; use the `fill` keyword instead for explicit"
            " splatting"
        )

        _simd_construction_checks[Self.dtype, Self.length]()
        self._mlir_value = rebind[Self._Mask._mlir_type](value._mlir_value)

    @always_inline("nodebug")
    def __init__(out self: SIMD[.bool, Self.length], *, fill: Bool):
        """Initializes the SIMD vector with a bool value.

        The bool value is splatted across all elements of the SIMD vector.

        Args:
            fill: The bool value to fill each element of the SIMD vector with.
        """
        _simd_construction_checks[Self.dtype, Self.length]()
        self._mlir_value = __mlir_op.`pop.simd.splat`[
            _type=Self._Mask._mlir_type
        ](fill._mlir_value)

    @doc_hidden
    @always_inline("builtin")
    def __init__(out self, *, mlir_value: Self._mlir_type):
        """Initializes the SIMD vector with the underlying mlir value.

        Args:
            mlir_value: The input value.
        """
        _simd_construction_checks[Self.dtype, Self.length]()
        self._mlir_value = mlir_value

    @doc_hidden
    @always_inline("builtin")
    def __init__(out self: Int, *, mlir_value: __mlir_type.index):
        self._mlir_value = __mlir_op.`pop.cast_from_builtin`[
            _type=__mlir_type.`!kgen.scalar<index>`
        ](mlir_value)

    @always_inline("builtin")
    @implicit
    def __init__(out self, value: Scalar[Self.dtype], /):
        """Constructs a SIMD vector by splatting a scalar value.

        The input value is splatted across all elements of the SIMD vector.

        Args:
            value: The value to splat to the elements of the vector.
        """
        _simd_construction_checks[Self.dtype, Self.length]()
        self._mlir_value = __mlir_op.`pop.simd.splat`[_type=Self._mlir_type](
            value._mlir_value
        )

    @always_inline("nodebug")
    def __init__(
        out self, *elems: Scalar[Self.dtype], __list_literal__: NoneType = None
    ):
        """Constructs a SIMD vector via a variadic list of elements.

        The input values are assigned to the corresponding elements of the SIMD
        vector.

        Constraints:
            The number of input values is equal to size of the SIMD vector.

        Args:
            elems: The variadic list of elements from which the SIMD vector is
                   constructed.
            __list_literal__: Tell Mojo to use this method for list literals.
        """
        _simd_construction_checks[Self.dtype, Self.length]()

        # TODO: Make this a compile-time check when possible.
        assert Self.length == len(
            elems
        ), "mismatch in the number of elements in the SIMD variadic constructor"

        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

        comptime for i in range(Self.length):
            self[i] = elems[i]

    # TODO: should be "builtin" when constrained is replaced with 'requires'.
    @always_inline("nodebug")
    @implicit
    def __init__(out self, value: FloatLiteral, /):
        """Initializes the SIMD vector with a float.

        The value is splatted across all the elements of the SIMD
        vector.

        Args:
            value: The input value.
        """
        _simd_construction_checks[Self.dtype, Self.length]()
        comptime assert (
            Self.dtype.is_floating_point()
        ), "the SIMD type must be floating point"
        var res = __mlir_attr[
            `#pop<float_literal_convert<`, value.value, `>> : `, Self._mlir_type
        ]
        self = Self(mlir_value=res)

    @staticmethod
    def __init__[
        int_dtype: DType, //
    ](out self, *, from_bits: SIMD[int_dtype, Self.length]):
        """Initializes the SIMD vector from the bits of an integral SIMD vector.

        Parameters:
            int_dtype: The integral type of the input SIMD vector.

        Args:
            from_bits: The SIMD vector to copy the bits from.
        """
        comptime assert (
            int_dtype.is_integral()
        ), "the SIMD type must be integral"

        comptime if Self.dtype == DType.bool and int_dtype in (
            DType.uint8,
            DType.int8,
        ):
            self = from_bits.ne(0)._refine[Self.dtype]()
        else:
            self = bitcast[Self.dtype, Self.length](from_bits)

    @always_inline
    def __init__(out self: Self, *, py: PythonObject) raises:
        """Initialize a SIMD value from a PythonObject.

        Args:
            py: The PythonObject to convert.

        Raises:
            If the conversion to double fails, or if the value is out of range
            for an unsigned dtype (a negative or too-large Python int).
        """

        comptime if Self.dtype.is_floating_point():
            ref cpy = Python().cpython()
            var float_value = cpy.PyFloat_AsDouble(py._obj_ptr)
            if float_value == -1.0 and cpy.PyErr_Occurred():
                # Note that -1.0 does not guarantee an error, it just means we
                # need to check if there was an exception.
                raise cpy.unsafe_get_error()
            # NOTE: if dtype is not float64, we truncate.
            self = Scalar[Self.dtype](float_value)
        elif Self.dtype.is_integral() and bit_width_of[Self.dtype]() <= 64:
            comptime if Self.dtype.is_unsigned():
                # Read unsigned dtypes through the unsigned entry point so that
                # values in `[2**63, 2**64)` round-trip (the signed path
                # overflows on them) and negative Python ints raise instead of
                # silently wrapping. Mirrors the Mojo -> Python direction, which
                # already uses the unsigned `PyLong_FromSize_t`.
                self = Scalar[Self.dtype](
                    Python.py_long_as_size_t(py.__int__())
                )
            else:
                self = Scalar[Self.dtype](
                    Python.py_long_as_ssize_t(py.__int__())
                )
        else:
            self = Scalar[Self.dtype]()
            comptime assert False, "unsupported dtype"

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    def __getitem__(self, idx: Int) -> Scalar[Self.dtype]:
        """Gets an element from the vector.

        Args:
            idx: The element index.

        Returns:
            The value at position `idx`.
        """
        return Scalar[Self.dtype](
            mlir_value=__mlir_op.`pop.simd.extractelement`(
                self._mlir_value, idx.__mlir_index__()
            )
        )

    @always_inline("nodebug")
    def __setitem__(mut self, idx: Int, val: Scalar[Self.dtype]):
        """Sets an element in the vector.

        Args:
            idx: The index to set.
            val: The value to set.
        """
        self._mlir_value = __mlir_op.`pop.simd.insertelement`(
            self._mlir_value, val._mlir_value, idx.__mlir_index__()
        )

    def __contains__(self, value: Scalar[Self.dtype]) -> Bool:
        """Whether the vector contains the value.

        Args:
            value: The value.

        Returns:
            Whether the vector contains the value.
        """
        return self.eq(value).reduce_or()

    @always_inline("builtin")
    def __add__(self, rhs: Self) -> Self:
        """Computes `self + rhs`.

        Args:
            rhs: The rhs value.

        Returns:
            A new vector whose element at position `i` is computed as
            `self[i] + rhs[i]`.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        return Self(
            mlir_value=__mlir_op.`pop.add`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __sub__(self, rhs: Self) -> Self:
        """Computes `self - rhs`.

        Args:
            rhs: The rhs value.

        Returns:
            A new vector whose element at position `i` is computed as
            `self[i] - rhs[i]`.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        return Self(
            mlir_value=__mlir_op.`pop.sub`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __mul__(self, rhs: Self) -> Self:
        """Computes `self * rhs`.

        Args:
            rhs: The rhs value.

        Returns:
            A new vector whose element at position `i` is computed as
            `self[i] * rhs[i]`.
        """

        return Self(
            mlir_value=__mlir_op.`pop.mul`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __truediv__(self, rhs: Self) -> Self:
        """Computes `self / rhs`.

        Args:
            rhs: The rhs value.

        Returns:
            A new vector whose element at position `i` is computed as
            `self[i] / rhs[i]`.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        return Self(
            mlir_value=__mlir_op.`pop.div`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("nodebug")
    def __floordiv__(self, rhs: Self) -> Self:
        """Returns the division of self and rhs rounded down to the nearest
        integer.

        Constraints:
            The element type of the SIMD vector must be numeric.

        Args:
            rhs: The value to divide with.

        Returns:
            `floor(self / rhs)` value.
        """
        comptime assert Self.dtype.is_numeric(), "the type must be numeric"

        var is_zero_mask = Self._Mask(fill=Self.dtype.is_integral()) and rhs.eq(
            0
        )
        var safe_divisor = is_zero_mask.select(Self(1), rhs)

        var floordiv = Self(
            mlir_value=__mlir_op.`pop.floordiv`(
                self._mlir_value, safe_divisor._mlir_value
            )
        )
        return is_zero_mask.select(Self(), floordiv)

    @always_inline("nodebug")
    def __mod__(self, rhs: Self) -> Self:
        """Returns the remainder of self divided by rhs.

        Args:
            rhs: The value to divide with.

        Returns:
            The remainder of dividing self by rhs.
        """
        comptime assert Self.dtype.is_numeric(), "the type must be numeric"

        var is_zero_mask = Self._Mask(fill=Self.dtype.is_integral()) and rhs.eq(
            0
        )
        var safe_divisor = is_zero_mask.select(Self(1), rhs)

        comptime if Self.dtype.is_unsigned():
            var rem = Self(
                mlir_value=__mlir_op.`pop.rem`(
                    self._mlir_value, safe_divisor._mlir_value
                )
            )
            return is_zero_mask.select(Self(), rem)
        else:
            var div = self / safe_divisor

            comptime if Self.dtype.is_floating_point():
                div = trunc(div)

            var mod = self - div * rhs
            var mask = (rhs.lt(0) ^ self.lt(0)) & mod.ne(0)
            var mod_result = mod + mask.select(rhs, Self(0))
            return is_zero_mask.select(Self(), mod_result)

    @always_inline("nodebug")
    def __divmod__(self, denominator: Self) -> Tuple[Self, Self]:
        """Computes both the quotient and remainder using floor division.

        Args:
            denominator: The value to divide on.

        Returns:
            The quotient and remainder as a
            `Tuple(self // denominator, self % denominator)`.
        """
        var is_zero_mask = denominator.eq(0)
        var safe_denominator = is_zero_mask.select(Self(1), denominator)

        comptime if Self.dtype.is_unsigned():
            var div = self // safe_denominator
            var mod = self % safe_denominator
            return is_zero_mask.select(Self(0), div), is_zero_mask.select(
                Self(0), mod
            )

        var div = self / safe_denominator

        comptime if Self.dtype.is_floating_point():
            div = trunc(div)

        var mod = self - div * denominator
        var mask = (denominator.lt(0) ^ self.lt(0)) & mod.ne(0)

        if any(mask):
            div = div - mask.cast[Self.dtype]()

        var div_res = is_zero_mask.select(Self(0), div)
        var mod_res = mod + mask.select(denominator, Self(0))
        return div_res, is_zero_mask.select(Self(0), mod_res)

    @always_inline("nodebug")
    def __pow__(self, exp: SIMD[_, _]) -> Self:
        """Computes the vector raised to the power of the input integer value.

        Args:
            exp: The exponent value.

        Returns:
            A SIMD vector where each element is raised to the power of the
            specified exponent value.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        comptime assert (
            Self.length == exp.length and Self.dtype == exp.dtype
        ) or exp.length == 1, (
            "The SIMD types must be identical or else the exponent must be a"
            " scalar."
        )
        comptime if Self.length == exp.length and self.dtype == exp.dtype:
            return _pow(self, rebind[Self](exp))
        else:
            return _pow(
                self,
                SIMD[exp.dtype, self.length](rebind[Scalar[exp.dtype]](exp)),
            )

    @always_inline("nodebug")
    def __pow__(self, exp: FloatLiteral) -> Self:
        """Computes the vector raised to the power of a float literal.

        The literal is converted to `Self`, so the exponent always has the
        same dtype as the base.

        Constraints:
            The SIMD dtype must be floating point.

        Args:
            exp: The exponent value.

        Returns:
            A SIMD vector where each element is raised to the power of the
            specified exponent value.
        """
        comptime assert (
            Self.dtype.is_floating_point()
        ), "a float literal exponent requires a floating point SIMD type"
        return _pow(self, Self(exp))

    # TODO(#22771): remove this overload.
    @always_inline("nodebug")
    def __pow__(self, exp: Self) -> Self:
        """Computes the vector raised elementwise to the right hand side power.

        Args:
            exp: The exponent value.

        Returns:
            A SIMD vector where each element is raised to the power of the
            specified exponent value.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        return _pow(self, exp)

    @always_inline("builtin")
    def __pos__(self) -> Self:
        """Defines the unary `+` operation.

        Returns:
            This SIMD vector.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        return self

    @always_inline("builtin")
    def __neg__(self) -> Self:
        """Defines the unary `-` operation.

        Returns:
            The negation of this SIMD vector.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        return Self(mlir_value=__mlir_op.`pop.neg`(self._mlir_value))

    @always_inline("builtin")
    def __and__(self, rhs: Self) -> Self:
        """Returns `self & rhs`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.

        Returns:
            `self & rhs`.
        """
        comptime assert (
            Self.dtype.is_integral() or Self.dtype == DType.bool
        ), "must be an integral or bool type"
        return Self(
            mlir_value=__mlir_op.`pop.simd.and`(
                self._mlir_value, rhs._mlir_value
            )
        )

    @always_inline("builtin")
    def __xor__(self, rhs: Self) -> Self:
        """Returns `self ^ rhs`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.

        Returns:
            `self ^ rhs`.
        """
        comptime assert (
            Self.dtype.is_integral() or Self.dtype == DType.bool
        ), "must be an integral or bool type"
        return Self(
            mlir_value=__mlir_op.`pop.simd.xor`(
                self._mlir_value, rhs._mlir_value
            )
        )

    @always_inline("builtin")
    def __or__(self, rhs: Self) -> Self:
        """Returns `self | rhs`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.

        Returns:
            `self | rhs`.
        """
        comptime assert (
            Self.dtype.is_integral() or Self.dtype == DType.bool
        ), "must be an integral or bool type"
        return Self(
            mlir_value=__mlir_op.`pop.simd.or`(
                self._mlir_value, rhs._mlir_value
            )
        )

    @always_inline("builtin")
    def __lshift__(self, rhs: Self) -> Self:
        """Returns `self << rhs`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            rhs: The RHS value.

        Returns:
            `self << rhs`.
        """
        comptime assert Self.dtype.is_integral(), "must be an integral type"
        return Self(
            mlir_value=__mlir_op.`pop.shl`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __rshift__(self, rhs: Self) -> Self:
        """Returns `self >> rhs`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            rhs: The RHS value.

        Returns:
            `self >> rhs`.
        """
        comptime assert Self.dtype.is_integral(), "must be an integral type"
        return Self(
            mlir_value=__mlir_op.`pop.shr`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __invert__(self) -> Self:
        """Returns `~self`.

        Constraints:
            The element type of the SIMD vector must be boolean or integral.

        Returns:
            The `~self` value.
        """
        comptime assert (
            Self.dtype.is_integral() or Self.dtype == DType.bool
        ), "must be an integral or bool type"

        return self ^ -1

    # ===------------------------------------------------------------------=== #
    # Boolean comparison operations.
    # ===------------------------------------------------------------------=== #

    @always_inline("builtin")
    def __eq__(self, rhs: Self) -> Bool:
        """Compares two SIMD vectors for equality.

        Args:
            rhs: The SIMD vector to compare with.

        Returns:
            True if all elements of the SIMD vectors are equal, False otherwise.
        """
        return Bool(
            mlir_value=__mlir_op.`pop.simd.reduce_and`(self.eq(rhs)._mlir_value)
        )

    @always_inline("builtin")
    def __ne__(self, rhs: Self) -> Bool:
        """Compares two SIMD vectors for inequality.

        Args:
            rhs: The SIMD vector to compare with.

        Returns:
            True if any elements of the SIMD vectors are not equal, False
            otherwise.
        """
        return not self == rhs

    @always_inline("builtin")
    def __gt__(self, rhs: Self) -> Bool:
        """Compares two Scalars using greater-than comparison.

        Args:
            rhs: The Scalar to compare with.

        Returns:
            True if `self` is greater than `rhs`, False otherwise.
        """
        comptime assert Self.length == 1, (
            "Strict inequality is only defined for `Scalar`s; "
            "did you mean to use `SIMD.gt(...)`?"
        )
        return self.gt(rhs).__bool__()

    @always_inline("builtin")
    def __ge__(self, rhs: Self) -> Bool:
        """Compares two Scalars using greater-than-or-equal comparison.

        Args:
            rhs: The Scalar to compare with.

        Returns:
            True if `self` is greater than or equal to `rhs`, False otherwise.
        """
        comptime assert Self.length == 1, (
            "Greater than or equal is only defined for `Scalar`s; "
            "did you mean to use `SIMD.ge(...)`?"
        )
        return self.ge(rhs).__bool__()

    @always_inline("builtin")
    def __lt__(self, rhs: Self) -> Bool:
        """Compares two Scalars using less-than comparison.

        Args:
            rhs: The Scalar to compare with.

        Returns:
            True if `self` is less than `rhs`, False otherwise.
        """
        comptime assert Self.length == 1, (
            "Strict inequality is only defined for `Scalar`s; "
            "did you mean to use `SIMD.lt(...)`?"
        )
        return self.lt(rhs).__bool__()

    @always_inline("builtin")
    def __le__(self, rhs: Self) -> Bool:
        """Compares two Scalars using less-than-or-equal comparison.

        Args:
            rhs: The Scalar to compare with.

        Returns:
            True if `self` is less than or equal to `rhs`, False otherwise.
        """
        comptime assert Self.length == 1, (
            "Less than or equal is only defined for `Scalar`s; "
            "did you mean to use `SIMD.le(...)`?"
        )
        return self.le(rhs).__bool__()

    # ===------------------------------------------------------------------=== #
    # Elementwise comparison operations.
    # ===------------------------------------------------------------------=== #

    @always_inline("builtin")
    def eq(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using elementwise equality.

        Args:
            rhs: The SIMD vector to compare with.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is the value of `self[i] == rhs[i]`.
        """

        var res = __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<eq>`](
            self._mlir_value, rhs._mlir_value
        )
        return Self._Mask(mlir_value=res)

    @always_inline("builtin")
    def ne(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using elementwise inequality.

        Args:
            rhs: The SIMD vector to compare with.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is the value of `self[i] != rhs[i]`.
        """

        var res = __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<ne>`](
            self._mlir_value, rhs._mlir_value
        )
        return Self._Mask(mlir_value=res)

    @always_inline("builtin")
    def gt(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using elementwise greater-than comparison.

        Args:
            rhs: The SIMD vector to compare with.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is the value of `self[i] > rhs[i]`.
        """

        var res = __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<gt>`](
            self._mlir_value, rhs._mlir_value
        )
        return Self._Mask(mlir_value=res)

    @always_inline("builtin")
    def ge(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using elementwise greater-than-or-equal
        comparison.

        Args:
            rhs: The SIMD vector to compare with.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is the value of `self[i] >= rhs[i]`.
        """

        var res = __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<ge>`](
            self._mlir_value, rhs._mlir_value
        )
        return Self._Mask(mlir_value=res)

    @always_inline("builtin")
    def lt(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using elementwise less-than comparison.

        Args:
            rhs: The SIMD vector to compare with.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is the value of `self[i] < rhs[i]`.
        """

        var res = __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<lt>`](
            self._mlir_value, rhs._mlir_value
        )
        return Self._Mask(mlir_value=res)

    @always_inline("builtin")
    def le(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using elementwise less-than-or-equal
        comparison.

        Args:
            rhs: The SIMD vector to compare with.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is the value of `self[i] <= rhs[i]`.
        """

        var res = __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<le>`](
            self._mlir_value, rhs._mlir_value
        )
        return Self._Mask(mlir_value=res)

    # ===------------------------------------------------------------------=== #
    # In place operations.
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    def __iadd__(mut self, rhs: Self):
        """Performs in-place addition.

        The vector is mutated where each element at position `i` is computed as
        `self[i] + rhs[i]`.

        Args:
            rhs: The rhs of the addition operation.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        self = self + rhs

    @always_inline("nodebug")
    def __isub__(mut self, rhs: Self):
        """Performs in-place subtraction.

        The vector is mutated where each element at position `i` is computed as
        `self[i] - rhs[i]`.

        Args:
            rhs: The rhs of the operation.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        self = self - rhs

    @always_inline("nodebug")
    def __imul__(mut self, rhs: Self):
        """Performs in-place multiplication.

        The vector is mutated where each element at position `i` is computed as
        `self[i] * rhs[i]`.

        Args:
            rhs: The rhs of the operation.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        self = self * rhs

    @always_inline("nodebug")
    def __itruediv__(mut self, rhs: Self):
        """In-place true divide operator.

        The vector is mutated where each element at position `i` is computed as
        `self[i] / rhs[i]`.

        Args:
            rhs: The rhs of the operation.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        self = self / rhs

    @always_inline("nodebug")
    def __ifloordiv__(mut self, rhs: Self):
        """In-place flood div operator.

        The vector is mutated where each element at position `i` is computed as
        `self[i] // rhs[i]`.

        Args:
            rhs: The rhs of the operation.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        self = self // rhs

    @always_inline("nodebug")
    def __imod__(mut self, rhs: Self):
        """In-place mod operator.

        The vector is mutated where each element at position `i` is computed as
        `self[i] % rhs[i]`.

        Args:
            rhs: The rhs of the operation.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        self = self.__mod__(rhs)

    @always_inline("nodebug")
    def __ipow__(mut self, rhs: Int):
        """In-place pow operator.

        The vector is mutated where each element at position `i` is computed as
        `pow(self[i], rhs)`.

        Args:
            rhs: The rhs of the operation.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        self = self.__pow__(rhs)

    @always_inline("nodebug")
    def __iand__(mut self, rhs: Self):
        """Computes `self & rhs` and save the result in `self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.
        """
        comptime assert (
            Self.dtype.is_integral() or Self.dtype == DType.bool
        ), "must be an integral or bool type"
        self = self & rhs

    @always_inline("nodebug")
    def __ixor__(mut self, rhs: Self):
        """Computes `self ^ rhs` and save the result in `self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.
        """
        comptime assert (
            Self.dtype.is_integral() or Self.dtype == DType.bool
        ), "must be an integral or bool type"
        self = self ^ rhs

    @always_inline("nodebug")
    def __ior__(mut self, rhs: Self):
        """Computes `self | rhs` and save the result in `self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.
        """
        comptime assert (
            Self.dtype.is_integral() or Self.dtype == DType.bool
        ), "must be an integral or bool type"
        self = self | rhs

    @always_inline("nodebug")
    def __ilshift__(mut self, rhs: Self):
        """Computes `self << rhs` and save the result in `self`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            rhs: The RHS value.
        """
        comptime assert Self.dtype.is_integral(), "must be an integral type"
        self = self << rhs

    @always_inline("nodebug")
    def __irshift__(mut self, rhs: Self):
        """Computes `self >> rhs` and save the result in `self`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            rhs: The RHS value.
        """
        comptime assert Self.dtype.is_integral(), "must be an integral type"
        self = self >> rhs

    # ===------------------------------------------------------------------=== #
    # Reversed operations
    # ===------------------------------------------------------------------=== #

    @always_inline("builtin")
    def __radd__(self, value: Self) -> Self:
        """Returns `value + self`.

        Args:
            value: The other value.

        Returns:
            `value + self`.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        return value + self

    @always_inline("builtin")
    def __rsub__(self, value: Self) -> Self:
        """Returns `value - self`.

        Args:
            value: The other value.

        Returns:
            `value - self`.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        return value - self

    @always_inline("builtin")
    def __rmul__(self, value: Self) -> Self:
        """Returns `value * self`.

        Args:
            value: The other value.

        Returns:
            `value * self`.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        return value * self

    @always_inline("nodebug")
    def __rfloordiv__(self, rhs: Self) -> Self:
        """Returns the division of rhs and self rounded down to the nearest
        integer.

        Constraints:
            The element type of the SIMD vector must be numeric.

        Args:
            rhs: The value to divide by self.

        Returns:
            `floor(rhs / self)` value.
        """
        comptime assert Self.dtype.is_numeric(), "the type must be numeric"
        return rhs // self

    @always_inline("nodebug")
    def __rtruediv__(self, value: Self) -> Self:
        """Returns `value / self`.

        Args:
            value: The other value.

        Returns:
            `value / self`.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"
        return value / self

    @always_inline("nodebug")
    def __rmod__(self, value: Self) -> Self:
        """Returns `value mod self`.

        Args:
            value: The other value.

        Returns:
            `value mod self`.
        """
        comptime assert Self.dtype.is_numeric(), "the type must be numeric"
        return value % self

    @always_inline("nodebug")
    def __rpow__(self, base: Self) -> Self:
        """Returns `base ** self`.

        Args:
            base: The base value.

        Returns:
            `base ** self`.
        """
        comptime assert Self.dtype.is_numeric(), "the type must be numeric"
        return base**self

    @always_inline("builtin")
    def __rand__(self, value: Self) -> Self:
        """Returns `value & self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            value: The other value.

        Returns:
            `value & self`.
        """
        comptime assert (
            Self.dtype.is_integral() or Self.dtype == DType.bool
        ), "must be an integral or bool type"
        return value & self

    @always_inline("builtin")
    def __rxor__(self, value: Self) -> Self:
        """Returns `value ^ self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            value: The other value.

        Returns:
            `value ^ self`.
        """
        comptime assert (
            Self.dtype.is_integral() or Self.dtype == DType.bool
        ), "must be an integral or bool type"
        return value ^ self

    @always_inline("builtin")
    def __ror__(self, value: Self) -> Self:
        """Returns `value | self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            value: The other value.

        Returns:
            `value | self`.
        """
        comptime assert (
            Self.dtype.is_integral() or Self.dtype == DType.bool
        ), "must be an integral or bool type"
        return value | self

    @always_inline("builtin")
    def __rlshift__(self, value: Self) -> Self:
        """Returns `value << self`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            value: The other value.

        Returns:
            `value << self`.
        """
        comptime assert Self.dtype.is_integral(), "must be an integral type"
        return value << self

    @always_inline("builtin")
    def __rrshift__(self, value: Self) -> Self:
        """Returns `value >> self`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            value: The other value.

        Returns:
            `value >> self`.
        """
        comptime assert Self.dtype.is_integral(), "must be an integral type"
        return value >> self

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    def __len__(self) -> Int:
        """Gets the length of the SIMD vector.

        Returns:
            The length of the SIMD vector.
        """

        return self.length

    @always_inline("builtin")
    def __bool__(self) -> Bool:
        """Converts the SIMD scalar into a boolean value.

        Returns:
            True if the SIMD scalar is non-zero and False otherwise.
        """

        var ne_zero = __mlir_op.`pop.cmp`[
            pred=__mlir_attr.`#kgen.cmp_pred<ne>`
        ](self._mlir_value, Self(0)._mlir_value)
        return Bool(mlir_value=__mlir_op.`pop.simd.reduce_or`(ne_zero))

    @always_inline("nodebug")
    def __int__(self) -> Int:
        """Casts to the value to an Int. If there is a fractional component,
        then the fractional part is truncated.

        Constraints:
            The size of the SIMD vector must be 1.

        Returns:
            The value as an integer.
        """
        comptime assert Int(Self.length) == 1, "expected a scalar type"

        comptime int_width = bit_width_of[Int]()
        comptime type_width = bit_width_of[Self.dtype]()

        comptime if Self.dtype.is_unsigned() and int_width > type_width:
            # If we are casting up, prevent sign extension by first casting to
            # a large unsigned
            return self.cast[_uint_type_of_width[int_width]()]().__int__()
        else:
            return self._refine[new_size=1]().cast[DType.int]()

    @doc_hidden
    @always_inline("builtin")
    def __mlir_index__(self) -> __mlir_type.index:
        """Convert to index.

        Returns:
            The corresponding __mlir_type.index value.
        """
        comptime assert (
            Self.dtype.is_integral()
        ), "cannot index using a floating point type"
        comptime assert Self.length == SIMDLength(
            1
        ), "cannot index using a non-scalar SIMD"

        return __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.index](
            __mlir_op.`pop.cast`[
                _type=SIMD[.int, 1]._mlir_type,
                fastmathFlags=__mlir_attr.`#pop.fmf<fast>`,
            ](rebind[SIMD[Self.dtype, SIMDLength(1)]](self)._mlir_value)
        )

    @always_inline("nodebug")
    def __float__(self) -> Float64:
        """Casts the value to a float.

        Constraints:
            The size of the SIMD vector must be 1.

        Returns:
            The value as a float.
        """
        comptime assert Self.length == 1, "expected a scalar type"
        return self._refine[new_size=1]().cast[DType.float64]()

    @always_inline("builtin")
    def __floor__(self) -> Self:
        """Performs elementwise floor on the elements of a SIMD vector.

        Returns:
            The elementwise floor of this SIMD vector.
        """
        return Self(mlir_value=__mlir_op.`pop.floor`(self._mlir_value))

    @always_inline("builtin")
    def __ceil__(self) -> Self:
        """Performs elementwise ceiling on the elements of a SIMD vector.

        Returns:
            The elementwise ceiling of this SIMD vector.
        """
        return Self(mlir_value=__mlir_op.`pop.ceil`(self._mlir_value))

    @always_inline("builtin")
    def __trunc__(self) -> Self:
        """Performs elementwise truncation on the elements of a SIMD vector.

        Returns:
            The elementwise truncated values of this SIMD vector.
        """
        return Self(mlir_value=__mlir_op.`pop.trunc`(self._mlir_value))

    @always_inline("builtin")
    def __abs__(self) -> Self:
        """Defines the absolute value operation.

        For signed integral element types, the absolute value of the minimum
        representable value is the minimum value itself.

        Returns:
            The absolute value of this SIMD vector.
        """
        return Self(mlir_value=__mlir_op.`pop.abs`(self._mlir_value))

    @always_inline("builtin")
    def __round__(self) -> Self:
        """Performs elementwise rounding on the elements of a SIMD vector.

        This rounding goes to the nearest integer with ties towards the nearest
        even value ("banker's rounding"). This is the default rounding mode for
        binary floating point in the IEEE 754 Standard for Floating Point
        Arithmetic.

        Returns:
            The elementwise rounded value of this SIMD vector.
        """
        return Self(mlir_value=__mlir_op.`pop.round`(self._mlir_value))

    @always_inline("nodebug")
    def __round__(self, ndigits: Int) -> Self:
        """Performs elementwise rounding on the elements of a SIMD vector.

        This rounding goes to the nearest integer with ties towards the nearest
        even value ("banker's rounding"). This is the default rounding mode for
        binary floating point in the IEEE 754 Standard for Floating Point
        Arithmetic.

        Args:
            ndigits: The number of digits to round to.

        Returns:
            The elementwise rounded value of this SIMD vector.
        """

        comptime if Self.dtype == DType.bool:
            return self

        comptime if Self.dtype.is_integral():
            if ndigits >= 0:
                return self
            return self - (self % Self(10) ** -(ndigits))

        var exp = Self(10) ** ndigits
        return (self * exp).__round__() / exp

    def __hash__[H: Hasher](self, mut hasher: H):
        """Updates hasher with this SIMD value.

        For a floating-point vector, `-0.0` hashes as `0.0`, since the two
        compare equal.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """

        # `float8_e8m0fnu` encodes no zero at all, so asking for the bit
        # pattern of one is a compile error rather than a value to fold.
        comptime if (
            Self.dtype.is_floating_point()
            and Self.dtype != DType.float8_e8m0fnu
        ):
            comptime neg_zero = Scalar[Self.dtype](-0.0).to_bits()

            # The `fnuz` encodings have no negative zero, so nothing to fold.
            comptime if neg_zero != 0:
                # Match on the bit pattern rather than the value: a float
                # compare crashes the compiler for every sub-16-bit encoding
                # (MOCO-4680). The hashers start from `to_bits()` either way,
                # so this leaves every other value's hash unchanged.
                var bits = self.to_bits()
                hasher._update_with_simd(
                    bits.eq(type_of(bits)(neg_zero)).select(
                        type_of(bits)(0), bits
                    )
                )
                return

        hasher._update_with_simd(self)

    @always_inline
    def __ceildiv__(self, denominator: Self) -> Self:
        """Return the rounded-up result of dividing self by denominator.


        Args:
            denominator: The denominator.

        Returns:
            The ceiling of dividing numerator by denominator.
        """

        comptime if Self.dtype.is_signed():
            return -(self // -denominator)
        # Biasing the numerator (`self + denominator - 1`) overflows and wraps
        # when `self` is near the unsigned type's max, so correct the floor
        # division with the remainder instead.
        var quotient, remainder = divmod(self, denominator)
        return quotient + remainder.ne(Self(0)).cast[Self.dtype]()

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    @__allow_legacy_custom_self_type
    def _decimal_digit_count(self: Int) -> Int:
        """
        Returns the number of decimal digits required to display this integer.

        Note that if this integer is negative, the returned count does not
        include space to store a leading minus character.

        Returns:
            A count of the number of decimal digits required to display this integer.

        Examples:

        ```mojo
        %# from testing import assert_equal
        assert_equal(Int(10)._decimal_digit_count(), 2)
        assert_equal(Int(-10)._decimal_digit_count(), 2)
        ```
        """

        var n = abs(self)

        comptime if is_32bit():
            return _calc_initial_buffer_size_int32(n)

        # The value only has low-bits.
        if n >> 32 == 0:
            return _calc_initial_buffer_size_int32(n)

        return _calc_initial_buffer_size_int64(UInt64(n))

    @always_inline("nodebug")
    def _refine[
        new_dtype: DType = Self.dtype, new_size: SIMDLength = Self.length
    ](self) -> SIMD[new_dtype, new_size]:
        """Manually refines the SIMD vector to a specific element type and size.

        Parameters:
            new_dtype: The target DType.
            new_size: The target size of the SIMD vector.

        Returns:
            The same SIMD vector with the specified element type and size.
        """
        return rebind[SIMD[new_dtype, new_size]](self)

    @always_inline("nodebug")
    def cast[target: DType](self) -> SIMD[target, Self.length]:
        """Casts the elements of the SIMD vector to the target element type.

        Parameters:
            target: The target DType.

        Returns:
            A new SIMD vector whose elements have been cast to the target
            element type.

        Casting behavior:

        ```mojo
        # Basic casting preserves value within range
        Int8(UInt8(127)) == Int8(127)

        # Numbers above signed max wrap to negative using two's complement
        Int8(UInt8(128)) == Int8(-128)
        Int8(UInt8(129)) == Int8(-127)
        Int8(UInt8(256)) == Int8(0)

        # Negative signed cast to unsigned using two's complement
        UInt8(Int8(-128)) == UInt8(128)
        UInt8(Int8(-127)) == UInt8(129)
        UInt8(Int8(-1)) == UInt8(255)

        # Truncate precision after downcast and upcast
        Float64(Float32(Float64(123456789.123456789))) == Float64(123456792.0)

        # Rightmost bits of significand become 0's on upcast
        Float64(Float32(0.3)) == Float64(0.30000001192092896)

        # Numbers equal after truncation of float literal and cast truncation
        Float32(Float64(123456789.123456789)) == Float32(123456789.123456789)

        # Float to int/uint floors
        Int64(Float64(42.2)) == Int64(42)
        ```
        """

        comptime if Self.dtype == target:
            return self._refine[target]()

        comptime if is_nvidia_gpu():
            comptime if Self.dtype == DType.bfloat16 and target == DType.float64:
                # Convert to F64 via a Float32 pathway. This would allow us to
                # use the optimizations defined above.
                return self.cast[DType.float32]().cast[target]()

        comptime if target in (
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
            DType.float8_e5m2,
            DType.float8_e5m2fnuz,
        ):
            # TODO(KERN-1488): use gpu (H100) instruction to convert from fp16 to fp8
            return _convert_f32_to_float8[target](self.cast[DType.float32]())

        comptime if target == DType.float8_e8m0fnu:
            return _convert_f32_to_float8_ue8m0[target, rounding_mode="rp"](
                self.cast[DType.float32]()
            )

        comptime if Self.dtype in (
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
            DType.float8_e5m2,
            DType.float8_e5m2fnuz,
        ):
            comptime assert target in (
                DType.bfloat16,
                DType.float16,
                DType.float32,
                DType.float64,
            ), String(
                (
                    "Only FP8->F64, FP8->F32, FP8->F16, and FP8->BF16"
                    " castings are implemented. "
                ),
                Self.dtype,
                "->",
                target,
            )

            comptime if target == DType.float16:
                return _convert_float8_to_f16(self).cast[target]()
            return _convert_float8_to_f32(self).cast[target]()

        comptime if Self.dtype == DType.float8_e8m0fnu:
            return _convert_float8_ue8m0_to_f32[DType.float32](self).cast[
                target
            ]()

        comptime if Self.dtype == DType.bool:
            return self.select[target](1, 0)
        elif target == DType.bool:
            return self.ne(0)._refine[target]()

        comptime if Self.dtype == DType.bfloat16 and is_amd_gpu():
            return _bfloat16_to_f32(
                self._refine[DType.bfloat16](),
            ).cast[target]()

        comptime if Self.dtype in (DType._uint1, DType._uint2, DType._uint4):
            # `pop.cast` doesn't support some conversions from `ui1`, `ui2`, or `ui4`
            var uint = __mlir_op.`pop.cast`[
                _type=SIMD[.uint32, Self.length]._mlir_type,
                fastmathFlags=__mlir_attr.`#pop.fmf<fast>`,
            ](self._mlir_value)
            return SIMD[.uint32, Self.length](mlir_value=uint).cast[target]()

        var res = __mlir_op.`pop.cast`[
            _type=SIMD[target, Self.length]._mlir_type,
            fastmathFlags=__mlir_attr.`#pop.fmf<fast>`,
        ](self._mlir_value)
        return SIMD(mlir_value=res)

    @always_inline("builtin")
    def is_power_of_two(self) -> SIMD[.bool, Self.length]:
        """Checks if the input value is a power of 2 for each element of a SIMD vector.

        Constraints:
            The element type of the input vector must be integral.

        Returns:
            A SIMD value where the element at position `i` is True if the integer at
            position `i` of the input value is a power of 2, False otherwise.
        """
        comptime assert Self.dtype.is_integral(), "must be integral"

        return self.gt(0) & (self & (self - 1)).eq(0)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """
        Formats this SIMD value to the provided Writer.

        Args:
            writer: The object to write to.
        """

        # `write_string` rather than `write`: `write` promotes each literal to a
        # `String`, so every literal costs a stack slot, the small-string branch
        # and a refcount decrement, all of which are re-elaborated for every
        # `SIMD[dtype, size]` that gets printed.
        comptime if Self.length > 1:
            writer.write_string("[")

        # Write each element.
        for i in range(Self.length):
            var element = self[i]
            # Write separators between each element.
            if i != 0:
                writer.write_string(", ")
            _write_scalar(writer, element)

        # Write a closing `]`.
        comptime if Self.length > 1:
            writer.write_string("]")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of the SIMD value.

        Args:
            writer: The value to write to.
        """
        comptime scalar_alias = _scalar_repr_alias[Self.dtype]()
        comptime if Self.length == 1 and scalar_alias:
            # Prefer the scalar type alias (e.g. `UInt32(4)`) over the verbose
            # `SIMD[DType.uint32, 1](4)` form for the common scalar case.
            writer.write_string(scalar_alias.value())
            writer.write_string("(")
        else:
            writer.write_string("SIMD[")
            Self.dtype.write_repr_to(writer)
            writer.write_string(", ")
            writer.write(Int(Self.length))
            writer.write_string("](")
        # Write each element.
        for i in range(Self.length):
            var element = self[i]
            # Write separators between each element.
            if i != 0:
                writer.write_string(", ")
            _write_scalar(writer, element)
        writer.write_string(")")

    def write_padded[
        W: Writer
    ](self, mut writer: W, width: Int) where self.dtype.is_integral():
        """Write the integral SIMD with each element right-aligned to a set
        padding. No additional space between elements is inserted.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
            width: The amount to pad to the left.
        """

        # Write an opening `[`.
        comptime if Self.length > 1:
            writer.write_string("[")

        # Write each element.
        for i in range(Self.length):
            var element = self[i]
            # _calc_initial_buffer_size adds an extra 1 for the terminator,
            # which we want to remove, but also doesn't include 1 for a
            # negative sign.
            var int_width = _calc_initial_buffer_size(abs(element)) - (
                1 if element >= 0 else 0
            )

            # Write separators between each element.
            if i != 0:
                writer.write_string(",")

            # TODO: Assumes user wants right-aligned content.
            if int_width < width:
                for _ in range(width - int_width):
                    writer.write_string(" ")

            _write_scalar(writer, element)

        # Write a closing `]`.
        comptime if Self.length > 1:
            writer.write_string("]")

    @always_inline
    def to_bits[
        _dtype: DType = _uint_type_of_width[bit_width_of[Self.dtype]()]()
    ](self) -> SIMD[_dtype, Self.length]:
        """Bitcasts the SIMD vector to an integer SIMD vector.

        Parameters:
            _dtype: The integer type to cast to.

        Returns:
            An integer representation of the floating-point value.
        """
        comptime assert (
            _dtype.is_unsigned()
        ), "the target type must be unsigned integral"
        comptime assert (
            bit_width_of[_dtype]() >= bit_width_of[Self.dtype]()
        ), "the target type must be at least as wide as the source type"

        comptime if Self.dtype == DType.bool:
            return self.cast[DType.uint8]().to_bits[_dtype]()
        else:
            comptime uint = _unsigned_integral_type_of[Self.dtype]()
            return bitcast[uint, Self.length](self).cast[_dtype]()

    @always_inline
    def _to_bits_signed(
        self,
    ) -> SIMD[_integral_type_of[Self.dtype](), Self.length]:
        return bitcast[_integral_type_of[Self.dtype](), Self.length](self)

    @staticmethod
    def from_bytes[
        *,
        big_endian: Bool = is_big_endian(),
    ](bytes: Array[Byte, _]) -> SIMD[Self.dtype, Self.length]:
        """Converts a byte array to a vector.

        Args:
            bytes: The byte array to convert.

        Parameters:
            big_endian: Whether the byte array is big-endian.

        Returns:
            The integer value.
        """
        comptime assert bytes.length == size_of[Self]()
        var ptr = bytes.unsafe_ptr().unsafe_bitcast[Self]()
        var value = ptr[]

        comptime if is_big_endian() != big_endian:
            return byte_swap(value)

        return value

    def as_bytes[
        *,
        big_endian: Bool = is_big_endian(),
    ](self) -> Array[Byte, size_of[Self]()]:
        """Convert the vector to a byte array.

        Parameters:
            big_endian: Whether the byte array should be big-endian.

        Returns:
            The byte array.
        """
        var value = self

        comptime if is_big_endian() != big_endian:
            value = byte_swap(value)

        var ptr = Pointer(to=value)
        var array = Array[Byte, size_of[Self]()](uninitialized=True)
        unsafe_memcpy(
            dest=array.unsafe_ptr(),
            src=ptr.unsafe_bitcast[Byte](),
            count=size_of[Self](),
        )
        return array^

    def clamp(self, lower_bound: Self, upper_bound: Self) -> Self:
        """Clamps the values in a SIMD vector to be in a certain range.

        Clamp cuts values in the input SIMD vector off at the upper bound and
        lower bound values. For example,  SIMD vector `[0, 1, 2, 3]` clamped to
        a lower bound of 1 and an upper bound of 2 would return `[1, 1, 2, 2]`.

        Args:
            lower_bound: Minimum of the range to clamp to.
            upper_bound: Maximum of the range to clamp to.

        Returns:
            A new SIMD vector containing x clamped to be within lower_bound and
            upper_bound.
        """
        return max(min(self, upper_bound), lower_bound)

    # TODO: Move to global function.
    @always_inline("nodebug")
    def fma[
        flag: FastMathFlag = FastMathFlag.CONTRACT
    ](self, multiplier: Self, accumulator: Self) -> Self:
        """Performs a fused multiply-add operation, i.e.
        `self*multiplier + accumulator`.

        Parameters:
            flag: Fast-math optimization flags to apply (default: CONTRACT).

        Args:
            multiplier: The value to multiply.
            accumulator: The value to accumulate.

        Returns:
            A new vector whose element at position `i` is computed as
            `self[i]*multiplier[i] + accumulator[i]`.
        """
        comptime assert Self.dtype.is_numeric(), "the SIMD type must be numeric"

        return Self(
            mlir_value=__mlir_op.`pop.fma`[fastmathFlags=flag._mlir_attr()](
                self._mlir_value,
                multiplier._mlir_value,
                accumulator._mlir_value,
            )
        )

    @always_inline("nodebug")
    def _shuffle_variadic[
        *mask: SIMDLength, output_size: Int = Self.length
    ](self, other: Self) -> SIMD[Self.dtype, output_size]:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask
        values must be within `2 * len(self)`.

        Parameters:
            mask: The permutation to use in the shuffle.
            output_size: The size of the output vector.

        Args:
            other: The other vector to shuffle with.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is `(self + other)[permutation[i]]`.
        """

        comptime assert (
            output_size == mask.size
        ), "size of the mask must match the output SIMD size"

        # FIXME: Support parameters on initializers better, removing __init__.
        comptime tup = StaticTuple[SIMDLength, output_size].__init__[*mask]()
        return self._shuffle_list[output_size, tup](other)

    @always_inline("nodebug")
    def _shuffle_list[
        output_size: SIMDLength, mask: StaticTuple[SIMDLength, output_size]
    ](self, other: Self) -> SIMD[Self.dtype, output_size]:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask
        values must be within `2 * len(self)`.

        Parameters:
            output_size: The output SIMD size.
            mask: The permutation to use in the shuffle.

        Args:
            other: The other vector to shuffle with.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is `(self + other)[permutation[i]]`.
        """

        comptime for i in range(output_size):
            comptime assert (
                0 <= mask[i] < 2 * Self.length
            ), "invalid index in the shuffle operation"

        var res = __mlir_op.`pop.simd.shuffle`[
            mask=mask._mlir_value,
            _type=SIMD[Self.dtype, output_size]._mlir_type,
        ](self._mlir_value, other._mlir_value)
        return SIMD[Self.dtype, output_size](mlir_value=res)

    @always_inline("nodebug")
    def shuffle[*mask: SIMDLength](self) -> Self:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask
        values must be within `2 * len(self)`.

        Parameters:
            mask: The permutation to use in the shuffle.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is `(self)[permutation[i]]`.
        """
        return self._shuffle_variadic[*mask](self)

    @always_inline("nodebug")
    def shuffle[*mask: SIMDLength](self, other: Self) -> Self:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask
        values must be within `2 * len(self)`.

        Parameters:
            mask: The permutation to use in the shuffle.

        Args:
            other: The other vector to shuffle with.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is `(self + other)[permutation[i]]`.
        """
        return self._shuffle_variadic[*mask](other)

    @always_inline("nodebug")
    def shuffle[mask: IndexList[Self.length, element_type=_]](self) -> Self:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask
        values must be within `2 * len(self)`.

        Parameters:
            mask: The permutation to use in the shuffle.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is `(self)[permutation[i]]`.
        """
        return self._shuffle_list[Self.length, mask.as_index_tuple()](self)

    @always_inline("nodebug")
    def shuffle[
        mask: IndexList[Self.length, element_type=_]
    ](self, other: Self) -> Self:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask
        values must be within `2 * len(self)`.

        Parameters:
            mask: The permutation to use in the shuffle.

        Args:
            other: The other vector to shuffle with.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is `(self + other)[permutation[i]]`.
        """
        return self._shuffle_list[Self.length, mask.as_index_tuple()](other)

    # Not an overload of shuffle because there is ambiguity
    # with def shuffle[*mask: Int](self, other: Self) -> Self:
    # TODO: move to the utils directory - see https://github.com/modular/modular/issues/3477
    @always_inline
    def _dynamic_shuffle[
        mask_size: SIMDLength
    ](self, mask: SIMD[.uint8, mask_size]) -> SIMD[Self.dtype, mask_size]:
        """Shuffles (also called blend) the values of the current vector.

        It's done using the specified mask (permutation). The mask
        values must be within `len(self)`. If that's not the case,
        the behavior is undefined.

        The mask is not known at compile time, unlike the `shuffle` method.

        Note that currently, this function is fast only if the following
        conditions are met:
        1) The SIMD vector `self` is of type uint8 and size 16
        2) The CPU supports SSE4 or NEON

        If that's not the case, the function will fallback on a slower path,
        which is an unrolled for loop.

        The pseudocode of this function is:
        ```
        result = SIMD[Self.type, mask_size]()
        for i in range(mask_size):
            result[i] = self[Int(mask[i])]
        ```

        Parameters:
            mask_size: The size of the mask.

        Args:
            mask: The mask to use. Contains the indices to use to shuffle.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is equal to `self[mask[i]]`.
        """

        comptime if (
            # TODO: Allow SSE3 when we have sys.has_sse3()
            (CompilationTarget.has_sse4() or CompilationTarget.has_neon())
            and Self.dtype == DType.uint8
            and Self.length == 16
        ):
            # The instruction works with mask size of 16
            comptime target_mask_size = 16

            # We know that simd sizes are powers of two, so we can use recursivity
            # to iterate on the method until we reach the target size.
            comptime if mask_size < target_mask_size:
                # Make a bigger mask (x2) and retry
                return self._dynamic_shuffle(mask.join({})).slice[mask_size]()
            elif mask_size == target_mask_size:
                return _pshuf_or_tbl1(
                    self._refine[DType.uint8, target_mask_size](),
                    mask._refine[DType.uint8, target_mask_size](),
                )._refine[Self.dtype, mask_size]()
            elif mask_size > target_mask_size:
                # We split it in two and call dynamic_shuffle twice.
                var fst_mask, snd_mask = mask.split()
                var fst = self._dynamic_shuffle(fst_mask)
                var snd = self._dynamic_shuffle(snd_mask)
                return fst.join(snd)._refine[Self.dtype, mask_size]()

        # Slow path, ~3x slower than pshuf for size 16
        var res = SIMD[Self.dtype, mask_size]()

        comptime for i in range(mask_size):
            res[i] = self[Int(mask[i])]
        return res

    @always_inline
    def slice[
        output_width: Int, /, *, offset: Int = 0
    ](self) -> SIMD[Self.dtype, output_width]:
        """Returns a slice of the vector of the specified width with the given
        offset.

        Constraints:
            `output_width + offset` must not exceed the size of this SIMD
            vector.

        Parameters:
            output_width: The output SIMD vector size.
            offset: The given offset for the slice.

        Returns:
            A new vector whose elements map to
            `self[offset:offset+output_width]`.
        """
        comptime assert (
            0 <= offset < output_width + offset <= Self.length
        ), "output width must be a positive integer less than simd size"

        @always_inline
        @__parameter
        def slice_body() -> SIMD[Self.dtype, output_width]:
            var tmp = SIMD[Self.dtype, output_width]()

            comptime for i in range(output_width):
                tmp[i] = self[i + offset]
            return tmp

        comptime if output_width == 1:
            return self[offset]
        elif offset % simd_width_of[Self.dtype]():
            return slice_body()

        if __is_run_in_comptime_interpreter:
            return slice_body()

        comptime if is_apple_gpu():
            return slice_body()

        return llvm_intrinsic[
            "llvm.vector.extract",
            SIMD[Self.dtype, output_width],
            has_side_effect=False,
        ](self, Int64(offset))

    @always_inline("nodebug")
    def insert[*, offset: Int = 0](self, value: SIMD[Self.dtype, _]) -> Self:
        """Returns a new vector where the elements between `offset` and
        `offset + input_width` have been replaced with the elements in `value`.

        Parameters:
            offset: The offset to insert at. This must be a multiple of value's
                    size.

        Args:
            value: The value to be inserted.

        Returns:
            A new vector whose elements at `self[offset:offset+input_width]`
            contain the values of `value`.
        """
        comptime assert (
            offset % value.length == 0
        ), "offset must be a multiple of the subvector's size"

        comptime input_width = value.length
        comptime assert (
            0 <= offset < input_width + offset <= Self.length
        ), "insertion position must not exceed the size of the vector"

        comptime if Self.length == 1:
            comptime assert (
                input_width == 1
            ), "the input width must be 1 if the size is 1"
            return value[0]

        return llvm_intrinsic[
            "llvm.vector.insert", Self, has_side_effect=False
        ](self, value, Int64(offset))

    @always_inline("nodebug")
    def join(self, other: Self) -> SIMD[Self.dtype, 2 * Int(Self.length)]:
        """Concatenates the two vectors together.

        Args:
            other: The other SIMD vector.

        Returns:
            A new vector `self_0, self_1, ..., self_n, other_0, ..., other_n`.
        """

        def indices() -> StaticTuple[SIMDLength, 2 * Int(Self.length)]:
            var res = StaticTuple[SIMDLength, 2 * Int(Self.length)](0)
            for i in range(len(res)):
                res[i] = i
            return res

        return self._shuffle_list[2 * Self.length, indices()](other)

    @always_inline("nodebug")
    def interleave(self, other: Self) -> SIMD[Self.dtype, Int(Self.length) * 2]:
        """Constructs a vector by interleaving two input vectors.

        Args:
            other: The other SIMD vector.

        Returns:
            A new vector `self_0, other_0, ..., self_n, other_n`.
        """

        comptime if Self.length == 1:
            return [self[0], other[0]]

        return llvm_intrinsic[
            "llvm.vector.interleave2",
            SIMD[Self.dtype, Int(Self.length) * 2],
            has_side_effect=False,
        ](self, other)

    @always_inline("nodebug")
    def split(
        self,
    ) -> Tuple[
        SIMD[Self.dtype, Self.length // 2], SIMD[Self.dtype, Self.length // 2]
    ]:
        """Splits the SIMD vector into 2 subvectors.

        Returns:
            A new vector `self_0:N/2, self_N/2:N`.
        """
        comptime assert Self.length > 1, "the simd width must be at least 2"
        comptime half_size = Self.length // 2
        var se = self.slice[half_size]()
        var lf = self.slice[half_size, offset=half_size]()
        return se, lf

    @always_inline("nodebug")
    def deinterleave(
        self,
    ) -> Tuple[
        SIMD[Self.dtype, Self.length / 2], SIMD[Self.dtype, Self.length / 2]
    ]:
        """Constructs two vectors by deinterleaving the even and odd lanes of
        the vector.

        Constraints:
            The vector size must be greater than 1.

        Returns:
            Two vectors the first of the form `self_0, self_2, ..., self_{n-2}`
            and the other being `self_1, self_3, ..., self_{n-1}`.
        """

        comptime assert (
            Self.length > 1
        ), "the vector size must be greater than 1."

        comptime if Self.length == 2:
            return self[0], self[1]

        var res = llvm_intrinsic[
            "llvm.vector.deinterleave2",
            _RegisterPackType[
                SIMD[Self.dtype, Self.length / 2],
                SIMD[Self.dtype, Self.length / 2],
            ],
            has_side_effect=False,
        ](self)
        return res[0], res[1]

    # ===------------------------------------------------------------------=== #
    # Reduce operations
    # ===------------------------------------------------------------------=== #

    comptime _T = SIMD[Self.dtype, _]

    @always_inline
    def reduce[
        func: def[width: Int](Self._T[width], Self._T[width]) thin -> Self._T[
            width
        ],
        size_out: Int = 1,
    ](self) -> Self._T[size_out]:
        """Reduces the vector using a provided reduce operator.

        Parameters:
            func: The reduce function to apply to elements in this SIMD.
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.

        Returns:
            A new scalar which is the reduction of all vector elements.
        """

        @always_inline
        @__parameter
        def body[
            width: SIMDLength
        ](lhs: Self._T[width], rhs: Self._T[width]) -> Self._T[width]:
            return func[width](lhs, rhs)

        return self.reduce[body, size_out]()

    # TODO: remove when non-capturing can be converted to capturing.
    @always_inline
    def reduce[
        func: def[width: SIMDLength](
            Self._T[width], Self._T[width]
        ) thin -> Self._T[width],
        size_out: Int = 1,
    ](self) -> Self._T[size_out]:
        """Reduces the vector using a provided reduce operator.

        Parameters:
            func: The reduce function to apply to elements in this SIMD.
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.

        Returns:
            A new scalar which is the reduction of all vector elements.
        """

        @always_inline
        @__parameter
        def body[w: Int](lhs: Self._T[w], rhs: Self._T[w]) -> Self._T[w]:
            return func(lhs, rhs)

        return self.reduce[body, size_out]()

    @always_inline
    def reduce[
        func: def[width: Int](
            Self._T[width], Self._T[width]
        ) capturing -> Self._T[width],
        size_out: Int = 1,
    ](self) -> Self._T[size_out]:
        """Reduces the vector using a provided reduce operator.

        Parameters:
            func: The reduce function to apply to elements in this SIMD.
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.

        Returns:
            A new scalar which is the reduction of all vector elements.
        """

        @always_inline
        @__parameter
        def body[
            width: SIMDLength
        ](lhs: Self._T[width], rhs: Self._T[width]) -> Self._T[width]:
            return func[width=width](lhs, rhs)

        return self.reduce[body, size_out]()

    @always_inline
    def reduce[
        func: def[width: SIMDLength](
            Self._T[width], Self._T[width]
        ) capturing -> Self._T[width],
        size_out: Int = 1,
    ](self) -> Self._T[size_out]:
        """Reduces the vector using a provided reduce operator.

        Parameters:
            func: The reduce function to apply to elements in this SIMD.
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.

        Returns:
            A new scalar which is the reduction of all vector elements.
        """
        comptime assert (
            size_out <= Self.length
        ), "reduction cannot increase simd width"

        comptime if Self.length == size_out:
            return self._refine[new_size=size_out]()
        else:
            var lhs, rhs = self.split()
            return func(lhs, rhs).reduce[func, size_out]()

    @always_inline("nodebug")
    def reduce_max[size_out: Int = 1](self) -> Self._T[size_out]:
        """Reduces the vector using the `max` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.
            The element type of the vector must be integer or FP.

        Returns:
            The maximum element of the vector.
        """

        comptime if Self.length == 1:
            return self[0]

        comptime if CompilationTarget.is_x86() or size_out > 1:
            return self.reduce[max[dtype=Self.dtype], size_out]()

        comptime if Self.dtype.is_unsigned():
            return llvm_intrinsic[
                "llvm.vector.reduce.umax",
                Scalar[Self.dtype],
                has_side_effect=False,
            ](self)._refine[new_size=size_out]()
        elif Self.dtype.is_integral():
            return llvm_intrinsic[
                "llvm.vector.reduce.smax",
                Scalar[Self.dtype],
                has_side_effect=False,
            ](self)._refine[new_size=size_out]()
        else:
            return llvm_intrinsic[
                "llvm.vector.reduce.fmax",
                Scalar[Self.dtype],
                has_side_effect=False,
            ](self)._refine[new_size=size_out]()

    @always_inline("nodebug")
    def reduce_min[size_out: Int = 1](self) -> SIMD[Self.dtype, size_out]:
        """Reduces the vector using the `min` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.
            The element type of the vector must be integer or FP.

        Returns:
            The minimum element of the vector.
        """

        comptime if Self.length == 1:
            return self[0]

        comptime if CompilationTarget.is_x86() or size_out > 1:
            return self.reduce[min[dtype=Self.dtype], size_out]()

        comptime if Self.dtype.is_unsigned():
            return llvm_intrinsic[
                "llvm.vector.reduce.umin",
                Scalar[Self.dtype],
                has_side_effect=False,
            ](self)._refine[new_size=size_out]()
        elif Self.dtype.is_integral():
            return llvm_intrinsic[
                "llvm.vector.reduce.smin",
                Scalar[Self.dtype],
                has_side_effect=False,
            ](self)._refine[new_size=size_out]()
        else:
            return llvm_intrinsic[
                "llvm.vector.reduce.fmin",
                Scalar[Self.dtype],
                has_side_effect=False,
            ](self)._refine[new_size=size_out]()

    @always_inline
    def reduce_add[size_out: Int = 1](self) -> SIMD[Self.dtype, size_out]:
        """Reduces the vector using the `add` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.

        Returns:
            The sum of all vector elements.

        """
        return self.reduce[Self._T.__add__, size_out]()

    @always_inline
    def reduce_mul[size_out: Int = 1](self) -> SIMD[Self.dtype, size_out]:
        """Reduces the vector using the `mul` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.
            The element type of the vector must be integer or FP.

        Returns:
            The product of all vector elements.
        """
        return self.reduce[Self._T.__mul__, size_out]()

    @always_inline
    def reduce_and[size_out: Int = 1](self) -> SIMD[Self.dtype, size_out]:
        """Reduces the vector using the bitwise `&` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.
            The element type of the vector must be integer or boolean.

        Returns:
            The reduced vector.
        """
        comptime assert (
            size_out <= Self.length
        ), "`size_out` must not exceed width of the vector."
        comptime assert (
            Self.dtype.is_integral() or Self.dtype == DType.bool
        ), "The element type of the vector must be integer or boolean."

        comptime if size_out > 1:
            return self.reduce[Self._T.__and__, size_out]()

        comptime if Self.length == 1:
            return self[0]

        return llvm_intrinsic[
            "llvm.vector.reduce.and",
            SIMD[Self.dtype, size_out],
            has_side_effect=False,
        ](self)

    @always_inline
    def reduce_or[size_out: Int = 1](self) -> SIMD[Self.dtype, size_out]:
        """Reduces the vector using the bitwise `|` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.
            The element type of the vector must be integer or boolean.

        Returns:
            The reduced vector.
        """
        comptime assert (
            size_out <= Self.length
        ), "`size_out` must not exceed width of the vector."
        comptime assert (
            Self.dtype.is_integral() or Self.dtype == DType.bool
        ), "The element type of the vector must be integer or boolean."

        comptime if size_out > 1:
            return self.reduce[Self._T.__or__, size_out]()

        comptime if Self.length == 1:
            return self[0]

        return llvm_intrinsic[
            "llvm.vector.reduce.or",
            SIMD[Self.dtype, size_out],
            has_side_effect=False,
        ](self)

    @always_inline
    def reduce_bit_count(self) -> Int:
        """Returns the total number of bits set in the SIMD vector.

        Constraints:
            Must be either an integral or a boolean type.

        Returns:
            Count of set bits across all elements of the vector.
        """
        comptime assert (
            Self.dtype.is_integral() or Self.dtype == DType.bool
        ), "Expected either integral or bool type"

        comptime if Self.dtype == DType.bool:
            comptime if Self.length == 1:
                return Int(self)
            else:
                var packed_mask = pack_bits(
                    rebind[SIMD[.bool, Self.length]](self)
                )
                var count = pop_count(packed_mask)
                return Int(count)
        else:
            return Int(pop_count(self).reduce_add())

    # ===------------------------------------------------------------------=== #
    # select
    # ===------------------------------------------------------------------=== #

    # TODO (7748): always_inline required to WAR LLVM codegen bug
    @always_inline("nodebug")
    def select[
        _dtype: DType
    ](
        self,
        true_case: SIMD[_dtype, Self.length],
        false_case: SIMD[_dtype, Self.length],
    ) -> SIMD[_dtype, Self.length]:
        """Selects the values of the `true_case` or the `false_case` based on
        the current boolean values of the SIMD vector.

        Parameters:
            _dtype: The element type of the input and output SIMD vectors.

        Args:
            true_case: The values selected if the positional value is True.
            false_case: The values selected if the positional value is False.

        Constraints:
            The element type of the vector must be boolean.

        Returns:
            A new vector of the form
            `[true_case[i] if elem else false_case[i] for i, elem in enumerate(self)]`.
        """
        comptime assert Self.dtype == DType.bool, "the simd type must be bool"
        var res = __mlir_op.`pop.simd.select`(
            self._refine[DType.bool]()._mlir_value,
            true_case._mlir_value,
            false_case._mlir_value,
        )
        return SIMD(mlir_value=res)

    # ===------------------------------------------------------------------=== #
    # Rotation operations
    # ===------------------------------------------------------------------=== #

    @always_inline
    def rotate_left[shift: Int](self) -> Self:
        """Shifts the elements of a SIMD vector to the left by `shift`
        elements (with wrap-around).

        Constraints:
            `-length <= shift < length`

        Parameters:
            shift: The number of positions by which to rotate the elements of
                   SIMD vector to the left (with wrap-around).

        Returns:
            The SIMD vector rotated to the left by `shift` elements
            (with wrap-around).
        """

        comptime assert (
            shift >= -Self.length and shift < Self.length
        ), "Constraint: -length <= shift < length"

        comptime if Self.length == 1:
            comptime assert shift == 0, "for scalars the shift must be 0"
            return self

        comptime if is_apple_gpu():
            return self.shuffle[mask=_apple_rotate_mask[Self.length, shift]()]()

        comptime if shift >= 0:
            return llvm_intrinsic[
                "llvm.vector.splice.left", Self, has_side_effect=False
            ](self, self, Int32(shift))
        else:
            return llvm_intrinsic[
                "llvm.vector.splice.right", Self, has_side_effect=False
            ](self, self, Int32(-shift))

    @always_inline
    def rotate_right[shift: Int](self) -> Self:
        """Shifts the elements of a SIMD vector to the right by `shift`
        elements (with wrap-around).

        Constraints:
            `-length < shift <= length`

        Parameters:
            shift: The number of positions by which to rotate the elements of
                   SIMD vector to the right (with wrap-around).

        Returns:
            The SIMD vector rotated to the right by `shift` elements
            (with wrap-around).
        """

        comptime assert (
            shift > -Self.length and shift <= Self.length
        ), "Constraint: -length < shift <= length"

        comptime if Self.length == 1:
            comptime assert shift == 0, "for scalars the shift must be 0"
            return self
        return self.rotate_left[-shift]()

    # ===------------------------------------------------------------------=== #
    # Shift operations
    # ===------------------------------------------------------------------=== #

    @always_inline
    def shift_left[shift: Int](self) -> Self:
        """Shifts the elements of a SIMD vector to the left by `shift`
        elements (no wrap-around, fill with zero).

        Constraints:
            `0 <= shift <= length`

        Parameters:
            shift: The number of positions by which to rotate the elements of
                   SIMD vector to the left (no wrap-around, fill with zero).

        Returns:
            The SIMD vector rotated to the left by `shift` elements (no
            wrap-around, fill with zero).
        """

        comptime assert 0 <= shift <= Self.length, (
            "shift must be greater than or equal to 0 and less than equal"
            " to the length"
        )

        comptime if shift == 0:
            return self
        elif shift == Self.length:
            return 0

        comptime if is_apple_gpu():
            return self.shuffle[mask=_apple_shift_mask[Self.length, shift]()](
                Self()
            )

        return llvm_intrinsic[
            "llvm.vector.splice.left", Self, has_side_effect=False
        ](self, Self(), Int32(shift))

    @always_inline
    def shift_right[shift: Int](self) -> Self:
        """Shifts the elements of a SIMD vector to the right by `shift`
        elements (no wrap-around, fill with zero).

        Constraints:
            `0 <= shift <= length`

        Parameters:
            shift: The number of positions by which to rotate the elements of
                   SIMD vector to the right (no wrap-around, fill with zero).

        Returns:
            The SIMD vector rotated to the right by `shift` elements (no
            wrap-around, fill with zero).
        """

        # Note the order of the llvm_intrinsic arguments below differ from
        # shift_left(), so we cannot directly reuse it here.

        comptime assert 0 <= shift <= Self.length, (
            "shift must be greater than or equal to 0 and less than equal"
            " to the length"
        )

        comptime if shift == 0:
            return self
        elif shift == Self.length:
            return 0

        comptime if is_apple_gpu():
            return self.shuffle[mask=_apple_shift_mask[Self.length, -shift]()](
                Self()
            )

        return llvm_intrinsic[
            "llvm.vector.splice.right", Self, has_side_effect=False
        ](Self(), self, Int32(shift))

    def reversed(self) -> Self:
        """Reverses the SIMD vector by indexes.

        Returns:
            The by index reversed vector.

        Examples:
        ```mojo
        print(SIMD[.uint8, 4](1, 2, 3, 4).reversed()) # [4, 3, 2, 1]
        ```
        """

        def indices() -> IndexList[Self.length]:
            var res = IndexList[Self.length]()
            for i in range(Self.length):
                res[i] = Self.length - i - 1
            return res

        return self.shuffle[mask=indices()]()

    # ===-------------------------------------------------------------------===#
    # CoordLike
    # ===-------------------------------------------------------------------===#

    comptime ParamListType = Coord[Self].element_types
    """The element types (Self for scalar types)."""

    comptime _ParamListType = Self.ParamListType.values
    """The low-level parameter list of element types."""

    comptime static_value: Int = -1
    """Always -1 for runtime values (not statically known)."""

    comptime DTYPE = Self.dtype
    """The data type for the runtime integer value."""

    @staticmethod
    @always_inline("nodebug")
    def __len__() -> Int:
        """Get the length (always 1 for scalar types).

        Returns:
            Always returns 1.
        """
        comptime assert (
            Self.dtype.is_integral()
        ), "CoordLike requires integral types"
        comptime assert Self.length == 1, "CoordLike requires length == 1"
        return 1

    @always_inline("nodebug")
    def product(self) -> Scalar[Self.dtype]:
        """Calculate the product (returns the value for scalar types).

        Returns:
            The integer value.
        """
        comptime assert (
            Self.dtype.is_integral()
        ), "CoordLike requires integral types"
        comptime assert Self.length == 1, "CoordLike requires length == 1"
        return self[0]

    @always_inline("nodebug")
    def sum(self) -> Scalar[Self.dtype]:
        """Calculate the sum (returns the value for scalar types).

        Returns:
            The integer value.
        """
        comptime assert (
            Self.dtype.is_integral()
        ), "CoordLike requires integral types"
        comptime assert Self.length == 1, "CoordLike requires length == 1"
        return self[0]

    @always_inline("nodebug")
    def value(self) -> Scalar[Self.dtype]:
        """Get the scalar value.

        Returns:
            The runtime integer value.
        """
        comptime assert (
            Self.dtype.is_integral()
        ), "CoordLike requires integral types"
        comptime assert Self.length == 1, "CoordLike requires length == 1"

        return self[0]

    @always_inline("nodebug")
    def tuple(var self) -> Coord[*Self.ParamListType]:
        """Get as a tuple (not valid for `Scalar` CoordLike).

        Returns:
            Never returns; aborts at compile time.
        """
        comptime assert False, "SIMD is not a tuple CoordLike type"


comptime U8x16 = SIMD[.uint8, 16]
"""A 16-element vector of unsigned 8-bit integers."""


def _pshuf_or_tbl1(lookup_table: U8x16, indices: U8x16) -> U8x16:
    comptime if CompilationTarget.has_sse4():
        return _pshuf(lookup_table, indices)
    elif CompilationTarget.has_neon():
        return _tbl1(lookup_table, indices)
    else:
        # TODO: Change the error message when we allow SSE3
        comptime assert False, "To call _pshuf_or_tbl1() you need sse4 or neon."


def _pshuf(lookup_table: U8x16, indices: U8x16) -> U8x16:
    """Shuffle operation using the SSSE3 `pshuf` instruction.

    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shuffle_epi8&ig_expand=6003
    """
    return llvm_intrinsic[
        "llvm.x86.ssse3.pshuf.b.128", U8x16, has_side_effect=False
    ](lookup_table, indices)


def _tbl1(lookup_table: U8x16, indices: U8x16) -> U8x16:
    """Shuffle operation using the aarch64 `tbl1` instruction.

    See https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/coding-for-neon---part-5-rearranging-vectors
    """
    return llvm_intrinsic[
        "llvm.aarch64.neon.tbl1", U8x16, has_side_effect=False
    ](lookup_table, indices)


# ===----------------------------------------------------------------------=== #
# _pow
# ===----------------------------------------------------------------------=== #


@always_inline
def _pow[
    width: SIMDLength
](base: SIMD[_, width], exp: SIMD[_, width], out result: type_of(base)):
    """Computes the power of the elements of a SIMD vector raised to the
    corresponding elements of another SIMD vector.

    Parameters:
        width: The width of the input and output SIMD vectors.

    Args:
        base: Base of the power operation.
        exp: Exponent of the power operation.

    Returns:
        A vector containing elementwise `base` raised to the power of `exp`.
    """

    comptime if exp.dtype.is_floating_point() and base.dtype == exp.dtype:
        comptime if is_apple_gpu():
            # AIR has no bf16 overload of `air.pow`; Metal rejects it, so
            # evaluate in f32 and narrow the result back.
            comptime if base.dtype == .bfloat16:
                return _pow(base.cast[.float32](), exp.cast[.float32]()).cast[
                    base.dtype
                ]()
            return llvm_intrinsic[
                "llvm.air.pow",
                type_of(base),
                type_of(base),
                type_of(exp),
                has_side_effect=False,
            ](base, exp)
        else:
            return _powf(base, exp)
    elif exp.dtype.is_integral():
        # Common cases
        if all(exp.eq(2)):
            return base * base
        if all(exp.eq(3)):
            return base * base * base

        result = {}

        comptime for i in range(width):
            result[i] = _powi(base[i], exp[i].cast[DType.int32]())
    else:
        comptime assert False, "unsupported type combination"


@always_inline
def _powf_scalar(
    base: Scalar, exponent: Scalar
) -> type_of(base) where base.dtype.is_floating_point():
    comptime assert (
        exponent.dtype.is_floating_point()
    ), "exponent must be floating point"

    var integral, fractional = _modf_scalar(exponent)

    if integral == exponent:
        return _powi(base, integral.cast[DType.int32]())

    if fractional and base < 0:
        return _nan[base.dtype]()

    return std.math.exp(exponent.cast[base.dtype]() * std.math.log(base))


@always_inline
def _powf[
    width: SIMDLength
](base: SIMD[_, width], exp: SIMD[_, width], out result: type_of(base)):
    comptime assert (
        exp.dtype.is_floating_point()
    ), "exponent must be floating point"
    comptime assert (
        base.dtype.is_floating_point()
    ), "base must be floating point"
    result = {}

    comptime for i in range(width):
        result[i] = _powf_scalar(base[i], exp[i])


@always_inline
def _powi(base: Scalar, exp: Int32) -> type_of(base):
    if base.dtype.is_integral() and exp < 0:
        if base == 1:
            return 1
        if base == -1:
            # (-1) ** n is 1 for even n, -1 for odd n.
            return Scalar[base.dtype](1 if (-exp) & 1 == 0 else -1)
        # For |base| > 1, this is the integer truncation of
        # 1 / base^|exp|. Note: 0 ** negative is mathematically
        # undefined, but we return 0 here for now.
        return 0

    var a = base
    var b = abs(exp) if base.dtype.is_floating_point() else exp
    var res: Scalar[base.dtype] = 1
    while b > 0:
        if b & 1:
            res *= a
        a *= a
        b >>= 1

    comptime if base.dtype.is_floating_point():
        if exp < 0:
            return 1.0 / res
    return res


# ===----------------------------------------------------------------------=== #
# float8
# ===----------------------------------------------------------------------=== #


@always_inline
def _convert_float8_to_f32_scalar[
    dtype: DType,
    //,
    result_dtype: DType,
](x: Scalar[dtype]) -> Scalar[result_dtype]:
    comptime FP8_EXPONENT_BIAS = FPUtils[dtype].exponent_bias()
    comptime FP8_NUM_MANTISSA_BITS = FPUtils[dtype].mantissa_width()
    comptime FP32_EXPONENT_BIAS = FPUtils[result_dtype].exponent_bias()
    comptime FP32_NUM_MANTISSA_BITS = FPUtils[result_dtype].mantissa_width()

    var exp = FPUtils.get_exponent_biased(x)
    var mantissa = FPUtils.get_mantissa(x)

    if exp == 0 and mantissa != 0:
        var subnormal_shift = FP8_NUM_MANTISSA_BITS - bit_width(mantissa)
        exp -= subnormal_shift
        mantissa <<= subnormal_shift + 1
        mantissa &= FPUtils[dtype].mantissa_mask()

    # TODO: To generalize this routine for float16, logic needs to be added
    # to convert float8 numbers that are in the normal exponent range to
    # float16 numbers that are in the subnormal exponent range. This is not
    # a problem for float32/bfloat16 as the exponent range is wider than any
    # currently supported float8 types.
    comptime assert result_dtype != DType.float16

    exp += FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS
    mantissa <<= FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS

    var result = Scalar[result_dtype](0)
    result = FPUtils.set_exponent(result, exp)
    result = FPUtils.set_mantissa(result, mantissa)

    var x_bits = FPUtils.bitcast_to_uint(x)
    var exp_mantissa = x_bits & type_of(x_bits)(
        FPUtils[dtype].exponent_mantissa_mask()
    )

    if exp_mantissa == 0x00:
        result = Scalar[result_dtype](0)

    comptime if dtype in (DType.float8_e4m3fn, DType.float8_e5m2):
        comptime if dtype == DType.float8_e4m3fn:
            if exp_mantissa == 0x7F:
                result = _nan[result_dtype]()
        else:
            if exp_mantissa == 0x7C:
                result = _inf[result_dtype]()
            elif exp_mantissa > 0x7C:
                result = _nan[result_dtype]()

    result = FPUtils.set_sign(result, FPUtils.get_sign(x))

    comptime if dtype in (DType.float8_e4m3fnuz, DType.float8_e5m2fnuz):
        if x_bits == 0x80:
            result = _nan[result_dtype]()

    return result


@always_inline
def _convert_float8_to_f32[
    dtype: DType,
    size: SIMDLength,
](val: SIMD[dtype, size]) -> SIMD[.float32, size]:
    comptime if _is_sm_9x_or_newer() and not _is_sm_120x_or_newer() and dtype in (
        DType.float8_e4m3fn,
        DType.float8_e5m2,
    ):
        return _convert_float8_to_f16(val).cast[DType.float32]()

    elif (
        _is_amd_mi300x()
        and dtype
        in (
            DType.float8_e4m3fnuz,
            DType.float8_e5m2fnuz,
        )
    ) or (
        _cdna_4_or_newer()
        and dtype
        in (
            DType.float8_e4m3fn,
            DType.float8_e5m2,
        )
    ):
        var res = __mlir_op.`pop.cast`[_type=SIMD[.float32, size]._mlir_type](
            val._mlir_value
        )
        return SIMD[.float32, size](mlir_value=res)

    else:

        @always_inline
        def wrapper_fn[
            input_dtype: DType, result_dtype: DType
        ](val: Scalar[input_dtype]) -> Scalar[result_dtype]:
            return _convert_float8_to_f32_scalar[result_dtype](val)

        return _simd_apply[wrapper_fn, result_dtype=DType.float32](val)


@always_inline
def _convert_float8_to_f16[
    dtype: DType,
    size: SIMDLength,
](val: SIMD[dtype, size]) -> SIMD[.float16, size]:
    comptime if _is_sm_9x_or_newer() and not _is_sm_120x_or_newer() and dtype in (
        DType.float8_e4m3fn,
        DType.float8_e5m2,
    ):
        # do not call `SIMD.cast` here; the inliner will diverge
        var res = __mlir_op.`pop.cast`[_type=SIMD[.float16, size]._mlir_type](
            val._mlir_value
        )
        return SIMD[.float16, size](mlir_value=res)
    else:
        return _convert_float8_to_f32(val).cast[DType.float16]()


@always_inline
def _convert_f32_to_float8[
    dtype: DType,
    size: SIMDLength,
    //,
    target: DType,
](val: SIMD[dtype, size]) -> SIMD[target, size]:
    comptime if (
        _is_sm_9x_or_newer() or _cdna_4_or_newer()
    ) and not _is_sm_120x_or_newer() and target in (
        DType.float8_e4m3fn,
        DType.float8_e5m2,
    ):
        # do not call `SIMD.cast` here; the inliner will diverge
        var res = __mlir_op.`pop.cast`[_type=SIMD[target, size]._mlir_type](
            val._mlir_value
        )
        return SIMD(mlir_value=res)
    elif _is_amd_mi300x() and target in (
        DType.float8_e4m3fnuz,
        DType.float8_e5m2fnuz,
    ):
        var res = __mlir_op.`pop.cast`[_type=SIMD[target, size]._mlir_type](
            val._mlir_value
        )
        return SIMD(mlir_value=res)
    else:

        @always_inline
        def wrapper_fn[
            input_dtype: DType, result_dtype: DType
        ](val: Scalar[input_dtype]) -> Scalar[result_dtype]:
            return _convert_f32_to_float8_scalar[result_dtype](val)

        return _simd_apply[wrapper_fn, result_dtype=target](val)


@always_inline
def _convert_f32_to_float8_scalar[
    dtype: DType,
    //,
    target: DType,
](x: Scalar[dtype]) -> Scalar[target]:
    # software implementation rounds toward nearest even

    @__parameter
    def max_finite_byte() -> UInt8:
        comptime if target == DType.float8_e4m3fn:
            return UInt8(0x7E)
        elif target in (DType.float8_e4m3fnuz, DType.float8_e5m2fnuz):
            return UInt8(0x7F)
        else:
            comptime assert target == DType.float8_e5m2
            return UInt8(0x7B)

    comptime FP8_NUM_MANTISSA_BITS = FPUtils[target].mantissa_width()
    comptime FP8_NUM_EXPONENT_BITS = FPUtils[target].exponent_width()
    comptime FP32_NUM_BITS = bit_width_of[dtype]()
    comptime FP8_EXPONENT_MASK: UInt8 = UInt8((1 << FP8_NUM_EXPONENT_BITS) - 1)
    comptime FP8_MANTISSA_MASK: UInt8 = UInt8((1 << FP8_NUM_MANTISSA_BITS) - 1)
    comptime FP8_EXPONENT_BIAS = FPUtils[target].exponent_bias()
    comptime FP8_MIN_EXPONENT = 1 - FP8_EXPONENT_BIAS
    comptime FP32_EXPONENT_BIAS = FPUtils[dtype].exponent_bias()
    comptime FP32_NUM_MANTISSA_BITS = FPUtils[dtype].mantissa_width()
    comptime FP8_MAX_FLT = max_finite_byte()

    # Extract the bits in the FP32 type
    var sign: UInt8 = UInt8(0x80) if FPUtils[dtype].get_sign(x) else UInt8(0x00)
    var exp = Int32(FPUtils[dtype].get_exponent_biased(x)) - Int32(
        FP32_EXPONENT_BIAS
    )
    var mantissa = Int32(FPUtils[dtype].get_mantissa(x))

    # NaN => NaN
    if _isnan(x):
        comptime if target in (DType.float8_e4m3fn, DType.float8_e5m2):
            return bitcast[target](UInt8(0x7F))
        else:
            comptime assert target in (
                DType.float8_e4m3fnuz,
                DType.float8_e5m2fnuz,
            )
            return bitcast[target](UInt8(0x80))

    # Inf => MAX_FLT (satfinite)
    if _isinf(x):
        return bitcast[target](sign | FP8_MAX_FLT)

    var sticky_bit: Int32 = 0

    var u: UInt8
    if abs(x) >= Scalar[dtype](_max_finite[target]()):
        # satfinite
        return bitcast[target](sign | FP8_MAX_FLT)
    elif exp >= Int32(FP8_MIN_EXPONENT):
        # normal fp32 to normal fp8
        exp += Int32(FP8_EXPONENT_BIAS)
        u = (
            (exp.cast[DType.uint32]() & FP8_EXPONENT_MASK.cast[DType.uint32]())
            << UInt32(FP8_NUM_MANTISSA_BITS)
        ).cast[DType.uint8]()
        u = (
            u
            | (
                mantissa
                >> Int32(FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)
            ).cast[DType.uint8]()
        )
    else:
        # normal single-precision to subnormal float8-precision representation
        var rshift: Int32 = Int32(FP8_MIN_EXPONENT) - exp
        if rshift < Int32(FP32_NUM_BITS):
            mantissa |= Int32(1 << Int32(FP32_NUM_MANTISSA_BITS))
            sticky_bit = (
                (mantissa & ((1 << rshift) - 1)).ne(0).cast[DType.int32]()
            )
            mantissa = mantissa >> rshift
            u = (
                mantissa
                >> Int32(FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)
            ).cast[DType.uint8]() & FP8_MANTISSA_MASK
        else:
            mantissa = 0
            u = 0

    # round to nearest even
    var NUM_BITS_SHIFT: Int32 = Int32(FP32_NUM_MANTISSA_BITS) - Int32(
        FP8_NUM_MANTISSA_BITS + 1
    )
    var round_bit: Int32 = (mantissa >> NUM_BITS_SHIFT) & 1
    sticky_bit |= (
        (mantissa & ((1 << NUM_BITS_SHIFT) - 1)).ne(0).cast[DType.int32]()
    )

    if (round_bit and sticky_bit) or (round_bit and (u & 1)):
        u = (u + 1).cast[DType.uint8]()

    # Special case for dtypes that lack a representation for signed zero.
    comptime if target in (DType.float8_e4m3fnuz, DType.float8_e5m2fnuz):
        if u == 0:
            return bitcast[target](UInt8(0))

    return bitcast[target](sign | u)


@always_inline
def _convert_f32_to_float8_ue8m0_scalar[
    dtype: DType,
    //,
    target: DType,
    *,
    satfinite: Bool = False,
    rounding_mode: String = "rp",
](x: Scalar[dtype]) -> Scalar[target]:
    """Convert float32 to float8_e8m0fnu (UE8M0).

    This follows CUTLASS and uses `rounding_mode="rp"` (round toward +infinity)
    when mapping float32 to a biased exponent byte. Other rounding modes are
    currently unsupported in the CPU fallback.
    """
    comptime assert not satfinite, (
        "satfinite is not implemented for CPU path. Extend this function to"
        " support it."
    )
    comptime assert rounding_mode == "rp", (
        "Only rounding mode 'rp' is supported for CPU path. Extend this"
        " function to support other rounding modes."
    )

    if _isnan(x) or _isinf(x):
        return bitcast[target, 1](UInt8(0xFF))

    var x_uint32: UInt32 = bitcast[DType.uint32, 1](x)
    var exp: UInt8 = UInt8(
        (x_uint32 >> 23) & 0xFF
    )  # Extract the 8 bit exponent
    var mant: UInt32 = x_uint32 & 0x7FFFFF  # Extract the 23 bit mantissa

    if (mant > 0) and (exp != 0xFE) and (not (exp == 0 and mant <= 0x00400000)):
        exp = exp + 1

    return bitcast[target, 1](exp)


@always_inline
def _convert_f32_to_float8_ue8m0[
    dtype: DType,
    size: SIMDLength,
    //,
    target: DType,
    *,
    satfinite: Bool = False,
    rounding_mode: String = "rp",
](val: SIMD[dtype, size],) -> SIMD[target, size]:
    """Convert float32 to float8_e8m0fnu (UE8M0).

    The default rounding mode is `rounding_mode="rp"` (round toward +infinity),
    matching CUTLASS. On SM100+ this lowers to `cvt.rp[.satfinite].ue8m0x2.f32`.
    """
    comptime assert dtype == DType.float32 and target == DType.float8_e8m0fnu, (
        "this conversion is only supported for float32 -> float8_e8m0fnu."
        " Exnted it if you need bfloat16 -> float8_e8m0fnu."
    )

    comptime if is_nvidia_gpu() and _is_sm_100x_or_newer():
        comptime satfinite_suffix = ".satfinite" if satfinite else ""
        comptime asm_prefix = "cvt." + rounding_mode + satfinite_suffix + ".ue8m0x2.f32"

        comptime if size > 1:
            var res = SIMD[target, size]()

            comptime for i in range(0, size, 2):
                var f8x2_f32x2 = inlined_assembly[
                    asm_prefix + " $0, $1, $2;",
                    UInt16,
                    constraints="=h,f,f",
                    has_side_effect=False,
                ](val[i + 1], val[i])
                var ui8x2 = bitcast[target, 2](f8x2_f32x2)
                res = res.insert[offset=i](ui8x2)
            return res
        else:
            var f8x2_f32x2 = inlined_assembly[
                asm_prefix + " $0, $1, $2;",
                UInt16,
                constraints="=h,f,f",
                has_side_effect=False,
            ](Float32(0.0), val[0])
            var ui8x2 = bitcast[target, 2](f8x2_f32x2)
            return ui8x2[0]
    else:

        @always_inline
        def wrapper_fn[
            input_dtype: DType, result_dtype: DType
        ](val: Scalar[input_dtype]) -> Scalar[result_dtype]:
            return _convert_f32_to_float8_ue8m0_scalar[
                result_dtype, satfinite=satfinite, rounding_mode=rounding_mode
            ](val)

        return _simd_apply[wrapper_fn, result_dtype=target](val)


@always_inline
def _convert_float8_ue8m0_to_f32[
    dtype: DType,
    size: SIMDLength,
    //,
    target: DType,
](val: SIMD[dtype, size]) -> SIMD[target, size]:
    """Convert float8_e8m0fnu to float32.

    float8_e8m0fnu stores an 8-bit biased exponent (bias=127). For values
    0x01..0xFE, the float32 representation is `exp << 23` (sign=0, mantissa=0).

    Special cases match CUTLASS:
      - 0x00 maps to 2**-127, which is a float32 subnormal (bits 0x00400000).
      - 0xFF maps to NaN (bits 0x7fffffff), not +infinity.
    """
    comptime assert (
        dtype == DType.float8_e8m0fnu and target == DType.float32
    ), "this conversion is only supported for float8_e8m0fnu -> float32."

    comptime if is_nvidia_gpu() and _is_sm_100x_or_newer():
        comptime asm_prefix = "cvt.rn.bf16x2.ue8m0x2"

        comptime if size > 1:
            var res = SIMD[target, size]()

            comptime for i in range(0, size, 2):
                var ue8m0x2 = SIMD[.uint8, 2](
                    bitcast[DType.uint8, 1](val[i]),
                    bitcast[DType.uint8, 1](val[i + 1]),
                )
                var bf16x2 = inlined_assembly[
                    asm_prefix + " $0, $1;",
                    UInt32,
                    constraints="=r,h",
                    has_side_effect=False,
                ](bitcast[DType.uint16, 1](ue8m0x2))
                var f32x2 = bitcast[DType.bfloat16, 2](bf16x2).cast[target]()
                res = res.insert[offset=i](f32x2)
            return res
        else:
            var ue8m0x2 = SIMD[.uint8, 2](
                bitcast[DType.uint8, 1](val[0]), UInt8(0)
            )
            var bf16x2 = inlined_assembly[
                asm_prefix + " $0, $1;",
                UInt32,
                constraints="=r,h",
                has_side_effect=False,
            ](bitcast[DType.uint16, 1](ue8m0x2))
            var f32x2 = bitcast[DType.bfloat16, 2](bf16x2).cast[target]()
            return f32x2[0]

    else:
        var exp = val.to_bits[DType.uint8]()
        var f32_bits = exp.cast[DType.uint32]() << 23

        # 0x00 represents 2**-127, which is a float32 subnormal.
        f32_bits = exp.eq(0).select(SIMD[.uint32, size](0x00400000), f32_bits)
        # 0xFF is NaN for this format; avoid creating +inf in float32.
        f32_bits = exp.eq(0xFF).select(
            SIMD[.uint32, size](0x7FFFFFFF), f32_bits
        )

        return SIMD[target, size](from_bits=f32_bits)


# ===----------------------------------------------------------------------=== #
# bfloat16
# ===----------------------------------------------------------------------=== #


@always_inline
def _bfloat16_to_f32_scalar(
    val: BFloat16,
) -> Float32:
    # For bfloat16, we can just do an unsafe_memcpy to perform the cast to
    # float32.
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "cvt.f32.bf16 $0, $1;" if _is_sm_9x_or_newer() else "mov.b32 $0, {0, $1};",
            Float32,
            constraints="=f,h",
            has_side_effect=False,
        ](bitcast[DType.int16](val))

    return bitcast[DType.float32, 1](SIMD[.bfloat16, 2](0, val))


@always_inline
def _bfloat16_to_f32[
    size: SIMDLength
](val: SIMD[.bfloat16, size]) -> SIMD[.float32, size]:
    @always_inline
    def wrapper_fn[
        input_dtype: DType, result_dtype: DType
    ](val: Scalar[input_dtype]) -> Scalar[result_dtype]:
        return _bfloat16_to_f32_scalar(
            val._refine[DType.bfloat16](),
        )._refine[result_dtype]()

    return _simd_apply[wrapper_fn, result_dtype=DType.float32](val)


# ===----------------------------------------------------------------------=== #
# _simd_apply
# ===----------------------------------------------------------------------=== #


@always_inline
def _simd_apply[
    input_dtype: DType,
    simd_width: SIMDLength,
    //,
    func: def[input_dtype: DType, result_dtype: DType](
        Scalar[input_dtype]
    ) thin -> Scalar[result_dtype],
    *,
    result_dtype: DType = input_dtype,
](x: SIMD[input_dtype, simd_width]) -> SIMD[result_dtype, simd_width]:
    """Returns a value whose elements corresponds to applying `func` to each
    element in the vector.

    Parameter:
      input_dtype: Type of the input to func.
      simd_width: Width of the input and output SIMD vectors.
      func: Function to apply to the SIMD vector.
      result_dtype: Result type of func.

    Args:
      x: the input value.

    Returns:
      A SIMD vector whose element at index `i` is `func(x[i])`.
    """
    var result = SIMD[result_dtype, simd_width]()

    comptime for i in range(simd_width):
        result[i] = func[input_dtype, result_dtype](x[i])

    return result


@always_inline
def _simd_apply[
    simd_width: SIMDLength,
    //,
    func: def[lhs_dtype: DType, rhs_dtype: DType, result_dtype: DType](
        Scalar[lhs_dtype], Scalar[rhs_dtype]
    ) thin -> Scalar[result_dtype],
    *,
    result_dtype: DType,
](x: SIMD[_, simd_width], y: SIMD[_, simd_width]) -> SIMD[
    result_dtype, simd_width
]:
    """Returns a value whose elements corresponds to applying `func` to each
    element in the vector.

    Parameter:
      simd_width: Width of the input and output SIMD vectors.
      func: Function to apply to the SIMD vector.
      result_dtype: Result type of func.

    Args:
      x: the lhs input value.
      y: the rhs input value.

    Returns:
      A SIMD vector whose element at index `i` is `func(x[i], y[i])`.
    """
    var result = SIMD[result_dtype, simd_width]()

    comptime for i in range(simd_width):
        result[i] = func[x.dtype, y.dtype, result_dtype](x[i], y[i])

    return result


# ===----------------------------------------------------------------------=== #
# modf
# ===----------------------------------------------------------------------=== #


def _modf_scalar(x: Scalar) -> Tuple[type_of(x), type_of(x)]:
    comptime assert (
        x.dtype.is_floating_point()
    ), "the type must be floating point"
    if x < 1:
        if x < 0:
            var res = _modf_scalar(-x)
            return (-res[0], -res[1])
        if x == 0:
            return (x, x)
        return (Scalar[x.dtype](0), x)

    var f = _floor(x)
    return (f, x - f)


def _modf(x: SIMD) -> Tuple[type_of(x), type_of(x)]:
    comptime assert x.dtype.is_numeric(), "the type must be numeric"

    comptime if x.dtype.is_integral():
        return (x, {0})

    var result_int: type_of(x) = {}
    var result_frac: type_of(x) = {}

    comptime for i in range(x.length):
        var tup = _modf_scalar(x[i])
        result_int[i] = tup[0]
        result_frac[i] = tup[1]

    return (result_int, result_frac)


# ===----------------------------------------------------------------------=== #
# floor
# ===----------------------------------------------------------------------=== #
def _floor(x: SIMD) -> type_of(x):
    comptime if x.dtype.is_integral():
        return x

    comptime integral_type = FPUtils[x.dtype].integral_type
    comptime bitwidth = bit_width_of[x.dtype]()
    comptime exponent_width = FPUtils[x.dtype].exponent_width()
    comptime mantissa_width = FPUtils[x.dtype].mantissa_width()
    comptime mask = FPUtils[x.dtype].exponent_mask()
    comptime bias = FPUtils[x.dtype].exponent_bias()
    comptime shift_factor = bitwidth - exponent_width - 1

    var bits = x._to_bits_signed()
    comptime BitsType = type_of(bits)
    var e = ((bits & BitsType(mask)) >> BitsType(mantissa_width)) - BitsType(
        bias
    )
    bits = e.lt(BitsType(shift_factor)).select(
        bits & type_of(bits)(~((1 << (BitsType(shift_factor) - e)) - 1)),
        bits,
    )
    return type_of(x)(from_bits=bits)


def _scalar_repr_alias[dtype: DType]() -> Optional[StaticString]:
    """Returns the scalar type alias name for a `dtype`, or `None`.

    This is used by `SIMD.write_repr_to` to print scalars using their friendly
    alias (e.g. `UInt32(4)`) instead of the verbose `SIMD[.uint32, 1](4)`
    form.

    The set of `dtype`s handled here must stay in sync with the `Scalar`
    aliases defined near the top of this file (e.g. `Int32 = Scalar[...]`), and
    parallels `DType.write_to`, the other exhaustive per-`dtype` chain that must
    be updated when a `dtype` is added. A `dtype` with no scalar alias returns
    `None` and keeps the verbose form: `DType.bool` is excluded because
    `SIMD[.bool, 1]` is distinct from the `Bool` struct, and
    `DType.float8_e3m4` has no `Scalar` alias. A `dtype` matching none of these
    cases fails a `comptime` assertion, so a newly added `dtype` is caught here
    rather than silently losing its alias.

    Parameters:
        dtype: The `DType` to look up the scalar alias name for.

    Returns:
        The scalar type alias name for `dtype`, or `None` when the `dtype` has
        no scalar alias.
    """
    comptime if dtype == DType.int:
        return StaticString("Int")
    elif dtype == DType.uint:
        return StaticString("UInt")
    elif dtype == DType.int8:
        return StaticString("Int8")
    elif dtype == DType.uint8:
        return StaticString("UInt8")
    elif dtype == DType.int16:
        return StaticString("Int16")
    elif dtype == DType.uint16:
        return StaticString("UInt16")
    elif dtype == DType.int32:
        return StaticString("Int32")
    elif dtype == DType.uint32:
        return StaticString("UInt32")
    elif dtype == DType.int64:
        return StaticString("Int64")
    elif dtype == DType.uint64:
        return StaticString("UInt64")
    elif dtype == DType.int128:
        return StaticString("Int128")
    elif dtype == DType.uint128:
        return StaticString("UInt128")
    elif dtype == DType.int256:
        return StaticString("Int256")
    elif dtype == DType.uint256:
        return StaticString("UInt256")
    elif dtype == DType.float4_e2m1fn:
        return StaticString("Float4_e2m1fn")
    elif dtype == DType.float8_e5m2:
        return StaticString("Float8_e5m2")
    elif dtype == DType.float8_e5m2fnuz:
        return StaticString("Float8_e5m2fnuz")
    elif dtype == DType.float8_e4m3fn:
        return StaticString("Float8_e4m3fn")
    elif dtype == DType.float8_e4m3fnuz:
        return StaticString("Float8_e4m3fnuz")
    elif dtype == DType.float8_e8m0fnu:
        return StaticString("Float8_e8m0fnu")
    elif dtype == DType.bfloat16:
        return StaticString("BFloat16")
    elif dtype == DType.float16:
        return StaticString("Float16")
    elif dtype == DType.float32:
        return StaticString("Float32")
    elif dtype == DType.float64:
        return StaticString("Float64")
    elif dtype == DType.bool or dtype == DType.float8_e3m4:
        return None
    else:
        comptime assert False, (
            "unhandled dtype in `_scalar_repr_alias`: add a `Scalar` alias"
            " branch or an explicit `None` case"
        )


def _write_scalar[
    dtype: DType,
    W: Writer,
    //,
](mut writer: W, value: Scalar[dtype]):
    comptime if dtype == DType.bool:
        if value:
            writer.write_string("True")
        else:
            writer.write_string("False")

    elif dtype.is_half_float():
        # `_write_float` widens everything but `float64` and the `float8`
        # variants to `Float32` before running dragonbox, so a half float and
        # its `Float32` widening already format to identical text. Widening here
        # instead lets `float16` and `bfloat16` share one `_write_float`
        # instantiation with `float32` rather than each getting their own copy of
        # the formatter, which dominates the compile cost of printing SIMD
        # values. `test_simd.test_write_half_float_matches_float32` pins the
        # equivalence.
        _write_float(writer, value.cast[DType.float32]())

    elif dtype.is_half_float():
        # `_write_float` widens everything but `float64` and the `float8`
        # variants to `Float32` before running dragonbox, so a half float and
        # its `Float32` widening already format to identical text. Widening here
        # instead lets `float16` and `bfloat16` share one `_write_float`
        # instantiation with `float32` rather than each getting their own copy of
        # the formatter, which dominates the compile cost of printing SIMD
        # values. `test_simd.test_write_half_float_matches_float32` pins the
        # equivalence.
        _write_float(writer, value.cast[DType.float32]())

    elif dtype.is_floating_point():
        _write_float(writer, value)

    # TODO(MSTDL-1039): bring in performant integer to string formatter
    elif dtype.is_integral():
        _ = _write_int(writer, value)
    else:
        comptime assert (
            False
        ), "unable to write dtype, only integral/float/bool supported"
