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
"""Defines utilities to work with numeric types.

You can import these APIs from the `utils` package. For example:

```mojo
from utils._numerics import FPUtils
```
"""

from sys import llvm_intrinsic, bitwidthof, has_neon, has_sse4
from sys._assembly import inlined_assembly

from builtin.dtype import _integral_type_of
from memory import UnsafePointer, bitcast

# ===----------------------------------------------------------------------===#
# _digits
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _digits[type: DType]() -> Int:
    """Returns the number of digits in base-radix that can be represented by
    the type without change.

    For integer types, this is the number of bits not counting the sign bit and
    the padding bits (if any). For floating-point types, this is the digits of
    the mantissa (for IEC 559/IEEE 754 implementations, this is the number of
    digits stored for the mantissa plus one, because the mantissa has an
    implicit leading 1 and binary point).

    Parameters:
        type: The type to get the digits for.

    Returns:
        The number of digits that can be represented by the type without change.
    """
    alias mlir_type = __mlir_type[`!pop.scalar<`, type.value, `>`]

    @parameter
    if type == DType.bool:
        return 1

    @parameter
    if type.is_integral():
        var bitwidth = bitwidthof[mlir_type]()
        return bitwidth - 1 if type.is_signed() else bitwidth

    @parameter
    if type == DType.float16:
        return 11

    @parameter
    if type == DType.bfloat16:
        return 8

    @parameter
    if type == DType.float32:
        return 24

    @parameter
    if type == DType.float64:
        return 53
    # Unreachable.
    return -1


# ===----------------------------------------------------------------------===#
# _fp_bitcast_to_integer
# ===----------------------------------------------------------------------===#


@always_inline
fn _fp_bitcast_to_integer[type: DType](value: Scalar[type]) -> Int:
    """Bitcasts the floating-point value to an integer.

    Parameters:
        type: The floating-point type.

    Args:
        value: The value to bitcast.

    Returns:
        An integer representation of the floating-point value.
    """
    alias integer_type = _integral_type_of[type]()
    return int(bitcast[integer_type, 1](value))


# ===----------------------------------------------------------------------===#
# _fp_bitcast_from_integer
# ===----------------------------------------------------------------------===#


@always_inline
fn _fp_bitcast_from_integer[type: DType](value: Int) -> Scalar[type]:
    """Bitcasts the integer value to a floating-point value.

    Parameters:
        type: The floating-point type.

    Args:
        value: The value to bitcast.

    Returns:
        A float-point representation of the integer value.
    """
    alias integer_type = _integral_type_of[type]()
    var int_val = SIMD[integer_type, 1](value)
    return bitcast[type, 1](int_val)


# ===----------------------------------------------------------------------===#
# FPUtils
# ===----------------------------------------------------------------------===#


struct FPUtils[type: DType]:
    """Collection of utility functions for working with FP values.

    Constraints:
        The type is floating point.

    Parameters:
        type: The concrete FP dtype (FP32/FP64/etc).
    """

    alias integral_type = _integral_type_of[type]()
    """The equivalent integer type of the float type."""

    @staticmethod
    @always_inline("nodebug")
    fn mantissa_width() -> Int:
        """Returns the mantissa width of a floating point type.

        Returns:
            The mantissa width.
        """
        constrained[
            type.is_floating_point(),
            "dtype must be a floating point type",
        ]()
        return _digits[type]() - 1

    @staticmethod
    @always_inline("nodebug")
    fn max_exponent() -> Int:
        """Returns the max exponent of a floating point type.

        Returns:
            The max exponent.
        """
        constrained[
            type.is_floating_point(),
            "dtype must be a floating point type",
        ]()

        @parameter
        if type == DType.float16:
            return 16
        elif type == DType.float32 or type == DType.bfloat16:
            return 128

        debug_assert(type == DType.float64, "must be float64")
        return 1024

    @staticmethod
    @always_inline("nodebug")
    fn exponent_width() -> Int:
        """Returns the exponent width of a floating point type.

        Returns:
            The exponent width.
        """
        constrained[
            type.is_floating_point(),
            "dtype must be a floating point type",
        ]()

        @parameter
        if type == DType.float16:
            return 5
        elif type == DType.float32 or type == DType.bfloat16:
            return 8

        debug_assert(type == DType.float64, "must be float64")
        return 11

    @staticmethod
    @always_inline
    fn mantissa_mask() -> Int:
        """Returns the mantissa mask of a floating point type.

        Returns:
            The mantissa mask.
        """
        constrained[
            type.is_floating_point(),
            "dtype must be a floating point type",
        ]()
        return (1 << Self.mantissa_width()) - 1

    @staticmethod
    @always_inline
    fn exponent_bias() -> Int:
        """Returns the exponent bias of a floating point type.

        Returns:
            The exponent bias.
        """
        constrained[
            type.is_floating_point(),
            "dtype must be a floating point type",
        ]()
        return Self.max_exponent() - 1

    @staticmethod
    @always_inline
    fn sign_mask() -> Int:
        """Returns the sign mask of a floating point type. It is computed by
        `1 << (exponent_width + mantissa_mask)`.

        Returns:
            The sign mask.
        """
        constrained[
            type.is_floating_point(),
            "dtype must be a floating point type",
        ]()
        return 1 << (Self.exponent_width() + Self.mantissa_width())

    @staticmethod
    @always_inline
    fn exponent_mask() -> Int:
        """Returns the exponent mask of a floating point type. It is computed by
        `~(sign_mask | mantissa_mask)`.

        Returns:
            The exponent mask.
        """
        constrained[
            type.is_floating_point(),
            "dtype must be a floating point type",
        ]()
        return ~(Self.sign_mask() | Self.mantissa_mask())

    @staticmethod
    @always_inline
    fn exponent_mantissa_mask() -> Int:
        """Returns the exponent and mantissa mask of a floating point type. It is
        computed by `exponent_mask + mantissa_mask`.

        Returns:
            The exponent and mantissa mask.
        """
        constrained[
            type.is_floating_point(),
            "dtype must be a floating point type",
        ]()
        return Self.exponent_mask() + Self.mantissa_mask()

    @staticmethod
    @always_inline
    fn quiet_nan_mask() -> Int:
        """Returns the quiet NaN mask for a floating point type.

        The mask is defined by evaluating:

        ```
        (1<<exponent_width-1)<<mantissa_width + 1<<(mantissa_width-1)
        ```

        Returns:
            The quiet NaN mask.
        """
        constrained[
            type.is_floating_point(),
            "dtype must be a floating point type",
        ]()
        var mantissa_width_val = Self.mantissa_width()
        return (1 << Self.exponent_width() - 1) << mantissa_width_val + (
            1 << (mantissa_width_val - 1)
        )

    @staticmethod
    @always_inline
    fn bitcast_to_integer(value: Scalar[type]) -> Int:
        """Bitcasts the floating-point value to an integer.

        Args:
            value: The floating-point type.

        Returns:
            An integer representation of the floating-point value.
        """
        return _fp_bitcast_to_integer[type](value)

    @staticmethod
    @always_inline
    fn bitcast_from_integer(value: Int) -> Scalar[type]:
        """Bitcasts the floating-point value from an integer.

        Args:
            value: The int value.

        Returns:
            An floating-point representation of the Int.
        """
        return _fp_bitcast_from_integer[type](value)

    @staticmethod
    @always_inline
    fn get_sign(value: Scalar[type]) -> Bool:
        """Returns the sign of the floating point value. True if the sign is set
        and False otherwise.

        Args:
            value: The floating-point type.

        Returns:
            Returns True if the sign is set and False otherwise.
        """

        return (Self.bitcast_to_integer(value) & Self.sign_mask()) != 0

    @staticmethod
    @always_inline
    fn set_sign(value: Scalar[type], sign: Bool) -> Scalar[type]:
        """Sets the sign of the floating point value.

        Args:
            value: The floating-point value.
            sign: True to set the sign and false otherwise.

        Returns:
            Returns the floating point value with the sign set.
        """
        var bits = Self.bitcast_to_integer(value)
        var sign_bits = Self.sign_mask()
        bits &= ~sign_bits
        if sign:
            bits |= sign_bits
        return Self.bitcast_from_integer(bits)

    @staticmethod
    @always_inline
    fn get_exponent(value: Scalar[type]) -> Int:
        """Returns the exponent bits of the floating-point value.

        Args:
            value: The floating-point value.

        Returns:
            Returns the exponent bits.
        """
        return (
            Self.bitcast_to_integer(value) & Self.exponent_mask()
        ) >> Self.mantissa_width()

    @staticmethod
    @always_inline
    fn get_exponent_without_bias(value: Scalar[type]) -> Int:
        """Returns the exponent bits of the floating-point value.

        Args:
            value: The floating-point value.

        Returns:
            Returns the exponent bits.
        """

        return Self.get_exponent(value) - Self.exponent_bias()

    @staticmethod
    @always_inline
    fn set_exponent(value: Scalar[type], exponent: Int) -> Scalar[type]:
        """Sets the exponent bits of the floating-point value.

        Args:
            value: The floating-point value.
            exponent: The exponent bits.

        Returns:
            Returns the floating-point value with the exponent bits set.
        """
        var bits = Self.bitcast_to_integer(value)
        bits &= ~Self.exponent_mask()
        bits |= (exponent << Self.mantissa_width()) & Self.exponent_mask()
        return Self.bitcast_from_integer(bits)

    @staticmethod
    @always_inline
    fn get_mantissa(value: Scalar[type]) -> Int:
        """Gets the mantissa bits of the floating-point value.

        Args:
            value: The floating-point value.

        Returns:
            The mantissa bits.
        """
        return Self.bitcast_to_integer(value) & Self.mantissa_mask()

    @staticmethod
    @always_inline
    fn set_mantissa(value: Scalar[type], mantissa: Int) -> Scalar[type]:
        """Sets the mantissa bits of the floating-point value.

        Args:
            value: The floating-point value.
            mantissa: The mantissa bits.

        Returns:
            Returns the floating-point value with the mantissa bits set.
        """
        var bits = Self.bitcast_to_integer(value)
        bits &= ~Self.mantissa_mask()
        bits |= mantissa & Self.mantissa_mask()
        return Self.bitcast_from_integer(bits)

    @staticmethod
    @always_inline
    fn pack(sign: Bool, exponent: Int, mantissa: Int) -> Scalar[type]:
        """Construct a floating-point value from its constituent sign, exponent,
        and mantissa.

        Args:
            sign: The sign of the floating-point value.
            exponent: The exponent of the floating-point value.
            mantissa: The mantissa of the floating-point value.

        Returns:
            Returns the floating-point value.
        """
        var res: Scalar[type] = 0
        res = Self.set_sign(res, sign)
        res = Self.set_exponent(res, exponent)
        res = Self.set_mantissa(res, mantissa)
        return res


# ===----------------------------------------------------------------------===#
# FlushDenormals
# ===----------------------------------------------------------------------===#


struct FlushDenormals:
    """Flushes and denormals are set to zero within the context and the state
    is restored to the prior value on exit."""

    var state: Int32
    """The current state."""

    @always_inline
    fn __init__(inout self):
        """Initializes the FlushDenormals."""
        self.state = Self._current_state()

    @always_inline
    fn __enter__(self):
        """Enters the context. This will set denormals to zero."""
        self._set_flush(True)

    @always_inline
    fn __exit__(self):
        """Exits the context. This will restore the prior FPState."""
        self._set_flush(False, True)

    @always_inline
    fn _set_flush(self, enable: Bool, force: Bool = False):
        @parameter
        if not has_sse4() and not has_neon():  # not supported, so skip
            return
        # Unless we forced to restore the prior state, we check if the flag
        # has already been enabled to avoid calling the intrinsic which can
        # be costly.
        if not force and enable == self._is_set(self.state):
            return

        # If the enable flag is set then we need to argument the register
        # value, otherwise we are in an exit state and we need to restore
        # the prior value.

        @parameter
        if has_sse4():
            var mxcsr = self.state
            if enable:
                mxcsr |= 0x8000  # flush to zero
                mxcsr |= 0x40  # denormals are zero
            llvm_intrinsic["llvm.x86.sse.ldmxcsr", NoneType](
                UnsafePointer[Int32].address_of(mxcsr)
            )
            return

        alias ARM_FPCR_FZ = Int64(1) << 24
        var fpcr = self.state.cast[DType.int64]()
        if enable:
            fpcr |= ARM_FPCR_FZ

        inlined_assembly[
            "msr fpcr, $0",
            NoneType,
            constraints="r",
            has_side_effect=True,
        ](fpcr)

    @always_inline
    fn _is_set(self, state: Int32) -> Bool:
        @parameter
        if has_sse4():
            return (state & 0x8000) != 0 and (state & 0x40) != 0

        alias ARM_FPCR_FZ = Int32(1) << 24
        return (state & ARM_FPCR_FZ) != 0

    @always_inline
    @staticmethod
    fn _current_state() -> Int32:
        """Gets the current denormal state."""

        @parameter
        if not has_sse4() and not has_neon():  # not supported, so skip
            return 0

        @parameter
        if has_sse4():
            var mxcsr = Int32()
            llvm_intrinsic["llvm.x86.sse.stmxcsr", NoneType](
                UnsafePointer[Int32].address_of(mxcsr)
            )
            return mxcsr

        var fpcr64 = inlined_assembly[
            "mrs $0, fpcr",
            UInt64,
            constraints="=r",
            has_side_effect=True,
        ]()

        return fpcr64.cast[DType.int32]()


# ===----------------------------------------------------------------------===#
# nan
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn nan[type: DType]() -> Scalar[type]:
    """Gets a NaN value for the given dtype.

    Constraints:
        Can only be used for FP dtypes.

    Parameters:
        type: The value dtype.

    Returns:
        The NaN value of the given dtype.
    """

    @parameter
    if type == DType.float16:
        return rebind[__mlir_type[`!pop.scalar<`, type.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f16>`],
                value = __mlir_attr[`#pop.simd<"nan"> : !pop.scalar<f16>`],
            ]()
        )
    elif type == DType.bfloat16:
        return rebind[__mlir_type[`!pop.scalar<`, type.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<bf16>`],
                value = __mlir_attr[`#pop.simd<"nan"> : !pop.scalar<bf16>`],
            ]()
        )
    elif type == DType.float32:
        return rebind[__mlir_type[`!pop.scalar<`, type.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f32>`],
                value = __mlir_attr[`#pop.simd<"nan"> : !pop.scalar<f32>`],
            ]()
        )
    elif type == DType.float64:
        return rebind[__mlir_type[`!pop.scalar<`, type.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f64>`],
                value = __mlir_attr[`#pop.simd<"nan"> : !pop.scalar<f64>`],
            ]()
        )
    else:
        constrained[False, "nan only support on floating point types"]()

    return 0


# ===----------------------------------------------------------------------===#
# isnan
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn isnan[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width]) -> SIMD[DType.bool, simd_width]:
    """Checks if the value is Not a Number (NaN).

    Parameters:
        type: The value dtype.
        simd_width: The width of the SIMD vector.

    Args:
        val: The value to check.

    Returns:
        True if val is NaN and False otherwise.
    """

    @parameter
    if not type.is_floating_point():
        return False

    @parameter
    if type == DType.bfloat16:
        alias int_dtype = _integral_type_of[type]()
        var int_val = bitcast[int_dtype, simd_width](val)
        return int_val & SIMD[int_dtype, simd_width](0x7FFF) > SIMD[
            int_dtype, simd_width
        ](0x7F80)

    alias signaling_nan_test: UInt32 = 0x0001
    alias quiet_nan_test: UInt32 = 0x0002
    return llvm_intrinsic[
        "llvm.is.fpclass", SIMD[DType.bool, simd_width], has_side_effect=False
    ](val.value, (signaling_nan_test | quiet_nan_test).value)


# ===----------------------------------------------------------------------===#
# inf
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn inf[type: DType]() -> Scalar[type]:
    """Gets a +inf value for the given dtype.

    Constraints:
        Can only be used for FP dtypes.

    Parameters:
        type: The value dtype.

    Returns:
        The +inf value of the given dtype.
    """

    @parameter
    if type == DType.float16:
        return rebind[__mlir_type[`!pop.scalar<`, type.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f16>`],
                value = __mlir_attr[`#pop.simd<"inf"> : !pop.scalar<f16>`],
            ]()
        )
    elif type == DType.bfloat16:
        return rebind[__mlir_type[`!pop.scalar<`, type.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<bf16>`],
                value = __mlir_attr[`#pop.simd<"inf"> : !pop.scalar<bf16>`],
            ]()
        )
    elif type == DType.float32:
        return rebind[__mlir_type[`!pop.scalar<`, type.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f32>`],
                value = __mlir_attr[`#pop.simd<"inf"> : !pop.scalar<f32>`],
            ]()
        )
    elif type == DType.float64:
        return rebind[__mlir_type[`!pop.scalar<`, type.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f64>`],
                value = __mlir_attr[`#pop.simd<"inf"> : !pop.scalar<f64>`],
            ]()
        )
    else:
        constrained[False, "+inf only support on floating point types"]()

    return 0


# ===----------------------------------------------------------------------===#
# isinf
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn isinf[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width]) -> SIMD[DType.bool, simd_width]:
    """Checks if the value is infinite.

    This is always False for non-FP data types.

    Parameters:
        type: The value dtype.
        simd_width: The width of the SIMD vector.

    Args:
        val: The value to check.

    Returns:
        True if val is infinite and False otherwise.
    """

    @parameter
    if not type.is_floating_point():
        return False

    alias negative_infinity_test: UInt32 = 0x0004
    alias positive_infinity_test: UInt32 = 0x0200
    return llvm_intrinsic["llvm.is.fpclass", SIMD[DType.bool, simd_width]](
        val.value, (negative_infinity_test | positive_infinity_test).value
    )


# ===----------------------------------------------------------------------===#
# isfinite
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn isfinite[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width]) -> SIMD[DType.bool, simd_width]:
    """Checks if the value is not infinite.

    This is always True for non-FP data types.

    Parameters:
        type: The value dtype.
        simd_width: The width of the SIMD vector.

    Args:
        val: The value to check.

    Returns:
        True if val is finite and False otherwise.
    """

    @parameter
    if not type.is_floating_point():
        return True

    return llvm_intrinsic["llvm.is.fpclass", SIMD[DType.bool, simd_width]](
        val.value, UInt32(0x1F8).value
    )


# ===----------------------------------------------------------------------===#
# get_accum_type
# ===----------------------------------------------------------------------===#


@always_inline
fn get_accum_type[type: DType]() -> DType:
    """Returns the recommended type for accumulation operations.

    Half precision types can introduce numerical error if they are used
    in reduction/accumulation operations. This method returns a higher precision
    type to use for accumulation if a half precision types is provided,
    otherwise it returns the original type.

    Parameters:
        type: The type of some accumulation operation.

    Returns:
        DType.float32 if type is a half-precision float, type otherwise.
    """

    return DType.float32 if type.is_half_float() else type
