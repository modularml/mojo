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
"""Implements SIMD struct.

These are Mojo built-ins, so you don't need to import them.
"""

from bit import pop_count
from sys import (
    llvm_intrinsic,
    has_neon,
    is_x86,
    triple_is_nvidia_cuda,
    simdwidthof,
    _RegisterPackType,
    PrefetchOptions,
    prefetch,
)

from builtin._math import Ceilable, CeilDivable, Floorable, Truncable
from builtin.hash import _hash_simd
from memory import bitcast

from utils.numerics import (
    FPUtils,
    isnan as _isnan,
    nan as _nan,
    max_finite as _max_finite,
    min_finite as _min_finite,
    max_or_inf as _max_or_inf,
    min_or_neg_inf as _min_or_neg_inf,
)
from utils._visualizers import lldb_formatter_wrapping_type
from utils import InlineArray, StringSlice

from .dtype import (
    _integral_type_of,
    _get_dtype_printf_format,
    _scientific_notation_digits,
)
from .io import _snprintf_scalar, _printf, _print_fmt
from .string import _calc_initial_buffer_size, _calc_format_buffer_size

# ===----------------------------------------------------------------------=== #
# Type Aliases
# ===----------------------------------------------------------------------=== #

alias Scalar = SIMD[size=1]
"""Represents a scalar dtype."""

alias Int8 = Scalar[DType.int8]
"""Represents an 8-bit signed scalar integer."""
alias UInt8 = Scalar[DType.uint8]
"""Represents an 8-bit unsigned scalar integer."""
alias Int16 = Scalar[DType.int16]
"""Represents a 16-bit signed scalar integer."""
alias UInt16 = Scalar[DType.uint16]
"""Represents a 16-bit unsigned scalar integer."""
alias Int32 = Scalar[DType.int32]
"""Represents a 32-bit signed scalar integer."""
alias UInt32 = Scalar[DType.uint32]
"""Represents a 32-bit unsigned scalar integer."""
alias Int64 = Scalar[DType.int64]
"""Represents a 64-bit signed scalar integer."""
alias UInt64 = Scalar[DType.uint64]
"""Represents a 64-bit unsigned scalar integer."""

alias BFloat16 = Scalar[DType.bfloat16]
"""Represents a 16-bit brain floating point value."""
alias Float16 = Scalar[DType.float16]
"""Represents a 16-bit floating point value."""
alias Float32 = Scalar[DType.float32]
"""Represents a 32-bit floating point value."""
alias Float64 = Scalar[DType.float64]
"""Represents a 64-bit floating point value."""

# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn _simd_construction_checks[type: DType, size: Int]():
    """Checks if the SIMD size is valid.

    The SIMD size is valid if it is a power of two and is positive.

    Parameters:
      type: The data type of SIMD vector elements.
      size: The number of elements in the SIMD vector.
    """
    constrained[type != DType.invalid, "simd type cannot be DType.invalid"]()
    constrained[size > 0, "simd width must be > 0"]()
    constrained[size & (size - 1) == 0, "simd width must be power of 2"]()
    constrained[
        type != DType.bfloat16 or not has_neon(),
        "bf16 is not supported for ARM architectures",
    ]()


@always_inline("nodebug")
fn _unchecked_zero[type: DType, size: Int]() -> SIMD[type, size]:
    var zero = __mlir_op.`pop.cast`[
        _type = __mlir_type[`!pop.scalar<`, type.value, `>`]
    ](
        __mlir_op.`kgen.param.constant`[
            _type = __mlir_type[`!pop.scalar<index>`],
            value = __mlir_attr[`#pop.simd<0> : !pop.scalar<index>`],
        ]()
    )
    return SIMD[type, size] {
        value: __mlir_op.`pop.simd.splat`[
            _type = __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`]
        ](zero)
    }


# ===----------------------------------------------------------------------=== #
# SIMD
# ===----------------------------------------------------------------------=== #


@lldb_formatter_wrapping_type
@register_passable("trivial")
struct SIMD[type: DType, size: Int = simdwidthof[type]()](
    Absable,
    Boolable,
    Ceilable,
    CeilDivable,
    CollectionElement,
    CollectionElementNew,
    Floorable,
    Hashable,
    Intable,
    Powable,
    Roundable,
    Sized,
    Stringable,
    Truncable,
    Representable,
):
    """Represents a small vector that is backed by a hardware vector element.

    SIMD allows a single instruction to be executed across the multiple data
    elements of the vector.

    Constraints:
        The size of the SIMD vector to be positive and a power of 2.

    Parameters:
        type: The data type of SIMD vector elements.
        size: The size of the SIMD vector.
    """

    # Fields
    alias _Mask = SIMD[DType.bool, size]

    alias element_type = type
    var value: __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`]
    """The underlying storage for the vector."""

    alias MAX = Self(_max_or_inf[type]())
    """Gets the maximum value for the SIMD value, potentially +inf."""

    alias MIN = Self(_min_or_neg_inf[type]())
    """Gets the minimum value for the SIMD value, potentially -inf."""

    alias MAX_FINITE = Self(_max_finite[type]())
    """Returns the maximum finite value of SIMD value."""

    alias MIN_FINITE = Self(_min_finite[type]())
    """Returns the minimum (lowest) finite value of SIMD value."""

    alias _default_alignment = alignof[
        Scalar[type]
    ]() if triple_is_nvidia_cuda() else 1

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __init__(inout self):
        """Default initializer of the SIMD vector.

        By default the SIMD vectors are initialized to all zeros.
        """
        _simd_construction_checks[type, size]()
        self = _unchecked_zero[type, size]()

    @always_inline("nodebug")
    fn __init__(inout self, value: SIMD[DType.float64, 1]):
        """Initializes the SIMD vector with a float.

        The value is splatted across all the elements of the SIMD
        vector.

        Args:
            value: The input value.
        """
        _simd_construction_checks[type, size]()

        var casted = __mlir_op.`pop.cast`[
            _type = __mlir_type[`!pop.simd<1,`, type.value, `>`]
        ](value.value)
        var vec = __mlir_op.`pop.simd.splat`[
            _type = __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`]
        ](casted)
        self.value = vec

    @always_inline("nodebug")
    fn __init__(inout self, *, other: SIMD[type, size]):
        """Explicitly copy the provided value.

        Args:
            other: The value to copy.
        """
        self.__copyinit__(other)

    @always_inline("nodebug")
    fn __init__(inout self, value: Int):
        """Initializes the SIMD vector with an integer.

        The integer value is splatted across all the elements of the SIMD
        vector.

        Args:
            value: The input value.
        """
        _simd_construction_checks[type, size]()

        var t0 = __mlir_op.`pop.cast_from_builtin`[
            _type = __mlir_type.`!pop.scalar<index>`
        ](value.value)
        var casted = __mlir_op.`pop.cast`[
            _type = __mlir_type[`!pop.simd<1,`, type.value, `>`]
        ](t0)
        self.value = __mlir_op.`pop.simd.splat`[
            _type = __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`]
        ](casted)

    @always_inline("nodebug")
    fn __init__(inout self, value: IntLiteral):
        """Initializes the SIMD vector with an integer.

        The integer value is splatted across all the elements of the SIMD
        vector.

        Args:
            value: The input value.
        """
        _simd_construction_checks[type, size]()

        var tn1 = __mlir_op.`kgen.int_literal.convert`[
            _type = __mlir_type.si128
        ](value.value)
        var t0 = __mlir_op.`pop.cast_from_builtin`[
            _type = __mlir_type.`!pop.scalar<si128>`
        ](tn1)
        var casted = __mlir_op.`pop.cast`[
            _type = __mlir_type[`!pop.simd<1,`, type.value, `>`]
        ](t0)
        self.value = __mlir_op.`pop.simd.splat`[
            _type = __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`]
        ](casted)

    @always_inline("nodebug")
    fn __init__(inout self, value: Bool):
        """Initializes the SIMD vector with a bool value.

        The bool value is splatted across all elements of the SIMD vector.

        Args:
            value: The bool value.
        """
        _simd_construction_checks[type, size]()

        var casted = __mlir_op.`pop.cast`[
            _type = __mlir_type[`!pop.simd<1,`, type.value, `>`]
        ](value._as_scalar_bool())
        self.value = __mlir_op.`pop.simd.splat`[
            _type = __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`]
        ](casted)

    @always_inline("nodebug")
    fn __init__(
        inout self,
        value: __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`],
    ):
        """Initializes the SIMD vector with the underlying mlir value.

        Args:
            value: The input value.
        """
        _simd_construction_checks[type, size]()
        self.value = value

    # Construct via a variadic type which has the same number of elements as
    # the SIMD value.
    @always_inline("nodebug")
    fn __init__(inout self, *elems: Scalar[type]):
        """Constructs a SIMD vector via a variadic list of elements.

        If there is just one input value, then it is splatted to all elements
        of the SIMD vector. Otherwise, the input values are assigned to the
        corresponding elements of the SIMD vector.

        Constraints:
            The number of input values is 1 or equal to size of the SIMD
            vector.

        Args:
            elems: The variadic list of elements from which the SIMD vector is
                   constructed.
        """
        _simd_construction_checks[type, size]()

        var num_elements = len(elems)
        if num_elements == 1:
            # Construct by broadcasting a scalar.
            self.value = __mlir_op.`pop.simd.splat`[
                _type = __mlir_type[
                    `!pop.simd<`,
                    size.value,
                    `, `,
                    type.value,
                    `>`,
                ]
            ](elems[0].value)
            return
        # TODO: Make this a compile-time check when possible.
        debug_assert(
            size == num_elements,
            (
                "mismatch in the number of elements in the SIMD variadic"
                " constructor"
            ),
        )

        self = __mlir_op.`kgen.undef`[
            _type = __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`]
        ]()

        @parameter
        for i in range(size):
            self[i] = elems[i]

    @always_inline("nodebug")
    fn __init__(inout self, value: FloatLiteral):
        """Initializes the SIMD vector with a float.

        The value is splatted across all the elements of the SIMD
        vector.

        Args:
            value: The input value.
        """
        _simd_construction_checks[type, size]()

        # TODO (#36686): This introduces uneeded casts here to work around
        # parameter if issues.
        @parameter
        if type == DType.float16:
            self = SIMD[type, size](
                __mlir_op.`pop.simd.splat`[
                    _type = __mlir_type[
                        `!pop.simd<`, size.value, `,`, type.value, `>`
                    ]
                ](
                    __mlir_op.`pop.cast`[
                        _type = __mlir_type[`!pop.scalar<`, type.value, `>`]
                    ](
                        __mlir_op.`pop.cast_from_builtin`[
                            _type = __mlir_type[`!pop.scalar<f16>`]
                        ](
                            __mlir_op.`kgen.float_literal.convert`[
                                _type = __mlir_type.f16
                            ](value.value)
                        )
                    )
                )
            )
        elif type == DType.bfloat16:
            self = Self(
                __mlir_op.`pop.simd.splat`[
                    _type = __mlir_type[
                        `!pop.simd<`, size.value, `,`, type.value, `>`
                    ]
                ](
                    __mlir_op.`pop.cast`[
                        _type = __mlir_type[`!pop.scalar<`, type.value, `>`]
                    ](
                        __mlir_op.`pop.cast_from_builtin`[
                            _type = __mlir_type[`!pop.scalar<bf16>`]
                        ](
                            __mlir_op.`kgen.float_literal.convert`[
                                _type = __mlir_type.bf16
                            ](value.value)
                        )
                    )
                )
            )
        elif type == DType.float32:
            self = Self(
                __mlir_op.`pop.simd.splat`[
                    _type = __mlir_type[
                        `!pop.simd<`, size.value, `,`, type.value, `>`
                    ]
                ](
                    __mlir_op.`pop.cast`[
                        _type = __mlir_type[`!pop.scalar<`, type.value, `>`]
                    ](
                        __mlir_op.`pop.cast_from_builtin`[
                            _type = __mlir_type[`!pop.scalar<f32>`]
                        ](
                            __mlir_op.`kgen.float_literal.convert`[
                                _type = __mlir_type.f32
                            ](value.value)
                        )
                    )
                )
            )
        else:
            self = Self(
                __mlir_op.`pop.simd.splat`[
                    _type = __mlir_type[
                        `!pop.simd<`, size.value, `,`, type.value, `>`
                    ]
                ](
                    __mlir_op.`pop.cast`[
                        _type = __mlir_type[`!pop.scalar<`, type.value, `>`]
                    ](
                        __mlir_op.`pop.cast_from_builtin`[
                            _type = __mlir_type[`!pop.scalar<f64>`]
                        ](
                            __mlir_op.`kgen.float_literal.convert`[
                                _type = __mlir_type.f64
                            ](value.value)
                        )
                    )
                )
            )

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline("nodebug")
    fn splat(x: Scalar[type]) -> Self:
        """Splats (broadcasts) the element onto the vector.

        Args:
            x: The input scalar value.

        Returns:
            A new SIMD vector whose elements are the same as the input value.
        """
        _simd_construction_checks[type, size]()
        return Self {
            value: __mlir_op.`pop.simd.splat`[
                _type = __mlir_type[
                    `!pop.simd<`, size.value, `, `, type.value, `>`
                ]
            ](x.value)
        }

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Scalar[type]:
        """Gets an element from the vector.

        Args:
            idx: The element index.

        Returns:
            The value at position `idx`.
        """
        return __mlir_op.`pop.simd.extractelement`[
            _type = __mlir_type[`!pop.scalar<`, type.value, `>`]
        ](self.value, index(idx).value)

    @always_inline("nodebug")
    fn __setitem__(inout self, idx: Int, val: Scalar[type]):
        """Sets an element in the vector.

        Args:
            idx: The index to set.
            val: The value to set.
        """
        self.value = __mlir_op.`pop.simd.insertelement`(
            self.value, val.value, index(idx).value
        )

    @always_inline("nodebug")
    fn __add__(self, rhs: Self) -> Self:
        """Computes `self + rhs`.

        Args:
            rhs: The rhs value.

        Returns:
            A new vector whose element at position `i` is computed as
            `self[i] + rhs[i]`.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return __mlir_op.`pop.add`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __sub__(self, rhs: Self) -> Self:
        """Computes `self - rhs`.

        Args:
            rhs: The rhs value.

        Returns:
            A new vector whose element at position `i` is computed as
            `self[i] - rhs[i]`.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return __mlir_op.`pop.sub`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __mul__(self, rhs: Self) -> Self:
        """Computes `self * rhs`.

        Args:
            rhs: The rhs value.

        Returns:
            A new vector whose element at position `i` is computed as
            `self[i] * rhs[i]`.
        """

        @parameter
        if type == DType.bool:
            return (rebind[Self._Mask](self) & rebind[Self._Mask](rhs)).cast[
                type
            ]()

        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return __mlir_op.`pop.mul`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __truediv__(self, rhs: Self) -> Self:
        """Computes `self / rhs`.

        Args:
            rhs: The rhs value.

        Returns:
            A new vector whose element at position `i` is computed as
            `self[i] / rhs[i]`.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return __mlir_op.`pop.div`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __floordiv__(self, rhs: Self) -> Self:
        """Returns the division of self and rhs rounded down to the nearest
        integer.

        Constraints:
            The element type of the SIMD vector must be numeric.

        Args:
            rhs: The value to divide with.

        Returns:
            `floor(self / rhs)` value.
        """
        constrained[type.is_numeric(), "the type must be numeric"]()

        if not any(rhs):
            # this should raise an exception.
            return 0

        var div = self / rhs

        @parameter
        if type.is_floating_point():
            return div.__floor__()
        elif type.is_unsigned():
            return div
        else:
            if all((self > 0) & (rhs > 0)):
                return div

            var mod = self - div * rhs
            var mask = ((rhs < 0) ^ (self < 0)) & (mod != 0)
            return div - mask.cast[type]()

    @always_inline("nodebug")
    fn __mod__(self, rhs: Self) -> Self:
        """Returns the remainder of self divided by rhs.

        Args:
            rhs: The value to divide on.

        Returns:
            The remainder of dividing self by rhs.
        """
        constrained[type.is_numeric(), "the type must be numeric"]()

        if not any(rhs):
            # this should raise an exception.
            return 0

        @parameter
        if type.is_unsigned():
            return __mlir_op.`pop.rem`(self.value, rhs.value)
        else:
            var div = self / rhs

            @parameter
            if type.is_floating_point():
                div = llvm_intrinsic["llvm.trunc", Self, has_side_effect=False](
                    div
                )

            var mod = self - div * rhs
            var mask = ((rhs < 0) ^ (self < 0)) & (mod != 0)
            return mod + mask.select(rhs, Self(0))

    @always_inline("nodebug")
    fn __pow__(self, exp: Int) -> Self:
        """Computes the vector raised to the power of the input integer value.

        Args:
            exp: The exponent value.

        Returns:
            A SIMD vector where each element is raised to the power of the
            specified exponent value.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return _pow[type, size, DType.index](self, exp)

    # TODO(#22771): remove this overload.
    @always_inline("nodebug")
    fn __pow__(self, exp: Self) -> Self:
        """Computes the vector raised elementwise to the right hand side power.

        Args:
            exp: The exponent value.

        Returns:
            A SIMD vector where each element is raised to the power of the
            specified exponent value.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return _pow(self, exp)

    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using less-than comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is True or False depending on the expression
            `self[i] < rhs[i]`.
        """

        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred lt>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __le__(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using less-than-or-equal comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is True or False depending on the expression
            `self[i] <= rhs[i]`.
        """

        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred le>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using equal-to comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is True or False depending on the expression
            `self[i] == rhs[i]`.
        """

        # TODO(KERN-228): support BF16 on neon systems.
        # As a workaround, we roll our own implementation
        @parameter
        if has_neon() and type == DType.bfloat16:
            var int_self = bitcast[_integral_type_of[type](), size](self)
            var int_rhs = bitcast[_integral_type_of[type](), size](rhs)
            return int_self == int_rhs
        else:
            return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred eq>`](
                self.value, rhs.value
            )

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using not-equal comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is True or False depending on the expression
            `self[i] != rhs[i]`.
        """

        # TODO(KERN-228): support BF16 on neon systems.
        # As a workaround, we roll our own implementation.
        @parameter
        if has_neon() and type == DType.bfloat16:
            var int_self = bitcast[_integral_type_of[type](), size](self)
            var int_rhs = bitcast[_integral_type_of[type](), size](rhs)
            return int_self != int_rhs
        else:
            return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](
                self.value, rhs.value
            )

    @always_inline("nodebug")
    fn __gt__(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using greater-than comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is True or False depending on the expression
            `self[i] > rhs[i]`.
        """

        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred gt>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __ge__(self, rhs: Self) -> Self._Mask:
        """Compares two SIMD vectors using greater-than-or-equal comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is True or False depending on the expression
            `self[i] >= rhs[i]`.
        """

        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ge>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __pos__(self) -> Self:
        """Defines the unary `+` operation.

        Returns:
            This SIMD vector.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return self

    @always_inline("nodebug")
    fn __neg__(self) -> Self:
        """Defines the unary `-` operation.

        Returns:
            The negation of this SIMD vector.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return __mlir_op.`pop.neg`(self.value)

    @always_inline("nodebug")
    fn __and__(self, rhs: Self) -> Self:
        """Returns `self & rhs`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.

        Returns:
            `self & rhs`.
        """
        constrained[
            type.is_integral() or type.is_bool(),
            "must be an integral or bool type",
        ]()
        return __mlir_op.`pop.and`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __xor__(self, rhs: Self) -> Self:
        """Returns `self ^ rhs`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.

        Returns:
            `self ^ rhs`.
        """
        constrained[
            type.is_integral() or type.is_bool(),
            "must be an integral or bool type",
        ]()
        return __mlir_op.`pop.xor`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __or__(self, rhs: Self) -> Self:
        """Returns `self | rhs`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.

        Returns:
            `self | rhs`.
        """
        constrained[
            type.is_integral() or type.is_bool(),
            "must be an integral or bool type",
        ]()
        return __mlir_op.`pop.or`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __lshift__(self, rhs: Self) -> Self:
        """Returns `self << rhs`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            rhs: The RHS value.

        Returns:
            `self << rhs`.
        """
        constrained[type.is_integral(), "must be an integral type"]()
        debug_assert(all(rhs >= 0), "unhandled negative value")
        return __mlir_op.`pop.shl`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __rshift__(self, rhs: Self) -> Self:
        """Returns `self >> rhs`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            rhs: The RHS value.

        Returns:
            `self >> rhs`.
        """
        constrained[type.is_integral(), "must be an integral type"]()
        debug_assert(all(rhs >= 0), "unhandled negative value")
        return __mlir_op.`pop.shr`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __invert__(self) -> Self:
        """Returns `~self`.

        Constraints:
            The element type of the SIMD vector must be boolean or integral.

        Returns:
            The `~self` value.
        """
        constrained[
            type.is_bool() or type.is_integral(),
            "must be an bool or integral type",
        ]()

        @parameter
        if type.is_bool():
            return self.select(Self(False), Self(True))
        else:
            return self ^ -1

    # ===------------------------------------------------------------------=== #
    # In place operations.
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __iadd__(inout self, rhs: Self):
        """Performs in-place addition.

        The vector is mutated where each element at position `i` is computed as
        `self[i] + rhs[i]`.

        Args:
            rhs: The rhs of the addition operation.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        self = self + rhs

    @always_inline("nodebug")
    fn __isub__(inout self, rhs: Self):
        """Performs in-place subtraction.

        The vector is mutated where each element at position `i` is computed as
        `self[i] - rhs[i]`.

        Args:
            rhs: The rhs of the operation.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        self = self - rhs

    @always_inline("nodebug")
    fn __imul__(inout self, rhs: Self):
        """Performs in-place multiplication.

        The vector is mutated where each element at position `i` is computed as
        `self[i] * rhs[i]`.

        Args:
            rhs: The rhs of the operation.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        self = self * rhs

    @always_inline("nodebug")
    fn __itruediv__(inout self, rhs: Self):
        """In-place true divide operator.

        The vector is mutated where each element at position `i` is computed as
        `self[i] / rhs[i]`.

        Args:
            rhs: The rhs of the operation.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        self = self / rhs

    @always_inline("nodebug")
    fn __ifloordiv__(inout self, rhs: Self):
        """In-place flood div operator.

        The vector is mutated where each element at position `i` is computed as
        `self[i] // rhs[i]`.

        Args:
            rhs: The rhs of the operation.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        self = self // rhs

    @always_inline("nodebug")
    fn __imod__(inout self, rhs: Self):
        """In-place mod operator.

        The vector is mutated where each element at position `i` is computed as
        `self[i] % rhs[i]`.

        Args:
            rhs: The rhs of the operation.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        self = self.__mod__(rhs)

    @always_inline("nodebug")
    fn __ipow__(inout self, rhs: Int):
        """In-place pow operator.

        The vector is mutated where each element at position `i` is computed as
        `pow(self[i], rhs)`.

        Args:
            rhs: The rhs of the operation.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        self = self.__pow__(rhs)

    @always_inline("nodebug")
    fn __iand__(inout self, rhs: Self):
        """Computes `self & rhs` and save the result in `self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.
        """
        constrained[
            type.is_integral() or type.is_bool(),
            "must be an integral or bool type",
        ]()
        self = self & rhs

    @always_inline("nodebug")
    fn __ixor__(inout self, rhs: Self):
        """Computes `self ^ rhs` and save the result in `self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.
        """
        constrained[
            type.is_integral() or type.is_bool(),
            "must be an integral or bool type",
        ]()
        self = self ^ rhs

    @always_inline("nodebug")
    fn __ior__(inout self, rhs: Self):
        """Computes `self | rhs` and save the result in `self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            rhs: The RHS value.
        """
        constrained[
            type.is_integral() or type.is_bool(),
            "must be an integral or bool type",
        ]()
        self = self | rhs

    @always_inline("nodebug")
    fn __ilshift__(inout self, rhs: Self):
        """Computes `self << rhs` and save the result in `self`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            rhs: The RHS value.
        """
        constrained[type.is_integral(), "must be an integral type"]()
        self = self << rhs

    @always_inline("nodebug")
    fn __irshift__(inout self, rhs: Self):
        """Computes `self >> rhs` and save the result in `self`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            rhs: The RHS value.
        """
        constrained[type.is_integral(), "must be an integral type"]()
        self = self >> rhs

    # ===------------------------------------------------------------------=== #
    # Reversed operations
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __radd__(self, value: Self) -> Self:
        """Returns `value + self`.

        Args:
            value: The other value.

        Returns:
            `value + self`.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return value + self

    @always_inline("nodebug")
    fn __rsub__(self, value: Self) -> Self:
        """Returns `value - self`.

        Args:
            value: The other value.

        Returns:
            `value - self`.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return value - self

    @always_inline("nodebug")
    fn __rmul__(self, value: Self) -> Self:
        """Returns `value * self`.

        Args:
            value: The other value.

        Returns:
            `value * self`.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return value * self

    @always_inline("nodebug")
    fn __rfloordiv__(self, rhs: Self) -> Self:
        """Returns the division of rhs and self rounded down to the nearest
        integer.

        Constraints:
            The element type of the SIMD vector must be numeric.

        Args:
            rhs: The value to divide by self.

        Returns:
            `floor(rhs / self)` value.
        """
        constrained[type.is_numeric(), "the type must be numeric"]()
        return rhs // self

    @always_inline("nodebug")
    fn __rtruediv__(self, value: Self) -> Self:
        """Returns `value / self`.

        Args:
            value: The other value.

        Returns:
            `value / self`.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return value / self

    @always_inline("nodebug")
    fn __rmod__(self, value: Self) -> Self:
        """Returns `value mod self`.

        Args:
            value: The other value.

        Returns:
            `value mod self`.
        """
        constrained[type.is_numeric(), "the type must be numeric"]()
        return value % self

    @always_inline("nodebug")
    fn __rand__(self, value: Self) -> Self:
        """Returns `value & self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            value: The other value.

        Returns:
            `value & self`.
        """
        constrained[
            type.is_integral() or type.is_bool(),
            "must be an integral or bool type",
        ]()
        return value & self

    @always_inline("nodebug")
    fn __rxor__(self, value: Self) -> Self:
        """Returns `value ^ self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            value: The other value.

        Returns:
            `value ^ self`.
        """
        constrained[
            type.is_integral() or type.is_bool(),
            "must be an integral or bool type",
        ]()
        return value ^ self

    @always_inline("nodebug")
    fn __ror__(self, value: Self) -> Self:
        """Returns `value | self`.

        Constraints:
            The element type of the SIMD vector must be bool or integral.

        Args:
            value: The other value.

        Returns:
            `value | self`.
        """
        constrained[
            type.is_integral() or type.is_bool(),
            "must be an integral or bool type",
        ]()
        return value | self

    @always_inline("nodebug")
    fn __rlshift__(self, value: Self) -> Self:
        """Returns `value << self`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            value: The other value.

        Returns:
            `value << self`.
        """
        constrained[type.is_integral(), "must be an integral type"]()
        return value << self

    @always_inline("nodebug")
    fn __rrshift__(self, value: Self) -> Self:
        """Returns `value >> self`.

        Constraints:
            The element type of the SIMD vector must be integral.

        Args:
            value: The other value.

        Returns:
            `value >> self`.
        """
        constrained[type.is_integral(), "must be an integral type"]()
        return value >> self

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Gets the length of the SIMD vector.

        Returns:
            The length of the SIMD vector.
        """

        return self.size

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        """Converts the SIMD scalar into a boolean value.

        Constraints:
            The size of the SIMD vector must be 1.

        Returns:
            True if the SIMD scalar is non-zero and False otherwise.
        """
        constrained[
            size == 1,
            (
                "The truth value of a SIMD vector with more than one element is"
                " ambiguous. Use the builtin `any()` or `all()` functions"
                " instead."
            ),
        ]()
        return rebind[Scalar[DType.bool]](self.cast[DType.bool]()).value

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Casts to the value to an Int. If there is a fractional component,
        then the fractional part is truncated.

        Constraints:
            The size of the SIMD vector must be 1.

        Returns:
            The value as an integer.
        """
        constrained[size == 1, "expected a scalar type"]()
        return __mlir_op.`pop.cast`[_type = __mlir_type.`!pop.scalar<index>`](
            rebind[Scalar[type]](self).value
        )

    @always_inline
    fn __str__(self) -> String:
        """Get the SIMD as a string.

        Returns:
            A string representation.
        """

        return String.format_sequence(self)

    @always_inline
    fn __repr__(self) -> String:
        """Get the representation of the SIMD value e.g. "SIMD[DType.int8, 2](1, 2)".

        Returns:
            The representation of the SIMD value.
        """

        var output = String()
        var writer = output._unsafe_to_formatter()
        self.format_to[use_scientific_notation=True](writer)

        var values = output.as_string_slice()

        @parameter
        if size > 1:
            # TODO: Fix when slice indexing is implemented on StringSlice
            values = StringSlice(unsafe_from_utf8=output.as_bytes_slice()[1:-1])
        return (
            "SIMD[" + type.__repr__() + ", " + str(size) + "](" + values + ")"
        )

    @always_inline("nodebug")
    fn __floor__(self) -> Self:
        """Performs elementwise floor on the elements of a SIMD vector.

        Returns:
            The elementwise floor of this SIMD vector.
        """
        return self._floor_ceil_trunc_impl["llvm.floor"]()

    @always_inline("nodebug")
    fn __ceil__(self) -> Self:
        """Performs elementwise ceiling on the elements of a SIMD vector.

        Returns:
            The elementwise ceiling of this SIMD vector.
        """
        return self._floor_ceil_trunc_impl["llvm.ceil"]()

    @always_inline("nodebug")
    fn __trunc__(self) -> Self:
        """Performs elementwise truncation on the elements of a SIMD vector.

        Returns:
            The elementwise truncated values of this SIMD vector.
        """

        return self._floor_ceil_trunc_impl["llvm.trunc"]()

    @always_inline
    fn __abs__(self) -> Self:
        """Defines the absolute value operation.

        Returns:
            The absolute value of this SIMD vector.
        """

        @parameter
        if type.is_unsigned() or type.is_bool():
            return self
        elif type.is_floating_point():
            alias integral_type = FPUtils[type].integral_type
            var m = self._float_to_bits[integral_type]()
            return (m & (FPUtils[type].sign_mask() - 1))._bits_to_float[type]()
        else:
            return (self < 0).select(-self, self)

    @always_inline("nodebug")
    fn __round__(self) -> Self:
        """Performs elementwise rounding on the elements of a SIMD vector.

        This rounding goes to the nearest integer with ties away from zero.

        Returns:
            The elementwise rounded value of this SIMD vector.
        """
        return llvm_intrinsic["llvm.round", Self, has_side_effect=False](self)

    @always_inline("nodebug")
    fn __round__(self, ndigits: Int) -> Self:
        """Performs elementwise rounding on the elements of a SIMD vector.
        This rounding goes to the nearest integer with ties away from zero.
        Args:
            ndigits: The number of digits to round to.
        Returns:
            The elementwise rounded value of this SIMD vector.
        """
        # TODO: see how can we implement this.
        return llvm_intrinsic["llvm.round", Self, has_side_effect=False](self)

    fn __hash__(self) -> Int:
        """Hash the value using builtin hash.

        Returns:
            A 64-bit hash value. This value is _not_ suitable for cryptographic
            uses. Its intended usage is for data structures. See the `hash`
            builtin documentation for more details.
        """
        return _hash_simd(self)

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn cast[target: DType](self) -> SIMD[target, size]:
        """Casts the elements of the SIMD vector to the target element type.

        Parameters:
            target: The target DType.

        Returns:
            A new SIMD vector whose elements have been casted to the target
            element type.
        """

        @parameter
        if type == target:
            return rebind[SIMD[target, size]](self)
        elif has_neon() and (
            type == DType.bfloat16 or target == DType.bfloat16
        ):
            # TODO(KERN-228): support BF16 on neon systems.
            return _unchecked_zero[target, size]()
        elif type == DType.bool:
            return self.select(SIMD[target, size](1), SIMD[target, size](0))
        elif target == DType.bool:
            return rebind[SIMD[target, size]](self != 0)
        elif type == DType.bfloat16:
            var cast_result = _bfloat16_to_f32(
                rebind[SIMD[DType.bfloat16, size]](self)
            ).cast[target]()
            return rebind[SIMD[target, size]](cast_result)
        elif target == DType.bfloat16:
            return rebind[SIMD[target, size]](
                _f32_to_bfloat16(self.cast[DType.float32]())
            )
        elif target == DType.address:
            var index_val = __mlir_op.`pop.cast`[
                _type = __mlir_type[`!pop.simd<`, size.value, `, index>`]
            ](self.value)
            var tmp = SIMD[DType.address, size](
                __mlir_op.`pop.index_to_pointer`[
                    _type = __mlir_type[
                        `!pop.simd<`,
                        size.value,
                        `, address >`,
                    ]
                ](index_val)
            )
            return rebind[SIMD[target, size]](tmp)
        elif (type == DType.address) and target.is_integral():
            var index_tmp = SIMD[DType.index, size](
                __mlir_op.`pop.pointer_to_index`[
                    _type = __mlir_type[
                        `!pop.simd<`,
                        size.value,
                        `, `,
                        DType.index.value,
                        `>`,
                    ]
                ](
                    rebind[
                        __mlir_type[
                            `!pop.simd<`,
                            size.value,
                            `, address >`,
                        ]
                    ](self.value)
                )
            )
            return index_tmp.cast[target]()
        else:
            return __mlir_op.`pop.cast`[
                _type = __mlir_type[
                    `!pop.simd<`,
                    size.value,
                    `, `,
                    target.value,
                    `>`,
                ]
            ](self.value)

    @always_inline
    fn format_to(self, inout writer: Formatter):
        """
        Formats this SIMD value to the provided formatter.

        Args:
            writer: The formatter to write to.
        """
        self.format_to[use_scientific_notation=False](writer)

    # This overload is required to keep SIMD compliant with the Formattable
    # trait, and the call to `String.format_sequence(self)` in SIMD.__str__ will
    # fail to compile.
    fn format_to[use_scientific_notation: Bool](self, inout writer: Formatter):
        """
        Formats this SIMD value to the provided formatter.

        Parameters:
            use_scientific_notation: Whether floats should use scientific notation.
                This parameter does not apply to integer types.

        Args:
            writer: The formatter to write to.
        """

        # Print an opening `[`.
        @parameter
        if size > 1:
            writer.write_str["["]()

        # Print each element.
        for i in range(size):
            var element = self[i]
            # Print separators between each element.
            if i != 0:
                writer.write_str[", "]()

            @parameter
            if triple_is_nvidia_cuda():

                @parameter
                if type.is_floating_point():
                    # get_dtype_printf_format hardcodes 17 digits of precision.
                    _printf["%g"](element)
                else:
                    # FIXME(MSTDL-406):
                    #   This prints "out of band" with the `Formatter` passed
                    #   in, meaning this will only work if `Formatter` is an
                    #   unbuffered wrapper around printf (which Formatter.stdout
                    #   currently is by default).
                    #
                    #   This is a workaround to permit debug formatting of
                    #   floating-point values on GPU, where printing to stdout
                    #   is the only way the Formatter framework is currently
                    #   used.
                    _printf[_get_dtype_printf_format[type]()](element)
            else:

                @parameter
                if use_scientific_notation and type.is_floating_point():
                    alias float_format = "%." + _scientific_notation_digits[
                        type
                    ]() + "e"
                    _format_scalar[type, float_format](writer, element)
                else:
                    _format_scalar(writer, element)

        # Print a closing `]`.
        @parameter
        if size > 1:
            writer.write_str["]"]()

    @always_inline
    fn _bits_to_float[dest_type: DType](self) -> SIMD[dest_type, size]:
        """Bitcasts the integer value to a floating-point value.

        Parameters:
            dest_type: DType to bitcast the input SIMD vector to.

        Returns:
            A floating-point representation of the integer value.
        """
        alias integral_type = FPUtils[type].integral_type
        return bitcast[dest_type, size](self.cast[integral_type]())

    @always_inline
    fn _float_to_bits[dest_type: DType](self) -> SIMD[dest_type, size]:
        """Bitcasts the floating-point value to an integer value.

        Parameters:
            dest_type: DType to bitcast the input SIMD vector to.

        Returns:
            An integer representation of the floating-point value.
        """
        alias integral_type = FPUtils[type].integral_type
        var v = bitcast[integral_type, size](self)
        return v.cast[dest_type]()

    fn _floor_ceil_trunc_impl[intrinsic: StringLiteral](self) -> Self:
        constrained[
            intrinsic == "llvm.floor"
            or intrinsic == "llvm.ceil"
            or intrinsic == "llvm.trunc",
            "unsupported intrinsic",
        ]()

        @parameter
        if type.is_bool() or type.is_integral():
            return self
        elif has_neon() and type == DType.bfloat16:
            # TODO(KERN-228): support BF16 on neon systems.
            # As a workaround, we cast to float32.
            return (
                self.cast[DType.float32]()
                ._floor_ceil_trunc_impl[intrinsic]()
                .cast[type]()
            )
        else:
            return llvm_intrinsic[intrinsic, Self, has_side_effect=False](self)

    fn clamp(self, lower_bound: Self, upper_bound: Self) -> Self:
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

        return self.min(upper_bound).max(lower_bound)

    @always_inline("nodebug")
    fn roundeven(self) -> Self:
        """Performs elementwise banker's rounding on the elements of a SIMD
        vector.

        This rounding goes to the nearest integer with ties toward the nearest
        even integer.

        Returns:
            The elementwise banker's rounding of this SIMD vector.
        """
        return llvm_intrinsic["llvm.roundeven", Self, has_side_effect=False](
            self
        )

    @always_inline
    fn add_with_overflow(self, rhs: Self) -> (Self, Self._Mask):
        """Computes `self + rhs` and a mask of which indices overflowed.

        Args:
            rhs: The rhs value.

        Returns:
            A tuple with the results of the operation and a mask for overflows.
            The first is a new vector whose element at position `i` is computed
            as `self[i] + rhs[i]`. The second item is a vector of booleans where
            a `1` at position `i` represents `self[i] + rhs[i]` overflowed.
        """
        constrained[type.is_integral()]()

        @parameter
        if type.is_signed():
            var result = llvm_intrinsic[
                "llvm.sadd.with.overflow",
                _RegisterPackType[Self, Self._Mask],
                Self,
                Self,
            ](self, rhs)
            return (result[0], result[1])
        else:
            var result = llvm_intrinsic[
                "llvm.uadd.with.overflow",
                _RegisterPackType[Self, Self._Mask],
                Self,
                Self,
            ](self, rhs)
            return (result[0], result[1])

    @always_inline
    fn sub_with_overflow(self, rhs: Self) -> (Self, Self._Mask):
        """Computes `self - rhs` and a mask of which indices overflowed.

        Args:
            rhs: The rhs value.

        Returns:
            A tuple with the results of the operation and a mask for overflows.
            The first is a new vector whose element at position `i` is computed
            as `self[i] - rhs[i]`. The second item is a vector of booleans where
            a `1` at position `i` represents `self[i] - rhs[i]` overflowed.
        """
        constrained[type.is_integral()]()

        @parameter
        if type.is_signed():
            var result = llvm_intrinsic[
                "llvm.ssub.with.overflow",
                _RegisterPackType[Self, Self._Mask],
                Self,
                Self,
            ](self, rhs)
            return (result[0], result[1])
        else:
            var result = llvm_intrinsic[
                "llvm.usub.with.overflow",
                _RegisterPackType[Self, Self._Mask],
                Self,
                Self,
            ](self, rhs)
            return (result[0], result[1])

    @always_inline
    fn mul_with_overflow(self, rhs: Self) -> (Self, Self._Mask):
        """Computes `self * rhs` and a mask of which indices overflowed.

        Args:
            rhs: The rhs value.

        Returns:
            A tuple with the results of the operation and a mask for overflows.
            The first is a new vector whose element at position `i` is computed
            as `self[i] * rhs[i]`. The second item is a vector of booleans where
            a `1` at position `i` represents `self[i] * rhs[i]` overflowed.
        """
        constrained[type.is_integral()]()

        @parameter
        if type.is_signed():
            var result = llvm_intrinsic[
                "llvm.smul.with.overflow",
                _RegisterPackType[Self, Self._Mask],
                Self,
                Self,
            ](self, rhs)
            return (result[0], result[1])
        else:
            var result = llvm_intrinsic[
                "llvm.umul.with.overflow",
                _RegisterPackType[Self, Self._Mask],
                Self,
                Self,
            ](self, rhs)
            return (result[0], result[1])

    # TODO: Move to global function.
    @always_inline("nodebug")
    fn fma(self, multiplier: Self, accumulator: Self) -> Self:
        """Performs a fused multiply-add operation, i.e.
        `self*multiplier + accumulator`.

        Args:
            multiplier: The value to multiply.
            accumulator: The value to accumulate.

        Returns:
            A new vector whose element at position `i` is computed as
            `self[i]*multiplier[i] + accumulator[i]`.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return __mlir_op.`pop.fma`(
            self.value, multiplier.value, accumulator.value
        )

    @always_inline("nodebug")
    fn _shuffle_list[
        *mask: Int, output_size: Int = size
    ](self, other: Self) -> SIMD[type, output_size]:
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

        @parameter
        fn variadic_len[*mask: Int]() -> Int:
            return __mlir_op.`pop.variadic.size`(mask)

        @parameter
        fn _convert_variadic_to_pop_array[
            *mask: Int
        ]() -> __mlir_type[`!pop.array<`, output_size.value, `, `, Int, `>`]:
            var array = __mlir_op.`kgen.undef`[
                _type = __mlir_type[
                    `!pop.array<`, output_size.value, `, `, Int, `>`
                ]
            ]()

            @parameter
            for idx in range(output_size):
                alias val = mask[idx]
                constrained[
                    0 <= val < 2 * size,
                    "invalid index in the shuffle operation",
                ]()
                var ptr = __mlir_op.`pop.array.gep`(
                    UnsafePointer.address_of(array).address, idx.value
                )
                __mlir_op.`pop.store`(val, ptr)

            return array

        alias length = variadic_len[mask]()
        constrained[
            output_size == length,
            "size of the mask must match the output SIMD size",
        ]()
        return __mlir_op.`pop.simd.shuffle`[
            mask = _convert_variadic_to_pop_array[mask](),
            _type = __mlir_type[
                `!pop.simd<`, output_size.value, `, `, type.value, `>`
            ],
        ](self.value, other.value)

    @always_inline("nodebug")
    fn _shuffle_list[
        output_size: Int, mask: StaticIntTuple[output_size]
    ](self, other: Self) -> SIMD[type, output_size]:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask
        values must be within `2 * len(self)`.

        Parameters:
            output_size: The size of the output vector.
            mask: The permutation to use in the shuffle.

        Args:
            other: The other vector to shuffle with.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is `(self + other)[permutation[i]]`.
        """

        @parameter
        for i in range(output_size):
            constrained[
                0 <= mask[i] < 2 * size,
                "invalid index in the shuffle operation",
            ]()

        return __mlir_op.`pop.simd.shuffle`[
            mask = mask.data.array,
            _type = __mlir_type[
                `!pop.simd<`, output_size.value, `, `, type.value, `>`
            ],
        ](self.value, other.value)

    @always_inline("nodebug")
    fn shuffle[*mask: Int](self) -> Self:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask
        values must be within `2 * len(self)`.

        Parameters:
            mask: The permutation to use in the shuffle.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is `(self)[permutation[i]]`.
        """
        return self._shuffle_list[mask](self)

    @always_inline("nodebug")
    fn shuffle[*mask: Int](self, other: Self) -> Self:
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
        return self._shuffle_list[mask](other)

    @always_inline("nodebug")
    fn shuffle[mask: StaticIntTuple[size]](self) -> Self:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask
        values must be within `2 * len(self)`.

        Parameters:
            mask: The permutation to use in the shuffle.

        Returns:
            A new vector with the same length as the mask where the value at
            position `i` is `(self)[permutation[i]]`.
        """
        return self._shuffle_list[size, mask](self)

    @always_inline("nodebug")
    fn shuffle[mask: StaticIntTuple[size]](self, other: Self) -> Self:
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
        return self._shuffle_list[size, mask](other)

    @always_inline("nodebug")
    fn slice[
        output_width: Int, /, *, offset: Int = 0
    ](self) -> SIMD[type, output_width]:
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
        constrained[
            0 < output_width + offset <= size,
            "output width must be a positive integer less than simd size",
        ]()

        @parameter
        if output_width == 1:
            return self[offset]

        @parameter
        if offset % simdwidthof[type]():
            var tmp = SIMD[type, output_width]()

            @parameter
            for i in range(output_width):
                tmp[i] = self[i + offset]
            return tmp

        return llvm_intrinsic[
            "llvm.vector.extract",
            SIMD[type, output_width],
            has_side_effect=False,
        ](self, offset)

    @always_inline("nodebug")
    fn insert[*, offset: Int = 0](self, value: SIMD[type, _]) -> Self:
        """Returns a the vector where the elements between `offset` and
        `offset + input_width` have been replaced with the elements in `value`.

        Parameters:
            offset: The offset to insert at.

        Args:
            value: The value to be inserted.

        Returns:
            A new vector whose elements at `self[offset:offset+input_width]`
            contain the values of `value`.
        """
        alias input_width = value.size
        constrained[
            0 < input_width + offset <= size,
            "insertion position must not exceed the size of the vector",
        ]()

        @parameter
        if size == 1:
            constrained[
                input_width == 1, "the input width must be 1 if the size is 1"
            ]()
            return rebind[Self](value)

        # You cannot insert into a SIMD value at positions that are not a
        # multiple of the SIMD width via the `llvm.vector.insert` intrinsic,
        # so resort to a for loop. Note that this can be made more intelligent
        # by dividing the problem into the offset, offset+val, val+input_width
        # where val is a value to align the offset to the simdwidth.
        @parameter
        if offset % simdwidthof[type]():
            var tmp = self

            @parameter
            for i in range(input_width):
                tmp[i + offset] = value[i]
            return tmp

        return llvm_intrinsic[
            "llvm.vector.insert", Self, has_side_effect=False
        ](self, value, offset)

    @always_inline("nodebug")
    fn join(self, other: Self) -> SIMD[type, 2 * size]:
        """Concatenates the two vectors together.

        Args:
            other: The other SIMD vector.

        Returns:
            A new vector `self_0, self_1, ..., self_n, other_0, ..., other_n`.
        """

        @always_inline
        @parameter
        fn build_indices() -> StaticIntTuple[2 * size]:
            var indices = StaticIntTuple[2 * size]()

            @parameter
            for i in range(2 * size):
                indices[i] = i

            return indices

        return self._shuffle_list[2 * size, build_indices()](other)

    @always_inline("nodebug")
    fn interleave(self, other: Self) -> SIMD[type, 2 * size]:
        """Constructs a vector by interleaving two input vectors.

        Args:
            other: The other SIMD vector.

        Returns:
            A new vector `self_0, other_0, ..., self_n, other_n`.
        """

        @parameter
        if size == 1:
            return SIMD[type, 2 * size](self[0], other[0])

        return llvm_intrinsic[
            "llvm.vector.interleave2",
            SIMD[type, 2 * size],
            has_side_effect=False,
        ](self, other)

    alias _SIMDHalfType = SIMD[type, size // 2]

    @always_inline("nodebug")
    fn deinterleave(
        self,
    ) -> (Self._SIMDHalfType, Self._SIMDHalfType):
        """Constructs two vectors by deinterleaving the even and odd lanes of
        the vector.

        Constraints:
            The vector size must be greater than 1.

        Returns:
            Two vectors the first of the form `self_0, self_2, ..., self_{n-2}`
            and the other being `self_1, self_3, ..., self_{n-1}`.
        """

        constrained[size > 1, "the vector size must be greater than 1."]()

        @parameter
        if size == 2:
            return (
                rebind[Self._SIMDHalfType](self[0]),
                rebind[Self._SIMDHalfType](self[1]),
            )

        var res = llvm_intrinsic[
            "llvm.vector.deinterleave2",
            _RegisterPackType[Self._SIMDHalfType, Self._SIMDHalfType],
            has_side_effect=False,
        ](self)
        return (
            rebind[Self._SIMDHalfType](res[0]),
            rebind[Self._SIMDHalfType](res[1]),
        )

    @always_inline("nodebug")
    fn min(self, other: Self) -> Self:
        """Computes the elementwise minimum between the two vectors.

        Args:
            other: The other SIMD vector.

        Returns:
            A new SIMD vector where each element at position `i` is
            `min(self[i], other[i])`.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return __mlir_op.`pop.min`(self.value, other.value)

    @always_inline("nodebug")
    fn max(self, other: Self) -> Self:
        """Computes the elementwise maximum between the two vectors.

        Args:
            other: The other SIMD vector.

        Returns:
            A new SIMD vector where each element at position `i` is
            `max(self[i], other[i])`.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return __mlir_op.`pop.max`(self.value, other.value)

    # ===------------------------------------------------------------------=== #
    # Reduce operations
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn reduce[
        func: fn[type: DType, width: Int] (
            SIMD[type, width], SIMD[type, width]
        ) capturing -> SIMD[type, width],
        size_out: Int = 1,
    ](self) -> SIMD[type, size_out]:
        """Reduces the vector using a provided reduce operator.

        Parameters:
            func: The reduce function to apply to elements in this SIMD.
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.

        Returns:
            A new scalar which is the reduction of all vector elements.
        """
        constrained[size_out <= size, "reduction cannot increase simd width"]()

        @parameter
        if size == size_out:
            return rebind[SIMD[type, size_out]](self)
        else:
            alias half_size = size // 2
            var lhs = self.slice[half_size, offset=0]()
            var rhs = self.slice[half_size, offset=half_size]()
            return func[type, half_size](lhs, rhs).reduce[func, size_out]()

    @always_inline("nodebug")
    fn reduce_max[size_out: Int = 1](self) -> SIMD[type, size_out]:
        """Reduces the vector using the `max` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.
            The element type of the vector must be integer or FP.

        Returns:
            The maximum element of the vector.
        """

        @parameter
        if size == 1:
            return self[0]

        @parameter
        if is_x86() or size_out > 1:

            @always_inline
            @parameter
            fn max_reduce_body[
                type: DType, width: Int
            ](v1: SIMD[type, width], v2: SIMD[type, width]) -> SIMD[
                type, width
            ]:
                return v1.max(v2)

            return self.reduce[max_reduce_body, size_out]()

        @parameter
        if type.is_floating_point():
            return rebind[SIMD[type, size_out]](
                llvm_intrinsic[
                    "llvm.vector.reduce.fmax",
                    Scalar[type],
                    has_side_effect=False,
                ](self)
            )

        @parameter
        if type.is_unsigned():
            return rebind[SIMD[type, size_out]](
                llvm_intrinsic[
                    "llvm.vector.reduce.umax",
                    Scalar[type],
                    has_side_effect=False,
                ](self)
            )
        return rebind[SIMD[type, size_out]](
            llvm_intrinsic[
                "llvm.vector.reduce.smax", Scalar[type], has_side_effect=False
            ](self)
        )

    @always_inline("nodebug")
    fn reduce_min[size_out: Int = 1](self) -> SIMD[type, size_out]:
        """Reduces the vector using the `min` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.
            The element type of the vector must be integer or FP.

        Returns:
            The minimum element of the vector.
        """

        @parameter
        if size == 1:
            return self[0]

        @parameter
        if is_x86() or size_out > 1:

            @always_inline
            @parameter
            fn min_reduce_body[
                type: DType, width: Int
            ](v1: SIMD[type, width], v2: SIMD[type, width]) -> SIMD[
                type, width
            ]:
                return v1.min(v2)

            return self.reduce[min_reduce_body, size_out]()

        @parameter
        if type.is_floating_point():
            return rebind[SIMD[type, size_out]](
                llvm_intrinsic[
                    "llvm.vector.reduce.fmin",
                    Scalar[type],
                    has_side_effect=False,
                ](self)
            )

        @parameter
        if type.is_unsigned():
            return rebind[SIMD[type, size_out]](
                llvm_intrinsic[
                    "llvm.vector.reduce.umin",
                    Scalar[type],
                    has_side_effect=False,
                ](self)
            )
        return rebind[SIMD[type, size_out]](
            llvm_intrinsic[
                "llvm.vector.reduce.smin", Scalar[type], has_side_effect=False
            ](self)
        )

    @always_inline
    fn reduce_add[size_out: Int = 1](self) -> SIMD[type, size_out]:
        """Reduces the vector using the `add` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.

        Returns:
            The sum of all vector elements.

        """

        @always_inline
        @parameter
        fn add_reduce_body[
            type: DType, width: Int
        ](v1: SIMD[type, width], v2: SIMD[type, width]) -> SIMD[type, width]:
            return v1 + v2

        return self.reduce[add_reduce_body, size_out]()

    @always_inline
    fn reduce_mul[size_out: Int = 1](self) -> SIMD[type, size_out]:
        """Reduces the vector using the `mul` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.
            The element type of the vector must be integer or FP.

        Returns:
            The product of all vector elements.
        """

        @always_inline
        @parameter
        fn mul_reduce_body[
            type: DType, width: Int
        ](v1: SIMD[type, width], v2: SIMD[type, width]) -> SIMD[type, width]:
            return v1 * v2

        return self.reduce[mul_reduce_body, size_out]()

    @always_inline
    fn reduce_and[size_out: Int = 1](self) -> SIMD[type, size_out]:
        """Reduces the vector using the bitwise `&` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.
            The element type of the vector must be integer or boolean.

        Returns:
            The reduced vector.
        """
        constrained[
            size_out <= size, "`size_out` must not exceed width of the vector."
        ]()
        constrained[
            type.is_integral() or type.is_bool(),
            "The element type of the vector must be integer or boolean.",
        ]()

        @parameter
        if size_out > 1:

            @always_inline
            @parameter
            fn and_reduce_body[
                type: DType, width: Int
            ](v1: SIMD[type, width], v2: SIMD[type, width]) -> SIMD[
                type, width
            ]:
                return v1 & v2

            return self.reduce[and_reduce_body, size_out]()

        @parameter
        if size == 1:
            return rebind[SIMD[type, size_out]](self)

        return llvm_intrinsic[
            "llvm.vector.reduce.and",
            SIMD[type, size_out],
            has_side_effect=False,
        ](self)

    @always_inline
    fn reduce_or[size_out: Int = 1](self) -> SIMD[type, size_out]:
        """Reduces the vector using the bitwise `|` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
            `size_out` must not exceed width of the vector.
            The element type of the vector must be integer or boolean.

        Returns:
            The reduced vector.
        """
        constrained[
            size_out <= size, "`size_out` must not exceed width of the vector."
        ]()
        constrained[
            type.is_integral() or type.is_bool(),
            "The element type of the vector must be integer or boolean.",
        ]()

        @parameter
        if size_out > 1:

            @always_inline
            @parameter
            fn or_reduce_body[
                type: DType, width: Int
            ](v1: SIMD[type, width], v2: SIMD[type, width]) -> SIMD[
                type, width
            ]:
                return v1 | v2

            return self.reduce[or_reduce_body, size_out]()

        @parameter
        if size == 1:
            return rebind[SIMD[type, size_out]](self)

        return llvm_intrinsic[
            "llvm.vector.reduce.or", SIMD[type, size_out], has_side_effect=False
        ](self)

    @always_inline
    fn reduce_bit_count(self) -> Int:
        """Returns the total number of bits set in the SIMD vector.

        Constraints:
            Must be either an integral or a boolean type.

        Returns:
            Count of set bits across all elements of the vector.
        """

        @parameter
        if type.is_bool():
            return int(self.cast[DType.uint8]().reduce_add())
        else:
            constrained[
                type.is_integral(), "Expected either integral or bool type"
            ]()
            return int(pop_count(self).reduce_add())

    # ===------------------------------------------------------------------=== #
    # select
    # ===------------------------------------------------------------------=== #

    # TODO (7748): always_inline required to WAR LLVM codegen bug
    @always_inline("nodebug")
    fn select[
        result_type: DType
    ](
        self,
        true_case: SIMD[result_type, size],
        false_case: SIMD[result_type, size],
    ) -> SIMD[result_type, size]:
        """Selects the values of the `true_case` or the `false_case` based on
        the current boolean values of the SIMD vector.

        Parameters:
            result_type: The element type of the input and output SIMD vectors.

        Args:
            true_case: The values selected if the positional value is True.
            false_case: The values selected if the positional value is False.

        Constraints:
            The element type of the vector must be boolean.

        Returns:
            A new vector of the form
            `[true_case[i] if elem else false_case[i] for i, elem in enumerate(self)]`.
        """
        constrained[type.is_bool(), "the simd dtype must be bool"]()
        return __mlir_op.`pop.simd.select`(
            rebind[Self._Mask](self).value,
            true_case.value,
            false_case.value,
        )

    # ===------------------------------------------------------------------=== #
    # Rotation operations
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn rotate_left[shift: Int](self) -> Self:
        """Shifts the elements of a SIMD vector to the left by `shift`
        elements (with wrap-around).

        Constraints:
            `-size <= shift < size`

        Parameters:
            shift: The number of positions by which to rotate the elements of
                   SIMD vector to the left (with wrap-around).

        Returns:
            The SIMD vector rotated to the left by `shift` elements
            (with wrap-around).
        """

        constrained[
            shift >= -size and shift < size,
            "Constraint: -size <= shift < size",
        ]()

        @parameter
        if size == 1:
            constrained[shift == 0, "for scalars the shift must be 0"]()
            return self
        return llvm_intrinsic[
            "llvm.vector.splice", Self, has_side_effect=False
        ](self, self, Int32(shift))

    @always_inline
    fn rotate_right[shift: Int](self) -> Self:
        """Shifts the elements of a SIMD vector to the right by `shift`
        elements (with wrap-around).

        Constraints:
            `-size < shift <= size`

        Parameters:
            shift: The number of positions by which to rotate the elements of
                   SIMD vector to the right (with wrap-around).

        Returns:
            The SIMD vector rotated to the right by `shift` elements
            (with wrap-around).
        """

        constrained[
            shift > -size and shift <= size,
            "Constraint: -size < shift <= size",
        ]()

        @parameter
        if size == 1:
            constrained[shift == 0, "for scalars the shift must be 0"]()
            return self
        return self.rotate_left[-shift]()

    # ===------------------------------------------------------------------=== #
    # Shift operations
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn shift_left[shift: Int](self) -> Self:
        """Shifts the elements of a SIMD vector to the left by `shift`
        elements (no wrap-around, fill with zero).

        Constraints:
            `0 <= shift <= size`

        Parameters:
            shift: The number of positions by which to rotate the elements of
                   SIMD vector to the left (no wrap-around, fill with zero).

        Returns:
            The SIMD vector rotated to the left by `shift` elements (no
            wrap-around, fill with zero).
        """

        constrained[
            0 <= shift <= size,
            (
                "shift must be greater than or equal to 0 and less than equal"
                " to the size"
            ),
        ]()

        @parameter
        if shift == 0:
            return self
        elif shift == size:
            return 0

        alias zero_simd = Self()

        return llvm_intrinsic[
            "llvm.vector.splice", Self, has_side_effect=False
        ](self, zero_simd, Int32(shift))

    @always_inline
    fn shift_right[shift: Int](self) -> Self:
        """Shifts the elements of a SIMD vector to the right by `shift`
        elements (no wrap-around, fill with zero).

        Constraints:
            `0 <= shift <= size`

        Parameters:
            shift: The number of positions by which to rotate the elements of
                   SIMD vector to the right (no wrap-around, fill with zero).

        Returns:
            The SIMD vector rotated to the right by `shift` elements (no
            wrap-around, fill with zero).
        """

        # Note the order of the llvm_intrinsic arguments below differ from
        # shift_left(), so we cannot directly reuse it here.

        constrained[
            0 <= shift <= size,
            (
                "shift must be greater than or equal to 0 and less than equal"
                " to the size"
            ),
        ]()

        @parameter
        if shift == 0:
            return self
        elif shift == size:
            return 0

        alias zero_simd = Self()

        return llvm_intrinsic[
            "llvm.vector.splice", Self, has_side_effect=False
        ](zero_simd, self, Int32(-shift))

    @staticmethod
    @always_inline
    fn prefetch[
        params: PrefetchOptions,
        *,
        address_space: AddressSpace = AddressSpace.GENERIC,
    ](ptr: DTypePointer[type, address_space]):
        # Prefetch at the underlying address.
        """Prefetches memory at the underlying address.

        Parameters:
            params: Prefetch options (see `PrefetchOptions` for details).
            address_space: The address space the pointer is in.

        Args:
            ptr: The pointer to prefetch from.
        """
        prefetch[params](ptr)

    @staticmethod
    @always_inline("nodebug")
    fn load[
        *,
        alignment: Int = Self._default_alignment,
        address_space: AddressSpace = AddressSpace.GENERIC,
    ](ptr: DTypePointer[type, address_space]) -> Self:
        """Loads the value the Pointer object points to.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            alignment: The minimal alignment of the address.
            address_space: The address space the pointer is in.

        Args:
            ptr: The pointer to load from.

        Returns:
            The loaded value.
        """
        return Self.load[alignment=alignment, address_space=address_space](
            ptr, 0
        )

    @staticmethod
    @always_inline("nodebug")
    fn load[
        T: Intable,
        *,
        alignment: Int = Self._default_alignment,
        address_space: AddressSpace = AddressSpace.GENERIC,
    ](ptr: DTypePointer[type, address_space], offset: T) -> Self:
        """Loads the value the Pointer object points to with the given offset.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            T: The Intable type of the offset.
            alignment: The minimal alignment of the address.
            address_space: The address space the pointer is in.

        Args:
            ptr: The pointer to load from.
            offset: The offset to load from.

        Returns:
            The loaded value.
        """

        @parameter
        if triple_is_nvidia_cuda() and sizeof[type]() == 1 and alignment == 1:
            # LLVM lowering to PTX incorrectly vectorizes loads for 1-byte types
            # regardless of the alignment that is passed. This causes issues if
            # this method is called on an unaligned pointer.
            # TODO #37823 We can make this smarter when we add an `aligned`
            # trait to the pointer class.
            var v = SIMD[type, size]()

            # intentionally don't unroll, otherwise the compiler vectorizes
            for i in range(size):
                v[i] = ptr.address.offset(int(offset) + i).load[
                    alignment=alignment
                ]()
            return v

        return (
            ptr.address.offset(offset)
            .bitcast[SIMD[type, size]]()
            .load[alignment=alignment]()
        )

    @staticmethod
    @always_inline("nodebug")
    fn store[
        T: Intable,
        /,
        *,
        alignment: Int = Self._default_alignment,
        address_space: AddressSpace = AddressSpace.GENERIC,
    ](ptr: DTypePointer[type, address_space], offset: T, val: Self):
        """Stores a single element value at the given offset.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            T: The Intable type of the offset.
            alignment: The minimal alignment of the address.
            address_space: The address space the pointer is in.

        Args:
            ptr: The pointer to store to.
            offset: The offset to store to.
            val: The value to store.
        """
        Self.store[alignment=alignment, address_space=address_space](
            ptr.offset(offset), val
        )

    @staticmethod
    @always_inline("nodebug")
    fn store[
        *,
        alignment: Int = Self._default_alignment,
        address_space: AddressSpace = AddressSpace.GENERIC,
    ](ptr: DTypePointer[type, address_space], val: Self):
        """Stores a single element value.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            alignment: The minimal alignment of the address.
            address_space: The address space the pointer is in.

        Args:
            ptr: The pointer to store to.
            val: The value to store.
        """
        constrained[size > 0, "width must be a positive integer value"]()
        constrained[
            alignment > 0, "alignment must be a positive integer value"
        ]()
        ptr.address.bitcast[SIMD[type, size]]().store[alignment=alignment](val)


# ===----------------------------------------------------------------------=== #
# _pow
# ===----------------------------------------------------------------------=== #


@always_inline
fn _pow[
    BaseTy: DType, simd_width: Int, ExpTy: DType
](base: SIMD[BaseTy, simd_width], exp: SIMD[ExpTy, simd_width]) -> __type_of(
    base
):
    """Computes the power of the elements of a SIMD vector raised to the
    corresponding elements of another SIMD vector.

    Parameters:
        BaseTy: The `dtype` of the `base` SIMD vector.
        simd_width: The width of the input and output SIMD vectors.
        ExpTy: The `dtype` of the `exp` SIMD vector.

    Args:
        base: Base of the power operation.
        exp: Exponent of the power operation.

    Returns:
        A vector containing elementwise `base` raised to the power of `exp`.
    """

    @parameter
    if ExpTy.is_floating_point() and BaseTy == ExpTy:
        var rhs_quotient = exp.__floor__()
        if all((exp >= 0) & (rhs_quotient == exp)):
            return _pow(base, rhs_quotient.cast[_integral_type_of[ExpTy]()]())

        var result = __type_of(base)()

        @parameter
        if triple_is_nvidia_cuda():
            _print_fmt(
                "ABORT: pow with two floating point operands is not supported"
                " on GPU"
            )
            abort()
        else:

            @parameter
            for i in range(simd_width):
                result[i] = llvm_intrinsic[
                    "llvm.pow", Scalar[BaseTy], has_side_effect=False
                ](base[i], exp[i])

        return result
    elif ExpTy.is_integral():
        # Common cases
        if all(exp == 2):
            return base * base
        if all(exp == 3):
            return base * base * base

        var result = __type_of(base)()

        @parameter
        for i in range(simd_width):
            result[i] = _powi(base[i], exp[i].cast[DType.int32]())
        return result
    else:
        constrained[False, "unsupported type combination"]()
        return __type_of(base)()


@always_inline
fn _powi[type: DType](base: Scalar[type], exp: Int32) -> __type_of(base):
    if type.is_integral() and exp < 0:
        # Not defined for Integers, this should raise an
        # exception.
        debug_assert(False, "exponent < 0 is undefined for integers")
        return 0
    var a = base
    var b = abs(exp) if type.is_floating_point() else exp
    var res: Scalar[type] = 1
    while b > 0:
        if b & 1:
            res *= a
        a *= a
        b >>= 1

    @parameter
    if type.is_floating_point():
        if exp < 0:
            return 1 / res
    return res


# ===----------------------------------------------------------------------=== #
# bfloat16
# ===----------------------------------------------------------------------=== #

alias _fp32_bf16_mantissa_diff = FPUtils[
    DType.float32
].mantissa_width() - FPUtils[DType.bfloat16].mantissa_width()


@always_inline
fn _bfloat16_to_f32_scalar(
    val: Scalar[DType.bfloat16],
) -> Scalar[DType.float32]:
    @parameter
    if has_neon():
        # TODO(KERN-228): support BF16 on neon systems.
        return _unchecked_zero[DType.float32, 1]()

    var bfloat_bits = FPUtils[DType.bfloat16].bitcast_to_integer(val)
    return FPUtils[DType.float32].bitcast_from_integer(
        bfloat_bits << _fp32_bf16_mantissa_diff
    )


@always_inline
fn _bfloat16_to_f32[
    size: Int
](val: SIMD[DType.bfloat16, size]) -> SIMD[DType.float32, size]:
    @parameter
    if has_neon():
        # TODO(KERN-228): support BF16 on neon systems.
        return _unchecked_zero[DType.float32, size]()

    @always_inline
    @parameter
    fn wrapper_fn[
        input_type: DType, result_type: DType
    ](val: Scalar[input_type]) capturing -> Scalar[result_type]:
        return rebind[Scalar[result_type]](
            _bfloat16_to_f32_scalar(rebind[Scalar[DType.bfloat16]](val))
        )

    return _simd_apply[wrapper_fn, DType.float32, size](val)


@always_inline
fn _f32_to_bfloat16_scalar(
    val: Scalar[DType.float32],
) -> Scalar[DType.bfloat16]:
    @parameter
    if has_neon():
        # TODO(KERN-228): support BF16 on neon systems.
        return _unchecked_zero[DType.bfloat16, 1]()

    if _isnan(val):
        return -_nan[DType.bfloat16]() if FPUtils[DType.float32].get_sign(
            val
        ) else _nan[DType.bfloat16]()

    var float_bits = FPUtils[DType.float32].bitcast_to_integer(val)

    var lsb = (float_bits >> _fp32_bf16_mantissa_diff) & 1
    var rounding_bias = 0x7FFF + lsb
    float_bits += rounding_bias

    var bfloat_bits = float_bits >> _fp32_bf16_mantissa_diff

    return FPUtils[DType.bfloat16].bitcast_from_integer(bfloat_bits)


@always_inline
fn _f32_to_bfloat16[
    size: Int
](val: SIMD[DType.float32, size]) -> SIMD[DType.bfloat16, size]:
    @parameter
    if has_neon():
        # TODO(KERN-228): support BF16 on neon systems.
        return _unchecked_zero[DType.bfloat16, size]()

    @always_inline
    @parameter
    fn wrapper_fn[
        input_type: DType, result_type: DType
    ](val: Scalar[input_type]) capturing -> Scalar[result_type]:
        return rebind[Scalar[result_type]](
            _f32_to_bfloat16_scalar(rebind[Scalar[DType.float32]](val))
        )

    return _simd_apply[wrapper_fn, DType.bfloat16, size](val)


# ===----------------------------------------------------------------------=== #
# _simd_apply
# ===----------------------------------------------------------------------=== #


@always_inline
fn _simd_apply[
    func: fn[input_type: DType, result_type: DType] (
        Scalar[input_type]
    ) capturing -> Scalar[result_type],
    result_type: DType,
    simd_width: Int,
](x: SIMD[_, simd_width]) -> SIMD[result_type, simd_width]:
    """Returns a value whose elements corresponds to applying `func` to each
    element in the vector.

    Parameter:
      simd_width: Width of the input and output SIMD vectors.
      input_type: Type of the input to func.
      result_type: Result type of func.
      func: Function to apply to the SIMD vector.

    Args:
      x: the input value.

    Returns:
      A SIMD vector whose element at index `i` is `func(x[i])`.
    """
    var result = SIMD[result_type, simd_width]()

    @parameter
    for i in range(simd_width):
        result[i] = func[x.type, result_type](x[i])

    return result


@always_inline
fn _simd_apply[
    func: fn[lhs_type: DType, rhs_type: DType, result_type: DType] (
        Scalar[lhs_type], Scalar[rhs_type]
    ) capturing -> Scalar[result_type],
    result_type: DType,
    simd_width: Int,
](x: SIMD[_, simd_width], y: SIMD[_, simd_width]) -> SIMD[
    result_type, simd_width
]:
    """Returns a value whose elements corresponds to applying `func` to each
    element in the vector.

    Parameter:
      simd_width: Width of the input and output SIMD vectors.
      input_type: Type of the input to func.
      result_type: Result type of func.
      func: Function to apply to the SIMD vector.

    Args:
      x: the lhs input value.
      y: the rhs input value.

    Returns:
      A SIMD vector whose element at index `i` is `func(x[i], y[i])`.
    """
    var result = SIMD[result_type, simd_width]()

    @parameter
    for i in range(simd_width):
        result[i] = func[x.type, y.type, result_type](x[i], y[i])

    return result


# ===----------------------------------------------------------------------=== #
# _format_scalar
# ===----------------------------------------------------------------------=== #


fn _format_scalar[
    dtype: DType,
    float_format: StringLiteral = "%.17g",
](inout writer: Formatter, value: Scalar[dtype]):
    # Stack allocate enough bytes to store any formatted Scalar value of any
    # type.
    alias size: Int = _calc_format_buffer_size[dtype]()

    var buf = InlineArray[UInt8, size](fill=0)

    var wrote = _snprintf_scalar[dtype, float_format](
        buf.unsafe_ptr(),
        size,
        value,
    )

    # SAFETY:
    #   Create a slice to only those bytes in `buf` that have been initialized.
    var str_slice = StringSlice[__lifetime_of(buf)](
        unsafe_from_utf8_ptr=buf.unsafe_ptr(), len=wrote
    )

    writer.write_str(str_slice)
