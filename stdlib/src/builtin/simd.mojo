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

from sys import llvm_intrinsic
from sys.info import has_neon, is_x86, simdwidthof

from builtin.hash import _hash_simd
from memory.unsafe import bitcast

from utils._numerics import FPUtils
from utils._numerics import isnan as _isnan
from utils._numerics import nan as _nan
from utils._visualizers import lldb_formatter_wrapping_type
from utils.static_tuple import StaticTuple

from .dtype import _integral_type_of
from .io import _snprintf_scalar
from .string import _calc_initial_buffer_size, _vec_fmt

# ===------------------------------------------------------------------------===#
# Type Aliases
# ===------------------------------------------------------------------------===#

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

# ===------------------------------------------------------------------------===#
# Utilities
# ===------------------------------------------------------------------------===#


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


@lldb_formatter_wrapping_type
@register_passable("trivial")
struct SIMD[type: DType, size: Int = simdwidthof[type]()](
    Sized,
    Intable,
    CollectionElement,
    Stringable,
    Hashable,
    Boolable,
):
    """Represents a small vector that is backed by a hardware vector element.

    SIMD allows a single instruction to be executed across the multiple data elements of the vector.

    Constraints:
        The size of the SIMD vector to be positive and a power of 2.

    Parameters:
        type: The data type of SIMD vector elements.
        size: The size of the SIMD vector.
    """

    alias element_type = type
    var value: __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`]
    """The underlying storage for the vector."""

    alias MAX = Self(_inf[type]())
    """Gets a +inf value for the SIMD value."""

    alias MIN = Self(_neginf[type]())
    """Gets a -inf value for the SIMD value."""

    alias MAX_FINITE = Self(_max_finite[type]())
    """Returns the maximum finite value of SIMD value."""

    alias MIN_FINITE = Self(_min_finite[type]())
    """Returns the minimum (lowest) finite value of SIMD value."""

    @always_inline("nodebug")
    fn __init__() -> Self:
        """Default initializer of the SIMD vector.

        By default the SIMD vectors are initialized to all zeros.

        Returns:
            SIMD vector whose elements are 0.
        """
        _simd_construction_checks[type, size]()
        return _unchecked_zero[type, size]()

    @always_inline("nodebug")
    fn __init__(value: SIMD[DType.float64, 1]) -> Self:
        """Initializes the SIMD vector with a float.

        The value is splatted across all the elements of the SIMD
        vector.

        Args:
            value: The input value.

        Returns:
            SIMD vector whose elements have the specified value.
        """
        _simd_construction_checks[type, size]()

        var casted = __mlir_op.`pop.cast`[
            _type = __mlir_type[`!pop.simd<1,`, type.value, `>`]
        ](value.value)
        var vec = __mlir_op.`pop.simd.splat`[
            _type = __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`]
        ](casted)
        return Self {value: vec}

    @always_inline("nodebug")
    fn __init__(value: Int) -> Self:
        """Initializes the SIMD vector with an integer.

        The integer value is splatted across all the elements of the SIMD
        vector.

        Args:
            value: The input value.

        Returns:
            SIMD vector whose elements have the specified value.
        """
        _simd_construction_checks[type, size]()

        var t0 = __mlir_op.`pop.cast_from_builtin`[
            _type = __mlir_type.`!pop.scalar<index>`
        ](value.value)
        var casted = __mlir_op.`pop.cast`[
            _type = __mlir_type[`!pop.simd<1,`, type.value, `>`]
        ](t0)
        var vec = __mlir_op.`pop.simd.splat`[
            _type = __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`]
        ](casted)
        return Self {value: vec}

    @always_inline("nodebug")
    fn __init__(value: IntLiteral) -> Self:
        """Initializes the SIMD vector with an integer.

        The integer value is splatted across all the elements of the SIMD
        vector.

        Args:
            value: The input value.

        Returns:
            SIMD vector whose elements have the specified value.
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
        var vec = __mlir_op.`pop.simd.splat`[
            _type = __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`]
        ](casted)
        return Self {value: vec}

    @always_inline("nodebug")
    fn __init__(value: Bool) -> Self:
        """Initializes the SIMD vector with a bool value.

        The bool value is splatted across all elements of the SIMD vector.

        Args:
            value: The bool value.

        Returns:
            SIMD vector whose elements have the specified value.
        """
        _simd_construction_checks[type, size]()

        var casted = __mlir_op.`pop.cast`[
            _type = __mlir_type[`!pop.simd<1,`, type.value, `>`]
        ](value.value)
        var vec = __mlir_op.`pop.simd.splat`[
            _type = __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`]
        ](casted)
        return Self {value: vec}

    @always_inline("nodebug")
    fn __init__(
        value: __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`]
    ) -> Self:
        """Initializes the SIMD vector with the underlying mlir value.

        Args:
            value: The input value.

        Returns:
            SIMD vector using the specified value.
        """
        _simd_construction_checks[type, size]()
        return Self {value: value}

    # Construct via a variadic type which has the same number of elements as
    # the SIMD value.
    @always_inline("nodebug")
    fn __init__(*elems: Scalar[type]) -> Self:
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

        Returns:
            The constructed SIMD vector.
        """
        _simd_construction_checks[type, size]()
        var num_elements: Int = len(elems)
        if num_elements == 1:
            # Construct by broadcasting a scalar.
            return Self {
                value: __mlir_op.`pop.simd.splat`[
                    _type = __mlir_type[
                        `!pop.simd<`,
                        size.value,
                        `, `,
                        type.value,
                        `>`,
                    ]
                ](elems[0].value)
            }

        debug_assert(size == num_elements, "mismatch in the number of elements")
        var result = Self()

        @unroll
        for i in range(size):
            result[i] = elems[i]

        return result

    @always_inline("nodebug")
    fn __init__(value: FloatLiteral) -> Self:
        """Initializes the SIMD vector with a float.

        The value is splatted across all the elements of the SIMD
        vector.

        Args:
            value: The input value.

        Returns:
            SIMD vector whose elements have the specified value.
        """
        _simd_construction_checks[type, size]()

        var tn1 = __mlir_op.`kgen.float_literal.convert`[
            _type = __mlir_type.f64
        ](value.value)
        var t0 = __mlir_op.`pop.cast_from_builtin`[
            _type = __mlir_type.`!pop.scalar<f64>`
        ](tn1)
        var casted = __mlir_op.`pop.cast`[
            _type = __mlir_type[`!pop.simd<1,`, type.value, `>`]
        ](t0)
        var vec = __mlir_op.`pop.simd.splat`[
            _type = __mlir_type[`!pop.simd<`, size.value, `, `, type.value, `>`]
        ](casted)
        return Self {value: vec}

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Gets the length of the SIMD vector.

        Returns:
            The length of the SIMD vector.
        """

        return size

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        """Converts the SIMD vector into a boolean scalar value.

        Returns:
            True if all the elements in the SIMD vector are non-zero and False
            otherwise.
        """

        @parameter
        if Self.element_type == DType.bool:
            return self.reduce_and()
        return (self != 0).reduce_and()

    @staticmethod
    @always_inline("nodebug")
    fn splat(x: Bool) -> Self:
        """Splats (broadcasts) the element onto the vector.

        Args:
            x: The input value.

        Returns:
            A new SIMD vector whose elements are the same as the input value.
        """
        _simd_construction_checks[type, size]()
        constrained[type == DType.bool, "input type must be boolean"]()
        var val = SIMD[DType.bool, size] {
            value: __mlir_op.`pop.simd.splat`[
                _type = __mlir_type[
                    `!pop.simd<`,
                    size.value,
                    `, `,
                    DType.bool.value,
                    `>`,
                ]
            ](x.value)
        }
        return rebind[Self](val)

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
        if has_neon() and (type == DType.bfloat16 or target == DType.bfloat16):
            # BF16 support on neon systems is not supported.
            return _unchecked_zero[target, size]()

        @parameter
        if type == DType.bool:
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

        # Reserve space for opening and closing brackets, plus each element and
        # its trailing commas.
        var buf = String._buffer_type()
        var initial_buffer_size = 2
        for i in range(size):
            initial_buffer_size += _calc_initial_buffer_size(self[i]) + 2
        buf.reserve(initial_buffer_size)

        # Print an opening `[`.
        @parameter
        if size > 1:
            buf.size += _vec_fmt(buf.data, 2, "[")
        # Print each element.
        for i in range(size):
            var element = self[i]
            # Print separators between each element.
            if i != 0:
                buf.size += _vec_fmt(buf.data + buf.size, 3, ", ")

            buf.size += _snprintf_scalar[type](
                rebind[Pointer[Int8]](buf.data + buf.size),
                _calc_initial_buffer_size(element),
                element,
            )

        # Print a closing `]`.
        @parameter
        if size > 1:
            buf.size += _vec_fmt(buf.data + buf.size, 2, "]")

        buf.size += 1  # for the null terminator.
        return String(buf^)

    @always_inline("nodebug")
    fn to_int(self) -> Int:
        """Casts to the value to an Int. If there is a fractional component,
        then the value is truncated towards zero.

        Constraints:
            The size of the SIMD vector must be 1.

        Returns:
            The value of the single integer element in the SIMD vector.
        """
        return self.__int__()

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
            return (
                rebind[SIMD[DType.bool, size]](self)
                & rebind[SIMD[DType.bool, size]](rhs)
            ).cast[type]()

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
            rhs: The value to divide on.

        Returns:
            `floor(self / rhs)` value.
        """
        constrained[type.is_numeric(), "the type must be numeric"]()

        if rhs == 0:
            # this should raise an exception.
            return 0

        var div = self / rhs

        @parameter
        if type.is_floating_point():
            return _floor(div)
        elif type.is_unsigned():
            return div
        else:
            if self > 0 and rhs > 0:
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

        if rhs == 0:
            # this should raise an exception.
            return 0

        @parameter
        if type.is_unsigned():
            return __mlir_op.`pop.rem`(self.value, rhs.value)
        else:
            var div = self / rhs

            @parameter
            if type.is_floating_point():
                div = llvm_intrinsic["llvm.trunc", Self](div)

            var mod = self - div * rhs
            var mask = ((rhs < 0) ^ (self < 0)) & (mod != 0)
            return mod + mask.select(rhs, Self(0))

    @always_inline("nodebug")
    fn __pow__(self, rhs: Int) -> Self:
        """Computes the vector raised to the power of the input integer value.

        Args:
            rhs: The exponential value.

        Returns:
            A SIMD vector where each element is raised to the power of the
            specified exponential value.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return _pow(self, rhs)

    # TODO(#22771): remove this overload.
    @always_inline("nodebug")
    fn __pow__(self, rhs: Self) -> Self:
        """Computes the vector raised elementwise to the right hand side power.

        Args:
            rhs: The exponential value.

        Returns:
            A SIMD vector where each element is raised to the power of the
            specified exponential value.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return _pow(self, rhs)

    @always_inline("nodebug")
    fn __pow__[rhs_type: DType](self, rhs: SIMD[rhs_type, size]) -> Self:
        """Computes the vector raised elementwise to the right hand side power.

        Parameters:
          rhs_type: The `dtype` of the rhs SIMD vector.

        Args:
            rhs: The exponential value.

        Returns:
            A SIMD vector where each element is raised to the power of the
            specified exponential value.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return _pow(self, rhs)

    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> SIMD[DType.bool, size]:
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
    fn __le__(self, rhs: Self) -> SIMD[DType.bool, size]:
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
    fn __eq__(self, rhs: Self) -> SIMD[DType.bool, size]:
        """Compares two SIMD vectors using equal-to comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is True or False depending on the expression
            `self[i] == rhs[i]`.
        """

        @parameter  # Because of #30525, we roll our own implementation for eq.
        if has_neon() and type == DType.bfloat16:
            var int_self = bitcast[_integral_type_of[type](), size](self)
            var int_rhs = bitcast[_integral_type_of[type](), size](rhs)
            return int_self == int_rhs

        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred eq>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> SIMD[DType.bool, size]:
        """Compares two SIMD vectors using not-equal comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            A new bool SIMD vector of the same size whose element at position
            `i` is True or False depending on the expression
            `self[i] != rhs[i]`.
        """

        @parameter  # Because of #30525, we roll our own implementation for ne.
        if has_neon() and type == DType.bfloat16:
            var int_self = bitcast[_integral_type_of[type](), size](self)
            var int_rhs = bitcast[_integral_type_of[type](), size](rhs)
            return int_self != int_rhs

        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __gt__(self, rhs: Self) -> SIMD[DType.bool, size]:
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
    fn __ge__(self, rhs: Self) -> SIMD[DType.bool, size]:
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

    # ===-------------------------------------------------------------------===#
    # In place operations.
    # ===-------------------------------------------------------------------===#

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

    # ===-------------------------------------------------------------------===#
    # Reversed operations
    # ===-------------------------------------------------------------------===#

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
    fn __rtruediv__(self, value: Self) -> Self:
        """Returns `value / self`.

        Args:
            value: The other value.

        Returns:
            `value / self`.
        """
        constrained[type.is_numeric(), "the SIMD type must be numeric"]()
        return value / self

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

    # ===-------------------------------------------------------------------===#
    # Bitwise operations
    # ===-------------------------------------------------------------------===#

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

    # ===-------------------------------------------------------------------===#
    # Shift operations
    # ===-------------------------------------------------------------------===#

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
        debug_assert(rhs >= 0, "unhandled negative value")
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
        debug_assert(rhs >= 0, "unhandled negative value")
        return __mlir_op.`pop.shr`(self.value, rhs.value)

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

    # ===-------------------------------------------------------------------===#
    # Shuffle operations
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn _shuffle_list[
        *mask: Int, output_size: Int = size
    ](self, other: Self) -> SIMD[type, output_size]:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask values
        must be within `2*len(self)`.

        Parameters:
            mask: The permutation to use in the shuffle.
            output_size: The size of the output vector.

        Args:
            other: The other vector to shuffle with.

        Returns:
            A new vector of length `len` where the value at position `i` is
            `(self+other)[permutation[i]]`.
        """

        @parameter
        fn variadic_len[*mask: Int]() -> Int:
            return __mlir_op.`pop.variadic.size`(mask)

        @parameter
        fn _convert_variadic_to_pop_array[
            *mask: Int
        ]() -> __mlir_type[
            `!pop.array<`, variadic_len[mask]().value, `, `, Int, `>`
        ]:
            alias size = variadic_len[mask]()
            var array = __mlir_op.`kgen.undef`[
                _type = __mlir_type[
                    `!pop.array<`, variadic_len[mask]().value, `, `, Int, `>`
                ]
            ]()

            @always_inline
            @parameter
            fn fill[idx: Int]():
                alias val = mask[idx]
                constrained[
                    0 <= val < 2 * size,
                    "invalid index in the shuffle operation",
                ]()
                var ptr = __mlir_op.`pop.array.gep`(
                    Pointer.address_of(array).address, idx.value
                )
                __mlir_op.`pop.store`(val, ptr)

            unroll[fill, size]()
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
    fn shuffle[*mask: Int](self) -> Self:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask values
        must be within `2*len(self)`.

        Parameters:
            mask: The permutation to use in the shuffle.

        Returns:
            A new vector of length `len` where the value at position `i` is
            `(self)[permutation[i]]`.
        """
        return self._shuffle_list[mask](self)

    @always_inline("nodebug")
    fn shuffle[*mask: Int](self, other: Self) -> Self:
        """Shuffles (also called blend) the values of the current vector with
        the `other` value using the specified mask (permutation). The mask values
        must be within `2*len(self)`.

        Parameters:
            mask: The permutation to use in the shuffle.

        Args:
            other: The other vector to shuffle with.

        Returns:
            A new vector of length `len` where the value at position `i` is
            `(self+other)[permutation[i]]`.
        """
        return self._shuffle_list[mask](other)

    # ===-------------------------------------------------------------------===#
    # Indexing operations
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
        ](self.value, idx.value)

    @always_inline("nodebug")
    fn __setitem__(inout self, idx: Int, val: Scalar[type]):
        """Sets an element in the vector.

        Args:
            idx: The index to set.
            val: The value to set.
        """
        self.value = __mlir_op.`pop.simd.insertelement`(
            self.value, val.value, idx.value
        )

    @always_inline("nodebug")
    fn __setitem__(
        inout self, idx: Int, val: __mlir_type[`!pop.scalar<`, type.value, `>`]
    ):
        """Sets an element in the vector.

        Args:
            idx: The index to set.
            val: The value to set.
        """
        self.value = __mlir_op.`pop.simd.insertelement`(
            self.value, val, idx.value
        )

    fn __hash__(self) -> Int:
        """Hash the value using builtin hash.

        Returns:
            A 64-bit hash value. This value is _not_ suitable for cryptographic
            uses. Its intended usage is for data structures. See the `hash`
            builtin documentation for more details.
        """
        return _hash_simd(self)

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

            @unroll
            for i in range(output_width):
                tmp[i] = self[i + offset]
            return tmp

        return llvm_intrinsic["llvm.vector.extract", SIMD[type, output_width]](
            self, offset
        )

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

            @unroll
            for i in range(input_width):
                tmp[i + offset] = value[i]
            return tmp

        return llvm_intrinsic["llvm.vector.insert", Self](self, value, offset)

    @always_inline("nodebug")
    fn join(self, other: Self) -> SIMD[type, 2 * size]:
        """Concatenates the two vectors together.

        Args:
            other: The other SIMD vector.

        Returns:
            A new vector `self_0, self_1, ..., self_n, other_0, ..., other_n`.
        """

        # Common cases will use shuffle which the compiler understands well.
        @parameter
        if size == 1:
            return self._shuffle_list[
                0,
                1,
                output_size = 2 * size,
            ](other)
        elif size == 2:
            return self._shuffle_list[
                0,
                1,
                2,
                3,
                output_size = 2 * size,
            ](other)
        elif size == 4:
            return self._shuffle_list[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                output_size = 2 * size,
            ](other)
        elif size == 8:
            return self._shuffle_list[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                output_size = 2 * size,
            ](other)
        elif size == 16:
            return self._shuffle_list[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                output_size = 2 * size,
            ](other)

        var res = SIMD[type, 2 * size]()
        res = res.insert(self)
        return res.insert[offset=size](other)

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
            "llvm.experimental.vector.interleave2", SIMD[type, 2 * size]
        ](self, other)

    @always_inline("nodebug")
    fn deinterleave(self) -> StaticTuple[SIMD[type, size // 2], 2]:
        """Constructs two vectors by deinterleaving the even and odd lanes of
        the vector.

        Constraints:
            The vector size must be greater than 1.

        Returns:
            Two vectors the first of the form `self_0, self_2, ..., self_{n-2}`
            and the other being `self_1, self_3, ..., self_{n-1}`.
        """

        constrained[size > 1, "the vector size must be greater than 1."]()

        var res = llvm_intrinsic[
            "llvm.experimental.vector.deinterleave2",
            (SIMD[type, size // 2], SIMD[type, size // 2]),
        ](self)
        return StaticTuple[SIMD[type, size // 2], 2](
            res.get[0, SIMD[type, size // 2]](),
            res.get[1, SIMD[type, size // 2]](),
        )

    # ===-------------------------------------------------------------------===#
    # Binary operations
    # ===-------------------------------------------------------------------===#

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

    # ===-------------------------------------------------------------------===#
    # Reduce operations
    # ===-------------------------------------------------------------------===#

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

        Returns:
            A new scalar which is the reduction of all vector elements.
        """
        constrained[
            size_out <= Self.size, "simd reduction cannot increase simd width"
        ]()

        @parameter
        if size == 1:
            return self[0]
        elif size == 2:
            return func[type, 1](self[0], self[1])
        elif size == size_out:
            return rebind[SIMD[Self.type, size_out]](self)
        else:
            alias half_size: Int = size // 2
            var lhs = self.slice[half_size, offset=0]()
            var rhs = self.slice[half_size, offset=half_size]()

            @parameter
            if half_size != size_out:
                return func[type, half_size](lhs, rhs).reduce[func, size_out]()
            return rebind[SIMD[type, size_out]](func[type, half_size](lhs, rhs))

    @always_inline("nodebug")
    fn reduce_max[size_out: Int = 1](self) -> SIMD[type, size_out]:
        """Reduces the vector using the `max` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
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
                llvm_intrinsic["llvm.vector.reduce.fmax", Scalar[type]](self)
            )

        @parameter
        if type.is_unsigned():
            return rebind[SIMD[type, size_out]](
                llvm_intrinsic["llvm.vector.reduce.umax", Scalar[type]](self)
            )
        return rebind[SIMD[type, size_out]](
            llvm_intrinsic["llvm.vector.reduce.smax", Scalar[type]](self)
        )

    @always_inline("nodebug")
    fn reduce_min[size_out: Int = 1](self) -> SIMD[type, size_out]:
        """Reduces the vector using the `min` operator.

        Parameters:
            size_out: The width of the reduction.

        Constraints:
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
                llvm_intrinsic["llvm.vector.reduce.fmin", Scalar[type]](self)
            )

        @parameter
        if type.is_unsigned():
            return rebind[SIMD[type, size_out]](
                llvm_intrinsic["llvm.vector.reduce.umin", Scalar[type]](self)
            )
        return rebind[SIMD[type, size_out]](
            llvm_intrinsic["llvm.vector.reduce.smin", Scalar[type]](self)
        )

    @always_inline
    fn reduce_add[size_out: Int = 1](self) -> SIMD[type, size_out]:
        """Reduces the vector using the `add` operator.

        Parameters:
            size_out: The width of the reduction.

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
    fn reduce_and(self) -> Bool:
        """Reduces the boolean vector using the `and` operator.

        Constraints:
            The element type of the vector must be boolean.

        Returns:
            True if all element in the vector is True and False otherwise.
        """

        @parameter
        if size == 1:
            return self.cast[DType.bool]()[0].value
        return llvm_intrinsic["llvm.vector.reduce.and", Scalar[DType.bool]](
            self
        )

    @always_inline
    fn reduce_or(self) -> Bool:
        """Reduces the boolean vector using the `or` operator.

        Constraints:
            The element type of the vector must be boolean.

        Returns:
            True if any element in the vector is True and False otherwise.
        """

        @parameter
        if size == 1:
            return self.cast[DType.bool]()[0].value
        return llvm_intrinsic["llvm.vector.reduce.or", Scalar[DType.bool]](self)

    # ===-------------------------------------------------------------------===#
    # select
    # ===-------------------------------------------------------------------===#

    # TODO (7748): always_inline required to WAR LLVM codegen bug
    @always_inline("nodebug")
    fn select[
        result_type: DType
    ](
        self,
        true_case: SIMD[result_type, size],
        false_case: SIMD[result_type, size],
    ) -> SIMD[result_type, size]:
        """Selects the values of the `true_case` or the `false_case` based on the
        current boolean values of the SIMD vector.

        Parameters:
            result_type: The element type of the input and output SIMD vectors.

        Args:
            true_case: The values selected if the positional value is True.
            false_case: The values selected if the positional value is False.

        Returns:
            A new vector of the form
            `[true_case[i] if elem else false_case[i] for i, elem in enumerate(self)]`.
        """
        constrained[type.is_bool(), "the simd dtype must be bool"]()
        return __mlir_op.`pop.simd.select`(
            rebind[SIMD[DType.bool, size]](self).value,
            true_case.value,
            false_case.value,
        )

    # ===-------------------------------------------------------------------===#
    # Rotation operations
    # ===-------------------------------------------------------------------===#

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
        return llvm_intrinsic["llvm.experimental.vector.splice", Self](
            self, self, Int32(shift)
        )

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

    # ===-------------------------------------------------------------------===#
    # Shift operations
    # ===-------------------------------------------------------------------===#

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

        return llvm_intrinsic["llvm.experimental.vector.splice", Self](
            self, zero_simd, Int32(shift)
        )

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

        return llvm_intrinsic["llvm.experimental.vector.splice", Self](
            zero_simd, self, Int32(-shift)
        )


# ===-------------------------------------------------------------------===#
# _pow
# ===-------------------------------------------------------------------===#


fn _pow[
    type: DType, simd_width: Int
](arg0: SIMD[type, simd_width], arg1: Int) -> SIMD[type, simd_width]:
    """Computes the `pow` of the inputs.

    Parameters:
      type: The `dtype` of the input and output SIMD vector.
      simd_width: The width of the input and output SIMD vector.

    Args:
      arg0: The first input argument.
      arg1: The second input argument.

    Returns:
      The `pow` of the inputs.
    """
    return _pow[type, DType.index, simd_width](arg0, arg1)


@always_inline
fn _pow[
    lhs_type: DType, rhs_type: DType, simd_width: Int
](lhs: SIMD[lhs_type, simd_width], rhs: SIMD[rhs_type, simd_width]) -> SIMD[
    lhs_type, simd_width
]:
    """Computes elementwise power of a type raised to another type.

    An element of the result SIMD vector will be the result of raising the
    corresponding element of lhs to the corresponding element of rhs.

    Parameters:
      lhs_type: The `dtype` of the lhs SIMD vector.
      rhs_type: The `dtype` of the rhs SIMD vector.
      simd_width: The width of the input and output SIMD vectors.

    Args:
      lhs: Base of the power operation.
      rhs: Exponent of the power operation.

    Returns:
      A SIMD vector containing elementwise lhs raised to the power of rhs.
    """

    @parameter
    if rhs_type.is_floating_point() and lhs_type == rhs_type:
        var rhs_quotient = _floor(rhs)
        if rhs >= 0 and rhs_quotient == rhs:
            return _pow(lhs, rhs_quotient.cast[_integral_type_of[rhs_type]()]())

        var result = SIMD[lhs_type, simd_width]()

        @unroll
        for i in range(simd_width):
            result[i] = llvm_intrinsic["llvm.pow", Scalar[lhs_type]](
                lhs[i], rhs[i]
            )

        return result
    elif rhs_type.is_integral():
        # Common cases
        if rhs == 2:
            return lhs * lhs
        if rhs == 3:
            return lhs * lhs * lhs

        var result = SIMD[lhs_type, simd_width]()

        @parameter
        if lhs_type.is_floating_point():

            @unroll
            for i in range(simd_width):
                result[i] = llvm_intrinsic["llvm.powi", Scalar[lhs_type]](
                    lhs[i], rhs[i].cast[DType.int32]()
                )
        else:
            for i in range(simd_width):
                if rhs[i] < 0:
                    # Not defined for Integers, this should raise an
                    # exception.
                    debug_assert(
                        False, "exponent < 0 is undefined for integers"
                    )
                    result[i] = 0
                    break
                var res: Scalar[lhs_type] = 1
                var x = lhs[i]
                var n = rhs[i]
                while n > 0:
                    if n & 1 != 0:
                        res *= x
                    x *= x
                    n >>= 1
                result[i] = res
        return result
    else:
        # Unsupported.
        return SIMD[lhs_type, simd_width]()


# ===----------------------------------------------------------------------===#
# floor
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _floor[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Performs elementwise floor on the elements of a SIMD vector.

    Parameters:
      type: The `dtype` of the input and output SIMD vector.
      simd_width: The width of the input and output SIMD vector.

    Args:
      x: SIMD vector to perform floor on.

    Returns:
      The elementwise floor of x.
    """

    @parameter
    if type.is_bool() or type.is_integral():
        return x

    @parameter
    if has_neon() and type == DType.bfloat16:
        return _floor(x.cast[DType.float32]()).cast[type]()

    return llvm_intrinsic["llvm.floor", SIMD[type, simd_width]](x)


# ===----------------------------------------------------------------------===#
# bfloat16
# ===----------------------------------------------------------------------===#

alias _fp32_bf16_mantissa_diff = FPUtils[
    DType.float32
].mantissa_width() - FPUtils[DType.bfloat16].mantissa_width()


@always_inline
fn _bfloat16_to_f32_scalar(
    val: Scalar[DType.bfloat16],
) -> Scalar[DType.float32]:
    @parameter
    if has_neon():
        # BF16 support on neon systems is not supported.
        return _unchecked_zero[DType.float32, 1]()

    var bfloat_bits = FPUtils.bitcast_to_integer(val)
    return FPUtils[DType.float32].bitcast_from_integer(
        bfloat_bits << _fp32_bf16_mantissa_diff
    )


@always_inline
fn _bfloat16_to_f32[
    size: Int
](val: SIMD[DType.bfloat16, size]) -> SIMD[DType.float32, size]:
    @parameter
    if has_neon():
        # BF16 support on neon systems is not supported.
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
        # BF16 support on neon systems is not supported.
        return _unchecked_zero[DType.bfloat16, 1]()

    if _isnan(val):
        return -_nan[DType.bfloat16]() if FPUtils.get_sign(val) else _nan[
            DType.bfloat16
        ]()

    var float_bits = FPUtils.bitcast_to_integer(val)

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
        # BF16 support on neon systems is not supported.
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


# ===----------------------------------------------------------------------===#
# Limits
# ===----------------------------------------------------------------------===#


# ===----------------------------------------------------------------------===#
# inf
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _inf[type: DType]() -> Scalar[type]:
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
    return _max_finite[type]()


# ===----------------------------------------------------------------------===#
# neginf
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _neginf[type: DType]() -> Scalar[type]:
    """Gets a -inf value for the given dtype.

    Constraints:
        Can only be used for FP dtypes.

    Parameters:
        type: The value dtype.

    Returns:
        The -inf value of the given dtype.
    """

    @parameter
    if type == DType.float16:
        return rebind[__mlir_type[`!pop.scalar<`, type.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f16>`],
                value = __mlir_attr[`#pop.simd<"-inf"> : !pop.scalar<f16>`],
            ]()
        )
    elif type == DType.bfloat16:
        return rebind[__mlir_type[`!pop.scalar<`, type.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<bf16>`],
                value = __mlir_attr[`#pop.simd<"-inf"> : !pop.scalar<bf16>`],
            ]()
        )
    elif type == DType.float32:
        return rebind[__mlir_type[`!pop.scalar<`, type.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f32>`],
                value = __mlir_attr[`#pop.simd<"-inf"> : !pop.scalar<f32>`],
            ]()
        )
    elif type == DType.float64:
        return rebind[__mlir_type[`!pop.scalar<`, type.value, `>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<f64>`],
                value = __mlir_attr[`#pop.simd<"-inf"> : !pop.scalar<f64>`],
            ]()
        )
    return _min_finite[type]()


# ===----------------------------------------------------------------------===#
# max_finite
# ===----------------------------------------------------------------------===#


@always_inline
fn _max_finite[type: DType]() -> Scalar[type]:
    """Returns the maximum finite value of type.

    Parameters:
        type: The value dtype.

    Returns:
        The maximum representable value of the type. Does not include infinity for
        floating-point types.
    """

    @parameter
    if type == DType.int8:
        return 127
    elif type == DType.uint8:
        return 255
    elif type == DType.int16:
        return 32767
    elif type == DType.uint16:
        return 65535
    elif type == DType.int32 or (
        type == DType.index and sizeof[DType.index]() == sizeof[DType.int32]()
    ):
        return 2147483647
    elif type == DType.uint32:
        return 4294967295
    elif type == DType.float32:
        return 3.40282346638528859812e38
    elif type == DType.int64 or (
        type == DType.index and sizeof[DType.index]() == sizeof[DType.int64]()
    ):
        return 9223372036854775807
    elif type == DType.uint64:
        return 18446744073709551615
    elif type == DType.float64:
        return 1.79769313486231570815e308
    elif type == DType.bfloat16:
        return 3.38953139e38
    else:
        constrained[False, "max_finite() called on unsupported type"]()
        return 0


# ===----------------------------------------------------------------------===#
# min_finite
# ===----------------------------------------------------------------------===#


@always_inline
fn _min_finite[type: DType]() -> Scalar[type]:
    """Returns the minimum (lowest) finite value of type.

    Parameters:
        type: The value dtype.

    Returns:
        The minimum representable value of the type. Does not include negative
        infinity for floating-point types.
    """

    @parameter
    if type.is_unsigned():
        return 0
    elif type == DType.int8:
        return -128
    elif type == DType.int16:
        return -32768
    elif type == DType.int32:
        return -2147483648
    elif type == DType.float32:
        return -_max_finite[type]()
    elif type == DType.int64:
        return -9223372036854775808
    elif type == DType.float64:
        return -_max_finite[type]()
    elif type == DType.bfloat16:
        return -_max_finite[type]()
    else:
        constrained[False, "min_finite() called on unsupported type"]()
        return 0


# ===----------------------------------------------------------------------===#
# _simd_apply
# ===----------------------------------------------------------------------===#


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

    @unroll
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

    @unroll
    for i in range(simd_width):
        result[i] = func[x.type, y.type, result_type](x[i], y[i])

    return result
