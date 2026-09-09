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
"""Utilities for doing unsigned division and modulo operations on `Int`.

These helpers allow performing unsigned division and modulo operations on Int
arguments. This can provide better performance, as these operations can be
slower when performed using signed integers on some accelerators, as correctly
handling negative values requires additional instructions.
"""

from std.builtin.dtype import _unsigned_integral_type_of
from std.math import align_up, align_down


@always_inline
def ufloordiv(a: Int, b: Int) -> Int:
    """Perform unsigned floor division (`//`) on Int arguments.

    This function treats both arguments as unsigned values and performs
    unsigned division, which is faster than signed division on NVIDIA GPUs.

    For correctness, both arguments should be non-negative integers.

    Args:
        a: The dividend (treated as unsigned).
        b: The divisor (treated as unsigned).

    Returns:
        The quotient of unsigned division.
    """
    return Int(UInt(a) // UInt(b))


@always_inline("nodebug")
def udiv_unchecked(a: Int, b: Int) -> Int:
    """Unsigned division without zero-guard.

    Unlike `ufloordiv`, this uses `UInt.__truediv__` (`pop.div`) which emits
    a raw unsigned division without the zero-guard that `UInt.__floordiv__`
    inserts. This produces tighter codegen on GPUs where the guard is
    unnecessary overhead.

    The caller must guarantee that `b > 0`. Behavior is undefined otherwise.

    Args:
        a: The dividend (must be non-negative).
        b: The divisor (must be positive).

    Returns:
        The quotient of unsigned division.
    """
    return Int(UInt(a) / UInt(b))


@always_inline
def umod[
    dtype: DType, width: SIMDLength, //
](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Perform unsigned modulo (`%`) on `SIMD` arguments.

    This function reinterprets both arguments as unsigned values of the same
    bit width and performs unsigned modulo, which is faster than signed modulo
    on NVIDIA GPUs.

    For correctness, both arguments should be non-negative integers.

    Constraints:
        `dtype` must be an integral type.

    Parameters:
        dtype: The integral data type of the operands.
        width: The number of elements in each `SIMD` vector.

    Args:
        a: The dividend (treated as unsigned).
        b: The divisor (treated as unsigned).

    Returns:
        The elementwise remainder of unsigned division.
    """
    comptime assert dtype.is_integral(), "umod requires an integral dtype"
    comptime utype = _unsigned_integral_type_of[dtype]()
    return (a.cast[utype]() % b.cast[utype]()).cast[dtype]()


@always_inline
def udivmod(a: Int, b: Int) -> Tuple[Int, Int]:
    """Perform unsigned divmod on Int arguments.

    Computes the quotient and remainder in a single unsigned division,
    which is faster than signed divmod on NVIDIA GPUs.

    For correctness, both arguments should be non-negative integers.

    Args:
        a: The dividend (treated as unsigned).
        b: The divisor (treated as unsigned).

    Returns:
        A `Tuple` of `(quotient, remainder)` from unsigned division.
    """
    var ua = UInt(a)
    var ub = UInt(b)
    var q, r = divmod(ua, ub)
    return Int(q), Int(r)


@always_inline("nodebug")
def udivmod_unchecked(a: Int, b: Int) -> Tuple[Int, Int]:
    """Unsigned divmod without zero-guard.

    Unlike `udivmod`, this uses `UInt.__truediv__` (`pop.div`) which emits
    a raw unsigned division without the zero-guard (`icmp eq 0` + `umax` +
    `select`) that `UInt.__floordiv__` and `UInt.__mod__` insert. This
    produces tighter codegen on GPUs where the guard is unnecessary overhead.

    The caller must guarantee that `b > 0`. Behavior is undefined otherwise.

    Args:
        a: The dividend (must be non-negative).
        b: The divisor (must be positive).

    Returns:
        A `Tuple` of `(quotient, remainder)` from unsigned division.
    """
    debug_assert(b > 0, "divisor must be positive")
    var ua = UInt(a)
    var ub = UInt(b)
    var q = ua / ub
    return Int(q), Int(ua - q * ub)


@always_inline
def ualign_up(value: Int, alignment: Int) -> Int:
    """Returns the closest multiple of alignment that is greater than or equal
    to value.

    Args:
        value: The value to align.
        alignment: Value to align to.

    Returns:
        Closest multiple of the alignment that is greater than or equal to the
        input value. In other words, ceiling(value / alignment) * alignment.
    """
    return Int(align_up(UInt(value), UInt(alignment)))


@always_inline
def ualign_down(value: Int, alignment: Int) -> Int:
    """Returns the closest multiple of alignment that is less than or equal to
    value.

    Args:
        value: The value to align.
        alignment: Value to align to.

    Returns:
        Closest multiple of the alignment that is less than or equal to the
        input value. In other words, floor(value / alignment) * alignment.
    """
    return Int(align_down(UInt(value), UInt(alignment)))


@always_inline
def uceildiv(numerator: Int, denominator: Int) -> Int:
    """Return the rounded-up result of dividing numerator by denominator.

    Args:
        numerator: The numerator.
        denominator: The denominator.

    Returns:
        The ceiling of dividing numerator by denominator.
    """
    return Int(UInt(numerator).__ceildiv__(UInt(denominator)))
