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
#
# This file contains wrappers around Intel VNNI intrinsics. See
# https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&avx512techs=AVX512_VNNI
#
# ===-----------------------------------------------------------------------===#

"""Provides wrappers around Intel VNNI (Vector Neural Network Instructions) for dot-product and multiply-accumulate operations on integer data."""

from std.sys import CompilationTarget, llvm_intrinsic

from std.memory.unsafe import bitcast

# ===-----------------------------------------------------------------------===#
# vpdpwssd
# ===-----------------------------------------------------------------------===#


def vpdpwssd[
    width: SIMDLength, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width],
    a: SIMD[a_type, width * 2],
    b: SIMD[b_type, width * 2],
) -> SIMD[c_type, width]:
    """Computes a multiply-accumulate of signed 16-bit integers using the VPDPWSSD Intel AVX-512 VNNI instruction.

    Multiplies pairs of adjacent signed 16-bit integers from a and b, accumulates
    the 32-bit products into src, and returns the result.

    Parameters:
        width: Number of int32 output elements (4, 8, or 16).
        a_type: DType of the a operand (int16).
        b_type: DType of the b operand (int16).
        c_type: DType of the accumulator; must be int32.

    Args:
        src: Int32 accumulator vector.
        a: Int16 input vector, twice the output width.
        b: Int16 input vector, twice the output width.

    Returns:
        Updated int32 accumulator after multiply-add.
    """
    comptime assert c_type == .int32, "the type of C must be int32"

    comptime if width == 16:
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpwssd.512", SIMD[c_type, width]
        ](src, a, b)
    elif width == 8:
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpwssd.256", SIMD[c_type, width]
        ](src, a, b)
    else:
        comptime assert width == 4
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpwssd.128", SIMD[c_type, width]
        ](src, a, b)


# ===-----------------------------------------------------------------------===#
# vpdpwssds
# ===-----------------------------------------------------------------------===#


def vpdpwssds[
    width: SIMDLength, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width],
    a: SIMD[a_type, width * 2],
    b: SIMD[b_type, width * 2],
) -> SIMD[c_type, width]:
    """Computes a saturating multiply-accumulate of signed 16-bit integers using the VPDPWSSDS Intel AVX-512 VNNI instruction.

    Like `vpdpwssd` but saturates the 32-bit accumulator on overflow instead of wrapping.

    Parameters:
        width: Number of int32 output elements (4, 8, or 16).
        a_type: DType of the a operand (int16).
        b_type: DType of the b operand (int16).
        c_type: DType of the accumulator; must be int32.

    Args:
        src: Int32 accumulator vector.
        a: Int16 input vector, twice the output width.
        b: Int16 input vector, twice the output width.

    Returns:
        Updated int32 accumulator after saturating multiply-add.
    """
    comptime assert c_type == .int32, "the type of C must be int32"

    comptime if width == 16:
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpwssds.512", SIMD[c_type, width]
        ](src, a, b)
    elif width == 8:
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpwssds.256", SIMD[c_type, width]
        ](src, a, b)
    else:
        comptime assert width == 4
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpwssds.128", SIMD[c_type, width]
        ](src, a, b)


# ===-----------------------------------------------------------------------===#
# vpdpbusd
# ===-----------------------------------------------------------------------===#


def vpdpbusd[
    width: SIMDLength, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width], a: SIMD[a_type, width], b: SIMD[b_type, width]
) -> SIMD[c_type, width]:
    """Computes a dot product of four unsigned-signed byte pairs per int32 element using the VPDPBUSD Intel AVX-512 VNNI instruction.

    For each int32 lane, treats the corresponding four bytes of a as uint8 and
    four bytes of b as int8, multiplies them element-wise, and adds the four
    products to the accumulator src.

    Parameters:
        width: Number of int32 output elements (4, 8, or 16).
        a_type: DType of the a operand.
        b_type: DType of the b operand.
        c_type: DType of the accumulator; must be int32.

    Args:
        src: Int32 accumulator vector.
        a: Uint8 input packed as int32-wide vector.
        b: Int8 input packed as int32-wide vector.

    Returns:
        Updated int32 accumulator after byte dot-product accumulation.
    """
    comptime assert c_type == .int32, "the type of C must be int32"

    comptime if width == 16:
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpbusd.512", SIMD[c_type, width]
        ](
            src,
            bitcast[.uint8, width * 4](a),
            bitcast[.uint8, width * 4](b),
        )
    elif width == 8:
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpbusd.256", SIMD[c_type, width]
        ](
            src,
            bitcast[.uint8, width * 4](a),
            bitcast[.uint8, width * 4](b),
        )
    else:
        comptime assert width == 4
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpbusd.128", SIMD[c_type, width]
        ](
            src,
            bitcast[.uint8, width * 4](a),
            bitcast[.uint8, width * 4](b),
        )


# ===-----------------------------------------------------------------------===#
# vpdpbusds
# ===-----------------------------------------------------------------------===#


def vpdpbusds[
    width: SIMDLength, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width], a: SIMD[a_type, width], b: SIMD[b_type, width]
) -> SIMD[c_type, width]:
    """Computes a saturating dot product of four unsigned-signed byte pairs per int32 element using the VPDPBUSDS Intel AVX-512 VNNI instruction.

    Like `vpdpbusd` but saturates the 32-bit accumulator on overflow.

    Parameters:
        width: Number of int32 output elements (4, 8, or 16).
        a_type: DType of the a operand.
        b_type: DType of the b operand.
        c_type: DType of the accumulator; must be int32.

    Args:
        src: Int32 accumulator vector.
        a: Uint8 input packed as int32-wide vector.
        b: Int8 input packed as int32-wide vector.

    Returns:
        Updated int32 accumulator after saturating byte dot-product accumulation.
    """
    comptime assert c_type == .int32, "the type of C must be int32"

    comptime if width == 16:
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpbusds.512", SIMD[c_type, width]
        ](
            src,
            bitcast[.uint8, width * 4](a),
            bitcast[.uint8, width * 4](b),
        )
    elif width == 8:
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpbusds.256", SIMD[c_type, width]
        ](
            src,
            bitcast[.uint8, width * 4](a),
            bitcast[.uint8, width * 4](b),
        )
    else:
        comptime assert width == 4
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpbusds.128", SIMD[c_type, width]
        ](
            src,
            bitcast[.uint8, width * 4](a),
            bitcast[.uint8, width * 4](b),
        )


def _dot_i8_to_i32_16(
    src: SIMD[.int32, 16], a: SIMD[.int8, 64], b: SIMD[.int8, 64]
) -> SIMD[.int32, 16]:
    var mask_hi = bitcast[.int8, 64](SIMD[.int16, 32](0x0100))
    var mask_lo = bitcast[.int8, 64](SIMD[.int16, 32](0x0001))
    var ah = llvm_intrinsic["llvm.x86.avx512.pmaddubs.w.512", SIMD[.int16, 32]](
        a, mask_hi
    )
    var bh = llvm_intrinsic["llvm.x86.avx512.pmaddubs.w.512", SIMD[.int16, 32]](
        mask_hi, b
    )
    var al = llvm_intrinsic["llvm.x86.avx512.pmaddubs.w.512", SIMD[.int16, 32]](
        a, mask_lo
    )
    var bl = llvm_intrinsic["llvm.x86.avx512.pmaddubs.w.512", SIMD[.int16, 32]](
        mask_lo, b
    )
    var t1 = llvm_intrinsic["llvm.x86.avx512.pmaddw.d.512", SIMD[.int32, 16]](
        al, bl
    )
    var t2 = llvm_intrinsic["llvm.x86.avx512.pmaddw.d.512", SIMD[.int32, 16]](
        ah, bh
    )
    return src + t1 + t2


def _dot_i8_to_i32_8(
    src: SIMD[.int32, 8], a: SIMD[.int8, 32], b: SIMD[.int8, 32]
) -> SIMD[.int32, 8]:
    var mask_hi = bitcast[.int8, 32](SIMD[.int16, 16](0x0100))
    var mask_lo = bitcast[.int8, 32](SIMD[.int16, 16](0x0001))

    var ah = llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[.int16, 16]](
        a, mask_hi
    )
    var bh = llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[.int16, 16]](
        mask_hi, b
    )
    var al = llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[.int16, 16]](
        a, mask_lo
    )
    var bl = llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[.int16, 16]](
        mask_lo, b
    )
    var t1 = llvm_intrinsic["llvm.x86.avx2.pmadd.wd", SIMD[.int32, 8]](al, bl)
    var t2 = llvm_intrinsic["llvm.x86.avx2.pmadd.wd", SIMD[.int32, 8]](ah, bh)
    return src + t1 + t2


def _dot_i8_to_i32_4(
    src: SIMD[.int32, 4], a: SIMD[.int8, 16], b: SIMD[.int8, 16]
) -> SIMD[.int32, 4]:
    var mask_hi = bitcast[.int8, 16](SIMD[.int16, 8](0x0100))
    var mask_lo = bitcast[.int8, 16](SIMD[.int16, 8](0x0001))

    var ah = llvm_intrinsic["llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[.int16, 8]](
        a, mask_hi
    )
    var bh = llvm_intrinsic["llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[.int16, 8]](
        mask_hi, b
    )
    var al = llvm_intrinsic["llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[.int16, 8]](
        a, mask_lo
    )
    var bl = llvm_intrinsic["llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[.int16, 8]](
        mask_lo, b
    )
    var t1 = llvm_intrinsic["llvm.x86.sse2.pmadd.wd", SIMD[.int32, 4]](al, bl)
    var t2 = llvm_intrinsic["llvm.x86.sse2.pmadd.wd", SIMD[.int32, 4]](ah, bh)
    return src + t1 + t2


def pmaddubs[
    width: SIMDLength
](a: SIMD[.int32, width], b: SIMD[.int32, width]) -> SIMD[.int32, width]:
    """Multiplies adjacent unsigned-signed byte pairs and returns the int16 results packed as int32.

    Parameters:
        width: Number of int32 output elements (4, 8, or 16).

    Args:
        a: Int32-typed SIMD vector reinterpreted as unsigned bytes.
        b: Int32-typed SIMD vector reinterpreted as signed bytes.
    """
    comptime if width == 16:
        return rebind[SIMD[.int32, width]](
            bitcast[.int32, 16](
                llvm_intrinsic[
                    "llvm.x86.avx512.pmaddubs.w.512", SIMD[.int16, 32]
                ](
                    bitcast[.int8, 64](a),
                    bitcast[.int8, 64](b),
                )
            )
        )
    elif width == 8:
        return rebind[SIMD[.int32, width]](
            bitcast[.int32, 8](
                llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[.int16, 16]](
                    bitcast[.int8, 32](a),
                    bitcast[.int8, 32](b),
                )
            )
        )
    else:
        comptime assert width == 4
        return rebind[SIMD[.int32, width]](
            bitcast[.int32, 4](
                llvm_intrinsic[
                    "llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[.int16, 8]
                ](
                    bitcast[.int8, 16](a),
                    bitcast[.int8, 16](b),
                )
            )
        )


def pmaddw[
    width: SIMDLength
](a: SIMD[.int32, width], b: SIMD[.int32, width]) -> SIMD[.int32, width]:
    """Multiplies adjacent signed 16-bit integer pairs and adds the products, returning int32 results.

    Parameters:
        width: Number of int32 output elements (4, 8, or 16).

    Args:
        a: Int32-typed SIMD vector reinterpreted as signed int16 pairs.
        b: Int32-typed SIMD vector reinterpreted as signed int16 pairs.
    """
    comptime if width == 16:
        return rebind[SIMD[.int32, width]](
            bitcast[.int32, 16](
                llvm_intrinsic[
                    "llvm.x86.avx512.pmaddw.d.512", SIMD[.int32, width]
                ](
                    bitcast[.int16, 32](a),
                    bitcast[.int16, 32](b),
                )
            )
        )
    elif width == 8:
        return rebind[SIMD[.int32, width]](
            bitcast[.int32, 8](
                llvm_intrinsic["llvm.x86.avx2.pmadd.wd", SIMD[.int32, width]](
                    bitcast[.int16, 16](a),
                    bitcast[.int16, 16](b),
                )
            )
        )
    else:
        comptime assert width == 4
        return rebind[SIMD[.int32, width]](
            bitcast[.int32, 16](
                llvm_intrinsic["llvm.x86.sse2.pmadd.wd", SIMD[.int32, width]](
                    bitcast[.int16, 8](a),
                    bitcast[.int16, 8](b),
                )
            )
        )


def _dot_i8_to_i32_saturated_16(
    src: SIMD[.int32, 16], a: SIMD[.int8, 64], b: SIMD[.int8, 64]
) -> SIMD[.int32, 16]:
    var t1 = llvm_intrinsic["llvm.x86.avx512.pmaddubs.w.512", SIMD[.int16, 32]](
        a, b
    )
    var t2 = llvm_intrinsic["llvm.x86.avx512.pmaddw.d.512", SIMD[.int32, 16]](
        t1, SIMD[.int16, 32](1)
    )
    return t2 + src


def _dot_i8_to_i32_saturated_8(
    src: SIMD[.int32, 8], a: SIMD[.int8, 32], b: SIMD[.int8, 32]
) -> SIMD[.int32, 8]:
    var t1 = llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[.int16, 16]](a, b)
    var t2 = llvm_intrinsic["llvm.x86.avx2.pmadd.wd", SIMD[.int32, 8]](
        t1, SIMD[.int16, 16](1)
    )
    return t2 + src


def _dot_i8_to_i32_saturated_4(
    src: SIMD[.int32, 4],
    a: SIMD[.int8, 16],
    b: SIMD[.int8, 16],
) -> SIMD[.int32, 4]:
    var t1 = llvm_intrinsic["llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[.int16, 8]](
        a, b
    )
    var t2 = llvm_intrinsic["llvm.x86.sse2.pmadd.wd", SIMD[.int32, 4]](
        t1, SIMD[.int16, 8](1)
    )
    return t2 + src


def dot_i8_to_i32_AVX2[
    width: SIMDLength, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width], a: SIMD[a_type, width], b: SIMD[b_type, width]
) -> SIMD[c_type, width]:
    """The dot product of the four bytes in each int32 element of a and b plus a int32 from src.

    Parameters:
        width: Size of the output SIMD vector.
        a_type: The DType for a.
        b_type: The DType for b.
        c_type: The DType for c.

    Args:
        src: A int32 SIMD vector.
        a: A uint8 SIMD vector.
        b: A int8 SIMD vector.

    Constraints:
        Requires AVX2.
        The size of the output vector must be 4, 8 or 16.
        The a argument has range [0,255].
        The b argument has range [-128,127].

    Returns:
        A SIMD vector of width elements.
    """

    comptime if width == 16:
        return rebind[SIMD[c_type, width]](
            _dot_i8_to_i32_16(
                rebind[SIMD[.int32, 16]](src),
                bitcast[.int8, 64](rebind[SIMD[.int32, 16]](a)),
                bitcast[.int8, 64](rebind[SIMD[.int32, 16]](b)),
            )
        )
    elif width == 8:
        return rebind[SIMD[c_type, width]](
            _dot_i8_to_i32_8(
                rebind[SIMD[.int32, 8]](src),
                bitcast[.int8, 32](rebind[SIMD[.int32, 8]](a)),
                bitcast[.int8, 32](rebind[SIMD[.int32, 8]](b)),
            )
        )
    else:
        comptime assert width == 4
        return rebind[SIMD[c_type, width]](
            _dot_i8_to_i32_4(
                rebind[SIMD[.int32, 4]](src),
                bitcast[.int8, 16](rebind[SIMD[.int32, 4]](a)),
                bitcast[.int8, 16](rebind[SIMD[.int32, 4]](b)),
            )
        )


def dot_i8_to_i32_saturated_AVX2[
    width: SIMDLength, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width], a: SIMD[a_type, width], b: SIMD[b_type, width]
) -> SIMD[c_type, width]:
    """The dot product of the four bytes in each int32 element of a and b plus a int32 from src.

    Parameters:
        width: Size of the output SIMD vector.
        a_type: The DType for a.
        b_type: The DType for b.
        c_type: The DType for c.

    Args:
        src: A int32 SIMD vector.
        a: A uint8 SIMD vector.
        b: A int8 SIMD vector.

    Constraints:
        Requires AVX2.
        The size of the output vector must be 4, 8 or 16.
        The a argument has range [0,127] not [0, 255].
        The b argument has range [-128,127].

    Returns:
        A SIMD vector of width elements.
    """

    comptime if width == 16:
        return rebind[SIMD[c_type, width]](
            _dot_i8_to_i32_saturated_16(
                rebind[SIMD[.int32, 16]](src),
                bitcast[.int8, 64](rebind[SIMD[.int32, 16]](a)),
                bitcast[.int8, 64](rebind[SIMD[.int32, 16]](b)),
            )
        )
    elif width == 8:
        return rebind[SIMD[c_type, width]](
            _dot_i8_to_i32_saturated_8(
                rebind[SIMD[.int32, 8]](src),
                bitcast[.int8, 32](rebind[SIMD[.int32, 8]](a)),
                bitcast[.int8, 32](rebind[SIMD[.int32, 8]](b)),
            )
        )
    else:
        comptime assert width == 4
        return rebind[SIMD[c_type, width]](
            _dot_i8_to_i32_saturated_4(
                rebind[SIMD[.int32, 4]](src),
                bitcast[.int8, 16](rebind[SIMD[.int32, 4]](a)),
                bitcast[.int8, 16](rebind[SIMD[.int32, 4]](b)),
            )
        )


def dot_i8_to_i32_x86[
    width: SIMDLength, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width], a: SIMD[a_type, width], b: SIMD[b_type, width]
) -> SIMD[c_type, width]:
    """The dot product of the four bytes in each int32 element of a and b plus a int32 from src using VNNI or AVX2.

    Parameters:
        width: Size of the output SIMD vector.
        a_type: The DType for a.
        b_type: The DType for b.
        c_type: The DType for c.

    Args:
        src: A int32 SIMD vector.
        a: A uint8 SIMD vector.
        b: A int8 SIMD vector.

    Constraints:
        Requires AVX512_VNNI or AVX2.
        The size of the output vector must be 4, 8 or 16.
        The a argument has range [0,255].
        The b argument has range [-128,127].

    Returns:
      A SIMD vector of width elements.
    """

    comptime if CompilationTarget.has_vnni():
        return vpdpbusd(src, a, b)
    else:
        return dot_i8_to_i32_AVX2(src, a, b)


# Saturation is much faster but limits input a to range [0, 127] instead of [0, 255]
def dot_i8_to_i32_saturated_x86[
    width: SIMDLength, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width], a: SIMD[a_type, width], b: SIMD[b_type, width]
) -> SIMD[c_type, width]:
    """The dot product of the four bytes in each int32 element of a and b plus a int32 from src using VNNI or AVX2.

    Parameters:
        width: Size of the output SIMD vector.
        a_type: The DType for a.
        b_type: The DType for b.
        c_type: The DType for c.

    Args:
        src: A int32 SIMD vector.
        a: A uint8 SIMD vector.
        b: A int8 SIMD vector.

    Constraints:
        Requires AVX512_VNNI or AVX2.
        The size of the output vector must be 4, 8 or 16.
        The a argument has range [0,127] not [0, 255].
        The b argument has range [-128,127].

    Returns:
      A SIMD vector of width elements.
    """

    comptime if CompilationTarget.has_vnni():
        return vpdpbusd(src, a, b)
    else:
        return dot_i8_to_i32_saturated_AVX2(src, a, b)


def dot_i16_to_i32_AVX2[
    width: SIMDLength, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width],
    a: SIMD[a_type, width * 2],
    b: SIMD[b_type, width * 2],
) -> SIMD[c_type, width]:
    """The dot product of the two words in each int32 element of a and b plus a int32 from src.

    Parameters:
        width: Size of the output SIMD vector.
        a_type: The DType for a.
        b_type: The DType for b.
        c_type: The DType for c.

    Args:
        src: A int32 SIMD vector.
        a: A int16 SIMD vector.
        b: A int16 SIMD vector.

    Constraints:
        Requires AVX2.
        The size of the output vector must be 4, 8 or 16.

    Returns:
        A SIMD vector of width elements.
    """

    var t: SIMD[c_type, width]

    comptime if width == 16:
        t = llvm_intrinsic["llvm.x86.avx512.pmaddw.d.512", SIMD[c_type, width]](
            a, b
        )
    elif width == 8:
        t = llvm_intrinsic["llvm.x86.avx2.pmadd.wd", SIMD[c_type, width]](a, b)
    else:
        comptime assert width == 4
        t = llvm_intrinsic["llvm.x86.sse2.pmadd.wd", SIMD[c_type, width]](a, b)

    return src + t


def dot_i16_to_i32_x86[
    width: SIMDLength, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width],
    a: SIMD[a_type, width * 2],
    b: SIMD[b_type, width * 2],
) -> SIMD[c_type, width]:
    """The dot product of the two words in each int32 element of a and b plus a int32 from src using VNNI or AVX2.

    Parameters:
        width: Size of the output SIMD vector.
        a_type: The DType for a.
        b_type: The DType for b.
        c_type: The DType for c.

    Args:
        src: A int32 SIMD vector.
        a: A int16 SIMD vector.
        b: A int16 SIMD vector.

    Constraints:
        Requires AVX512_VNNI or AVX2.
        The size of the output vector must be 4, 8 or 16.

    Returns:
      A SIMD vector of width elements.
    """

    comptime if CompilationTarget.has_vnni():
        return vpdpwssd(src, a, b)
    else:
        return dot_i16_to_i32_AVX2(src, a, b)
