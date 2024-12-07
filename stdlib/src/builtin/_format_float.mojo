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
# This file is derived from the dragonbox reference implementation
# (https://github.com/jk-jeon/dragonbox). Dragonbox contains the following
# attribution notice:
#
# Copyright 2020-2024 Junekey Jeon
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, this software
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.
# ===----------------------------------------------------------------------=== #
from collections import InlineArray
from math import log2
from sys.info import sizeof

from builtin.io import _printf
from memory import bitcast, Span

from utils import StaticTuple
from utils.numerics import FPUtils, isinf, isnan


@value
@register_passable("trivial")
struct _UInt128:
    var high: UInt64
    var low: UInt64

    fn __iadd__(mut self, n: UInt64):
        var sum = (self.low + n) & UInt64.MAX
        self.high += 1 if sum < self.low else 0
        self.low = sum


@value
@register_passable("trivial")
struct _MulParity:
    var parity: Bool
    var is_integer: Bool

    fn __init__(out self, parity: Bool, is_integer: Bool):
        self.parity = parity
        self.is_integer = is_integer


@value
@register_passable("trivial")
struct _MulResult[CarrierDType: DType]:
    var integer_part: Scalar[CarrierDType]
    var is_integer: Bool


@value
struct FP[type: DType, CarrierDType: DType = FPUtils[type].uint_type]:
    alias CarrierType = Scalar[Self.CarrierDType]
    alias total_bits = sizeof[type]() * 8
    alias carrier_bits = sizeof[Self.CarrierDType]() * 8
    alias sig_bits = FPUtils[type].mantissa_width()
    alias exp_bits = FPUtils[type].exponent_width()
    alias neg_exp_bias = -FPUtils[type].exponent_bias()
    alias min_normal_exp = Self.neg_exp_bias + 1
    alias cache_bits = 64 if Self.CarrierDType == DType.uint32 else 128
    alias min_k = -31 if Self.CarrierDType == DType.uint32 else -292
    alias max_k = 46 if Self.CarrierDType == DType.uint32 else 326
    alias divide_magic_number = StaticTuple[UInt32, 2](6554, 656)
    alias n_max = (
        (Scalar[Self.CarrierDType](2) << Self.sig_bits) + 1
    ) // 3 + 1 * 20
    alias n_max_larger = (
        Scalar[Self.CarrierDType](2) << Self.sig_bits
    ) * Self.big_divisor - 1
    alias kappa = _floor_log10_pow2(Self.carrier_bits - Self.sig_bits - 2) - 1
    alias big_divisor = pow(10, Self.kappa + 1)
    alias small_divisor = pow(10, Self.kappa)


fn _write_float[W: Writer, type: DType, //](mut writer: W, value: Scalar[type]):
    """Write a SIMD float type into a Writer, using the dragonbox algorithm for
    perfect roundtrip, shortest representable format, and high performance.
    Paper: https://github.com/jk-jeon/dragonbox/blob/master/other_files/Dragonbox.pdf
    Reference Implementation: https://github.com/jk-jeon/dragonbox.

    Parameters:
        W: The type of the Writer.
        type: The type of the float being passed in.

    Args:
        writer: The Writer to write the float to.
        value: The float to write into the Writer.
    """
    constrained[type.is_floating_point()]()

    @parameter
    if type is DType.float8e5m2:
        return writer.write(float8e5m2_to_str[int(bitcast[DType.uint8](value))])
    elif type is DType.float8e4m3:
        return writer.write(float8e4m3_to_str[int(bitcast[DType.uint8](value))])
    elif type is DType.float8e5m2fnuz:
        return writer.write(
            float8e5m2fnuz_to_str[int(bitcast[DType.uint8](value))]
        )
    elif type is DType.float8e4m3fnuz:
        return writer.write(
            float8e4m3fnuz_to_str[int(bitcast[DType.uint8](value))]
        )
    else:
        # Upcast the float16 types to float32
        casted = value.cast[
            DType.float64 if type == DType.float64 else DType.float32
        ]()

        # Bitcast the float and separate the sig and exp, to enable manipulating
        # bits as a UInt64 and Int:
        #  - The significand (sig) is the raw binary fraction
        #  - The exponent (exp) is still in biased form
        var sig = FPUtils.get_mantissa_uint(casted)
        var exp = FPUtils.get_exponent_biased(casted)
        var sign = FPUtils.get_sign(casted)

        if isinf(value):
            if sign:
                return writer.write("-inf")
            return writer.write("inf")

        if isnan(value):
            return writer.write("nan")

        if sign:
            writer.write("-")

        if not sig and not exp:
            return writer.write("0.0")

        # Convert the binary components to a decimal representation:
        #   - The raw binary sig into a decimal sig
        #   - The biased binary exp into a decimal power of 10 exp
        # This does all the heavy lifting for perfect roundtrip, shortest
        # representable format, bankers rounding etc.
        _to_decimal[casted.type](sig, exp)

        # This is a custom routine for writing the decimal following python
        # behavior.  it can be further optimized with a lookup table, there is
        # overhead here compared to snprintf.
        var orig_sig = sig
        var abs_exp = abs(exp)
        var digits = StaticTuple[Byte, 21]()
        var idx = 0
        while sig > 0:
            digits[idx] = (sig % 10).cast[DType.uint8]()
            sig //= 10
            idx += 1
            if sig > 0:
                exp += 1
        var leading_zeroes = abs_exp - idx

        # Write in scientific notation if < 0.0001 or exp > 15
        if (exp < 0 and leading_zeroes > 3) or exp > 15:
            # Handle single digit case
            if orig_sig < 10:
                writer.write(orig_sig)
            else:
                # Write digit before decimal point
                writer.write(digits[idx - 1])
                writer.write(".")
            # Write digits after decimal point
            for i in reversed(range(idx - 1)):
                writer.write(digits[i])
            # Write exponent
            if exp < 0:
                writer.write("e-")
                exp = -exp
            else:
                writer.write("e+")
            # Pad exponent with a 0 if less than two digits
            if exp < 10:
                writer.write("0")
            var exp_digits = StaticTuple[Byte, 10]()
            var exp_idx = 0
            while exp > 0:
                exp_digits[exp_idx] = exp % 10
                exp //= 10
                exp_idx += 1
            for i in reversed(range(exp_idx)):
                writer.write(exp_digits[i])
        # If between 0 and 0.0001
        elif exp < 0 and leading_zeroes > 0:
            writer.write("0.")
            for _ in range(leading_zeroes):
                writer.write("0")
            for i in reversed(range(idx)):
                writer.write(digits[i])
        # All other floats > 0.0001 with an exponent <= 15
        else:
            var point_written = False
            for i in reversed(range(idx)):
                if leading_zeroes < 1 and exp == idx - i - 2:
                    # No integer part so write leading 0
                    if i == idx - 1:
                        writer.write("0")
                    writer.write(".")
                    point_written = True
                writer.write(digits[i])

            # If exp - idx + 1 > 0 it's a positive number with more 0's than the
            # sig
            for _ in range(exp - idx + 1):
                writer.write("0")
            if not point_written:
                writer.write(".0")


fn _to_decimal[
    CarrierDType: DType, //, type: DType
](mut sig: Scalar[CarrierDType], mut exp: Int):
    """Transform the raw binary significand to decimal significand,
    and biased binary exponent into a decimal power of 10 exponent.
    """
    var two_fc = sig * 2
    var binary_exp = exp

    # For normal numbers
    if binary_exp != 0:
        binary_exp += FP[type].neg_exp_bias - FP[type].sig_bits
        if two_fc == 0:
            var minus_k = (binary_exp * 631305 - 261663) >> 21
            var beta = binary_exp + _floor_log2_pow10(-minus_k)
            var cache_index = -minus_k - FP[type].min_k

            var xi = _compute_endpoint[
                CarrierDType,
                FP[type].sig_bits,
                FP[type].total_bits,
                FP[type].cache_bits,
            ](cache_index, beta, left_endpoint=True)

            var zi = _compute_endpoint[
                CarrierDType,
                FP[type].sig_bits,
                FP[type].total_bits,
                FP[type].cache_bits,
            ](cache_index, beta, left_endpoint=False)

            # If we don't accept the left endpoint or if the left endpoint is
            # not an integer, increase it.
            if not _is_left_endpoint_integer_shorter_interval[
                CarrierDType, FP[type].sig_bits
            ](binary_exp):
                xi += 1

            # Try bigger divisor.
            # zi is at most floor((f_c + 1/2) * 2^e * 10^k0).
            # Substituting f_c = 2^p and k0 = -floor(log10(3 * 2^(e-2))), we get
            # zi <= floor((2^(p+1) + 1) * 20/3) <= ceil((2^(p+1) + 1)/3) * 20.
            sig = _divide_by_pow10[1, FP[type, CarrierDType].n_max](zi)

            # On success, remove trailing zeros and return.
            if sig * 10 >= xi:
                exp = minus_k + 1
                return _remove_trailing_zeros(sig, exp)

            # Otherwise, compute the round-up of y.
            sig = _compute_round_up_for_shorter_interval_case[
                CarrierDType,
                FP[type].total_bits,
                FP[type].sig_bits,
                FP[type].cache_bits,
            ](cache_index, beta)

            # When tie occurs
            if sig < xi:
                sig += 1

            # No trailing zeroes, so set exp and return
            exp = minus_k
            return

        # Normal interval case
        two_fc |= Scalar[CarrierDType](1) << (FP[type].sig_bits + 1)
    else:
        # For subnormal numbers
        binary_exp = FP[type].min_normal_exp - FP[type].sig_bits

    ##########################################
    # Step 1: Schubfach multiplier calculation
    ##########################################
    var minus_k = _floor_log10_pow2(binary_exp) - FP[type].kappa
    var beta = binary_exp + _floor_log2_pow10(-minus_k)
    var cache_index = -minus_k - FP[type].min_k
    var deltai = _compute_delta[
        CarrierDType, FP[type].total_bits, FP[type].cache_bits
    ](cache_index, beta)
    var z_result = _compute_mul[CarrierDType](
        Scalar[CarrierDType]((two_fc | 1) << beta), cache_index
    )

    ################################################################
    # Step 2: Try larger divisor, remove trailing zeros if necessary
    ################################################################
    sig = _divide_by_pow10[
        FP[type, CarrierDType].kappa + 1, FP[type, CarrierDType].n_max_larger
    ](z_result.integer_part)
    var r = (z_result.integer_part - FP[type].big_divisor * sig)

    while True:
        if r < deltai:
            # Exclude the right endpoint if necessary
            if (
                r
                | Scalar[CarrierDType](not z_result.is_integer)
                | Scalar[CarrierDType](1)
            ) == 0:
                sig -= 1
                r = FP[type].big_divisor
                break
        elif r > deltai:
            break
        else:
            # r == deltai, compare fractional parts
            var x_result = _compute_mul_parity(
                (two_fc - 1).cast[DType.uint64](), cache_index, beta
            )
            if not (x_result.parity | x_result.is_integer):
                break
        # If no break conditions were met
        exp = minus_k + FP[type].kappa + 1
        return _remove_trailing_zeros(sig, exp)

    #######################################################
    # Step 3: Find the significand with the smaller divisor
    #######################################################
    sig *= 10

    # delta is equal to 10^(kappa + elog10(2) - floor(elog10(2))), so dist cannot
    # be larger than r.
    var dist = r - (deltai // 2) + (FP[type].small_divisor // 2)
    var approx_y_parity = ((dist ^ (FP[type].small_divisor // 2)) & 1) != 0

    # Is dist divisible by 10^kappa
    var divisible_by_small_divisor = _check_divisibility_and_divide_by_pow10[
        FP[type].carrier_bits, FP[type].divide_magic_number
    ](dist, FP[type].kappa)

    # Add dist / 10^kappa to the significand.
    sig += dist

    if divisible_by_small_divisor:
        # Check z^(f) >= epsilon^(f).
        # We have either yi == zi - epsiloni or yi == (zi - epsiloni) - 1,
        # where yi == zi - epsilon if and only if z^(f) >= epsilon^(f).
        # Since there are only 2 possibilities, we only need to care about the
        # parity. Also, zi and r should have the same parity since the divisor
        # is an even number.
        var y_result = _compute_mul_parity(
            two_fc.cast[DType.uint64](), cache_index, beta
        )
        if y_result.parity != approx_y_parity:
            sig -= 1

    # No trailing zeroes on this branch, so set exp as the final step
    exp = minus_k + FP[type].kappa


fn _compute_endpoint[
    CarrierDType: DType, sig_bits: Int, total_bits: Int, cache_bits: Int
](cache_index: Int, beta: Int, left_endpoint: Bool) -> Scalar[CarrierDType]:
    @parameter
    if CarrierDType is DType.uint64:
        var cache = cache_f64[cache_index]
        if left_endpoint:
            return (
                (cache.high - (cache.high >> (sig_bits + 2)))
                >> (total_bits - sig_bits - 1 - beta)
            ).cast[CarrierDType]()
        else:
            return (
                (cache.high + (cache.high >> (sig_bits + 1)))
                >> (total_bits - sig_bits - 1 - beta)
            ).cast[CarrierDType]()
    else:
        var cache = cache_f32[cache_index]
        if left_endpoint:
            return (
                (cache - (cache >> (sig_bits + 2)))
                >> (cache_bits - sig_bits - 1 - beta)
            ).cast[CarrierDType]()
        else:
            return (
                (cache + (cache >> (sig_bits + 1)))
                >> (cache_bits - sig_bits - 1 - beta)
            ).cast[CarrierDType]()


fn _print_bits[type: DType](x: Scalar[type]) -> String:
    alias total_bits = sizeof[type]() * 8
    var output = String()

    @parameter
    if not type.is_floating_point():
        for i in reversed(range(total_bits)):
            output.write((x >> i) & 1)
            if i % 8 == 0:
                output.write(" ")
    else:
        alias sig_bits = 23 if type is DType.float32 else 52
        alias exp_bits = 8 if type is DType.float32 else 11
        alias cast_type = DType.uint32 if type is DType.float32 else DType.uint64
        var casted = bitcast[cast_type](x)
        for i in reversed(range(total_bits)):
            output.write((casted >> i) & 1)
            if i == total_bits - 1:
                output.write(" ")
            if i == sig_bits:
                if total_bits == 64:
                    output.write(" ")
                else:
                    output.write("    ")
            if i < sig_bits:
                if (
                    total_bits == 32
                    and (total_bits - sig_bits - 2 - i) % 8 == 0
                ):
                    output.write(" ")
                if total_bits == 64 and (total_bits - sig_bits - i) % 8 == 0:
                    output.write(" ")
    return output


fn _rotr[
    CarrierDType: DType
](n: Scalar[CarrierDType], r: Scalar[CarrierDType]) -> Scalar[CarrierDType]:
    @parameter
    if CarrierDType is DType.uint32:
        var r_masked = r & 31
        return (n >> r_masked) | (n << ((32 - r_masked) & 31))
    else:
        var r_masked = r & 63
        return (n >> r_masked) | (n << ((64 - r_masked) & 63))


fn _floor_log2(n: UInt64) -> Int:
    var count = -1
    var num = n
    while num != 0:
        count += 1
        num >>= 1
    return count


fn _floor_log10_pow2(e: Int) -> Int:
    return (e * 315653) >> 20


fn _floor_log2_pow10(e: Int) -> Int:
    return (e * 1741647) >> 19


fn _umul64(x: UInt32, y: UInt32) -> UInt64:
    return x.cast[DType.uint64]() * y.cast[DType.uint64]()


fn _umul128[
    CarrierDType: DType
](x: Scalar[CarrierDType], y: UInt64) -> _UInt128:
    var a = (x >> 32).cast[DType.uint32]()
    var b = x.cast[DType.uint32]()
    var c = (y >> 32).cast[DType.uint32]()
    var d = y.cast[DType.uint32]()

    var ac = _umul64(a, c)
    var bc = _umul64(b, c)
    var ad = _umul64(a, d)
    var bd = _umul64(b, d)

    var intermediate = (bd >> 32) + _truncate[DType.uint32](ad) + _truncate[
        DType.uint32
    ](bc)

    return _UInt128(
        ac + (intermediate >> 32) + (ad >> 32) + (bc >> 32),
        (intermediate << 32) + _truncate[DType.uint32](bd),
    )


fn _remove_trailing_zeros[
    CarrierDType: DType
](mut sig: Scalar[CarrierDType], mut exp: Int):
    """Fastest alg for removing trailing zeroes:
    https://github.com/jk-jeon/rtz_benchmark.
    """

    @parameter
    if CarrierDType is DType.uint64:
        var r = _rotr(sig * 28999941890838049, 8)
        var b = r < 184467440738
        var s = int(b)
        if b:
            sig = r

        r = _rotr(sig * 182622766329724561, 4)
        b = r < 1844674407370956
        s = s * 2 + int(b)
        if b:
            sig = r

        r = _rotr(sig * 10330176681277348905, 2)
        b = r < 184467440737095517
        s = s * 2 + int(b)
        if b:
            sig = r

        r = _rotr(sig * 14757395258967641293, 1)
        b = r < 1844674407370955162
        s = s * 2 + int(b)
        if b:
            sig = r

        exp += s
    else:
        var r = _rotr(sig * 184254097, 4)
        var b = r < 429497
        var s = int(b)
        if b:
            sig = r

        r = _rotr(sig * 42949673, 2)
        b = r < 42949673
        s = s * 2 + int(b)
        if b:
            sig = r

        r = _rotr(sig * 1288490189, 1)
        b = r < 429496730
        s = s * 2 + int(b)
        if b:
            sig = r

        exp += s


fn _divide_by_pow10[
    CarrierDType: DType, //, N: Int, n_max: Scalar[CarrierDType]
](n: Scalar[CarrierDType]) -> Scalar[CarrierDType]:
    @parameter
    if CarrierDType is DType.uint64:

        @parameter
        if N == 1 and bool(n_max <= 4611686018427387908):
            return _umul128_upper64(n, 1844674407370955162)
        elif N == 3 and bool(n_max <= 15534100272597517998):
            return _umul128_upper64(n, 4722366482869645214) >> 8
        else:
            return n / pow(10, N)
    else:

        @parameter
        if N == 1 and bool(n_max <= 1073741828):
            return (_umul64(n.cast[DType.uint32](), 429496730) >> 32).cast[
                CarrierDType
            ]()
        elif N == 2:
            return (_umul64(n.cast[DType.uint32](), 1374389535) >> 37).cast[
                CarrierDType
            ]()
        else:
            return n / pow(10, N)


fn _umul192_lower128(x: UInt64, y: _UInt128) -> _UInt128:
    """Get lower 128-bits of multiplication of a 64-bit unsigned integer and a
    128-bit unsigned integer.
    """
    var high = x * y.high
    var high_low = _umul128(x, y.low)
    return _UInt128((high + high_low.high) & UInt64.MAX, high_low.low)


fn _compute_mul_parity[
    CarrierDType: DType
](two_f: Scalar[CarrierDType], cache_index: Int, beta: Int) -> _MulParity:
    if CarrierDType is DType.uint64:
        debug_assert(1 <= beta < 64, "beta must be between 1 and 64")
        var r = _umul192_lower128(
            two_f.cast[DType.uint64](), cache_f64[cache_index]
        )
        return _MulParity(
            ((r.high >> (64 - beta)) & 1) != 0,
            (
                ((r.high << beta) & UInt64(0xFFFFFFFFFFFFFFFF))
                | (r.low >> (64 - beta))
            )
            == 0,
        )
    else:
        debug_assert(
            1 <= beta < 32,
            "beta for float types 32bits must be between 1 and 32",
        )
        var r = _umul96_lower64(
            two_f.cast[DType.uint32](), cache_f32[cache_index]
        )
        return _MulParity(
            ((r >> (64 - beta)) & 1) != 0,
            (UInt32(0xFFFFFFFF).cast[DType.uint64]() & (r >> (32 - beta))) == 0,
        )


fn _umul96_lower64(x: UInt32, y: UInt64) -> UInt64:
    return (x.cast[DType.uint64]() * y) & UInt64(0xFFFFFFFFFFFFFFFF)


fn _check_divisibility_and_divide_by_pow10[
    CarrierDType: DType, //,
    carrier_bits: Int,
    divide_magic_number: StaticTuple[UInt32, 2],
](mut n: Scalar[CarrierDType], N: Int) -> Bool:
    # Make sure the computation for max_n does not overflow.
    debug_assert(N + 1 <= _floor_log10_pow2(carrier_bits))

    var magic_number = divide_magic_number[N - 1]
    var prod = (n * magic_number.cast[CarrierDType]()).cast[DType.uint32]()

    var mask = UInt32((UInt32(1) << 16) - 1)
    var result = ((prod & mask) < magic_number)

    n = (prod >> 16).cast[CarrierDType]()
    return result


fn _truncate[
    D: DType, S: Int, //, TruncateType: DType
](u: SIMD[D, S]) -> SIMD[D, S]:
    """Cast to DType to truncate to the width of that type, then cast back to
    original DType.
    """
    return u.cast[TruncateType]().cast[D]()


fn _umul96_upper64[
    CarrierDType: DType
](x: Scalar[CarrierDType], y: UInt64) -> UInt64:
    var yh = (y >> 32).cast[DType.uint32]()
    var yl = y.cast[DType.uint32]()

    var xyh = _umul64(x.cast[DType.uint32](), yh)
    var xyl = _umul64(x.cast[DType.uint32](), yl)
    return xyh + (xyl >> 32)


fn _compute_mul[
    CarrierDType: DType
](u: Scalar[CarrierDType], cache_index: Int) -> _MulResult[CarrierDType]:
    if CarrierDType is DType.uint64:
        var r = _umul192_upper128(u, cache_f64[cache_index])
        return _MulResult[CarrierDType](r.high.cast[CarrierDType](), r.low == 0)
    else:
        var cache_value = cache_f32[cache_index]
        var r = _umul96_upper64(u, cache_value)
        return _MulResult[CarrierDType](
            (r >> 32).cast[CarrierDType](), r.cast[CarrierDType]() == 0
        )


fn _compute_delta[
    CarrierDType: DType, total_bits: Int, cache_bits: Int
](cache_index: Int, beta: Int) -> Scalar[CarrierDType]:
    if CarrierDType is DType.uint64:
        var cache = cache_f64[cache_index]
        return (cache.high >> (total_bits - 1 - beta)).cast[CarrierDType]()
    else:
        var cache = cache_f32[cache_index]
        return (cache >> (cache_bits - 1 - beta)).cast[CarrierDType]()


fn _umul192_upper128[
    CarrierDType: DType
](x: Scalar[CarrierDType], y: _UInt128) -> _UInt128:
    var r = _umul128(x, y.high)
    r += _umul128_upper64(x, y.low).cast[DType.uint64]()
    return r


fn _umul128_upper64[
    CarrierDType: DType
](x: Scalar[CarrierDType], y: UInt64) -> Scalar[CarrierDType]:
    var a = (x >> 32).cast[DType.uint32]()
    var b = x.cast[DType.uint32]()
    var c = (y >> 32).cast[DType.uint32]()
    var d = y.cast[DType.uint32]()

    var ac = _umul64(a, c)
    var bc = _umul64(b, c)
    var ad = _umul64(a, d)
    var bd = _umul64(b, d)

    var intermediate = (bd >> 32) + _truncate[DType.uint32](ad) + _truncate[
        DType.uint32
    ](bc)
    return (ac + (intermediate >> 32) + (ad >> 32) + (bc >> 32)).cast[
        CarrierDType
    ]()


fn _is_finite[exp_bits: Int](exponent: Int) -> Bool:
    return exponent != (1 << exp_bits) - 1


fn _count_factors[
    CarrierDType: DType
](owned n: Scalar[CarrierDType], a: Int) -> Int:
    debug_assert(a > 1)
    var c = 0
    while n % a == 0:
        n /= a
        c += 1
    return c


fn _compute_round_up_for_shorter_interval_case[
    CarrierDType: DType, total_bits: Int, sig_bits: Int, cache_bits: Int
](cache_index: Int, beta: Int) -> Scalar[CarrierDType]:
    if CarrierDType is DType.uint64:
        var cache = cache_f64[cache_index]
        return (
            (
                (cache.high >> (total_bits - sig_bits - 2 - beta)).cast[
                    CarrierDType
                ]()
            )
            + 1
        ) / 2
    else:
        var cache = cache_f32[cache_index]
        return (
            (cache >> (cache_bits - sig_bits - 2 - beta)).cast[CarrierDType]()
            + 1
        ) / 2


fn _case_shorter_interval_left_endpoint_upper_threshold[
    CarrierDType: DType, sig_bits: Int
]() -> Int:
    var k = _count_factors(
        (Scalar[CarrierDType](1) << (sig_bits + 2)) - 1, 5
    ) + 1
    return 2 + _floor_log2(pow(10, k)) // 3


fn _is_left_endpoint_integer_shorter_interval[
    CarrierDType: DType, sig_bits: Int
](binary_exp: Int) -> Bool:
    return (
        binary_exp >= 2
        and binary_exp
        <= _case_shorter_interval_left_endpoint_upper_threshold[
            CarrierDType, sig_bits
        ]()
    )


# fmt: off
alias cache_f32 = StaticTuple[UInt64, 78](
    0x81CEB32C4B43FCF5, 0xA2425FF75E14FC32,
    0xCAD2F7F5359A3B3F, 0xFD87B5F28300CA0E,
    0x9E74D1B791E07E49, 0xC612062576589DDB,
    0xF79687AED3EEC552, 0x9ABE14CD44753B53,
    0xC16D9A0095928A28, 0xF1C90080BAF72CB2,
    0x971DA05074DA7BEF, 0xBCE5086492111AEB,
    0xEC1E4A7DB69561A6, 0x9392EE8E921D5D08,
    0xB877AA3236A4B44A, 0xE69594BEC44DE15C,
    0x901D7CF73AB0ACDA, 0xB424DC35095CD810,
    0xE12E13424BB40E14, 0x8CBCCC096F5088CC,
    0xAFEBFF0BCB24AAFF, 0xDBE6FECEBDEDD5BF,
    0x89705F4136B4A598, 0xABCC77118461CEFD,
    0xD6BF94D5E57A42BD, 0x8637BD05AF6C69B6,
    0xA7C5AC471B478424, 0xD1B71758E219652C,
    0x83126E978D4FDF3C, 0xA3D70A3D70A3D70B,
    0xCCCCCCCCCCCCCCCD, 0x8000000000000000,
    0xA000000000000000, 0xC800000000000000,
    0xFA00000000000000, 0x9C40000000000000,
    0xC350000000000000, 0xF424000000000000,
    0x9896800000000000, 0xBEBC200000000000,
    0xEE6B280000000000, 0x9502F90000000000,
    0xBA43B74000000000, 0xE8D4A51000000000,
    0x9184E72A00000000, 0xB5E620F480000000,
    0xE35FA931A0000000, 0x8E1BC9BF04000000,
    0xB1A2BC2EC5000000, 0xDE0B6B3A76400000,
    0x8AC7230489E80000, 0xAD78EBC5AC620000,
    0xD8D726B7177A8000, 0x878678326EAC9000,
    0xA968163F0A57B400, 0xD3C21BCECCEDA100,
    0x84595161401484A0, 0xA56FA5B99019A5C8,
    0xCECB8F27F4200F3A, 0x813F3978F8940985,
    0xA18F07D736B90BE6, 0xC9F2C9CD04674EDF,
    0xFC6F7C4045812297, 0x9DC5ADA82B70B59E,
    0xC5371912364CE306, 0xF684DF56C3E01BC7,
    0x9A130B963A6C115D, 0xC097CE7BC90715B4,
    0xF0BDC21ABB48DB21, 0x96769950B50D88F5,
    0xBC143FA4E250EB32, 0xEB194F8E1AE525FE,
    0x92EFD1B8D0CF37BF, 0xB7ABC627050305AE,
    0xE596B7B0C643C71A, 0x8F7E32CE7BEA5C70,
    0xB35DBF821AE4F38C, 0xE0352F62A19E306F,
)
# fmt: on

alias cache_f64 = StaticTuple[_UInt128, 619](
    _UInt128(0xFF77B1FCBEBCDC4F, 0x25E8E89C13BB0F7B),
    _UInt128(0x9FAACF3DF73609B1, 0x77B191618C54E9AD),
    _UInt128(0xC795830D75038C1D, 0xD59DF5B9EF6A2418),
    _UInt128(0xF97AE3D0D2446F25, 0x4B0573286B44AD1E),
    _UInt128(0x9BECCE62836AC577, 0x4EE367F9430AEC33),
    _UInt128(0xC2E801FB244576D5, 0x229C41F793CDA740),
    _UInt128(0xF3A20279ED56D48A, 0x6B43527578C11110),
    _UInt128(0x9845418C345644D6, 0x830A13896B78AAAA),
    _UInt128(0xBE5691EF416BD60C, 0x23CC986BC656D554),
    _UInt128(0xEDEC366B11C6CB8F, 0x2CBFBE86B7EC8AA9),
    _UInt128(0x94B3A202EB1C3F39, 0x7BF7D71432F3D6AA),
    _UInt128(0xB9E08A83A5E34F07, 0xDAF5CCD93FB0CC54),
    _UInt128(0xE858AD248F5C22C9, 0xD1B3400F8F9CFF69),
    _UInt128(0x91376C36D99995BE, 0x23100809B9C21FA2),
    _UInt128(0xB58547448FFFFB2D, 0xABD40A0C2832A78B),
    _UInt128(0xE2E69915B3FFF9F9, 0x16C90C8F323F516D),
    _UInt128(0x8DD01FAD907FFC3B, 0xAE3DA7D97F6792E4),
    _UInt128(0xB1442798F49FFB4A, 0x99CD11CFDF41779D),
    _UInt128(0xDD95317F31C7FA1D, 0x40405643D711D584),
    _UInt128(0x8A7D3EEF7F1CFC52, 0x482835EA666B2573),
    _UInt128(0xAD1C8EAB5EE43B66, 0xDA3243650005EED0),
    _UInt128(0xD863B256369D4A40, 0x90BED43E40076A83),
    _UInt128(0x873E4F75E2224E68, 0x5A7744A6E804A292),
    _UInt128(0xA90DE3535AAAE202, 0x711515D0A205CB37),
    _UInt128(0xD3515C2831559A83, 0x0D5A5B44CA873E04),
    _UInt128(0x8412D9991ED58091, 0xE858790AFE9486C3),
    _UInt128(0xA5178FFF668AE0B6, 0x626E974DBE39A873),
    _UInt128(0xCE5D73FF402D98E3, 0xFB0A3D212DC81290),
    _UInt128(0x80FA687F881C7F8E, 0x7CE66634BC9D0B9A),
    _UInt128(0xA139029F6A239F72, 0x1C1FFFC1EBC44E81),
    _UInt128(0xC987434744AC874E, 0xA327FFB266B56221),
    _UInt128(0xFBE9141915D7A922, 0x4BF1FF9F0062BAA9),
    _UInt128(0x9D71AC8FADA6C9B5, 0x6F773FC3603DB4AA),
    _UInt128(0xC4CE17B399107C22, 0xCB550FB4384D21D4),
    _UInt128(0xF6019DA07F549B2B, 0x7E2A53A146606A49),
    _UInt128(0x99C102844F94E0FB, 0x2EDA7444CBFC426E),
    _UInt128(0xC0314325637A1939, 0xFA911155FEFB5309),
    _UInt128(0xF03D93EEBC589F88, 0x793555AB7EBA27CB),
    _UInt128(0x96267C7535B763B5, 0x4BC1558B2F3458DF),
    _UInt128(0xBBB01B9283253CA2, 0x9EB1AAEDFB016F17),
    _UInt128(0xEA9C227723EE8BCB, 0x465E15A979C1CADD),
    _UInt128(0x92A1958A7675175F, 0x0BFACD89EC191ECA),
    _UInt128(0xB749FAED14125D36, 0xCEF980EC671F667C),
    _UInt128(0xE51C79A85916F484, 0x82B7E12780E7401B),
    _UInt128(0x8F31CC0937AE58D2, 0xD1B2ECB8B0908811),
    _UInt128(0xB2FE3F0B8599EF07, 0x861FA7E6DCB4AA16),
    _UInt128(0xDFBDCECE67006AC9, 0x67A791E093E1D49B),
    _UInt128(0x8BD6A141006042BD, 0xE0C8BB2C5C6D24E1),
    _UInt128(0xAECC49914078536D, 0x58FAE9F773886E19),
    _UInt128(0xDA7F5BF590966848, 0xAF39A475506A899F),
    _UInt128(0x888F99797A5E012D, 0x6D8406C952429604),
    _UInt128(0xAAB37FD7D8F58178, 0xC8E5087BA6D33B84),
    _UInt128(0xD5605FCDCF32E1D6, 0xFB1E4A9A90880A65),
    _UInt128(0x855C3BE0A17FCD26, 0x5CF2EEA09A550680),
    _UInt128(0xA6B34AD8C9DFC06F, 0xF42FAA48C0EA481F),
    _UInt128(0xD0601D8EFC57B08B, 0xF13B94DAF124DA27),
    _UInt128(0x823C12795DB6CE57, 0x76C53D08D6B70859),
    _UInt128(0xA2CB1717B52481ED, 0x54768C4B0C64CA6F),
    _UInt128(0xCB7DDCDDA26DA268, 0xA9942F5DCF7DFD0A),
    _UInt128(0xFE5D54150B090B02, 0xD3F93B35435D7C4D),
    _UInt128(0x9EFA548D26E5A6E1, 0xC47BC5014A1A6DB0),
    _UInt128(0xC6B8E9B0709F109A, 0x359AB6419CA1091C),
    _UInt128(0xF867241C8CC6D4C0, 0xC30163D203C94B63),
    _UInt128(0x9B407691D7FC44F8, 0x79E0DE63425DCF1E),
    _UInt128(0xC21094364DFB5636, 0x985915FC12F542E5),
    _UInt128(0xF294B943E17A2BC4, 0x3E6F5B7B17B2939E),
    _UInt128(0x979CF3CA6CEC5B5A, 0xA705992CEECF9C43),
    _UInt128(0xBD8430BD08277231, 0x50C6FF782A838354),
    _UInt128(0xECE53CEC4A314EBD, 0xA4F8BF5635246429),
    _UInt128(0x940F4613AE5ED136, 0x871B7795E136BE9A),
    _UInt128(0xB913179899F68584, 0x28E2557B59846E40),
    _UInt128(0xE757DD7EC07426E5, 0x331AEADA2FE589D0),
    _UInt128(0x9096EA6F3848984F, 0x3FF0D2C85DEF7622),
    _UInt128(0xB4BCA50B065ABE63, 0x0FED077A756B53AA),
    _UInt128(0xE1EBCE4DC7F16DFB, 0xD3E8495912C62895),
    _UInt128(0x8D3360F09CF6E4BD, 0x64712DD7ABBBD95D),
    _UInt128(0xB080392CC4349DEC, 0xBD8D794D96AACFB4),
    _UInt128(0xDCA04777F541C567, 0xECF0D7A0FC5583A1),
    _UInt128(0x89E42CAAF9491B60, 0xF41686C49DB57245),
    _UInt128(0xAC5D37D5B79B6239, 0x311C2875C522CED6),
    _UInt128(0xD77485CB25823AC7, 0x7D633293366B828C),
    _UInt128(0x86A8D39EF77164BC, 0xAE5DFF9C02033198),
    _UInt128(0xA8530886B54DBDEB, 0xD9F57F830283FDFD),
    _UInt128(0xD267CAA862A12D66, 0xD072DF63C324FD7C),
    _UInt128(0x8380DEA93DA4BC60, 0x4247CB9E59F71E6E),
    _UInt128(0xA46116538D0DEB78, 0x52D9BE85F074E609),
    _UInt128(0xCD795BE870516656, 0x67902E276C921F8C),
    _UInt128(0x806BD9714632DFF6, 0x00BA1CD8A3DB53B7),
    _UInt128(0xA086CFCD97BF97F3, 0x80E8A40ECCD228A5),
    _UInt128(0xC8A883C0FDAF7DF0, 0x6122CD128006B2CE),
    _UInt128(0xFAD2A4B13D1B5D6C, 0x796B805720085F82),
    _UInt128(0x9CC3A6EEC6311A63, 0xCBE3303674053BB1),
    _UInt128(0xC3F490AA77BD60FC, 0xBEDBFC4411068A9D),
    _UInt128(0xF4F1B4D515ACB93B, 0xEE92FB5515482D45),
    _UInt128(0x991711052D8BF3C5, 0x751BDD152D4D1C4B),
    _UInt128(0xBF5CD54678EEF0B6, 0xD262D45A78A0635E),
    _UInt128(0xEF340A98172AACE4, 0x86FB897116C87C35),
    _UInt128(0x9580869F0E7AAC0E, 0xD45D35E6AE3D4DA1),
    _UInt128(0xBAE0A846D2195712, 0x8974836059CCA10A),
    _UInt128(0xE998D258869FACD7, 0x2BD1A438703FC94C),
    _UInt128(0x91FF83775423CC06, 0x7B6306A34627DDD0),
    _UInt128(0xB67F6455292CBF08, 0x1A3BC84C17B1D543),
    _UInt128(0xE41F3D6A7377EECA, 0x20CABA5F1D9E4A94),
    _UInt128(0x8E938662882AF53E, 0x547EB47B7282EE9D),
    _UInt128(0xB23867FB2A35B28D, 0xE99E619A4F23AA44),
    _UInt128(0xDEC681F9F4C31F31, 0x6405FA00E2EC94D5),
    _UInt128(0x8B3C113C38F9F37E, 0xDE83BC408DD3DD05),
    _UInt128(0xAE0B158B4738705E, 0x9624AB50B148D446),
    _UInt128(0xD98DDAEE19068C76, 0x3BADD624DD9B0958),
    _UInt128(0x87F8A8D4CFA417C9, 0xE54CA5D70A80E5D7),
    _UInt128(0xA9F6D30A038D1DBC, 0x5E9FCF4CCD211F4D),
    _UInt128(0xD47487CC8470652B, 0x7647C32000696720),
    _UInt128(0x84C8D4DFD2C63F3B, 0x29ECD9F40041E074),
    _UInt128(0xA5FB0A17C777CF09, 0xF468107100525891),
    _UInt128(0xCF79CC9DB955C2CC, 0x7182148D4066EEB5),
    _UInt128(0x81AC1FE293D599BF, 0xC6F14CD848405531),
    _UInt128(0xA21727DB38CB002F, 0xB8ADA00E5A506A7D),
    _UInt128(0xCA9CF1D206FDC03B, 0xA6D90811F0E4851D),
    _UInt128(0xFD442E4688BD304A, 0x908F4A166D1DA664),
    _UInt128(0x9E4A9CEC15763E2E, 0x9A598E4E043287FF),
    _UInt128(0xC5DD44271AD3CDBA, 0x40EFF1E1853F29FE),
    _UInt128(0xF7549530E188C128, 0xD12BEE59E68EF47D),
    _UInt128(0x9A94DD3E8CF578B9, 0x82BB74F8301958CF),
    _UInt128(0xC13A148E3032D6E7, 0xE36A52363C1FAF02),
    _UInt128(0xF18899B1BC3F8CA1, 0xDC44E6C3CB279AC2),
    _UInt128(0x96F5600F15A7B7E5, 0x29AB103A5EF8C0BA),
    _UInt128(0xBCB2B812DB11A5DE, 0x7415D448F6B6F0E8),
    _UInt128(0xEBDF661791D60F56, 0x111B495B3464AD22),
    _UInt128(0x936B9FCEBB25C995, 0xCAB10DD900BEEC35),
    _UInt128(0xB84687C269EF3BFB, 0x3D5D514F40EEA743),
    _UInt128(0xE65829B3046B0AFA, 0x0CB4A5A3112A5113),
    _UInt128(0x8FF71A0FE2C2E6DC, 0x47F0E785EABA72AC),
    _UInt128(0xB3F4E093DB73A093, 0x59ED216765690F57),
    _UInt128(0xE0F218B8D25088B8, 0x306869C13EC3532D),
    _UInt128(0x8C974F7383725573, 0x1E414218C73A13FC),
    _UInt128(0xAFBD2350644EEACF, 0xE5D1929EF90898FB),
    _UInt128(0xDBAC6C247D62A583, 0xDF45F746B74ABF3A),
    _UInt128(0x894BC396CE5DA772, 0x6B8BBA8C328EB784),
    _UInt128(0xAB9EB47C81F5114F, 0x066EA92F3F326565),
    _UInt128(0xD686619BA27255A2, 0xC80A537B0EFEFEBE),
    _UInt128(0x8613FD0145877585, 0xBD06742CE95F5F37),
    _UInt128(0xA798FC4196E952E7, 0x2C48113823B73705),
    _UInt128(0xD17F3B51FCA3A7A0, 0xF75A15862CA504C6),
    _UInt128(0x82EF85133DE648C4, 0x9A984D73DBE722FC),
    _UInt128(0xA3AB66580D5FDAF5, 0xC13E60D0D2E0EBBB),
    _UInt128(0xCC963FEE10B7D1B3, 0x318DF905079926A9),
    _UInt128(0xFFBBCFE994E5C61F, 0xFDF17746497F7053),
    _UInt128(0x9FD561F1FD0F9BD3, 0xFEB6EA8BEDEFA634),
    _UInt128(0xC7CABA6E7C5382C8, 0xFE64A52EE96B8FC1),
    _UInt128(0xF9BD690A1B68637B, 0x3DFDCE7AA3C673B1),
    _UInt128(0x9C1661A651213E2D, 0x06BEA10CA65C084F),
    _UInt128(0xC31BFA0FE5698DB8, 0x486E494FCFF30A63),
    _UInt128(0xF3E2F893DEC3F126, 0x5A89DBA3C3EFCCFB),
    _UInt128(0x986DDB5C6B3A76B7, 0xF89629465A75E01D),
    _UInt128(0xBE89523386091465, 0xF6BBB397F1135824),
    _UInt128(0xEE2BA6C0678B597F, 0x746AA07DED582E2D),
    _UInt128(0x94DB483840B717EF, 0xA8C2A44EB4571CDD),
    _UInt128(0xBA121A4650E4DDEB, 0x92F34D62616CE414),
    _UInt128(0xE896A0D7E51E1566, 0x77B020BAF9C81D18),
    _UInt128(0x915E2486EF32CD60, 0x0ACE1474DC1D122F),
    _UInt128(0xB5B5ADA8AAFF80B8, 0x0D819992132456BB),
    _UInt128(0xE3231912D5BF60E6, 0x10E1FFF697ED6C6A),
    _UInt128(0x8DF5EFABC5979C8F, 0xCA8D3FFA1EF463C2),
    _UInt128(0xB1736B96B6FD83B3, 0xBD308FF8A6B17CB3),
    _UInt128(0xDDD0467C64BCE4A0, 0xAC7CB3F6D05DDBDF),
    _UInt128(0x8AA22C0DBEF60EE4, 0x6BCDF07A423AA96C),
    _UInt128(0xAD4AB7112EB3929D, 0x86C16C98D2C953C7),
    _UInt128(0xD89D64D57A607744, 0xE871C7BF077BA8B8),
    _UInt128(0x87625F056C7C4A8B, 0x11471CD764AD4973),
    _UInt128(0xA93AF6C6C79B5D2D, 0xD598E40D3DD89BD0),
    _UInt128(0xD389B47879823479, 0x4AFF1D108D4EC2C4),
    _UInt128(0x843610CB4BF160CB, 0xCEDF722A585139BB),
    _UInt128(0xA54394FE1EEDB8FE, 0xC2974EB4EE658829),
    _UInt128(0xCE947A3DA6A9273E, 0x733D226229FEEA33),
    _UInt128(0x811CCC668829B887, 0x0806357D5A3F5260),
    _UInt128(0xA163FF802A3426A8, 0xCA07C2DCB0CF26F8),
    _UInt128(0xC9BCFF6034C13052, 0xFC89B393DD02F0B6),
    _UInt128(0xFC2C3F3841F17C67, 0xBBAC2078D443ACE3),
    _UInt128(0x9D9BA7832936EDC0, 0xD54B944B84AA4C0E),
    _UInt128(0xC5029163F384A931, 0x0A9E795E65D4DF12),
    _UInt128(0xF64335BCF065D37D, 0x4D4617B5FF4A16D6),
    _UInt128(0x99EA0196163FA42E, 0x504BCED1BF8E4E46),
    _UInt128(0xC06481FB9BCF8D39, 0xE45EC2862F71E1D7),
    _UInt128(0xF07DA27A82C37088, 0x5D767327BB4E5A4D),
    _UInt128(0x964E858C91BA2655, 0x3A6A07F8D510F870),
    _UInt128(0xBBE226EFB628AFEA, 0x890489F70A55368C),
    _UInt128(0xEADAB0ABA3B2DBE5, 0x2B45AC74CCEA842F),
    _UInt128(0x92C8AE6B464FC96F, 0x3B0B8BC90012929E),
    _UInt128(0xB77ADA0617E3BBCB, 0x09CE6EBB40173745),
    _UInt128(0xE55990879DDCAABD, 0xCC420A6A101D0516),
    _UInt128(0x8F57FA54C2A9EAB6, 0x9FA946824A12232E),
    _UInt128(0xB32DF8E9F3546564, 0x47939822DC96ABFA),
    _UInt128(0xDFF9772470297EBD, 0x59787E2B93BC56F8),
    _UInt128(0x8BFBEA76C619EF36, 0x57EB4EDB3C55B65B),
    _UInt128(0xAEFAE51477A06B03, 0xEDE622920B6B23F2),
    _UInt128(0xDAB99E59958885C4, 0xE95FAB368E45ECEE),
    _UInt128(0x88B402F7FD75539B, 0x11DBCB0218EBB415),
    _UInt128(0xAAE103B5FCD2A881, 0xD652BDC29F26A11A),
    _UInt128(0xD59944A37C0752A2, 0x4BE76D3346F04960),
    _UInt128(0x857FCAE62D8493A5, 0x6F70A4400C562DDC),
    _UInt128(0xA6DFBD9FB8E5B88E, 0xCB4CCD500F6BB953),
    _UInt128(0xD097AD07A71F26B2, 0x7E2000A41346A7A8),
    _UInt128(0x825ECC24C873782F, 0x8ED400668C0C28C9),
    _UInt128(0xA2F67F2DFA90563B, 0x728900802F0F32FB),
    _UInt128(0xCBB41EF979346BCA, 0x4F2B40A03AD2FFBA),
    _UInt128(0xFEA126B7D78186BC, 0xE2F610C84987BFA9),
    _UInt128(0x9F24B832E6B0F436, 0x0DD9CA7D2DF4D7CA),
    _UInt128(0xC6EDE63FA05D3143, 0x91503D1C79720DBC),
    _UInt128(0xF8A95FCF88747D94, 0x75A44C6397CE912B),
    _UInt128(0x9B69DBE1B548CE7C, 0xC986AFBE3EE11ABB),
    _UInt128(0xC24452DA229B021B, 0xFBE85BADCE996169),
    _UInt128(0xF2D56790AB41C2A2, 0xFAE27299423FB9C4),
    _UInt128(0x97C560BA6B0919A5, 0xDCCD879FC967D41B),
    _UInt128(0xBDB6B8E905CB600F, 0x5400E987BBC1C921),
    _UInt128(0xED246723473E3813, 0x290123E9AAB23B69),
    _UInt128(0x9436C0760C86E30B, 0xF9A0B6720AAF6522),
    _UInt128(0xB94470938FA89BCE, 0xF808E40E8D5B3E6A),
    _UInt128(0xE7958CB87392C2C2, 0xB60B1D1230B20E05),
    _UInt128(0x90BD77F3483BB9B9, 0xB1C6F22B5E6F48C3),
    _UInt128(0xB4ECD5F01A4AA828, 0x1E38AEB6360B1AF4),
    _UInt128(0xE2280B6C20DD5232, 0x25C6DA63C38DE1B1),
    _UInt128(0x8D590723948A535F, 0x579C487E5A38AD0F),
    _UInt128(0xB0AF48EC79ACE837, 0x2D835A9DF0C6D852),
    _UInt128(0xDCDB1B2798182244, 0xF8E431456CF88E66),
    _UInt128(0x8A08F0F8BF0F156B, 0x1B8E9ECB641B5900),
    _UInt128(0xAC8B2D36EED2DAC5, 0xE272467E3D222F40),
    _UInt128(0xD7ADF884AA879177, 0x5B0ED81DCC6ABB10),
    _UInt128(0x86CCBB52EA94BAEA, 0x98E947129FC2B4EA),
    _UInt128(0xA87FEA27A539E9A5, 0x3F2398D747B36225),
    _UInt128(0xD29FE4B18E88640E, 0x8EEC7F0D19A03AAE),
    _UInt128(0x83A3EEEEF9153E89, 0x1953CF68300424AD),
    _UInt128(0xA48CEAAAB75A8E2B, 0x5FA8C3423C052DD8),
    _UInt128(0xCDB02555653131B6, 0x3792F412CB06794E),
    _UInt128(0x808E17555F3EBF11, 0xE2BBD88BBEE40BD1),
    _UInt128(0xA0B19D2AB70E6ED6, 0x5B6ACEAEAE9D0EC5),
    _UInt128(0xC8DE047564D20A8B, 0xF245825A5A445276),
    _UInt128(0xFB158592BE068D2E, 0xEED6E2F0F0D56713),
    _UInt128(0x9CED737BB6C4183D, 0x55464DD69685606C),
    _UInt128(0xC428D05AA4751E4C, 0xAA97E14C3C26B887),
    _UInt128(0xF53304714D9265DF, 0xD53DD99F4B3066A9),
    _UInt128(0x993FE2C6D07B7FAB, 0xE546A8038EFE402A),
    _UInt128(0xBF8FDB78849A5F96, 0xDE98520472BDD034),
    _UInt128(0xEF73D256A5C0F77C, 0x963E66858F6D4441),
    _UInt128(0x95A8637627989AAD, 0xDDE7001379A44AA9),
    _UInt128(0xBB127C53B17EC159, 0x5560C018580D5D53),
    _UInt128(0xE9D71B689DDE71AF, 0xAAB8F01E6E10B4A7),
    _UInt128(0x9226712162AB070D, 0xCAB3961304CA70E9),
    _UInt128(0xB6B00D69BB55C8D1, 0x3D607B97C5FD0D23),
    _UInt128(0xE45C10C42A2B3B05, 0x8CB89A7DB77C506B),
    _UInt128(0x8EB98A7A9A5B04E3, 0x77F3608E92ADB243),
    _UInt128(0xB267ED1940F1C61C, 0x55F038B237591ED4),
    _UInt128(0xDF01E85F912E37A3, 0x6B6C46DEC52F6689),
    _UInt128(0x8B61313BBABCE2C6, 0x2323AC4B3B3DA016),
    _UInt128(0xAE397D8AA96C1B77, 0xABEC975E0A0D081B),
    _UInt128(0xD9C7DCED53C72255, 0x96E7BD358C904A22),
    _UInt128(0x881CEA14545C7575, 0x7E50D64177DA2E55),
    _UInt128(0xAA242499697392D2, 0xDDE50BD1D5D0B9EA),
    _UInt128(0xD4AD2DBFC3D07787, 0x955E4EC64B44E865),
    _UInt128(0x84EC3C97DA624AB4, 0xBD5AF13BEF0B113F),
    _UInt128(0xA6274BBDD0FADD61, 0xECB1AD8AEACDD58F),
    _UInt128(0xCFB11EAD453994BA, 0x67DE18EDA5814AF3),
    _UInt128(0x81CEB32C4B43FCF4, 0x80EACF948770CED8),
    _UInt128(0xA2425FF75E14FC31, 0xA1258379A94D028E),
    _UInt128(0xCAD2F7F5359A3B3E, 0x096EE45813A04331),
    _UInt128(0xFD87B5F28300CA0D, 0x8BCA9D6E188853FD),
    _UInt128(0x9E74D1B791E07E48, 0x775EA264CF55347E),
    _UInt128(0xC612062576589DDA, 0x95364AFE032A819E),
    _UInt128(0xF79687AED3EEC551, 0x3A83DDBD83F52205),
    _UInt128(0x9ABE14CD44753B52, 0xC4926A9672793543),
    _UInt128(0xC16D9A0095928A27, 0x75B7053C0F178294),
    _UInt128(0xF1C90080BAF72CB1, 0x5324C68B12DD6339),
    _UInt128(0x971DA05074DA7BEE, 0xD3F6FC16EBCA5E04),
    _UInt128(0xBCE5086492111AEA, 0x88F4BB1CA6BCF585),
    _UInt128(0xEC1E4A7DB69561A5, 0x2B31E9E3D06C32E6),
    _UInt128(0x9392EE8E921D5D07, 0x3AFF322E62439FD0),
    _UInt128(0xB877AA3236A4B449, 0x09BEFEB9FAD487C3),
    _UInt128(0xE69594BEC44DE15B, 0x4C2EBE687989A9B4),
    _UInt128(0x901D7CF73AB0ACD9, 0x0F9D37014BF60A11),
    _UInt128(0xB424DC35095CD80F, 0x538484C19EF38C95),
    _UInt128(0xE12E13424BB40E13, 0x2865A5F206B06FBA),
    _UInt128(0x8CBCCC096F5088CB, 0xF93F87B7442E45D4),
    _UInt128(0xAFEBFF0BCB24AAFE, 0xF78F69A51539D749),
    _UInt128(0xDBE6FECEBDEDD5BE, 0xB573440E5A884D1C),
    _UInt128(0x89705F4136B4A597, 0x31680A88F8953031),
    _UInt128(0xABCC77118461CEFC, 0xFDC20D2B36BA7C3E),
    _UInt128(0xD6BF94D5E57A42BC, 0x3D32907604691B4D),
    _UInt128(0x8637BD05AF6C69B5, 0xA63F9A49C2C1B110),
    _UInt128(0xA7C5AC471B478423, 0x0FCF80DC33721D54),
    _UInt128(0xD1B71758E219652B, 0xD3C36113404EA4A9),
    _UInt128(0x83126E978D4FDF3B, 0x645A1CAC083126EA),
    _UInt128(0xA3D70A3D70A3D70A, 0x3D70A3D70A3D70A4),
    _UInt128(0xCCCCCCCCCCCCCCCC, 0xCCCCCCCCCCCCCCCD),
    _UInt128(0x8000000000000000, 0x0000000000000000),
    _UInt128(0xA000000000000000, 0x0000000000000000),
    _UInt128(0xC800000000000000, 0x0000000000000000),
    _UInt128(0xFA00000000000000, 0x0000000000000000),
    _UInt128(0x9C40000000000000, 0x0000000000000000),
    _UInt128(0xC350000000000000, 0x0000000000000000),
    _UInt128(0xF424000000000000, 0x0000000000000000),
    _UInt128(0x9896800000000000, 0x0000000000000000),
    _UInt128(0xBEBC200000000000, 0x0000000000000000),
    _UInt128(0xEE6B280000000000, 0x0000000000000000),
    _UInt128(0x9502F90000000000, 0x0000000000000000),
    _UInt128(0xBA43B74000000000, 0x0000000000000000),
    _UInt128(0xE8D4A51000000000, 0x0000000000000000),
    _UInt128(0x9184E72A00000000, 0x0000000000000000),
    _UInt128(0xB5E620F480000000, 0x0000000000000000),
    _UInt128(0xE35FA931A0000000, 0x0000000000000000),
    _UInt128(0x8E1BC9BF04000000, 0x0000000000000000),
    _UInt128(0xB1A2BC2EC5000000, 0x0000000000000000),
    _UInt128(0xDE0B6B3A76400000, 0x0000000000000000),
    _UInt128(0x8AC7230489E80000, 0x0000000000000000),
    _UInt128(0xAD78EBC5AC620000, 0x0000000000000000),
    _UInt128(0xD8D726B7177A8000, 0x0000000000000000),
    _UInt128(0x878678326EAC9000, 0x0000000000000000),
    _UInt128(0xA968163F0A57B400, 0x0000000000000000),
    _UInt128(0xD3C21BCECCEDA100, 0x0000000000000000),
    _UInt128(0x84595161401484A0, 0x0000000000000000),
    _UInt128(0xA56FA5B99019A5C8, 0x0000000000000000),
    _UInt128(0xCECB8F27F4200F3A, 0x0000000000000000),
    _UInt128(0x813F3978F8940984, 0x4000000000000000),
    _UInt128(0xA18F07D736B90BE5, 0x5000000000000000),
    _UInt128(0xC9F2C9CD04674EDE, 0xA400000000000000),
    _UInt128(0xFC6F7C4045812296, 0x4D00000000000000),
    _UInt128(0x9DC5ADA82B70B59D, 0xF020000000000000),
    _UInt128(0xC5371912364CE305, 0x6C28000000000000),
    _UInt128(0xF684DF56C3E01BC6, 0xC732000000000000),
    _UInt128(0x9A130B963A6C115C, 0x3C7F400000000000),
    _UInt128(0xC097CE7BC90715B3, 0x4B9F100000000000),
    _UInt128(0xF0BDC21ABB48DB20, 0x1E86D40000000000),
    _UInt128(0x96769950B50D88F4, 0x1314448000000000),
    _UInt128(0xBC143FA4E250EB31, 0x17D955A000000000),
    _UInt128(0xEB194F8E1AE525FD, 0x5DCFAB0800000000),
    _UInt128(0x92EFD1B8D0CF37BE, 0x5AA1CAE500000000),
    _UInt128(0xB7ABC627050305AD, 0xF14A3D9E40000000),
    _UInt128(0xE596B7B0C643C719, 0x6D9CCD05D0000000),
    _UInt128(0x8F7E32CE7BEA5C6F, 0xE4820023A2000000),
    _UInt128(0xB35DBF821AE4F38B, 0xDDA2802C8A800000),
    _UInt128(0xE0352F62A19E306E, 0xD50B2037AD200000),
    _UInt128(0x8C213D9DA502DE45, 0x4526F422CC340000),
    _UInt128(0xAF298D050E4395D6, 0x9670B12B7F410000),
    _UInt128(0xDAF3F04651D47B4C, 0x3C0CDD765F114000),
    _UInt128(0x88D8762BF324CD0F, 0xA5880A69FB6AC800),
    _UInt128(0xAB0E93B6EFEE0053, 0x8EEA0D047A457A00),
    _UInt128(0xD5D238A4ABE98068, 0x72A4904598D6D880),
    _UInt128(0x85A36366EB71F041, 0x47A6DA2B7F864750),
    _UInt128(0xA70C3C40A64E6C51, 0x999090B65F67D924),
    _UInt128(0xD0CF4B50CFE20765, 0xFFF4B4E3F741CF6D),
    _UInt128(0x82818F1281ED449F, 0xBFF8F10E7A8921A5),
    _UInt128(0xA321F2D7226895C7, 0xAFF72D52192B6A0E),
    _UInt128(0xCBEA6F8CEB02BB39, 0x9BF4F8A69F764491),
    _UInt128(0xFEE50B7025C36A08, 0x02F236D04753D5B5),
    _UInt128(0x9F4F2726179A2245, 0x01D762422C946591),
    _UInt128(0xC722F0EF9D80AAD6, 0x424D3AD2B7B97EF6),
    _UInt128(0xF8EBAD2B84E0D58B, 0xD2E0898765A7DEB3),
    _UInt128(0x9B934C3B330C8577, 0x63CC55F49F88EB30),
    _UInt128(0xC2781F49FFCFA6D5, 0x3CBF6B71C76B25FC),
    _UInt128(0xF316271C7FC3908A, 0x8BEF464E3945EF7B),
    _UInt128(0x97EDD871CFDA3A56, 0x97758BF0E3CBB5AD),
    _UInt128(0xBDE94E8E43D0C8EC, 0x3D52EEED1CBEA318),
    _UInt128(0xED63A231D4C4FB27, 0x4CA7AAA863EE4BDE),
    _UInt128(0x945E455F24FB1CF8, 0x8FE8CAA93E74EF6B),
    _UInt128(0xB975D6B6EE39E436, 0xB3E2FD538E122B45),
    _UInt128(0xE7D34C64A9C85D44, 0x60DBBCA87196B617),
    _UInt128(0x90E40FBEEA1D3A4A, 0xBC8955E946FE31CE),
    _UInt128(0xB51D13AEA4A488DD, 0x6BABAB6398BDBE42),
    _UInt128(0xE264589A4DCDAB14, 0xC696963C7EED2DD2),
    _UInt128(0x8D7EB76070A08AEC, 0xFC1E1DE5CF543CA3),
    _UInt128(0xB0DE65388CC8ADA8, 0x3B25A55F43294BCC),
    _UInt128(0xDD15FE86AFFAD912, 0x49EF0EB713F39EBF),
    _UInt128(0x8A2DBF142DFCC7AB, 0x6E3569326C784338),
    _UInt128(0xACB92ED9397BF996, 0x49C2C37F07965405),
    _UInt128(0xD7E77A8F87DAF7FB, 0xDC33745EC97BE907),
    _UInt128(0x86F0AC99B4E8DAFD, 0x69A028BB3DED71A4),
    _UInt128(0xA8ACD7C0222311BC, 0xC40832EA0D68CE0D),
    _UInt128(0xD2D80DB02AABD62B, 0xF50A3FA490C30191),
    _UInt128(0x83C7088E1AAB65DB, 0x792667C6DA79E0FB),
    _UInt128(0xA4B8CAB1A1563F52, 0x577001B891185939),
    _UInt128(0xCDE6FD5E09ABCF26, 0xED4C0226B55E6F87),
    _UInt128(0x80B05E5AC60B6178, 0x544F8158315B05B5),
    _UInt128(0xA0DC75F1778E39D6, 0x696361AE3DB1C722),
    _UInt128(0xC913936DD571C84C, 0x03BC3A19CD1E38EA),
    _UInt128(0xFB5878494ACE3A5F, 0x04AB48A04065C724),
    _UInt128(0x9D174B2DCEC0E47B, 0x62EB0D64283F9C77),
    _UInt128(0xC45D1DF942711D9A, 0x3BA5D0BD324F8395),
    _UInt128(0xF5746577930D6500, 0xCA8F44EC7EE3647A),
    _UInt128(0x9968BF6ABBE85F20, 0x7E998B13CF4E1ECC),
    _UInt128(0xBFC2EF456AE276E8, 0x9E3FEDD8C321A67F),
    _UInt128(0xEFB3AB16C59B14A2, 0xC5CFE94EF3EA101F),
    _UInt128(0x95D04AEE3B80ECE5, 0xBBA1F1D158724A13),
    _UInt128(0xBB445DA9CA61281F, 0x2A8A6E45AE8EDC98),
    _UInt128(0xEA1575143CF97226, 0xF52D09D71A3293BE),
    _UInt128(0x924D692CA61BE758, 0x593C2626705F9C57),
    _UInt128(0xB6E0C377CFA2E12E, 0x6F8B2FB00C77836D),
    _UInt128(0xE498F455C38B997A, 0x0B6DFB9C0F956448),
    _UInt128(0x8EDF98B59A373FEC, 0x4724BD4189BD5EAD),
    _UInt128(0xB2977EE300C50FE7, 0x58EDEC91EC2CB658),
    _UInt128(0xDF3D5E9BC0F653E1, 0x2F2967B66737E3EE),
    _UInt128(0x8B865B215899F46C, 0xBD79E0D20082EE75),
    _UInt128(0xAE67F1E9AEC07187, 0xECD8590680A3AA12),
    _UInt128(0xDA01EE641A708DE9, 0xE80E6F4820CC9496),
    _UInt128(0x884134FE908658B2, 0x3109058D147FDCDE),
    _UInt128(0xAA51823E34A7EEDE, 0xBD4B46F0599FD416),
    _UInt128(0xD4E5E2CDC1D1EA96, 0x6C9E18AC7007C91B),
    _UInt128(0x850FADC09923329E, 0x03E2CF6BC604DDB1),
    _UInt128(0xA6539930BF6BFF45, 0x84DB8346B786151D),
    _UInt128(0xCFE87F7CEF46FF16, 0xE612641865679A64),
    _UInt128(0x81F14FAE158C5F6E, 0x4FCB7E8F3F60C07F),
    _UInt128(0xA26DA3999AEF7749, 0xE3BE5E330F38F09E),
    _UInt128(0xCB090C8001AB551C, 0x5CADF5BFD3072CC6),
    _UInt128(0xFDCB4FA002162A63, 0x73D9732FC7C8F7F7),
    _UInt128(0x9E9F11C4014DDA7E, 0x2867E7FDDCDD9AFB),
    _UInt128(0xC646D63501A1511D, 0xB281E1FD541501B9),
    _UInt128(0xF7D88BC24209A565, 0x1F225A7CA91A4227),
    _UInt128(0x9AE757596946075F, 0x3375788DE9B06959),
    _UInt128(0xC1A12D2FC3978937, 0x0052D6B1641C83AF),
    _UInt128(0xF209787BB47D6B84, 0xC0678C5DBD23A49B),
    _UInt128(0x9745EB4D50CE6332, 0xF840B7BA963646E1),
    _UInt128(0xBD176620A501FBFF, 0xB650E5A93BC3D899),
    _UInt128(0xEC5D3FA8CE427AFF, 0xA3E51F138AB4CEBF),
    _UInt128(0x93BA47C980E98CDF, 0xC66F336C36B10138),
    _UInt128(0xB8A8D9BBE123F017, 0xB80B0047445D4185),
    _UInt128(0xE6D3102AD96CEC1D, 0xA60DC059157491E6),
    _UInt128(0x9043EA1AC7E41392, 0x87C89837AD68DB30),
    _UInt128(0xB454E4A179DD1877, 0x29BABE4598C311FC),
    _UInt128(0xE16A1DC9D8545E94, 0xF4296DD6FEF3D67B),
    _UInt128(0x8CE2529E2734BB1D, 0x1899E4A65F58660D),
    _UInt128(0xB01AE745B101E9E4, 0x5EC05DCFF72E7F90),
    _UInt128(0xDC21A1171D42645D, 0x76707543F4FA1F74),
    _UInt128(0x899504AE72497EBA, 0x6A06494A791C53A9),
    _UInt128(0xABFA45DA0EDBDE69, 0x0487DB9D17636893),
    _UInt128(0xD6F8D7509292D603, 0x45A9D2845D3C42B7),
    _UInt128(0x865B86925B9BC5C2, 0x0B8A2392BA45A9B3),
    _UInt128(0xA7F26836F282B732, 0x8E6CAC7768D7141F),
    _UInt128(0xD1EF0244AF2364FF, 0x3207D795430CD927),
    _UInt128(0x8335616AED761F1F, 0x7F44E6BD49E807B9),
    _UInt128(0xA402B9C5A8D3A6E7, 0x5F16206C9C6209A7),
    _UInt128(0xCD036837130890A1, 0x36DBA887C37A8C10),
    _UInt128(0x802221226BE55A64, 0xC2494954DA2C978A),
    _UInt128(0xA02AA96B06DEB0FD, 0xF2DB9BAA10B7BD6D),
    _UInt128(0xC83553C5C8965D3D, 0x6F92829494E5ACC8),
    _UInt128(0xFA42A8B73ABBF48C, 0xCB772339BA1F17FA),
    _UInt128(0x9C69A97284B578D7, 0xFF2A760414536EFC),
    _UInt128(0xC38413CF25E2D70D, 0xFEF5138519684ABB),
    _UInt128(0xF46518C2EF5B8CD1, 0x7EB258665FC25D6A),
    _UInt128(0x98BF2F79D5993802, 0xEF2F773FFBD97A62),
    _UInt128(0xBEEEFB584AFF8603, 0xAAFB550FFACFD8FB),
    _UInt128(0xEEAABA2E5DBF6784, 0x95BA2A53F983CF39),
    _UInt128(0x952AB45CFA97A0B2, 0xDD945A747BF26184),
    _UInt128(0xBA756174393D88DF, 0x94F971119AEEF9E5),
    _UInt128(0xE912B9D1478CEB17, 0x7A37CD5601AAB85E),
    _UInt128(0x91ABB422CCB812EE, 0xAC62E055C10AB33B),
    _UInt128(0xB616A12B7FE617AA, 0x577B986B314D600A),
    _UInt128(0xE39C49765FDF9D94, 0xED5A7E85FDA0B80C),
    _UInt128(0x8E41ADE9FBEBC27D, 0x14588F13BE847308),
    _UInt128(0xB1D219647AE6B31C, 0x596EB2D8AE258FC9),
    _UInt128(0xDE469FBD99A05FE3, 0x6FCA5F8ED9AEF3BC),
    _UInt128(0x8AEC23D680043BEE, 0x25DE7BB9480D5855),
    _UInt128(0xADA72CCC20054AE9, 0xAF561AA79A10AE6B),
    _UInt128(0xD910F7FF28069DA4, 0x1B2BA1518094DA05),
    _UInt128(0x87AA9AFF79042286, 0x90FB44D2F05D0843),
    _UInt128(0xA99541BF57452B28, 0x353A1607AC744A54),
    _UInt128(0xD3FA922F2D1675F2, 0x42889B8997915CE9),
    _UInt128(0x847C9B5D7C2E09B7, 0x69956135FEBADA12),
    _UInt128(0xA59BC234DB398C25, 0x43FAB9837E699096),
    _UInt128(0xCF02B2C21207EF2E, 0x94F967E45E03F4BC),
    _UInt128(0x8161AFB94B44F57D, 0x1D1BE0EEBAC278F6),
    _UInt128(0xA1BA1BA79E1632DC, 0x6462D92A69731733),
    _UInt128(0xCA28A291859BBF93, 0x7D7B8F7503CFDCFF),
    _UInt128(0xFCB2CB35E702AF78, 0x5CDA735244C3D43F),
    _UInt128(0x9DEFBF01B061ADAB, 0x3A0888136AFA64A8),
    _UInt128(0xC56BAEC21C7A1916, 0x088AAA1845B8FDD1),
    _UInt128(0xF6C69A72A3989F5B, 0x8AAD549E57273D46),
    _UInt128(0x9A3C2087A63F6399, 0x36AC54E2F678864C),
    _UInt128(0xC0CB28A98FCF3C7F, 0x84576A1BB416A7DE),
    _UInt128(0xF0FDF2D3F3C30B9F, 0x656D44A2A11C51D6),
    _UInt128(0x969EB7C47859E743, 0x9F644AE5A4B1B326),
    _UInt128(0xBC4665B596706114, 0x873D5D9F0DDE1FEF),
    _UInt128(0xEB57FF22FC0C7959, 0xA90CB506D155A7EB),
    _UInt128(0x9316FF75DD87CBD8, 0x09A7F12442D588F3),
    _UInt128(0xB7DCBF5354E9BECE, 0x0C11ED6D538AEB30),
    _UInt128(0xE5D3EF282A242E81, 0x8F1668C8A86DA5FB),
    _UInt128(0x8FA475791A569D10, 0xF96E017D694487BD),
    _UInt128(0xB38D92D760EC4455, 0x37C981DCC395A9AD),
    _UInt128(0xE070F78D3927556A, 0x85BBE253F47B1418),
    _UInt128(0x8C469AB843B89562, 0x93956D7478CCEC8F),
    _UInt128(0xAF58416654A6BABB, 0x387AC8D1970027B3),
    _UInt128(0xDB2E51BFE9D0696A, 0x06997B05FCC0319F),
    _UInt128(0x88FCF317F22241E2, 0x441FECE3BDF81F04),
    _UInt128(0xAB3C2FDDEEAAD25A, 0xD527E81CAD7626C4),
    _UInt128(0xD60B3BD56A5586F1, 0x8A71E223D8D3B075),
    _UInt128(0x85C7056562757456, 0xF6872D5667844E4A),
    _UInt128(0xA738C6BEBB12D16C, 0xB428F8AC016561DC),
    _UInt128(0xD106F86E69D785C7, 0xE13336D701BEBA53),
    _UInt128(0x82A45B450226B39C, 0xECC0024661173474),
    _UInt128(0xA34D721642B06084, 0x27F002D7F95D0191),
    _UInt128(0xCC20CE9BD35C78A5, 0x31EC038DF7B441F5),
    _UInt128(0xFF290242C83396CE, 0x7E67047175A15272),
    _UInt128(0x9F79A169BD203E41, 0x0F0062C6E984D387),
    _UInt128(0xC75809C42C684DD1, 0x52C07B78A3E60869),
    _UInt128(0xF92E0C3537826145, 0xA7709A56CCDF8A83),
    _UInt128(0x9BBCC7A142B17CCB, 0x88A66076400BB692),
    _UInt128(0xC2ABF989935DDBFE, 0x6ACFF893D00EA436),
    _UInt128(0xF356F7EBF83552FE, 0x0583F6B8C4124D44),
    _UInt128(0x98165AF37B2153DE, 0xC3727A337A8B704B),
    _UInt128(0xBE1BF1B059E9A8D6, 0x744F18C0592E4C5D),
    _UInt128(0xEDA2EE1C7064130C, 0x1162DEF06F79DF74),
    _UInt128(0x9485D4D1C63E8BE7, 0x8ADDCB5645AC2BA9),
    _UInt128(0xB9A74A0637CE2EE1, 0x6D953E2BD7173693),
    _UInt128(0xE8111C87C5C1BA99, 0xC8FA8DB6CCDD0438),
    _UInt128(0x910AB1D4DB9914A0, 0x1D9C9892400A22A3),
    _UInt128(0xB54D5E4A127F59C8, 0x2503BEB6D00CAB4C),
    _UInt128(0xE2A0B5DC971F303A, 0x2E44AE64840FD61E),
    _UInt128(0x8DA471A9DE737E24, 0x5CEAECFED289E5D3),
    _UInt128(0xB10D8E1456105DAD, 0x7425A83E872C5F48),
    _UInt128(0xDD50F1996B947518, 0xD12F124E28F7771A),
    _UInt128(0x8A5296FFE33CC92F, 0x82BD6B70D99AAA70),
    _UInt128(0xACE73CBFDC0BFB7B, 0x636CC64D1001550C),
    _UInt128(0xD8210BEFD30EFA5A, 0x3C47F7E05401AA4F),
    _UInt128(0x8714A775E3E95C78, 0x65ACFAEC34810A72),
    _UInt128(0xA8D9D1535CE3B396, 0x7F1839A741A14D0E),
    _UInt128(0xD31045A8341CA07C, 0x1EDE48111209A051),
    _UInt128(0x83EA2B892091E44D, 0x934AED0AAB460433),
    _UInt128(0xA4E4B66B68B65D60, 0xF81DA84D56178540),
    _UInt128(0xCE1DE40642E3F4B9, 0x36251260AB9D668F),
    _UInt128(0x80D2AE83E9CE78F3, 0xC1D72B7C6B42601A),
    _UInt128(0xA1075A24E4421730, 0xB24CF65B8612F820),
    _UInt128(0xC94930AE1D529CFC, 0xDEE033F26797B628),
    _UInt128(0xFB9B7CD9A4A7443C, 0x169840EF017DA3B2),
    _UInt128(0x9D412E0806E88AA5, 0x8E1F289560EE864F),
    _UInt128(0xC491798A08A2AD4E, 0xF1A6F2BAB92A27E3),
    _UInt128(0xF5B5D7EC8ACB58A2, 0xAE10AF696774B1DC),
    _UInt128(0x9991A6F3D6BF1765, 0xACCA6DA1E0A8EF2A),
    _UInt128(0xBFF610B0CC6EDD3F, 0x17FD090A58D32AF4),
    _UInt128(0xEFF394DCFF8A948E, 0xDDFC4B4CEF07F5B1),
    _UInt128(0x95F83D0A1FB69CD9, 0x4ABDAF101564F98F),
    _UInt128(0xBB764C4CA7A4440F, 0x9D6D1AD41ABE37F2),
    _UInt128(0xEA53DF5FD18D5513, 0x84C86189216DC5EE),
    _UInt128(0x92746B9BE2F8552C, 0x32FD3CF5B4E49BB5),
    _UInt128(0xB7118682DBB66A77, 0x3FBC8C33221DC2A2),
    _UInt128(0xE4D5E82392A40515, 0x0FABAF3FEAA5334B),
    _UInt128(0x8F05B1163BA6832D, 0x29CB4D87F2A7400F),
    _UInt128(0xB2C71D5BCA9023F8, 0x743E20E9EF511013),
    _UInt128(0xDF78E4B2BD342CF6, 0x914DA9246B255417),
    _UInt128(0x8BAB8EEFB6409C1A, 0x1AD089B6C2F7548F),
    _UInt128(0xAE9672ABA3D0C320, 0xA184AC2473B529B2),
    _UInt128(0xDA3C0F568CC4F3E8, 0xC9E5D72D90A2741F),
    _UInt128(0x8865899617FB1871, 0x7E2FA67C7A658893),
    _UInt128(0xAA7EEBFB9DF9DE8D, 0xDDBB901B98FEEAB8),
    _UInt128(0xD51EA6FA85785631, 0x552A74227F3EA566),
    _UInt128(0x8533285C936B35DE, 0xD53A88958F872760),
    _UInt128(0xA67FF273B8460356, 0x8A892ABAF368F138),
    _UInt128(0xD01FEF10A657842C, 0x2D2B7569B0432D86),
    _UInt128(0x8213F56A67F6B29B, 0x9C3B29620E29FC74),
    _UInt128(0xA298F2C501F45F42, 0x8349F3BA91B47B90),
    _UInt128(0xCB3F2F7642717713, 0x241C70A936219A74),
    _UInt128(0xFE0EFB53D30DD4D7, 0xED238CD383AA0111),
    _UInt128(0x9EC95D1463E8A506, 0xF4363804324A40AB),
    _UInt128(0xC67BB4597CE2CE48, 0xB143C6053EDCD0D6),
    _UInt128(0xF81AA16FDC1B81DA, 0xDD94B7868E94050B),
    _UInt128(0x9B10A4E5E9913128, 0xCA7CF2B4191C8327),
    _UInt128(0xC1D4CE1F63F57D72, 0xFD1C2F611F63A3F1),
    _UInt128(0xF24A01A73CF2DCCF, 0xBC633B39673C8CED),
    _UInt128(0x976E41088617CA01, 0xD5BE0503E085D814),
    _UInt128(0xBD49D14AA79DBC82, 0x4B2D8644D8A74E19),
    _UInt128(0xEC9C459D51852BA2, 0xDDF8E7D60ED1219F),
    _UInt128(0x93E1AB8252F33B45, 0xCABB90E5C942B504),
    _UInt128(0xB8DA1662E7B00A17, 0x3D6A751F3B936244),
    _UInt128(0xE7109BFBA19C0C9D, 0x0CC512670A783AD5),
    _UInt128(0x906A617D450187E2, 0x27FB2B80668B24C6),
    _UInt128(0xB484F9DC9641E9DA, 0xB1F9F660802DEDF7),
    _UInt128(0xE1A63853BBD26451, 0x5E7873F8A0396974),
    _UInt128(0x8D07E33455637EB2, 0xDB0B487B6423E1E9),
    _UInt128(0xB049DC016ABC5E5F, 0x91CE1A9A3D2CDA63),
    _UInt128(0xDC5C5301C56B75F7, 0x7641A140CC7810FC),
    _UInt128(0x89B9B3E11B6329BA, 0xA9E904C87FCB0A9E),
    _UInt128(0xAC2820D9623BF429, 0x546345FA9FBDCD45),
    _UInt128(0xD732290FBACAF133, 0xA97C177947AD4096),
    _UInt128(0x867F59A9D4BED6C0, 0x49ED8EABCCCC485E),
    _UInt128(0xA81F301449EE8C70, 0x5C68F256BFFF5A75),
    _UInt128(0xD226FC195C6A2F8C, 0x73832EEC6FFF3112),
    _UInt128(0x83585D8FD9C25DB7, 0xC831FD53C5FF7EAC),
    _UInt128(0xA42E74F3D032F525, 0xBA3E7CA8B77F5E56),
    _UInt128(0xCD3A1230C43FB26F, 0x28CE1BD2E55F35EC),
    _UInt128(0x80444B5E7AA7CF85, 0x7980D163CF5B81B4),
    _UInt128(0xA0555E361951C366, 0xD7E105BCC3326220),
    _UInt128(0xC86AB5C39FA63440, 0x8DD9472BF3FEFAA8),
    _UInt128(0xFA856334878FC150, 0xB14F98F6F0FEB952),
    _UInt128(0x9C935E00D4B9D8D2, 0x6ED1BF9A569F33D4),
    _UInt128(0xC3B8358109E84F07, 0x0A862F80EC4700C9),
    _UInt128(0xF4A642E14C6262C8, 0xCD27BB612758C0FB),
    _UInt128(0x98E7E9CCCFBD7DBD, 0x8038D51CB897789D),
    _UInt128(0xBF21E44003ACDD2C, 0xE0470A63E6BD56C4),
    _UInt128(0xEEEA5D5004981478, 0x1858CCFCE06CAC75),
    _UInt128(0x95527A5202DF0CCB, 0x0F37801E0C43EBC9),
    _UInt128(0xBAA718E68396CFFD, 0xD30560258F54E6BB),
    _UInt128(0xE950DF20247C83FD, 0x47C6B82EF32A206A),
    _UInt128(0x91D28B7416CDD27E, 0x4CDC331D57FA5442),
    _UInt128(0xB6472E511C81471D, 0xE0133FE4ADF8E953),
    _UInt128(0xE3D8F9E563A198E5, 0x58180FDDD97723A7),
    _UInt128(0x8E679C2F5E44FF8F, 0x570F09EAA7EA7649),
    _UInt128(0xB201833B35D63F73, 0x2CD2CC6551E513DB),
    _UInt128(0xDE81E40A034BCF4F, 0xF8077F7EA65E58D2),
    _UInt128(0x8B112E86420F6191, 0xFB04AFAF27FAF783),
    _UInt128(0xADD57A27D29339F6, 0x79C5DB9AF1F9B564),
    _UInt128(0xD94AD8B1C7380874, 0x18375281AE7822BD),
    _UInt128(0x87CEC76F1C830548, 0x8F2293910D0B15B6),
    _UInt128(0xA9C2794AE3A3C69A, 0xB2EB3875504DDB23),
    _UInt128(0xD433179D9C8CB841, 0x5FA60692A46151EC),
    _UInt128(0x849FEEC281D7F328, 0xDBC7C41BA6BCD334),
    _UInt128(0xA5C7EA73224DEFF3, 0x12B9B522906C0801),
    _UInt128(0xCF39E50FEAE16BEF, 0xD768226B34870A01),
    _UInt128(0x81842F29F2CCE375, 0xE6A1158300D46641),
    _UInt128(0xA1E53AF46F801C53, 0x60495AE3C1097FD1),
    _UInt128(0xCA5E89B18B602368, 0x385BB19CB14BDFC5),
    _UInt128(0xFCF62C1DEE382C42, 0x46729E03DD9ED7B6),
    _UInt128(0x9E19DB92B4E31BA9, 0x6C07A2C26A8346D2),
    _UInt128(0xC5A05277621BE293, 0xC7098B7305241886),
    _UInt128(0xF70867153AA2DB38, 0xB8CBEE4FC66D1EA8),
)

alias float8e5m2_to_str = StaticTuple[StringLiteral, 256](
    "0.0",
    "1.52587890625e-05",
    "3.0517578125e-05",
    "4.57763671875e-05",
    "6.103515625e-05",
    "7.62939453125e-05",
    "9.1552734375e-05",
    "0.0001068115234375",
    "0.0001220703125",
    "0.000152587890625",
    "0.00018310546875",
    "0.000213623046875",
    "0.000244140625",
    "0.00030517578125",
    "0.0003662109375",
    "0.00042724609375",
    "0.00048828125",
    "0.0006103515625",
    "0.000732421875",
    "0.0008544921875",
    "0.0009765625",
    "0.001220703125",
    "0.00146484375",
    "0.001708984375",
    "0.001953125",
    "0.00244140625",
    "0.0029296875",
    "0.00341796875",
    "0.00390625",
    "0.0048828125",
    "0.005859375",
    "0.0068359375",
    "0.0078125",
    "0.009765625",
    "0.01171875",
    "0.013671875",
    "0.015625",
    "0.01953125",
    "0.0234375",
    "0.02734375",
    "0.03125",
    "0.0390625",
    "0.046875",
    "0.0546875",
    "0.0625",
    "0.078125",
    "0.09375",
    "0.109375",
    "0.125",
    "0.15625",
    "0.1875",
    "0.21875",
    "0.25",
    "0.3125",
    "0.375",
    "0.4375",
    "0.5",
    "0.625",
    "0.75",
    "0.875",
    "1.0",
    "1.25",
    "1.5",
    "1.75",
    "2.0",
    "2.5",
    "3.0",
    "3.5",
    "4.0",
    "5.0",
    "6.0",
    "7.0",
    "8.0",
    "10.0",
    "12.0",
    "14.0",
    "16.0",
    "20.0",
    "24.0",
    "28.0",
    "32.0",
    "40.0",
    "48.0",
    "56.0",
    "64.0",
    "80.0",
    "96.0",
    "112.0",
    "128.0",
    "160.0",
    "192.0",
    "224.0",
    "256.0",
    "320.0",
    "384.0",
    "448.0",
    "512.0",
    "640.0",
    "768.0",
    "896.0",
    "1024.0",
    "1280.0",
    "1536.0",
    "1792.0",
    "2048.0",
    "2560.0",
    "3072.0",
    "3584.0",
    "4096.0",
    "5120.0",
    "6144.0",
    "7168.0",
    "8192.0",
    "10240.0",
    "12288.0",
    "14336.0",
    "16384.0",
    "20480.0",
    "24576.0",
    "28672.0",
    "32768.0",
    "40960.0",
    "49152.0",
    "57344.0",
    "inf",
    "nan",
    "nan",
    "nan",
    "-0.0",
    "-1.52587890625e-05",
    "-3.0517578125e-05",
    "-4.57763671875e-05",
    "-6.103515625e-05",
    "-7.62939453125e-05",
    "-9.1552734375e-05",
    "-0.0001068115234375",
    "-0.0001220703125",
    "-0.000152587890625",
    "-0.00018310546875",
    "-0.000213623046875",
    "-0.000244140625",
    "-0.00030517578125",
    "-0.0003662109375",
    "-0.00042724609375",
    "-0.00048828125",
    "-0.0006103515625",
    "-0.000732421875",
    "-0.0008544921875",
    "-0.0009765625",
    "-0.001220703125",
    "-0.00146484375",
    "-0.001708984375",
    "-0.001953125",
    "-0.00244140625",
    "-0.0029296875",
    "-0.00341796875",
    "-0.00390625",
    "-0.0048828125",
    "-0.005859375",
    "-0.0068359375",
    "-0.0078125",
    "-0.009765625",
    "-0.01171875",
    "-0.013671875",
    "-0.015625",
    "-0.01953125",
    "-0.0234375",
    "-0.02734375",
    "-0.03125",
    "-0.0390625",
    "-0.046875",
    "-0.0546875",
    "-0.0625",
    "-0.078125",
    "-0.09375",
    "-0.109375",
    "-0.125",
    "-0.15625",
    "-0.1875",
    "-0.21875",
    "-0.25",
    "-0.3125",
    "-0.375",
    "-0.4375",
    "-0.5",
    "-0.625",
    "-0.75",
    "-0.875",
    "-1.0",
    "-1.25",
    "-1.5",
    "-1.75",
    "-2.0",
    "-2.5",
    "-3.0",
    "-3.5",
    "-4.0",
    "-5.0",
    "-6.0",
    "-7.0",
    "-8.0",
    "-10.0",
    "-12.0",
    "-14.0",
    "-16.0",
    "-20.0",
    "-24.0",
    "-28.0",
    "-32.0",
    "-40.0",
    "-48.0",
    "-56.0",
    "-64.0",
    "-80.0",
    "-96.0",
    "-112.0",
    "-128.0",
    "-160.0",
    "-192.0",
    "-224.0",
    "-256.0",
    "-320.0",
    "-384.0",
    "-448.0",
    "-512.0",
    "-640.0",
    "-768.0",
    "-896.0",
    "-1024.0",
    "-1280.0",
    "-1536.0",
    "-1792.0",
    "-2048.0",
    "-2560.0",
    "-3072.0",
    "-3584.0",
    "-4096.0",
    "-5120.0",
    "-6144.0",
    "-7168.0",
    "-8192.0",
    "-10240.0",
    "-12288.0",
    "-14336.0",
    "-16384.0",
    "-20480.0",
    "-24576.0",
    "-28672.0",
    "-32768.0",
    "-40960.0",
    "-49152.0",
    "-57344.0",
    "-inf",
    "nan",
    "nan",
    "nan",
)

alias float8e4m3_to_str = StaticTuple[StringLiteral, 256](
    "0.0",
    "0.001953125",
    "0.00390625",
    "0.005859375",
    "0.0078125",
    "0.009765625",
    "0.01171875",
    "0.013671875",
    "0.015625",
    "0.017578125",
    "0.01953125",
    "0.021484375",
    "0.0234375",
    "0.025390625",
    "0.02734375",
    "0.029296875",
    "0.03125",
    "0.03515625",
    "0.0390625",
    "0.04296875",
    "0.046875",
    "0.05078125",
    "0.0546875",
    "0.05859375",
    "0.0625",
    "0.0703125",
    "0.078125",
    "0.0859375",
    "0.09375",
    "0.1015625",
    "0.109375",
    "0.1171875",
    "0.125",
    "0.140625",
    "0.15625",
    "0.171875",
    "0.1875",
    "0.203125",
    "0.21875",
    "0.234375",
    "0.25",
    "0.28125",
    "0.3125",
    "0.34375",
    "0.375",
    "0.40625",
    "0.4375",
    "0.46875",
    "0.5",
    "0.5625",
    "0.625",
    "0.6875",
    "0.75",
    "0.8125",
    "0.875",
    "0.9375",
    "1.0",
    "1.125",
    "1.25",
    "1.375",
    "1.5",
    "1.625",
    "1.75",
    "1.875",
    "2.0",
    "2.25",
    "2.5",
    "2.75",
    "3.0",
    "3.25",
    "3.5",
    "3.75",
    "4.0",
    "4.5",
    "5.0",
    "5.5",
    "6.0",
    "6.5",
    "7.0",
    "7.5",
    "8.0",
    "9.0",
    "10.0",
    "11.0",
    "12.0",
    "13.0",
    "14.0",
    "15.0",
    "16.0",
    "18.0",
    "20.0",
    "22.0",
    "24.0",
    "26.0",
    "28.0",
    "30.0",
    "32.0",
    "36.0",
    "40.0",
    "44.0",
    "48.0",
    "52.0",
    "56.0",
    "60.0",
    "64.0",
    "72.0",
    "80.0",
    "88.0",
    "96.0",
    "104.0",
    "112.0",
    "120.0",
    "128.0",
    "144.0",
    "160.0",
    "176.0",
    "192.0",
    "208.0",
    "224.0",
    "240.0",
    "256.0",
    "288.0",
    "320.0",
    "352.0",
    "384.0",
    "416.0",
    "448.0",
    "nan",
    "-0.0",
    "-0.001953125",
    "-0.00390625",
    "-0.005859375",
    "-0.0078125",
    "-0.009765625",
    "-0.01171875",
    "-0.013671875",
    "-0.015625",
    "-0.017578125",
    "-0.01953125",
    "-0.021484375",
    "-0.0234375",
    "-0.025390625",
    "-0.02734375",
    "-0.029296875",
    "-0.03125",
    "-0.03515625",
    "-0.0390625",
    "-0.04296875",
    "-0.046875",
    "-0.05078125",
    "-0.0546875",
    "-0.05859375",
    "-0.0625",
    "-0.0703125",
    "-0.078125",
    "-0.0859375",
    "-0.09375",
    "-0.1015625",
    "-0.109375",
    "-0.1171875",
    "-0.125",
    "-0.140625",
    "-0.15625",
    "-0.171875",
    "-0.1875",
    "-0.203125",
    "-0.21875",
    "-0.234375",
    "-0.25",
    "-0.28125",
    "-0.3125",
    "-0.34375",
    "-0.375",
    "-0.40625",
    "-0.4375",
    "-0.46875",
    "-0.5",
    "-0.5625",
    "-0.625",
    "-0.6875",
    "-0.75",
    "-0.8125",
    "-0.875",
    "-0.9375",
    "-1.0",
    "-1.125",
    "-1.25",
    "-1.375",
    "-1.5",
    "-1.625",
    "-1.75",
    "-1.875",
    "-2.0",
    "-2.25",
    "-2.5",
    "-2.75",
    "-3.0",
    "-3.25",
    "-3.5",
    "-3.75",
    "-4.0",
    "-4.5",
    "-5.0",
    "-5.5",
    "-6.0",
    "-6.5",
    "-7.0",
    "-7.5",
    "-8.0",
    "-9.0",
    "-10.0",
    "-11.0",
    "-12.0",
    "-13.0",
    "-14.0",
    "-15.0",
    "-16.0",
    "-18.0",
    "-20.0",
    "-22.0",
    "-24.0",
    "-26.0",
    "-28.0",
    "-30.0",
    "-32.0",
    "-36.0",
    "-40.0",
    "-44.0",
    "-48.0",
    "-52.0",
    "-56.0",
    "-60.0",
    "-64.0",
    "-72.0",
    "-80.0",
    "-88.0",
    "-96.0",
    "-104.0",
    "-112.0",
    "-120.0",
    "-128.0",
    "-144.0",
    "-160.0",
    "-176.0",
    "-192.0",
    "-208.0",
    "-224.0",
    "-240.0",
    "-256.0",
    "-288.0",
    "-320.0",
    "-352.0",
    "-384.0",
    "-416.0",
    "-448.0",
    "nan",
)

alias float8e5m2fnuz_to_str = StaticTuple[StringLiteral, 256](
    "0.0",
    "7.62939453125e-06",
    "1.52587890625e-05",
    "2.288818359375e-05",
    "3.0517578125e-05",
    "3.814697265625e-05",
    "4.57763671875e-05",
    "5.340576171875e-05",
    "6.103515625e-05",
    "7.62939453125e-05",
    "9.1552734375e-05",
    "0.0001068115234375",
    "0.0001220703125",
    "0.000152587890625",
    "0.00018310546875",
    "0.000213623046875",
    "0.000244140625",
    "0.00030517578125",
    "0.0003662109375",
    "0.00042724609375",
    "0.00048828125",
    "0.0006103515625",
    "0.000732421875",
    "0.0008544921875",
    "0.0009765625",
    "0.001220703125",
    "0.00146484375",
    "0.001708984375",
    "0.001953125",
    "0.00244140625",
    "0.0029296875",
    "0.00341796875",
    "0.00390625",
    "0.0048828125",
    "0.005859375",
    "0.0068359375",
    "0.0078125",
    "0.009765625",
    "0.01171875",
    "0.013671875",
    "0.015625",
    "0.01953125",
    "0.0234375",
    "0.02734375",
    "0.03125",
    "0.0390625",
    "0.046875",
    "0.0546875",
    "0.0625",
    "0.078125",
    "0.09375",
    "0.109375",
    "0.125",
    "0.15625",
    "0.1875",
    "0.21875",
    "0.25",
    "0.3125",
    "0.375",
    "0.4375",
    "0.5",
    "0.625",
    "0.75",
    "0.875",
    "1.0",
    "1.25",
    "1.5",
    "1.75",
    "2.0",
    "2.5",
    "3.0",
    "3.5",
    "4.0",
    "5.0",
    "6.0",
    "7.0",
    "8.0",
    "10.0",
    "12.0",
    "14.0",
    "16.0",
    "20.0",
    "24.0",
    "28.0",
    "32.0",
    "40.0",
    "48.0",
    "56.0",
    "64.0",
    "80.0",
    "96.0",
    "112.0",
    "128.0",
    "160.0",
    "192.0",
    "224.0",
    "256.0",
    "320.0",
    "384.0",
    "448.0",
    "512.0",
    "640.0",
    "768.0",
    "896.0",
    "1024.0",
    "1280.0",
    "1536.0",
    "1792.0",
    "2048.0",
    "2560.0",
    "3072.0",
    "3584.0",
    "4096.0",
    "5120.0",
    "6144.0",
    "7168.0",
    "8192.0",
    "10240.0",
    "12288.0",
    "14336.0",
    "16384.0",
    "20480.0",
    "24576.0",
    "28672.0",
    "32768.0",
    "40960.0",
    "49152.0",
    "57344.0",
    "nan",
    "-7.62939453125e-06",
    "-1.52587890625e-05",
    "-2.288818359375e-05",
    "-3.0517578125e-05",
    "-3.814697265625e-05",
    "-4.57763671875e-05",
    "-5.340576171875e-05",
    "-6.103515625e-05",
    "-7.62939453125e-05",
    "-9.1552734375e-05",
    "-0.0001068115234375",
    "-0.0001220703125",
    "-0.000152587890625",
    "-0.00018310546875",
    "-0.000213623046875",
    "-0.000244140625",
    "-0.00030517578125",
    "-0.0003662109375",
    "-0.00042724609375",
    "-0.00048828125",
    "-0.0006103515625",
    "-0.000732421875",
    "-0.0008544921875",
    "-0.0009765625",
    "-0.001220703125",
    "-0.00146484375",
    "-0.001708984375",
    "-0.001953125",
    "-0.00244140625",
    "-0.0029296875",
    "-0.00341796875",
    "-0.00390625",
    "-0.0048828125",
    "-0.005859375",
    "-0.0068359375",
    "-0.0078125",
    "-0.009765625",
    "-0.01171875",
    "-0.013671875",
    "-0.015625",
    "-0.01953125",
    "-0.0234375",
    "-0.02734375",
    "-0.03125",
    "-0.0390625",
    "-0.046875",
    "-0.0546875",
    "-0.0625",
    "-0.078125",
    "-0.09375",
    "-0.109375",
    "-0.125",
    "-0.15625",
    "-0.1875",
    "-0.21875",
    "-0.25",
    "-0.3125",
    "-0.375",
    "-0.4375",
    "-0.5",
    "-0.625",
    "-0.75",
    "-0.875",
    "-1.0",
    "-1.25",
    "-1.5",
    "-1.75",
    "-2.0",
    "-2.5",
    "-3.0",
    "-3.5",
    "-4.0",
    "-5.0",
    "-6.0",
    "-7.0",
    "-8.0",
    "-10.0",
    "-12.0",
    "-14.0",
    "-16.0",
    "-20.0",
    "-24.0",
    "-28.0",
    "-32.0",
    "-40.0",
    "-48.0",
    "-56.0",
    "-64.0",
    "-80.0",
    "-96.0",
    "-112.0",
    "-128.0",
    "-160.0",
    "-192.0",
    "-224.0",
    "-256.0",
    "-320.0",
    "-384.0",
    "-448.0",
    "-512.0",
    "-640.0",
    "-768.0",
    "-896.0",
    "-1024.0",
    "-1280.0",
    "-1536.0",
    "-1792.0",
    "-2048.0",
    "-2560.0",
    "-3072.0",
    "-3584.0",
    "-4096.0",
    "-5120.0",
    "-6144.0",
    "-7168.0",
    "-8192.0",
    "-10240.0",
    "-12288.0",
    "-14336.0",
    "-16384.0",
    "-20480.0",
    "-24576.0",
    "-28672.0",
    "-32768.0",
    "-40960.0",
    "-49152.0",
    "-57344.0",
)

alias float8e4m3fnuz_to_str = StaticTuple[StringLiteral, 256](
    "0.0",
    "0.0009765625",
    "0.001953125",
    "0.0029296875",
    "0.00390625",
    "0.0048828125",
    "0.005859375",
    "0.0068359375",
    "0.0078125",
    "0.0087890625",
    "0.009765625",
    "0.0107421875",
    "0.01171875",
    "0.0126953125",
    "0.013671875",
    "0.0146484375",
    "0.015625",
    "0.017578125",
    "0.01953125",
    "0.021484375",
    "0.0234375",
    "0.025390625",
    "0.02734375",
    "0.029296875",
    "0.03125",
    "0.03515625",
    "0.0390625",
    "0.04296875",
    "0.046875",
    "0.05078125",
    "0.0546875",
    "0.05859375",
    "0.0625",
    "0.0703125",
    "0.078125",
    "0.0859375",
    "0.09375",
    "0.1015625",
    "0.109375",
    "0.1171875",
    "0.125",
    "0.140625",
    "0.15625",
    "0.171875",
    "0.1875",
    "0.203125",
    "0.21875",
    "0.234375",
    "0.25",
    "0.28125",
    "0.3125",
    "0.34375",
    "0.375",
    "0.40625",
    "0.4375",
    "0.46875",
    "0.5",
    "0.5625",
    "0.625",
    "0.6875",
    "0.75",
    "0.8125",
    "0.875",
    "0.9375",
    "1.0",
    "1.125",
    "1.25",
    "1.375",
    "1.5",
    "1.625",
    "1.75",
    "1.875",
    "2.0",
    "2.25",
    "2.5",
    "2.75",
    "3.0",
    "3.25",
    "3.5",
    "3.75",
    "4.0",
    "4.5",
    "5.0",
    "5.5",
    "6.0",
    "6.5",
    "7.0",
    "7.5",
    "8.0",
    "9.0",
    "10.0",
    "11.0",
    "12.0",
    "13.0",
    "14.0",
    "15.0",
    "16.0",
    "18.0",
    "20.0",
    "22.0",
    "24.0",
    "26.0",
    "28.0",
    "30.0",
    "32.0",
    "36.0",
    "40.0",
    "44.0",
    "48.0",
    "52.0",
    "56.0",
    "60.0",
    "64.0",
    "72.0",
    "80.0",
    "88.0",
    "96.0",
    "104.0",
    "112.0",
    "120.0",
    "128.0",
    "144.0",
    "160.0",
    "176.0",
    "192.0",
    "208.0",
    "224.0",
    "240.0",
    "nan",
    "-0.0009765625",
    "-0.001953125",
    "-0.0029296875",
    "-0.00390625",
    "-0.0048828125",
    "-0.005859375",
    "-0.0068359375",
    "-0.0078125",
    "-0.0087890625",
    "-0.009765625",
    "-0.0107421875",
    "-0.01171875",
    "-0.0126953125",
    "-0.013671875",
    "-0.0146484375",
    "-0.015625",
    "-0.017578125",
    "-0.01953125",
    "-0.021484375",
    "-0.0234375",
    "-0.025390625",
    "-0.02734375",
    "-0.029296875",
    "-0.03125",
    "-0.03515625",
    "-0.0390625",
    "-0.04296875",
    "-0.046875",
    "-0.05078125",
    "-0.0546875",
    "-0.05859375",
    "-0.0625",
    "-0.0703125",
    "-0.078125",
    "-0.0859375",
    "-0.09375",
    "-0.1015625",
    "-0.109375",
    "-0.1171875",
    "-0.125",
    "-0.140625",
    "-0.15625",
    "-0.171875",
    "-0.1875",
    "-0.203125",
    "-0.21875",
    "-0.234375",
    "-0.25",
    "-0.28125",
    "-0.3125",
    "-0.34375",
    "-0.375",
    "-0.40625",
    "-0.4375",
    "-0.46875",
    "-0.5",
    "-0.5625",
    "-0.625",
    "-0.6875",
    "-0.75",
    "-0.8125",
    "-0.875",
    "-0.9375",
    "-1.0",
    "-1.125",
    "-1.25",
    "-1.375",
    "-1.5",
    "-1.625",
    "-1.75",
    "-1.875",
    "-2.0",
    "-2.25",
    "-2.5",
    "-2.75",
    "-3.0",
    "-3.25",
    "-3.5",
    "-3.75",
    "-4.0",
    "-4.5",
    "-5.0",
    "-5.5",
    "-6.0",
    "-6.5",
    "-7.0",
    "-7.5",
    "-8.0",
    "-9.0",
    "-10.0",
    "-11.0",
    "-12.0",
    "-13.0",
    "-14.0",
    "-15.0",
    "-16.0",
    "-18.0",
    "-20.0",
    "-22.0",
    "-24.0",
    "-26.0",
    "-28.0",
    "-30.0",
    "-32.0",
    "-36.0",
    "-40.0",
    "-44.0",
    "-48.0",
    "-52.0",
    "-56.0",
    "-60.0",
    "-64.0",
    "-72.0",
    "-80.0",
    "-88.0",
    "-96.0",
    "-104.0",
    "-112.0",
    "-120.0",
    "-128.0",
    "-144.0",
    "-160.0",
    "-176.0",
    "-192.0",
    "-208.0",
    "-224.0",
    "-240.0",
)
