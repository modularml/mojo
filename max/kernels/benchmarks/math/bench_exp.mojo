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

from std.math import *
from std.math.math import _exp_taylor, _ldexp_impl
from std.math.polynomial import polynomial_evaluate
from std.ffi import external_call
from std.sys import llvm_intrinsic, simd_width_of, size_of
from std.sys.arg import argv
from std.utils import IndexList
from std.algorithm.functional import vectorize
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)
from layout import Coord, TileTensor, row_major
from std.builtin.range import _StridedRange
from std.compile import compile_info
from std.memory import bitcast, dealloc, unsafe_stack_allocation


def _ri(v: Int) -> Int64:
    return Int64(v)


def apply[
    func: def[dtype: DType, width: SIMDLength](SIMD[dtype, width]) thin -> SIMD[
        dtype, width
    ],
    dtype: DType,
](
    input: TileTensor[mut=False, dtype, ...],
    output: TileTensor[mut=True, dtype, ...],
):
    def _func[width: Int](idx: Int) {imm}:
        output.store_linear[width=width](
            IndexList[1](idx),
            func(input.load_linear[width=width](IndexList[1](idx))),
        )

    vectorize[simd_width_of[dtype]()](input.num_elements(), _func)


def bench_unary[
    func: def[dtype: DType, width: SIMDLength](SIMD[dtype, width]) thin -> SIMD[
        dtype, width
    ],
    dtype: DType,
](mut m: Bench, size_range: _StridedRange[.int], op_name: String) raises:
    for i in size_range:
        bench_unary[func, dtype](m, i, op_name)


def bench_unary[
    func: def[dtype: DType, width: SIMDLength](SIMD[dtype, width]) thin -> SIMD[
        dtype, width
    ],
    dtype: DType,
](mut m: Bench, size: Int, op_name: String) raises:
    comptime alignment = 64
    var input_ptr_alloc = alloc[Scalar[dtype]](
        {count = size, alignment = alignment}
    ).into_managed()
    var input_ptr = input_ptr_alloc.unsafe_ptr()
    var output_ptr_alloc = alloc[Scalar[dtype]](
        {count = size, alignment = alignment}
    ).into_managed()
    var output_ptr = output_ptr_alloc.unsafe_ptr()

    var linspace = range(0x3000_0000, 0x42B0_0000, 1)
    for i in range(size):
        var f = bitcast[dtype](UInt32(linspace[i % len(linspace)]))
        input_ptr.unsafe_offset(i).write(f)

    def bench(mut b: Bencher, size: Int) raises {imm}:
        def iter_fn() {imm}:
            apply[func](
                TileTensor(input_ptr, row_major(Coord(_ri(size)))),
                TileTensor(output_ptr, row_major(Coord(_ri(size)))),
            )
            keep(output_ptr)

        b.iter(iter_fn)

    var elements = ThroughputMeasure(
        BenchMetric.elements, size * size_of[dtype]()
    )
    m.bench_with_input(
        bench,
        BenchId(op_name, String(size)),
        size,
        # TODO: Pick relevant benchmetric.
        [elements],
    )

    dealloc(input_ptr_alloc^)
    dealloc(output_ptr_alloc^)


def ldexp2kf_opt[
    dtype: DType, simd_width: SIMDLength
](x_in: SIMD[dtype, simd_width], q_in: SIMD[.int32, simd_width]) -> SIMD[
    dtype, simd_width
]:
    var m = q_in >> 31
    m = (((m + q_in) >> 6) - m) << 4
    var q = q_in - (m << 2)
    m += 127
    if m < 0:
        m = 0

    # m = m <   0 ?   0 : m;
    # m = m > 255 ? 255 : m;
    if m > 255:
        m = 255

    #   u = intBitsToFloat(((int32_t)m) << 23);
    var u = bitcast[dtype, simd_width](m << 23)
    var x = x_in * u * u * u * u
    #   u = intBitsToFloat(((int32_t)(q + 0x7f)) << 23);
    var xu = ((q + SIMD[.int32, simd_width](0x7F)).cast[.int32]()) << 23
    return x * xu.cast[dtype]()


def pow2if[
    simd_width: SIMDLength
](q: SIMD[.int32, simd_width]) -> SIMD[.float32, simd_width]:
    var x = (((q + SIMD[.int32, simd_width](0x7F)).cast[.int32]())) << 23
    return bitcast[.float32, simd_width](x)


def ldexp2kf[
    dtype: DType, simd_width: SIMDLength
](d: SIMD[dtype, simd_width], e: SIMD[.int32, simd_width]) -> SIMD[
    dtype, simd_width
]:
    # return d * (pow2if[simd_width](e >> 1) * pow2if[simd_width](e - (e >> 1))).cast[dtype]();
    var result = d * (pow2if[simd_width](e)).cast[dtype]()
    var y = bitcast[.int32, simd_width](result)

    var msb = y
    for _ in range(32):
        if (msb & 0x1).ne(0).reduce_and():
            break
        msb = msb >> 1

    # if e>=23:
    #     y=y-(y&mask)
    # if e>=23:
    #     y=y+1
    result = bitcast[dtype, simd_width](y)
    return result


# `bench_unary` takes an unconstrained function, so the stdlib `exp` reaches it
# through a wrapper that states no obligation of its own.
@always_inline
def exp_mojo[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    comptime assert dtype.is_floating_point(), "must be a floating point value"
    return exp(x)


@always_inline
def exp_libm[
    dtype: DType, simd_width: SIMDLength
](arg: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    var res = SIMD[dtype, simd_width]()

    comptime for i in range(simd_width):
        res[i] = external_call["expf", Scalar[dtype]](arg[i])
    return res


@always_inline
def ldexp_libm[
    dtype: DType, simd_width: SIMDLength
](arg: SIMD[dtype, simd_width], e: SIMD[.int32, simd_width]) -> SIMD[
    dtype, simd_width
]:
    var res = SIMD[dtype, simd_width]()

    comptime for i in range(simd_width):
        res[i] = external_call["ldexpf", Scalar[dtype]](arg)
    return res


def exp_sleef[
    dtype: DType, simd_width: SIMDLength
](d: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    comptime inv_lg2 = SIMD[dtype, simd_width](1.4426950408889634)
    comptime lg2it = SIMD[dtype, simd_width](0.6931471805599453)

    var q = floor(d.fma(inv_lg2, 0.5))

    ## upper and lower parts of log(2)
    comptime L2Uf = SIMD[dtype, simd_width](0.693145751953125)
    comptime L2Lf = SIMD[dtype, simd_width](1.428606765330187045e-06)

    var s = q.fma(-L2Uf, d)
    s = q.fma(-L2Lf, s)

    var u = SIMD[dtype, simd_width](0.000198527617612853646278381)
    u = u.fma(s, 0.00139304355252534151077271)
    u = u.fma(s, 0.00833336077630519866943359)
    u = u.fma(s, 0.0416664853692054748535156)
    u = u.fma(s, 0.166666671633720397949219)
    u = u.fma(s, 0.5)
    u = s * s * u + s

    return q.eq(0).select(u, ldexp2kf(u + 1, q.cast[.int32]()) - 1)


@always_inline
def _exp_taylor0[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    comptime coefficients: List[Scalar[dtype]] = [
        1.0,
        1.0,
        0.5,
        0.16666666666666666667,
        0.041666666666666666667,
        0.0083333333333333333333,
        0.0013888888888888888889,
        0.00019841269841269841270,
    ]
    return polynomial_evaluate[coefficients](x)


@always_inline
def exp_mojo_opt[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    comptime assert dtype.is_floating_point(), "must be a floating point value"
    comptime neg_ln2 = -0.69314718055966295651160180568695068359375
    comptime inv_lg2 = 1.442695040888963407359924681001892137426646

    # upper and lower parts of log(2)=[L2Uf,L2Lf]
    comptime L2Uf = 0.693145751953125
    comptime L2Lf = 1.428606765330187045e-06

    comptime min_val = SIMD[dtype, simd_width](-88.3762626647949)
    comptime max_val = SIMD[dtype, simd_width](88.3762626647950)

    comptime im_type = DType.float64
    var xc = x.clamp(min_val, max_val).cast[im_type]()
    var k = floor(xc.fma(inv_lg2, 0.5)).cast[im_type]()

    var r = k.fma(neg_ln2, xc)
    # var r = k.fma(-L2Lf, k.fma(-L2Uf, xc))
    var taylor_result = _exp_taylor0(r.cast[im_type]()).cast[dtype]()
    var expr = ldexp(taylor_result, k.cast[.int32]())
    return expr
    # var val1 = (expr > min_val).select(expr, SIMD[dtype,simd_width](0))
    # return (val1 < max_val).select(val1, SIMD[dtype,simd_width](inf[dtype]()))


@always_inline
def exp_mojo_opt2[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    comptime assert dtype.is_floating_point(), "must be a floating point value"
    comptime inv_lg2 = 1.44269504088896340736  # 1/log(2)

    # upper and lower parts of log(2)=[L2Uf,L2Lf]
    comptime L2Uf = -0.693359375
    comptime L2Lf = -2.12194440e-4

    comptime min_val = SIMD[dtype, simd_width](-87.3)
    comptime max_val = SIMD[dtype, simd_width](87.3)

    var xc = x.clamp(min_val, max_val)
    var k = floor(xc * inv_lg2)

    var r = k.fma(L2Lf, k.fma(L2Uf, xc))

    var taylor_result = _exp_taylor(r)

    var expr = ldexp(taylor_result, k.cast[.int32]())
    return expr


@always_inline
def _exp_taylor3[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    comptime coefficients: List[Scalar[dtype]] = [
        0.5,
        0.16666666666666666667,
        0.041666666666666666667,
        0.0083333333333333333333,
        0.0013888888888888888889,
        0.00019841269841269841270,
    ]
    return polynomial_evaluate[coefficients](x)


@always_inline
def exp_mojo_opt3[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    comptime assert dtype.is_floating_point(), "must be a floating point value"
    comptime inv_lg2 = 1.44269504088896340736  # 1/log(2)

    # upper and lower parts of log(2)=[L2Uf,L2Lf]
    comptime L2Uf = -0.693359375
    comptime L2Lf = -2.12194440e-4

    comptime min_val = SIMD[dtype, simd_width](-87.3)
    comptime max_val = SIMD[dtype, simd_width](87.3)

    var xc = x.clamp(min_val, max_val)
    var k = floor(xc * inv_lg2)

    var r = k.fma(L2Lf, k.fma(L2Uf, xc))
    var taylor_result = _exp_taylor3(r).fma(r * r, r) + 1

    var expr = _ldexp_impl(taylor_result, k)
    return expr


@always_inline
def _exp_taylor_mlas[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    return polynomial_evaluate[
        [
            Scalar[dtype](1.0),
            1.0,
            0.499999851,
            0.16666472,
            0.0416695364,
            0.00837312452,
            0.00137805939,
        ],
    ](x)


@always_inline
def exp_mlas[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    comptime assert dtype.is_floating_point(), "must be a floating point value"
    comptime neg_ln2 = -0.69314718055966295651160180568695068359375
    comptime inv_lg2 = 1.442695040888963407359924681001892137426646

    comptime neg_ln2_hi = -6.93145752e-1
    comptime neg_ln2_lo = -1.42860677e-6

    comptime min_val = -88.3762626647949
    comptime max_val = 88.3762626647950

    var xc = x.clamp(min_val, max_val)
    var k = floor(xc.fma(inv_lg2, 0.5))
    var r = k.fma(neg_ln2_hi, xc)
    var rr = k.fma(neg_ln2_lo, r)
    return max(ldexp(_exp_taylor_mlas(rr), k.cast[.int32]()), xc)


@always_inline
def llvm_ldexp[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width], exp: SIMD[.int32, simd_width]) -> SIMD[
    dtype, simd_width
]:
    return llvm_intrinsic["llvm.ldexp", type_of(x)](x, exp)


@always_inline
def mlas_llvm_ldexp[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    comptime assert dtype.is_floating_point(), "must be a floating point value"
    comptime neg_ln2 = -0.69314718055966295651160180568695068359375
    comptime inv_lg2 = 1.442695040888963407359924681001892137426646

    comptime neg_ln2_hi = -6.93145752e-1
    comptime neg_ln2_lo = -1.42860677e-6

    comptime min_val = -88.3762626647949
    comptime max_val = 88.3762626647950

    var xc = x.clamp(min_val, max_val)
    var k = floor(xc.fma(inv_lg2, 0.5))
    var r = k.fma(neg_ln2_hi, xc)
    var rr = k.fma(neg_ln2_lo, r)
    return max(llvm_ldexp(_exp_taylor_mlas(rr), k.cast[.int32]()), xc)


def accuracy_test() raises:
    comptime delta_min = -16
    comptime delta_max = 15
    comptime delta_range = delta_max - delta_min + 1

    var deltas_ptr = unsafe_stack_allocation[delta_range, DType.int32]()
    var deltas = TileTensor(deltas_ptr, row_major[delta_range]())
    _ = deltas.fill(Int32(0))

    for i in range(0x3000_0000, 0x42B0_0000, 1):
        var f = bitcast[.float32, 1](UInt32(i))

        var r1 = exp_mojo_opt3(f)
        var r2 = exp_libm(f)

        var i1 = bitcast[.int32, 1](r1)
        var i2 = bitcast[.int32, 1](r2)

        var diff = i1 - i2
        var id = Int(diff.clamp(delta_min, delta_max))
        deltas[id - delta_min] = deltas[id - delta_min] + 1

        if id == delta_max:
            print(f, r1, r2, diff)

    for i in range(delta_range):
        print("deltas", i + delta_min, deltas[i])


def main() raises:
    var args = argv()
    for arg in args:
        if arg == "-c":
            print(compile_info[llvm_ldexp[.float32, 4]]())
            return

    var m = Bench()
    var problem_size = range(1 << 24, 1 << 26, 1 << 25)
    bench_unary[exp_mojo, DType.float32](m, problem_size, "mojo")
    bench_unary[exp_mojo_opt, DType.float32](m, problem_size, "mojo_opt")
    bench_unary[exp_mojo_opt2, DType.float32](m, problem_size, "mojo_opt2")
    bench_unary[exp_mojo_opt3, DType.float32](m, problem_size, "mojo_opt3")
    bench_unary[exp_libm, DType.float32](m, problem_size, "libm")
    bench_unary[exp_sleef, DType.float32](m, problem_size, "sleef")
    bench_unary[exp_mlas, DType.float32](m, problem_size, "mlas")
    bench_unary[mlas_llvm_ldexp, DType.float32](
        m, problem_size, "mlas_llvm_ldexp"
    )
    accuracy_test()
    m.dump_report()
