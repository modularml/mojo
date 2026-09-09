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

from std.collections import Optional
from std.math import inf, isnan, log, nan, sqrt
from std.sys import simd_width_of

from std.algorithm import vectorize

from max.algorithm import elementwise, mean, sum
from std.algorithm.functional import unswitch
from max.gpu.host import DeviceContext

from std.utils import IndexList
from std.utils.coord import Coord

# ===----------------------------------------------------------------------=== #
# kl_div
# ===----------------------------------------------------------------------=== #


def kl_div(
    x: SIMD, y: type_of(x)
) -> type_of(x) where x.dtype.is_floating_point():
    """Elementwise function for computing Kullback-Leibler divergence.

    $$
    \\mathrm{kl\\_div}(x, y) =
      \\begin{cases}
        x \\log(x / y) - x + y & x > 0, y > 0 \\\\
        y & x = 0, y \\ge 0 \\\\
        \\infty & \\text{otherwise}
      \\end{cases}
    $$
    """
    return (isnan(x) | isnan(y)).select(
        nan[x.dtype](),
        (x.gt(0) & y.gt(0)).select(
            x * log(x / y) - x + y,
            (x.eq(0) & y.ge(0)).select(y, inf[x.dtype]()),
        ),
    )


def kl_div[
    dtype: DType, //
](
    output: UnsafePointer[mut=True, Scalar[dtype], _],
    x: UnsafePointer[mut=False, Scalar[dtype], _],
    y: UnsafePointer[mut=False, Scalar[dtype], _],
    len: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    def kl_div_elementwise[
        simd_width: Int, alignment: Int = 1
    ](idx: Coord) {var}:
        output.store(
            idx[0].value(),
            kl_div(
                x.load[width=simd_width](idx[0].value()),
                y.load[width=simd_width](idx[0].value()),
            ),
        )

    elementwise[simd_width_of[dtype]()](kl_div_elementwise, Coord(len), ctx)


def kl_div[
    dtype: DType, //, out_type: DType = .float64
](
    x: UnsafePointer[mut=False, Scalar[dtype], _],
    y: UnsafePointer[mut=False, Scalar[dtype], _],
    len: Int,
) -> Scalar[
    out_type
] where dtype.is_floating_point() where out_type.is_floating_point():
    comptime simd_width = simd_width_of[dtype]()
    var accum_simd = SIMD[out_type, simd_width](0)
    var accum_scalar = Scalar[out_type](0)

    def kl_div_elementwise[simd_width: Int](idx: Int) {x, y, mut}:
        var xi = x.load[width=simd_width](idx).cast[out_type]()
        var yi = y.load[width=simd_width](idx).cast[out_type]()
        var kl = kl_div(xi, yi)

        # TODO: should use VDPBF16PS when applicable
        # (i.e., host has avx512_bf16, type = bf16, out_type = float32)
        comptime if simd_width == 1:
            accum_scalar += kl[0]
        else:
            accum_simd += rebind[type_of(accum_simd)](kl)

    vectorize[simd_width](len, kl_div_elementwise)

    return accum_simd.reduce_add() + accum_scalar


# ===----------------------------------------------------------------------=== #
# correlation
# ===----------------------------------------------------------------------=== #


def correlation[
    dtype: DType, //, out_type: DType = dtype
](
    u: UnsafePointer[mut=False, Scalar[dtype], _],
    v: UnsafePointer[mut=False, Scalar[dtype], _],
    len: Int,
    ctx: DeviceContext,
    *,
    w: OptionalPointer[mut=True, u.T, _] = Optional[
        UnsafePointer[u.T, MutUntrackedOrigin]
    ](),
    centered: Bool = True,
) raises -> Scalar[out_type]:
    """Compute the correlation distance between two 1-D arrays.

    The correlation distance between `u` and `v`, is
    defined as

    $$
        1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                  {{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}
    $$

    where $`\\bar{u}`$ is the mean of the elements of `u`
    and $`x \\cdot y`$ is the dot product of $x$ and $y$.
    """
    var umu = Scalar[out_type]()
    var vmu = Scalar[out_type]()
    var w_list = List[Scalar[dtype]]()
    if w:
        w_list = List[Scalar[dtype]](capacity=len)
        _div(w_list.unsafe_ptr(), w.value(), _sum(w.value(), len), len, ctx)
    if centered:
        if w:
            umu = _dot[out_type=out_type](u, w_list.unsafe_ptr(), len)
            vmu = _dot[out_type=out_type](v, w_list.unsafe_ptr(), len)
        else:
            umu = _mean(u, len).cast[out_type]()
            vmu = _mean(v, len).cast[out_type]()

    var uv = Scalar[out_type]()
    var uu = Scalar[out_type]()
    var vv = Scalar[out_type]()

    comptime simd_width = simd_width_of[dtype]()
    var uv_simd = SIMD[out_type, simd_width]()
    var uu_simd = SIMD[out_type, simd_width]()
    var vv_simd = SIMD[out_type, simd_width]()

    var w_val: UnsafePointer[w_list.T, origin_of(w_list)] = w_list.unsafe_ptr()

    def accumulate[weighted: Bool]() {u, v, len, mut}:
        def apply_wfn[simd_width: Int](idx: Int) {u, v, mut}:
            var ui = u.load[width=simd_width](idx).cast[out_type]() - umu
            var vi = v.load[width=simd_width](idx).cast[out_type]() - vmu
            var uw = ui
            var vw = vi

            comptime if weighted:
                var wi = w_val.load[width=simd_width](idx).cast[out_type]()
                uw *= wi
                vw *= wi

            var uvw = ui * vw
            var uuw = ui * uw
            var vvw = vi * vw

            comptime if simd_width == 1:
                uv += uvw[0]
                uu += uuw[0]
                vv += vvw[0]
            else:
                uv_simd += rebind[type_of(uv_simd)](uvw)
                uu_simd += rebind[type_of(uu_simd)](uuw)
                vv_simd += rebind[type_of(vv_simd)](vvw)

        vectorize[simd_width](len, apply_wfn)

    unswitch(w.__bool__(), accumulate)

    uv += uv_simd.reduce_add()
    uu += uu_simd.reduce_add()
    vv += vv_simd.reduce_add()

    return (uv / sqrt(uu * vv)).clamp(-1, 1)


def uncentered_unweighted_correlation[
    dtype: DType, //, out_type: DType = dtype
](
    u: UnsafePointer[mut=False, Scalar[dtype], _],
    v: UnsafePointer[mut=False, Scalar[dtype], _],
    len: Int,
) -> Scalar[out_type]:
    """Compute the uncentered and unweighted correlation
    distance between two 1-D arrays.
    Unlike `correlation` with arguments set, this does not raise.

    The correlation distance between `u` and `v`, is
    defined as

    $$
        1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                  {{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}
    $$

    where $`\\bar{u}`$ is the mean of the elements of `u`
    and $`x \\cdot y`$ is the dot product of $x$ and $y$.
    """
    var uv = _dot[out_type=out_type](u, v, len)
    var uu = _dot[out_type=out_type](u, u, len)
    var vv = _dot[out_type=out_type](v, v, len)
    comptime eps = Scalar[out_type](1e-6)
    return uv / (sqrt(uu * vv) + eps)


# ===----------------------------------------------------------------------=== #
# cosine
# ===----------------------------------------------------------------------=== #


def cosine[
    dtype: DType,
    //,
](
    u: UnsafePointer[mut=False, Scalar[dtype], _],
    v: UnsafePointer[mut=False, Scalar[dtype], _],
    len: Int,
) -> Float64:
    """Compute the Cosine distance between 1-D arrays.

    The Cosine distance between `u` and `v`, is defined as

    $$
    1 - \\frac{u \\cdot v}{\\|u\\|_2 \\|v\\|_2}.
    $$

    where $u \\cdot v$ is the dot product of $u$ and $v$.

    The cosine distance is also referred to as 'uncentered correlation',
    or 'reflective correlation'.
    """
    return 1 - uncentered_unweighted_correlation[out_type=DType.float64](
        u, v, len
    )


def relative_difference[
    dtype: DType,
    //,
](
    output: UnsafePointer[mut=False, Scalar[dtype], _],
    ref_out: UnsafePointer[mut=False, Scalar[dtype], _],
    len: Int,
) -> Float64:
    var sum_abs_diff: Float64 = 0.0
    var sum_abs_ref: Float64 = 0.0
    var size = len

    for idx in range(len):
        var ui = output[idx].cast[.float64]()
        var vi = ref_out[idx].cast[.float64]()

        sum_abs_diff += abs(ui - vi).cast[.float64]()

        sum_abs_ref += abs(vi).cast[.float64]()

    var mean_abs_diff = sum_abs_diff / Float64(size)
    var mean_abs_ref = sum_abs_ref / Float64(size)

    var rel_diff = mean_abs_diff / mean_abs_ref
    return rel_diff


# ===----------------------------------------------------------------------=== #
# utils
# ===----------------------------------------------------------------------=== #


def _sqrt[
    dtype: DType, //
](
    output: UnsafePointer[mut=True, Scalar[dtype], _],
    x: UnsafePointer[mut=False, Scalar[dtype], _],
    len: Int,
    ctx: DeviceContext,
) raises:
    def apply_fn[simd_width: Int, alignment: Int = 1](idx: Coord) {var}:
        output.store(
            idx[0].value(),
            rebind[SIMD[dtype, simd_width]](
                sqrt(x.load[width=simd_width](idx[0].value()))
            ),
        )

    elementwise[simd_width_of[dtype]()](apply_fn, Coord(len), ctx)


def _mul[
    dtype: DType, //
](
    output: UnsafePointer[mut=True, Scalar[dtype], _],
    x: UnsafePointer[mut=False, Scalar[dtype], _],
    y: UnsafePointer[mut=False, Scalar[dtype], _],
    len: Int,
    ctx: DeviceContext,
) raises:
    def apply_fn[simd_width: Int, alignment: Int = 1](idx: Coord) {var}:
        output.store(
            idx[0].value(),
            rebind[SIMD[dtype, simd_width]](
                x.load[width=simd_width](idx[0].value())
                * y.load[width=simd_width](idx[0].value())
            ),
        )

    elementwise[simd_width_of[dtype]()](apply_fn, Coord(len), ctx)


def _div[
    dtype: DType, //
](
    output: UnsafePointer[mut=True, Scalar[dtype], _],
    x: UnsafePointer[mut=False, Scalar[dtype], _],
    c: Scalar[dtype],
    len: Int,
    ctx: DeviceContext,
) raises:
    def apply_fn[simd_width: Int, alignment: Int = 1](idx: Coord) {var}:
        output.store(
            idx[0].value(),
            rebind[SIMD[dtype, simd_width]](
                x.load[width=simd_width](idx[0].value())
            )
            / c,
        )

    elementwise[simd_width_of[dtype]()](apply_fn, Coord(len), ctx)


def _sum[
    dtype: DType, //
](src: UnsafePointer[mut=False, Scalar[dtype], _], len: Int) raises -> Scalar[
    dtype
]:
    return sum(Span[Scalar[dtype]](unsafe_ptr=src, length=len))


def _mean[
    dtype: DType, //
](src: UnsafePointer[mut=False, Scalar[dtype], _], len: Int) raises -> Scalar[
    dtype
]:
    return mean(Span[Scalar[dtype]](unsafe_ptr=src, length=len))


def _dot[
    dtype: DType, //, out_type: DType = dtype
](
    x: UnsafePointer[mut=False, Scalar[dtype], _],
    y: UnsafePointer[mut=False, Scalar[dtype], _],
    len: Int,
) -> Scalar[out_type]:
    # loads are the expensive part, so we use the (probably) smaller
    # input type for determining simd width.
    comptime simd_width = simd_width_of[dtype]()
    var accum_simd = SIMD[out_type, simd_width](0)
    var accum_scalar = Scalar[out_type](0)

    def apply_fn[simd_width: Int](idx: Int) {x, y, mut}:
        var xi = x.load[width=simd_width](idx).cast[out_type]()
        var yi = y.load[width=simd_width](idx).cast[out_type]()

        # TODO: should use VDPBF16PS when applicable
        # (i.e., host has avx512_bf16, type = bf16, out_type = float32)
        comptime if simd_width == 1:
            accum_scalar += xi[0] * yi[0]
        else:
            accum_simd += rebind[type_of(accum_simd)](xi * yi)

    vectorize[simd_width](len, apply_fn)

    return accum_simd.reduce_add() + accum_scalar
