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

from std.math import tanh
from std.random import randn, seed

from test_utils import compare, libm_call
from std.testing import assert_almost_equal, TestSuite


def tanh_libm[
    dtype: DType, simd_width: SIMDLength
](arg: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
] where dtype.is_floating_point():
    return libm_call["tanhf", "tanh"](arg)


def test_tanh_tfvals_fp32() raises:
    comptime dtype = DType.float32

    # The following input values for x are taken from
    # https://github.com/modularml/modular/issues/28981#issuecomment-1890182667
    var x_stack = Array[Scalar[dtype], 4](fill={})
    var x = Span(x_stack)
    x.unsafe_ptr().unsafe_store[width=4](
        0,
        SIMD[dtype, 4](
            -1.2583316564559937,
            -8.081921577453613,
            -8.626264572143555,
            -0.7127348184585571,
        ),
    )

    var y_stack = Array[Scalar[dtype], 4](fill={})
    var y = Span(y_stack)
    for i in range(4):
        y[i] = tanh(x[i])

    #################################################
    # TF results
    # use `tf.print(tf.math.tanh(numpy.float32(x)))`
    var tfvals_stack = Array[Scalar[dtype], 4](fill={})
    var tfvals_fp32 = Span(tfvals_stack)
    tfvals_fp32.unsafe_ptr().unsafe_store[width=4](
        0, SIMD[dtype, 4](-0.850603521, -1, -1, -0.612388909)
    )

    # abs_rel_err = (abs_min, abs_max, rel_min, rel_max)
    var abs_rel_err = SIMD[dtype, 4](
        0.0, 1.1920928955078125e-07, 0.0, 1.1920928955078125e-07
    )
    var err = compare[dtype](
        y.unsafe_ptr(),
        tfvals_fp32.unsafe_ptr(),
        4,
        msg="Compare Mojo vs. Tensorflow FP32",
    )
    # check that tolerances are better than or almost equal to abs_rel_err
    for i in range(4):
        if not err[i] <= abs_rel_err[i]:
            assert_almost_equal(err[i], abs_rel_err[i])


def test_tanh_tfvals_fp64() raises:
    comptime dtype = DType.float64

    # The following input values for x are taken from
    # https://github.com/modularml/modular/issues/28981#issuecomment-1890182667
    var x_stack = Array[Scalar[dtype], 4](fill={})
    var x = Span(x_stack)
    x.unsafe_ptr().unsafe_store[width=4](
        0,
        SIMD[dtype, 4](
            -1.2583316564559937,
            -8.081921577453613,
            -8.626264572143555,
            -0.7127348184585571,
        ),
    )

    var y_stack = Array[Scalar[dtype], 4](fill={})
    var y = Span(y_stack)
    for i in range(4):
        y[i] = tanh(x[i])

    #################################################
    # TF results
    # use `tf.print(tf.math.tanh(numpy.float64(x)))`
    var tfvals_stack = Array[Scalar[dtype], 4](fill={})
    var tfvals_fp64 = Span(tfvals_stack)
    tfvals_fp64.unsafe_ptr().unsafe_store[width=4](
        0,
        SIMD[dtype, 4](
            -0.85060351067231821,
            -0.99999980894339091,
            -0.99999993567914991,
            -0.61238890225714893,
        ),
    )

    # abs_rel_err = (abs_min, abs_max, rel_min, rel_max)
    var abs_rel_err = SIMD[dtype, 4](
        7.2062200651146213e-09,
        1.2149700800989649e-08,
        8.3577847290501252e-09,
        1.4283624095774667e-08,
    )

    var err = compare[dtype](
        y.unsafe_ptr(),
        tfvals_fp64.unsafe_ptr(),
        4,
        msg="Compare Mojo vs. Tensorflow FP64",
    )
    # check that tolerances are better than or almost equal to abs_rel_err
    for i in range(4):
        if not err[i] <= abs_rel_err[i]:
            assert_almost_equal(err[i], abs_rel_err[i])


def _test_tanh_libm[N: Int = 8192]() raises:
    seed(0)
    comptime test_dtype = DType.float32
    var x32 = List(length=N, fill=Scalar[test_dtype](0))
    randn(x32, 0, 9.0)
    print("For N=", N, " randomly generated vals; mean=0.0, var=9.0")

    ####################
    # mojo tanh result
    ####################
    var y32 = List(length=N, fill=Scalar[test_dtype](0))
    for i in range(N):
        y32[i] = tanh(x32[i])

    ####################
    ## libm tanh result
    ####################
    var libm_out = List(length=N, fill=Scalar[test_dtype](0))
    for i in range(N):
        libm_out[i] = tanh_libm(x32[i])

    # abs_rel_err = (abs_min, abs_max, rel_min, rel_max)
    var abs_rel_err = SIMD[test_dtype, 4](
        0.0, 2.384185791015625e-07, 0.0, 3.2e-07
    )

    var err = compare[test_dtype](
        y32.unsafe_ptr(),
        libm_out.unsafe_ptr(),
        N,
        msg="Compare Mojo vs. LibM",
    )
    # check that tolerances are better than or almost equal to abs_rel_err
    for i in range(4):
        if not err[i] <= abs_rel_err[i]:
            assert_almost_equal(err[i], abs_rel_err[i])


def test_direct() raises:
    comptime F32x4 = SIMD[.float32, 4]
    var f32x4 = 0.5 * F32x4(0.0, 1.0, 2.0, 3.0)
    assert_almost_equal(
        tanh(f32x4), F32x4(0.0, 0.462117165, 0.761594176, 0.905148208)
    )
    assert_almost_equal(
        tanh(0.5 * f32x4), F32x4(0.0, 0.244918659, 0.462117165, 0.635149002)
    )

    comptime F64x4 = SIMD[.float64, 4]
    var f64x4 = 0.5 * F64x4(0.0, 1.0, 2.0, 3.0)
    assert_almost_equal(
        tanh(f64x4), F64x4(0.0, 0.462117165, 0.761594176, 0.905148208)
    )
    assert_almost_equal(
        tanh(0.5 * f64x4), F64x4(0.0, 0.244918659, 0.462117165, 0.635149002)
    )


def test_tanh_libm() raises:
    _test_tanh_libm[]()


def test_tanh_libm_f64() raises:
    comptime N = 8192
    seed(0)
    comptime test_dtype = DType.float64
    var x64 = List(length=N, fill=Scalar[test_dtype](0))
    randn[test_dtype](x64, 0, 9.0)
    print("For N=", N, " randomly generated float64 vals; mean=0.0, var=9.0")

    var y64 = List(length=N, fill=Scalar[test_dtype](0))
    for i in range(N):
        y64[i] = tanh(x64[i])

    var libm_out = List(length=N, fill=Scalar[test_dtype](0))
    for i in range(N):
        libm_out[i] = tanh_libm(x64[i])

    # Expect near machine-epsilon accuracy after float64 fix.
    # Allow up to 1e-14 absolute and relative error as a conservative bound.
    var abs_rel_err = SIMD[test_dtype, 4](0.0, 1e-14, 0.0, 1e-14)

    var err = compare[test_dtype](
        y64.unsafe_ptr(),
        libm_out.unsafe_ptr(),
        N,
        msg="Compare Mojo float64 vs. LibM tanh",
    )
    for i in range(4):
        if not err[i] <= abs_rel_err[i]:
            assert_almost_equal(err[i], abs_rel_err[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
