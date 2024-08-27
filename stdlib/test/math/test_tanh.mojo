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
# REQUIRES: linux
# RUN: %mojo %s

from math import tanh
from random import randn, seed

from test_utils import compare, libm_call
from testing import assert_almost_equal


fn tanh_libm[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return libm_call[type, simd_width, "tanhf", "tanh"](arg)


def test_tanh_libm[N: Int = 8192]():
    seed(0)
    alias test_dtype = DType.float32
    var x32 = UnsafePointer[Scalar[test_dtype]].alloc(N)
    randn[test_dtype](x32, N, 0, 9.0)
    print("For N=" + str(N) + " randomly generated vals; mean=0.0, var=9.0")

    ####################
    # mojo tanh result
    ####################
    var y32 = UnsafePointer[Scalar[test_dtype]].alloc(N)
    for i in range(N):
        y32[i] = tanh(x32[i])

    ####################
    ## libm tanh result
    ####################
    var libm_out = UnsafePointer[Scalar[test_dtype]].alloc(N)
    for i in range(N):
        libm_out[i] = tanh_libm(x32[i])

    # abs_rel_err = (abs_min, abs_max, rel_min, rel_max)
    var abs_rel_err = SIMD[test_dtype, 4](
        0.0, 2.384185791015625e-07, 0.0, 2.5438197326366208e-07
    )

    var err = compare[test_dtype](y32, libm_out, N, msg="Compare Mojo vs. LibM")
    assert_almost_equal(err, abs_rel_err)

    x32.free()
    y32.free()
    libm_out.free()


def test_direct():
    alias F32x4 = SIMD[DType.float32, 4]
    var f32x4 = 0.5 * F32x4(0.0, 1.0, 2.0, 3.0)
    assert_almost_equal(
        tanh(f32x4), F32x4(0.0, 0.462117165, 0.761594176, 0.905148208)
    )
    assert_almost_equal(
        tanh(0.5 * f32x4), F32x4(0.0, 0.244918659, 0.462117165, 0.635149002)
    )

    alias F64x4 = SIMD[DType.float64, 4]
    var f64x4 = 0.5 * F64x4(0.0, 1.0, 2.0, 3.0)
    assert_almost_equal(
        tanh(f64x4), F64x4(0.0, 0.462117165, 0.761594176, 0.905148208)
    )
    assert_almost_equal(
        tanh(0.5 * f64x4), F64x4(0.0, 0.244918659, 0.462117165, 0.635149002)
    )


def main():
    test_direct()
    test_tanh_libm()
