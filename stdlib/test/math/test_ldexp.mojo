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
# RUN: %mojo %s

from math import frexp, ldexp
from sys import external_call

from test_utils import libm_call
from testing import assert_almost_equal, assert_equal


def test_ldexp():
    assert_equal(ldexp(Float32(1.5), 4), 24)
    assert_equal(ldexp(Float64(1.5), Int32(4)), 24)


def test_ldexp_vector():
    assert_equal(
        ldexp(SIMD[DType.float32, 4](1.5), SIMD[DType.int32, 4](4)),
        SIMD[DType.float32, 4](24),
    )
    assert_equal(
        ldexp(SIMD[DType.float64, 4](1.5), SIMD[DType.int32, 4](4)),
        SIMD[DType.float64, 4](24),
    )
    assert_equal(
        ldexp(SIMD[DType.float32, 32](1.5), SIMD[DType.int32, 32](4)),
        SIMD[DType.float32, 32](24),
    )
    assert_equal(
        ldexp(SIMD[DType.float64, 32](1.5), SIMD[DType.int32, 32](4)),
        SIMD[DType.float64, 32](24),
    )


fn ldexp_libm[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width], e: SIMD[DType.int32, simd_width]) -> SIMD[
    type, simd_width
]:
    var res = SIMD[type, simd_width]()

    for i in range(simd_width):
        res[i] = external_call["ldexpf", Scalar[type]](arg, e)
    return res


def test_ldexp_extensive_float32():
    var i = -1e3
    while i < 1e3:
        var out = frexp(i.cast[DType.float32]())
        var frac = out[0]
        var exp = out[1].cast[DType.int32]()
        assert_almost_equal(
            ldexp(frac, exp),
            ldexp_libm(frac, exp),
            msg="unmatched results for frac="
            + str(frac)
            + " and exp="
            + str(exp)
            + " at index "
            + str(i),
        )
        i += 1007


def main():
    test_ldexp()
    test_ldexp_vector()
    test_ldexp_extensive_float32()
