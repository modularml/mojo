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

# RUN: %mojo -debug-level full %s | FileCheck %s


# Evaluates the exp function using 6th order taylor series expansion. This is
# the same expansion used by MLAS internally. The expansion is:
#
# Exp[x] = 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120 + x^6/720 + x^7/5040
#        = 1 + x (1 + x (1/2 + x (1/6 + x (1/24 + (1/120 + x/720) x))))
def exp_scalar_taylor_float32(x: Float32) -> Float32:
    return 1.0 + x * (
        1.0
        + x
        * (
            0.5
            + x
            * (0.166667 + x * (0.0416667 + (0.00833333 + 0.00138889 * x) * x))
        )
    )


def erf_taylor_vector(x: Float32) -> Float32:
    return x * (x * x).fma(-0.37612638903183752463, 1.1283791670955125739)


def main():
    var res_exp = exp_scalar_taylor_float32(2.3)
    # CHECK: 9.88
    print(res_exp)

    var res_erf = erf_taylor_vector(0.8)
    # CHECK: 0.71
    print(res_erf)
