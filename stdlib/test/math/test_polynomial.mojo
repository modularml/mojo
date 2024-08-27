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

from math.polynomial import _horner_evaluate, polynomial_evaluate

from testing import assert_equal


def test_polynomial_evaluate_degree3():
    # Evaluate 1000 + x + x^2
    alias coeffs = List[SIMD[DType.float64, 1]](1000.0, 1.0, 1.0)

    assert_equal(_horner_evaluate[coeffs](1.0), 1002.0)
    assert_equal(polynomial_evaluate[coeffs](1.0), 1002.0)
    assert_equal(_horner_evaluate[coeffs](0.1), 1000.11)
    assert_equal(polynomial_evaluate[coeffs](0.1), 1000.11)


def test_polynomial_evaluate_degree4():
    # Evalaute 1000 + 99 x - 43 x^2 + 12 x^3 - 14 x^4
    alias coeffs = List[SIMD[DType.float64, 1]](
        1000.0, 99.0, -43.0, 12.0, -14.0
    )

    assert_equal(_horner_evaluate[coeffs](1.0), 1054.0)
    assert_equal(polynomial_evaluate[coeffs](1.0), 1054.0)
    assert_equal(_horner_evaluate[coeffs](0.1), 1009.4806)
    assert_equal(polynomial_evaluate[coeffs](0.1), 1009.4806)


def test_polynomial_evaluate_degree10():
    # Evaluate 20.0 + 9.0 x + 1.0 x^2 + 1.0 x^3 + 1.0 x^4 + 1.0 x^5 + 1.0 x^6 +
    # 1.0 x^7 + 1.0 x^8 + 43.0 x^9 + 10.0 x^10
    alias coeffs = List[SIMD[DType.float64, 1]](
        20.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 43.0, 10.0
    )

    assert_equal(_horner_evaluate[coeffs](1.0), 89.0)
    assert_equal(polynomial_evaluate[coeffs](1.0), 89.0)
    assert_equal(_horner_evaluate[coeffs](0.1), 20.911111154)
    assert_equal(polynomial_evaluate[coeffs](0.1), 20.911111154)


def main():
    test_polynomial_evaluate_degree3()
    test_polynomial_evaluate_degree4()
    test_polynomial_evaluate_degree10()
