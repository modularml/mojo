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

from my_complex import Complex
from std.testing import *


def test_init() raises:
    var re1 = -1.2
    var im1 = 6.5
    var c1 = Complex(re1, im1)
    var re2 = 3.14159
    var im2 = 0.0
    var c2 = Complex(re2)

    assert_equal(re1, c1.re)
    assert_equal(im1, c1.im)
    assert_equal(re2, c2.re)
    assert_equal(im2, c2.im)


def test_bool() raises:
    var c1 = Complex(0.0, 0.0)
    var c2 = Complex(1.0, 0.0)
    var c3 = Complex(0.0, 1.0)
    var c4 = Complex(-1.2, 6.5)

    assert_false(c1)
    assert_true(c2)
    assert_true(c3)
    assert_true(c4)


def test_str() raises:
    var re1 = 3.3
    var im1 = 5.1
    var str1 = "({} + {}i)".format(re1, im1)
    var out_str1 = String()
    var c1 = Complex(re1, im1)
    assert_equal(str1, String(c1))

    c1.write_to(out_str1)
    assert_equal(str1, out_str1)

    var re2 = -1.2
    var im2 = -3.4
    var str2 = "({} - {}i)".format(re2, abs(im2))
    var out_str2 = String()
    var c2 = Complex(re2, im2)
    assert_equal(str2, String(c2))

    c2.write_to(out_str2)
    assert_equal(str2, out_str2)


def test_indexing() raises:
    var re1 = -1.2
    var im1 = 6.5
    var c1 = Complex(re1, im1)
    assert_equal(re1, c1[0])
    assert_equal(im1, c1[1])

    var re2 = 4.5
    var im2 = 7.8
    c1[0] = re2
    c1[1] = im2
    assert_equal(re2, c1[0])
    assert_equal(im2, c1[1])


def test_unary() raises:
    var re1 = -1.2
    var im1 = 6.5
    var c1 = Complex(re1, im1)
    var re2 = 4.5
    var im2 = -7.8
    var c2 = Complex(re2, im2)

    var c1_pos = +c1
    assert_equal(c1.re, c1_pos.re)
    assert_equal(c1.im, c1_pos.im)

    var c2_pos = +c2
    assert_equal(c2.re, c2_pos.re)
    assert_equal(c2.im, c2_pos.im)

    var c1_neg = -c1
    assert_equal(-c1.re, c1_neg.re)
    assert_equal(-c1.im, c1_neg.im)

    var c2_neg = -c2
    assert_equal(-c2.re, c2_neg.re)
    assert_equal(-c2.im, c2_neg.im)


def test_binary_complex() raises:
    var c1 = Complex(-1.2, 6.5)
    var c2 = Complex(3.14159, -2.71828)

    var sum = c1 + c2
    var diff = c1 - c2
    var prod = c1 * c2
    var quot = c1 / c2

    var sum_re = 1.94159
    var sum_im = 3.78172
    var diff_re = -4.34159
    var diff_im = 9.21828
    var prod_re = 13.898912
    var prod_im = 23.682271
    var quot_re = -1.242203
    var quot_im = 0.994192

    assert_almost_equal(sum_re, sum.re, atol=0.00001)
    assert_almost_equal(sum_im, sum.im, atol=0.00001)
    assert_almost_equal(diff_re, diff.re, atol=0.00001)
    assert_almost_equal(diff_im, diff.im, atol=0.00001)
    assert_almost_equal(prod_re, prod.re, atol=0.000001)
    assert_almost_equal(prod_im, prod.im, atol=0.000001)
    assert_almost_equal(quot_re, quot.re, atol=0.000001)
    assert_almost_equal(quot_im, quot.im, atol=0.000001)


def test_binary_float() raises:
    var c1 = Complex(-1.2, 6.5)
    var f1 = 2.5

    var sum = c1 + f1
    var diff = c1 - f1
    var prod = c1 * f1
    var quot = c1 / f1

    var sum_re = 1.3
    var sum_im = 6.5
    var diff_re = -3.7
    var diff_im = 6.5
    var prod_re = -3.0
    var prod_im = 16.25
    var quot_re = -0.48
    var quot_im = 2.6

    assert_almost_equal(sum_re, sum.re, atol=0.00001)
    assert_almost_equal(sum_im, sum.im, atol=0.00001)
    assert_almost_equal(diff_re, diff.re, atol=0.00001)
    assert_almost_equal(diff_im, diff.im, atol=0.00001)
    assert_almost_equal(prod_re, prod.re, atol=0.000001)
    assert_almost_equal(prod_im, prod.im, atol=0.000001)
    assert_almost_equal(quot_re, quot.re, atol=0.000001)
    assert_almost_equal(quot_im, quot.im, atol=0.000001)


def test_binary_rfloat() raises:
    var c1 = Complex(-1.2, 6.5)
    var f1 = 2.5

    var sum = f1 + c1
    var diff = f1 - c1
    var prod = f1 * c1
    var quot = f1 / c1

    var sum_re = 1.3
    var sum_im = 6.5
    var diff_re = 3.7
    var diff_im = -6.5
    var prod_re = -3.0
    var prod_im = 16.25
    var quot_re = -0.068666
    var quot_im = -0.371939

    assert_almost_equal(sum_re, sum.re, atol=0.00001)
    assert_almost_equal(sum_im, sum.im, atol=0.00001)
    assert_almost_equal(diff_re, diff.re, atol=0.00001)
    assert_almost_equal(diff_im, diff.im, atol=0.00001)
    assert_almost_equal(prod_re, prod.re, atol=0.000001)
    assert_almost_equal(prod_im, prod.im, atol=0.000001)
    assert_almost_equal(quot_re, quot.re, atol=0.000001)
    assert_almost_equal(quot_im, quot.im, atol=0.000001)


def test_complex_inplace() raises:
    var c1 = Complex(-1, -1)
    c1 += Complex(0.5, -0.5)
    assert_almost_equal(-0.5, c1.re, atol=0.000001)
    assert_almost_equal(-1.5, c1.im, atol=0.000001)

    c1 = Complex(-0.5, -1.5)
    c1 += 2.75
    assert_almost_equal(2.25, c1.re, atol=0.000001)
    assert_almost_equal(-1.5, c1.im, atol=0.000001)

    c1 = Complex(2.25, -1.5)
    c1 -= Complex(0.25, 1.5)
    assert_almost_equal(2.0, c1.re, atol=0.000001)
    assert_almost_equal(-3.0, c1.im, atol=0.000001)

    c1 = Complex(2.0, -3.0)
    c1 -= 3
    assert_almost_equal(-1.0, c1.re, atol=0.000001)
    assert_almost_equal(-3.0, c1.im, atol=0.000001)

    c1 = Complex(-1.0, -3.0)
    c1 *= Complex(-3.0, 2.0)
    assert_almost_equal(9.0, c1.re, atol=0.000001)
    assert_almost_equal(7.0, c1.im, atol=0.000001)

    c1 = Complex(9.0, 7.0)
    c1 *= 0.75
    assert_almost_equal(6.75, c1.re, atol=0.000001)
    assert_almost_equal(5.25, c1.im, atol=0.000001)

    c1 = Complex(6.75, 5.25)
    c1 /= Complex(1.25, 2.0)
    assert_almost_equal(3.404494, c1.re, atol=0.000001)
    assert_almost_equal(-1.247191, c1.im, atol=0.000001)

    c1 = Complex(-9.0, 7.0)
    c1 /= 2.0
    assert_almost_equal(-4.5, c1.re, atol=0.000001)
    assert_almost_equal(3.5, c1.im, atol=0.000001)


def test_equality() raises:
    var c1 = Complex(-1.2, 6.5)
    var c2 = Complex(-1.2, 0.0)
    var c3 = Complex(0.0, 6.5)

    assert_true(c1 == c1)
    assert_false(c1 != c1)

    assert_true(c1 != c2)
    assert_false(c1 == c2)

    assert_true(c1 != c3)
    assert_false(c1 == c3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
