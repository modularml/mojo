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

from my_complex import Complex
from testing import *


def test_init():
    re1 = -1.2
    im1 = 6.5
    c1 = Complex(re1, im1)
    re2 = 3.14159
    im2 = 0.0
    c2 = Complex(re2)

    assert_equal(re1, c1.re)
    assert_equal(im1, c1.im)
    assert_equal(re2, c2.re)
    assert_equal(im2, c2.im)


def test_str():
    re1 = 3.3
    im1 = 5.1
    str1 = "({} + {}i)".format(re1, im1)
    out_str1 = String()
    c1 = Complex(re1, im1)
    assert_equal(str1, str(c1))

    c1.write_to(out_str1)
    assert_equal(str1, out_str1)

    re2 = -1.2
    im2 = -3.4
    str2 = "({} - {}i)".format(re2, abs(im2))
    out_str2 = String()
    c2 = Complex(re2, im2)
    assert_equal(str2, str(c2))

    c2.write_to(out_str2)
    assert_equal(str2, out_str2)


def test_indexing():
    err_msg = "index out of bounds"
    re1 = -1.2
    im1 = 6.5
    c1 = Complex(re1, im1)
    assert_equal(re1, c1[0])
    assert_equal(im1, c1[1])

    re2 = 4.5
    im2 = 7.8
    c1[0] = re2
    c1[1] = im2
    assert_equal(re2, c1[0])
    assert_equal(im2, c1[1])

    with assert_raises(contains=err_msg):
        _ = c1[-1]

    with assert_raises(contains=err_msg):
        _ = c1[2]

    with assert_raises(contains=err_msg):
        c1[-1] = 1.0

    with assert_raises(contains=err_msg):
        c1[2] = 1.0


def test_unary():
    re1 = -1.2
    im1 = 6.5
    c1 = Complex(re1, im1)
    re2 = 4.5
    im2 = -7.8
    c2 = Complex(re2, im2)

    c1_pos = +c1
    assert_equal(c1.re, c1_pos.re)
    assert_equal(c1.im, c1_pos.im)

    c2_pos = +c2
    assert_equal(c2.re, c2_pos.re)
    assert_equal(c2.im, c2_pos.im)

    c1_neg = -c1
    assert_equal(-c1.re, c1_neg.re)
    assert_equal(-c1.im, c1_neg.im)

    c2_neg = -c2
    assert_equal(-c2.re, c2_neg.re)
    assert_equal(-c2.im, c2_neg.im)


def test_binary_complex():
    c1 = Complex(-1.2, 6.5)
    c2 = Complex(3.14159, -2.71828)

    sum = c1 + c2
    diff = c1 - c2
    prod = c1 * c2
    quot = c1 / c2

    sum_re = 1.94159
    sum_im = 3.78172
    diff_re = -4.34159
    diff_im = 9.21828
    prod_re = 13.898912
    prod_im = 23.682271
    quot_re = -1.242203
    quot_im = 0.994192

    assert_almost_equal(sum_re, sum.re, atol=0.00001)
    assert_almost_equal(sum_im, sum.im, atol=0.00001)
    assert_almost_equal(diff_re, diff.re, atol=0.00001)
    assert_almost_equal(diff_im, diff.im, atol=0.00001)
    assert_almost_equal(prod_re, prod.re, atol=0.000001)
    assert_almost_equal(prod_im, prod.im, atol=0.000001)
    assert_almost_equal(quot_re, quot.re, atol=0.000001)
    assert_almost_equal(quot_im, quot.im, atol=0.000001)


def test_binary_float():
    c1 = Complex(-1.2, 6.5)
    f1 = 2.5

    sum = c1 + f1
    diff = c1 - f1
    prod = c1 * f1
    quot = c1 / f1

    sum_re = 1.3
    sum_im = 6.5
    diff_re = -3.7
    diff_im = 6.5
    prod_re = -3.0
    prod_im = 16.25
    quot_re = -0.48
    quot_im = 2.6

    assert_almost_equal(sum_re, sum.re, atol=0.00001)
    assert_almost_equal(sum_im, sum.im, atol=0.00001)
    assert_almost_equal(diff_re, diff.re, atol=0.00001)
    assert_almost_equal(diff_im, diff.im, atol=0.00001)
    assert_almost_equal(prod_re, prod.re, atol=0.000001)
    assert_almost_equal(prod_im, prod.im, atol=0.000001)
    assert_almost_equal(quot_re, quot.re, atol=0.000001)
    assert_almost_equal(quot_im, quot.im, atol=0.000001)


def test_binary_rfloat():
    c1 = Complex(-1.2, 6.5)
    f1 = 2.5

    sum = f1 + c1
    diff = f1 - c1
    prod = f1 * c1
    quot = f1 / c1

    sum_re = 1.3
    sum_im = 6.5
    diff_re = 3.7
    diff_im = -6.5
    prod_re = -3.0
    prod_im = 16.25
    quot_re = -0.068666
    quot_im = -0.371939

    assert_almost_equal(sum_re, sum.re, atol=0.00001)
    assert_almost_equal(sum_im, sum.im, atol=0.00001)
    assert_almost_equal(diff_re, diff.re, atol=0.00001)
    assert_almost_equal(diff_im, diff.im, atol=0.00001)
    assert_almost_equal(prod_re, prod.re, atol=0.000001)
    assert_almost_equal(prod_im, prod.im, atol=0.000001)
    assert_almost_equal(quot_re, quot.re, atol=0.000001)
    assert_almost_equal(quot_im, quot.im, atol=0.000001)


def test_complex_inplace():
    c1 = Complex(-1, -1)
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


def test_equality():
    c1 = Complex(-1.2, 6.5)
    c2 = Complex(3.14159, -2.71828)
    c3 = Complex(-1.2, 6.5)

    assert_false(c1 == c2)
    assert_true(c1 != c2)

    assert_true(c1 == c3)
    assert_false(c1 != c3)
