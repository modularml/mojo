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
from std.testing import assert_equal, assert_true
from std.math import sqrt, isclose


@fieldwise_init
struct Complex(
    Equatable,
    TrivialRegisterPassable,
    Writable,
):
    var re: Float64
    var im: Float64

    def __init__(out self, re: Float64):
        self.re = re
        self.im = 0.0

    def write_to(self, mut writer: Some[Writer]):
        writer.write("(", self.re)
        if self.im < 0:
            writer.write(" - ", -self.im)
        else:
            writer.write(" + ", self.im)
        writer.write("i)")

    def __pos__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        return Self(-self.re, -self.im)

    def __add__(self, rhs: Self) -> Self:
        return Self(self.re + rhs.re, self.im + rhs.im)

    def __sub__(self, rhs: Self) -> Self:
        return Self(self.re - rhs.re, self.im - rhs.im)

    def __mul__(self, rhs: Self) -> Self:
        return Self(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )

    def squared_norm(self) -> Float64:
        return self.re * self.re + self.im * self.im

    def norm(self) -> Float64:
        return sqrt(self.squared_norm())

    def __truediv__(self, rhs: Self) -> Self:
        var denom = rhs.squared_norm()
        return Self(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )

    def __add__(self, rhs: Float64) -> Self:
        return Self(self.re + rhs, self.im)

    def __radd__(self, lhs: Float64) -> Self:
        return Self(self.re + lhs, self.im)

    def __sub__(self, rhs: Float64) -> Self:
        return Self(self.re - rhs, self.im)

    def __rsub__(self, lhs: Float64) -> Self:
        return Self(lhs - self.re, -self.im)

    def __mul__(self, rhs: Float64) -> Self:
        return Self(self.re * rhs, self.im * rhs)

    def __rmul__(self, lhs: Float64) -> Self:
        return Self(lhs * self.re, lhs * self.im)

    def __truediv__(self, rhs: Float64) -> Self:
        return Self(self.re / rhs, self.im / rhs)

    def __rtruediv__(self, lhs: Float64) -> Self:
        var denom = self.squared_norm()
        return Self(
            (lhs * self.re) / denom,
            (-lhs * self.im) / denom,
        )

    def __iadd__(mut self, rhs: Self):
        self.re += rhs.re
        self.im += rhs.im

    def __iadd__(mut self, rhs: Float64):
        self.re += rhs

    def __isub__(mut self, rhs: Self):
        self.re -= rhs.re
        self.im -= rhs.im

    def __isub__(mut self, rhs: Float64):
        self.re -= rhs

    def __imul__(mut self, rhs: Self):
        var new_re = self.re * rhs.re - self.im * rhs.im
        var new_im = self.re * rhs.im + self.im * rhs.re
        self.re = new_re
        self.im = new_im

    def __imul__(mut self, rhs: Float64):
        self.re *= rhs
        self.im *= rhs

    def __itruediv__(mut self, rhs: Self):
        var denom = rhs.squared_norm()
        var new_re = (self.re * rhs.re + self.im * rhs.im) / denom
        var new_im = (self.im * rhs.re - self.re * rhs.im) / denom
        self.re = new_re
        self.im = new_im

    def __itruediv__(mut self, rhs: Float64):
        self.re /= rhs
        self.im /= rhs

    def __eq__(self, other: Self) -> Bool:
        return self.re == other.re and self.im == other.im

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def __getitem__(self, idx: Int) raises -> Float64:
        if idx == 0:
            return self.re
        if idx == 1:
            return self.im
        raise "index out of bounds"

    def __setitem__(mut self, idx: Int, value: Float64) raises:
        if idx == 0:
            self.re = value
        elif idx == 1:
            self.im = value
        else:
            raise "index out of bounds"


def test_complex() raises:
    var c = Complex(3.14, -2.72)
    assert_equal(c.re, 3.14)
    assert_equal(c.im, -2.72)
    assert_equal(String(c), "(3.14 - 2.72i)")

    c = Complex(-1.2, 6.5)
    assert_equal(String(+c), "(-1.2 + 6.5i)")
    assert_equal(String(-c), "(1.2 - 6.5i)")

    assert_true(isclose(c.squared_norm(), 43.69))
    assert_true(isclose(c.norm(), 6.6098))

    var c1 = Complex(-1.2, 6.5)
    var c2 = Complex(3.14, -2.72)
    assert_true(isclose((c1 + c2).re, 1.94))
    assert_true(isclose((c1 + c2).im, 3.78))
    assert_true(isclose((c1 - c2).re, -4.34))
    assert_true(isclose((c1 - c2).im, 9.22))

    c = Complex(-1.2, 6.5)
    assert_true(isclose((c + 2.5).re, 1.3))
    assert_true(isclose((c + 2.5).im, 6.5))
    assert_true(isclose((2.5 + c).re, 1.3))
    assert_true(isclose((2.5 + c).im, 6.5))
    assert_true(isclose((c * 2.5).re, -3.0))
    assert_true(isclose((c * 2.5).im, 16.25))

    c = Complex(-1.0, -1.0)
    c += Complex(0.5, -0.5)
    assert_true(isclose(c.re, -0.5))
    assert_true(isclose(c.im, -1.5))
    c += 2.75
    assert_true(isclose(c.re, 2.25))
    assert_true(isclose(c.im, -1.5))
    c -= Complex(1.25, 0.25)
    assert_true(isclose(c.re, 1.0))
    assert_true(isclose(c.im, -1.75))
    c -= 0.75
    assert_true(isclose(c.re, 0.25))
    assert_true(isclose(c.im, -1.75))
    c *= 0.75
    assert_true(isclose(c.re, 0.1875))
    assert_true(isclose(c.im, -1.3125))
    c /= 2.0
    assert_true(isclose(c.re, 0.09375))
    assert_true(isclose(c.im, -0.65625))

    c1 = Complex(-1.2, 6.5)
    c2 = Complex(-1.2, 6.5)
    var c3 = Complex(3.14, -2.72)
    assert_true(c1 == c2)
    assert_true(c1 != c3)

    c = Complex(3.14)
    assert_true(isclose(c[0], 3.14))
    assert_true(isclose(c[1], 0.0))
    c[1] = 42.0
    assert_true(isclose(c[1], 42.0))


def main() raises:
    test_complex()
