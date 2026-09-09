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

from std.math import sqrt


@fieldwise_init
struct Complex(
    Boolable,
    Equatable,
    TrivialRegisterPassable,
    Writable,
):
    """Represents a complex value.

    The struct provides basic methods for manipulating complex values.
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var re: Float64
    var im: Float64

    # ===-------------------------------------------------------------------===#
    # Initializers
    # ===-------------------------------------------------------------------===#

    def __init__(out self, re: Float64):
        self.re = re
        self.im = 0.0

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def write_to(self, mut writer: Some[Writer]):
        writer.write("(", self.re)
        if self.im < 0:
            writer.write(" - ", -self.im)
        else:
            writer.write(" + ", self.im)
        writer.write("i)")

    def write_repr_to(self, mut writer: Some[Writer]):
        t"Complex(re = {self.re}, im = {self.im})".write_to(writer)

    def __bool__(self) -> Bool:
        return self.re != 0.0 or self.im != 0.0

    # ===-------------------------------------------------------------------===#
    # Unary arithmetic operator dunders
    # ===-------------------------------------------------------------------===#

    def __pos__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        return Self(-self.re, -self.im)

    # ===-------------------------------------------------------------------===#
    # Binary arithmetic operator dunders
    # ===-------------------------------------------------------------------===#

    def __add__(self, rhs: Self) -> Self:
        return Self(self.re + rhs.re, self.im + rhs.im)

    def __add__(self, rhs: Float64) -> Self:
        return Self(self.re + rhs, self.im)

    def __radd__(self, lhs: Float64) -> Self:
        return Self(self.re + lhs, self.im)

    def __iadd__(mut self, rhs: Self):
        self.re += rhs.re
        self.im += rhs.im

    def __iadd__(mut self, rhs: Float64):
        self.re += rhs

    def __sub__(self, rhs: Self) -> Self:
        return Self(self.re - rhs.re, self.im - rhs.im)

    def __sub__(self, rhs: Float64) -> Self:
        return Self(self.re - rhs, self.im)

    def __rsub__(self, lhs: Float64) -> Self:
        return Self(lhs - self.re, -self.im)

    def __isub__(mut self, rhs: Self):
        self.re -= rhs.re
        self.im -= rhs.im

    def __isub__(mut self, rhs: Float64):
        self.re -= rhs

    def __mul__(self, rhs: Self) -> Self:
        return Self(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )

    def __mul__(self, rhs: Float64) -> Self:
        return Self(self.re * rhs, self.im * rhs)

    def __rmul__(self, lhs: Float64) -> Self:
        return Self(lhs * self.re, lhs * self.im)

    def __imul__(mut self, rhs: Self):
        var new_re = self.re * rhs.re - self.im * rhs.im
        var new_im = self.re * rhs.im + self.im * rhs.re
        self.re = new_re
        self.im = new_im

    def __imul__(mut self, rhs: Float64):
        self.re *= rhs
        self.im *= rhs

    def __truediv__(self, rhs: Self) -> Self:
        var denom = rhs.squared_norm()
        return Self(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )

    def __truediv__(self, rhs: Float64) -> Self:
        return Self(self.re / rhs, self.im / rhs)

    def __rtruediv__(self, lhs: Float64) -> Self:
        var denom = self.squared_norm()
        return Self(
            (lhs * self.re) / denom,
            (-lhs * self.im) / denom,
        )

    def __itruediv__(mut self, rhs: Self):
        var denom = rhs.squared_norm()
        var new_re = (self.re * rhs.re + self.im * rhs.im) / denom
        var new_im = (self.im * rhs.re - self.re * rhs.im) / denom
        self.re = new_re
        self.im = new_im

    def __itruediv__(mut self, rhs: Float64):
        self.re /= rhs
        self.im /= rhs

    # ===-------------------------------------------------------------------===#
    # Indexing
    # ===-------------------------------------------------------------------===#

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

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def squared_norm(self) -> Float64:
        return self.re * self.re + self.im * self.im

    def norm(self) -> Float64:
        return sqrt(self.squared_norm())
