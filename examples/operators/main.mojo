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


def main():
    # Examples of creating Complex instances
    c1 = Complex(-1.2, 6.5)
    print(String("c1: Real: {}; Imaginary: {}").format(c1.re, c1.im))

    c2 = Complex(3.14159)
    print(String("c2: Real: {}; Imaginary: {}").format(c2.re, c2.im))

    print()

    # Examples of using Complex values with str(), repr(), and print()
    c3 = Complex(3.14159, -2.71828)
    print("c3 =", c3)
    var msg: String = "The value is: " + str(c3)
    print(msg)
    print(String("{!r}").format(c3))

    print()

    # Examples of using Complex indexing
    print(String("c2[0]: {}; c2[1]: {}").format(c2[0], c2[1]))
    c2[0] = 2.71828
    c2[1] = 42
    print("c2[0] = 2.71828; c2[1] = 42; c2:", c2)

    print()

    # Examples of unary arithmetic operators
    print("+c1:", +c1)
    print("-c1:", -c1)

    print()

    # Examples of binary arithmetic operators
    print("c1 + c3 =", c1 + c3)
    print("c1 - c3 =", c1 - c3)
    print("c1 * c3 =", c1 * c3)
    print("c1 / c3 =", c1 / c3)

    print()

    # Examples of binary arithmetic operators mixing Complex and Float64 values
    f1 = 2.5
    print("c1 + f1 =", c1 + f1)
    print("f1 + c1 =", f1 + c1)
    print("c1 - f1 =", c1 - f1)
    print("f1 - c1 =", f1 - c1)
    print("c1 * f1 =", c1 * f1)
    print("f1 * c1 =", f1 * c1)
    print("c1 / f1 =", c1 / f1)
    print("f1 / c1 =", f1 / c1)

    print()

    # Examples of in-place arithmetic operators
    c4 = Complex(-1, -1)
    print("c4 =", c4)
    c4 += Complex(0.5, -0.5)
    print("c4 += Complex(0.5, -0.5) =>", c4)
    c4 += 2.75
    print("c4 += 2.75 =>", c4)
    c4 -= Complex(0.25, 1.5)
    print("c4 -= Complex(0.25, 1.5) =>", c4)
    c4 -= 3
    print("c4 -= 3 =>", c4)
    c4 *= Complex(-3.0, 2.0)
    print("c4 *= Complex(-3.0, 2.0) =>", c4)
    c4 *= 0.75
    print("c4 *= 0.75 =>", c4)
    c4 /= Complex(1.25, 2.0)
    print("c4 /= Complex(1.25, 2.0) =>", c4)
    c4 /= 2.0
    print("c4 /= 2.0 =>", c4)

    print()

    # Examples of equality and inequality comparison operators
    c1 = Complex(-1.2, 6.5)
    c3 = Complex(3.14159, -2.71828)
    c5 = Complex(-1.2, 6.5)

    if c1 == c5:
        print("c1 is equal to c5")
    else:
        print("c1 is not equal to c5")

    if c1 != c3:
        print("c1 is not equal to c3")
    else:
        print("c1 is equal to c3")
