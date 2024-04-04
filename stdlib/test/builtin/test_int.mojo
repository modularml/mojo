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
# RUN: %mojo -debug-level full %s

from testing import assert_equal


def test_constructors():
    var i1 = Int(3)  # Constructible from IntLiteral
    var i2 = Int(Int(5))  # Constructible from Int


def test_add():
    assert_equal(6, Int(3) + Int(3))


def test_sub():
    assert_equal(3, Int(4) - Int(1))
    assert_equal(5, Int(6) - Int(1))


def test_div():
    var n = Int(5)
    var d = Int(2)
    assert_equal(2.5, n / d)
    n /= d
    assert_equal(2, n)


def test_pow():
    assert_equal(1, Int(3) ** Int(0))
    assert_equal(27, Int(3) ** Int(3))
    assert_equal(81, Int(3) ** Int(4))


def test_floordiv():
    assert_equal(1, Int(2) // Int(2))
    assert_equal(0, Int(2) // Int(3))
    assert_equal(-1, Int(2) // Int(-2))
    assert_equal(-50, Int(99) // Int(-2))
    assert_equal(-1, Int(-1) // Int(10))


def test_mod():
    assert_equal(0, Int(99) % Int(1))
    assert_equal(0, Int(99) % Int(3))
    assert_equal(-1, Int(99) % Int(-2))
    assert_equal(3, Int(99) % Int(8))
    assert_equal(-5, Int(99) % Int(-8))
    assert_equal(0, Int(2) % Int(-1))
    assert_equal(0, Int(2) % Int(-2))
    assert_equal(-1, Int(3) % Int(-2))
    assert_equal(1, Int(-3) % Int(2))


def main():
    test_constructors()
    test_add()
    test_sub()
    test_div()
    test_pow()
    test_floordiv()
    test_mod()
