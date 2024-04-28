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

from testing import assert_equal


def test_int():
    assert_equal(3, 3)
    assert_equal(3 + 3, 6)
    assert_equal(4 - 1, 3)
    assert_equal(6 - 1, 5)


def test_floordiv():
    assert_equal(2 // 2, 1)
    assert_equal(2 // 3, 0)
    assert_equal(2 // -2, -1)
    assert_equal(99 // -2, -50)


def test_mod():
    assert_equal(99 % 1, 0)
    assert_equal(99 % 3, 0)
    assert_equal(99 % -2, -1)
    assert_equal(99 % 8, 3)
    assert_equal(99 % -8, -5)
    assert_equal(2 % -1, 0)
    assert_equal(2 % -2, 0)
    assert_equal(3 % -2, -1)
    assert_equal(-3 % 2, 1)


def test_bit_width():
    assert_equal((0)._bit_width(), 1)
    assert_equal((-1)._bit_width(), 1)
    assert_equal((255)._bit_width(), 9)
    assert_equal((-256)._bit_width(), 9)


def test_abs():
    assert_equal(abs(-5), 5)
    assert_equal(abs(2), 2)
    assert_equal(abs(0), 0)


def main():
    test_int()
    test_floordiv()
    test_mod()
    test_bit_width()
    test_abs()
